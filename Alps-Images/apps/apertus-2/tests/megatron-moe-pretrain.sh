#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
MEGATRON_LM_DIR="${MEGATRON_LM_DIR:-/opt/megatron}"
PYTHON="${PYTHON:-python3}"

[[ -d "$MEGATRON_LM_DIR" ]] || { echo "ERROR: MEGATRON_LM_DIR missing: $MEGATRON_LM_DIR" >&2; exit 1; }
[[ -f "$MEGATRON_LM_DIR/pretrain_gpt.py" ]] || { echo "ERROR: pretrain_gpt.py missing in $MEGATRON_LM_DIR" >&2; exit 1; }

export PYTHONPATH="$MEGATRON_LM_DIR:${PYTHONPATH:-}"

# ------------------------------------------------------------
# Distributed env
# ------------------------------------------------------------
export RANK="${RANK:-${SLURM_PROCID:-0}}"
export LOCAL_RANK="${LOCAL_RANK:-${SLURM_LOCALID:-0}}"
export WORLD_SIZE="${WORLD_SIZE:-${SLURM_NTASKS:-1}}"

export MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "${SLURM_JOB_NODELIST:-}" 2>/dev/null | head -n 1 || echo 127.0.0.1)}"
export MASTER_PORT="${MASTER_PORT:-$(( 20000 + (${SLURM_JOB_ID:-1} % 40000) ))}"

# ------------------------------------------------------------
# ENV knobs
# ------------------------------------------------------------
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB__FILE_STREAM_RETRY_MAX="${WANDB__FILE_STREAM_RETRY_MAX:-10}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

# Respect Slurm cpus-per-task if present
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"

# Caches (use tmp; runner images often don't have persistent dirs)
TORCH_INDUCTOR_CACHE_DIR="${TORCH_INDUCTOR_CACHE_DIR:-/tmp/.torch_inductor}"
TRITON_HOME_DIR="${TRITON_HOME_DIR:-/tmp/.triton}"
PYTHON_CACHE_DIR="${PYTHON_CACHE_DIR:-/tmp/.python_cache}"

mkdir -p "$TORCH_INDUCTOR_CACHE_DIR" "$TRITON_HOME_DIR" "$TRITON_HOME_DIR/cache" "$PYTHON_CACHE_DIR"
export TRITON_HOME="$TRITON_HOME_DIR"
export TRITON_CACHE_DIR="$TRITON_HOME_DIR/cache"

# ------------------------------------------------------------
# “Real” defaults (can be overridden from CI/job vars)
# ------------------------------------------------------------
MBS="${MBS:-2}"
GBS=$((2 * SLURM_NPROCS * MBS))
SEQ_LEN="${SEQ_LEN:-4096}"
TRAINING_STEPS="${TRAINING_STEPS:-40000}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-2000}"
MOCK_DATA="true"

# ------------------------------------------------------------
# CI overrides
# ------------------------------------------------------------
# Set CI_FAST=1 in CI to make it cheap but keep “shape” of args.
CI_FAST="${CI_FAST:-1}"
if [[ "$CI_FAST" == "1" ]]; then
  # keep same “kind” of run, just tiny
  SEQ_LEN="${SEQ_LEN_CI:-512}"
  TRAINING_STEPS="${TRAINING_STEPS_CI:-10}"
  CHECKPOINT_STEPS="${CHECKPOINT_STEPS_CI:-10000000}"  # effectively disables
  GBS="${GBS_CI:-$WORLD_SIZE}"                         # sane default for small runs
  LR_WARMUP_ITERS="${LR_WARMUP_ITERS_CI:-1}"
  # Keep decay steps >= training iters, but strictly > warmup
  LR_WSD_DECAY_ITERS="${LR_WSD_DECAY_ITERS_CI:-$TRAINING_STEPS}"
  if (( LR_WSD_DECAY_ITERS <= LR_WARMUP_ITERS )); then
    LR_WSD_DECAY_ITERS=$((LR_WARMUP_ITERS + 1))
  fi
fi

# ------------------------------------------------------------
# Logging / ckpt dirs
# ------------------------------------------------------------
RUN_ROOT="${RUN_ROOT:-/tmp/megatron-ci}"
EXP_NAME="${EXP_NAME:-apertus-2-ci-${WORLD_SIZE}ranks}"
EXP_DIR="$RUN_ROOT/$EXP_NAME"
CKPT_DIR="$EXP_DIR/checkpoints"
DEBUG_DIR="$EXP_DIR/debug/${SLURM_JOB_ID:-0}"
LOGGING_DIR="$EXP_DIR/logging"
TENSORBOARD_DIR="$LOGGING_DIR/tensorboard"

mkdir -p "$CKPT_DIR" "$DEBUG_DIR" "$LOGGING_DIR" "$TENSORBOARD_DIR"

# ------------------------------------------------------------
# Megatron args
# ------------------------------------------------------------
# Network size: keep flags, but allow CI to shrink sizes while preserving “features”
NUM_LAYERS="${NUM_LAYERS:-2}"
HIDDEN_SIZE="${HIDDEN_SIZE:-2048}"
FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE:-6144}"
MOE_FFN_HIDDEN_SIZE="${MOE_FFN_HIDDEN_SIZE:-768}"
NUM_ATTN_HEADS="${NUM_ATTN_HEADS:-20}"
NUM_QUERY_GROUPS="${NUM_QUERY_GROUPS:-8}"
NUM_EXPERTS="${NUM_EXPERTS:-128}"

if [[ "$CI_FAST" == "1" ]]; then
  NUM_LAYERS="${NUM_LAYERS_CI:-4}"
  HIDDEN_SIZE="${HIDDEN_SIZE_CI:-512}"
  FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE_CI:-2048}"
  NUM_ATTN_HEADS="${NUM_ATTN_HEADS_CI:-8}"
  NUM_QUERY_GROUPS="${NUM_QUERY_GROUPS_CI:-2}"
  NUM_EXPERTS="${NUM_EXPERTS_CI:-16}"
fi

MODEL_ARGS=(
    --num-layers 2   # TODO:
    --hidden-size 2048
    --ffn-hidden-size "$FFN_HIDDEN_SIZE"
    --num-attention-heads "$NUM_ATTN_HEADS"
    --kv-channels 192
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --apply-layernorm-1p
    --swiglu
    --position-embedding-type rope
    --rotary-base 10000
    --disable-bias-linear
    --untie-embeddings-and-output-weights
    --hidden-dropout 0.0
    --attention-dropout 0.0
    --seq-length "$SEQ_LEN"
    --max-position-embeddings 4096
    --make-vocab-size-divisible-by 128
    --init-method-std 0.008
)

MLA_ARGS=(
    --multi-latent-attention                      # = glm4.7's (attn)
    --q-lora-rank 768                             # = glm4.7's 768 (attn)
    --kv-lora-rank 512                             # = glm4.7's 512 (attn)
    --qk-head-dim 192                             # = glm4.7's 192 (attn)
    --qk-pos-emb-head-dim 64                      # = glm4.7's 64 (attn)
    --v-head-dim 256                             # = glm4.7's 256 (attn)
    --qk-clip                                    # = kimi2 (attn+muon)
)

MOE_ARGS=(
    --num-experts "$NUM_EXPERTS"
    --moe-ffn-hidden-size "$MOE_FFN_HIDDEN_SIZE"
    --moe-router-topk 7
    --moe-shared-expert-intermediate-size 768
    --moe-layer-freq "'([0]+[1]*1)'"
    --moe-router-score-function sigmoid
    --moe-router-pre-softmax
    --moe-router-load-balancing-type seq_aux_loss
    --moe-router-bias-update-rate 0.001
    --moe-aux-loss-coeff 0.0001
    --moe-router-enable-expert-bias
    --moe-router-topk-scaling-factor 2.5
    --moe-router-dtype fp32
)

OPTIMIZER_ARGS=(
    --optimizer dist_muon
    --lr 1e-3
    --min-lr 1e-4
    --adam-beta1 0.9
    --adam-beta2 0.999
    --adam-eps 1e-8
    --muon-momentum 0.95
    --muon-use-nesterov
    --muon-scale-mode shape_scaling
    --muon-extra-scale-factor 1.0
    --muon-num-ns-steps 5
    --weight-decay 0.1
    --lr-decay-style WSD
    --lr-warmup-samples $((20 * GBS))
    --lr-wsd-decay-style cosine
    --lr-wsd-decay-samples 0
    --clip-grad 1.0
)
    # --rampup-batch-size 8 8 10000

# Distributed: keep the flags, but allow CI to force TP=1 to fit small jobs.
TP="${TP:-1}"
PP="${PP:-1}"
EP="${EP:-4}"
if [[ "$CI_FAST" == "1" ]]; then
  TP="${TP_CI:-1}"
  PP="${PP_CI:-1}"
  EP="${EP:-4}"
fi
# Note: Muon/dist_muon does not support --use-distributed-optimizer
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size "$TP"
    --pipeline-model-parallel-size "$PP"
    --expert-model-parallel-size "$EP"
    --overlap-grad-reduce
    --moe-grouped-gemm
    --moe-token-dispatcher-type alltoall
    --moe-per-layer-logging
    --overlap-moe-expert-parallel-comm
)

FUSING_ARGS=(
    --no-rope-fusion
    --mla-down-proj-fusion
    --moe-grouped-gemm
    --moe-router-fusion
    --moe-permute-fusion
    --cross-entropy-loss-fusion
    --cross-entropy-fusion-impl te
    --disable-symmetric-registration
    #TODO: fusion
    #TODO: recomputation
    #TODO: cuda graph
    #TODO: miscellaneous
)

TRAINING_ARGS=(
    --micro-batch-size ${MBS}
    --global-batch-size ${GBS}
    --train-samples $((TRAINING_STEPS * GBS))
    --eval-interval 999999999
    --eval-iters 0
    --bf16
    --attention-backend fused # TODO: for return_max_logit
    --transformer-impl transformer_engine
    --ckpt-format torch_dist
    --no-check-for-nan-in-loss-and-grad
    --seed 41
)

if [[ "$CI_FAST" != "1" ]]; then
  TRAINING_ARGS+=( --async-save --ckpt-fully-parallel-load )
fi

TOKENIZER_ARGS=( --tokenizer-type NullTokenizer --vocab-size 32000 )

DATA_ARGS=(
    --dataloader-type single
    --num-workers 0
    --num-dataset-builder-threads 1
    --reset-position-ids
    --reset-attention-mask
    --eod-mask-loss
    --mock-data
)

LOGGING_ARGS=(
    --log-interval 1
    --log-throughput
    --log-timers-to-tensorboard
    --log-validation-ppl-to-tensorboard
    --tensorboard-queue-size 1
    --tensorboard-dir "${TENSORBOARD_DIR}"
    --save-interval "${CHECKPOINT_STEPS}"
    --save "${CKPT_DIR}"
    --distributed-timeout-minutes 600
    --manual-gc
    --manual-gc-interval 500
)

export NVTE_DEBUG=1
export NVTE_DEBUG_LEVEL=2
export NVTE_NORM_FWD_USE_CUDNN=1
export NVTE_NORM_BWD_USE_CUDNN=1
export NVTE_FWD_LAYERNORM_SM_MARGIN=0
export NVTE_BWD_LAYERNORM_SM_MARGIN=0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export NVTE_USE_CUTLASS_GROUPED_GEMM=0
export NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1

exec "$PYTHON" "$MEGATRON_LM_DIR/pretrain_gpt.py" \
  "${MODEL_ARGS[@]}" \
  "${MLA_ARGS[@]}" \
  "${MOE_ARGS[@]}" \
  "${MODEL_PARALLEL_ARGS[@]}" \
  "${FUSING_ARGS[@]}" \
  "${TRAINING_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${TOKENIZER_ARGS[@]}" \
  "${DATA_ARGS[@]}" \
  "${LOGGING_ARGS[@]}"
