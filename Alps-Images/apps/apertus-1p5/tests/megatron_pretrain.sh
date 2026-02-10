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
MBS="${MBS:-1}"
GBS="${GBS:-32}"
SEQ_LEN="${SEQ_LEN:-8192}"
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
EXP_NAME="${EXP_NAME:-apertus-1p5-ci-${WORLD_SIZE}ranks}"
EXP_DIR="$RUN_ROOT/$EXP_NAME"
CKPT_DIR="$EXP_DIR/checkpoints"
DEBUG_DIR="$EXP_DIR/debug/${SLURM_JOB_ID:-0}"
LOGGING_DIR="$EXP_DIR/logging"
TENSORBOARD_DIR="$LOGGING_DIR/tensorboard"

mkdir -p "$CKPT_DIR" "$DEBUG_DIR" "$LOGGING_DIR" "$TENSORBOARD_DIR"

TRIGGER_DIR="/tmp/trigger_${SLURM_JOB_ID:-ci}"
mkdir -p "$TRIGGER_DIR"

# ------------------------------------------------------------
# Megatron args
# ------------------------------------------------------------
TRANSFORMER_ENGINE_ARGS=(
  --main-grads-dtype fp32
  --log-params-norm
)

# Network size: keep flags, but allow CI to shrink sizes while preserving “features”
NUM_LAYERS="${NUM_LAYERS:-32}"
HIDDEN_SIZE="${HIDDEN_SIZE:-4096}"
FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE:-21504}"
NUM_ATTN_HEADS="${NUM_ATTN_HEADS:-32}"
NUM_QUERY_GROUPS="${NUM_QUERY_GROUPS:-8}"

if [[ "$CI_FAST" == "1" ]]; then
  NUM_LAYERS="${NUM_LAYERS_CI:-4}"
  HIDDEN_SIZE="${HIDDEN_SIZE_CI:-512}"
  FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE_CI:-2048}"
  NUM_ATTN_HEADS="${NUM_ATTN_HEADS_CI:-8}"
  NUM_QUERY_GROUPS="${NUM_QUERY_GROUPS_CI:-2}"
fi

NETWORK_SIZE_ARGS=(
  --num-layers "$NUM_LAYERS"
  --hidden-size "$HIDDEN_SIZE"
  --ffn-hidden-size "$FFN_HIDDEN_SIZE"
  --num-attention-heads "$NUM_ATTN_HEADS"
  --group-query-attention
  --num-query-groups "$NUM_QUERY_GROUPS"
  --max-position-embeddings "$SEQ_LEN"
  --position-embedding-type rope
  --rotary-base 500000
  --use-rope-scaling
  --rope-scaling-factor 8
  --make-vocab-size-divisible-by 128
  --normalization RMSNorm
  --xielu
  --qk-layernorm
  --qknorm-impl apex
  --untie-embeddings-and-output-weights
)

LOGGING_ARGS=(
  --log-throughput
  --tensorboard-dir "$TENSORBOARD_DIR"
  --no-log-loss-scale-to-tensorboard
  --log-memory-to-tensorboard
)

REGULARIZATION_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --weight-decay 0.1
  --clip-grad 0.1
  --adam-beta1 0.9
  --adam-beta2 0.999
  --ademamix-alpha 8
  --ademamix-beta3 0.9999
  --ademamix-beta3-warmup 100000
  --ademamix-alpha-warmup 100000
)

TRAINING_ARGS=(
  --micro-batch-size "$MBS"
  --global-batch-size "$GBS"
  --no-check-for-nan-in-loss-and-grad
  --train-iters "$TRAINING_STEPS"
  --log-interval 1
  --cross-entropy-loss-fusion
  --disable-bias-linear
  --optimizer ademamix
  --dataloader-type single
  --manual-gc
  --manual-gc-interval 500
  --exit-signal-handler
  --eval-interval 10000000
  --eval-iters 0
)

INITIALIZATION_ARGS=(
  --seed 41
  --init-method-std 0.008944
)

LEARNING_RATE_ARGS=(
  --lr 0.00011
  --min-lr 0.000011
  --lr-decay-style WSD
  --lr-warmup-iters "${LR_WARMUP_ITERS:-1000}"
  --lr-wsd-decay-style minus_sqrt
  --lr-wsd-decay-iters "${LR_WSD_DECAY_ITERS:-0}"
)

# In CI we don’t want async-save / frequent ckpt I/O.
CHECKPOINTING_ARGS=(
  --load "$CKPT_DIR"
  --save "$CKPT_DIR"
  --save-interval "$CHECKPOINT_STEPS"
  --ckpt-format torch_dist
  --dist-ckpt-strictness assume_ok_unexpected
  --override-opt_param-scheduler
  --trigger-path "${TRIGGER_DIR}"
)

if [[ "$CI_FAST" != "1" ]]; then
  CHECKPOINTING_ARGS+=( --async-save --ckpt-fully-parallel-load )
fi

MIXED_PRECISION_ARGS=( --bf16 )

# Distributed: keep the flags, but allow CI to force TP=1 to fit small jobs.
TP="${TP:-2}"
PP="${PP:-1}"
if [[ "$CI_FAST" == "1" ]]; then
  TP="${TP_CI:-1}"
  PP="${PP_CI:-1}"
fi

DISTRIBUTED_ARGS=(
  --tensor-model-parallel-size "$TP"
  --pipeline-model-parallel-size "$PP"
  --use-distributed-optimizer
  --overlap-grad-reduce
  --overlap-param-gather
)

TOKENIZER_ARGS=( --tokenizer-type NullTokenizer --vocab-size 32000 )

DATA_ARGS=(
  --split 100,0,0
  --seq-length "$SEQ_LEN"
  --reset-position-ids
  --reset-attention-mask
  --eod-mask-loss
  --num-workers 0
  --num-dataset-builder-threads 1
  --goldfish-loss
  --goldfish-k 50
  --goldfish-h 50
  --mock-data
)

# ------------------------------------------------------------
# Launch
# ------------------------------------------------------------
echo "START TIME: $(date)"
echo "MOCK_DATA=$MOCK_DATA CI_FAST=$CI_FAST TP=$TP PP=$PP SEQ_LEN=$SEQ_LEN TRAIN_ITERS=$TRAINING_STEPS GBS=$GBS"

exec "$PYTHON" "$MEGATRON_LM_DIR/pretrain_gpt.py" \
  "${TRANSFORMER_ENGINE_ARGS[@]}" \
  "${NETWORK_SIZE_ARGS[@]}" \
  "${LOGGING_ARGS[@]}" \
  "${REGULARIZATION_ARGS[@]}" \
  "${TRAINING_ARGS[@]}" \
  "${INITIALIZATION_ARGS[@]}" \
  "${LEARNING_RATE_ARGS[@]}" \
  "${CHECKPOINTING_ARGS[@]}" \
  "${MIXED_PRECISION_ARGS[@]}" \
  "${DISTRIBUTED_ARGS[@]}" \
  "${TOKENIZER_ARGS[@]}" \
  "${DATA_ARGS[@]}"
