#!/bin/bash
#SBATCH --job-name=kda-120b-moe-256e-latent-kda31-nope-muonmd-lr6.16e-3-mlr3.08e-2-chonk
#SBATCH --nodes=256
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=36
#SBATCH --mem=800G
#SBATCH --time=12:00:00
#SBATCH --dependency=singleton   # one job of this name at a time: a resubmit queues instead of double-writing CKPT_DIR
#SBATCH --no-requeue             # a crashed job is not restarted by Slurm, so its logs survive
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# chonk.sh - ~110B-parameter KDA MoE pre-training (chonk), one self-contained script.
#
# Source: swiss-ai/pretrain@6cb927e8b153e2fe9cd660fe0a6332c2ee7ce2a2
#         apertus-2/pretraining/chonk/chonk.sh
# Local changes: use ../environments/chonk.toml and select the account at submission.
# This is a 256-node production configuration, not a smoke test.
#
# This is the launch framework of swiss-ai/pretrain, branch chonk @ b463bd4, flattened:
#   sizes/scaling-ladder/kda-120b-moe-256e-latent-kda31-nope.sh + recipes/md_decoupling.sh + clusters/alps3.sh
#   + lib/common.sh + lib/kda_fused.sh + lib/triton_warm.sh + train.sbatch + launch/install_python_deps.sh,
# i.e. what `bash _research/launch/framework/submit.sh --size scaling-ladder/kda-120b-moe-256e-latent-kda31-nope
# --recipe md_decoupling` launches. The Megatron argument list is reproduced token for token (DRY_RUN=1 prints
# it), plus --kda-legacy-gate-out-proj-bias; the README lists where the run itself differed.
#
# Usage (submit from WORK_DIR):
#   mkdir -p logs
#   SBATCH_ACCOUNT=<account> bash chonk.sh --warmup
#   sbatch -A <account> chonk.sh
#   DRY_RUN=1 bash chonk.sh
# Repeat the warm-up until it reports SATURATED. Then submit training.
# Training resumes from CKPT_DIR if a checkpoint exists.
# DRY_RUN prints the arguments and environment without starting a job.
#
# The warm-up is not optional: a cold start JIT-compiles ~390 Triton kernels per node during the first
# iteration, at 64+ nodes the inter-node skew exceeds DeepEP's device-side barrier, and the job hangs at
# iteration 0. At 32 nodes the skew is small, and the kernels are keyed on tensor shapes (model config, TP,
# head geometry), so a 32-node archive is valid for the big run. Training refuses to start without an
# archive unless ALLOW_COLD_START=1. Every warm-up pass seeds from the existing archive, compiles what is
# missing and writes the union back, so re-running accumulates until no new kernel appears.
# The warm-up never runs on the training allocation: `bash chonk.sh --warmup` is the entry point and submits
# the 32-node job itself; `sbatch chonk.sh --warmup` on more than WARMUP_NODES nodes, or under the training
# job's name, is refused at startup. Extra sbatch options for the warm-up (partition, reservation) go through
# Slurm's SBATCH_* environment variables, e.g. SBATCH_RESERVATION=... bash chonk.sh --warmup.

set -euo pipefail

WARMUP=0
while [ $# -gt 0 ]; do
    case $1 in
        --warmup) WARMUP=1; shift ;;
        *) echo "Unknown argument: $1"; echo "Usage: $0 [--warmup]"; exit 1 ;;
    esac
done

# Under sbatch, $0 is a copy in Slurm's spool directory; the submitted path is in the job record. A failed scontrol
# (busy controller) must not end the job under set -e: the spool copy still runs, and the checks below report what
# the fallback breaks (the tokenizer default next to this file).
SELF=""
if [ -n "${SLURM_JOB_ID:-}" ]; then
    SELF=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | sed -n 's/.*Command=\([^ ]*\).*/\1/p' | head -1 || true)
    [ -f "$SELF" ] || echo "WARNING: could not read this job's script path from scontrol; falling back to ${BASH_SOURCE[0]}"
fi
[ -f "$SELF" ] || SELF=$(readlink -f "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(cd "$(dirname "$SELF")" && pwd)
JOB_NAME=$(sed -n 's/^#SBATCH --job-name=\([^ ]*\).*/\1/p' "$SELF" | head -1)   # the training job's singleton name (header above)
[ -n "$JOB_NAME" ] || { echo "ERROR: no '#SBATCH --job-name=' line in $SELF"; exit 1; }

## Megatron-LM-MoE is cloned on first use when MEGATRON_LM_DIR does not hold it yet (shallow, into a temporary directory
## and published with one rename, so jobs starting together cannot see a half-written checkout). An existing checkout is
## used as it is: nothing is pulled or switched, so pin a version by cloning it yourself or by setting MEGATRON_REF.
fetch_megatron() {   # $1 = target dir, $2 = repository URL, $3 = branch or tag
    local dir=$1 repo=$2 ref=$3 tmp   # tmp: unique per call (not the PID: two nodes of one filesystem can share it)
    [ -f "$dir/pretrain_gpt.py" ] && return 0
    if [ -n "$(ls -A "$dir" 2>/dev/null)" ]; then echo "ERROR: $dir exists but holds no pretrain_gpt.py; remove it or point MEGATRON_LM_DIR elsewhere"; return 1; fi
    echo "megatron  cloning $repo ($ref) into $dir"
    mkdir -p "$(dirname "$dir")" && tmp=$(mktemp -d "$dir.clone.XXXXXX") \
        && git clone --quiet --depth 1 --branch "$ref" "$repo" "$tmp" \
        || { [ -z "${tmp:-}" ] || rm -rf "$tmp"; echo "ERROR: could not clone $repo ($ref); clone it yourself and set MEGATRON_LM_DIR"; return 1; }
    mv -T "$tmp" "$dir" 2>/dev/null || { rm -rf "$tmp"; [ -f "$dir/pretrain_gpt.py" ] || { echo "ERROR: could not move the clone to $dir"; return 1; }; }
}

## Paths. Defaults are the production Clariden paths; override from the environment. ##
MEGATRON_REPO=${MEGATRON_REPO:-https://github.com/swiss-ai/Megatron-LM-MoE.git}
MEGATRON_REF=${MEGATRON_REF:-main}                         # branch or tag cloned when MEGATRON_LM_DIR does not exist yet
MEGATRON_LM_DIR=${MEGATRON_LM_DIR:-/ritom/scratch/cscs/$USER/megatron-apertus-moe}  # cloned from MEGATRON_REPO on first use
CONTAINER_ENV=${CONTAINER_ENV:-$SCRIPT_DIR/../environments/chonk.toml}   # local Chonk EDF
DATA_ROOT=${DATA_ROOT:-/ritom/scratch/cscs/$USER/chonk-stage1-every6}   # stage-1 blend (every 6th shard of apertus-v2-20260722/first_stage)
TOKENIZER_MODEL=${TOKENIZER_MODEL:-$SCRIPT_DIR/../../tokenizer/apertus-mul-200k-tokenizer}  # apertus multilingual 200k tokenizer (HF json)
WORK_DIR=${WORK_DIR:-$MEGATRON_LM_DIR/_research}            # checkpoints, tensorboard, wandb, nsys, pip packages, Triton caches and archive
DATASET_CACHE_DIR=${DATASET_CACHE_DIR:-$MEGATRON_LM_DIR/cache/dataset}
WANDB_PROJECT=${WANDB_PROJECT:-apertus-moe-arch-ablations_lc}
WANDB_ENTITY=${WANDB_ENTITY:-apertus}
# WANDB_API_KEY is read from the environment. Leave it empty to disable WandB logging.

if [ -z "${DRY_RUN:-}" ]; then fetch_megatron "$MEGATRON_LM_DIR" "$MEGATRON_REPO" "$MEGATRON_REF" || exit 1
elif [ ! -f "$MEGATRON_LM_DIR/pretrain_gpt.py" ]; then echo "megatron  (DRY_RUN) $MEGATRON_LM_DIR is absent; a real run clones $MEGATRON_REPO ($MEGATRON_REF) there"; fi
[ -e "$CONTAINER_ENV" ]                    || { echo "ERROR: container definition not found: $CONTAINER_ENV"; exit 1; }
[ -d "$DATA_ROOT" ]                        || { echo "ERROR: DATA_ROOT not found: $DATA_ROOT"; exit 1; }
[ -f "$TOKENIZER_MODEL/tokenizer.json" ]   || { echo "ERROR: no tokenizer.json under TOKENIZER_MODEL=$TOKENIZER_MODEL (submit from this folder or set TOKENIZER_MODEL)"; exit 1; }

## Run identity (= the framework's EXP_NAME, so CKPT_DIR is the original run's directory). ##
RUN_TAG=${RUN_TAG-chonk}                                    # the original run's tag; RUN_TAG= drops it
EXP_NAME=${EXP_NAME:-scaling-ladder/kda-120b-moe-256e-latent-kda31-nope-muonmd-lr6.16e-3-mlr3.08e-2-chonk${RUN_TAG:+-$RUN_TAG}}
CKPT_DIR=${CKPT_DIR:-$WORK_DIR/results/ckpts/$EXP_NAME}
TENSORBOARD_DIR=$WORK_DIR/results/runs/tb-${SLURM_JOB_ID:-interactive}
WANDB_SAVE_DIR=$WORK_DIR/results/runs
NSYS_DIR=$WORK_DIR/results/nsys/${SLURM_JOB_ID:-interactive}
PACKAGE_DIR=$WORK_DIR/packages                              # wandb, omegaconf (+ emerging_optimizers if the image lacks it)
FLA_DIR=$WORK_DIR/packages-fla                              # flash-linear-attention 0.5.2, ahead of the container's 0.4.0 on PYTHONPATH
JITCACHE_ROOT=$WORK_DIR/jitcache                            # per-node Triton/inductor caches: jitcache/triton/job-<id>/node-<n>
TRITON_TAR=${TRITON_TAR:-$WORK_DIR/triton-warm/kda-120b-moe-256e-latent-kda31-nope.tar}  # warm Triton archive: written by --warmup, seeded by training
MIN_TRITON_ARCHIVE_KEYS=${MIN_TRITON_ARCHIVE_KEYS:-300}     # sanity floor; this config saturates at ~390 keys

## Model: 42 layers, hidden 3584, 256 latent experts (top-8, latent 1792), 3 KDA : 1 global attention, NoPE. ##
NUM_LAYERS=42
HIDDEN=3584
FFN_HIDDEN=8960               # dense-layer MLP
NUM_HEADS=28                  # head_dim = 128
NUM_KV_HEADS=7                # GQA 4:1 on the global softmax layers
SEQ_LEN=4096
INIT_STD=0.0167038            # 1/sqrt(HIDDEN)

## Parallelism and batch. TP1/PP4/EP16/CP1: 256 nodes -> DP = 1024/4 = 256, expert-DP = 16. ##
MBS=2
GBS=4096
TP=1
PP=4
EP=16
CP=1                          # KDA requires context_parallel_size 1

## Token budget: 1T tokens = 59,604 steps of 4096 x 4096. ##
TRAIN_SAMPLES=244137984       # sequences; = LR_DECAY_SAMPLES, so the LR decays all the way to MIN_LR
LR_DECAY_SAMPLES=244137984
LR_WARMUP_SAMPLES=409600      # 100 steps
SAVE_INTERVAL=200
EXIT_DURATION_MINS=${EXIT_DURATION_MINS:-705}    # exit cleanly 15 min before the 12 h limit
DIST_TIMEOUT_MIN=${DIST_TIMEOUT_MIN:-20}         # rank 0 must survive the one-time dataset index build at the barrier

## md_decoupling recipe: scalar / matrix (Muon) / gains learning rates, all flooring at MIN_LR. ##
LR=6.16e-3
MLR=3.08e-2
GAINS_LR=6.16e-3
MIN_LR=1e-4

## nsys profile of iterations 50-51 on ranks 64 and 128 (the size file has it on; NSYS_PROFILER=false disables it). ##
NSYS_PROFILER=${NSYS_PROFILER:-true}
NSYS_RANKS="64 128"
NSYS_START_ITER=50
NSYS_END_ITER=51

## Warm-up pass (--warmup). ##
WARMUP_NODES=${WARMUP_NODES:-32}           # half the training size, as the production_kda reference: compile skew is harmless here
WARMUP_TIME=${WARMUP_TIME:-01:00:00}       # WARMUP_ITERS iterations at a fraction of the training DP, plus the cold first iteration
WARMUP_ITERS=${WARMUP_ITERS:-6}            # iterations before --exit-interval stops the run
WARMUP_DATA_SHARDS=${WARMUP_DATA_SHARDS:-8}  # kernels are keyed on tensor shapes, not data: a few shards skip the big index build

## Entry points. Training runs under sbatch only. A warm-up started with bash on a login node submits this script at
## the warm-up size under its own job name (<job-name>-warmup: its own singleton slot and log files, the training
## job's --dependency=singleton is untouched). A warm-up that is already under sbatch runs in place only when the
## allocation is no larger than WARMUP_NODES and does not carry the training job's name; the rest of the environment
## (MEGATRON_LM_DIR, WARMUP_NODES, ... overrides) reaches the submitted job through sbatch's default --export=ALL.
if [ -z "${DRY_RUN:-}" ]; then
    if [ -z "${SLURM_JOB_ID:-}" ]; then
        [ "$WARMUP" = 1 ] || { echo "ERROR: training is submitted with 'sbatch $(basename "$SELF")' (DRY_RUN=1 prints the argv instead)"; exit 1; }
        mkdir -p logs   # the #SBATCH --output path is relative to the submit directory, as for 'sbatch $(basename "$SELF")'
        WARMUP_JOBID=$(sbatch --parsable --nodes="$WARMUP_NODES" --time="$WARMUP_TIME" --job-name="$JOB_NAME-warmup" "$SELF" --warmup)
        echo "warm-up: submitted job $WARMUP_JOBID ($WARMUP_NODES nodes, $WARMUP_TIME, job name $JOB_NAME-warmup, log logs/$JOB_NAME-warmup-$WARMUP_JOBID.out)"
        exit 0
    elif [ "$WARMUP" = 1 ]; then
        [ "${SLURM_JOB_NUM_NODES:-0}" -le "$WARMUP_NODES" ] \
            || { echo "ERROR: --warmup on $SLURM_JOB_NUM_NODES nodes; a warm-up pass needs WARMUP_NODES=$WARMUP_NODES. Submit it with 'bash $(basename "$SELF") --warmup'."; exit 1; }
        [ "${SLURM_JOB_NAME:-}" != "$JOB_NAME" ] \
            || { echo "ERROR: --warmup under the training job name $JOB_NAME would take its singleton slot. Submit it with 'bash $(basename "$SELF") --warmup'."; exit 1; }
    fi
fi

echo "START TIME: $(date)"

## Environment (inherited by every srun task). ##
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_CPP_LOG_LEVEL=WARNING
export OMP_NUM_THREADS=18
export NUMEXPR_NUM_THREADS=18
export NUMEXPR_MAX_THREADS=512
export TOKENIZERS_PARALLELISM=false
export TMPDIR=/tmp/$USER-${SLURM_JOB_ID:-$$}
export NCCL_GRAPH_REGISTER=0              # CUDA graphs on the router path
# UCCL / DeepEP tuning for the flex dispatcher at EP=16 with fp8 dispatch (read by UCCL, not Megatron):
# num_sms, nvl_chunked_send, nvl_chunked_recv, rdma_chunked_send, rdma_chunked_recv.
export UCCL_EP_TRANSPORT=cxi
export UCCL_EP_DISPATCH_CONFIG=24,40,512,32,512
export UCCL_EP_COMBINE_CONFIG=24,7,512,32,512
export UCCL_EP_CPU_TIMEOUT_SECS=6000
# Fused KDA runtime levers, resolved by kimi_delta_attention.py at layer construction. A_log stays PER HEAD (28,):
# the checkpoint was trained without KDA_ALOG_PER_CHANNEL and would not load with it.
KDA_LEVERS="KDA_USE_GATE_IN_KERNEL=1 KDA_FUSED_GATE=1 KDA_QK_L2NORM_IN_KERNEL=1 KDA_CONV1D_CUDA=1 KDA_FUSED_LOW_RANK_GATHER=1"

if [ -z "${DRY_RUN:-}" ]; then
    # Not `export X=$(...)`: export returns 0 whatever the substitution did, and an empty MASTER_ADDR only surfaces
    # as a rendezvous timeout 2048 ranks later. head closes the pipe early, hence the explicit || true under pipefail.
    MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1 || true)
    [ -n "$MASTER_ADDR" ] || { echo "ERROR: could not resolve the first node of $SLURM_JOB_NODELIST"; exit 1; }
else
    MASTER_ADDR=$(hostname)
fi
export MASTER_ADDR
export MASTER_PORT=${MASTER_PORT:-25678}

## Megatron args. Check megatron/training/arguments.py for more information. ##
TRANSFORMER_ENGINE_ARGS=(
    --transformer-impl transformer_engine
    --main-grads-dtype fp32
)

NETWORK_SIZE_ARGS=(
    --num-layers $NUM_LAYERS
    --hidden-size $HIDDEN
    --ffn-hidden-size $FFN_HIDDEN
    --num-attention-heads $NUM_HEADS
    --group-query-attention
    --num-query-groups $NUM_KV_HEADS
    --max-position-embeddings $SEQ_LEN
    --position-embedding-type rope
    --rotary-base 10000
    --make-vocab-size-divisible-by 128
    --normalization RMSNorm
    --qk-layernorm
    --untie-embeddings-and-output-weights
    --attention-backend auto
    --seq-length $SEQ_LEN
    --scale-embeddings-by-sqrt-hidden   # embedding x sqrt(hidden): RMS ~1 entering the network with INIT_STD = 1/sqrt(hidden)
    --residual-output-scaling           # x = x + f(x)/sqrt(2*num_layers), the single depth-scaling mechanism
    # Hybrid attention: 3 KDA : 1 global over 42 layers (1 = KDA, 0 = global), globals at 4,8,...,40,42. NoPE everywhere.
    --experimental-attention-variant kda
    --linear-attention-safe-output-gate # bounded decay gate (needed for FlashKDA inference)
    --attention-output-gate             # gates only the global softmax layers
    # The KDA output-gate projection (gate_out_proj) carries a bias in this checkpoint: the run trained before Megatron-LM-MoE
    # 34d15931c ("disable bias arg for kda gate out projection", 2026-09-12), when the bias was hardcoded. Since then it follows
    # --disable-bias-linear, so without this flag the bias is dropped and the checkpoint does not load. Needs a Megatron at or after that commit.
    --kda-legacy-gate-out-proj-bias
    --linear-attention-freq "([1,1,1,0]*10+[1,0])"
    --no-rope-freq 1
    --linear-conv-kernel-dim 4
    --linear-key-head-dim 128
    --linear-value-head-dim 128
    --linear-num-key-heads 28           # HIDDEN/128
    --linear-num-value-heads 28
    --sssglu
    --sandwich-norm
    --enable-experimental
)

MOE_ARGS=(
    --num-experts 256
    --moe-router-topk 8
    --moe-ffn-hidden-size 1920
    --moe-shared-expert-intermediate-size 1920
    --moe-latent-size 1792
    --moe-layer-freq "([0]*3+[1]*39)"   # first 3 layers dense
    --expert-tensor-parallel-size 1     # experts sharded over EP only
    --pipeline-model-parallel-layout "Ettt|(tttttt|)*2,(ttttt|)*2,(tttttt|)*2,tttttL"
    # DeepSeek-V3-style routing: sigmoid gate, top-k renorm scaled by 2.5, fp32 router math, quantile balancing (QB).
    --moe-router-score-function sigmoid
    --moe-router-topk-scaling-factor 2.5
    --moe-router-dtype fp32
    --moe-router-load-balancing-type quantile_balancing
    --moe-aux-loss-coeff 1e-3
    --moe-grouped-gemm
    --moe-permute-fusion
    --moe-token-dispatcher-type flex
    --moe-flex-dispatcher-backend deepep
    --moe-per-layer-logging
    # Overlap the expert-parallel all-to-all with compute; fp8-quantize the tokens crossing it.
    --overlap-moe-expert-parallel-comm
    --delay-wgrad-compute
    --moe-use-fp8-dispatch
    # Expert offloading: fp8 expert params staged in host memory, streamed back in 8 chunks / 2 stages.
    --moe-use-offloading-experts
    --moe-offloading-num-chunks 8
    --moe-offloading-num-stages 2
    --moe-use-inplace-fp8-param
    --moe-use-extra-fp8-param-storage
    --moe-offloading-mode coarse-grained
    --moe-offload-main-grad
    --moe-offload-activations moe_input moe_fc1_output
    # CUDA graphs on the router path. Two separate tokens for the scope, not one quoted string.
    --cuda-graph-impl transformer_engine
    --cuda-graph-scope moe_router moe_preprocess
)

TRAINING_ARGS=(
    --micro-batch-size $MBS
    --global-batch-size $GBS
    --exit-duration-in-mins $EXIT_DURATION_MINS
    --log-interval 1
    --eval-iters 0
    --optimizer md_decoupling
    --dataloader-type single
    --manual-gc
    --manual-gc-interval 100
    --log-memory-interval 20
    --log-device-memory-used
    --cross-entropy-loss-fusion
    --disable-bias-linear
    --no-create-attention-mask-in-dataloader
    --goldfish-loss
    --goldfish-k 50
    --goldfish-h 50
    --distributed-timeout-minutes $DIST_TIMEOUT_MIN
    # Rerun engine: a step whose grad norm exceeds 5.0 is recomputed instead of applied. The per-step NaN check stays off.
    --no-check-for-nan-in-loss-and-grad
    --check-grad-norm
    --check-grad-norm-threshold 5.0
    # Selective activation recompute.
    --recompute-granularity selective
    --recompute-modules layernorm linear_attn qkv shared_experts
)

REGULARIZATION_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --weight-decay 0.0
    --adam-beta1 0.9
    --adam-beta2 0.95
)

# md_decoupling: hypersphere-normalized matrices (direction) + learnable per-axis gains (magnitude) + Muon updates.
# Embedding and LM head use the Adam branch. Incompatible with --use-distributed-optimizer; the layer-wise
# distributed optimizer shards the state instead. --overlap-param-gather is omitted: it corrupts the Newton-Schulz step.
OPTIMIZER_ARGS=(
    --matrix-lr $MLR
    --gains-lr $GAINS_LR
    --min-lr-mode absolute              # every group floors at --min-lr
    --use-orthogonal-updates
    --muon-coefficient-type polar_express
    --muon-scale-mode shape_up
    --muon-router-scale-mode none
    --muon-momentum 0.95
    --muon-use-nesterov
    --hypersphere-mode row
    --hypersphere-embedding-mode row
    --hypersphere-router-mode row
    --hypersphere-radius-mode fan_in
    --md-router-use-orthogonal-updates True
    --hypersphere-gains-mode rowcol
    --hypersphere-gains-mode-router rowcol
    --hypersphere-gains-mode-embedding none
    --gain-parametrization softplus
    --use-layer-wise-distributed-optimizer
    --overlap-grad-reduce
    --log-muon-gains
    --log-muon-sparsity
    --log-muon-per-layer
    --muon-sparsity-thresholds 1e-8 1e-12
    --muon-log-interval 200
)

LEARNING_RATE_ARGS=(
    --lr $LR
    --min-lr $MIN_LR
    --lr-decay-style linear
    --lr-warmup-samples $LR_WARMUP_SAMPLES
    --lr-decay-samples $LR_DECAY_SAMPLES
)

INITIALIZATION_ARGS=(
    --seed 42
    --init-method-std $INIT_STD
)

MIXED_PRECISION_ARGS=(
    --bf16
)

DISTRIBUTED_ARGS=(
    --tensor-model-parallel-size $TP
    --pipeline-model-parallel-size $PP
    --context-parallel-size $CP
    --expert-model-parallel-size $EP
)

LOGGING_ARGS=(
    --log-throughput
    --log-progress
    --log-params-norm
    --log-memory-to-tensorboard
    --tensorboard-dir "$TENSORBOARD_DIR"
    --tensorboard-log-interval 1
    --log-timers-to-tensorboard
)

TOKENIZER_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "$TOKENIZER_MODEL"
)

# Data: every .bin/.idx shard prefix under DATA_ROOT; Megatron infers token-proportional blend weights.
# Above 256 shards the list goes through a file (--data-args-path), which is the same blend but keeps argv under ARG_MAX.
# LC_ALL=C: the run's shard list was sorted in the C locale; under en_US.UTF-8 the same names sort differently, and the
# blend index depends on the order of the list.
mapfile -t DATA_PATH_LIST < <(find -L "$DATA_ROOT" -type f \( -name '*.bin' -o -name '*.idx' \) | sed -E 's/\.[^.]+$//' | LC_ALL=C sort -u)
[ ${#DATA_PATH_LIST[@]} -gt 0 ] || { echo "ERROR: no .bin/.idx shards under $DATA_ROOT"; exit 1; }
if [ "$WARMUP" = 1 ] && [ ${#DATA_PATH_LIST[@]} -gt "$WARMUP_DATA_SHARDS" ]; then
    DATA_PATH_LIST=("${DATA_PATH_LIST[@]:0:$WARMUP_DATA_SHARDS}")
fi
if [ ${#DATA_PATH_LIST[@]} -gt 256 ]; then
    DATA_ARGS_FILE=$DATASET_CACHE_DIR/train_data_paths-${SLURM_JOB_ID:-local}.txt
    [ -n "${DRY_RUN:-}" ] && DATA_ARGS_FILE=$TMPDIR/train_data_paths-dry.txt
    mkdir -p "$(dirname "$DATA_ARGS_FILE")"
    printf '%s\n' "${DATA_PATH_LIST[@]}" > "$DATA_ARGS_FILE"
    DATA_ARGS=(--data-args-path "$DATA_ARGS_FILE" --split 100,0,0)
else
    DATA_ARGS=(--train-data-path "${DATA_PATH_LIST[@]}")
fi
DATA_ARGS+=(
    --data-cache-path "$DATASET_CACHE_DIR"
    --num-workers 1
)

NSYS_SELECT=""
if [ "$WARMUP" = 1 ]; then
    # A short run from random init with the full argument list: no checkpoint is written or read,
    # --exit-interval stops it, and --train-samples is shrunk so the dataset index build is instant.
    RUN_ARGS=(
        --train-samples $(( 10 * GBS ))
        --eval-interval $(( 10 * GBS ))
        --exit-interval $WARMUP_ITERS
    )
    export WANDB_MODE=disabled
else
    RUN_ARGS=(
        --train-samples $TRAIN_SAMPLES
        --eval-interval $TRAIN_SAMPLES
        --save "$CKPT_DIR"
        --load "$CKPT_DIR"
        --save-interval $SAVE_INTERVAL
        --save-iters "1,2,4,8,16,32,64,128,256,512"
        --ckpt-format torch_dist
    )
    if [ "$NSYS_PROFILER" = true ]; then
        RUN_ARGS+=(
            --profile
            --profile-step-start $NSYS_START_ITER
            --profile-step-end $NSYS_END_ITER
            --profile-ranks $NSYS_RANKS       # unquoted on purpose: two tokens, "64 128"
        )
        # Only the profiled ranks get the nsys wrapper; the choice is made inside the task, where SLURM_PROCID exists.
        NSYS_SELECT="NSYS=''; case \" $NSYS_RANKS \" in *\" \$SLURM_PROCID \"*) NSYS='nsys profile -s none --trace=nvtx,cudnn,cublas,cuda --output=$NSYS_DIR/nsys-${EXP_NAME##*/}-rank'\$SLURM_PROCID' --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop';; esac;"
    fi
    if [ -n "${WANDB_API_KEY:-}" ]; then
        export WANDB_PROJECT
        export WANDB_NAME=$EXP_NAME-${SLURM_JOB_ID:-interactive}
        RUN_ARGS+=(
            --wandb-project "$WANDB_PROJECT"
            --wandb-exp-name "$EXP_NAME-${SLURM_JOB_ID:-interactive}"
            --wandb-save-dir "$WANDB_SAVE_DIR"
        )
        if [ -n "$WANDB_ENTITY" ]; then
            export WANDB_ENTITY
            RUN_ARGS+=(--wandb-entity "$WANDB_ENTITY")
        fi
    else
        export WANDB_MODE=disabled
    fi
fi

MEGATRON_ARGS=(
    "${TRANSFORMER_ENGINE_ARGS[@]}"
    "${NETWORK_SIZE_ARGS[@]}"
    "${MOE_ARGS[@]}"
    "${TRAINING_ARGS[@]}"
    "${REGULARIZATION_ARGS[@]}"
    "${OPTIMIZER_ARGS[@]}"
    "${LEARNING_RATE_ARGS[@]}"
    "${INITIALIZATION_ARGS[@]}"
    "${MIXED_PRECISION_ARGS[@]}"
    "${DISTRIBUTED_ARGS[@]}"
    "${LOGGING_ARGS[@]}"
    "${TOKENIZER_ARGS[@]}"
    "${DATA_ARGS[@]}"
    "${RUN_ARGS[@]}"
)

## Helpers that run INSIDE the container (shipped to every task through declare -f). ##

# Populate this node's Triton cache from the warm archive, once per node (local rank 0 extracts, the
# others wait). A missing archive is a cold start: the run compiles, which is what the warm-up pass wants.
triton_warm_seed() {
    local dest=$1 tar_path=$2 stamp=$1/.triton_warm_done staging i
    [ -f "$tar_path" ] || { echo "[triton-warm] no archive at $tar_path -- cold start (will JIT)" >&2; mkdir -p "$dest"; return 0; }
    if [ "${SLURM_LOCALID:-0}" = 0 ]; then
        if [ ! -f "$stamp" ]; then
            staging=$dest.staging.$$
            rm -rf "$staging"; mkdir -p "$staging"
            if tar -xf "$tar_path" -C "$staging" 2>/dev/null; then
                : > "$staging/.triton_warm_done"
                # mv -T is one rename(2): it also replaces an EMPTY $dest another rank may just have created, so the
                # stamp can never be seen before the files. A non-empty $dest (an earlier attempt) is merged instead,
                # with the stamp written last.
                if ! mv -T "$staging" "$dest" 2>/dev/null; then
                    rm -f "$staging/.triton_warm_done"; mkdir -p "$dest"
                    cp -a "$staging/." "$dest/" 2>/dev/null; rm -rf "$staging"; : > "$stamp"
                fi
                echo "[triton-warm] seeded $dest from $(basename "$tar_path") ($(ls -1 "$dest" | wc -l) entries)" >&2
            else
                echo "[triton-warm] FAILED to extract $tar_path -- cold start (will JIT)" >&2
                rm -rf "$staging"; mkdir -p "$dest"
            fi
        fi
    else
        for i in $(seq 1 900); do [ -f "$stamp" ] && break; sleep 1; done
        [ -f "$stamp" ] || echo "[triton-warm] localid=$SLURM_LOCALID timed out waiting for the seed -- proceeding" >&2
    fi
    mkdir -p "$dest"
}

# Install the python packages the image lacks into PACKAGE_DIR, once per allocation (marker + lock directory).
# Only container-absent packages, --no-deps where possible: PACKAGE_DIR precedes site-packages on PYTHONPATH.
install_python_deps() {
    local pkg=$1 marker=$1/.deps_installed_moe_v1 lock=$1/.install.lockdir install="pip install --quiet" waited=0
    mkdir -p "$pkg"
    [ -f "$marker" ] && return 0
    while ! mkdir "$lock" 2>/dev/null; do
        [ -f "$marker" ] && return 0
        [ "$waited" -lt 1800 ] || { echo "install_python_deps: $lock held for 30 min without $marker appearing (stale lock from a killed job? rmdir it)" >&2; return 1; }
        sleep 5; waited=$((waited + 5))
    done
    [ -f "$marker" ] && { rmdir "$lock"; return 0; }
    command -v uv >/dev/null 2>&1 && install="uv pip install --quiet"
    { $install --target="$pkg" wandb omegaconf \
        && { python3 -c "import emerging_optimizers" 2>/dev/null \
             || $install --no-deps --target="$pkg" 'git+https://github.com/NVIDIA-NeMo/Emerging-Optimizers.git@v0.2.0'; } \
        && touch "$marker"; } || { rmdir "$lock"; return 1; }
    rmdir "$lock"
}

## Host-side helper: archive the union of every node's compiled kernels (pipeline stages compile different sets),
## plus everything the previous archive ($3) held, so a pass whose nodes failed to seed can never shrink it. ##
triton_warm_build() {
    local src=$1 out=$2 seed=${3:-} merge map d k n_new=0
    merge=$(mktemp -d "$TMPDIR/triton-warm-merge.XXXXXX") || return 1
    map=$(mktemp) || return 1
    for d in "$src"/node-*; do
        [ -d "$d" ] || continue
        for k in $(ls -1 "$d" 2>/dev/null); do printf '%s\t%s/%s\n' "$k" "$d" "$k"; done
    done | sort -u -k1,1 > "$map"
    while IFS=$'\t' read -r k path; do
        [ -n "$k" ] && [ ! -e "$merge/$k" ] || continue
        cp -a "$path" "$merge/$k" 2>/dev/null && n_new=$((n_new + 1))
    done < "$map"
    rm -f "$map"
    [ "$n_new" -gt 0 ] || { echo "triton_warm_build: nothing collected from $src" >&2; rm -rf "$merge"; return 1; }
    if [ -n "$seed" ] && [ -f "$seed" ]; then
        tar -xf "$seed" -C "$merge" --skip-old-files 2>/dev/null || echo "triton_warm_build: could not merge the previous archive $seed" >&2
    fi
    mkdir -p "$(dirname "$out")"
    tar -cf "$out.tmp" -C "$merge" . && mv "$out.tmp" "$out"
    rm -rf "$merge"
}
archive_keys() { tar -tf "$1" | awk -F/ 'NF>1 && $2!=""{print $2}' | sort -u | wc -l; }

## In-container command. Host variables expand now (paths through ${var@Q}, which survives the bash -c round trip
## with spaces or metacharacters), \$SLURM_* and \$TMPDIR resolve inside each task. TRITON_CACHE_DIR is created by
## triton_warm_seed alone, never before it: a mkdir by local ranks 1-3 would race with rank 0's rename of the seed. ##
IN_CONTAINER="$(declare -f triton_warm_seed install_python_deps)
export MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT
export RANK=\$SLURM_PROCID WORLD_SIZE=\$SLURM_NTASKS LOCAL_RANK=\$SLURM_LOCALID LOCAL_WORLD_SIZE=$(( ${SLURM_NTASKS:-4} / ${SLURM_NNODES:-1} ))
export PYTHONPATH=${FLA_DIR@Q}:${MEGATRON_LM_DIR@Q}:${PACKAGE_DIR@Q}
export $KDA_LEVERS
export TRITON_CACHE_DIR=${JITCACHE_ROOT@Q}/triton/job-${SLURM_JOB_ID:-interactive}/node-\$SLURM_NODEID
export TORCHINDUCTOR_CACHE_DIR=${JITCACHE_ROOT@Q}/inductor/job-${SLURM_JOB_ID:-interactive}/node-\$SLURM_NODEID
export TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER=0
mkdir -p \"\$TORCHINDUCTOR_CACHE_DIR\" \"\$TMPDIR\"
triton_warm_seed \"\$TRITON_CACHE_DIR\" ${TRITON_TAR@Q}
cd ${MEGATRON_LM_DIR@Q}
${NSYS_SELECT}
install_python_deps ${PACKAGE_DIR@Q} && \$NSYS python3 pretrain_gpt.py ${MEGATRON_ARGS[*]@Q}"
[ -n "$NSYS_SELECT" ] || IN_CONTAINER=${IN_CONTAINER//\$NSYS /}

if [ -n "${DRY_RUN:-}" ]; then
    echo "=== Megatron argv (DRY_RUN) ==="; printf '%s\n' "${MEGATRON_ARGS[@]}"
    echo "=== in-container command (DRY_RUN) ==="; echo "$IN_CONTAINER" | sed -n '/^export MASTER_ADDR/,$p'
    if [ "$WARMUP" = 1 ]; then
        echo "=== warm-up submission (DRY_RUN) ==="
        echo "sbatch --parsable --nodes=$WARMUP_NODES --time=$WARMUP_TIME --job-name=$JOB_NAME-warmup $SELF --warmup"
    fi
    exit 0
fi

mkdir -p "$WORK_DIR/results/runs" "$DATASET_CACHE_DIR" "$PACKAGE_DIR" "$JITCACHE_ROOT" "$TMPDIR"
[ "$WARMUP" = 1 ] || mkdir -p "$CKPT_DIR"
[ "$WARMUP" = 1 ] || [ "$NSYS_PROFILER" != true ] || mkdir -p "$NSYS_DIR"

## Triton archive check. ##
if [ -f "$TRITON_TAR" ]; then
    SEED_KEYS=$(archive_keys "$TRITON_TAR") || { echo "ERROR: cannot list $TRITON_TAR (corrupt archive? move it away and run the warm-up again)"; exit 1; }
    echo "TRITON ARCHIVE: $TRITON_TAR ($SEED_KEYS keys)"
else
    SEED_KEYS=0
    if [ "$WARMUP" = 1 ]; then
        echo "TRITON ARCHIVE: none yet; this pass compiles from scratch -> $TRITON_TAR"
    elif [ "${ALLOW_COLD_START:-0}" = 1 ]; then
        echo "TRITON ARCHIVE: none at $TRITON_TAR -- ALLOW_COLD_START=1, compiling at run time (may hang at iteration 0)"
    else
        echo "ERROR: no Triton archive at $TRITON_TAR. Run 'bash $(basename "$SELF") --warmup' first (or ALLOW_COLD_START=1)."
        exit 1
    fi
fi

## flash-linear-attention 0.5.2 (+ the swiss-ai decay-gate kernel) into FLA_DIR, once, before the launch.
## The image ships 0.4.0, on which every KDA_* lever disables itself. fla-core is required too: the two
## distributions merge into one fla/ namespace. --no-deps keeps the container's torch/triton untouched.
## The install goes to a temporary directory and is published with one rename: a concurrent job (a warm-up and a
## training job may overlap) or a job killed mid-install can never leave a half-written FLA_DIR that later jobs skip.
install_fla() {   # runs inside the container on one node; $1 = FLA_DIR
    local dst=$1 tmp=$1.tmp.$$
    if [ ! -f "$dst/fla/__init__.py" ]; then
        rm -rf "$tmp"
        pip install --quiet --no-deps --ignore-installed --target="$tmp" 'flash-linear-attention==0.5.2' 'fla-core==0.5.2' || return 1
        curl -sfL https://raw.githubusercontent.com/swiss-ai/flash-linear-attention/main/fla/ops/kda/gate.py -o "$tmp/fla/ops/kda/gate.py" || return 1
        if ! mv -T "$tmp" "$dst" 2>/dev/null; then                      # $dst appeared meanwhile
            if [ -f "$dst/fla/__init__.py" ]; then rm -rf "$tmp"         # another job finished first: keep theirs
            else rm -rf "$dst" && mv -T "$tmp" "$dst" || return 1; fi    # a half-installed leftover: replace it
        fi
    fi
    PYTHONPATH=$dst python3 -c "import fla; assert fla.__version__.startswith('0.5.2'), 'got fla '+fla.__version__; print('fla', fla.__version__, 'OK')"
}
srun --nodes=1 --ntasks=1 --overlap --network=disable_rdzv_get --environment="$CONTAINER_ENV" \
    bash -c "$(declare -f install_fla); install_fla ${FLA_DIR@Q}"

## Auto-requeue (AUTO_REQUEUE=1): queue the successor BEFORE training with --dependency=afterany, so the
## chain survives a crash or a timeout. Same job name -> same CKPT_DIR -> it resumes via --load. When the
## token budget is reached the pending successor is cancelled below. MAX_REQUEUES caps the chain.
training_complete() {
    local marker=$CKPT_DIR/latest_checkpointed_iteration.txt iter
    [ -f "$marker" ] || return 1
    iter=$(tr -dc '0-9' < "$marker"); [ -n "$iter" ] || return 1
    [ "$iter" -ge $(( TRAIN_SAMPLES / GBS )) ]
}
REQUEUE_JOBID=""
if [ "$WARMUP" = 0 ] && [ "${AUTO_REQUEUE:-0}" = 1 ] && [ "${REQUEUE_COUNT:-0}" -lt "${MAX_REQUEUES:-50}" ] && ! training_complete; then
    # The successor copies this job's placement (nodes, time limit, account, partition, QOS, reservation, submit
    # directory), so a command-line override of the header survives every link; an unreadable time limit falls back
    # to the header's. A fresh rendezvous port per link: a lingering TCPStore on the old port hangs torch.distributed
    # init. A failed sbatch (busy controller) is reported, not fatal: this job still trains, the chain just ends here.
    REQUEUE_TIME=$(squeue -h -j "$SLURM_JOB_ID" -o %l 2>/dev/null | tr -d ' ' || true)
    REQUEUE_JOBID=$(sbatch --parsable \
        --dependency=afterany:"$SLURM_JOB_ID",singleton \
        --nodes="$SLURM_NNODES" ${REQUEUE_TIME:+--time="$REQUEUE_TIME"} --job-name="$SLURM_JOB_NAME" \
        ${SLURM_JOB_ACCOUNT:+--account="$SLURM_JOB_ACCOUNT"} --partition="$SLURM_JOB_PARTITION" \
        ${SLURM_JOB_QOS:+--qos="$SLURM_JOB_QOS"} ${SLURM_JOB_RESERVATION:+--reservation="$SLURM_JOB_RESERVATION"} \
        --chdir="$SLURM_SUBMIT_DIR" \
        --export=ALL,AUTO_REQUEUE=1,REQUEUE_COUNT=$(( ${REQUEUE_COUNT:-0} + 1 )),MASTER_PORT=$(( 20000 + RANDOM % 20000 )) \
        "$SELF") || REQUEUE_JOBID=""
    if [ -n "$REQUEUE_JOBID" ]; then
        echo "auto-requeue: pre-submitted job $REQUEUE_JOBID (afterany:$SLURM_JOB_ID, link $(( ${REQUEUE_COUNT:-0} + 1 ))/${MAX_REQUEUES:-50})"
    else
        echo "WARNING: auto-requeue: sbatch failed; this job trains without a successor (resubmit by hand when it ends)"
    fi
fi

echo "MODE:      $([ "$WARMUP" = 1 ] && echo "warm-up ($WARMUP_ITERS iterations, ${#DATA_PATH_LIST[@]} shards)" || echo training)"
echo "EXP_NAME:  $EXP_NAME"
echo "CKPT_DIR:  $CKPT_DIR"
echo "MEGATRON:  $MEGATRON_LM_DIR @ $(git -C "$MEGATRON_LM_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "CONTAINER: $CONTAINER_ENV"
echo "NODES:     $SLURM_NNODES ($SLURM_NTASKS GPUs) | TP$TP PP$PP EP$EP CP$CP -> DP=$(( SLURM_NTASKS / (TP * PP * CP) ))"
echo "CMD:       python3 pretrain_gpt.py ${MEGATRON_ARGS[*]@Q}"

SRUN_RC=0
srun -lu --mpi=pmix --network=disable_rdzv_get \
    --environment="$CONTAINER_ENV" \
    --cpus-per-task "$SLURM_CPUS_PER_TASK" --wait 120 \
    bash -c "$IN_CONTAINER" || SRUN_RC=$?

echo "END TIME: $(date) (srun exit code $SRUN_RC)"

if [ "$WARMUP" = 1 ]; then
    # Harvest whatever compiled, even after a failed pass: the union of the seeded archive and every node's new kernels.
    CACHE_ROOT=$JITCACHE_ROOT/triton/job-$SLURM_JOB_ID
    [ -d "$CACHE_ROOT" ] || { echo "ERROR: no Triton cache at $CACHE_ROOT"; exit 1; }
    triton_warm_build "$CACHE_ROOT" "$TRITON_TAR.new" "$TRITON_TAR" && mv "$TRITON_TAR.new" "$TRITON_TAR" \
        || { echo "ERROR: no compiled kernels under $CACHE_ROOT (srun exit $SRUN_RC); the archive is unchanged"; exit 1; }
    KEYS=$(archive_keys "$TRITON_TAR") || { echo "ERROR: cannot list $TRITON_TAR"; exit 1; }
    echo "ARCHIVE:   $TRITON_TAR ($KEYS keys, $(du -h "$TRITON_TAR" | cut -f1)); seeded $SEED_KEYS, added $(( KEYS - SEED_KEYS ))"
    rm -rf "$CACHE_ROOT" "$JITCACHE_ROOT/inductor/job-$SLURM_JOB_ID"
    if [ "$KEYS" -lt "$MIN_TRITON_ARCHIVE_KEYS" ]; then
        echo "WARNING:   only $KEYS keys, below the $MIN_TRITON_ARCHIVE_KEYS floor. Run the warm-up again; it seeds from this archive and compiles the rest."
        exit 1
    elif [ "$SRUN_RC" != 0 ] && [ "$KEYS" -le "$SEED_KEYS" ]; then
        # Seeded caches alone make KEYS == SEED_KEYS; a pass that died before compiling anything says nothing about saturation.
        echo "WARNING:   the pass failed (srun exit $SRUN_RC) and added no kernels; the archive is unchanged. Fix the failure and run the warm-up again."
        exit 1
    elif [ "$KEYS" -le "$SEED_KEYS" ]; then
        echo "SATURATED: no new kernels this pass. Ready for training."
    else
        # A non-zero exit that still grew the archive is accepted: this model segfaults in teardown (exit 139) after
        # "exiting program at iteration N", with every kernel already compiled.
        [ "$SRUN_RC" = 0 ] || echo "NOTE:      srun exit $SRUN_RC (teardown crash after the iterations); the kernels it compiled are in the archive."
        echo "Added $(( KEYS - SEED_KEYS )) kernels this pass. Run the warm-up again until it reports SATURATED."
    fi
    exit 0
fi

# This job's per-node caches are the archive plus whatever JIT-compiled: hundreds of directories of small files
# per job, which a long chain turns into millions. KEEP_JITCACHE=1 keeps them (e.g. to harvest an ALLOW_COLD_START run).
[ "${KEEP_JITCACHE:-0}" = 1 ] || rm -rf "$JITCACHE_ROOT/triton/job-$SLURM_JOB_ID" "$JITCACHE_ROOT/inductor/job-$SLURM_JOB_ID"

if [ -n "$REQUEUE_JOBID" ]; then
    if training_complete; then
        echo "auto-requeue: token budget reached, cancelling pending job $REQUEUE_JOBID"; scancel "$REQUEUE_JOBID" || true
    else
        echo "auto-requeue: training incomplete, pending job $REQUEUE_JOBID will resume"
    fi
fi
exit "$SRUN_RC"
