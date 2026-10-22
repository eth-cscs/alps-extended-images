#!/usr/bin/env bash
set -Eeuo pipefail
# Keep crash dumps out of the submission directory.
ulimit -c 0

# ----- CLI and recipe selection -----

usage() {
    printf '%s\n' \
        'Usage: bash train.sh RECIPE [--profile NAME] [training options] -- [sbatch options]' \
        'Example: bash train.sh vllm/gsm8k --profile muon --grpo-steps 58 -- --account=YOUR_PROJECT' \
        'RECIPE is a directory relative to this launcher. Select NAME.env with --profile.' \
        'Without --profile, the launcher uses recipe.env.' \
        'Training options: --grpo-steps, --prompts, --generations-per-prompt, --updates-per-rollout,' \
        '  --micro-batch, --train-nodes, --generation-nodes, --inference-requests,' \
        '  --lr, --min-lr, --matrix-lr, --gains-lr, --train-mb-tokens, --logprob-mb-tokens.' \
        'CLI options override the shared .env/environment and profile defaults.' \
        'Use this command rather than direct sbatch; Slurm options must follow --.'
}
if [[ $# == 0 || "${1:-}" == -h || "${1:-}" == --help ]]; then
    usage
    exit 0
fi
recipe="$1"
shift
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then usage; exit 0; fi
mode=submit
case "${1:-}" in
    --batch|--server|--driver) mode="${1#--}"; shift ;;
esac
if [[ "$mode" == submit ]]; then
    [[ -z "${SLURM_JOB_ID:-}" ]] || {
        echo 'Use bash train.sh from outside an allocation, not direct sbatch.' >&2
        exit 2
    }
    LAUNCHER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
else
    [[ $# == 0 ]] || { echo 'Unexpected arguments in internal job mode.' >&2; exit 2; }
    : "${SLURM_JOB_ID:?Internal job modes require Slurm}"
    # Slurm executes a spooled script, so its BASH_SOURCE is not the repository path.
    : "${CHONK_EXAMPLES_DIR:?Submit with bash train.sh RECIPE}"
    LAUNCHER_DIR="$CHONK_EXAMPLES_DIR"
fi
LAUNCHER="$LAUNCHER_DIR/train.sh"
# Save explicit CLI values, or the resolved values inherited by internal modes.
knobs=(TRAIN_NODES INFERENCE_NODES MAX_NUM_STEPS NUM_PROMPTS GENERATIONS_PER_PROMPT
    UPDATES_PER_ROLLOUT TRAIN_MICRO_BATCH INFERENCE_REQUESTS TRAIN_EXPERT_PARALLEL_SIZE
    TRAIN_MB_TOKENS LOGPROB_MB_TOKENS BASE_LR MIN_LR MATRIX_LR GAINS_LR)
declare -A overrides=()
profile="${CHONK_RECIPE_PROFILE:-recipe}"
if [[ "$mode" == submit ]]; then
    profile=recipe
    while (( $# )); do
        case "$1" in
            --) shift; break ;;
            -h|--help) usage; exit 0 ;;
            --*=*) option="${1%%=*}"; value="${1#*=}"; shift ;;
            --*)
                (( $# >= 2 )) || { echo "Missing value for $1" >&2; exit 2; }
                option="$1"; value="$2"; shift 2 ;;
            *) echo "Unexpected argument: $1. Put Slurm options after --." >&2; exit 2 ;;
        esac
        case "$option" in
            --profile) profile="$value"; continue ;;
            --grpo-steps) key=MAX_NUM_STEPS ;;
            --prompts) key=NUM_PROMPTS ;;
            --generations-per-prompt) key=GENERATIONS_PER_PROMPT ;;
            --updates-per-rollout) key=UPDATES_PER_ROLLOUT ;;
            --micro-batch) key=TRAIN_MICRO_BATCH ;;
            --train-nodes) key=TRAIN_NODES ;;
            --generation-nodes) key=INFERENCE_NODES ;;
            --inference-requests) key=INFERENCE_REQUESTS ;;
            --lr) key=BASE_LR ;;
            --min-lr) key=MIN_LR ;;
            --matrix-lr) key=MATRIX_LR ;;
            --gains-lr) key=GAINS_LR ;;
            --train-mb-tokens) key=TRAIN_MB_TOKENS ;;
            --logprob-mb-tokens) key=LOGPROB_MB_TOKENS ;;
            *) echo "Unknown training option: $option. Put Slurm options after --." >&2; exit 2 ;;
        esac
        overrides[$key]="$value"
    done
else
    for key in "${knobs[@]}"; do
        if [[ -v "$key" ]]; then overrides[$key]="${!key}"; fi
    done
fi
[[ "$profile" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid profile name.' >&2; exit 2; }
recipe_profile="$LAUNCHER_DIR/$recipe/$profile.env"
[[ -f "$recipe_profile" ]] || { echo "Profile not found: $recipe/$profile.env" >&2; exit 2; }

# ----- User settings and recipe defaults -----

# Read .env only at submission so queued jobs retain their CLI overrides.
if [[ "$mode" == submit ]] && [[ -f "$LAUNCHER_DIR/.env" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "$LAUNCHER_DIR/.env"
    set +a
fi
recipe_profile="$(readlink -f -- "$recipe_profile")"
set -a
# shellcheck source=/dev/null
source "$recipe_profile"
set +a
for key in "${!overrides[@]}"; do printf -v "$key" '%s' "${overrides[$key]}"; done
EXAMPLE_DIR="${EXAMPLE_DIR:-$(dirname -- "$recipe_profile")}"
EXAMPLE_DIR="$(cd -- "$EXAMPLE_DIR" && pwd -P)"

NEMO_RL_SOURCE="${NEMO_RL_SOURCE:-https://github.com/swiss-ai/Nemo-RL.git}"
BRIDGE_SOURCE="${BRIDGE_SOURCE:-https://github.com/swiss-ai/Megatron-Bridge.git}"
MEGATRON_SOURCE="${MEGATRON_SOURCE:-https://github.com/swiss-ai/Megatron-LM-MoE.git}"
HF_MODEL="${HF_MODEL:-/capstor/store/cscs/swissai/infra01/apertus_checkpoints/v2/hf/chonk/pretraining/long_context/long-context-iter-0003576}"
ENV_FILE="${NEMO_RL_ENVIRONMENT:-$LAUNCHER_DIR/environment.toml}"
RUN_ROOT="${CHONK_RUN_ROOT:-/iopsstor/scratch/cscs/$USER/tmp/$RUN_NAME}"
if [[ "$GYM_ENABLED" == 1 ]]; then
    NEMO_GYM_SOURCE="${NEMO_GYM_SOURCE:-https://github.com/swiss-ai/nemo-gym.git}"
    NEMO_GYM_REF="${NEMO_GYM_REF:-93f232f02074deda2fec13a6848f22e5dca741f5}"
    export NEMO_GYM_SOURCE NEMO_GYM_REF
fi

# ----- Fixed source revisions -----
NEMO_REF=f196ef0a1ccb54d6ce71cd63277fb6a0e01168db
NEMO_FP8_REF=19843e5a523254ebf3bed49a47a1e852dca29a70
NEMO_VLLM_DP_REF=05db6ab36b7a0c528c178cbeb44903dbe82fda5d # Pin PR #2517 to the revision tested with the FP8 cache fix.
MCORE_REF=0536034901962160aac9742d134a84c55e28d224
BRIDGE_REF=077224531ad4d6d0caef767a84e619b256aa5cb6
BRIDGE_PR5_REF=6ccdb3c369491374f1fef4320886e8bb1394bb3e
declare -A NEMO_PR_REFS=(
    [1]=40a766b21dcfc65c9548fd0cc1ff02869c585869
    [5]=fe7e3fe71c80c1630487c09a7d49a27e27a3b71e
    [6]=2bd5dc71c955d9d7bdb12a9d3849529a4ad0d292
    [7]=e7784d7c052b9d13960dafa15694355b4edf660c
    [9]=1abfe7fff686168b0cd97161e8ebe064f67a4c05
)
for pr in "${NEMO_PRS[@]}"; do
    [[ -n "${NEMO_PR_REFS[$pr]:-}" ]] || {
        echo "No pinned commit for NeMo PR #$pr." >&2; exit 2;
    }
done

# ----- Batch sizing and validation -----
TRAIN_MICRO_BATCH="${TRAIN_MICRO_BATCH:-$DEFAULT_MICRO_BATCH}"
INFERENCE_REQUESTS="${INFERENCE_REQUESTS:-32}"
MAX_NUM_STEPS="${MAX_NUM_STEPS:-$DEFAULT_MAX_NUM_STEPS}"
NUM_PROMPTS="${NUM_PROMPTS:-$DEFAULT_NUM_PROMPTS}"
# Check integers before Bash arithmetic; invalid input must fail before submission.
TRAIN_MODEL_PARALLEL_SIZE="${TRAIN_MODEL_PARALLEL_SIZE:-4}"
for key in TRAIN_NODES INFERENCE_NODES TRAIN_MODEL_PARALLEL_SIZE TRAIN_MICRO_BATCH \
    INFERENCE_REQUESTS NUM_PROMPTS MAX_NUM_STEPS GENERATIONS_PER_PROMPT \
    UPDATES_PER_ROLLOUT TRAIN_EXPERT_PARALLEL_SIZE GENERATION_EXPERT_PARALLEL_SIZE \
    TRAIN_MB_TOKENS LOGPROB_MB_TOKENS; do
    [[ ! -v "$key" ]] || [[ "${!key}" =~ ^[1-9][0-9]*$ ]] || {
        echo "$key must be a positive integer." >&2; exit 2;
    }
done
for key in BASE_LR MIN_LR MATRIX_LR GAINS_LR; do
    [[ ! -v "$key" ]] || [[ "${!key}" =~ ^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$ ]] || {
        echo "$key must be a non-negative number." >&2; exit 2;
    }
done
if [[ -n "${BASE_LR:-}" && -n "${MIN_LR:-}" ]]; then
    awk -v base="$BASE_LR" -v minimum="$MIN_LR" 'BEGIN { exit(minimum > base) }' || {
        echo 'Minimum learning rate must not exceed the base learning rate.' >&2; exit 2;
    }
fi
TOTAL_NODES=$((TRAIN_NODES + INFERENCE_NODES))
TOTAL_GPUS=$((TOTAL_NODES * 4))
if [[ -n "${UPDATES_PER_ROLLOUT:-}" ]]; then
    (( (NUM_PROMPTS * GENERATIONS_PER_PROMPT) % UPDATES_PER_ROLLOUT == 0 )) || {
        echo 'Rollout batch must divide into the requested optimizer updates.' >&2; exit 2;
    }
    TRAIN_GLOBAL_BATCH=$((NUM_PROMPTS * GENERATIONS_PER_PROMPT / UPDATES_PER_ROLLOUT))
fi
[[ "${TRAIN_GLOBAL_BATCH:-}" =~ ^[1-9][0-9]*$ ]] || {
    echo 'Derived global batch must be positive.' >&2; exit 2;
}
# TRAIN_MODEL_PARALLEL_SIZE is TP × PP × CP.
(( (TRAIN_NODES * 4) % TRAIN_MODEL_PARALLEL_SIZE == 0 )) || {
    echo 'Training GPUs must divide into complete model-parallel groups.' >&2; exit 2;
}
TRAIN_DATA_PARALLEL_SIZE=$((TRAIN_NODES * 4 / TRAIN_MODEL_PARALLEL_SIZE))
(( TRAIN_GLOBAL_BATCH % (TRAIN_DATA_PARALLEL_SIZE * TRAIN_MICRO_BATCH) == 0 )) || {
    echo "Global batch $TRAIN_GLOBAL_BATCH must divide into complete training microbatches." >&2; exit 2;
}
(( (NUM_PROMPTS * GENERATIONS_PER_PROMPT) % TRAIN_GLOBAL_BATCH == 0 )) || {
    echo 'Rollout batch must divide into training batches.' >&2; exit 2;
}
if [[ -n "${TRAIN_EXPERT_PARALLEL_SIZE:-}" ]]; then
    (( TRAIN_DATA_PARALLEL_SIZE % TRAIN_EXPERT_PARALLEL_SIZE == 0 )) || {
        echo 'Training data-parallel ranks must divide into complete expert-parallel groups.' >&2; exit 2;
    }
fi
if [[ -n "${GENERATION_EXPERT_PARALLEL_SIZE:-}" ]]; then
    (( (INFERENCE_NODES * 4) % GENERATION_EXPERT_PARALLEL_SIZE == 0 )) || {
        echo 'Generation GPUs must divide into complete expert-parallel groups.' >&2; exit 2;
    }
fi
UPDATES_PER_ROLLOUT=$((NUM_PROMPTS * GENERATIONS_PER_PROMPT / TRAIN_GLOBAL_BATCH))
[[ -f "$ENV_FILE" ]] || { echo "EDF not found: $ENV_FILE" >&2; exit 2; }
[[ -f "$EXAMPLE_DIR/$CONFIG_FILE" ]] || { echo "Config not found: $EXAMPLE_DIR/$CONFIG_FILE" >&2; exit 2; }
case "$RUN_ROOT/" in "$HOME/"*) echo 'RUN_ROOT must not be under HOME.' >&2; exit 2 ;; esac

export CHONK_EXAMPLES_DIR="$LAUNCHER_DIR" EXAMPLE_DIR HF_MODEL
export NEMO_RL_SOURCE BRIDGE_SOURCE MEGATRON_SOURCE
export NEMO_RL_ENVIRONMENT="$ENV_FILE" CHONK_RUN_ROOT="$RUN_ROOT"
export TRAIN_NODES INFERENCE_NODES TOTAL_NODES TRAIN_GLOBAL_BATCH GENERATIONS_PER_PROMPT
export TRAIN_MICRO_BATCH INFERENCE_REQUESTS MAX_NUM_STEPS NUM_PROMPTS GYM_ENABLED
export UPDATES_PER_ROLLOUT TRAIN_EXPERT_PARALLEL_SIZE TRAIN_MB_TOKENS LOGPROB_MB_TOKENS
export BASE_LR MIN_LR MATRIX_LR GAINS_LR
export CHONK_RECIPE_PROFILE="$profile"

# ----- Submission -----

if [[ "$mode" == submit ]]; then
    printf '%s GRPO steps × %s updates = %s optimizer updates; global batch %s; %s training + %s generation nodes\n' \
        "$MAX_NUM_STEPS" "$UPDATES_PER_ROLLOUT" "$((MAX_NUM_STEPS * UPDATES_PER_ROLLOUT))" \
        "$TRAIN_GLOBAL_BATCH" "$TRAIN_NODES" "$INFERENCE_NODES" >&2
    command -v sbatch >/dev/null || { echo 'sbatch is not available.' >&2; exit 127; }
    sources=("$NEMO_RL_SOURCE" "$BRIDGE_SOURCE" "$MEGATRON_SOURCE")
    if [[ "$GYM_ENABLED" == 1 ]]; then sources+=("$NEMO_GYM_SOURCE"); fi
    for source in "${sources[@]}"; do git ls-remote "$source" HEAD >/dev/null || exit 2; done
    mkdir -p "$RUN_ROOT" "/iopsstor/scratch/cscs/$USER/tmp"
    # Command-line flags work here; #SBATCH directives cannot expand recipe values.
    exec sbatch \
        --nodes="$TOTAL_NODES" --ntasks-per-node=1 --cpus-per-task=288 --gres=gpu:4 \
        --exclusive --account=csstaff --partition=normal --time="$JOB_TIME" \
        --job-name="$JOB_NAME" --output="/iopsstor/scratch/cscs/$USER/tmp/$JOB_NAME-%j.out" \
        --export=ALL "$@" "$LAUNCHER" "$recipe" --batch
fi

# ----- Source preparation -----

: "${SCRATCH:?SCRATCH must point to scratch storage}"
[[ "${SLURM_JOB_NUM_NODES:-$TOTAL_NODES}" == "$TOTAL_NODES" ]] || {
    echo "Requires exactly $TOTAL_NODES nodes." >&2; exit 2;
}
RUN_DIR="$RUN_ROOT/job-${SLURM_JOB_ID}"
MEGATRON_PATH="/tmp/chonk-mcore-$SLURM_JOB_ID"
if [[ "$mode" == batch ]]; then
    mkdir -p "$RUN_DIR"
    [[ ! -e "$RUN_DIR/sources" ]] || { echo "Refusing to overwrite snapshots: $RUN_DIR/sources" >&2; exit 1; }
    mkdir -p "$RUN_DIR/sources/megatron"
    git clone --no-checkout "$NEMO_RL_SOURCE" "$RUN_DIR/sources/nemo-rl"
    git -C "$RUN_DIR/sources/nemo-rl" checkout --detach "$NEMO_REF"
    for pr in "${NEMO_PRS[@]}"; do
        ref="${NEMO_PR_REFS[$pr]}"
        git -C "$RUN_DIR/sources/nemo-rl" fetch https://github.com/swiss-ai/Nemo-RL.git "$ref"
        printf 'NeMo PR #%s %s\n' "$pr" "$ref" >> "$RUN_DIR/applied-prs.txt"
        git -C "$RUN_DIR/sources/nemo-rl" -c user.name='Alps example' \
            -c user.email=alps@example.invalid -c commit.gpgsign=false merge --no-edit "$ref"
    done
    if [[ "$GENERATION_BACKEND" == vllm ]]; then
        git -C "$RUN_DIR/sources/nemo-rl" fetch https://github.com/NVIDIA-NeMo/RL.git "$NEMO_VLLM_DP_REF"
        printf 'NeMo upstream PR #2517 ' >> "$RUN_DIR/applied-prs.txt"
        git -C "$RUN_DIR/sources/nemo-rl" rev-parse FETCH_HEAD >> "$RUN_DIR/applied-prs.txt"
        # Keep the PR's EP-only multiprocessing executor and the fork's nightly-test budget.
        git -C "$RUN_DIR/sources/nemo-rl" -c user.name='Alps example' \
            -c user.email=alps@example.invalid -c commit.gpgsign=false merge --no-commit --no-ff -Xtheirs FETCH_HEAD
        git -C "$RUN_DIR/sources/nemo-rl" restore --source=HEAD -- tests/unit/test_recipes_and_test_suites.py
        git -C "$RUN_DIR/sources/nemo-rl" add tests/unit/test_recipes_and_test_suites.py
        git -C "$RUN_DIR/sources/nemo-rl" -c user.name='Alps example' \
            -c user.email=alps@example.invalid -c commit.gpgsign=false \
            commit -m 'Merge upstream vLLM DP PR 2517' \
            -m 'Use the PR multiprocessing executor and retain the fork nightly-test budget.'
    fi
    # PR #8 refreshes FP8 expert caches after optimizer and reference/policy updates.
    git -C "$RUN_DIR/sources/nemo-rl" fetch https://github.com/swiss-ai/Nemo-RL.git "$NEMO_FP8_REF"
    git -C "$RUN_DIR/sources/nemo-rl" -c user.name='Alps example' \
        -c user.email=alps@example.invalid -c commit.gpgsign=false merge --no-edit "$NEMO_FP8_REF"
    printf 'NeMo PR #8 %s\n' "$NEMO_FP8_REF" >> "$RUN_DIR/applied-prs.txt"
    for patch in "$LAUNCHER_DIR"/patches/nemo-rl/*.patch; do
        git -C "$RUN_DIR/sources/nemo-rl" apply "$patch"
        sha256sum "$patch" >> "$RUN_DIR/applied-patches.txt"
    done

    if [[ "$GYM_ENABLED" == 1 ]]; then
        NEMO_GYM_PATH="$RUN_DIR/sources/nemo-rl/3rdparty/Gym-workspace/Gym"
        git clone --no-checkout "$NEMO_GYM_SOURCE" "$NEMO_GYM_PATH"
        git -C "$NEMO_GYM_PATH" checkout --detach "$NEMO_GYM_REF"
        printf 'NeMo Gym %s\n' "$NEMO_GYM_REF" >> "$RUN_DIR/applied-prs.txt"
        export NEMO_GYM_PATH NEMO_GYM_VENV_DIR="$RUN_DIR/gym-venvs"
    fi
    git clone --no-checkout "$MEGATRON_SOURCE" "$RUN_DIR/megatron-repo"
    git -C "$RUN_DIR/megatron-repo" fetch https://github.com/swiss-ai/Megatron-LM-MoE.git "$MCORE_REF"
    printf 'Megatron PR #88 ' >> "$RUN_DIR/applied-prs.txt"
    git -C "$RUN_DIR/megatron-repo" rev-parse FETCH_HEAD >> "$RUN_DIR/applied-prs.txt"
    git -C "$RUN_DIR/megatron-repo" archive "$MCORE_REF" |
        tar --extract --touch --directory="$RUN_DIR/sources/megatron"
    for patch in "$LAUNCHER_DIR"/patches/megatron/*.patch; do
        git -C "$RUN_DIR/sources/megatron" apply "$patch"
        sha256sum "$patch" >> "$RUN_DIR/applied-patches.txt"
    done
    git clone --no-checkout "$BRIDGE_SOURCE" "$RUN_DIR/sources/bridge"
    git -C "$RUN_DIR/sources/bridge" checkout --detach "$BRIDGE_REF"
    git -C "$RUN_DIR/sources/bridge" fetch https://github.com/swiss-ai/Megatron-Bridge.git "$BRIDGE_PR5_REF"
    git -C "$RUN_DIR/sources/bridge" merge --ff-only "$BRIDGE_PR5_REF"
    {
        printf 'NeMo-RL %s\nMegatron-Core %s\nBridge base %s\n' "$NEMO_REF" "$MCORE_REF" "$BRIDGE_REF"
        cat "$RUN_DIR/applied-prs.txt" "$RUN_DIR/applied-patches.txt"
        printf 'NeMo merged tree (before local patches) '
        git -C "$RUN_DIR/sources/nemo-rl" rev-parse 'HEAD^{tree}'
        printf 'Bridge PR #5 '
        git -C "$RUN_DIR/sources/bridge" rev-parse HEAD
    } > "$RUN_DIR/source-provenance.txt"

    NRL_MEGATRON_CHECKPOINT_DIR="${NRL_MEGATRON_CHECKPOINT_DIR:-$RUN_DIR/hf-import}"
    export RUN_DIR NRL_MEGATRON_CHECKPOINT_DIR MEGATRON_PATH
    export NEMO_RL_PATH="$RUN_DIR/sources/nemo-rl" MEGATRON_BRIDGE_PATH="$RUN_DIR/sources/bridge/src"
    # W&B job-local paths. The shared artifact cache is set in environment.toml.
    export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$RUN_DIR/wandb/config}"
    export WANDB_DATA_DIR="${WANDB_DATA_DIR:-$RUN_DIR/wandb/data}"
    export WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-$RUN_DIR/wandb/artifacts}"
    if [[ "$GYM_ENABLED" == 1 ]]; then
        # GYM_ENV (profile) selects the remote environment served by the rl-gym
        # Argo CD project on the CSCS TDS cluster: gym-$GYM_ENV.yaml (Gym config)
        # and $GYM_ENV-chat-template.jinja. Math prepares the full dataset splits;
        # blackjack uses the bundled smoke rows (every reset deals a fresh hand).
        GYM_ENV="${GYM_ENV:-math}"
        case "$GYM_ENV" in
            math)
                gym_default_url="https://gym-math.shared-int-tds.tds.cscs.ch"
                gym_default_train="$RUN_DIR/data/openmathreasoning-train.jsonl"
                gym_default_val="$RUN_DIR/data/openmathreasoning-validation.jsonl" ;;
            blackjack)
                gym_default_url="https://gym-blackjack.shared-int-tds.tds.cscs.ch"
                gym_default_train="$EXAMPLE_DIR/blackjack-smoke.jsonl"
                gym_default_val="$EXAMPLE_DIR/blackjack-smoke.jsonl" ;;
            *) echo "Unknown GYM_ENV '$GYM_ENV' (expected math or blackjack)." >&2; exit 2 ;;
        esac
        export GYM_ENV
        export GYM_URL="${GYM_URL:-$gym_default_url}"
        export GYM_CONFIG="${GYM_CONFIG:-$EXAMPLE_DIR/gym-$GYM_ENV.yaml}"
        export GYM_TRAIN_DATA="${GYM_TRAIN_DATA:-$gym_default_train}"
        export GYM_VAL_DATA="${GYM_VAL_DATA:-$gym_default_val}"
        GYM_CHAT_TEMPLATE="$(< "${GYM_CHAT_TEMPLATE_FILE:-$EXAMPLE_DIR/$GYM_ENV-chat-template.jinja}")"
        export GYM_CHAT_TEMPLATE
    fi
    export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
    export RAY_USAGE_STATS_ENABLED=0 RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 PMIX_MCA_psec=native
    if [[ "$GENERATION_BACKEND" == vllm ]]; then export VLLM_WORKER_MULTIPROC_METHOD=spawn; fi
    export NRL_REFIT_BUFFER_MEMORY_RATIO=0.005 NRL_REFIT_NUM_BUFFERS=1
fi

# ----- Runtime environment and caches -----

if [[ "$mode" == server || "$mode" == driver ]]; then
    unset BASH_ENV SSH_CONNECTION SSH_CLIENT SSH_TTY
    cache_root="/tmp/chonk-caches-$SLURM_JOB_ID"
    export CUDA_CACHE_PATH="$cache_root/cuda" TRITON_CACHE_DIR="$cache_root/triton"
    export TRITON_HOME="$cache_root/triton-home" TORCHINDUCTOR_CACHE_DIR="$cache_root/inductor"
    export TORCH_EXTENSIONS_DIR="$cache_root/extensions" FLASHINFER_WORKSPACE_BASE="$cache_root/flashinfer"
    export DG_JIT_CACHE_DIR="$cache_root/deep-gemm"
    export MEGATRON_CONFIG_LOCK_DIR="${MEGATRON_CONFIG_LOCK_DIR:-$cache_root/megatron-config-locks}"
    mkdir -p "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
        "$TORCH_EXTENSIONS_DIR" "$FLASHINFER_WORKSPACE_BASE" "$MEGATRON_CONFIG_LOCK_DIR"
    if [[ ! -f "$MEGATRON_PATH/megatron/core/__init__.py" ]]; then
        mkdir -p "$MEGATRON_PATH"
        cp -R --no-preserve=mode,ownership "$RUN_DIR/sources/megatron/." "$MEGATRON_PATH/"
    fi
    export PYTHONPATH="$RUN_DIR/dependencies:$NEMO_RL_PATH:$MEGATRON_BRIDGE_PATH:$MEGATRON_PATH:$EXAMPLE_DIR"
    if [[ "$GYM_ENABLED" == 1 ]]; then
        export PYTHONPATH="$RUN_DIR/dependencies:$NEMO_RL_PATH:$MEGATRON_BRIDGE_PATH:$MEGATRON_PATH:$NEMO_GYM_PATH:$EXAMPLE_DIR"
    fi
fi

# ----- Ray servers -----

# Rank zero starts the driver in the head's container.
if [[ "$mode" == server ]]; then
    node_ip="$(hostname -I | awk '{print $1}')"
    [[ -n "$node_ip" ]] || { echo 'Cannot determine node IP.' >&2; exit 1; }
    args=(start --node-ip-address="$node_ip" --num-cpus="${SLURM_CPUS_PER_TASK:-288}" --num-gpus=4 --disable-usage-stats)
    if [[ "$SLURM_PROCID" == 0 ]]; then
        if [[ "$GYM_ENABLED" == 1 ]]; then
            uv run --no-project --python /usr/bin/python3.12 python -c \
                'from nemo_gym.cli import RunHelper; from nemo_gym.rollout_collection import RolloutCollectionHelper; import responses_api_agents.simple_agent.app; import responses_api_models.vllm_model.app; print("GYM_CLIENT_PREFLIGHT_PASS")'
        fi
        args+=(--head --port="${RAY_PORT:-6379}" --include-dashboard=false --temp-dir="/tmp/chonk-ray-$SLURM_JOB_ID")
        printf '%s:%s\n' "$node_ip" "${RAY_PORT:-6379}" > "$RUN_DIR/ray-address.txt.tmp"
        mv "$RUN_DIR/ray-address.txt.tmp" "$RUN_DIR/ray-address.txt"
    else
        deadline=$((SECONDS + 900))
        until [[ -s "$RUN_DIR/ray-address.txt" ]]; do
            (( SECONDS < deadline )) || { echo 'Timed out waiting for Ray head address.' >&2; exit 1; }
            sleep 2
        done
        read -r RAY_ADDRESS < "$RUN_DIR/ray-address.txt"
        args+=(--address="$RAY_ADDRESS")
    fi
    ray "${args[@]}"
    if [[ "$SLURM_PROCID" == 0 ]]; then
        if [[ "$GYM_ENABLED" == 1 ]]; then
            curl --fail --silent --show-error --max-time 15 "$GYM_URL/" > "$RUN_DIR/gym-health.json"
        fi
        set +e
        bash "$LAUNCHER" "$recipe" --driver
        driver_status=$?
        set -e
        printf '%s\n' "$driver_status" > "$RUN_DIR/driver-exit.txt.tmp"
        mv "$RUN_DIR/driver-exit.txt.tmp" "$RUN_DIR/driver-exit.txt"
    else
        until [[ -s "$RUN_DIR/driver-exit.txt" ]]; do sleep 2; done
        read -r driver_status < "$RUN_DIR/driver-exit.txt"
    fi
    exit "$driver_status"
fi

# ----- Training driver -----

# Run beside the head's Raylet after the full resource set is ready.
if [[ "$mode" == driver ]]; then
    deadline=$((SECONDS + 900))
    ready=0
    while (( SECONDS < deadline )); do
        if [[ -s "$RUN_DIR/ray-address.txt" ]]; then
            read -r RAY_ADDRESS < "$RUN_DIR/ray-address.txt"
            if status="$(timeout 15 ray status --address="$RAY_ADDRESS" 2>&1)"; then
                nodes="$(awk '/^Active:/ {active=1; next} /^Pending:/ {active=0} active && $1 ~ /^[0-9]+$/ && $2 ~ /^node_/ {n += $1} END {print n+0}' <<< "$status")"
                gpus="$(awk '$2 == "GPU" {split($1, a, "/"); print int(a[2]); exit}' <<< "$status")"
                if [[ "$nodes" == "$TOTAL_NODES" && "${gpus:-0}" == "$TOTAL_GPUS" ]]; then ready=1; break; fi
                echo "Waiting for Ray: $nodes/$TOTAL_NODES nodes, ${gpus:-0}/$TOTAL_GPUS GPUs"
            fi
        fi
        sleep 5
    done
    (( ready )) || { echo "Timed out waiting for all $TOTAL_NODES Ray nodes and $TOTAL_GPUS GPUs." >&2; exit 1; }
    export RAY_ADDRESS
    exec uv run --no-project --python /usr/bin/python3.12 python \
        "$NEMO_RL_PATH/$ENTRYPOINT" --config "$EXAMPLE_DIR/$CONFIG_FILE"
fi

# ----- Gym client and data -----

if [[ "$GYM_ENABLED" == 1 ]]; then
    srun --nodes=1 --ntasks=1 --relative=0 --cpus-per-task=1 --gres=none \
        --mpi=pmix --network=disable_rdzv_get --export=ALL --environment="$ENV_FILE" \
        -u bash -c '
        set -euo pipefail
        uv pip install --python /usr/bin/python3.12 --no-deps --target "$RUN_DIR/dependencies" \
            --requirement "$EXAMPLE_DIR/gym-client-requirements.txt"
        if [[ "$GYM_TRAIN_DATA" == "$RUN_DIR/data/openmathreasoning-train.jsonl" ]]; then
            bash "$EXAMPLE_DIR/prepare-data.sh" train "$GYM_TRAIN_DATA"
        fi
        if [[ "$GYM_VAL_DATA" == "$RUN_DIR/data/openmathreasoning-validation.jsonl" ]]; then
            bash "$EXAMPLE_DIR/prepare-data.sh" validation "$GYM_VAL_DATA"
        fi
        [[ -f "$GYM_TRAIN_DATA" ]] || { echo "Dataset not found: $GYM_TRAIN_DATA" >&2; exit 2; }
        [[ -f "$GYM_VAL_DATA" ]] || { echo "Dataset not found: $GYM_VAL_DATA" >&2; exit 2; }'
fi

# ----- Launch -----

srun --nodes="$TOTAL_NODES" --ntasks="$TOTAL_NODES" --ntasks-per-node=1 \
    --cpus-per-task=288 --gres=gpu:4 --mpi=pmix --network=disable_rdzv_get \
    --export=ALL --environment="$ENV_FILE" --container-mounts="$HF_MODEL:$HF_MODEL:ro" \
    --kill-on-bad-exit=1 -u bash "$LAUNCHER" "$recipe" --server
