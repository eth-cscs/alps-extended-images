#!/usr/bin/env bash
#SBATCH --nodes=24
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --account=csstaff
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/tmp/nemo-gym-chonk-%j.out
#SBATCH --job-name=nemo-gym-chonk
set -Eeuo pipefail

EXAMPLE_DIR="${EXAMPLE_DIR:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}}"
GYM_ENV="${GYM_ENV:-math}"
NEMO_RL_SOURCE="${NEMO_RL_SOURCE:-https://github.com/swiss-ai/Nemo-RL.git}"
NEMO_GYM_SOURCE="${NEMO_GYM_SOURCE:-https://github.com/swiss-ai/nemo-gym.git}"
NEMO_GYM_REF="${NEMO_GYM_REF:-93f232f02074deda2fec13a6848f22e5dca741f5}"
BRIDGE_SOURCE="${BRIDGE_SOURCE:-https://github.com/swiss-ai/Megatron-Bridge.git}"
MEGATRON_SOURCE="${MEGATRON_SOURCE:-https://github.com/swiss-ai/Megatron-LM-MoE.git}"
HF_MODEL="${HF_MODEL:-/capstor/store/cscs/swissai/infra01/apertus_checkpoints/v2/hf/chonk/pretraining/long_context/long-context-iter-0003576}"
ENV_FILE="${NEMO_RL_ENVIRONMENT:-$EXAMPLE_DIR/../environment.toml}"
RUN_ROOT="${CHONK_RUN_ROOT:-/iopsstor/scratch/cscs/$USER/tmp/nemo-rl-nemo-gym-chonk}"
NEMO_REF=f196ef0a1ccb54d6ce71cd63277fb6a0e01168db
NEMO_FP8_REF=19843e5a523254ebf3bed49a47a1e852dca29a70
MCORE_REF=0536034901962160aac9742d134a84c55e28d224
BRIDGE_REF=077224531ad4d6d0caef767a84e619b256aa5cb6

usage() {
    printf '%s\n' \
        'Submit from this directory: sbatch train-chonk.sh' \
        'Or: bash train-chonk.sh [sbatch options, e.g. --time 02:00:00]' \
        'NEMO_RL_SOURCE, NEMO_GYM_SOURCE, BRIDGE_SOURCE and MEGATRON_SOURCE accept local paths or Git URLs.' \
        'Overrides: NEMO_RL_ENVIRONMENT, HF_MODEL, MAX_NUM_STEPS, TRAIN_MICRO_BATCH,' \
        'INFERENCE_REQUESTS, NUM_PROMPTS, NRL_MEGATRON_CHECKPOINT_DIR, CHONK_RUN_ROOT,' \
        'GYM_ENV (math|blackjack), GYM_URL, GYM_CONFIG, GYM_TRAIN_DATA, GYM_CHAT_TEMPLATE_FILE.'
}
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
    usage
    exit 0
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    command -v sbatch >/dev/null || { echo 'sbatch is not available.' >&2; exit 127; }
    for source in "$NEMO_RL_SOURCE" "$NEMO_GYM_SOURCE" "$BRIDGE_SOURCE" "$MEGATRON_SOURCE"; do
        git ls-remote "$source" HEAD >/dev/null || exit 2
    done
    mkdir -p "$RUN_ROOT"
    exec sbatch \
        --output="$RUN_ROOT/slurm-%j.out" \
        --export=ALL,EXAMPLE_DIR="$EXAMPLE_DIR",NEMO_RL_SOURCE="$NEMO_RL_SOURCE",BRIDGE_SOURCE="$BRIDGE_SOURCE",MEGATRON_SOURCE="$MEGATRON_SOURCE",HF_MODEL="$HF_MODEL",NEMO_RL_ENVIRONMENT="$ENV_FILE",CHONK_RUN_ROOT="$RUN_ROOT",GYM_ENV="$GYM_ENV" \
        "$@" "$EXAMPLE_DIR/train-chonk.sh"
fi

: "${SCRATCH:?SCRATCH must point to Iopsstor scratch}"
[[ -f "$ENV_FILE" ]] || { echo "EDF not found: $ENV_FILE" >&2; exit 2; }
[[ "${SLURM_JOB_NUM_NODES:-24}" == 24 ]] || { echo 'Requires exactly 24 nodes.' >&2; exit 2; }
case "$RUN_ROOT/" in "$HOME/"*) echo 'RUN_ROOT must not be under HOME.' >&2; exit 2 ;; esac

RUN_DIR="$RUN_ROOT/job-${SLURM_JOB_ID}"
MEGATRON_PATH="/tmp/chonk-mcore-$SLURM_JOB_ID"
if [[ "${1:-}" != server && "${1:-}" != driver ]]; then
    mkdir -p "$RUN_DIR"
    [[ ! -e "$RUN_DIR/sources" ]] || {
        echo "Refusing to overwrite snapshots: $RUN_DIR/sources" >&2
        exit 1
    }

    mkdir -p "$RUN_DIR/sources/megatron"
    git clone --no-checkout "$NEMO_RL_SOURCE" "$RUN_DIR/sources/nemo-rl"
    git -C "$RUN_DIR/sources/nemo-rl" checkout --detach "$NEMO_REF"
    for pr in 5 1; do
        git -C "$RUN_DIR/sources/nemo-rl" fetch \
            https://github.com/swiss-ai/Nemo-RL.git "pull/$pr/head"
        printf 'NeMo PR #%s ' "$pr" >> "$RUN_DIR/applied-prs.txt"
        git -C "$RUN_DIR/sources/nemo-rl" rev-parse FETCH_HEAD >> "$RUN_DIR/applied-prs.txt"
        git -C "$RUN_DIR/sources/nemo-rl" -c user.name='Alps example' \
            -c user.email=alps@example.invalid -c commit.gpgsign=false \
            merge --no-edit FETCH_HEAD
    done
    git -C "$RUN_DIR/sources/nemo-rl" fetch \
        https://github.com/NVIDIA-NeMo/RL.git pull/2517/head
    printf 'NeMo upstream PR #2517 ' >> "$RUN_DIR/applied-prs.txt"
    git -C "$RUN_DIR/sources/nemo-rl" rev-parse FETCH_HEAD >> "$RUN_DIR/applied-prs.txt"
    # Keep the PR's EP-only multiprocessing executor in conflicting hunks.
    git -C "$RUN_DIR/sources/nemo-rl" -c user.name='Alps example' \
        -c user.email=alps@example.invalid -c commit.gpgsign=false \
        merge --no-commit --no-ff -Xtheirs FETCH_HEAD
    # The fork already has a larger nightly-test budget than the old PR.
    git -C "$RUN_DIR/sources/nemo-rl" restore --source=HEAD -- \
        tests/unit/test_recipes_and_test_suites.py
    git -C "$RUN_DIR/sources/nemo-rl" add tests/unit/test_recipes_and_test_suites.py
    git -C "$RUN_DIR/sources/nemo-rl" -c user.name='Alps example' \
        -c user.email=alps@example.invalid -c commit.gpgsign=false \
        commit -m 'Merge upstream vLLM DP PR 2517' \
        -m 'Use the PR multiprocessing executor and retain the fork nightly-test budget.'
    # PR #8 refreshes FP8 expert caches after optimizer and reference/policy updates.
    git -C "$RUN_DIR/sources/nemo-rl" fetch \
        https://github.com/swiss-ai/Nemo-RL.git pull/8/head
    git -C "$RUN_DIR/sources/nemo-rl" cat-file -e "$NEMO_FP8_REF^{commit}"
    git -C "$RUN_DIR/sources/nemo-rl" -c user.name='Alps example' \
        -c user.email=alps@example.invalid -c commit.gpgsign=false \
        merge --no-edit "$NEMO_FP8_REF"
    printf 'NeMo PR #8 %s\n' "$NEMO_FP8_REF" >> "$RUN_DIR/applied-prs.txt"

    NEMO_GYM_PATH="$RUN_DIR/sources/nemo-rl/3rdparty/Gym-workspace/Gym"
    git clone --no-checkout "$NEMO_GYM_SOURCE" "$NEMO_GYM_PATH"
    git -C "$NEMO_GYM_PATH" checkout --detach "$NEMO_GYM_REF"
    printf 'NeMo Gym %s\n' "$NEMO_GYM_REF" >> "$RUN_DIR/applied-prs.txt"
    export NEMO_GYM_PATH
    export NEMO_GYM_VENV_DIR="$RUN_DIR/gym-venvs"

    git clone --no-checkout "$MEGATRON_SOURCE" "$RUN_DIR/megatron-repo"
    git -C "$RUN_DIR/megatron-repo" fetch \
        https://github.com/swiss-ai/Megatron-LM-MoE.git pull/88/head
    printf 'Megatron PR #88 ' >> "$RUN_DIR/applied-prs.txt"
    git -C "$RUN_DIR/megatron-repo" rev-parse FETCH_HEAD >> "$RUN_DIR/applied-prs.txt"
    git -C "$RUN_DIR/megatron-repo" archive "$MCORE_REF" |
        tar --extract --touch --directory="$RUN_DIR/sources/megatron"
    git clone --no-checkout "$BRIDGE_SOURCE" "$RUN_DIR/sources/bridge"

    git -C "$RUN_DIR/sources/bridge" checkout --detach "$BRIDGE_REF"
    git -C "$RUN_DIR/sources/bridge" fetch https://github.com/swiss-ai/Megatron-Bridge.git pull/5/head
    git -C "$RUN_DIR/sources/bridge" merge --ff-only FETCH_HEAD

    {
        printf 'NeMo-RL %s\nMegatron-Core %s\nBridge base %s\n' "$NEMO_REF" "$MCORE_REF" "$BRIDGE_REF"
        cat "$RUN_DIR/applied-prs.txt"
        printf 'NeMo result tree '
        git -C "$RUN_DIR/sources/nemo-rl" rev-parse 'HEAD^{tree}'
        printf 'Bridge PR #5 '
        git -C "$RUN_DIR/sources/bridge" rev-parse HEAD
    } > "$RUN_DIR/source-provenance.txt"

    TRAIN_NODES=16
    INFERENCE_NODES=8
    TOTAL_NODES=24
    TRAIN_MICRO_BATCH="${TRAIN_MICRO_BATCH:-1}"
    INFERENCE_REQUESTS="${INFERENCE_REQUESTS:-32}"
    MAX_NUM_STEPS="${MAX_NUM_STEPS:-2}"
    NUM_PROMPTS="${NUM_PROMPTS:-32}"
    TRAIN_GLOBAL_BATCH=128
    (( TRAIN_MICRO_BATCH > 0 && INFERENCE_REQUESTS > 0 && NUM_PROMPTS > 0 && MAX_NUM_STEPS > 0 )) || {
        echo 'Batch settings and steps must be positive.' >&2
        exit 2
    }
    # TP1/PP4/CP1 on four GPUs per node gives TRAIN_NODES data-parallel ranks.
    (( TRAIN_GLOBAL_BATCH % (TRAIN_NODES * TRAIN_MICRO_BATCH) == 0 )) || {
        echo 'Global batch 128 must divide into complete training microbatches.' >&2
        exit 2
    }
    (( (NUM_PROMPTS * 8) % TRAIN_GLOBAL_BATCH == 0 )) || {
        echo 'Rollout batch must divide into training batches.' >&2
        exit 2
    }
    NRL_MEGATRON_CHECKPOINT_DIR="${NRL_MEGATRON_CHECKPOINT_DIR:-$RUN_DIR/hf-import}"

    export RUN_DIR EXAMPLE_DIR HF_MODEL
    # GYM_ENV selects the remote environment: gym-$GYM_ENV.yaml (Gym config),
    # $GYM_ENV-smoke.jsonl (dataset) and $GYM_ENV-chat-template.jinja.
    # The rl-gym Argo CD project serves both on the CSCS TDS cluster.
    case "$GYM_ENV" in
        math) gym_default_url="https://gym-math.shared-int-tds.tds.cscs.ch" ;;
        blackjack) gym_default_url="https://gym-blackjack.shared-int-tds.tds.cscs.ch" ;;
        *) echo "Unknown GYM_ENV '$GYM_ENV' (expected math or blackjack)." >&2; exit 2 ;;
    esac
    export GYM_URL="${GYM_URL:-$gym_default_url}"
    export GYM_CONFIG="${GYM_CONFIG:-$EXAMPLE_DIR/gym-$GYM_ENV.yaml}"
    export GYM_TRAIN_DATA="${GYM_TRAIN_DATA:-$EXAMPLE_DIR/$GYM_ENV-smoke.jsonl}"
    GYM_CHAT_TEMPLATE="$(< "${GYM_CHAT_TEMPLATE_FILE:-$EXAMPLE_DIR/$GYM_ENV-chat-template.jinja}")"
    export GYM_CHAT_TEMPLATE
    export NEMO_RL_PATH="$RUN_DIR/sources/nemo-rl"
    export MEGATRON_BRIDGE_PATH="$RUN_DIR/sources/bridge/src"
    export MEGATRON_PATH TRAIN_NODES INFERENCE_NODES TOTAL_NODES

    export TRAIN_MICRO_BATCH TRAIN_GLOBAL_BATCH INFERENCE_REQUESTS NUM_PROMPTS MAX_NUM_STEPS
    export NRL_MEGATRON_CHECKPOINT_DIR
    export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
    export RAY_USAGE_STATS_ENABLED=0 RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
    export PMIX_MCA_psec=native
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    export NRL_REFIT_BUFFER_MEMORY_RATIO=0.005 NRL_REFIT_NUM_BUFFERS=1
fi

if [[ "${1:-}" == server || "${1:-}" == driver ]]; then
    unset BASH_ENV SSH_CONNECTION SSH_CLIENT SSH_TTY

    cache_root="/tmp/chonk-caches-$SLURM_JOB_ID"
    export CUDA_CACHE_PATH="$cache_root/cuda"
    export TRITON_CACHE_DIR="$cache_root/triton"
    export TRITON_HOME="$cache_root/triton-home"
    export TORCHINDUCTOR_CACHE_DIR="$cache_root/inductor"
    export TORCH_EXTENSIONS_DIR="$cache_root/extensions"
    export FLASHINFER_WORKSPACE_BASE="$cache_root/flashinfer"
    export DG_JIT_CACHE_DIR="$cache_root/deep-gemm"
    export MEGATRON_CONFIG_LOCK_DIR="${MEGATRON_CONFIG_LOCK_DIR:-$cache_root/megatron-config-locks}"

    mkdir -p "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
        "$TORCH_EXTENSIONS_DIR" "$FLASHINFER_WORKSPACE_BASE" "$MEGATRON_CONFIG_LOCK_DIR"

    if [[ ! -f "$MEGATRON_PATH/megatron/core/__init__.py" ]]; then
        mkdir -p "$MEGATRON_PATH"
        cp -R --no-preserve=mode,ownership "$RUN_DIR/sources/megatron/." "$MEGATRON_PATH/"
    fi
    export PYTHONPATH="$RUN_DIR/dependencies:$NEMO_RL_PATH:$MEGATRON_BRIDGE_PATH:$MEGATRON_PATH:$NEMO_GYM_PATH:$EXAMPLE_DIR"
fi

# Internal server mode: srun starts one Ray server per node.
# Rank 0 is the head and starts the driver; the other ranks are Ray workers.
if [[ "${1:-}" == server ]]; then
    node_ip="$(hostname -I | awk '{print $1}')"
    [[ -n "$node_ip" ]] || { echo 'Cannot determine node IP.' >&2; exit 1; }
    args=(start
        --node-ip-address="$node_ip"
        --num-cpus="${SLURM_CPUS_PER_TASK:-288}"
        --num-gpus=4
        --disable-usage-stats)
    if [[ "$SLURM_PROCID" == 0 ]]; then
        uv run --no-project --python /usr/bin/python3.12 python -c \
            'from nemo_gym.cli import RunHelper; from nemo_gym.rollout_collection import RolloutCollectionHelper; import responses_api_agents.simple_agent.app; import responses_api_models.vllm_model.app; print("GYM_CLIENT_PREFLIGHT_PASS")'
        port="${RAY_PORT:-6379}"
        args+=(--head
            --port="$port"
            --include-dashboard=false
            --temp-dir="/tmp/chonk-ray-$SLURM_JOB_ID")
        printf '%s:%s\n' "$node_ip" "$port" > "$RUN_DIR/ray-address.txt.tmp"
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
        curl --fail --silent --show-error --max-time 15 "$GYM_URL/" \
            > "$RUN_DIR/gym-health.json"
        # The driver must share the head's container-local Raylet socket.
        set +e
        bash "$EXAMPLE_DIR/train-chonk.sh" driver
        driver_status=$?
        set -e
        printf '%s\n' "$driver_status" > "$RUN_DIR/driver-exit.txt.tmp"
        mv "$RUN_DIR/driver-exit.txt.tmp" "$RUN_DIR/driver-exit.txt"
    else
        until [[ -s "$RUN_DIR/driver-exit.txt" ]]; do
            sleep 2
        done
        read -r driver_status < "$RUN_DIR/driver-exit.txt"
    fi
    exit "$driver_status"
fi

# Internal driver mode: run GRPO in the head's container to share its Ray socket.
# Wait for all nodes and GPUs before starting GRPO.
if [[ "${1:-}" == driver ]]; then
    deadline=$((SECONDS + 900))
    ready=0
    while (( SECONDS < deadline )); do
        if [[ -s "$RUN_DIR/ray-address.txt" ]]; then
            read -r RAY_ADDRESS < "$RUN_DIR/ray-address.txt"
            if status="$(timeout 15 ray status --address="$RAY_ADDRESS" 2>&1)"; then
                nodes="$(awk '/^Active:/ {active=1; next} /^Pending:/ {active=0} active && $1 ~ /^[0-9]+$/ && $2 ~ /^node_/ {n += $1} END {print n+0}' <<< "$status")"
                gpus="$(awk '$2 == "GPU" {split($1, a, "/"); print int(a[2]); exit}' <<< "$status")"
                if [[ "$nodes" == 24 && "${gpus:-0}" == 96 ]]; then
                    ready=1
                    break
                fi
                echo "Waiting for Ray: $nodes/24 nodes, ${gpus:-0}/96 GPUs"
            fi
        fi
        sleep 5
    done
    (( ready )) || { echo 'Timed out waiting for all 24 Ray nodes and 96 GPUs.' >&2; exit 1; }
    export RAY_ADDRESS
    exec uv run --no-project --python /usr/bin/python3.12 python \
        "$RUN_DIR/sources/nemo-rl/examples/nemo_gym/run_grpo_nemo_gym.py" \
        --config "$EXAMPLE_DIR/grpo-chonk.yaml"
fi

# Job-local dependencies; preserve the image's Torch stack.
srun \
    --nodes=1 \
    --ntasks=1 \
    --relative=0 \
    --cpus-per-task=1 \
    --gres=none \
    --mpi=pmix \
    --network=disable_rdzv_get \
    --export=ALL \
    --environment="$ENV_FILE" \
    -u bash -c \
    'uv pip install --python /usr/bin/python3.12 --no-deps --target "$RUN_DIR/dependencies" "nemo-lens[sdk] @ git+https://github.com/NVIDIA-NeMo/Lens.git@b85578fc2b736a1804705e537001b5f45e9c715d" soundfile==0.13.1 --requirement "$EXAMPLE_DIR/gym-client-requirements.txt"'

# One container per node; rank zero runs the Ray head and driver.
srun \
    --nodes=24 \
    --ntasks=24 \
    --ntasks-per-node=1 \
    --cpus-per-task=288 \
    --gres=gpu:4 \
    --mpi=pmix \
    --network=disable_rdzv_get \
    --export=ALL \
    --environment="$ENV_FILE" \
    --container-mounts="$HF_MODEL:$HF_MODEL:ro" \
    --kill-on-bad-exit=1 \
    -u bash "$EXAMPLE_DIR/train-chonk.sh" server
