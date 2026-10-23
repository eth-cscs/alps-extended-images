#!/usr/bin/env bash
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --output=/iopsstor/scratch/cscs/%u/tmp/%x-%j.out

# Slurm batch script. Use directly for baked sources, or through submit.sh.
set -Eeuo pipefail
ulimit -c 0

die() { printf '%s\n' "$*" >&2; exit 2; }
if [[ $# == 0 || "${1:-}" == -h || "${1:-}" == --help ]]; then
    echo 'Usage: sbatch [Slurm options] train.sh RECIPE [--profile NAME] [training options]'
    echo 'For automatic resources or development sources, use bash submit.sh RECIPE ...'
    exit 0
fi
recipe="$1"
shift
mode=job
case "${1:-}" in
    --server|--driver|--prepare-dev) mode="${1#--}"; shift ;;
esac
: "${SLURM_JOB_ID:?Run with sbatch or submit.sh}"
CHONK_EXAMPLES_DIR="${CHONK_EXAMPLES_DIR:-${SLURM_SUBMIT_DIR:?Submit from the example directory or set CHONK_EXAMPLES_DIR}}"

# Load shared settings for direct sbatch runs.
if [[ "${CHONK_SETTINGS_READY:-0}" != 1 ]]; then
    source "$CHONK_EXAMPLES_DIR/submit.sh"
    load_settings "$recipe" "$@"
    [[ ${#slurm_args[@]} == 0 ]] || die 'Put Slurm options before train.sh.'
    [[ "$DEV_ENABLED" == 0 ]] || die 'Use submit.sh for development source overrides.'
    set --
fi
[[ $# == 0 ]] || die 'Training and Slurm options belong to submit.sh.'

LAUNCHER="$CHONK_EXAMPLES_DIR/train.sh"
export RUN_DIR="$CHONK_RUN_ROOT/job-${SLURM_JOB_ID}"
[[ "${SLURM_JOB_NUM_NODES:-$TOTAL_NODES}" == "$TOTAL_NODES" ]] || die "Requires exactly $TOTAL_NODES nodes."

# ----- Optional development preparation (inside the allocation) -----

prepare_development_sources() {
    mkdir "$RUN_DIR/dev"
    export GIT_AUTHOR_NAME='Alps development' GIT_COMMITTER_NAME='Alps development'
    export GIT_AUTHOR_EMAIL=alps@example.invalid GIT_COMMITTER_EMAIL=alps@example.invalid
    export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=commit.gpgsign GIT_CONFIG_VALUE_0=false
    : > "$RUN_DIR/development-provenance.txt"
    : > "$RUN_DIR/development-applied-patches.txt"
    for component in NEMO CORE BRIDGE GYM; do
        checkout_key="DEV_${component}_CHECKOUT"
        repo_key="DEV_${component}_REPO"
        ref_key="DEV_${component}_REF"
        merge_key="DEV_${component}_MERGES"
        patch_key="DEV_${component}_PATCHES"
        [[ -n "${!checkout_key:-}${!ref_key:-}" ]] || continue
        case "$component" in
            NEMO) repository=https://github.com/swiss-ai/Nemo-RL.git ;;
            CORE) repository=https://github.com/swiss-ai/Megatron-LM-MoE.git ;;
            BRIDGE) repository=https://github.com/swiss-ai/Megatron-Bridge.git ;;
            GYM) repository=https://github.com/swiss-ai/nemo-gym.git ;;
        esac
        repository="${!repo_key:-$repository}"
        target="$RUN_DIR/dev/${component,,}"
        if [[ -n "${!ref_key:-}" ]]; then
            base_kind=ref
            git clone --no-checkout "$repository" "$target"
            git -C "$target" fetch origin "${!ref_key}"
            git -C "$target" checkout --detach FETCH_HEAD
        else
            base_kind=checkout
            source_path="${!checkout_key}"
            git -c safe.directory="$source_path" clone --no-hardlinks --no-checkout "$source_path" "$target"
            git -C "$target" reset --mixed --quiet HEAD
            tar_excludes=(--exclude=.git --exclude='.env*' --exclude=.venv --exclude=__pycache__)
            git_excludes=()
            # Select submodules separately; do not copy them with the parent.
            while IFS= read -r -d '' entry; do
                [[ "$entry" == '160000 '* ]] || continue
                submodule="${entry#*$'\t'}"
                tar_excludes+=("--exclude=./$submodule")
                git_excludes+=(":(exclude,literal)$submodule")
                if [[ -e "$source_path/$submodule/.git" ]]; then
                    echo "Skipping $component submodule $submodule; override that repository separately if needed." >&2
                fi
            done < <(git -C "$source_path" ls-files --stage -z)
            tar -C "$source_path" "${tar_excludes[@]}" --dereference -cf - . | tar -C "$target" -xf -
            git -C "$target" add -A -- . "${git_excludes[@]}"
            if ! git -C "$target" diff --cached --quiet; then
                git -C "$target" commit -m 'Snapshot development working tree'
            fi
        fi
        printf '%s base %s %s\n' "$component" "$base_kind" "$(git -C "$target" rev-parse HEAD)" >> "$RUN_DIR/development-provenance.txt"
        while IFS= read -r ref; do
            [[ -n "$ref" ]] || continue
            merge_repository="$repository"
            if [[ "$ref" == *'#'* ]]; then
                merge_repository="${ref%%#*}"
                ref="${ref#*#}"
            fi
            git -C "$target" fetch "$merge_repository" "$ref"
            printf '%s merge %s\n' "$component" "$(git -C "$target" rev-parse FETCH_HEAD)" >> "$RUN_DIR/development-provenance.txt"
            git -C "$target" merge --no-edit FETCH_HEAD
        done <<< "${!merge_key:-}"
        while IFS= read -r patch; do
            [[ -n "$patch" ]] || continue
            git -C "$target" apply --index "$patch"
            printf '%s patch ' "$component" >> "$RUN_DIR/development-provenance.txt"
            sha256sum "$patch" | tee -a "$RUN_DIR/development-applied-patches.txt" >> "$RUN_DIR/development-provenance.txt"
        done <<< "${!patch_key:-}"
        printf '%s tree %s\n' "$component" "$(git -C "$target" write-tree)" >> "$RUN_DIR/development-provenance.txt"
    done
}
if [[ "$mode" == prepare-dev ]]; then prepare_development_sources; exit 0; fi

# ----- Container environment (EDF) -----
ENV_FILE="$RUN_DIR/environment.toml"
write_environment() {
    local mounts="\"/capstor/\", \"/iopsstor/\", \"/ritom/\", \"${HOME}:${HOME}\"" component target
    for component in nemo core bridge gym; do
        [[ "$DEV_ENABLED" == 1 && -d "$RUN_DIR/dev/$component" ]] || continue
        case "$component" in
            nemo) target=/opt/nemo-rl ;; core) target=/opt/megatron-lm ;;
            bridge) target=/opt/megatron-bridge ;; gym) target=/opt/nemo-gym ;;
        esac
        mounts+=", \"$RUN_DIR/dev/$component:$target:ro\""
    done
    cat > "$ENV_FILE" <<EOF
image = "$NEMO_RL_IMAGE"
mounts = [$mounts]
workdir = "/tmp"
writable = true

[env]
PATH = "/usr/local/lib/python3.12/dist-packages/torch_tensorrt/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/mpi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/ucx/bin:/opt/amazon/efa/bin:/opt/tensorrt/bin"
PMIX_MCA_psec = "native"
NCCL_PXN_DISABLE = "1"
NCCL_CROSS_NIC = "0"
UCCL_EP_TRANSPORT = "cxi"
XDG_CACHE_HOME = "/ritom/scratch/cscs/${USER}/cache"
HF_HOME = "/ritom/scratch/cscs/${USER}/cache/huggingface"
HF_HUB_CACHE = "/ritom/scratch/cscs/${USER}/cache/huggingface/hub"
TORCH_HOME = "/ritom/scratch/cscs/${USER}/cache/torch"
VLLM_CACHE_ROOT = "/ritom/scratch/cscs/${USER}/cache/vllm"
WANDB_CACHE_DIR = "/ritom/scratch/cscs/${USER}/cache/wandb"
FLASHINFER_WORKSPACE_BASE = "/ritom/scratch/cscs/${USER}/cache/flashinfer"
PIP_CACHE_DIR = "/ritom/scratch/cscs/${USER}/cache/pip"
UV_CACHE_DIR = "/ritom/scratch/cscs/${USER}/cache/uv"

[annotations]
com.hooks.cxi.enabled = "false"
com.hooks.netstack.source = "host"
EOF
}

# ----- Job setup -----

if [[ "$mode" == job ]]; then
    mkdir -p "$RUN_DIR"
    [[ "$DEV_ENABLED" == 0 || ! -e "$RUN_DIR/dev" ]] ||
        die "Refusing to overwrite development snapshots: $RUN_DIR/dev"
    write_environment
    if [[ "$DEV_ENABLED" == 1 ]]; then
        srun --nodes=1 --ntasks=1 --relative=0 --cpus-per-task=1 --gres=none \
            --mpi=pmix --network=disable_rdzv_get --export=ALL --environment="$ENV_FILE" \
            -u bash "$LAUNCHER" "$recipe" --prepare-dev
        write_environment
    fi
    NRL_MEGATRON_CHECKPOINT_DIR="${NRL_MEGATRON_CHECKPOINT_DIR:-$RUN_DIR/hf-import}"
    export RUN_DIR NRL_MEGATRON_CHECKPOINT_DIR
    export NEMO_GYM_VENV_DIR="$RUN_DIR/gym-venvs"
    export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$RUN_DIR/wandb/config}"
    export WANDB_DATA_DIR="${WANDB_DATA_DIR:-$RUN_DIR/wandb/data}"
    export WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-$RUN_DIR/wandb/artifacts}"

    if [[ "$GYM_ENABLED" == 1 ]]; then
        export GYM_MATH_URL="${GYM_MATH_URL:-https://gym-dev.swissai.svc.cscs.ch/math_with_judge}"
        export GYM_TRAIN_DATA="${GYM_TRAIN_DATA:-$RUN_DIR/data/openmathreasoning-train.jsonl}"
        export GYM_VAL_DATA="${GYM_VAL_DATA:-$RUN_DIR/data/openmathreasoning-validation.jsonl}"
        GYM_CHAT_TEMPLATE="$(< "$EXAMPLE_DIR/math-chat-template.jinja")"
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
    : "${NEMO_RL_PATH:?The image must include the installed training stack}"
    if [[ "$mode" == server && "$SLURM_PROCID" == 0 ]]; then
        if [[ "$DEV_ENABLED" == 1 ]]; then
            cp /opt/alps/source-provenance.txt "$RUN_DIR/image-source-provenance.txt"
            cp /opt/alps/applied-patches.txt "$RUN_DIR/image-applied-patches.txt"
            printf 'Image defaults: image-source-provenance.txt; selected replacements follow.\n' > "$RUN_DIR/source-provenance.txt"
            cat "$RUN_DIR/development-provenance.txt" >> "$RUN_DIR/source-provenance.txt"
            cp "$RUN_DIR/development-applied-patches.txt" "$RUN_DIR/applied-patches.txt"
        else
            cp /opt/alps/source-provenance.txt "$RUN_DIR/source-provenance.txt"
            cp /opt/alps/applied-patches.txt "$RUN_DIR/applied-patches.txt"
        fi
    fi
    export PYTHONPATH="$EXAMPLE_DIR"
fi

# ----- Ray servers -----

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
            curl --fail --silent --show-error --max-time 15 "$GYM_MATH_URL/" > "$RUN_DIR/gym-health.json"
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

# ----- Gym client and data downloading -----

if [[ "$GYM_ENABLED" == 1 ]]; then
    srun --nodes=1 --ntasks=1 --relative=0 --cpus-per-task=1 --gres=none \
        --mpi=pmix --network=disable_rdzv_get --export=ALL --environment="$ENV_FILE" \
        -u bash -c '
        set -euo pipefail
        if [[ "$GYM_TRAIN_DATA" == "$RUN_DIR/data/openmathreasoning-train.jsonl" ]]; then
            bash "$EXAMPLE_DIR/prepare-data.sh" train "$GYM_TRAIN_DATA"
        fi
        if [[ "$GYM_VAL_DATA" == "$RUN_DIR/data/openmathreasoning-validation.jsonl" ]]; then
            bash "$EXAMPLE_DIR/prepare-data.sh" validation "$GYM_VAL_DATA"
        fi
        for path in "$GYM_TRAIN_DATA" "$GYM_VAL_DATA"; do
            [[ -f "$path" ]] || { echo "Dataset not found: $path" >&2; exit 2; }
        done'
fi

# ----- Launch containers on the compute nodes -----

srun --mpi=pmix --network=disable_rdzv_get \
    --export=ALL --environment="$ENV_FILE" --container-mounts="$HF_MODEL:$HF_MODEL:ro" \
    --kill-on-bad-exit=1 -u bash "$LAUNCHER" "$recipe" --server
