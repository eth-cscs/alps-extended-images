#!/usr/bin/env bash
# Read settings and submit train.sh. The settings function also serves direct sbatch runs.
set -Eeuo pipefail

die() { printf '%s\n' "$*" >&2; exit 2; }
usage() {
    printf '%s\n' \
        'Usage: bash submit.sh RECIPE [--profile NAME] [training options] [sbatch options]' \
        'Example: bash submit.sh vllm/gsm8k --profile muon --account=YOUR_PROJECT --partition=YOUR_PARTITION' \
        'RECIPE is relative to this script. --profile NAME selects NAME.env; default: recipe.env.' \
        'Training options: --grpo-steps, --prompts, --generations-per-prompt, --updates-per-rollout,' \
        '  --micro-batch, --train-nodes, --generation-nodes, --inference-requests,' \
        '  --lr, --min-lr, --matrix-lr, --gains-lr, --train-mb-tokens, --logprob-mb-tokens.' \
        'Development: --dev-COMPONENT-checkout PATH, or --dev-COMPONENT-ref REF' \
        '  with optional --dev-COMPONENT-repo URL; COMPONENT is nemo, core, bridge or gym.' \
        '  Repeat --dev-COMPONENT-pr NUMBER_OR_GITHUB_URL, --dev-COMPONENT-merge REF' \
        '  and --dev-COMPONENT-patch FILE (merges first, then patches).' \
        '  PRs, merges and patches require a checkout or ref for that component.' \
        'Other arguments pass unchanged to sbatch. No -- separator is required.' \
        'CLI training options override .env/environment and profile defaults.' \
        'NEMO_RL_IMAGE overrides the default container image.' \
        'Returns after submission. The job runs independently of your login shell.'
}

load_settings() {
    recipe="$1"
    shift
    LAUNCHER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    profile=recipe
    declare -A overrides=()
    slurm_args=()
    while (( $# )); do
        option="${1%%=*}"
        field=''
        case "$option" in
            -h|--help) usage; exit 0 ;;
            --) shift; slurm_args+=("$@"); break ;;
            --profile) key=profile ;;
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
            --dev-*)
                [[ "$option" =~ ^--dev-(nemo|core|bridge|gym)-(checkout|repo|ref|merge|pr|patch)$ ]] ||
                    die "Unknown development option: $option"
                component="${BASH_REMATCH[1]^^}"
                field="${BASH_REMATCH[2]^^}"
                case "$field" in MERGE|PR) key="DEV_${component}_MERGES" ;; PATCH) key="DEV_${component}_PATCHES" ;; *) key="DEV_${component}_${field}" ;; esac ;;
            *) slurm_args+=("$1"); shift; continue ;;
        esac
        if [[ "$1" == *=* ]]; then
            value="${1#*=}"
            shift
        else
            (( $# >= 2 )) || die "Missing value for $option"
            value="$2"
            shift 2
        fi
        [[ -n "$value" && "$value" != *$'\n'* ]] || die "Invalid value for $option"
        if [[ "$key" == profile ]]; then profile="$value"; continue; fi
        if [[ "$field" == PR ]]; then
            if [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
                value="refs/pull/$value/head"
            elif [[ "$value" =~ ^https://github.com/([^/]+/[^/]+)/pull/([1-9][0-9]*)/?$ ]]; then
                value="https://github.com/${BASH_REMATCH[1]}.git#refs/pull/${BASH_REMATCH[2]}/head"
            else
                die "Invalid PR number or GitHub PR URL: $value"
            fi
        fi
        if [[ "$field" == MERGE || "$field" == PR || "$field" == PATCH ]]; then
            overrides[$key]="${overrides[$key]:+${overrides[$key]}$'\n'}$value"
        else
            overrides[$key]="$value"
        fi
    done

    # ----- User settings and recipe defaults -----
    [[ "$profile" =~ ^[a-zA-Z0-9_-]+$ ]] || die 'Invalid profile name.'
    recipe_profile="$LAUNCHER_DIR/$recipe/$profile.env"
    [[ -f "$recipe_profile" ]] || die "Profile not found: $recipe/$profile.env"
    if [[ -f "$LAUNCHER_DIR/.env" ]]; then
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
    EXAMPLE_DIR="$(cd -- "${EXAMPLE_DIR:-$(dirname -- "$recipe_profile")}" && pwd -P)"
    NEMO_RL_IMAGE="${NEMO_RL_IMAGE:-jfrog.svc.cscs.ch/docker-group-csstaff/alps-images/nemo-rl-cuda:alps7-dev-62e3c495bd4200880fc6c652bf9fc68c}"
    HF_MODEL="${HF_MODEL:-/capstor/store/cscs/swissai/infra01/apertus_checkpoints/v2/hf/chonk/pretraining/long_context/long-context-iter-0003576}"
    RUN_ROOT="$(readlink -m -- "${CHONK_RUN_ROOT:-/iopsstor/scratch/cscs/$USER/tmp/$RUN_NAME}")"

    # ----- Development source selection -----
    DEV_ENABLED=0
    for component in NEMO CORE BRIDGE GYM; do
        checkout_key="DEV_${component}_CHECKOUT"
        repo_key="DEV_${component}_REPO"
        ref_key="DEV_${component}_REF"
        patch_key="DEV_${component}_PATCHES"
        merge_key="DEV_${component}_MERGES"
        [[ -n "${!checkout_key:-}${!ref_key:-}" || -z "${!merge_key:-}${!patch_key:-}" ]] ||
            die "$component PRs, merges and patches require --dev-${component,,}-checkout or --dev-${component,,}-ref."
        [[ -z "${!checkout_key:-}" || ( -z "${!repo_key:-}" && -z "${!ref_key:-}" ) ]] ||
            die "Choose a checkout OR a repository/ref for $component."
        [[ -z "${!repo_key:-}" || -n "${!ref_key:-}" ]] || die "$repo_key requires $ref_key."
        for field in CHECKOUT REPO REF MERGES PATCHES; do
            key="DEV_${component}_${field}"
            [[ -z "${!key:-}" ]] || DEV_ENABLED=1
            export "$key"
        done
        if [[ -n "${!checkout_key:-}" ]]; then
            [[ -d "${!checkout_key}" ]] || die "Checkout not found: ${!checkout_key}"
            printf -v "$checkout_key" '%s' "$(cd -- "${!checkout_key}" && pwd -P)"
            [[ "$(git -C "${!checkout_key}" rev-parse --show-toplevel)" == "${!checkout_key}" ]] ||
                die "$checkout_key must be a Git checkout root."
        fi
        if [[ -n "${!patch_key:-}" ]]; then
            patches=''
            while IFS= read -r patch; do
                [[ -n "$patch" ]] || continue
                [[ -f "$patch" ]] || die "Patch not found: $patch"
                patch="$(readlink -f -- "$patch")"
                patches+="${patches:+$'\n'}$patch"
            done <<< "${!patch_key}"
            printf -v "$patch_key" '%s' "$patches"
        fi
    done

    # ----- Batch sizing -----
    TRAIN_MICRO_BATCH="${TRAIN_MICRO_BATCH:-$DEFAULT_MICRO_BATCH}"
    INFERENCE_REQUESTS="${INFERENCE_REQUESTS:-32}"
    MAX_NUM_STEPS="${MAX_NUM_STEPS:-$DEFAULT_MAX_NUM_STEPS}"
    NUM_PROMPTS="${NUM_PROMPTS:-$DEFAULT_NUM_PROMPTS}"
    for key in TRAIN_NODES INFERENCE_NODES TRAIN_MICRO_BATCH INFERENCE_REQUESTS NUM_PROMPTS \
        MAX_NUM_STEPS GENERATIONS_PER_PROMPT UPDATES_PER_ROLLOUT TRAIN_EXPERT_PARALLEL_SIZE \
        GENERATION_EXPERT_PARALLEL_SIZE TRAIN_MB_TOKENS LOGPROB_MB_TOKENS; do
        [[ ! -v "$key" ]] || [[ "${!key}" =~ ^[1-9][0-9]*$ ]] || die "$key must be a positive integer."
    done
    for key in BASE_LR MIN_LR MATRIX_LR GAINS_LR; do
        [[ ! -v "$key" ]] || [[ "${!key}" =~ ^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$ ]] ||
            die "$key must be a non-negative number."
    done
    if [[ -n "${BASE_LR:-}" && -n "${MIN_LR:-}" ]]; then
        awk -v base="$BASE_LR" -v minimum="$MIN_LR" 'BEGIN { exit(minimum > base) }' ||
            die 'Minimum learning rate must not exceed the base learning rate.'
    fi
    TOTAL_NODES=$((TRAIN_NODES + INFERENCE_NODES))
    TOTAL_GPUS=$((TOTAL_NODES * 4))
    if [[ -n "${UPDATES_PER_ROLLOUT:-}" ]]; then
        (( (NUM_PROMPTS * GENERATIONS_PER_PROMPT) % UPDATES_PER_ROLLOUT == 0 )) ||
            die 'Rollout batch must divide into the requested optimizer updates.'
        TRAIN_GLOBAL_BATCH=$((NUM_PROMPTS * GENERATIONS_PER_PROMPT / UPDATES_PER_ROLLOUT))
    fi
    [[ "${TRAIN_GLOBAL_BATCH:-}" =~ ^[1-9][0-9]*$ ]] || die 'Derived global batch must be positive.'
    (( (NUM_PROMPTS * GENERATIONS_PER_PROMPT) % TRAIN_GLOBAL_BATCH == 0 )) ||
        die 'Rollout batch must divide into training batches.'
    UPDATES_PER_ROLLOUT=$((NUM_PROMPTS * GENERATIONS_PER_PROMPT / TRAIN_GLOBAL_BATCH))
    [[ -f "$EXAMPLE_DIR/$CONFIG_FILE" ]] || die "Config not found: $EXAMPLE_DIR/$CONFIG_FILE"
    case "$RUN_ROOT/" in "$(readlink -m -- "$HOME")/"*) die 'RUN_ROOT must not be under HOME.' ;; esac
    case "$RUN_ROOT" in *\"*|*\\*|*:*|*$'\n'*) die 'RUN_ROOT cannot contain quotes, backslashes, colons or newlines.' ;; esac

    export CHONK_EXAMPLES_DIR="$LAUNCHER_DIR" EXAMPLE_DIR HF_MODEL
    export NEMO_RL_IMAGE CHONK_RUN_ROOT="$RUN_ROOT" DEV_ENABLED TOTAL_GPUS
    export TRAIN_NODES INFERENCE_NODES TOTAL_NODES TRAIN_GLOBAL_BATCH GENERATIONS_PER_PROMPT
    export TRAIN_MICRO_BATCH INFERENCE_REQUESTS MAX_NUM_STEPS NUM_PROMPTS GYM_ENABLED
    export UPDATES_PER_ROLLOUT TRAIN_EXPERT_PARALLEL_SIZE TRAIN_MB_TOKENS LOGPROB_MB_TOKENS
    export BASE_LR MIN_LR MATRIX_LR GAINS_LR
    export CHONK_SETTINGS_READY=1
}

# Sourcing this file only loads the settings functions only.
[[ "${BASH_SOURCE[0]}" == "$0" ]] || return 0
if [[ $# == 0 || "${1:-}" == -h || "${1:-}" == --help ]]; then usage; exit 0; fi
[[ -z "${SLURM_JOB_ID:-}" ]] || die 'Run submit.sh outside an allocation.'
load_settings "$@"

# Submit job
printf '%s GRPO steps × %s updates = %s optimizer updates; global batch %s; %s training + %s generation nodes\n' \
    "$MAX_NUM_STEPS" "$UPDATES_PER_ROLLOUT" "$((MAX_NUM_STEPS * UPDATES_PER_ROLLOUT))" \
    "$TRAIN_GLOBAL_BATCH" "$TRAIN_NODES" "$INFERENCE_NODES" >&2
command -v sbatch >/dev/null || { echo 'sbatch is not available.' >&2; exit 127; }
mkdir -p "$RUN_ROOT" "/iopsstor/scratch/cscs/$USER/tmp"
exec sbatch \
    --nodes="$TOTAL_NODES" --time="$JOB_TIME" \
    --job-name="$JOB_NAME" \
    --export=ALL "${slurm_args[@]}" "$LAUNCHER_DIR/train.sh" "$recipe"
