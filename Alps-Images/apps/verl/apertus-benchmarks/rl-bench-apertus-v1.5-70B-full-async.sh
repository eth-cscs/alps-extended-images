#!/bin/bash

#SBATCH --nodes=16
#SBATCH --account=csstaff
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=12:00:00

export VERL_IMAGE="jfrog.svc.cscs.ch/docker-group-csstaff/alps-images/verl:alps7-dev-0f334b540ccc7034" #alps7-dev-0f334b540ccc7034 image with megatron

export MODEL_NAME="Apertus-v1.5-70B"
export MODEL_REPO="swiss-ai"

export PROJECT_NAME="apertus-benchmarks"
export EXPERIMENT_NAME="${MODEL_NAME}-verl-sglang-fsdp2-async-${SLURM_JOB_NUM_NODES}n"
export RUN_NAME="${EXPERIMENT_NAME}-run-${SLURM_JOB_ID}"
export TRAINING_HOME=/capstor/scratch/cscs/${USER}/RL/${MODEL_NAME}
export TRAINING_CONFIG=/tmp
export CHECKPOINT_HOME=${TRAINING_HOME}/checkpoints/${EXPERIMENT_NAME}-run-${SLURM_JOB_ID} #remove "run-${SLURM_JOB_ID}" to enable checkpoint resuming


mkdir -p $TRAINING_HOME
cd $TRAINING_HOME



export ROLLOUT_NNODES=$(python3 -c "import math; print(max(1, math.ceil($SLURM_JOB_NUM_NODES * 0.25)))")
export TRAINING_NNODES=$(( SLURM_JOB_NUM_NODES - ROLLOUT_NNODES ))

cat > "${TRAINING_CONFIG}/env.toml" <<- EOF
image = "${VERL_IMAGE}"
mounts = ["/capstor", "/iopsstor", "/users","/tmp"]
workdir = "/workspace/verl"
writable = true
entrypoint = true
[env]
PMIX_MCA_psec = "native"
HF_TOKEN = "$(cat ~/HF_TOKEN)"
[annotations]
com.hooks.cxi.enabled = "false"
EOF

cat > "${TRAINING_CONFIG}/grpo_gsm8k.yaml" <<- EOF
defaults:
  - ppo_trainer
  - override rollout@actor_rollout_ref.rollout: rollout
  - override actor@actor_rollout_ref.actor: dp_actor
  - override data@data: legacy_data
  - _self_

# ── Required by fully_async_main ──────────────────────────────────────────────
async_training:
  
  # On Policy Settings
  # staleness_threshold: 0
  # trigger_parameter_sync_step: 1
  
  # Stream Pipeline Settings
  staleness_threshold: 0.1 
  trigger_parameter_sync_step: 2

  require_batches: 1
  partial_rollout: False
  use_trainer_do_validate: False

# Top-level rollout block — fully_async_main copies .nnodes/.n_gpus_per_node
# into actor_rollout_ref.rollout, so keep these in sync with the rollout block below.
rollout:
  nnodes: ${ROLLOUT_NNODES}
  n_gpus_per_node: 4
  total_rollout_steps: 22419
  test_freq: 10
# ──────────────────────────────────────────────────────────────────────────────

data:
  train_files: ${TRAINING_HOME}/data/gsm8k/train.parquet
  val_files:   ${TRAINING_HOME}/data/gsm8k/test.parquet
  train_batch_size: 0    # must be 0 in fully-async mode
  gen_batch_size: 1      # must be 1 in fully-async mode
  return_raw_chat: True

actor_rollout_ref:
  hybrid_engine: False

  model:
    path: ${TRAINING_HOME}/models/${MODEL_NAME}
    # eager, not flash_attention_2/sdpa: Apertus-v1.5 instantiates a vision
    # tokenizer submodule (Apertus1p5VisionTokenizerModel) even for this
    # text-only GSM8K benchmark, and that submodule declares neither FA2 (run
    # 3130014) nor SDPA (run 3130169) support — only the language-model
    # backbone does. transformers' own error for the SDPA case points at
    # eager as the fallback; it is the one implementation every model
    # supports (no _supports_eager gate).
    override_config:
      attn_implementation: eager
    use_shm: false

  actor:
    strategy: fsdp2
    ppo_mini_batch_size: 48 #must be divisible by (rollout.n_gpus_per_node * rollout.nnodes)
    ppo_micro_batch_size_per_gpu: 1
    use_rollout_log_probs: True   # required for fully-async log prob correctness
    use_dynamic_bsz: True

  rollout:
    name: sglang
    mode: async
    load_format: dummy
    n_gpus_per_node: 4
    temperature: 1.0
    n: 16 #num responses per prompt 
    tensor_model_parallel_size: 4
    gpu_memory_utilization: 0.75
    log_prob_use_dynamic_bsz: True
    checkpoint_engine:
      backend: nccl # weight sync via NCCL broadcast

  ref:
    log_prob_use_dynamic_bsz: True

algorithm:
  adv_estimator: grpo
  kl_ctrl:
    type: adaptive
    kl_coef: 0.001
    target_kl: 0.05
    horizon: 10000
  rollout_correction:
    bypass_mode: True   # required for off-policy log prob correction

reward:
  custom_reward_function:
    path: ${TRAINING_CONFIG}/gsm8k_reward.py
    name: compute_reward

trainer:
  total_epochs: 3
  project_name: ${PROJECT_NAME}
  experiment_name: ${RUN_NAME}
  nnodes: ${TRAINING_NNODES}
  n_gpus_per_node: 4
  save_freq: 50
  default_local_dir: ${CHECKPOINT_HOME}
  logger: ["console", "wandb"]

ray_kwargs:
  ray_init:
    address: "auto"

critic:
  enable: false

distillation:
  enabled: false
EOF

cat > "${TRAINING_CONFIG}/gsm8k_reward.py" <<- EOF
# gsm8k_reward.py
import re
import math
from typing import Optional


def extract_model_answer(response: str) -> Optional[str]:
    """
    Pull the content of the last <answer>...</answer> block.
    Returns None if the model did not produce the expected format.
    """
    matches = re.findall(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if not matches:
        return None
    raw = matches[-1].strip().replace(",", "")
    try:
        val = float(raw)
        return str(val) if not math.isfinite(val) else (str(int(val)) if val == int(val) else str(val))
    except ValueError:
        return raw


def compute_reward(
    data_source, solution_str, ground_truth, extra_info=None, **kwargs
) -> float:
    # Truncated response (thinking opened but never closed): return 0, not a large
    # negative, to avoid extreme GRPO advantages that cause gradient spikes.
    if "<think>" in solution_str and "</think>" not in solution_str:
        return 0.0

    model_ans = extract_model_answer(solution_str)
    has_answer = "<answer>" in solution_str and "</answer>" in solution_str
    format_reward  = 0.1 if has_answer else 0.0
    outcome_reward = 1.0 if (model_ans is not None and model_ans == str(ground_truth)) else 0.0

    # Smooth length penalty starting at 350 words (~455 tokens), max -0.2 at 700 words.
    # Keeps thinking chains well below the 1024-token hard cap so truncation stays rare.
    words = len(solution_str.split())
    length_penalty = -0.2 * min(1.0, max(0.0, (words - 350) / 350))

    return outcome_reward + format_reward + length_penalty
EOF

cat > "${TRAINING_CONFIG}/prepare_gsm8k.py" <<- EOF
import re
import os
import datasets
import pandas as pd
from pathlib import Path

SYSTEM_PROMPT = """You are a precise math solver.
Solve the problem step by step, then give your final answer as a single number inside <answer>...</answer> tags.

Example:
<answer>42</answer>"""

def extract_ground_truth(solution: str) -> str:
    """Pull the number after #### from a GSM8K solution string."""
    match = re.search(r"####\s*([\d,\-\.]+)", solution)
    return match.group(1).replace(",", "").strip() if match else ""

def make_prompt(question: str) -> list:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question},
    ]

def prepare(split: str, output_path: str):
    training_home = os.environ.get("TRAINING_HOME", ".")
    raw_path = os.path.join(training_home, "data/gsm8k_raw")

    if os.path.exists(raw_path):
        print(f"Loading {split} from local cache: {raw_path}")
        ds = datasets.load_from_disk(raw_path)[split]
    else:
        print(f"Downloading {split} from HuggingFace...")
        ds = datasets.load_dataset("openai/gsm8k", "main", split=split)

    rows = []
    skipped = 0
    for item in ds:
        gt = extract_ground_truth(item["answer"])
        if not gt:
            skipped += 1
            continue
        rows.append({
            "prompt": make_prompt(item["question"]),
            "data_source": "gsm8k",
            "reward_model": {"ground_truth": gt},
        })

    df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"[{split}] Saved {len(df)} rows → {output_path} (skipped {skipped})")

if __name__ == "__main__":
    training_home = os.environ.get("TRAINING_HOME", ".")
    prepare("train", os.path.join(training_home, "data/gsm8k/train.parquet"))
    prepare("test",  os.path.join(training_home, "data/gsm8k/test.parquet"))
EOF

sbcast -f ${TRAINING_CONFIG}/gsm8k_reward.py ${TRAINING_CONFIG}/gsm8k_reward.py
sbcast -f ${TRAINING_CONFIG}/prepare_gsm8k.py ${TRAINING_CONFIG}/prepare_gsm8k.py

# The swiss-ai/transformers fork pinned below already registers some newer model
# types (e.g. "qwen3_asr") that stock transformers does not. SGLang's own
# compatibility shims (sglang/srt/configs/qwen3_asr.py) call
# AutoConfig.register("qwen3_asr", ...) unconditionally on import (exist_ok
# defaults to False), assuming the running transformers predates that type —
# against this fork that now collides: ValueError: 'qwen3_asr' is already used
# by a Transformers config, pick another name (run 3130211), raised inside the
# FullyAsyncRollouter actor the first time it imports SGLang. Same
# PYTHONPATH/sitecustomize.py mechanism as the GLM v1-separate-async script
# (see CLAUDE.md) — applies cluster-wide, including inside Ray actors, without
# touching verl or the fork's source.
cat > "${TRAINING_CONFIG}/sitecustomize-autoconfig-register.py" <<- 'EOF'
"""
Make transformers.models.auto.configuration_auto._LazyConfigMapping.register
tolerate re-registration of a model type that already exists.

Injected at Python startup via PYTHONPATH — does not modify transformers or
SGLang source. Must be staged as the process's actual "sitecustomize.py" on
sys.path to be auto-loaded.

Why: the swiss-ai/transformers fork used for Apertus-v1.5 already registers
some newer model types (e.g. "qwen3_asr") that stock/older transformers
releases do not. SGLang's compatibility shims for those same model types
(e.g. sglang/srt/configs/qwen3_asr.py) call
AutoConfig.register(model_type, config) with the default exist_ok=False,
assuming the running transformers predates the type. Against this fork the
type is already present, so the call raises ValueError instead of being a
harmless no-op, crashing every process that imports SGLang (run 3130211).
"""
import sys

_PATCHED: set = set()


class _AutoConfigRegisterPatcher:
    """sys.meta_path hook: patches the module immediately after it loads."""

    _TARGET = "transformers.models.auto.configuration_auto"

    def find_module(self, fullname, path=None):
        if fullname == self._TARGET and fullname not in _PATCHED:
            return self
        return None

    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]
        sys.meta_path[:] = [m for m in sys.meta_path if m is not self]
        try:
            import importlib
            mod = importlib.import_module(fullname)
        finally:
            sys.meta_path.append(self)
        _PATCHED.add(fullname)
        _orig_register = mod._LazyConfigMapping.register

        def _register(self, key, value, exist_ok=False):
            return _orig_register(self, key, value, exist_ok=True)

        mod._LazyConfigMapping.register = _register
        print(
            "[sitecustomize-autoconfig-register] patched "
            "_LazyConfigMapping.register to tolerate duplicate model types",
            flush=True,
        )
        return mod


sys.meta_path.append(_AutoConfigRegisterPatcher())
EOF
sbcast -f ${TRAINING_CONFIG}/sitecustomize-autoconfig-register.py ${TRAINING_CONFIG}/sitecustomize-autoconfig-register.py

# The image's pinned transformers==5.8.1 does not recognize model_type
# "apertus1p5" (Apertus-v1.5's architecture is not merged into any mainline
# transformers release yet). Build the swiss-ai fork commit the model card
# points at into a wheel once, on a single node, and stash it on Lustre —
# every node then does a plain wheel install (unzip + copy, no building, no
# per-node network). Originally this sbcast the ~large source tarball to every
# node directly, but run 3129881 hit "Bus error (core dumped)" from sbcast on
# that tarball, leaving several nodes with a truncated archive
# ("tar: Unexpected EOF") and a broken transformers install
# ("No module named 'transformers.utils'"); nodes where tar did succeed still
# built their own wheel independently inside the main srun — non-deterministic
# (different sha256 per build) and at least one hit
# "OSError: [Errno 116] Stale file handle" mid-install, the same Lustre
# file-locking hazard as the model-config loader (see Known hazards in
# CLAUDE.md). A single ~5MB wheel, built once, sidesteps all three.
#
# Also fetch a safetensors wheel meeting the fork's floor (>=0.8.0,
# dependency_versions_table.py) alongside it: the image's pinned safetensors==0.7.0
# fails transformers/dependency_versions_check.py at import time
# ("ImportError: safetensors>=0.8.0 is required ... but found safetensors==0.7.0",
# run 3129970) once --no-deps stops transformers from pulling it in itself.
# safetensors has no torch/torchvision ABI coupling, so upgrading it doesn't risk
# the ABI --no-deps is otherwise protecting (see the install step below).
export SWISS_AI_TRANSFORMERS_SHA=3797303dda74844e3d1f8977ff5518bb91f818b4
export SWISS_AI_WHEEL_DIR=${TRAINING_HOME}/wheels
mkdir -p ${SWISS_AI_WHEEL_DIR}
if ! ls ${SWISS_AI_WHEEL_DIR}/transformers-*.whl >/dev/null 2>&1 \
    || ! ls ${SWISS_AI_WHEEL_DIR}/safetensors-*.whl >/dev/null 2>&1; then
    echo "Building swiss-ai/transformers@${SWISS_AI_TRANSFORMERS_SHA} + safetensors wheels..."
    srun --mpi=pmix --network=disable_rdzv_get -N 1 --ntasks=1 -u \
        --environment="${TRAINING_CONFIG}/env.toml" \
        --container-writable bash -c '
        set -e
        curl -sfL "https://github.com/swiss-ai/transformers/archive/${SWISS_AI_TRANSFORMERS_SHA}.tar.gz" \
            -o /tmp/swiss-ai-transformers.tar.gz
        [ -s /tmp/swiss-ai-transformers.tar.gz ]
        mkdir -p /tmp/swiss-ai-transformers
        tar xzf /tmp/swiss-ai-transformers.tar.gz -C /tmp/swiss-ai-transformers --strip-components=1
        pip wheel --no-deps -w /tmp/swiss-ai-wheel /tmp/swiss-ai-transformers
        pip download --no-deps -d /tmp/swiss-ai-wheel "safetensors>=0.8.0"
        cp /tmp/swiss-ai-wheel/transformers-*.whl /tmp/swiss-ai-wheel/safetensors-*.whl ${SWISS_AI_WHEEL_DIR}/
    '
    ls ${SWISS_AI_WHEEL_DIR}/transformers-*.whl >/dev/null 2>&1 \
        && ls ${SWISS_AI_WHEEL_DIR}/safetensors-*.whl >/dev/null 2>&1 \
        || { echo "FATAL: swiss-ai/transformers or safetensors wheel build failed"; exit 1; }
else
    echo "swiss-ai/transformers and safetensors wheels already present, skipping build."
fi


# Download model (skip if already present)
if [ ! -d "${TRAINING_HOME}/models/${MODEL_NAME}" ]; then
    echo "Downloading ${MODEL_NAME}..."
    srun --mpi=pmix --network=disable_rdzv_get -N 1 --ntasks=1 -u \
        --environment="${TRAINING_CONFIG}/env.toml" \
        --container-writable bash -c '
        hf auth login
        hf download ${MODEL_REPO}/${MODEL_NAME} \
            --local-dir ${TRAINING_HOME}/models/${MODEL_NAME} \
    '
else
    echo "Model already present, skipping download."
fi

# Prepare dataset (skip if already present)
if [ ! -f "${TRAINING_HOME}/data/gsm8k/train.parquet" ]; then
    echo "Preparing GSM8K dataset..."
    srun --mpi=pmix --network=disable_rdzv_get -N 1 --ntasks=1 -u \
        --environment="${TRAINING_CONFIG}/env.toml" \
        --container-writable bash -c '
        # Try loading from cached raw download first, otherwise fetch from HF
        python ${TRAINING_CONFIG}/prepare_gsm8k.py
    '
else
    echo "Dataset already present, skipping preparation."
fi


export MASTER_NODE=$(hostname)
export MASTER_NODE_IP=$(hostname -i)
export PORT=6382
export RAY_ADDRESS="${MASTER_NODE_IP}:${PORT}"

export WANDB_API_KEY=$(cat /users/${USER}/.wandb_api_key)
export WANDB_SILENT=true # Suppress WandB logs

export RAY_memory_usage_threshold=0.99



srun --mpi=pmix --network=disable_rdzv_get -N ${SLURM_JOB_NUM_NODES} --ntasks-per-node=1 -u \
    --environment="${TRAINING_CONFIG}/env.toml" \
    --container-writable bash -c '


# Upgrade Verl to v0.9.0.
# The image clones with --branch ${VERL_REF} --depth 1, so no other ref is present
# locally and the tag has to be fetched explicitly before it can be checked out.
# verl is installed editable (pip install -e) from /workspace/verl, so the checkout
# takes effect without reinstalling; -f discards any dirty state in the clone.
export VERL_REF=v0.9.0
git -C /workspace/verl fetch --depth 1 origin +refs/tags/${VERL_REF}:refs/tags/${VERL_REF} \
    && git -C /workspace/verl checkout -f ${VERL_REF} \
    || { echo "FATAL: could not check out verl ${VERL_REF}"; exit 1; }
git -C /workspace/verl --no-pager log --oneline -1


# Apply Verl fixes
git remote add pr_origin https://github.com/theely/verl.git 2>/dev/null || true
git fetch pr_origin Fix-fsdp-model-loading-on-async
git reset --hard pr_origin/Fix-fsdp-model-loading-on-async


# Install the swiss-ai transformers + safetensors wheels built once above (adds
# the apertus1p5 architecture used by Apertus-v1.5; not yet in any mainline
# transformers release). Plain wheel installs straight off Lustre: no building,
# no per-node network, no isolated build env, so none of run 3129881 failure
# modes apply here. --no-deps: the image already carries a torch/torchvision
# pair whose ABI transformers pip installs are normally pinned against (see
# torch_constraints.txt in the Containerfile); pulling in the
# [torch,vision,audio] extras here could silently upgrade them. Those extras
# are for Apertus-v1.5 image/audio inputs, unused by this text-only GSM8K
# benchmark. safetensors is bumped alongside it to satisfy the forks
# safetensors>=0.8.0 floor (dependency_versions_check.py rejects the images
# pinned safetensors==0.7.0 at import time, run 3129970) — safetensors has no
# torch/torchvision ABI coupling, so this does not carry the same risk.
pip install --no-deps ${SWISS_AI_WHEEL_DIR}/transformers-*.whl ${SWISS_AI_WHEEL_DIR}/safetensors-*.whl
python3 -c "import transformers, safetensors; print(\"transformers:\", transformers.__version__, transformers.__file__); print(\"safetensors:\", safetensors.__version__)"


# Redirect all JIT/kernel caches to local tmpfs — Lustre does not support file locking
export FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer_${SLURM_JOB_ID}
mkdir -p $FLASHINFER_WORKSPACE_BASE

# Pre-warm FlashInfer JIT cache to avoid contention during training
python3 -c "
import os
import flashinfer
from flashinfer.prefill import get_batch_prefill_module
" 2>/dev/null || true

# Also disable CUDA graphs in SGLang to avoid the capture issue
export SGLANG_DISABLE_CUDA_GRAPH=1

# Disable SGlang TP memory imbalance, we need this because on some nodes FSDP takes more memory.
export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0

export VERL_LOGGING_LEVEL=INFO

# Stage the AutoConfig.register patch (sbcast to ${TRAINING_CONFIG} above) as
# this node sitecustomize.py — Python only auto-loads a module literally
# named "sitecustomize" found on sys.path. --ntasks-per-node=1 here, so there
# is exactly one task per node and no cross-task race to guard against.
export SITECUSTOMIZE_LOCAL=/tmp/verl_patches_${SLURM_JOB_ID}
mkdir -p $SITECUSTOMIZE_LOCAL
cp "${TRAINING_CONFIG}/sitecustomize-autoconfig-register.py" "$SITECUSTOMIZE_LOCAL/sitecustomize.py"
export PYTHONPATH="${SITECUSTOMIZE_LOCAL}:${PYTHONPATH:-}"


if [ $SLURM_PROCID -eq 0 ]; then
    # Start Ray head on rank 0
    ray start --head \
        --node-ip-address=$MASTER_NODE_IP \
        --port=$PORT \
        --num-cpus=${SLURM_CPUS_PER_TASK} \
        --num-gpus=4 \
        --disable-usage-stats || true

    while true; do
            alive_nodes=$(ray status | awk "/Active:/{flag=1;next}/Pending:/{flag=0}flag" | grep "node_" | wc -l)
            if ! [[ "$alive_nodes" =~ ^[0-9]+$ ]]; then
                alive_nodes=0
            fi
            if [ "$alive_nodes" -ge "$SLURM_JOB_NUM_NODES" ]; then
                break
            fi
            echo "Waiting for all nodes to join [$alive_nodes/$SLURM_JOB_NUM_NODES]"
            sleep 5
    done

    HYDRA_FULL_ERROR=1 python -m verl.experimental.fully_async_policy.fully_async_main \
        --config-path ${TRAINING_CONFIG} \
        --config-name grpo_gsm8k \
        --config-dir /workspace/verl/verl/trainer/config
else
    # Worker nodes join the Ray cluster
    sleep 15
    ray start \
        --address="${RAY_ADDRESS}" \
        --node-ip-address=$(hostname -i) \
        --num-cpus=${SLURM_CPUS_PER_TASK} \
        --num-gpus=4 \
        --block || true
fi


'
