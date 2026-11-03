#!/bin/bash

#SBATCH --nodes=8
#SBATCH --account=csstaff
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=12:00:00

export VERL_IMAGE="jfrog.svc.cscs.ch/docker-group-csstaff/alps-images/verl:alps7-dev"


export MODEL_NAME="Qwen2.5-3B-Instruct"
export MODEL_REPO="Qwen"

export PROJECT_NAME="async-grpo-gsm8k"
export EXPERIMENT_NAME="${MODEL_NAME}-verl-sglang-megatron-async-${SLURM_JOB_NUM_NODES}n"
export RUN_NAME="${EXPERIMENT_NAME}-${SLURM_JOB_ID}"
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
[annotations]
com.hooks.cxi.enabled = "false"
EOF

cat > "${TRAINING_CONFIG}/grpo_gsm8k.yaml" <<- EOF
defaults:
  - ppo_megatron_trainer
  - override model_engine: megatron
  - override rollout@actor_rollout_ref.rollout: rollout
  - override data@data: legacy_data
  - _self_

# ── Required by fully_async_main ──────────────────────────────────────────────
async_training:
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
    use_remove_padding: True  # Megatron THD layout requires sequence packing
    use_shm: false
    trust_remote_code: True  # GLM-5.2 uses a custom TokenizersBackend tokenizer

  actor:
    # 8 training nodes × 4 GPUs = 32 GPUs; TP=4, PP=2, EP=4 → 4×2×4=32
    # 2 training nodes × 4 GPUs = 8 GPUs;  TP=2, PP=2, EP=4 → 2×2×2=8
    optim:
      lr_decay_steps: 22419  # must be positive at init; set_total_train_steps is called too late (after init_workers)
    ppo_mini_batch_size: 48
    ppo_micro_batch_size_per_gpu: 1
    ppo_max_token_len_per_gpu: 16384
    use_rollout_log_probs: True   # required for fully-async log prob correctness
    use_dynamic_bsz: True
    megatron:
      tensor_model_parallel_size: 2
      pipeline_model_parallel_size: 2
      expert_model_parallel_size: 1  # Apertus-8B is dense (no MoE); EP must be 1
      param_offload: True
      grad_offload: True
      optimizer_offload: True
      vanilla_mbridge: False  # use Megatron-Bridge (NVIDIA) instead of mbridge
      override_transformer_config:
        recompute_granularity: full
        recompute_method: uniform
        recompute_num_layers: 1
        use_cpu_initialization: True  # keep params on CPU during DDP init to avoid 44 GiB flat-buffer OOM

  rollout:
    name: sglang
    mode: async
    load_format: dummy
    n_gpus_per_node: 4
    temperature: 1.0
    n: 16 #num responses per prompt
    tensor_model_parallel_size: 4  # must be <= n_gpus_per_node; cross-node TP kills decode throughput
    gpu_memory_utilization: 0.85
    log_prob_use_dynamic_bsz: True
    checkpoint_engine:
      backend: nccl # weight sync via NCCL broadcast

  ref:
    log_prob_use_dynamic_bsz: True
    log_prob_max_token_len_per_gpu: 16384
    megatron:
      param_offload: True  # keep ref params on CPU when not computing log probs
      tensor_model_parallel_size: 2
      pipeline_model_parallel_size: 2
      vanilla_mbridge: False  # use Megatron-Bridge (NVIDIA) instead of mbridge

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

    # Smooth length penalty starting at 1000 words, max -0.2 at 2000 words.
    # Kimi-K2.6 produces long thinking traces; penalise only runaway verbosity.
    words = len(solution_str.split())
    length_penalty = -0.2 * min(1.0, max(0.0, (words - 1000) / 1000))

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


# Download model (skip if already present)
if [ ! -d "${TRAINING_HOME}/models/${MODEL_NAME}" ]; then
    echo "Downloading ${MODEL_NAME}..."
    srun --mpi=pmix --network=disable_rdzv_get -N 1 --ntasks=1 -u \
        --environment="${TRAINING_CONFIG}/env.toml" \
        --container-writable bash -c '
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

# Redirect pip cache to local tmpfs — ~/.cache/pip is on Lustre which causes
# "Stale file handle" (ESTALE) errors during package downloads.
export PIP_CACHE_DIR=/tmp/pip_cache_${SLURM_JOB_ID}
export TMPDIR=/tmp
mkdir -p $PIP_CACHE_DIR

# Cluster proxy intercepts PyPI HTTPS; --trusted-host bypasses the Content-Type check.
PIP_TRUSTED="--trusted-host pypi.org --trusted-host files.pythonhosted.org"

pip install "PyYAML>=6.0.2" --ignore-installed ${PIP_TRUSTED} --quiet

# megatron-bridges PyPI wheel ships only megatron/bridge/* — its megatron-core dep
# (a local path in the source repo) is excluded from requires_dist, so install separately.
pip install megatron-bridge "megatron-core>=0.17.0" "nvidia-resiliency-ext>=0.6.0" \
    ${PIP_TRUSTED} --quiet
pip install nvidia-modelopt[hf] ${PIP_TRUSTED} --quiet

# megatron-bridges mlflow/comet-ml deps upgrade opentelemetry-exporter-prometheus
# to 0.64b0 which requires opentelemetry-sdk~=1.43.0, but the SDK stays at 1.33.1
# → Ray dashboard crashes with AttributeError → GCS never starts → all workers fail.
# Fix: upgrade the SDK to match the already-upgraded exporters.
pip install "opentelemetry-sdk>=1.43.0" ${PIP_TRUSTED} --quiet

# Restore sglang-compatible versions that megatron-bridge overwrites
pip install "transformers==5.8.1" "timm==1.0.16" ${PIP_TRUSTED} --quiet
pip install "flashinfer_cubin==0.6.12" "flashinfer_python[cu13]==0.6.12" \
    ${PIP_TRUSTED} --quiet





# Redirect all JIT/kernel caches to local tmpfs — Lustre does not support file locking
export FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer_${SLURM_JOB_ID}
mkdir -p $FLASHINFER_WORKSPACE_BASE

export TRITON_CACHE_DIR=/tmp/triton_${SLURM_JOB_ID}
mkdir -p $TRITON_CACHE_DIR

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

# Required for Megatron communication/computation overlapping
export CUDA_DEVICE_MAX_CONNECTIONS=1

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
