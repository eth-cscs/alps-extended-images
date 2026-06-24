#!/bin/bash

#SBATCH --nodes=8
#SBATCH --account=csstaff
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=5:00:00

export VERL_IMAGE="jfrog.svc.cscs.ch/docker-group-csstaff/alps-images/verl:alps5"

#export MODEL_NAME="Apertus-70B-Instruct-2509"
export MODEL_NAME="Apertus-8B-Instruct-2509"
export MODEL_REPO="swiss-ai"

export PROJECT_NAME="scale-grpo-gsm8k"
export EXPERIMENT_NAME="${MODEL_NAME}-grpo-gsm8k-full-async"
export RUN_NAME="${EXPERIMENT_NAME}-run-${SLURM_JOB_ID}"
export TRAINING_HOME=/capstor/scratch/cscs/${USER}/RL/${MODEL_NAME}
export CHECKPOINT_HOME=${TRAINING_HOME}/checkpoints/${EXPERIMENT_NAME}-run-${SLURM_JOB_ID} #remove "run-${SLURM_JOB_ID}" to enable checkpoint resuming


mkdir -p $TRAINING_HOME
cd $TRAINING_HOME

cat > "env.toml" <<- EOF
image = "${VERL_IMAGE}"
mounts = ["/capstor", "/iopsstor", "/users"]
workdir = "/workspace/verl"
writable = true
[env]
PMIX_MCA_psec = "native"
[annotations]
com.hooks.cxi.enabled = "false"
EOF

cat > "grpo_gsm8k.yaml" <<- EOF
defaults:
  - ppo_trainer
  - override rollout@actor_rollout_ref.rollout: rollout
  - override actor@actor_rollout_ref.actor: dp_actor
  - override data@data: legacy_data
  - _self_

# ── Required by fully_async_main ──────────────────────────────────────────────
async_training:
  staleness_threshold: 0.1
  trigger_parameter_sync_step: 4
  require_batches: 2
  partial_rollout: True
  use_trainer_do_validate: False

# Top-level rollout block — fully_async_main copies .nnodes/.n_gpus_per_node
# into actor_rollout_ref.rollout, so keep these in sync with the rollout block below.
rollout:
  nnodes: 4
  n_gpus_per_node: 4
  n: 8
  total_rollout_steps: 51200   # adjust to your dataset size * desired steps
# ──────────────────────────────────────────────────────────────────────────────

data:
  train_files: ${TRAINING_HOME}/data/gsm8k/train.parquet
  val_files:   ${TRAINING_HOME}/data/gsm8k/test.parquet
  train_batch_size: 0    # must be 0 in fully-async mode
  gen_batch_size: 1      # must be 1 in fully-async mode
  max_prompt_length: 512
  max_response_length: 1024
  apply_chat_template_kwargs:
    enable_thinking: true

actor_rollout_ref:
  hybrid_engine: False

  model:
    path: ${TRAINING_HOME}/models/${MODEL_NAME}
    override_config:
      attn_implementation: flash_attention_2
    use_shm: false

  actor:
    strategy: fsdp2
    rollout_n: 8
    ppo_mini_batch_size: 32
    ppo_micro_batch_size_per_gpu: 2
    use_torch_compile: false
    use_rollout_log_probs: True   # required for fully-async log prob correctness
    optim:
      lr: 1.0e-6

  rollout:
    name: sglang
    mode: async
    load_format: dummy
    nnodes: 4
    n_gpus_per_node: 4
    temperature: 1.0
    n: 8
    tensor_model_parallel_size: 2
    gpu_memory_utilization: 0.8 #don't set too high, otherwise the rollout will OOM, need to leave a buffer for NCCL comms.
    log_prob_micro_batch_size_per_gpu: 4
    calculate_log_probs: True     # required; log probs must come from rollout
    checkpoint_engine:
      backend: nccl               # weight sync via NCCL broadcast
    engine_kwargs:
      sglang:
        disable_piecewise_cuda_graph: true

  ref:
    fsdp_config:
      param_offload: true

algorithm:
  adv_estimator: grpo
  kl_ctrl:
    type: adaptive
    kl_coef: 0.001
    target_kl: 0.05
    horizon: 10000
  rollout_correction:
    bypass_mode: True             # required for off-policy log prob correction

reward:
  custom_reward_function:
    path: ${TRAINING_HOME}/gsm8k_reward.py
    name: compute_reward

trainer:
  total_epochs: 3
  project_name: ${PROJECT_NAME}
  experiment_name: ${RUN_NAME}
  nnodes: 4
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

cat > "gsm8k_reward.py" <<- EOF
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
    model_ans = extract_model_answer(solution_str)
    has_answer = "<answer>" in solution_str and "</answer>" in solution_str
    format_reward  = 0.1 if has_answer else 0.0
    outcome_reward = 1.0 if (model_ans is not None and model_ans == str(ground_truth)) else 0.0

    # Soft length penalty: discourage responses over 800 tokens
    # No penalty under 800, linear penalty above up to -0.1 at 2048 tokens
    length = len(solution_str.split())
    length_penalty = 0.0
    if length > 800:
        length_penalty = -0.1 * min(1.0, (length - 800) / 1248)

    return outcome_reward + format_reward + length_penalty
EOF

cat > "prepare_gsm8k.py" <<- EOF
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

# Download model (skip if already present)
if [ ! -d "${TRAINING_HOME}/models/${MODEL_NAME}" ]; then
    echo "Downloading ${MODEL_NAME}..."
    srun --mpi=pmix --network=disable_rdzv_get -N 1 --ntasks=1 -u \
        --environment="${TRAINING_HOME}/env.toml" \
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
        --environment="${TRAINING_HOME}/env.toml" \
        --container-writable bash -c '
        # Try loading from cached raw download first, otherwise fetch from HF
        python ${TRAINING_HOME}/prepare_gsm8k.py
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
    --environment="${TRAINING_HOME}/env.toml" \
    --container-writable bash -c '

# Patch flash attention NoneType bug for Qwen2 on this transformers version
sed -i "s/s_aux=s_aux\.to(query\.dtype),/s_aux=s_aux.to(query.dtype) if s_aux is not None else None,/" \
    /usr/local/lib/python3.12/dist-packages/transformers/integrations/flash_attention.py


git remote add pr_origin https://github.com/theely/verl.git 2>/dev/null || true
git fetch pr_origin Full-async-SGLang-weight-broadcasting

BASE=$(git merge-base HEAD pr_origin/Full-async-SGLang-weight-broadcasting)
git diff --name-only $BASE pr_origin/Full-async-SGLang-weight-broadcasting \
  | xargs git checkout pr_origin/Full-async-SGLang-weight-broadcasting --


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

#TODO: move this inside the verl image build.
pip install cupy-cuda13x

export NCCL_TIMEOUT=600
export NCCL_DEBUG=INFO
export NCCL_NET_GDR_LEVEL=0 #DMA-BUF registration cause hang when transfering large tensors.


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

    ROLLOUT_NNODES=$(python3 -c "import math; print(max(1, math.ceil($SLURM_JOB_NUM_NODES * 0.5)))")
    TRAINING_NNODES=$(( SLURM_JOB_NUM_NODES - ROLLOUT_NNODES ))

    HYDRA_FULL_ERROR=1 python -m verl.experimental.fully_async_policy.fully_async_main \
        --config-path ${TRAINING_HOME} \
        --config-name grpo_gsm8k \
        --config-dir /workspace/verl/verl/trainer/config \
        trainer.nnodes=${TRAINING_NNODES} \
        actor_rollout_ref.rollout.nnodes=${ROLLOUT_NNODES}
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