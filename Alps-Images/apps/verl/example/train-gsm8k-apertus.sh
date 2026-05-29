#!/bin/bash

#SBATCH --nodes=2
#SBATCH --account=csstaff
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=8:00:00

export VERL_IMAGE="jfrog.svc.cscs.ch/docker-group-csstaff/alps-images/verl:alps5-dev-3c816d5334982789"

export MODEL_NAME="Apertus-8B-Instruct-2509"
export MODEL_REPO="swiss-ai"

export TRAINING_HOME=/capstor/scratch/cscs/${USER}/RL/${MODEL_NAME}

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

data:
  train_files: ${TRAINING_HOME}/data/gsm8k/train.parquet
  val_files:   ${TRAINING_HOME}/data/gsm8k/test.parquet
  train_batch_size: 256
  max_prompt_length: 512
  max_response_length: 1024
  apply_chat_template_kwargs:
    enable_thinking: true

actor_rollout_ref:
  model:
    path: ${TRAINING_HOME}/models/${MODEL_NAME}
    override_config:
      attn_implementation: flash_attention_2

  actor:
    strategy: fsdp
    rollout_n: 8
    ppo_mini_batch_size: 32
    ppo_micro_batch_size_per_gpu: 2
    use_torch_compile: false
    optim:
      lr: 1.0e-6

  rollout:
    name: sglang
    nnodes: 0
    n_gpus_per_node: 4
    temperature: 1.0
    n: 8
    tensor_model_parallel_size: 2
    gpu_memory_utilization: 0.5
    log_prob_micro_batch_size_per_gpu: 4

  ref:
    fsdp_config:
      param_offload: true

algorithm:
  adv_estimator: grpo
  kl_ctrl:
    kl_coef: 0.001

reward:
  custom_reward_function:
    path: ${TRAINING_HOME}/gsm8k_reward.py
    name: compute_reward

trainer:
  total_epochs: 3
  project_name: grpo-gsm8k
  experiment_name: ${MODEL_NAME}-grpo-run1
  nnodes: \${oc.env:SLURM_JOB_NUM_NODES,1}
  n_gpus_per_node: 4
  save_freq: 50
  default_local_dir: ${TRAINING_HOME}/checkpoints/grpo-gsm8k
  logger: ["console"]

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
        return str(int(val)) if val == int(val) else str(val)
    except ValueError:
        return raw


def compute_reward(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    **kwargs,
) -> float:
    """
    Reward function for GSM8K with Apertus (or any model using <answer> tags).

    Reward breakdown:
      +1.0  correct answer inside <answer> tags
      +0.1  correct format (<answer> tag present) even if answer wrong
       0.0  no <answer> tag at all

    Thinking format (deliberation) is not required for format reward —
    this avoids penalising the model early in training before it learns
    to produce reasoning chains.
    """
    model_ans = extract_model_answer(solution_str)

    # Format reward: model produced an <answer> tag
    has_answer = "<answer>" in solution_str and "</answer>" in solution_str
    format_reward = 0.1 if has_answer else 0.0

    # Outcome reward: answer is correct
    outcome_reward = 1.0 if (
        model_ans is not None and model_ans == str(ground_truth)
    ) else 0.0

    return outcome_reward + format_reward
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


srun --mpi=pmix --network=disable_rdzv_get -N ${SLURM_JOB_NUM_NODES} --ntasks-per-node=1 -u \
    --environment="${TRAINING_HOME}/env.toml" \
    --container-writable bash -c '

# Patch flash attention NoneType bug for Qwen2 on this transformers version
sed -i "s/s_aux=s_aux\.to(query\.dtype),/s_aux=s_aux.to(query.dtype) if s_aux is not None else None,/" \
    /usr/local/lib/python3.12/dist-packages/transformers/integrations/flash_attention.py


# Redirect all JIT/kernel caches to local tmpfs — Lustre does not support file locking
export TRITON_CACHE_DIR="/tmp/triton_cache_${SLURM_JOB_ID}_${SLURM_PROCID}"
export TRITON_HOME="/tmp/triton_home_${SLURM_JOB_ID}_${SLURM_PROCID}"
export FLASHINFER_CACHE_DIR="/tmp/flashinfer_cache_${SLURM_JOB_ID}_${SLURM_PROCID}"
export FLASHINFER_JIT_CACHE_DIR="/tmp/flashinfer_jit_${SLURM_JOB_ID}_${SLURM_PROCID}"
export TORCHINDUCTOR_CACHE_DIR="/tmp/inductor_cache_${SLURM_JOB_ID}_${SLURM_PROCID}"
mkdir -p $TRITON_CACHE_DIR $TRITON_HOME $FLASHINFER_CACHE_DIR $FLASHINFER_JIT_CACHE_DIR $TORCHINDUCTOR_CACHE_DIR

# Also disable CUDA graphs in SGLang to avoid the capture issue
export SGLANG_DISABLE_CUDA_GRAPH=1


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

    HYDRA_FULL_ERROR=1 python -m verl.trainer.main_ppo_sync \
        --config-path ${TRAINING_HOME} \
        --config-name grpo_gsm8k \
        --config-dir /workspace/verl/verl/trainer/config \
        trainer.nnodes=${SLURM_JOB_NUM_NODES}
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