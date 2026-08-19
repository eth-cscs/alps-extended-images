#!/bin/bash

#SBATCH --nodes=4
#SBATCH --account=csstaff
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=01:00:00

# =============================================================================
# SMALL-MODEL LEAK-REPRO RIG (see HANDOFF.md, "the shm staircase").
#
# Purpose: reproduce the per-refit shared-memory staircase of the GLM-5.1
# campaign on 2 nodes in minutes instead of 104 nodes in hours.  This is NOT
# a training run — rewards and loss are irrelevant; the deliverable is
# ${TRAINING_HOME}/mem_trace/<jobid>/ after ~8 training steps (= ~10 refit
# events including collective setup + initial refit).
#
# What must stay FAITHFUL to the big run (the suspect code path):
#   - Megatron training backend (not dtensor) — export_hf_weights path
#   - non-colocated vLLM with the collective-broadcast refit
#   - async GRPO with max_trajectory_age_steps=1 (refit every step)
#   - distributed optimizer, same refit env knobs, same driver+tracer
#
# What to look for in mem_trace: Shmem stepping up at every refit and never
# coming down.  Expected quantum scales with the per-rank param shard —
# Qwen3-30B-A3B at EP=8 puts ~3.8B params (~7.6 GB bf16) on each training
# rank, so staircase steps of very roughly that order per rank, NOT the
# 95 GiB of the 700B.  Flat Shmem across 8 steps = no repro at this scale/
# MoE shape — also a result; see HANDOFF.
#
# Interactive tip: while it runs, on the training node
#   watch -n5 "grep Shmem /proc/meminfo"          # the staircase live
#   ls -l /proc/<worker-pid>/fd | grep deleted    # the segments live
# =============================================================================

SCRIPT_DIR="${PWD}"

export NEMORL_IMAGE="/capstor/scratch/cscs/phimuell/.uenv-images/__ML__/nemo_rl_2026_08_13_I.sqsh"

export WANDB_API_KEY="${WANDB_API_KEY:-}"
if [[ -z "$WANDB_API_KEY" ]]; then
    echo "[WARNING] WANDB_API_KEY is not set; W&B logging will fail or run in offline mode." >&2
    unset WANDB_API_KEY
fi

if [ -n "${HF_HOME}" ]; then
    echo "[INFO] Found HF_HOME." >&2
elif [ -n "$HF_TOKEN_PATH" ]; then
    echo "[INFO] Found HF_TOKEN_PATH." >&2
else
    echo "[WARNING] HF_TOKEN_PATH is not set; HuggingFace downloads may be rate-limited." >&2
fi

# Qwen3-30B-A3B per the upstream async recipe this rig inherits from
# (examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-async-1off
# .yaml) — an upstream-TESTED MoE async config on exactly our 4-GPU-node
# shape, which also exercises the expert-gather export path (prime leak
# suspect).  ~60 GB download on first run.
export MODEL_NAME="${QWEN_MODEL:-Qwen3-30B-A3B}"
export MODEL_REPO="Qwen"
export WANDB_PROJECT_NAME="shm-leak-repro"
export WANDB_RUN_NAME="${MODEL_NAME}-repro-${SLURM_JOB_NUM_NODES}n-${SLURM_JOB_ID}"
export EXPERIMENT_NAME="${WANDB_RUN_NAME}"
export TRAINING_HOME="/capstor/scratch/cscs/${USER}/RL/qwen-small-repro"
export TRAINING_CONFIG="${TRAINING_HOME}/config"
export CHECKPOINT_HOME="${TRAINING_HOME}/checkpoints/${EXPERIMENT_NAME}"
export LOCAL_MODEL_DIR="${TRAINING_HOME}/models/${MODEL_NAME}"

export NRL_MEGATRON_CHECKPOINT_DIR="${TRAINING_HOME}/megatron_ckpts"
mkdir -p "${NRL_MEGATRON_CHECKPOINT_DIR}"

export NEMORL_DIR="/workdir/nemo_rl"
export NEMORL_FORK_URL="https://github.com/philip-paul-mueller/RL.git"
# Image and branch ship together (see the GLM entry script).
export NEMORL_BRANCH="${NEMORL_BRANCH:-glm51-megatron-fsdp-wiring}"

# Recipe split: 2 training nodes (8 GPUs, EP=8, dense DP=8) + 2 rollout
# nodes (8 x TP=1 vLLM engines).
export ROLLOUT_NNODES=2
export TRAINING_NNODES=$(( SLURM_JOB_NUM_NODES - ROLLOUT_NNODES ))

export YAML_NAME="grpo_gsm8k_qwen_small_repro.yaml"

mkdir -p "${TRAINING_HOME}"
mkdir -p "${TRAINING_CONFIG}"
cd "${TRAINING_HOME}"

export MASTER_NODE=$(hostname)
export MASTER_NODE_IP=$(hostname -i)
export PORT=6382
export RAY_ADDRESS="${MASTER_NODE_IP}:${PORT}"

# -----------------------------------------------------------------------------
# Container environment — IDENTICAL knobs to the GLM run on purpose: the rig
# must exercise the same refit/env code path (incl. the small packed-broadcast
# buckets and symlink venvs).  NCCL INFO stays on so communicator creation can
# be correlated with Shmem jumps in the trace.
# -----------------------------------------------------------------------------
cat > "${TRAINING_CONFIG}/env.toml" <<- EOF
image = "${NEMORL_IMAGE}"
mounts = ["/capstor", "/iopsstor", "/users", "/tmp"]
workdir = "${NEMORL_DIR}"
writable = true
entrypoint = true
[env]
PMIX_MCA_psec = "native"
WANDB_API_KEY = "${WANDB_API_KEY}"
HF_TOKEN_PATH = "${HF_TOKEN_PATH}"
RAY_ADDRESS = "${RAY_ADDRESS}"
MASTER_NODE_IP = "${MASTER_NODE_IP}"
PORT = "${PORT}"
TRITON_CACHE_DIR = "/tmp/triton_cache_${SLURM_JOB_ID}_${SLURM_PROCID}"
TRITON_HOME = "/tmp/triton_home_${SLURM_JOB_ID}_${SLURM_PROCID}"
VLLM_TORCH_COMPILE_CACHE_DIR = "/tmp/vllm_torch_compile_cache_${SLURM_JOB_ID}_${SLURM_PROCID}"
TORCHINDUCTOR_CACHE_DIR = "/tmp/torchinductor_cache_${SLURM_JOB_ID}_${SLURM_PROCID}"
VLLM_CACHE_ROOT = "/tmp/vllm_cache_${SLURM_JOB_ID}_${SLURM_PROCID}"
NRL_MEGATRON_CHECKPOINT_DIR = "${NRL_MEGATRON_CHECKPOINT_DIR}"
PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
NCCL_DEBUG = "INFO"
NCCL_DEBUG_SUBSYS = "INIT,NET"
NCCL_NVLS_ENABLE = "0"
NRL_REFIT_BUFFER_MEMORY_RATIO = "0.005"
NRL_REFIT_NUM_BUFFERS = "1"
UV_LINK_MODE = "symlink"
[annotations]
com.hooks.cxi.enabled = "false"
EOF

# -----------------------------------------------------------------------------
# GRPO config: INHERITS the upstream-tested async MoE recipe
#   grpo-qwen3-30ba3b-4n4g-async-1off.yaml
# (megatron backend, EP=8, non-colocated 2+2 nodes, async lag-1, in-flight
# weight updates, vLLM TP=1 engines, moe_backend=triton refit workaround)
# and overrides ONLY: model path, size/speed knobs (8 steps of 8x8), data
# (GSM8K), and output dirs.  Everything else is the recipe as upstream ships
# and tests it.
# -----------------------------------------------------------------------------
cat > "${TRAINING_CONFIG}/${YAML_NAME}" <<- EOF
defaults: ${NEMORL_DIR}/examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-async-1off.yaml

grpo:
  num_prompts_per_step: 8
  num_generations_per_prompt: 8
  max_num_epochs: 1
  # 8 steps = ~10 refit events incl. setup — enough staircase to judge.
  max_num_steps: 8
  val_period: 0
  val_at_start: false
  val_at_end: false

checkpointing:
  enabled: false
  checkpoint_dir: ${CHECKPOINT_HOME}

policy:
  # Locally cached copy (driver downloads it once, striped).
  model_name: "${LOCAL_MODEL_DIR}"
  train_global_batch_size: 64
  max_total_sequence_length: 2048

  generation:
    max_new_tokens: 512
    vllm_cfg:
      enable_vllm_metrics_logger: true
      vllm_metrics_logger_interval: 30.0
    colocated:
      enabled: false
      resources:
        num_nodes: ${ROLLOUT_NNODES}
        gpus_per_node: 4

# GSM8K instead of the parent chain math data — small and familiar.
data:
  _override_: true
  max_input_seq_length: \${policy.max_total_sequence_length}
  shuffle: true
  num_workers: 1
  use_multiple_dataloader: false
  train:
    dataset_name: "gsm8k"
    split: train
  validation:
    dataset_name: "gsm8k"
    split: test
  default:
    prompt_file: null
    system_prompt_file: ${NEMORL_DIR}/examples/prompts/gsm8k.txt
    processor: "math_hf_data_processor"
    env_name: "math"

env:
  math:
    num_workers: 2
    math_verify_impl: "hf_math_verify"

logger:
  log_dir: ${TRAINING_HOME}/logs
  wandb_enabled: true
  tensorboard_enabled: false
  mlflow_enabled: false
  swanlab_enabled: false
  monitor_gpus: true
  wandb:
    project: ${WANDB_PROJECT_NAME}
    name: ${WANDB_RUN_NAME}

cluster:
  num_nodes: ${SLURM_JOB_NUM_NODES}
  gpus_per_node: 4
  master_port_range_low: 25000
  master_port_range_high: 28000
EOF

# -----------------------------------------------------------------------------
# Hand over to the shared driver (model download, striping, Ray, tracer,
# salvage — everything identical to the big run).
# -----------------------------------------------------------------------------
source "${SCRIPT_DIR}/nemorl-grpo-driver.sh"
