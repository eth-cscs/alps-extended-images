#!/bin/bash

#SBATCH --nodes=10
#SBATCH --account=csstaff
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=04:00:00

# =============================================================================
# GRPO on Apertus v1.5 70B (swiss-ai) — dedicated fork stack, dtensor backend.
#
# TARGET: Apertus-v1.5-70B on the DEDICATED APERTUS STACK — the NeMoRL
# branch `apertus-stack` pins swiss-ai's forked transformers (@3797303d)
# and modified vLLM (@a601a9d9) in the lockfile, and the image is built
# from that branch (see NeMoRL/APERTUS_STACK.md for the relock + image
# build procedure; both MUST exist before this script can run).  The
# shared driver is stack-agnostic — only image+branch change.
#   - Megatron-Bridge has ZERO Apertus support (grep-verified) → the
#     megatron backend is impossible for ANY Apertus.  Hence dtensor/FSDP2.
#   - v1.5 loads via the new AutoModelForMultimodalLM class — the NeMo-RL
#     automodel load path may need extending (first light will tell; the
#     vlm_grpo automodel recipes are the in-tree precedent).
#   - Fallback that runs on the NORMAL stack today:
#     APERTUS_MODEL=Apertus-70B-Instruct-2509 NEMORL_BRANCH=glm51-megatron-fsdp-wiring \
#         sbatch ... (text-only v1.0, upstreamed in vllm 0.25.1/transformers 5.5).
#
# Geometry (Apertus-70B v1.0): 80 layers, hidden 8192, 64 attention heads,
# 8 KV heads (GQA), vocab 131,072.  Divisibility: TP=4 → 64/4=16 q-heads,
# 8/4=2 kv-heads per rank ✓ (both training dtensor-TP and vLLM TP).
# ⚠️ After the first model download, sanity-check
#   grep -E "num_attention_heads|num_key_value_heads|num_hidden_layers" \
#     ${LOCAL_MODEL_DIR}/config.json
# against the numbers above before blaming anything else.
#
# The repo is GATED on HF — accept the license with the account whose token
# HF_TOKEN_PATH points at, or the driver's download step will 401.
#
# Sizing: 8 training nodes (32 GPUs; FSDP2 across 8 with TP=4) + 2 rollout
# nodes (2 vLLM engines, TP=4, ~35 GB weights/rank).  Optimizer state
# (~840 GB FP32 total) shards to ~26 GB/rank.  Async lag-1, non-colocated —
# same pipeline family as the GLM run, different training backend.
# =============================================================================

SCRIPT_DIR="${PWD}"

# The Apertus-stack image (built from NEMORL_COMMIT=apertus-stack per
# APERTUS_STACK.md).  Override APERTUS_IMAGE while iterating on builds.
export NEMORL_IMAGE="${APERTUS_IMAGE:-/capstor/scratch/cscs/phimuell/.uenv-images/__ML__/nemo_rl_apertus15.sqsh}"

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
    echo "[WARNING] HF_TOKEN_PATH is not set — the Apertus repo is GATED; the download WILL fail without an authorized token." >&2
fi

export MODEL_NAME="${APERTUS_MODEL:-Apertus-v1.5-70B}"
export MODEL_REPO="swiss-ai"
export WANDB_PROJECT_NAME="async-grpo-gsm8k"
export WANDB_RUN_NAME="${MODEL_NAME}-nemorl-vllm-dtensor-async-${SLURM_JOB_NUM_NODES}n-${SLURM_JOB_ID}"
export EXPERIMENT_NAME="${WANDB_RUN_NAME}"
export TRAINING_HOME="/capstor/scratch/cscs/${USER}/RL/${MODEL_NAME}"
export TRAINING_CONFIG="${TRAINING_HOME}/config"
export CHECKPOINT_HOME="${TRAINING_HOME}/checkpoints/${EXPERIMENT_NAME}"
export LOCAL_MODEL_DIR="${TRAINING_HOME}/models/${MODEL_NAME}"

# Unused by the dtensor path but expected by the shared env.toml.
export NRL_MEGATRON_CHECKPOINT_DIR="${TRAINING_HOME}/megatron_ckpts"
mkdir -p "${NRL_MEGATRON_CHECKPOINT_DIR}"

export NEMORL_DIR="/workdir/nemo_rl"
export NEMORL_FORK_URL="https://github.com/philip-paul-mueller/RL.git"
# Image and branch ship together (see the GLM entry script).
export NEMORL_BRANCH="${NEMORL_BRANCH:-apertus-stack}"

export ROLLOUT_NNODES=2
export TRAINING_NNODES=$(( SLURM_JOB_NUM_NODES - ROLLOUT_NNODES ))

export YAML_NAME="grpo_gsm8k_apertus70b_async_nemorl.yaml"

mkdir -p "${TRAINING_HOME}"
mkdir -p "${TRAINING_CONFIG}"
cd "${TRAINING_HOME}"

export MASTER_NODE=$(hostname)
export MASTER_NODE_IP=$(hostname -i)
export PORT=6382
export RAY_ADDRESS="${MASTER_NODE_IP}:${PORT}"

# -----------------------------------------------------------------------------
# Container environment — same hardened knobs as the GLM run.
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
# GRPO config: dtensor (FSDP2) training + non-colocated async vLLM rollout.
#   Training: 8 nodes x 4 GPUs; dtensor TP=4, FSDP2 across the rest.
#   Rollout:  2 nodes x 4 GPUs = two TP=4 engines.
# First-light run: 20 steps, checkpointing off.  Scale max_num_steps and
# enable checkpointing once it holds.
# -----------------------------------------------------------------------------
cat > "${TRAINING_CONFIG}/${YAML_NAME}" <<- EOF
defaults: ${NEMORL_DIR}/examples/configs/grpo_math_1B.yaml

grpo:
  num_prompts_per_step: 16
  num_generations_per_prompt: 8
  max_num_epochs: 1
  max_num_steps: 20
  val_period: 0
  val_at_start: false
  val_at_end: false
  normalize_rewards: true
  use_leave_one_out_baseline: true

  async_grpo:
    enabled: true
    max_trajectory_age_steps: 1
    in_flight_weight_updates: true

loss_fn:
  reference_policy_kl_penalty: 0.001
  use_importance_sampling_correction: true

checkpointing:
  enabled: false
  checkpoint_dir: ${CHECKPOINT_HOME}

policy:
  model_name: "${LOCAL_MODEL_DIR}"
  train_global_batch_size: 128
  train_micro_batch_size: 1
  logprob_batch_size: 2
  max_total_sequence_length: 2048
  precision: "bfloat16"
  make_sequence_length_divisible_by: 4

  # dtensor/FSDP2 backend — Megatron-Bridge has no Apertus mapping.
  dtensor_cfg:
    enabled: true
    _v2: true
    tensor_parallel_size: 4
    context_parallel_size: 1
    sequence_parallel: false
    activation_checkpointing: true
    cpu_offload: false

  megatron_cfg:
    enabled: false

  generation:
    backend: "vllm"
    max_new_tokens: 1024
    temperature: 1.0
    top_p: 1.0
    top_k: null
    vllm_cfg:
      async_engine: true
      precision: bfloat16
      kv_cache_dtype: "auto"
      # 64 q-heads / 8 kv-heads: TP=4 divides both (16 q, 2 kv per rank).
      tensor_parallel_size: 4
      pipeline_parallel_size: 1
      gpu_memory_utilization: 0.7
      max_model_len: \${policy.max_total_sequence_length}
      enforce_eager: true
      use_tqdm: true
      enable_vllm_metrics_logger: true
      vllm_metrics_logger_interval: 30.0
    vllm_kwargs: {}
    colocated:
      enabled: false
      resources:
        num_nodes: ${ROLLOUT_NNODES}
        gpus_per_node: 4

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
    num_workers: 4
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
# Hand over to the shared driver (download+striping, Ray, tracer, salvage).
# -----------------------------------------------------------------------------
source "${SCRIPT_DIR}/nemorl-grpo-driver.sh"
