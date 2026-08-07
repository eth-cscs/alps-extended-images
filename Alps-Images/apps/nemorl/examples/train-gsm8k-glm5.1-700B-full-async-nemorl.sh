#!/bin/bash

#SBATCH --nodes=136
#SBATCH --account=csstaff
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=12:00:00

# Because slurm makes a copy of the script we can not find it through `BASH_SOURCE`
#  instead we relly on that it is in the folder we started.
#SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="${PWD}"

# Specific settings
#export NEMORL_IMAGE="/capstor/scratch/cscs/phimuell/.uenv-images/__ML__/nemo_rl_2026_07_21_I.sqsh"
export NEMORL_IMAGE="/capstor/scratch/cscs/phimuell/.uenv-images/__ML__/nemo_rl_2026_08_04_II.sqsh"

# W&B is enabled in the NeMo-RL config below. Make sure the API key is exported
# in the environment before submitting; it is forwarded into the container.
export WANDB_API_KEY="${WANDB_API_KEY:-}"
if [[ -z "$WANDB_API_KEY" ]]; then
    echo "[WARNING] WANDB_API_KEY is not set; W&B logging will fail or run in offline mode." >&2
    unset WANDB_API_KEY
fi

# HuggingFace token: strongly recommended for multi-node runs. Without it, all
# policy workers download the model unauthenticated and easily hit rate limits,
# causing stragglers that fail the torch distributed rendezvous.
# Prefer HF_TOKEN_PATH so the secret is not embedded in the job script/env.toml.
if [ -n "${HF_HOME}" ]
then
	echo "[INFO] Found HF_HOME." >&2
elif [ -n "$HF_TOKEN_PATH" ]
then
	echo "[INFO] Found HF_TOKEN_PATH." >&2
else
	echo "[WARNING] HF_TOKEN_PATH is not set; HuggingFace downloads may be rate-limited on many workers." >&2
fi

export MODEL_NAME="GLM-5.1"
export MODEL_REPO="zai-org"
export WANDB_PROJECT_NAME="async-grpo-gsm8k"
export WANDB_RUN_NAME="${MODEL_NAME}-nemorl-vllm-megatron-async-${SLURM_JOB_NUM_NODES}n-${SLURM_JOB_ID}"
export EXPERIMENT_NAME="${WANDB_RUN_NAME}"
export TRAINING_HOME="/capstor/scratch/cscs/${USER}/RL/${MODEL_NAME}"
export TRAINING_CONFIG="${TRAINING_HOME}/config"
export CHECKPOINT_HOME="${TRAINING_HOME}/checkpoints/${EXPERIMENT_NAME}"
export LOCAL_MODEL_DIR="${TRAINING_HOME}/models/${MODEL_NAME}"

# Megatron-Bridge converts the HF checkpoint to a torch_dist checkpoint once on
# startup. That output must live on a filesystem with enough quota for a 700B
# sharded model; /users/$USER/.hf is usually too small. Use the scratch area.
export NRL_MEGATRON_CHECKPOINT_DIR="${TRAINING_HOME}/megatron_ckpts"
mkdir -p "${NRL_MEGATRON_CHECKPOINT_DIR}"

# Path where the container file installs the repo.
export NEMORL_DIR="/workdir/nemo_rl"

# Fork and branch carrying NeMo-RL patches for this run (see HANDOFF.md).
# The shared driver checks out this branch inside the container before Ray starts.
export NEMORL_FORK_URL="https://github.com/philip-paul-mueller/RL.git"
export NEMORL_BRANCH="glm51-megatron-fsdp-wiring"

# Node split: 8 nodes for non-colocated vLLM generation (TP=32, one replica),
# remaining 128 nodes for Megatron training (TP=8, PP=8, EP=64 → 512 GPUs, DP=8).
#
# This mirrors the validated demo config
# (examples/configs/recipes/llm/grpo-glm5.1-64n8g-megatron.yaml: TP=8, PP=8,
# EP=64, DP=8 on 512 GPUs) adapted to our 4-GPU async non-colocated setup.
# TP=8 spans 2 nodes per TP group (cross-node TP via NCCL — slower than
# node-local TP=4 but required: at TP=4/PP=8 the legacy DDP param+grad buffer
# is ~88 GB, exceeding the 95 GB GH200).
#
# Divisibility: expert_TP=1 (parent default), so
#   expert_tensor_model_pipeline_parallel = 1 * EP * PP = 512.
#   world=512 % 512 = 0 ✓ (expert_DP=1).
#   world=512 % (TP*PP) = 512 % 64 = 0 ✓ (DP=8).
# Fewer training nodes (e.g. 64 → world=256) fails: 256 % 512 ≠ 0.
#
# GLM-5.1 has 78 layers; PP=8 with num_layers_in_first/last_pipeline_stage=9
# gives 9 + 6×10 + 9 = 78.  512 total experts; EP=64 → 8/rank (matches
# megatron-bridge's GLM-5.1 mapping, same as the demo).
export ROLLOUT_NNODES=8
export TRAINING_NNODES=$(( SLURM_JOB_NUM_NODES - ROLLOUT_NNODES ))
export ROLLOUT_NNODES=8
export TRAINING_NNODES=$(( SLURM_JOB_NUM_NODES - ROLLOUT_NNODES ))

# Name of the generated YAML config consumed by the shared driver.
export YAML_NAME="grpo_gsm8k_glm51_async_nemorl.yaml"

mkdir -p "${TRAINING_HOME}"
mkdir -p "${TRAINING_CONFIG}"
cd "${TRAINING_HOME}"

# Ray cluster settings (must be defined before the env.toml heredoc).
export MASTER_NODE=$(hostname)
export MASTER_NODE_IP=$(hostname -i)
export PORT=6382
export RAY_ADDRESS="${MASTER_NODE_IP}:${PORT}"

# -----------------------------------------------------------------------------
# Container environment (matches the verl env.toml style).
# -----------------------------------------------------------------------------
# TODO: Store the file into a temporary folder such that the script can run in parallel.
cat > "${TRAINING_CONFIG}/env.toml" <<- EOF
image = "${NEMORL_IMAGE}"
mounts = ["/capstor", "/iopsstor", "/users", "/tmp"]
workdir = "${NEMORL_DIR}"
# Needed because of venvs that is written inside the container.
writable = true
entrypoint = true
[env]
PMIX_MCA_psec = "native"
WANDB_API_KEY = "${WANDB_API_KEY}"
HF_TOKEN_PATH = "${HF_TOKEN_PATH}"
RAY_ADDRESS = "${RAY_ADDRESS}"
MASTER_NODE_IP = "${MASTER_NODE_IP}"
PORT = "${PORT}"
# Keep Triton/torch.compile caches on node-local storage; Lustre-backed \$HOME
# gives "Stale file handle" (ESTALE) when many ranks JIT-compile simultaneously.
# Make each rank use its own sub-directory to avoid intra-node races.
TRITON_CACHE_DIR = "/tmp/triton_cache_${SLURM_JOB_ID}_${SLURM_PROCID}"
TRITON_HOME = "/tmp/triton_home_${SLURM_JOB_ID}_${SLURM_PROCID}"
VLLM_TORCH_COMPILE_CACHE_DIR = "/tmp/vllm_torch_compile_cache_${SLURM_JOB_ID}_${SLURM_PROCID}"
TORCHINDUCTOR_CACHE_DIR = "/tmp/torchinductor_cache_${SLURM_JOB_ID}_${SLURM_PROCID}"
# vLLM V1 engine writes compile artifacts under \$HOME/.cache/vllm_0 by default;
# point the whole vllm cache tree to /tmp as well.
VLLM_CACHE_ROOT = "/tmp/vllm_cache_${SLURM_JOB_ID}_${SLURM_PROCID}"
# Megatron-Bridge initial HF -> Mcore conversion writes a large sharded
# checkpoint. Point it to the same scratch area used for the model/logs.
NRL_MEGATRON_CHECKPOINT_DIR = "${NRL_MEGATRON_CHECKPOINT_DIR}"
# Mitigate fragmentation during startup: the param+grad buffer (~30 GB at
# PP=13/DP=2) plus the FP32 master clone (slurm-3007452 OOM'd by 96 MiB).
# PyTorch recommends expandable_segments when small allocations fail near
# the limit (see the CUDA out of memory error message in slurm-3007452).
PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
[annotations]
com.hooks.cxi.enabled = "false"
EOF

# -----------------------------------------------------------------------------
# NeMo-RL GRPO config for GLM-5.1 (700B) with Megatron training + non-colocated
# async vLLM rollout.  Ported from the verl script
# train-gsm8k-glm5.1-700B-full-async-megatron.sh.
#
# Key mapping notes:
#   verl SGLang rollout  → NeMo-RL vLLM with async_engine=true (when EP = TP)
#     (NeMo-RL async GRPO requires vLLM or Megatron backend, not SGLang;
#      see nemo_rl/algorithms/grpo.py:3732.)
#     NOTE: vLLM async_engine with EP > TP (native DP inside vLLM) is currently
#     unsupported in NeMo-RL main; see https://github.com/NVIDIA-NeMo/RL/issues/1101
#     and open PR https://github.com/NVIDIA-NeMo/RL/pull/2517. We therefore set
#     EP = TP for the rollout so vLLM DP is not created. This is a NeMo-RL/vLLM
#     integration limitation, not a problem in the verl/SGLang path.
#   verl ppo_mini_batch_size (48) → grpo.num_prompts_per_step (48)
#   verl rollout.n (16)          → grpo.num_generations_per_prompt (16)
#   verl total_epochs (3)        → grpo.max_num_epochs (3)
#   verl total_rollout_steps (22419) → grpo.max_num_steps (468)
#     (Different counting: verl counts per-prompt rollouts over 3 epochs;
#      NeMo-RL counts training steps. 468 ≈ 3 epochs × 156 steps/epoch,
#      where 156 = ceil(7473 / 48). Verify this matches your intent on
#      first run.)
#   verl max_response_length (1024) → max_total_sequence_length (2048)
#     (1024 prompt + 1024 response.)
#   verl gpu_memory_utilization (0.75) → vllm_cfg.gpu_memory_utilization (0.75)
#
# Reward function (NOT ported — verify before relying on this run):
#   The verl script uses a custom reward (gsm8k_reward.py) with <answer>
#   extraction, a +0.1 format reward, a +1.0 exact-match outcome reward,
#   and a smooth length penalty (-0.2 at 2000+ words). NeMo-RL here uses
#   its built-in `math` env with `math_verify_impl: hf_math_verify`, which
#   extracts answers via math_verify and does exact match. The custom
#   length penalty and truncated-thinking handling are NOT directly
#   portable; if needed, implement a custom environment or reward wrapper.
#
# Offload settings (verify sufficient for 700B on 96 GPUs):
#   The verl script sets param_offload/grad_offload/optimizer_offload: True
#   explicitly. NeMo-RL's megatron_cfg does not have these fields directly;
#   use_distributed_optimizer: true + empty_unused_memory_level: 1 provide
#   the equivalent offload semantics in the Megatron-LM backend.
#
# lr_decay_iters (verify on first run):
#   Set to 468 to match max_num_steps. The verl script notes that
#   lr_decay_steps "must be positive at init; set_total_train_steps is
#   called too late (after init_workers)" — verify whether the same
#   applies to NeMo-RL's megatron path.
#
# Parallelism (mirrors examples/configs/recipes/llm/grpo-glm5.1-64n8g-megatron.yaml,
# adapted to our 4-GPU async non-colocated setup):
#   Training: 64 nodes × 4 GPUs = 256 GPUs; TP=8, PP=8, EP=64 → DP=4
#   Rollout:   8 nodes × 4 GPUs =  32 GPUs; TP=32 (one replica)
#
# TP=8 spans 2 nodes per TP group (cross-node TP via NCCL).  At TP=4/PP=8 the
# legacy DDP param+grad buffer is ~88 GB, exceeding the 95 GB GH200; TP=8
# halves per-rank params to ~10.9B → buffer ~43.8 GB (fits, matches the demo).
#
# PP=8 with num_layers_in_first/last_pipeline_stage=9: 9 + 6×10 + 9 = 78 ✓.
# EP=64 → 8 experts/rank (matches megatron-bridge's GLM-5.1 mapping, same as
# the demo).  expert_tensor_parallel_size defaults to 1 (parent
# grpo_math_1B.yaml:191); the demo does not override it.
#
# Legacy DDP (use_megatron_fsdp absent/false).  See HANDOFF.md for the FSDP
# dead end (upstream _get_dp_tp_mesh bug with PP>1) and why NeMo-RL has no
# equivalent of verl's param_offload/grad_offload.
# -----------------------------------------------------------------------------
cat > "${TRAINING_CONFIG}/${YAML_NAME}" <<- EOF
defaults: ${NEMORL_DIR}/examples/configs/grpo_math_1B_megatron.yaml

grpo:
  # 48 prompts per step x 16 generations = 768 sequences per global batch.
  num_prompts_per_step: 48
  num_generations_per_prompt: 16
  # 3 epochs over GSM8K (7,473 samples / 48 per step ≈ 156 steps/epoch → 468
  # steps total, bounded by min(max_num_epochs * len(dataloader), max_num_steps)).
  # NOTE: async GRPO does NOT consult grpo.max_num_epochs for data iteration;
  # the trajectory collector re-iterates the dataloader indefinitely and
  # training is bounded solely by grpo.max_num_steps. See the warning at
  # nemo_rl/algorithms/grpo.py:2534 and the multi-epoch support added in
  # nemo_rl/algorithms/async_utils/trajectory_collector.py (_collection_loop).
  max_num_epochs: 3
  max_num_steps: 468
  # Validation disabled: greedy decode over 1319 GSM8K test samples causes
  # cuEventSynchronize deadlocks at this scale (as observed in the verl run).
  val_period: 0
  val_at_start: false
  val_at_end: false
  max_val_samples: 1319
  val_batch_size: 256
  normalize_rewards: true
  use_leave_one_out_baseline: true

  async_grpo:
    enabled: true
    # Lag-1 async: trajectories may be at most 1 training step old.
    max_trajectory_age_steps: 1
    # Allow weight updates to be pushed while generation is still running.
    in_flight_weight_updates: true
    recompute_kv_cache_after_weight_updates: false

loss_fn:
  # Matches verl algorithm.kl_ctrl.kl_coef.
  reference_policy_kl_penalty: 0.001
  # Required by async GRPO for off-policy correction
  # (matches verl rollout_correction.bypass_mode: True).
  use_importance_sampling_correction: true

checkpointing:
  checkpoint_dir: ${CHECKPOINT_HOME}
  save_period: 50
  keep_top_k: 10
  # NOTE: ckpt_format lives under policy.megatron_cfg.checkpoint below, not here.
  # The top-level checkpointing block is the NeMo-RL PolicyConfig; the Megatron
  # Bridge CheckpointConfig (what the assertion reads) is fed from
  # policy.megatron_cfg.checkpoint via _create_checkpoint_config.

policy:
  # Use the locally cached copy downloaded once below; avoids all workers
  # hitting the HuggingFace Hub simultaneously.
  model_name: "${LOCAL_MODEL_DIR}"
  train_global_batch_size: 768
  train_micro_batch_size: 1
  logprob_batch_size: 1
  # 1024 prompt + 1024 response (matches verl max_response_length: 1024).
  max_total_sequence_length: 2048
  precision: "bfloat16"

  dtensor_cfg:
    enabled: false

  # Megatron backend for 700B training.
  # GLM-5.1 model_type=glm_moe_dsa requires Megatron-Bridge (vanilla_mbridge: False).
  #
  # Divisibility (mirrors the validated demo config
  # examples/configs/recipes/llm/grpo-glm5.1-64n8g-megatron.yaml):
  #   - GLM-5.1 has 78 transformer layers; PP=8 with
  #     num_layers_in_first/last_pipeline_stage=9 gives 9+6x10+9=78.
  #   - expert_tensor_parallel_size=1 (parent grpo_math_1B.yaml:191, not
  #     overridden), so expert_tensor_model_pipeline_parallel = 1*EP*PP = 512.
  #     world=512 % 512 = 0 (expert_DP=1).
  #   - world=512 % (TP*PP) = 512 % 64 = 0 (DP=8).
  #   - EP=64 -> 8 experts/rank (matches megatron-bridge GLM-5.1 mapping).
  #
  # Legacy DDP (use_megatron_fsdp absent).  At TP=8/PP=8, per-rank params are
  # ~10.9B -> param+grad buffer ~43.8 GB (fits 95 GB GH200, matches the demo).
  megatron_cfg:
    enabled: true
    empty_unused_memory_level: 1
    # 128 training nodes x 4 GPUs = 512 GPUs; TP=8, PP=8 -> DP=8
    # GLM-5.1 has 78 layers; num_layers_in_first/last=9 -> 9+6*10+9=78.
    tensor_model_parallel_size: 8
    pipeline_model_parallel_size: 8
    num_layers_in_first_pipeline_stage: 9
    num_layers_in_last_pipeline_stage: 9
    expert_model_parallel_size: 64
    context_parallel_size: 1
    sequence_parallel: true
    pipeline_dtype: \${policy.precision}

    # Activation checkpointing (matches verl recompute_granularity: full,
    # recompute_method: uniform, recompute_num_layers: 1).
    activation_checkpointing: true
    recompute_granularity: full
    recompute_num_layers: 1

    # GLM-5.1 DSA-specific settings (from examples/configs/recipes/llm/
    # grpo-glm5.1-64n8g-megatron.yaml).
    apply_rope_fusion: false
    defer_fp32_logits: true
    moe_token_dispatcher_type: allgather
    moe_permute_fusion: true
    moe_grouped_gemm: true

    # Freeze MoE router for GRPO (prevents logprob error divergence).
    freeze_moe_router: true
    moe_router_dtype: "fp64"
    moe_router_load_balancing_type: "none"
    moe_router_bias_update_rate: 0.0

    # Adam optimizer for GRPO.  Legacy DDP path: the bf16 param+grad buffer
    # is unsharded on GPU (~43.8 GB at TP=8/PP=8, matching the demo).  The
    # FP32 master + Adam m/v states are offloaded to CPU and sharded across
    # DP=8 by use_distributed_optimizer.  This is the closest NeMo-RL
    # equivalent to verl's optimizer_offload (NeMo-RL has no
    # param_offload/grad_offload — those are verl-only features in
    # verl/verl/workers/engine/megatron/transformer_impl.py).
    optimizer:
      optimizer: "adam"
      lr: 5.0e-7
      min_lr: 5.0e-8
      weight_decay: 0.0
      bf16: true
      fp16: false
      params_dtype: "float32"
      adam_beta1: 0.9
      adam_beta2: 0.999
      adam_eps: 1e-8
      use_distributed_optimizer: true
      use_precision_aware_optimizer: false
      clip_grad: 1.0
      # Offload FP32 master + Adam states to CPU (sharded across DP=2).
      # NeMo-RL requires optimizer_offload_fraction=1.0 for full offload.
      optimizer_cpu_offload: true
      optimizer_offload_fraction: 1.0
      overlap_cpu_optimizer_d2h_h2d: true

    scheduler:
      start_weight_decay: \${policy.megatron_cfg.optimizer.weight_decay}
      end_weight_decay: \${policy.megatron_cfg.optimizer.weight_decay}
      weight_decay_incr_style: "constant"
      lr_decay_style: "constant"
      lr_decay_iters: 468
      lr_warmup_iters: 13
      lr_warmup_init: 5.0e-8

    distributed_data_parallel_config:
      grad_reduce_in_fp32: false
      overlap_grad_reduce: true
      overlap_param_gather: true
      use_custom_fsdp: false
      data_parallel_sharding_strategy: "optim_grads_params"
      # Legacy DDP (no use_megatron_fsdp).  Megatron-FSDP with PP>1 is blocked
      # by an upstream bug in mcore_fsdp_adapter._get_dp_tp_mesh (ignores PP,
      # so world_size=104 != dp_cp*ep*tp=52).  See HANDOFF.md.

  generation:
    backend: "vllm"
    max_new_tokens: 1024
    temperature: 1.0
    top_p: 1.0
    top_k: null
    vllm_cfg:
      # Async engine is required for async GRPO.
      async_engine: true
      precision: bfloat16
      kv_cache_dtype: "auto"
      # 8 rollout nodes × 4 GPUs = 32 GPUs; TP=32, EP=32 (one replica).
      # GLM-5.1 (700B) needs all 32 GPUs to fit for generation.
      # We deliberately set EP = TP here because vLLM async_engine with EP > TP
      # (native vLLM DP) is not supported on NeMo-RL main; see issue #1101 / PR #2517.
      # With EP = TP there is no vLLM DP dimension and async_engine can be used.
      tensor_parallel_size: 32
      pipeline_parallel_size: 1
      expert_parallel_size: 32
      # Raised from the demo's 0.5 to 0.7: with TP=32, EP=32 each rollout
      # GPU holds ~47.5 GiB of bf16 weights (logged in slurm-3020427). At
      # util=0.5 the vLLM budget is ~48 GiB, leaving ~0.5 GiB for KV cache,
      # which is below the minimum block size and triggers
      # `ValueError: No available memory for the cache blocks`. At util=0.7
      # the budget is ~67 GiB, leaving ~20 GiB for KV cache.
      gpu_memory_utilization: 0.7
      max_model_len: \${policy.max_total_sequence_length}
      enforce_eager: true
      use_tqdm: true
      use_deep_gemm: false
      num_last_layers_in_bf16: 0
      num_first_layers_in_bf16: 0
      enable_vllm_metrics_logger: false
      vllm_metrics_logger_interval: 0.5
    vllm_kwargs: {}
    # Non-colocated generation: dedicated rollout nodes separate from training.
    colocated:
      enabled: false
      resources:
        num_nodes: ${ROLLOUT_NNODES}
        gpus_per_node: 4

# Replace the parent's OpenMathInstruct-2 data setup with GSM8K.
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
    num_workers: 8
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
  # Total SLURM nodes.  NeMo-RL's non-colocated logic subtracts the rollout nodes
  # configured under policy.generation.colocated.resources from this total to obtain
  # the training node count, so this must be 32 (not 24).
  num_nodes: ${SLURM_JOB_NUM_NODES}
  gpus_per_node: 4
  master_port_range_low: 25000
  master_port_range_high: 28000
EOF

# -----------------------------------------------------------------------------
# Hand over to the shared driver for model download and execution.
# -----------------------------------------------------------------------------
source "${SCRIPT_DIR}/nemorl-grpo-driver.sh"
