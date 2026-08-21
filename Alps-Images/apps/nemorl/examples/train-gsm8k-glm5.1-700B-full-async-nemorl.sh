#!/bin/bash

#SBATCH --nodes=56
#SBATCH --account=csstaff
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=12:00:00

# Because slurm makes a copy of the script we can not find it through `BASH_SOURCE`
#  instead we relly on that it is in the folder we started.
#SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="${PWD}"

# Specific settings
export NEMORL_IMAGE="/capstor/scratch/cscs/phimuell/.uenv-images/__ML__/nemo_rl_2026_08_13_I.sqsh"

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
# ⚠️ IMAGE AND BRANCH SHIP TOGETHER.  The driver git-fetches this branch at
# job start INTO the image checkout: running the 3.13.14 image with the old
# branch (or vice versa) recreates the mixed-version breakage of 3103672 at
# the python level.  Old image (nemo_rl_2026_08_13_I) ↔ glm51-megatron-
# fsdp-wiring; offline image ↔ glm51-megatron-fsdp-wiring_merged.
export NEMORL_BRANCH="${NEMORL_BRANCH:-glm51-megatron-fsdp-wiring}"

# Node split: 8 nodes for non-colocated vLLM generation (TP=32, one replica),
# the rest for Megatron training with fixed TP=4, PP=3, EP=8, ETP=4.
#
# DEFAULT: 56 nodes = 48 training (192 GPUs, expert_DP=2, dense DP=16),
#   with the CPU-offloaded optimizer (see the OPTIMIZER_CPU_OFFLOAD toggle
#   below — slurm-3031411's measured numbers ruled out the GPU-resident
#   optimizer at this node count).  PRIORITY IS A RUN THAT TRAINS AT ALL —
#   efficiency comes after (see HANDOFF §8).
# Variant: `sbatch --nodes=32` = 24 training (96 GPUs, expert_DP=1) — verl
#   parity, cheapest, but the unsharded ~87 GB expert optimizer state does
#   not fit on GPU and its CPU offload recreates 3029253's host burden;
#   only viable if the verl baseline proves the update phase fits.
#
# Why 32 nodes works (see HANDOFF.md §8): GLM-5.1 is ~94% expert weights, and
# Megatron shards experts over ETP × EP × PP ranks.  verl leaves
# expert_tensor_parallel_size unset, which Megatron defaults to TP
# (parallel_state.py:781: None → tensor_model_parallel_size), giving 4×8×3 =
# 96-way expert sharding on 96 GPUs → ~10.9B params/rank (~22 GB bf16, DDP
# param+grad buffer ~44 GB — fits the 95 GB GH200).  NeMo-RL's parent config
# pins expert_tensor_parallel_size=1 (grpo_math_1B.yaml:191), which is why the
# earlier TP=4/PP=3/EP=8 attempts saw 30–33B params/rank (slurm-3004663: 24-way
# expert sharding) and OOM'd, and why the interim workaround needed EP=64/PP=8
# = 512 GPUs (128 nodes) to reach the same per-rank footprint.  Setting
# expert_tensor_parallel_size: 4 explicitly below restores the verl layout.
#
# Divisibility: expert_tensor_model_pipeline_parallel = ETP*EP*PP = 4*8*3 = 96,
# so the training world must be a multiple of 96 GPUs = 24 nodes:
#   24 training nodes (world  96): expert_DP=1, dense DP=8
#   48 training nodes (world 192): expert_DP=2, dense DP=16  <- default
# (anything in between is invalid — Megatron asserts world % 96 == 0).
#
# GLM-5.1 has 78 layers; PP=3 → 26 layers/stage (no uneven first/last stage
# needed).  512 total experts; EP=8 → 64 experts/rank, each split 4-way by ETP
# (matches megatron-bridge's GLM-5.1 mapping and the verl run).
#
# Host-RAM note (only when OPTIMIZER_CPU_OFFLOAD=true): the CPU-offloaded
# optimizer state (FP32 master + Adam m/v ≈ 12 B/param) is sharded only
# expert_DP ways for the expert part.  At expert_DP=1 that is ~87 GB/rank
# × 4 ranks/node ≈ 345 GB of the 450 GB node RAM — and the
# HybridDeviceOptimizer allocates its pinned CPU clones eagerly at init
# (see HANDOFF §8, slurm-3029253).  At the default expert_DP=2 it halves
# to ~51 GB/rank × 4 ≈ 204 GB — the acceptable-risk regime.
export ROLLOUT_NNODES=8
export TRAINING_NNODES=$(( SLURM_JOB_NUM_NODES - ROLLOUT_NNODES ))

# Name of the generated YAML config consumed by the shared driver.
export YAML_NAME="grpo_gsm8k_glm51_async_nemorl.yaml"

# -----------------------------------------------------------------------------
# Optimizer placement toggle (see HANDOFF.md §8).
#
# "true" (default since slurm-3031411): CPU-offloaded optimizer via
#   HybridDeviceOptimizer + CPUAdam.  Measured numbers from 3031411 killed
#   the GPU-resident path at 56 nodes: buffers 39 GB + masters ~17-21 GB +
#   Adam moments 2x masters (~34-42 GB at the first step) + ~9 GB measured
#   NCCL overhead = 90-100+ GB > 95 GB.  With offload: GPU ~= 39 buffers +
#   ~17 GPU shard-mains + 9 NCCL ~= 65 GB (~30 GB headroom vs the phantom);
#   host ~= ~51 GB/rank x 4 ~= 204 GB of 450 GB — about half the burden that
#   correlated with the init-phantom in slurm-3029253 (expert_DP=2 halves
#   the eager pinned clones).
#
# "false": fully GPU-resident optimizer (FusedAdam, moments lazy at first
#   step, idle host).  Fits only at >= 80 total nodes (72 training,
#   expert_DP=3, ~82 GB) — and even there only ~13 GB headroom against the
#   27-45 GB phantom seen in 3031411.
# -----------------------------------------------------------------------------
export OPTIMIZER_CPU_OFFLOAD="${OPTIMIZER_CPU_OFFLOAD:-true}"
if [[ "${OPTIMIZER_CPU_OFFLOAD}" == "true" ]]; then
    OPTIMIZER_PLACEMENT_YAML="      optimizer_cpu_offload: true
      optimizer_offload_fraction: 1.0
      overlap_cpu_optimizer_d2h_h2d: true"
else
    OPTIMIZER_PLACEMENT_YAML="      optimizer_cpu_offload: false"
fi

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
# Log which network transport NCCL actually selects (Slingshot/CXI via
# aws-ofi-nccl vs. TCP-socket fallback).  Look for "NCCL INFO NET/..." lines:
# "Using network AWS Libfabric" = HSN; "Using network Socket" = slow fallback.
# Added to diagnose the extremely slow TP=32 rollout decode in slurm-3030645.
NCCL_DEBUG = "INFO"
NCCL_DEBUG_SUBSYS = "INIT,NET"
# Phantom-memory hedge (HANDOFF §8, slurm-3031411): 27-45 GB/GPU of memory
# attributed to no process appears between init and the first training
# transition.  Prime suspect: NCCL NVLS multicast buffers, allocated lazily
# per communicator via cuMem — invisible to NVML per-process accounting.
# NVLS only speeds up large all-reduces on the NVLink domain; TP=4
# all-reduces at 2k tokens lose little.
NCCL_NVLS_ENABLE = "0"
# STEP-2 WEDGE FIX (slurm-3129449/3129890/3132942 — deterministic NCCL
# watchdog wedge at step-2 training, memory exonerated by the KL-free run:
# worst node 64 GiB Shmem / 231 GiB avail and it STILL hung).  On Hopper
# with TP>1 + sequence_parallel (non-FSDP), Megatron-LM MANDATES
# CUDA_DEVICE_MAX_CONNECTIONS=1 — its launcher asserts on it — so that
# comm kernels launch in enqueue order and concurrent collectives cannot
# interleave differently across ranks (= deadlock).  NeMo-RL drives
# megatron-core directly and skips that validation; TransformerEngine
# warned about the missing setting on all 384 ranks in every run.  Step 2
# is where the DDP overlap machinery (overlap_grad_reduce +
# overlap_param_gather) first runs fully concurrent.  verl exports this
# for every job unconditionally (verl/trainer/constants_ppo.py).
CUDA_DEVICE_MAX_CONNECTIONS = "1"
# Post-step weight-broadcast staging (slurm-3062482): the packed refit
# broadcast (nemo_rl/utils/packed_tensor.py) targets 2% of GPU memory
# (~1.9 GB) per bucket, overshoots by up to one tensor, and double-buffers
# (x2) — ~3.55 GB allocations against ~2.7 GB free at the post-step peak.
# Shrink the target to 0.5% and use a single buffer; largest single tensor
# (~0.5 GB grouped-MoE layer shard) still fits the pack.
NRL_REFIT_BUFFER_MEMORY_RATIO = "0.005"
NRL_REFIT_NUM_BUFFERS = "1"
# Venvs built at RUNTIME (e.g. AsyncTrajectoryCollector — not in the image's
# prefetch list) land in the writable container overlay, which on diskless
# GH200 nodes is tmpfs = RAM.  uv's default link mode (hardlink) cannot cross
# from the squashfs lower layer (/opt/uv_cache) into the overlay and silently
# falls back to FULL COPIES — potentially 15-30 GB of RAM per node.  Symlink
# mode makes runtime venvs point into the read-only cache instead (same mode
# the image build itself uses for the prefetched venvs).
UV_LINK_MODE = "symlink"
# THE LEAK FIX (named via the 2+2-node Qwen rig, slurm-3125000): every refit
# left ~7.5 GB/rank of pinned host staging in torch's CachingHostAllocator
# ("/dev/zero (deleted)" mappings, Shmem-accounted on GH200) — the staircase
# that killed every run since 3100132 at ~step 2-3.  Fork commit 552d7d131
# flushes the pinned-host cache after each weight broadcast.
NRL_EMPTY_HOST_CACHE_AFTER_REFIT = "${NRL_EMPTY_HOST_CACHE_AFTER_REFIT:-1}"
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
# Offload settings:
#   The verl script sets param_offload/grad_offload/optimizer_offload: True.
#   NeMo-RL has no param_offload/grad_offload equivalent, but those are NOT
#   what lets verl fit on 96 GPUs (offload cannot shrink the in-step working
#   set — params+grads must be on GPU during fwd/bwd anyway).  The fit comes
#   from ETP=4 expert sharding (see the node-split comment above).
#   NOTE: verl's optimizer_offload is verl-level lifecycle code
#   (verl/verl/utils/megatron_utils.py:714,764 — lazy FusedAdam states moved
#   at phase boundaries), NOT Megatron's optimizer_cpu_offload /
#   HybridDeviceOptimizer.  Our optimizer placement is controlled by the
#   OPTIMIZER_CPU_OFFLOAD toggle above (default: GPU-resident, like verl).
#   The reference model is already held as a CPU state dict by NeMo-RL
#   (setup.py:1756).
#
# lr_decay_iters (verify on first run):
#   Set to 468 to match max_num_steps. The verl script notes that
#   lr_decay_steps "must be positive at init; set_total_train_steps is
#   called too late (after init_workers)" — verify whether the same
#   applies to NeMo-RL's megatron path.
#
# Parallelism (verl-style sharding, NOT the NVIDIA demo — the demo's
# EP=64/ETP=1 layout needs 512 training GPUs; ETP=4 reaches the same
# per-rank footprint at 96):
#   Training: TP=4, PP=3, EP=8, ETP=4 fixed; node count sets expert_DP/DP
#             (default 48 training nodes = 192 GPUs → expert_DP=2, DP=16)
#   Rollout:  8 nodes × 4 GPUs = 32 GPUs; TP=32 (one replica)
#
# TP=4 is node-local (4 GPUs/node — no cross-node TP).  Per-rank params
# 9.77B measured (slurm-3029253 and the verl run slurm-3028399 — identical)
# → legacy DDP param+grad buffer ~39 GB, independent of node count.
#
# Legacy DDP (use_megatron_fsdp absent/false).  See HANDOFF.md for the FSDP
# dead end (upstream _get_dp_tp_mesh bug with PP>1) and §8 for why ETP — not
# verl's param/grad offload — is what closes the 96-GPU gap.
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
  # ⚠️ PORT-FIDELITY FIX (2026-08-20): the verl source recipe NEVER used a
  # reference policy.  verl only builds one when actor.use_kl_loss or
  # algorithm.use_kl_in_reward is true — both default false, the recipe sets
  # neither, and its kl_ctrl.kl_coef: 0.001 was dead config on the disabled
  # reward-KL path (verl/trainer/ppo/utils.py need_reference_policy()).  Our
  # earlier translation to 0.001 here ACCIDENTALLY ENABLED the reference
  # machinery: a permanent pinned CPU copy of the reference weights
  # (~95 GiB/node) plus a per-step pinned copy of the current weights for the
  # model<->reference swap — the host-memory staircase that killed every
  # multi-step run (see HANDOFF §8 and CACHING_HOST_ALLOCATOR_FINDINGS.md).
  # 0 = faithful to verl AND skips the whole reference stack (NeMo-RL
  # auto-skips ref logprobs when the penalty is 0, grpo.py:990).
  # Override NRL_REFERENCE_KL_PENALTY to re-enable a KL leash.
  reference_policy_kl_penalty: ${NRL_REFERENCE_KL_PENALTY:-0}
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
  # Divisibility (verl-style sharding: TP=4/PP=3/EP=8 with explicit ETP=4 —
  # the parent grpo_math_1B.yaml:191 pins ETP to 1, which caused the 30-33B
  # params/rank OOMs; see HANDOFF §8):
  #   - GLM-5.1 has 78 transformer layers; PP=3 -> 26 layers/stage (even).
  #   - expert_tensor_model_pipeline_parallel = 4*EP*PP = 4*8*3 = 96;
  #     training world must be a multiple of 96 GPUs (= 24 nodes).
  #   - EP=8 -> 64 experts/rank, each split 4-way by ETP (matches
  #     megatron-bridge GLM-5.1 mapping and the verl run).
  #
  # Legacy DDP (use_megatron_fsdp absent).  Per-rank params are 9.77B
  # measured -> param+grad buffer ~39 GB regardless of node count (model
  # build succeeded on all 96 GPUs in slurm-3029253).  The node count only
  # changes how the optimizer state is sharded (expert_DP) — see the
  # node-split comment at the top.
  megatron_cfg:
    enabled: true
    empty_unused_memory_level: 1
    # TP=4 (node-local), PP=3; DP derived from the training world size.
    # GLM-5.1 has 78 layers; 78 / PP=3 = 26 layers/stage.
    tensor_model_parallel_size: 4
    pipeline_model_parallel_size: 3
    expert_model_parallel_size: 8
    # Shard each expert's FFN 4-way across the TP group, like verl (verl
    # leaves this null and Megatron defaults it to TP; NeMo-RL's parent
    # config would otherwise pin it to 1).  This is what makes 96 training
    # GPUs sufficient: expert weights shard ETP*EP*PP = 96 ways instead of 24.
    expert_tensor_parallel_size: 4
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
    # is unsharded on GPU (~39 GB at TP=4/PP=3/EP=8/ETP=4).
    # use_distributed_optimizer shards optimizer state across dense DP for
    # dense params and expert_DP for expert params (expert_DP=2 at the
    # default 48 training nodes).  Placement is controlled by
    # OPTIMIZER_CPU_OFFLOAD at the top of this script — default is
    # CPU-offloaded since slurm-3031411 measured that the GPU-resident
    # optimizer (masters ~17-21 GB + moments 2x) exceeds 95 GB at the
    # first optimizer.step even at 56 nodes.
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
${OPTIMIZER_PLACEMENT_YAML}

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
      # "ValueError: No available memory for the cache blocks". At util=0.7
      # the budget is ~67 GiB, leaving ~20 GiB for KV cache.
      gpu_memory_utilization: 0.7
      max_model_len: \${policy.max_total_sequence_length}
      enforce_eager: true
      use_tqdm: true
      use_deep_gemm: false
      num_last_layers_in_bf16: 0
      num_first_layers_in_bf16: 0
      # Periodic tokens/s + running/waiting-request logging from the vLLM
      # engine.  Enabled after slurm-3030645, where generation ran blind for
      # minutes with no way to distinguish "slow" from "wedged" in the log.
      enable_vllm_metrics_logger: true
      vllm_metrics_logger_interval: 30.0
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
