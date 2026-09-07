#!/bin/bash

#SBATCH --nodes=128
#SBATCH --account=csstaff
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=5:00:00

# ─────────────────────────────────────────────────────────────────────────────
# DeepSeek-V3 (671B, MoE + MLA) variant of
# train-gsm8k-glm5.1-700B-v1-separate-async-megatron.sh (2026-09-03). Same image, same
# verl V1 separate-async trainer, same patch stack, same 128-node TP=4/PP=3/EP=8 layout
# that was validated for GLM-5.1 -- only the model-specific knobs differ. What changed
# vs the GLM-5.1 recipe (search for "DeepSeek-V3" to find every edit):
#
#   * model           : zai-org/GLM-5.1 (glm_moe_dsa, 78 layers, DSA attention)
#                       -> deepseek-ai/DeepSeek-V3 (deepseek_v3, 61 layers + 1 MTP layer,
#                       MLA attention, 256 routed + 1 shared expert, 8 active, yarn RoPE).
#                       megatron-bridge 0.6.1 ships a native DeepSeekV3Bridge
#                       (megatron/bridge/models/deepseek/deepseek_v3_bridge.py) -- no
#                       runtime bridge patch needed, unlike the Apertus recipes.
#   * FP8 checkpoint  : the HF checkpoint is FP8 block-quantized (config.json
#                       quantization_config fmt=e4m3, weight_block_size 128x128, with
#                       *_scale_inv tensors). The trainer is fine as-is: DeepSeekV3Bridge
#                       dequantizes to bf16 on the fly (maybe_modify_loaded_hf_weight ->
#                       quantization_utils.maybe_dequantize_fp8_blockwise, keyed on the
#                       tensor dtype, NOT on the config). The standalone SGLang rollout is
#                       NOT fine as-is: it reads config.json, and with quantization_config
#                       present it would build FP8 block-scaled weights expecting
#                       *_scale_inv, while load_format=dummy + the trainer->rollout weight
#                       sync push plain bf16 tensors into it. verl's own DeepSeek-V3 example
#                       (examples/grpo_trainer/run_deepseek_v3_671b_megatron.sh) says to
#                       "remove quantization_config from config.json and set
#                       num_nextn_predict_layers=0". We do exactly that, but on the per-node
#                       /tmp config mirror only (see MODEL_LOCAL below) -- the downloaded
#                       checkpoint on Lustre is left untouched.
#   * MTP             : num_nextn_predict_layers=1 in the checkpoint. verl zeroes it when
#                       actor_rollout_ref.model.mtp.enable is False (the default;
#                       verl/workers/config/model.py) so Megatron builds no MTP block and
#                       the layer-61 MTP weights are simply never mapped; the config mirror
#                       also sets it to 0 so SGLang sees the same thing. "MTP and
#                       quantization is disabled during RL training" -- verl docs/perf/dpsk.md.
#   * pipeline split  : 61 layers is prime, so PP=3 cannot split evenly. Uneven PP via
#                       megatron-core's num_layers_in_first/last_pipeline_stage = 20/20
#                       (middle stage gets 21); validated against core_v0.19.0's
#                       TransformerConfig checks (61-20-20 = 21 layers over PP-2 = 1 middle
#                       stage). verl's own DeepSeek-V3 example uses the same knob
#                       (num_layers_in_last_pipeline_stage).
#   * trust_remote_code: True -> False. transformers has had native deepseek_v3 support
#                       since 4.51; the checkpoint's bundled modeling_deepseek.py targets
#                       transformers 4.33 and must not be loaded under this image's 5.x.
#                       auto_map is stripped from the config mirror for the same reason.
#   * MLA overrides   : apply_rope_fusion: False (yarn RoPE + MLA, no fused kernel; the
#                       bridge already sets this, made explicit) and moe_router_dtype: fp32
#                       (sigmoid / noaux_tc routing, per verl's DeepSeek-V3 example).
#   * PR #7421        : DSA-specific (experimental_attention_variant=dsa). Not exercised by
#                       MLA; kept applied so the verl source tree is byte-identical to the
#                       validated GLM-5.1 recipe.
#   * memory          : per-rank params at TP=4/PP=3/EP=8 are ~8.3B (vs GLM-5.1's 9.24B):
#                       routed experts 654B/(ETP 4 x EP 8 x PP 3) + attention 11.4B/(4x3) +
#                       shared/dense 3.7B/(4x3) + embed/head 0.93B/4 on the first/last
#                       stage -- so the GLM-validated 128-node layout has slightly MORE
#                       headroom here. Untouched for the first run (isolate the model
#                       change); 104 nodes / DP=4 is the obvious follow-up reduction.
#
# VALIDATED end-to-end 2026-09-07 (run 3309430): 40/40 GRPO steps, exit 0, ~58 min wall on 128
# nodes, ppo_kl ~0.001, critic/score/mean 0.81 -> ~1.05, every delta_sharded steady sync 25-34 s.
# It took 5 runs to get here -- see the DeepSeek-V3 section of .claude/CLAUDE.md for the three
# bugs found on the way (sglang qkv_a fusion cache, delta-sync rank-0 padded gather OOM, and the
# 480-peer gather-to-one NCCL hang), all fixed by patches embedded below. Checkpoint saving is
# still disabled (save_freq: -1) -- see the trainer.save_freq comment.
#
# Everything below this block is inherited verbatim from the GLM-5.1 recipe's own header.
# ─────────────────────────────────────────────────────────────────────────────
# Copy of train-gsm8k-glm5.1-700B-full-async-megatron.sh switched from the
# experimental fully-async recipe (verl.experimental.fully_async_policy) to the
# V1 trainer (verl/trainer/ppo/v1) in separate-async mode:
#
#   trainer.use_v1=True
#   trainer.v1.trainer_mode=separate_async
#
# What changed relative to the fully-async script:
#   * entrypoint      : verl.experimental.fully_async_policy.fully_async_main
#                       -> verl.trainer.main_ppo
#   * async_training  : the whole block is gone; the equivalent knobs are
#                       trainer.v1.separate_async.* and trainer.v1.sampler.*
#   * top-level rollout: gone; the standalone rollout resources are now declared
#                       in actor_rollout_ref.rollout.{nnodes,n_gpus_per_node}
#   * data batching   : the V1 separate-async trainer asserts
#                       train_batch_size == parameter_sync_step * ppo_mini_batch_size
#                       (fully-async required train_batch_size=0)
#   * TransferQueue   : V1 stores all experience in TransferQueue; it is forced on
#                       by main_ppo, we set it explicitly and size the storage units
#   * old_log_probs   : driven by rollout.calculate_log_probs + algorithm.rollout_correction
#                       (actor.use_rollout_log_probs is unused by V1)
#   * lr schedule     : V1 sets actor.optim.total_training_steps in _init_dataloader,
#                       i.e. before the workers are created, so the lr_decay_steps
#                       workaround the fully-async script needed is dropped.
#
# CAVEAT — the V1 separate-async trainer is *not* a pure disaggregated setup:
# PPOTrainer._setup() always builds hybrid rollout replicas on top of the training
# worker group (trainer world size / rollout world size = 288/32 = 9 replicas here)
# *in addition to* the standalone rollout on ROLLOUT_NNODES. actor_rollout_ref.
# hybrid_engine is not consulted anywhere in the V1 path, so this cannot be turned
# off from the config. Those replicas are put to sleep as soon as the first sample
# batch is drawn (should_switch_to_rollout() is hard-coded to False), but they are
# instantiated at init with gpu_memory_utilization=0.75 on the training GPUs — this
# is the first thing to look at if init OOMs.
# ─────────────────────────────────────────────────────────────────────────────

export VERL_IMAGE="jfrog.svc.cscs.ch/docker-group-csstaff/alps-images/verl-cuda:alps7-dev-621fa40275c4f036" #alps7-dev-a9f9e56471c0574e image with update dependencies #alps7-dev-0f334b540ccc7034 image with megatron

# deepseek-ai/DeepSeek-V3 is the CHAT checkpoint (DeepSeek-V3-Base is the base model); the
# GSM8K system prompt + <answer> format below needs an instruction-following model.
# DeepSeek-V3-0324 (same architecture, later chat post-training) is a drop-in alternative.
export MODEL_NAME="${MODEL_NAME:-DeepSeek-V3}"
export MODEL_REPO="${MODEL_REPO:-deepseek-ai}"

export PROJECT_NAME="async-grpo-gsm8k"
export EXPERIMENT_NAME="${MODEL_NAME}-verl-sglang-megatron-v1-separate-async-${SLURM_JOB_NUM_NODES}n"
export RUN_NAME="${EXPERIMENT_NAME}-${SLURM_JOB_ID}"
export TRAINING_HOME=/capstor/scratch/cscs/${USER}/RL/${MODEL_NAME}
export TRAINING_CONFIG=/tmp
export CHECKPOINT_HOME=${TRAINING_HOME}/checkpoints/${EXPERIMENT_NAME}-run-${SLURM_JOB_ID} #remove "run-${SLURM_JOB_ID}" to enable checkpoint resuming


mkdir -p $TRAINING_HOME
cd $TRAINING_HOME



# Standalone rollout needs exactly 8 nodes for TP=32 (8 nodes × 4 GPUs = 32 GPUs, 1 replica).
# Training gets the remaining 120 nodes (480 GPUs) for TP=4, PP=3, EP=8, DP=5.
# DeepSeek-V3: layout inherited unchanged from the GLM-5.1 recipe. The node-count history
# below is GLM-5.1's; DeepSeek-V3's per-rank footprint is ~10% smaller (see the header), so
# this is a conservative starting point, not a tuned one.
# (80->104->128 nodes, 2026-08-31/09-01: the step-1 fused_adam optimizer-state OOM on trainer
# local-GPU-0. Runs 3241496-3247517 (5x) all die in the first optimizer.step(): [MEMDUMP]
# (run 3246683/3247517) proved it is a genuine hairline miss -- ~38 GiB free on GPU 0 right
# before the step, fused_adam then allocates the full ~38 GiB DP-shard optimizer state (fp32
# master-remainders + exp_avg + exp_avg_sq) and misses by 12-24 MiB, every run. NOT a phantom
# process (nvidia-smi: one WorkerDict/GPU), NOT fragmentation (unconditional empty_cache()
# closed the reserved-vs-alloc gap to 0.15 GiB and it still OOM'd), NOT token-dependent
# (ppo_max_token_len_per_gpu 8192->4096 had zero effect). DP 4->5 shrinks the optimizer shard
# ~38 -> ~30 GiB -- ~7.6 GiB margin, overwhelming for a MiB miss.)
export ROLLOUT_NNODES=8
export TRAINING_NNODES=$(( SLURM_JOB_NUM_NODES - ROLLOUT_NNODES ))

# V1 separate-async batching contract:
#   data.train_batch_size == trainer.v1.separate_async.parameter_sync_step * actor.ppo_mini_batch_size
# parameter_sync_step is the number of actor updates between two weight syncs to the
# standalone rollout (the fully-async script called this trigger_parameter_sync_step).
#
# Both train_batch_size and ppo_mini_batch_size are PROMPT counts, not row counts:
# actor.ppo_mini_batch_size gets multiplied by actor_rollout_ref.rollout.n internally
# (verl/trainer/ppo/v1/trainer_base.py:1652) to get the actual row count fed to the
# DP-parallel actor mini-batch, and that product is what the dp_size sizing constraint
# (trainer_base.py:1441-1447, _get_required_batch_multiple) actually checks. Keep
# ppo_mini_batch_size * rollout.n at least 2x dp_size (DP=3 x EP=8 = 24) — not just
# >= dp_size, and not just checking ppo_mini_batch_size in isolation: at exactly 1x, every
# DP rank gets exactly 1 row per mini-batch, and (historically, before the real fix in the
# "TransferQueue==0.1.6 shape-equality bug" hazard above) a length-1 list made verl's/
# TransferQueue's nested-vs-stacked batch assembly stack a plain Tensor instead of building
# a jagged NestedTensor, crashing engine_workers.py's train_mini_batch (run 3134772:
# AttributeError: 'Tensor' object has no attribute 'offsets'). That bug is fixed upstream
# now (TransferQueue 0.1.7), but the >=2x margin costs nothing and remains cheap insurance.
export ROLLOUT_N=8                   # responses per prompt -- GRPO advantages degenerate at n=1
export PPO_MINI_BATCH_SIZE=10        # prompts; x ROLLOUT_N = 80 rows = 2x dp_size (DP=5 x EP=8 = 40)
export PARAMETER_SYNC_STEP=2
export TRAIN_BATCH_SIZE=$(( PARAMETER_SYNC_STEP * PPO_MINI_BATCH_SIZE ))

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

# ── TransferQueue: mandatory experience store for the V1 trainer ──────────────
transfer_queue:
  enable: True
  backend:
    storage_backend: SimpleStorage
    SimpleStorage:
      # verl recommends >= 2 x number of nodes for load balancing
      num_data_storage_units: $(( SLURM_JOB_NUM_NODES * 2 ))

data:
  train_files: ${TRAINING_HOME}/data/gsm8k/train.parquet
  val_files:   ${TRAINING_HOME}/data/gsm8k/test.parquet
  train_batch_size: ${TRAIN_BATCH_SIZE}   # == parameter_sync_step * ppo_mini_batch_size
  gen_batch_size: 1      # prompts are submitted to the rollout one at a time
  return_raw_chat: True
  max_response_length: 256  # reduced from 1024 — at ~10 tok/s for a ~700B model at TP=32, 1024 tokens takes 83min for 48 samples (GLM-5.1 measurement; DeepSeek-V3 671B expected similar)

actor_rollout_ref:
  model:
    path: ${TRAINING_HOME}/models/${MODEL_NAME}
    # THD (packed-sequence) mode. Required by R3 (align_r3_router_replay_data hard-requires
    # jagged input_ids) and by use_fused_kernels below. DeepSeek-V3 is plain MLA (no DSA), so
    # none of the GLM-5.1 recipe's DSA-vs-THD history applies here; MLA + THD is the standard
    # Megatron path (verl's own DeepSeek-V3 example runs it).
    use_remove_padding: True
    use_shm: false
    # DeepSeek-V3: False (GLM-5.1 needed True for its custom tokenizer). transformers has
    # native deepseek_v3 support; the checkpoint's bundled modeling_deepseek.py /
    # configuration_deepseek.py target transformers 4.33 and would break under this image's
    # 5.x if trust_remote_code let AutoModel pick them via auto_map. The tokenizer is a plain
    # LlamaTokenizerFast (tokenizer.json), no remote code. auto_map is also stripped from the
    # per-node config mirror (see MODEL_LOCAL in the srun body) so there is no ambiguity.
    trust_remote_code: False
    # Fused linear cross-entropy (2026-09-01, added after run 3251006's backward-pass OOM):
    # patches the Megatron forward to compute LM-head logits + log-probs + entropy WITHOUT
    # materializing the [num_tokens, vocab~151K] logits tensor (and its bwd grad/softmax copies)
    # -- several GiB of activation relief in exactly the fwd/bwd phase that OOM'd. verl gates it
    # (verl/workers/engine/megatron/transformer_impl.py:_maybe_enable_fused_kernels) on
    # use_remove_padding (set), not-value-model (actor), mtp.enable=False (confirmed in the
    # 3251006 config dump), and uniform temperature (rollout temp 1.0) -- all satisfied. If any
    # prereq were unmet verl auto-disables with a warning (not fatal). Numerically equivalent.
    use_fused_kernels: True

  actor:
    # 120 training nodes x 4 GPUs = 480 GPUs; TP=4, PP=3, EP=8 -> 4x3x8=96, DP=5 (see
    # TRAINING_NNODES above for the 80->104->128 node history and why DP had to keep growing:
    # fused_adam optimizer-state init needs ~38 GiB on GPU 0 and DP scaling is the only lever
    # that shrinks it -- ~38 GiB at DP=4 -> ~30 GiB at DP=5).
    # DeepSeek-V3: EP=8 -> 256 routed experts / 8 = 32 experts per EP rank (the shared expert
    # is replicated). PP=3 over 61 layers is UNEVEN -- 61 is prime -- so the split is pinned
    # to 20 / 21 / 20 via num_layers_in_first/last_pipeline_stage in override_transformer_config
    # below (layers 0-2 are dense per first_k_dense_replace=3, all on stage 0, which is why
    # the first stage takes the lighter share). Without those two keys megatron-core asserts
    # num_layers % pipeline_model_parallel_size == 0 at TransformerConfig init.
    ppo_mini_batch_size: ${PPO_MINI_BATCH_SIZE}
    ppo_micro_batch_size_per_gpu: 1
    # 16384 -> 12288 (2026-08-29) -> 8192 -> 4096 (2026-08-31): the Megatron fused_adam
    # optimizer-state OOM on trainer local-GPU-0. Runs 3241496 / 3243323 / 3244653 / 3246683 all
    # die in the FIRST optimizer.step(). Run 3246683's [MEMDUMP] proved it is a GENUINE HAIRLINE
    # MISS (~12-24 MiB), not a phantom co-resident process: nvidia-smi shows exactly one
    # WorkerDict per GPU, ~35 GiB free right before the step, ~12.3 GiB unavoidable
    # NCCL/context, and fused_adam then allocates the full ~35 GiB DP-shard optimizer state in
    # large contiguous blocks. NOT fragmentation (reserved-but-unallocated only ~40-65 MiB at
    # OOM); expandable_segments would not help and breaks the shared SGLang rollout (run 3219305).
    # Primary fix: the unconditional empty_cache() in optimizer_step (step1-oom-memdump.patch)
    # returns ~3 GiB of cached-but-unallocated blocks to the driver first. This 8192->4096 cut is
    # paired insurance -- lowers the fwd/bwd reserved high-water carried into the optimizer step.
    ppo_max_token_len_per_gpu: 4096
    use_dynamic_bsz: True
    megatron:
      tensor_model_parallel_size: 4
      pipeline_model_parallel_size: 3
      expert_model_parallel_size: 8
      # param_offload: True -> False (2026-09-02, after run 3262227): the two host-RAM tuning
      # knobs (optimizer_offload_fraction, update_weights_bucket_megabytes) moved the 446/450 GB
      # host-RAM ceiling by ~0 -- the HybridDeviceOptimizer apparently keeps its full state
      # pinned on host regardless of fraction, so that was the wrong lever. Step 1 itself now
      # completes fully clean with 35-44 GB GPU free at [MEMDUMP], so GPU can afford to keep the
      # ~18.5 GB bf16 params resident instead of CPU-bouncing them every step -- removes that
      # chunk from host RAM entirely rather than tuning it.
      param_offload: False
      grad_offload: True
      optimizer_offload: True
      vanilla_mbridge: False  # DeepSeek-V3 model_type=deepseek_v3 (MLA + MoE) uses megatron-bridge's native DeepSeekV3Bridge, which also does the on-the-fly FP8->bf16 dequant of the checkpoint
      # R3 (Rollout Router Replay) — verl's MoE routing-alignment mechanism. verl's own docs
      # (docs/ascend_tutorial/.../transfer_to_npu_guide.md) name DeepSeek-V3.2, GLM-5 and
      # MiMo-V2 as adopters and recommend R3 for any large MoE; DeepSeek-V3 has the same
      # MoE routing surface (flat num_hidden_layers / num_experts_per_tok, sglang captures via
      # the generic TopK layer, verl replays via its TopKRouter patch), so it is kept on,
      # inherited from the GLM-5.1 recipe where it is validated (20+ clean steps). The correct
      # config path, confirmed against v0.9.0 source, is router_replay/mode: R3 right here
      # under actor.megatron — not the top-level actor.router_replay.mode field.
      # DeepSeek-V3: UNVERIFIED on this model. If the first run fails inside R3
      # (router_replay_utils / routed_experts), flip this to disabled and drop
      # rollout.enable_rollout_routing_replay together to get a baseline first.
      router_replay:
        mode: R3
      override_transformer_config:
        recompute_granularity: full
        recompute_method: uniform
        recompute_num_layers: 1
        use_cpu_initialization: True
        moe_grouped_gemm: True
        moe_permute_fusion: True
        # DeepSeek-V3: uneven PP=3 over 61 layers -> 20 / 21 / 20 (see the actor comment
        # above). megatron-core 0.19.0 TransformerConfig: with both set, the remaining
        # 61-20-20 = 21 layers must spread over the PP-2 = 1 middle stage -- exact. Must NOT be
        # combined with account_for_embedding/loss_in_pipeline_split (mcore rejects that).
        num_layers_in_first_pipeline_stage: 20
        num_layers_in_last_pipeline_stage: 20
        # DeepSeek-V3 MLA + yarn RoPE has no fused-rope kernel path; the bridge already sets
        # this False, made explicit here (verl's own DeepSeek-V3 example does the same).
        apply_rope_fusion: False
        # sigmoid / noaux_tc expert-bias routing: keep the router in fp32 (bridge default,
        # explicit per verl's DeepSeek-V3 example).
        moe_router_dtype: fp32
    optim:
      # CPU-streaming (HybridDevice) optimizer -- the step-1 fused_adam OOM fix, 2026-09-01.
      # Runs 3241496-3250425 (6x) all die in the first optimizer.step(): [MEMDUMP] (3246683,
      # 3247517, 3250425) proved it is a genuine hairline miss -- ~40 GiB free on GPU 0 right
      # before the step, and optimizer.step() then allocates a ~40 GiB transient (exp_avg +
      # exp_avg_sq creation, grad->fp32-main copy, distributed-optimizer all-gather buffer,
      # grad-norm + .float() upcast scratch) and misses by ~19 MiB, every run. DP scaling stalled
      # (3247517 DP=4 / 3250425 DP=5 moved "free" only ~2 GiB -- the all-gather buffer and bf16
      # params are a fixed floor that DP does not shrink).
      #
      # override_optimizer_config is forwarded verbatim to Megatron-core's OptimizerConfig
      # (verl/utils/megatron/optimizer.py:210). This block is verl's OWN canonical large-MoE
      # Megatron-async recipe -- see verl/experimental/fully_async_policy/shell/
      # grpo_30b_a3b_base_math_megatron_96_32_mis.sh (Qwen 30B-A3B MoE, 96+32 nodes) and the
      # other 30B/35B MoE megatron-async scripts, which all set exactly these four keys.
      # optimizer_cpu_offload builds Megatron's HybridDeviceOptimizer: the optimizer state lives
      # on CPU and is streamed bucket-by-bucket during the step, so the GPU never materializes
      # the full moments. overlap_cpu_optimizer_d2h_h2d hides the transfer. use_precision_aware_
      # optimizer is required by that path. verl's own megatron.optimizer_offload:True stays on
      # alongside (its offload code already handles the HybridDeviceOptimizer via
      # _move_new_state_to_right_device).
      #
      # main_grads_dtype: bf16 (2026-09-01, added after run 3251006): the CPU-offload optimizer
      # above cleared the step-1 optimizer.step() OOM (6 runs died there), and step 1 then died
      # EARLIER in the backward-pass DP grad reduce-scatter -- the fp32 DDP grad buffer holds
      # grads for all 9.24B local params (~37 GiB) resident before the reduce-scatter shards it.
      # bf16 halves it to ~18.5 GiB (this key also drives the DDP grad-bucket dtype). Safe here:
      # token budget 4096 + ~2 rows/DP-rank = 1 micro-batch, so NO grad accumulation (bf16 error
      # only compounds over many micro-batches); the fp32 master-param update is unchanged.
      # Adam moments left fp32 (on CPU now anyway).
      override_optimizer_config:
        optimizer_cpu_offload: True
        # 1.0 -> 0.7 (2026-09-02, after run 3257320): the CPU-offload optimizer + verl param/grad
        # offload + the delta engine's pinned CPU snapshot tipped trainer node RAM to 446/450 GB
        # (Ray OOM-killed a WorkerDict, hung the post-step delta sync). Keeping ~30% of the ~22 GB
        # optimizer state on GPU (which has 22-45 GB free per [MEMDUMP]) frees ~7 GB host/rank.
        optimizer_offload_fraction: 0.7
        overlap_cpu_optimizer_d2h_h2d: True
        use_precision_aware_optimizer: True
        main_grads_dtype: bf16

  rollout:
    name: sglang
    mode: async
    load_format: dummy
    # Standalone (disaggregated) rollout resources — V1 separate-async reads the
    # rollout pool size from here instead of a top-level rollout: block.
    nnodes: ${ROLLOUT_NNODES}
    n_gpus_per_node: 4
    temperature: 1.0
    n: ${ROLLOUT_N} # responses per prompt — GRPO group size for the relative-advantage baseline
    # 8 rollout nodes × 4 GPUs = 32 GPUs; TP=32 (one replica). DeepSeek-V3 is served in bf16
    # (the FP8 quantization_config is stripped from the config mirror; the weight sync pushes
    # bf16): 671B x 2 B = ~1.34 TB over 32 x 95 GiB x 0.75 = ~2.2 TB — fits with KV headroom.
    # (An FP8 rollout via rollout.quantization: fp8 is a verl option, deliberately not used:
    # it changes the weight-sync path this recipe has validated.)
    tensor_model_parallel_size: 32
    gpu_memory_utilization: 0.75
    free_cache_engine: false  # keep KV cache alive across weight syncs — avoids engine rebuild + CUDA graph re-capture across TP=32 (8-node deadlock)
    calculate_log_probs: True   # required: bypass_mode reads rollout_log_probs as old_log_probs
    log_prob_use_dynamic_bsz: True
    # R3 companion flag (see actor.megatron.router_replay above). Re-enabled 2026-08-28 alongside
    # the megatron-core upgrade — r3-sglang-routed-experts-import-fix.patch (still applied below)
    # was never actually exercised in run 3201189, since the job died in Megatron model init
    # before rollout ever started; this is the first attempt where it should actually run.
    enable_rollout_routing_replay: True
    # delta_sharded (was: nccl) — 2026-08-29. The nccl backend streams the FULL 700B model every
    # sync via megatron-bridge per-tensor gather_from_ep_ranks (~6000 back-to-back collectives),
    # and ~1 run in 2 hangs when one rank silently drops a collective (runs 3141801 / 3207923 /
    # 3219811). delta_sharded exports each rank's LOCAL mcore shard (no cross-rank gather in the
    # export; bridge param-mapping run comm-stubbed) and ships only the changed (position,value)
    # pairs with count-lockstepped sparse gathers — designed by verl specifically to keep the
    # gather sequence identical across ranks. Only the one-time seed sync still uses the fragile
    # full path; every steady sync is on the robust delta path (and far smaller / faster). verl
    # auto-wires the SGLang delta_loader when backend==delta_sharded (async_sglang_server.py:263).
    # See CLAUDE.md's Run log / this script's Configuration audit for the full investigation.
    checkpoint_engine:
      backend: delta_sharded  # separate-async rejects "naive"; delta_sharded extends the nccl engine
      # 2048 -> 512 (2026-09-02, after run 3257320): the per-bucket megatron-bridge HF-conversion
      # buffer during the weight sync -- 4x smaller cuts host RAM on the trainer nodes (446/450 GB
      # OOM in the post-step delta sync).
      update_weights_bucket_megabytes: 512
      engine_kwargs:
        delta_sharded:
          rebuild_group: false
          # DeepSeek-V3 (2026-09-04, run 3289294): per-rank byte budget of one padded gather
          # round in the STEADY delta sync (added by delta-sharded-gather-round-size.patch).
          # Under PP>1 every param is gathered over the 480-rank WORLD group and rank 0 holds
          # world x round buffers, so the default (== update_weights_bucket_megabytes, 512 MB)
          # can need up to 240 GB on rank 0 and OOMed at ~77 GB on the first real-gradient
          # sync. 16 MB -> rank-0 peak ~ (480 + ~96 contributing) x 16 MB = ~9 GB against the
          # ~47 GB free measured at [MEMDUMP]. More rounds (each a 480-way gather of 16 MB/rank,
          # ~0.4 s), same total wire; the 512 MB flush bucket to the rollout is unchanged.
          # 16 -> 64 (2026-09-06, with delta-sharded-p2p-gather.patch): rank 0 now receives only
          # the REAL entries of the <= 32 contributing ranks per round (no world x max_n padding),
          # so a 64 MB per-rank cap costs at most ~2-6 GB on rank 0 while cutting the round count 4x.
          gather_round_megabytes: 64
    engine_kwargs:
      sglang:
        watchdog_timeout: 300
        disable_cuda_graph: true
        max_running_requests: 128  # limit concurrent decode batch; keeps per-step latency and KV usage manageable

  ref:
    log_prob_use_dynamic_bsz: True
    log_prob_max_token_len_per_gpu: 16384
    megatron:
      param_offload: True  # keep ref params on CPU when not computing log probs
      tensor_model_parallel_size: 4
      pipeline_model_parallel_size: 3
      expert_model_parallel_size: 8
      vanilla_mbridge: False  # DeepSeek-V3 model_type=deepseek_v3 (MLA + MoE) uses megatron-bridge's native DeepSeekV3Bridge, which also does the on-the-fly FP8->bf16 dequant of the checkpoint
      # DeepSeek-V3: the ref is never built here (no KL loss / KL-in-reward -> no reference
      # policy), but keep its PP split consistent with the actor so enabling a KL term later
      # does not trip the uneven-61-layers assertion.
      override_transformer_config:
        num_layers_in_first_pipeline_stage: 20
        num_layers_in_last_pipeline_stage: 20
        apply_rope_fusion: False

algorithm:
  adv_estimator: grpo
  kl_ctrl:
    type: adaptive
    kl_coef: 0.001
    target_kl: 0.05
    horizon: 10000
  rollout_correction:
    # Bypass mode: old_log_probs = rollout_log_probs. Decoupled mode (False) would make
    # the trainer save/restore a CPU copy of the 700B model on every mini-batch.
    bypass_mode: True

reward:
  custom_reward_function:
    path: ${TRAINING_CONFIG}/gsm8k_reward.py
    name: compute_reward

trainer:
  use_v1: True
  v1:
    trainer_mode: separate_async
    separate_async:
      # batches pushed to the rollout before the training loop starts
      num_warmup_batches: 1
      # actor updates between two weight syncs to the standalone rollout
      parameter_sync_step: ${PARAMETER_SYNC_STEP}
    sampler:
      # staleness bound, in model versions, for a trajectory to remain usable
      max_off_policy_threshold: 8
      max_off_policy_strategy: drop
  total_epochs: 3
  # Overrides the epoch-based step count (verl/trainer/ppo/v1/trainer_base.py:712-716) so the
  # run stops after a fixed number of steps regardless of dataset size/epochs, and the LR
  # scheduler (fed total_training_steps * parameter_sync_step) decays over the right horizon
  # instead of the full 231-step schedule. Sized against the confirmed ~350-360s/step (run
  # 3149736, GLM-5.1; DeepSeek-V3 step time is unmeasured but the model is ~10% smaller
  # per rank): 40 * 360s = 4h worst case + ~20min setup overhead, comfortably inside the 5h
  # limit. Enough to see whether training is moving at all (KL-from-reference nonzero, reward
  # mean not degenerate) but not enough for a trustworthy reward trend, which needs closer to
  # 50-100 steps on GSM8K's noisy per-step reward signal.
  total_training_steps: 40
  project_name: ${PROJECT_NAME}
  experiment_name: ${RUN_NAME}
  nnodes: ${TRAINING_NNODES}
  n_gpus_per_node: 4
  # DeepSeek-V3: 20 -> -1 (2026-09-04, run 3279810). With the default use_dist_checkpointing:
  # False, 'model' in save_contents means an HF-format export via megatron-bridge, which gathers
  # the full 671B model through rank 0 on HOST RAM: the step-20 save put the rank-0 WorkerDict at
  # 302 GB, the node at 446/450 GB, Ray OOM-killed workers and the job died. Disabled for the
  # shakedown. Real fix for a run that needs checkpoints: actor.megatron.use_dist_checkpointing:
  # True (sharded Megatron save, no gather) -- untested on this recipe.
  save_freq: -1
  test_freq: -1   # disable validation — greedy decode over 1319 samples hangs at 24min (cuEventSynchronize deadlock)
  val_before_train: false
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


_DUMP_LEFT = [6]  # DeepSeek-V3 diagnostic (2026-09-04): print a few raw rollouts per reward process


def _maybe_dump(solution_str: str, ground_truth, model_ans) -> None:
    # Run 3279810 scored 0.0 on every sample with no way to see WHAT the model generated
    # (verl does not log generations). Print the first few raw responses per process so the
    # next log shows whether the rollout is coherent text in the wrong format or garbage.
    if _DUMP_LEFT[0] <= 0:
        return
    _DUMP_LEFT[0] -= 1
    text = solution_str.replace("\n", "\\n")
    if len(text) > 900:
        text = text[:900] + "...[truncated " + str(len(solution_str)) + " chars]"
    print("[REWARD-DUMP] gt=" + repr(str(ground_truth)) + " parsed=" + repr(model_ans)
          + " words=" + str(len(solution_str.split())) + " text=" + text, flush=True)


def compute_reward(
    data_source, solution_str, ground_truth, extra_info=None, **kwargs
) -> float:
    # Truncated response (thinking opened but never closed): return 0, not a large
    # negative, to avoid extreme GRPO advantages that cause gradient spikes.
    if "<think>" in solution_str and "</think>" not in solution_str:
        return 0.0

    model_ans = extract_model_answer(solution_str)
    _maybe_dump(solution_str, ground_truth, model_ans)
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

# Content of example/patches/v1-separate-async-fixes.patch, embedded here
# rather than read via a script-relative path: under sbatch, BASH_SOURCE[0]
# resolves to the spool-staged copy of this script, not its checkout location,
# so a script-relative read silently fails on every node (this is the exact
# failure mode that lost the old sitecustomize.py-based fallback patch in run
# 3129805 -- see Known hazards in CLAUDE.md). Applied via git apply after the
# verl v0.9.0 checkout below, alongside the upstream PR patches -- three real,
# stable fixes converted from runtime sitecustomize.py monkeypatches (used
# while still iterating) once run 3149736 confirmed the whole recipe works
# end-to-end with them as a plain source patch instead.
cat > "${TRAINING_CONFIG}/v1-separate-async-fixes.patch" <<- 'EOF'
# Local fixes for verl v0.9.0's V1 separate-async trainer, discovered debugging
# train-gsm8k-glm5.1-700B-v1-separate-async-megatron.sh (see Known hazards and the
# Run log in CLAUDE.md for the full incident history). Re-diff against a newer verl
# ref if this stops applying.
#
# 1. LLMServerManager._initialize_llm_servers (verl/workers/rollout/llm_server.py):
#    PPOTrainer._setup() always builds hybrid rollout replicas on top of the
#    training worker group (trainer_world_size / rollout_world_size replicas) *in
#    addition to* the standalone rollout -- actor_rollout_ref.hybrid_engine is not
#    consulted anywhere in the V1 path, so this could not be disabled from config.
#    Those replicas are instantiated at gpu_memory_utilization=0.75 on the training
#    GPUs during trainer.init(), before the first on_sample_end() ever runs, and
#    free_cache_engine=false (required for TP=32 SGLang stability) makes their
#    sleep() a no-op -- so they held ~71 GiB per training GPU for the life of the
#    run. trainer.init() then needs ~15 GiB back on those same GPUs to stage
#    Megatron params for the NCCL export to the standalone rollout, and OOMed
#    (runs 3121001, 3125195, 3129805). separate-async never actually needs the
#    hybrid engine: get_llm_client() is overridden in PPOTrainerSeparateAsync to
#    always route through the standalone rollout, so the hybrid replicas existed
#    only to be immediately put to sleep. Fixed by no-oping hybrid-mode calls
#    (worker_group is not None); standalone-mode calls (worker_group is None) are
#    unaffected. Confirmed fixed in run 3134772 (no OOM, training reached step 0).
#
# 2. PPOTrainerSeparateAsync.on_init_end (verl/trainer/ppo/v1/trainer_separate_async.py):
#    drops the self.checkpoint_manager.update_weights(...) call. That manager's
#    backend is forced to "naive" (trainer_base.py), which pushes weights into
#    each worker's colocated hybrid engine directly rather than going through a
#    replica list -- with hybrid replicas disabled by fix 1 above, that colocated
#    engine is never created, so the call would push into nothing.
#    self.standalone_checkpoint_manager.update_weights(...), the actual sync to
#    the standalone rollout, is left untouched. Paired with fix 1; same runs.
#
# 3. list_of_dict_to_tensordict (verl/utils/tensordict_utils.py): decided
#    nested-vs-stacked per field by checking `all(item.shape == val_list[0].shape
#    for item in val_list)` -- trivially true for a length-1 list (this function
#    is called once per rollout output, so len(list_of_dicts) is often 1) and also
#    true whenever several ragged items (e.g. GRPO rollout-group responses)
#    coincidentally share a length, most commonly by all saturating
#    max_response_length. Silently produced a dense Tensor for fields callers
#    assume are nested (input_ids, prompts, responses, position_ids), and any
#    downstream .offsets() call then raised
#    AttributeError: 'Tensor' object has no attribute 'offsets' (runs 3134772,
#    3136766, 3137775). Fixed by delegating non-scalar tensor fields
#    unconditionally to this same file's own nested_tensor_from_tensor_list (used
#    elsewhere in the file for chunking/dispatch, and already correct there) --
#    no more shape-equality guessing. This was a real, independent bug, but ended
#    up never being the sole cause of the offsets crashes in this chain: see
#    CLAUDE.md's run 3144665/3149736 entries for the actual root cause, a
#    same-shaped bug in the separately pip-installed TransferQueue==0.1.6 (fixed
#    upstream in 0.1.7; the training script upgrades the wheel at runtime rather
#    than patching it here, since it isn't part of this checkout). Confirmed fixed
#    end-to-end for the whole recipe in run 3149736 (14/231 training steps
#    completed, sane metrics, zero offsets crashes).
#
# Originally shipped as sitecustomize.py runtime monkeypatches (fast to iterate on
# mid-debugging -- no hand-crafted diff needed while the exact fix was still
# changing), converted to this source patch once run 3149736 confirmed all three
# were correct and stable: same reasoning as the Apertus benchmark's equivalent
# conversion (apertus-benchmarks/patches/sglang-apertus1p5-local-fixes.patch) --
# a runtime monkeypatch is great for fast iteration but not the form a stable fix
# should end up in. A plain source patch is one less moving part (no
# sys.meta_path machinery, no PYTHONPATH staging, no import-timing dependency)
# and the actual behavior is just readable in the file.
diff --git a/verl/trainer/ppo/v1/trainer_separate_async.py b/verl/trainer/ppo/v1/trainer_separate_async.py
index 18a06ee2..4c1815a5 100644
--- a/verl/trainer/ppo/v1/trainer_separate_async.py
+++ b/verl/trainer/ppo/v1/trainer_separate_async.py
@@ -133,7 +133,13 @@ class PPOTrainerSeparateAsync(PPOTrainer):
     def on_init_end(self):
         # update weights after loading checkpoint
         self.standalone_checkpoint_manager.update_weights(self.global_steps)
-        self.checkpoint_manager.update_weights(self.global_steps)
+        # self.checkpoint_manager (the hybrid-replica sync path) is skipped: its
+        # backend is forced to "naive", which pushes weights into each worker's
+        # colocated hybrid engine directly rather than through a replica list --
+        # with LLMServerManager._initialize_llm_servers disabling hybrid replicas
+        # (see llm_server.py), that colocated engine is never created, so this call
+        # would push into nothing. self.standalone_checkpoint_manager.update_weights
+        # above, the actual sync to the standalone rollout, is unaffected.

     def on_train_begin(self):
         if self.config.skip.rollout_tq.enable:
diff --git a/verl/utils/tensordict_utils.py b/verl/utils/tensordict_utils.py
index 91d82b15..810ccbee 100644
--- a/verl/utils/tensordict_utils.py
+++ b/verl/utils/tensordict_utils.py
@@ -930,20 +930,21 @@ def list_of_dict_to_tensordict(list_of_dicts: list[dict[str, Any]]) -> TensorDic
     dict_of_lists = {key: [d[key] for d in list_of_dicts] for key in keys}
     batch_size = len(list_of_dicts)

-    final_data = {
-        key: (
-            torch.stack(val_list)
-            if val_list
-            and all(isinstance(item, torch.Tensor) for item in val_list)
-            and all(item.shape == val_list[0].shape for item in val_list)
-            else (
-                torch.nested.as_nested_tensor(val_list, layout=torch.jagged)
-                if val_list and all(isinstance(item, torch.Tensor) for item in val_list)
-                else NonTensorStack(*val_list)
-            )
-        )
-        for key, val_list in dict_of_lists.items()
-    }
+    def _pack(val_list):
+        if not val_list or not all(isinstance(item, torch.Tensor) for item in val_list):
+            return NonTensorStack(*val_list)
+        # Scalar tensors have no dimension to make ragged along.
+        if all(item.dim() == 0 for item in val_list):
+            return torch.stack(val_list)
+        # Always nested -- never guess dense-vs-ragged from shape equality: that
+        # heuristic is trivially wrong for a length-1 list (every item "matches"
+        # its own shape) and misfires whenever several ragged items coincidentally
+        # share a length (e.g. multiple GRPO rollout responses that all saturate
+        # max_response_length), silently producing a dense Tensor for fields
+        # callers assume are nested and crashing their .offsets() calls downstream.
+        return nested_tensor_from_tensor_list(val_list)
+
+    final_data = {key: _pack(val_list) for key, val_list in dict_of_lists.items()}

     td = TensorDict(final_data, batch_size=[batch_size])

diff --git a/verl/workers/rollout/llm_server.py b/verl/workers/rollout/llm_server.py
index d9beede7..4e2d767e 100644
--- a/verl/workers/rollout/llm_server.py
+++ b/verl/workers/rollout/llm_server.py
@@ -523,6 +523,18 @@ class LLMServerManager:
                 so standalone replicas can avoid Ray named-actor collisions with hybrid
                 replicas (which start at 0) when both coexist (e.g. separate async).
         """
+        # separate-async's standalone rollout makes the hybrid replicas this method
+        # would otherwise build on top of the training worker group pure overhead:
+        # get_llm_client() is overridden elsewhere to always route through the
+        # standalone rollout, so hybrid replicas exist only to be put to sleep. Building
+        # them anyway costs ~gpu_memory_utilization worth of every training GPU during
+        # init, which the trainer's own weight-sync needs back (OOM without this).
+        if self.worker_group is not None:
+            self.rollout_replicas = []
+            self.server_handles = []
+            self.server_addresses = []
+            print("LLMServerManager: hybrid replicas disabled (worker_group is set) — skipping init_hybrid()")
+            return
         if start_rank is None:
             start_rank = self.start_rank
         rollout_world_size = (
EOF
sbcast -f ${TRAINING_CONFIG}/v1-separate-async-fixes.patch ${TRAINING_CONFIG}/v1-separate-async-fixes.patch

# example/patches/r3-sglang-routed-experts-import-fix.patch, embedded here for the same
# reason as v1-separate-async-fixes.patch above (BASH_SOURCE[0] resolves to the sbatch
# spool copy under srun, not this file's real checkout location, so a script-relative
# read silently fails on every node). Applied via git apply after the verl v0.9.0
# checkout below, alongside the other verl-source patches. See CLAUDE.md's Configuration
# audit entry and the Run log entries for run 3199623 / probe job 3199799 for the full
# story: verl v0.9.0 imports R3's rollout-side capture helper from a path this image's
# sglang (0.5.16) no longer has it at -- relocated, not missing.
cat > "${TRAINING_CONFIG}/r3-sglang-routed-experts-import-fix.patch" <<- 'EOF'
# Fixes R3 (Rollout Router Replay)'s rollout-side routed-experts capture for this image's
# actual SGLang version (0.5.16). Discovered debugging
# train-gsm8k-glm5.1-700B-v1-separate-async-megatron.sh -- see CLAUDE.md's Configuration audit
# and the Run log entries for runs 3199623 / 3207151 and probe job 3199799.
#
# Two separate defects in verl v0.9.0's async_sglang_server.py, both fixed here (this patch
# supersedes the earlier import-path-only version):
#
#  1. The `skip_tokenizer_init: True` branch did `captured.numpy()` on
#     meta_info["routed_experts"], assuming a raw torch tensor. Current sglang stores that
#     field as a base64-encoded int32 buffer regardless of skip_tokenizer_init (the encode
#     happens in DetokenizerManager._b64_encode_per_request, which runs even with no
#     tokenizer) -- so the branch raised `AttributeError: 'str' object has no attribute
#     'numpy'` on every rollout call, cluster-wide, in run 3207151. This recipe's TP=32
#     standalone rollout runs with skip_tokenizer_init=True, so it always hit this branch.
#     Fix: drop the dead tensor branch; always decode via
#     extract_routed_experts_from_meta_info + reshape (what the other branch already did).
#
#  2. The surviving branch imported extract_routed_experts_from_meta_info from
#     sglang.srt.layers.moe.routed_experts_capturer, a path that does not exist in sglang
#     0.5.16 -- the module was relocated/renamed to sglang.srt.state_capturer.routed_experts
#     (confirmed by fetching the real sglang v0.5.16 tag: same function name, same
#     single-arg signature). Fix: import from the new path, fall back to the old one so the
#     patch is also correct against an older sglang / usable as an upstream PR.
#
# The reshape target -- [num_tokens, num_hidden_layers, num_experts_per_tok] -- matches
# sglang 0.5.16's own RoutedExpertsCapturer buffer, which is allocated
# num_layers=hf_text_config.num_hidden_layers wide (dense-layer slots stay zero); confirmed
# by reading sglang/srt/state_capturer/routed_experts.py at the v0.5.16 tag. GLM-5.1
# (glm_moe_dsa) exposes num_hidden_layers and num_experts_per_tok as flat top-level config
# fields (same layout as transformers' glm4_moe), so the hasattr guard passes; if a future
# checkpoint nests them, the guard raises a clear AttributeError rather than mis-shaping.
#
# Generated from a real edited git worktree at the verl v0.9.0 tag (not hand-written),
# verified against a fresh checkout: applies cleanly (git apply --check), the patched file
# compiles (python3 -m py_compile), and git apply --reverse --check correctly detects
# "already applied" -- same verification discipline as this script's other patches.
#
# This only fixes the rollout (SGLang) side of R3. The actor-side fused DSA-THD kernel that
# R3 also needs is covered by this script's megatron-core 0.19.0 + megatron-bridge 0.6.1
# runtime upgrade block.
diff --git a/verl/workers/rollout/sglang_rollout/async_sglang_server.py b/verl/workers/rollout/sglang_rollout/async_sglang_server.py
index b3329a8..fe20e55 100644
--- a/verl/workers/rollout/sglang_rollout/async_sglang_server.py
+++ b/verl/workers/rollout/sglang_rollout/async_sglang_server.py
@@ -658,12 +658,24 @@ class SGLangHttpServer:
 
         routed_experts = None
         if self.config.enable_rollout_routing_replay:
-            if self.config.skip_tokenizer_init:
-                # convert to numpy
-                captured = output.get("meta_info", {}).get("routed_experts", None)
-                routed_experts = captured.numpy() if captured is not None else None
-            else:
-                from sglang.srt.layers.moe.routed_experts_capturer import extract_routed_experts_from_meta_info
+            # sglang stores meta_info["routed_experts"] as a base64-encoded int32
+            # buffer (this is true regardless of skip_tokenizer_init -- the encode
+            # happens in DetokenizerManager._b64_encode_per_request, which runs even
+            # with no tokenizer). Decode + reshape to
+            # [num_tokens, num_hidden_layers, num_experts_per_tok]. The former
+            # skip_tokenizer_init branch assumed a raw tensor exposing .numpy(),
+            # which no current sglang provides -- it raised
+            # AttributeError: 'str' object has no attribute 'numpy'.
+            captured = output.get("meta_info", {}).get("routed_experts", None)
+            if captured is not None:
+                try:
+                    from sglang.srt.state_capturer.routed_experts import (
+                        extract_routed_experts_from_meta_info,
+                    )
+                except ImportError:
+                    from sglang.srt.layers.moe.routed_experts_capturer import (
+                        extract_routed_experts_from_meta_info,
+                    )
 
                 hf_config = self.model_config.hf_config
                 if not hasattr(hf_config, "num_hidden_layers") or not hasattr(hf_config, "num_experts_per_tok"):
EOF
sbcast -f ${TRAINING_CONFIG}/r3-sglang-routed-experts-import-fix.patch ${TRAINING_CONFIG}/r3-sglang-routed-experts-import-fix.patch

# example/patches/wsync-debug-progress-log.patch, embedded (same BASH_SOURCE-under-sbatch
# reason as the patches above). Per-tensor progress logging for the trainer->rollout weight
# sync (diagnostic) PLUS the seed-sync lockstep barrier that fixes the megatron-bridge
# collective desync hit in runs 3141801/3207923/3219811/3240762 (see the patch header and
# CLAUDE.md). Applied after the verl checkout, alongside the other verl-source patches.
cat > "${TRAINING_CONFIG}/wsync-debug-progress-log.patch" <<- 'EOF'
# [CSCS, 2026-09-02] Weight-sync per-tensor progress logging + seed AND steady-sync lockstep
# barrier.
#
# (1) DIAGNOSTIC: wraps the two Megatron weight-export generators (get_per_tensor_param = the
#     full seed export; get_per_tensor_param_delta_shard = the delta_sharded steady export) so a
#     collective hang shows, per rank, how far streaming got ("[WSYNC-DBG] rank=N ... tensor#K",
#     every 1000 tensors + on exit). Pairs with TORCH_NCCL_TRACE_BUFFER_SIZE / _DUMP_ON_TIMEOUT
#     (set in the srun env).
#
# (2) FIX: a collective desync during weight sync (CLAUDE.md "Gloo all_gather_object" /
#     gather_from_ep_ranks NCCL ALLGATHER 30-min timeout in stream_weights_megatron_to_hf, hit on
#     the SEED path in runs 3141801/3207923/3219811/3240762 -- ~1 run in 2, the single biggest
#     blocker to a full run for a long stretch). The seed sync (_send_full_seed) streams the full
#     get_per_tensor_param() HF export, running a chain of PP/EP/TP assembly collectives per HF
#     tensor; _send_full_seed drives that generator ASYMMETRICALLY -- rank 0 buckets + broadcasts
#     each flush to the rollout CE group between pulls, every non-master rank just discards its
#     tensor and pulls the next -- so non-master ranks race ahead in the per-tensor collective
#     chain until a later cross-group collective deadlocks. Fix: for the seed export,
#     torch.distributed.barrier() on the trainer WORLD group every VERL_WSYNC_SEED_BARRIER_EVERY
#     (default 1) HF tensors, AFTER the consumer processed each -- no rank can start tensor K+1's
#     assembly collectives until every rank finished K, so drift is bounded to
#     VERL_WSYNC_SEED_BARRIER_EVERY. Validated across 10+ runs with zero recurrence.
#
#     Run 3263683 (the first run to ever reach a real, full-scale delta_sharded STEADY sync --
#     every prior run died earlier, on the seed hang or a step-1 GPU/host-RAM OOM) hit the SAME
#     desync class there: an NCCL ALLGATHER hang on the steady sync's very first flush. The
#     steady path's _GatherQueue was assumed "count-lockstepped by design" (CLAUDE.md run
#     3219811's investigation) and deliberately left unbarriered -- that assumption is now
#     disproven by direct evidence. Fix: extend the identical barrier to the "delta-steady" tag
#     too. Steady syncs are far smaller than the seed (only changed values), so per-item
#     barriering there is cheap.
#
# Generated from a real edited git worktree at the verl v0.9.0 tag; verified against a fresh
# checkout: git apply --check clean, py_compile clean, git apply --reverse --check detects
# "already applied".
diff --git a/verl/workers/engine/megatron/transformer_impl.py b/verl/workers/engine/megatron/transformer_impl.py
index e8a6c56..259cdd7 100644
--- a/verl/workers/engine/megatron/transformer_impl.py
+++ b/verl/workers/engine/megatron/transformer_impl.py
@@ -87,6 +87,53 @@ logger = logging.getLogger(__file__)
 logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))
 
 
+
+def _wsync_progress_log(gen, tag):
+    """[weight-sync debug + seed/steady-sync lockstep fix, CSCS] Wrap a per-tensor weight-export
+    generator.
+
+    (1) Diagnostic: on a collective hang during weight sync, print per rank how far the
+        streaming got (which tensor / how long). Pairs with TORCH_NCCL_TRACE_BUFFER_SIZE.
+    (2) Fix for megatron-bridge / delta-engine collective desyncs during weight sync (CLAUDE.md:
+        gather_from_ep_ranks / broadcast_obj_from_pp_rank timeout on the seed path, runs
+        3141801 / 3207923 / 3219811 / 3240762; and an NCCL ALLGATHER hang on the delta-STEADY
+        path's very first flush, run 3263683 -- the first run to ever reach that path at real
+        scale). For BOTH the "seed/full" and "delta-steady" exports, torch.distributed.barrier()
+        on the trainer WORLD group every VERL_WSYNC_SEED_BARRIER_EVERY (default 1) items, AFTER
+        the consumer has processed each -- no rank can start the next item's assembly/gather
+        collectives until every rank finished the current one, so drift is bounded to
+        VERL_WSYNC_SEED_BARRIER_EVERY. The seed export drives its generator asymmetrically (rank
+        0 buckets + broadcasts each flush to the rollout CE group between pulls, non-master ranks
+        discard and pull the next) -- the delta-steady _GatherQueue path was assumed
+        "count-lockstepped by design" and left unbarriered, but run 3263683 showed that
+        assumption does not hold at real scale. Steady syncs are far smaller than the seed
+        (only changed values), so per-item barriering there is cheap."""
+    import time as _t
+
+    _rank = os.environ.get("RANK") or os.environ.get("SLURM_PROCID") or "?"
+    _barrier_tag = tag in ("seed/full", "delta-steady")
+    try:
+        _every = max(1, int(os.environ.get("VERL_WSYNC_SEED_BARRIER_EVERY", "1")))
+    except ValueError:
+        _every = 1
+    _t0 = _t.time()
+    _n = 0
+    try:
+        for _item in gen:
+            _n += 1
+            if _n % 1000 == 1:
+                try:
+                    _name = getattr(_item, "name", None) or (_item[0] if isinstance(_item, (tuple, list)) else "?")
+                except Exception:
+                    _name = "?"
+                print(f"[WSYNC-DBG] rank={_rank} {tag} tensor#{_n} name={_name} t+{_t.time() - _t0:.0f}s", flush=True)
+            yield _item
+            if _barrier_tag and (_n % _every == 0) and torch.distributed.is_initialized():
+                torch.distributed.barrier()
+    finally:
+        print(f"[WSYNC-DBG] rank={_rank} {tag} generator exited after {_n} tensors, {_t.time() - _t0:.0f}s", flush=True)
+
+
 def _resolve_fused_temperature(temperature: float | torch.Tensor) -> float:
     """Return the scalar temperature required by fused linear cross entropy."""
     values = torch.as_tensor(temperature).detach().flatten()
@@ -1042,7 +1089,7 @@ class MegatronEngine(BaseEngine):
 
             per_tensor_param = export_qat_weights(per_tensor_param, self.module, self._qat_config.mode, self.bridge)
 
-        return per_tensor_param, peft_config
+        return _wsync_progress_log(per_tensor_param, "seed/full"), peft_config
 
     def _mcore_export_index(self):
         """Build (once) the per-parameter delta export index: geometry specs and
@@ -1104,7 +1151,9 @@ class MegatronEngine(BaseEngine):
 
         self._delta_shard_snap = getattr(self, "_delta_shard_snap", {})
         gen, _ = self.get_per_tensor_param_shard()
-        return hf_delta_export(gen, self._delta_shard_snap, self._hf_delta_entry), None
+        return _wsync_progress_log(
+            hf_delta_export(gen, self._delta_shard_snap, self._hf_delta_entry), "delta-steady"
+        ), None
 
     def disable_adapter(self) -> ContextManager:
         return self.peft_cls.disable_adapter(self.module)
EOF
sbcast -f ${TRAINING_CONFIG}/wsync-debug-progress-log.patch ${TRAINING_CONFIG}/wsync-debug-progress-log.patch

# example/patches/delta-sharded-localserializedtensor-import-fix.patch, embedded (same reason).
# verl v0.9.0's delta_sharded rollout path imports LocalSerializedTensor from a sglang path that
# moved in sglang 0.5.16; run 3223205 hit this on the first delta flush. Applied after the verl
# checkout alongside the other verl-source patches.
cat > "${TRAINING_CONFIG}/delta-sharded-localserializedtensor-import-fix.patch" <<- 'EOF'
# [CSCS, 2026-08-30] verl v0.9.0's delta_sharded weight-sync path
# (_update_weights_delta_flush in sglang_rollout.py) hardcodes
#   from sglang.srt.model_executor.model_runner import LocalSerializedTensor
# but this image's sglang (0.5.16, the GLM-5.1 DSA build) moved that class to
#   sglang.srt.model_executor.model_runner_components.weight_updater
# -- confirmed by reading sglang v0.5.16's own weight_sync/utils.py, which imports it from the
# new path (line 10-12), and the class definition at model_runner_components/weight_updater.py:370.
# Same class of verl-vs-sglang import drift as r3-sglang-routed-experts-import-fix.patch. Found by
# run 3223205: delta_sharded's trainer-side export worked cleanly (no gather_from_ep_ranks hang!)
# but the first delta flush died on the rollout side with
#   ImportError: cannot import name 'LocalSerializedTensor' from 'sglang.srt.model_executor.model_runner'
#
# Fix: try the new path, fall back to the old (so it also works against an older sglang / as an
# upstream PR). Only the import moves; the LocalSerializedTensor(values=[...]) construction a few
# lines below is unchanged. The other three sglang imports in the same function
# (UpdateWeightsFromTensorReqInput, MultiprocessingSerializer, monkey_patch_torch_reductions) were
# checked against sglang v0.5.16 and are still at their hardcoded paths.
#
# Generated from a real edited git worktree at the verl v0.9.0 tag; verified against a fresh
# checkout: git apply --check clean, py_compile clean, git apply --reverse --check detects
# "already applied".
diff --git a/verl/workers/rollout/sglang_rollout/sglang_rollout.py b/verl/workers/rollout/sglang_rollout/sglang_rollout.py
index 07e30d0..bf05f07 100644
--- a/verl/workers/rollout/sglang_rollout/sglang_rollout.py
+++ b/verl/workers/rollout/sglang_rollout/sglang_rollout.py
@@ -401,7 +401,12 @@ class ServerAdapter(BaseRollout):
         """
         import torch.distributed as dist
         from sglang.srt.managers.io_struct import UpdateWeightsFromTensorReqInput
-        from sglang.srt.model_executor.model_runner import LocalSerializedTensor
+
+        try:
+            # sglang >= ~0.5.16 (the version this image ships): LocalSerializedTensor moved here
+            from sglang.srt.model_executor.model_runner_components.weight_updater import LocalSerializedTensor
+        except ImportError:
+            from sglang.srt.model_executor.model_runner import LocalSerializedTensor
         from sglang.srt.utils import MultiprocessingSerializer
         from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
 
EOF
sbcast -f ${TRAINING_CONFIG}/delta-sharded-localserializedtensor-import-fix.patch ${TRAINING_CONFIG}/delta-sharded-localserializedtensor-import-fix.patch

# example/patches/step1-oom-memdump.patch, embedded (same reason). DIAGNOSTIC ONLY: dumps GPU
# memory accounting + full nvidia-smi right before the first optimizer.step() to identify the
# ~19 GiB of non-trainer memory on trainer GPU-0 that OOMs fused_adam at step 1 (runs 3241496 /
# 3243323 / 3244653). Remove once identified.
cat > "${TRAINING_CONFIG}/step1-oom-memdump.patch" <<- 'EOF'
# [CSCS diagnostic, 2026-09-01/02] step-1 GPU memory instrumentation + the empty_cache() fix.
#
# _step1_oom_memdump() fires ONCE per rank, right before the first optimizer.step(): prints this
# process's torch CUDA accounting (every rank) plus a full nvidia-smi -- all GPUs, all compute
# processes, per-process used_memory (local rank 0 only, i.e. the GPU that used to OOM). The
# [MEMDUMP] lines land in the main slurm log. This diagnostic block (run 3246683) proved the
# step-1 GPU OOM (runs 3241496-3246683) was a genuine hairline miss, not a phantom co-resident
# process -- nvidia-smi always showed exactly one WorkerDict per GPU.
#
# Also makes optimizer_step()'s empty_cache() call UNCONDITIONAL (was gated on the unused
# _distillation_use_topk_active path): returns PyTorch's reserved-but-unallocated cache to the
# driver right before the optimizer's large contiguous allocations, closing the hairline miss.
#
# MUST be applied AFTER wsync-debug-progress-log.patch -- both touch transformer_impl.py and this
# diff's context lines assume _wsync_progress_log is already present.
#
# DIAGNOSTIC (memdump) kept indefinitely for now -- cheap, and useful signal on any future
# step-1 memory question. Generated from a real edited git worktree at the verl v0.9.0 tag
# (+ wsync-debug-progress-log.patch applied); git apply --check clean, py_compile clean,
# git apply --reverse --check detects "already applied".
diff --git a/verl/workers/engine/megatron/transformer_impl.py b/verl/workers/engine/megatron/transformer_impl.py
index 259cdd7..fa9d585 100644
--- a/verl/workers/engine/megatron/transformer_impl.py
+++ b/verl/workers/engine/megatron/transformer_impl.py
@@ -134,6 +134,57 @@ def _wsync_progress_log(gen, tag):
         print(f"[WSYNC-DBG] rank={_rank} {tag} generator exited after {_n} tensors, {_t.time() - _t0:.0f}s", flush=True)
 
 
+
+def _step1_oom_memdump(engine):
+    """[CSCS step-1 OOM diagnostic] Fire once, on this rank's FIRST optimizer.step(), right
+    before it runs. Dumps this process's torch CUDA accounting + a full nvidia-smi (all GPUs,
+    all compute processes, per-process used memory) so GPU memory pressure at the optimizer step
+    can be attributed to a concrete pid/process. Every rank dumps its own torch numbers; only
+    local rank 0 (the GPU that OOMs) also shells out to nvidia-smi. Remove once no longer needed
+    for diagnosis."""
+    if getattr(engine, "_step1_oom_memdump_done", False):
+        return
+    engine._step1_oom_memdump_done = True
+    import subprocess
+
+    _rank = os.environ.get("RANK") or os.environ.get("SLURM_PROCID") or "?"
+    _lrank = os.environ.get("LOCAL_RANK", "0")
+    _cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
+    try:
+        _dev = torch.cuda.current_device()
+        _free, _total = torch.cuda.mem_get_info(_dev)
+        _alloc = torch.cuda.memory_allocated(_dev)
+        _resv = torch.cuda.memory_reserved(_dev)
+        print(
+            f"[MEMDUMP] rank={_rank} lrank={_lrank} pid={os.getpid()} CUDA_VISIBLE_DEVICES={_cvd} "
+            f"cur_dev={_dev} free={_free / 2**30:.2f}G total={_total / 2**30:.2f}G "
+            f"gpu_used={(_total - _free) / 2**30:.2f}G torch_alloc={_alloc / 2**30:.2f}G "
+            f"torch_reserved={_resv / 2**30:.2f}G "
+            f"non_torch_this_proc={((_total - _free) - _resv) / 2**30:.2f}G(incl other procs)",
+            flush=True,
+        )
+    except Exception as _e:
+        print(f"[MEMDUMP] rank={_rank} torch mem read failed: {_e}", flush=True)
+
+    if str(_lrank) != "0":
+        return
+    for _args, _tag in (
+        (["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
+          "--format=csv,noheader"], "compute-apps"),
+        (["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total",
+          "--format=csv,noheader"], "gpu-mem"),
+    ):
+        try:
+            _out = subprocess.run(_args, capture_output=True, text=True, timeout=25).stdout.strip()
+            print(f"[MEMDUMP] rank={_rank} nvidia-smi {_tag}:\n{_out}", flush=True)
+        except Exception as _e:
+            print(f"[MEMDUMP] rank={_rank} nvidia-smi {_tag} failed: {_e}", flush=True)
+    try:
+        print(f"[MEMDUMP] rank={_rank} torch.cuda.memory_summary():\n{torch.cuda.memory_summary()}", flush=True)
+    except Exception as _e:
+        print(f"[MEMDUMP] rank={_rank} memory_summary failed: {_e}", flush=True)
+
+
 def _resolve_fused_temperature(temperature: float | torch.Tensor) -> float:
     """Return the scalar temperature required by fused linear cross entropy."""
     values = torch.as_tensor(temperature).detach().flatten()
@@ -733,8 +784,13 @@ class MegatronEngine(BaseEngine):
         """
         # forward_kl_topk leaves large fp32 vocab tensors until backward ends;
         # free cached blocks before grad-norm all_reduce to reduce OOM on tight VRAM.
-        if getattr(self, "_distillation_use_topk_active", False):
-            get_torch_device().empty_cache()
+        # [CSCS] made unconditional (was gated on _distillation_use_topk_active): the
+        # Megatron distributed-optimizer state init allocates large contiguous blocks on GPU 0
+        # during the FIRST optimizer.step() and used to OOM a few MiB short against PyTorch's
+        # reserved-but-unallocated cache (runs 3241496-3246683). Returning that cache to the
+        # driver first hands the optimizer a clean arena.
+        get_torch_device().empty_cache()
+        _step1_oom_memdump(self)
         update_successful, grad_norm, num_zeros_in_grad = self.optimizer.step()
 
         if update_successful:
EOF
sbcast -f ${TRAINING_CONFIG}/step1-oom-memdump.patch ${TRAINING_CONFIG}/step1-oom-memdump.patch

# DeepSeek-V3 (2026-09-04, after run 3289294): example/patches/delta-sharded-gather-round-size.patch,
# embedded (same BASH_SOURCE-under-sbatch reason). Adds the gather_round_megabytes engine kwarg
# that rollout.checkpoint_engine.engine_kwargs.delta_sharded sets below; without it the first
# steady delta sync with real gradients OOMs rank 0 (480-rank WORLD gather x 512 MB rounds).
# Applied via the same git apply loop as the other verl-source patches. See the patch header.
cat > "${TRAINING_CONFIG}/delta-sharded-gather-round-size.patch" <<- 'EOF'
# [CSCS, 2026-09-04] delta_sharded steady sync: decouple the per-round GATHER byte budget from
# the rollout FLUSH bucket. Found debugging train-gsm8k-deepseek-v3-671B-v1-separate-async-
# megatron.sh run 3289294: the first steady sync with REAL gradients (every prior delta_sharded
# run in this repo had ~zero weight change, so this path was never exercised at scale) OOMed
# the wire-master rank 0 GPU:
#   sparse_gather.py:134 gather_slot_entries_to_rank0
#     idx_list = [torch.zeros(max_n, ...) for _ in range(world)] if rank == 0 else None
#   torch.OutOfMemoryError ... 80.55 GiB allocated by PyTorch (was ~35 GiB before the sync)
# Mechanism: under PP>1 delta_export.py merges EVERY param over the WORLD group (480 ranks
# here) so the wire master needs no relay; rank 0 then allocates world x max_n padded idx/val
# lists per round, and max_round_bytes (the per-rank cap on max_n) was hard-wired to
# self.bucket_size = update_weights_bucket_megabytes (512 MB) -- up to 480 x 512 MB = 240 GB
# on rank 0 for one round, even though only the owner-stage / non-replica ranks contribute
# anything but zeros. The observed round (max_n = 26.7M elems, ~160 MB/rank) needed ~77 GB.
#
# Fix: new engine kwarg `gather_round_megabytes` (engine_kwargs.delta_sharded.
# gather_round_megabytes in the rollout config) sizing ONLY the gather round; None keeps the
# old coupling (behavior-preserving default). Rank-0 peak per round is then
# ~ (world + contributing) x gather_round_megabytes, independent of the flush bucket shipped to
# the rollout, which can stay large (fewer update_weights_from_tensor requests). Upstreamable
# as-is; a fuller fix would gather only from contributing ranks.
#
# Generated from a real edited git worktree at the verl v0.9.0 tag (not hand-written);
# verified: git apply --check clean, py_compile clean, git apply --reverse --check detects
# "already applied". Touches only delta_checkpoint_engine.py (no overlap with the other
# patches in this recipe).
diff --git a/verl/checkpoint_engine/delta_checkpoint_engine.py b/verl/checkpoint_engine/delta_checkpoint_engine.py
index f15eb357..55663e44 100644
--- a/verl/checkpoint_engine/delta_checkpoint_engine.py
+++ b/verl/checkpoint_engine/delta_checkpoint_engine.py
@@ -396,9 +396,26 @@ class DeltaShardedCheckpointEngine(NCCLCheckpointEngine):
         logger.info("delta recv v=%s flushes=%d (yielded to server adapter)", global_steps, applied)
 
     def __init__(
-        self, *args, encoding: str = "indices", batch_gather: int = 32, verify_every: int = 0, **kwargs
+        self,
+        *args,
+        encoding: str = "indices",
+        batch_gather: int = 32,
+        verify_every: int = 0,
+        gather_round_megabytes: int | None = None,
+        **kwargs,
     ) -> None:
         super().__init__(*args, **kwargs)
+        # Per-rank byte budget of ONE padded gather round in the steady sync
+        # (sparse_gather.gather_slot_entries_to_rank0's max_round_bytes). Rank 0
+        # allocates world x budget for the padded idx/val gather lists, and under
+        # PP>1 every param merges over the WORLD group (delta_export.py), so with
+        # the default (== bucket_size, the flush size shipped to the rollout) a
+        # 480-rank trainer can need 480 x 512 MB = 240 GB on rank 0 for a single
+        # round. Decoupled so the gather round can be sized to rank 0's headroom
+        # while the rollout flush bucket stays large. None keeps the old coupling.
+        self.gather_round_bytes = (
+            int(gather_round_megabytes) << 20 if gather_round_megabytes else None
+        )
         assert encoding == "indices", f"delta_sharded ships only the 'indices' position encoding; got {encoding!r}"
         self.encoding = encoding
         # every K-th steady sync appends a full state-verification sweep
@@ -626,7 +643,7 @@ class DeltaShardedCheckpointEngine(NCCLCheckpointEngine):
             wire_bytes += int(aidx.numel()) * (4 + aval.element_size())
             _bucket_sliced(bkt, name, dtype_str, full_shape, aidx, aval)
 
-        gq = _GatherQueue(batch_k, self.bucket_size, is_r0, _bucket_slot_delta)
+        gq = _GatherQueue(batch_k, self.gather_round_bytes or self.bucket_size, is_r0, _bucket_slot_delta)
 
         # ``weights`` is the BACKEND's HF delta stream (hf_delta_export): entries
         # already carry final HF coordinates -- naming, conversion, diff and
EOF
sbcast -f ${TRAINING_CONFIG}/delta-sharded-gather-round-size.patch ${TRAINING_CONFIG}/delta-sharded-gather-round-size.patch

# DeepSeek-V3 (2026-09-06, after run 3293605): example/patches/delta-sharded-p2p-gather.patch,
# embedded (same reason). THE FIX for the steady delta-sync hang: the padded 480-way
# dist.gather-to-rank-0 in sparse_gather.py (rank 0 stuck inside NCCL's launch, per the
# [WSYNC-STALL] dump) becomes targeted P2P from the <= 32 ranks that hold entries. Bit-identical
# output (gloo test, 6 procs). Touches only sparse_gather.py. See the patch header.
cat > "${TRAINING_CONFIG}/delta-sharded-p2p-gather.patch" <<- 'EOF'
# [CSCS, 2026-09-06] delta_sharded steady sync: replace the padded group-wide dist.gather pair
# in sparse_gather.gather_slot_entries_to_rank0 with targeted point-to-point transfers from the
# ranks that actually hold entries.
#
# Found on train-gsm8k-deepseek-v3-671B-v1-separate-async-megatron.sh. Runs 3289420 and 3293605
# (and GLM-5.1 run 3263683): the first steady delta sync carrying real gradients hangs for the
# full 30-min NCCL timeout. The rank-0 stall dumper (delta-sharded-steady-stall-diag.patch, run
# 3293605) pinned it: rank 0 (the wire master) is blocked INSIDE torch.distributed.gather at
# sparse_gather.py:137 (the value gather of a sub-round; the index gather right before it
# completed) -- the call never returns and the work is never enqueued, i.e. NCCL's host-side
# launch of a 480-peer gather-to-one stalls. Under PP>1 delta_export.py merges every param over
# the WORLD group, so each round was a 479->1 P2P fan-in into one GPU over Slingshot with
# world x max_n padded receive lists (the OOM of run 3289294, fixed by the gather-round patch,
# was the same design). Rounds carrying ~1700 changed elements (run 3279810, zero-gradient
# training) always worked; real payloads never did.
#
# Fix: the counts matrix is already all-gathered, so every rank knows which ranks contribute to
# this round. Contributing ranks (<= a stage's tp x ep group, typically 4 or 32) send idx then
# val to rank 0 in one batch_isend_irecv; rank 0 posts matching irecvs of exactly their sizes.
# No padding, no world-wide fan-in, rank-0 memory = the real gathered bytes. Output is
# bit-identical to the padded version (per slot, pieces concatenated in rank order) --
# verified locally on gloo/CPU with 6 processes, 6 random trials x {no cuts, sub-round cuts},
# including an all-empty round and ranks with zero entries. The all_gather of counts, the
# sub-round cut logic and gather_round_megabytes (now a cap on the real per-round data) are
# unchanged. Upstreamable.
#
# Generated from a real git worktree at verl v0.9.0 (this file is touched by no other patch);
# git apply --check / py_compile / --reverse --check verified.
diff --git a/verl/checkpoint_engine/delta_sync/sparse_gather.py b/verl/checkpoint_engine/delta_sync/sparse_gather.py
index b3ce23f3..2198548d 100644
--- a/verl/checkpoint_engine/delta_sync/sparse_gather.py
+++ b/verl/checkpoint_engine/delta_sync/sparse_gather.py
@@ -125,28 +125,56 @@ def gather_slot_entries_to_rank0(
         empty_v = torch.empty(0, dtype=val_concat.dtype, device=dev)
         return [(empty_i, empty_v) for _ in range(k)]
 
-    idx_pad = torch.zeros(max_n, dtype=idx_concat.dtype, device=dev)
-    val_pad = torch.zeros(max_n, dtype=val_concat.dtype, device=dev)
-    n = int(idx_concat.numel())
-    idx_pad[:n] = idx_concat
-    val_pad[:n] = val_concat
-
-    idx_list = [torch.zeros(max_n, dtype=idx_pad.dtype, device=dev) for _ in range(world)] if rank == 0 else None
-    val_list = [torch.zeros(max_n, dtype=val_concat.dtype, device=dev) for _ in range(world)] if rank == 0 else None
-    dist.gather(idx_pad, idx_list, dst=dst, group=group)
-    dist.gather(val_pad, val_list, dst=dst, group=group)
+    # [CSCS, 2026-09-06] Targeted point-to-point transfer instead of two padded, group-wide
+    # dist.gather calls. The counts matrix above already tells every rank which ranks hold
+    # entries for this round, so only THOSE ranks send (idx then val, one batched P2P group)
+    # and rank 0 receives exactly their sizes -- no world x max_n padded lists, and the
+    # fan-in is the number of contributing ranks (<= a stage's tp x ep group under PP>1)
+    # instead of the whole group. Under PP>1 the delta export merges every param over the
+    # 480-rank WORLD group; the padded gather-to-one from 479 peers hung inside NCCL's
+    # host-side launch on Slingshot (rank 0 blocked in dist.gather before the work was even
+    # enqueued -- DeepSeek-V3 runs 3289420 / 3293605, GLM-5.1 run 3263683). Output is
+    # bit-identical to the padded version: per slot, the pieces are concatenated in rank order.
+    contributors = [r for r in range(world) if totals[r] > 0]
     if rank != 0:
+        if totals[rank] > 0:
+            ops = [
+                dist.P2POp(dist.isend, idx_concat.contiguous(), dst, group=group),
+                dist.P2POp(dist.isend, val_concat.contiguous(), dst, group=group),
+            ]
+            for w in dist.batch_isend_irecv(ops):
+                w.wait()
         return None
 
+    recv_idx: dict[int, torch.Tensor] = {}
+    recv_val: dict[int, torch.Tensor] = {}
+    ops = []
+    for r in contributors:
+        if r == 0:
+            recv_idx[0] = idx_concat
+            recv_val[0] = val_concat
+            continue
+        src = dist.get_global_rank(group, r) if group is not None else r
+        recv_idx[r] = torch.empty(totals[r], dtype=idx_concat.dtype, device=dev)
+        recv_val[r] = torch.empty(totals[r], dtype=val_concat.dtype, device=dev)
+        ops.append(dist.P2POp(dist.irecv, recv_idx[r], src, group=group))
+        ops.append(dist.P2POp(dist.irecv, recv_val[r], src, group=group))
+    if ops:
+        for w in dist.batch_isend_irecv(ops):
+            w.wait()
+
     # per-rank cumulative offsets into each blob, sliced per param then stitched across ranks
-    offs = [[0] * (k + 1) for _ in range(world)]
-    for r in range(world):
-        for i in range(k):
-            offs[r][i + 1] = offs[r][i] + counts_cpu[r][i]
+    offs = {r: 0 for r in contributors}
     out = []
     for i in range(k):
-        idx_pieces = [idx_list[r][offs[r][i] : offs[r][i + 1]] for r in range(world) if counts_cpu[r][i]]
-        val_pieces = [val_list[r][offs[r][i] : offs[r][i + 1]] for r in range(world) if counts_cpu[r][i]]
+        idx_pieces = []
+        val_pieces = []
+        for r in contributors:
+            c = counts_cpu[r][i]
+            if c:
+                idx_pieces.append(recv_idx[r][offs[r] : offs[r] + c])
+                val_pieces.append(recv_val[r][offs[r] : offs[r] + c])
+                offs[r] += c
         if idx_pieces:
             out.append((torch.cat(idx_pieces), torch.cat(val_pieces)))
         else:
EOF
sbcast -f ${TRAINING_CONFIG}/delta-sharded-p2p-gather.patch ${TRAINING_CONFIG}/delta-sharded-p2p-gather.patch

# CSCS SGLang scheduler-watchdog diagnostic (2026-09-02, after run 3264247). The TP=32 standalone
# rollout's own 300s scheduler watchdog has now hung at 3 distinct call sites across this recipe's
# history (a torch.distributed broadcast, run 3152802; flashinfer MLA plan(), run 3209484, fixed
# by the flashinfer 0.6.14 pin; R3's routed-experts get_topk D2H copy, run 3264247) -- a real,
# recurring TP=32 SGLang fragility, not (so far) a single flake. The stock watchdog only py-spy
# dumps Python stacks on timeout -- shows WHERE a rank is stuck, not the underlying CUDA state,
# which is what actually found + fixed the flashinfer-MLA occurrence. This file defines an extra
# diagnostic hook (nvidia-smi + local CUDA memory/stream state) wired into SGLang's own watchdog
# dump_info callback -- see the srun-side patcher below. Deliberately does NOT attempt an NCCL
# flight-recorder dump: that API's behavior when called ad hoc (outside torch's own watchdog
# handler) is not confirmed safe/fast, and this code runs INSIDE the watchdog thread that is
# about to SIGQUIT the stuck process -- anything slow or blocking here would delay recovery
# instead of just diagnosing it. Kept to fast, synchronous, local reads only.
cat > "${TRAINING_CONFIG}/cscs_watchdog_diag.py" <<- 'PYEOF'
def _cscs_watchdog_dump_info():
    """[CSCS diagnostic] Extra state captured on any SGLang scheduler watchdog timeout, beyond
    the stock py-spy dump: nvidia-smi (all GPUs + compute processes) and local CUDA memory/stream
    state. See CLAUDE.md's "TP=32 SGLang" Known-hazards entry and run 3264247 for why."""
    import os as _os
    import subprocess as _sp

    parts = [f"[CSCS-WATCHDOG] pid={_os.getpid()}"]
    try:
        import torch as _torch

        if _torch.cuda.is_available():
            _dev = _torch.cuda.current_device()
            _free, _total = _torch.cuda.mem_get_info(_dev)
            parts.append(
                f"[CSCS-WATCHDOG] cuda dev={_dev} free={_free / 2**30:.2f}G "
                f"total={_total / 2**30:.2f}G alloc={_torch.cuda.memory_allocated(_dev) / 2**30:.2f}G "
                f"reserved={_torch.cuda.memory_reserved(_dev) / 2**30:.2f}G "
                f"stream_idle={_torch.cuda.current_stream().query()}"
            )
    except Exception as _e:
        parts.append(f"[CSCS-WATCHDOG] torch cuda read failed: {_e}")
    try:
        _out = _sp.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()
        parts.append(f"[CSCS-WATCHDOG] nvidia-smi compute-apps:\n{_out}")
    except Exception as _e:
        parts.append(f"[CSCS-WATCHDOG] nvidia-smi failed: {_e}")
    return "\n".join(parts)
PYEOF
sbcast -f ${TRAINING_CONFIG}/cscs_watchdog_diag.py ${TRAINING_CONFIG}/cscs_watchdog_diag.py

# DeepSeek-V3 (2026-09-04, after run 3279810): example/patches/sglang-deepseek-qkv-a-cache-fix.py,
# embedded (same BASH_SOURCE-under-sbatch reason as every other patch here). sglang 0.5.16's
# DeepSeek loader fuses q_a_proj + kv_a_proj_with_mqa using a cache local to each
# load_weights() call; verl's delta_sharded sync feeds load_weights in 512 MB chunks, so any
# layer whose pair straddles a chunk keeps its dummy-init fused projection -> the rollout served
# a different model than the trainer (run 3279810: reward 0.0 / grad_norm 0.0 every step,
# ppo_kl ~1.5). Load-bearing, applied fatal-if-missing in the srun below. See the file header.
cat > "${TRAINING_CONFIG}/sglang-deepseek-qkv-a-cache-fix.py" <<- 'PYEOF'
# [CSCS, 2026-09-04] Runtime patch for sglang 0.5.16's DeepSeek weight loader
# (sglang/srt/models/deepseek_common/deepseek_weight_loader.py, DeepseekV2WeightLoaderMixin
# .do_load_weights), found debugging train-gsm8k-deepseek-v3-671B-v1-separate-async-megatron.sh
# run 3279810 (19 steps, reward 0.0 / grad_norm 0.0 on every step, ppo_kl ~1.5: the SGLang
# rollout was serving a different model than the trainer).
#
# Mechanism: for q_lora_rank models (DeepSeek-V2/V3) sglang fuses q_a_proj and
# kv_a_proj_with_mqa into one fused_qkv_a_proj_with_mqa parameter at load time, using a
# `cached_a_proj = {}` dict that is LOCAL to each do_load_weights() call -- the fused
# parameter is only written once BOTH halves of a layer arrived in the SAME call. That holds
# for a from-disk load (one call, all tensors), but verl's delta_sharded weight sync
# (verl/workers/rollout/sglang_rollout/delta_loader.py) feeds model.load_weights() in 512 MB
# chunks -- run 3279810's seed sync was 2386 flushes for 45395 tensors -- so any layer whose
# pair straddles a chunk boundary silently keeps its dummy-init (load_format=dummy) fused
# projection. Garbage attention in a handful of layers = garbage generations = zero reward.
#
# Fix: (1) persist the cache on the model object across calls, so a pair completes whenever
# its second half arrives; (2) when a still-unpaired half arrives again (steady syncs ship only
# CHANGED params as NaN-masked full-shape tensors, and skip unchanged ones entirely), MERGE the
# new NaN-masked delta into the cached one instead of overwriting it, so no changed positions
# are lost while waiting for the partner. Dense (seed) tensors carry no NaNs and take the plain
# overwrite path. Correct for both the seed and steady paths; a no-op for non-q_lora models.
#
# Applied on every node inside the main srun (dist-packages is per-container), BEFORE the
# SGLang servers start. Fatal if the anchors are missing (load-bearing for DeepSeek-V3, unlike
# the best-effort watchdog diagnostic) -- re-anchor against the new sglang source if it moves.
# Idempotent. Usage: python3 sglang-deepseek-qkv-a-cache-fix.py [path-to-file-for-testing]
import importlib.util
import py_compile
import sys

MARK = "_cscs_cached_a_proj"

ANCHOR_INIT = "        cached_a_proj = {} if fuse_qkv_a_proj else None\n"
NEW_INIT = (
    "        # [CSCS] cache persists across load_weights calls: verl delta_sharded feeds weights\n"
    "        # in 512 MB chunks, and a per-call cache silently drops any layer whose q_a_proj /\n"
    "        # kv_a_proj_with_mqa pair straddles a chunk boundary (fused projection left at its\n"
    "        # dummy-init value). See sglang-deepseek-qkv-a-cache-fix.py.\n"
    "        if fuse_qkv_a_proj:\n"
    "            if not hasattr(self, \"" + MARK + "\"):\n"
    "                self." + MARK + " = {}\n"
    "            cached_a_proj = self." + MARK + "\n"
    "        else:\n"
    "            cached_a_proj = None\n"
)

ANCHOR_STORE = (
    "                            cached_a_proj[name] = _clone_if_runai_streamed_tensor(\n"
    "                                loaded_weight\n"
    "                            )\n"
)
NEW_STORE = (
    "                            _cscs_new = _clone_if_runai_streamed_tensor(loaded_weight)\n"
    "                            _cscs_old = cached_a_proj.get(name)\n"
    "                            if (\n"
    "                                _cscs_old is not None\n"
    "                                and _cscs_old.shape == _cscs_new.shape\n"
    "                                and _cscs_new.is_floating_point()\n"
    "                            ):\n"
    "                                # [CSCS] steady-sync deltas are NaN-masked; merge, do not overwrite\n"
    "                                _cscs_new = torch.where(torch.isnan(_cscs_new), _cscs_old, _cscs_new)\n"
    "                            cached_a_proj[name] = _cscs_new\n"
)


def main() -> int:
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        spec = importlib.util.find_spec("sglang.srt.models.deepseek_common.deepseek_weight_loader")
        if spec is None or not spec.origin:
            print("FATAL: sglang.srt.models.deepseek_common.deepseek_weight_loader not found")
            return 1
        path = spec.origin
    with open(path) as f:
        src = f.read()
    if MARK in src:
        print("sglang DeepSeek qkv_a cache fix already applied in " + path)
        return 0
    if src.count(ANCHOR_INIT) != 1 or src.count(ANCHOR_STORE) != 1:
        print(
            "FATAL: sglang DeepSeek qkv_a cache fix anchors not found (init=%d store=%d) in %s -- sglang version drift, re-anchor"
            % (src.count(ANCHOR_INIT), src.count(ANCHOR_STORE), path)
        )
        return 1
    if "import torch\n" not in src:
        print("FATAL: expected `import torch` in " + path)
        return 1
    src = src.replace(ANCHOR_INIT, NEW_INIT, 1).replace(ANCHOR_STORE, NEW_STORE, 1)
    with open(path, "w") as f:
        f.write(src)
    py_compile.compile(path, doraise=True)
    print("Applied sglang DeepSeek qkv_a cache fix to " + path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
PYEOF
sbcast -f ${TRAINING_CONFIG}/sglang-deepseek-qkv-a-cache-fix.py ${TRAINING_CONFIG}/sglang-deepseek-qkv-a-cache-fix.py

# Fetch the upstream PR patches once here and sbcast them to every node, instead of
# curl-ing them from inside the srun. In run 3124273 each of the 80 nodes downloaded
# them independently and only ~45/80 succeeded, so the cluster ran mixed verl code:
# the nodes that missed PR #7422 flipped the standalone rollout back to
# load_format=auto and their TP ranks started loading the real 700B weights while
# the patched ranks dummy-initialised, blowing up SGLang's post-load barrier.
# -f makes an HTTP error page a hard failure instead of a "patch" that applies as a no-op.
for pr in 7421 7422 7423; do
    curl -sfL "https://github.com/verl-project/verl/pull/${pr}.patch" -o "${TRAINING_CONFIG}/${pr}.patch" \
        || { echo "FATAL: could not download PR #${pr}"; exit 1; }
    [ -s "${TRAINING_CONFIG}/${pr}.patch" ] \
        || { echo "FATAL: PR #${pr} patch is empty"; exit 1; }
    sbcast -f "${TRAINING_CONFIG}/${pr}.patch" "${TRAINING_CONFIG}/${pr}.patch"
done

# NOTE (2026-08-30): the runtime pip upgrades that used to be here — TransferQueue 0.1.7,
# megatron-core 0.19.0, megatron-bridge 0.6.1, flashinfer 0.6.14 (matched python+cubin) — are now
# all baked into the image (tag alps7-dev-a9f9e56471c0574e, built from
# Alps-Images/apps/verl/Containerfile). Removed. The srun still verifies the versions non-fatally
# and applies the verl *source* patches (PR #7421/#7422/#7423 + the local .patch files), which
# are NOT in the image.


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


# verl is baked into the image at v0.9.0 (Containerfile VERL_REF=v0.9.0), editable-installed from
# /workspace/verl — no runtime checkout needed. The source patches below (PR #7421/#7422/#7423 +
# the local .patch files) are still applied on top.
git -C /workspace/verl --no-pager log --oneline -1 || true


# Redirect pip cache to local tmpfs — ~/.cache/pip is on Lustre which causes
# "Stale file handle" (ESTALE) errors during package downloads.
export PIP_CACHE_DIR=/tmp/pip_cache_${SLURM_JOB_ID}
export TMPDIR=/tmp
mkdir -p $PIP_CACHE_DIR

# Image-version + import smoke test (diagnostic only, non-fatal). Every dependency below is baked
# into the image now; this block just makes a wrong image tag obvious in the first ~30 s of the
# log instead of via a downstream crash. A failure here does NOT stop the job — the real step
# will fail loudly with a clear error if the image is actually wrong.
python3 -c "
import importlib.metadata as _m
for _p in (\"verl\", \"TransferQueue\", \"megatron-core\", \"megatron-bridge\", \"flashinfer-python\", \"flashinfer-cubin\", \"sglang\", \"transformers\", \"transformer-engine\", \"transformer-engine-torch\", \"nvidia-cutlass-dsl\"):
    try:
        print(f\"  {_p}: {_m.version(_p)}\")
    except Exception as _e:
        print(f\"  {_p}: <NOT INSTALLED> ({_e})\")
# import chains the recipe / source patches depend on:
try:
    from megatron.training.models.gpt import GPTModelBuilder, GPTModelConfig, mtp_block_spec  # noqa: F401
    print(\"  megatron.training.models.gpt: OK\")
except ImportError as _e:
    print(f\"  WARNING megatron.training.models.gpt NOT importable ({_e}) — broken image: wrong tag, or transformer-engine trouble: a torch ABI mismatch (run 3235127) or an unpinned TE pulling a cutlass-dsl it needs block_copy from (run 3236353). Check the transformer-engine / nvidia-cutlass-dsl versions above; TE must be 2.12.0.\")
# DeepSeek-V3: the bridge + the native transformers config class this recipe relies on.
try:
    from megatron.bridge.models.deepseek.deepseek_v3_bridge import DeepSeekV3Bridge  # noqa: F401
    print(\"  megatron-bridge DeepSeekV3Bridge: OK\")
except ImportError as _e:
    print(f\"  WARNING megatron-bridge DeepSeekV3Bridge NOT importable ({_e}) — the megatron-bridge in this image has no native DeepSeek-V3 support; model init will fail with: Model architecture DeepseekV3ForCausalLM is not yet supported\")
try:
    from transformers import DeepseekV3Config  # noqa: F401
    print(\"  transformers native deepseek_v3 config: OK\")
except ImportError as _e:
    print(f\"  WARNING transformers has no native DeepseekV3Config ({_e}) — trust_remote_code is False in this recipe, so config load will fail\")
try:
    from sglang.srt.state_capturer.routed_experts import extract_routed_experts_from_meta_info  # noqa: F401
    print(\"  sglang routed_experts capture: OK at sglang.srt.state_capturer.routed_experts\")
except ImportError as _e:
    print(f\"  WARNING sglang routed_experts capture NOT at the patched path ({_e}) — r3-sglang-routed-experts-import-fix.patch needs re-pointing\")
try:
    from sglang.srt.model_executor.model_runner_components.weight_updater import LocalSerializedTensor  # noqa: F401
    print(\"  sglang LocalSerializedTensor: OK at model_runner_components.weight_updater\")
except ImportError as _e:
    print(f\"  WARNING sglang LocalSerializedTensor NOT at the patched path ({_e}) — delta-sharded-localserializedtensor-import-fix.patch needs re-pointing\")
"


# Patch megatron-bridge safe_config_loader to skip filelock.
# /dev/shm and /tmp on CSCS Alps do not support fcntl.flock in the container
# (ENOLCK / ESTALE on every attempt). The lock is unnecessary because the config
# files are written by localid=0 before any reader starts (purely read-only after that).
#
# We match line-by-line on the "with filelock." prefix rather than using a regex
# that tries to parse the argument, because FileLock() arguments often contain
# nested parens (e.g. os.path.join(...)) which break [^)]* patterns.
python3 -c "
import importlib.util
spec = importlib.util.find_spec(\"megatron.bridge.models.hf_pretrained.safe_config_loader\")
if not spec:
    print(\"safe_config_loader not found — skipping patch\")
else:
    p = spec.origin
    with open(p) as f:
        lines = f.readlines()
    if not any(\"import contextlib\" in l for l in lines):
        lines.insert(0, \"import contextlib\n\")
    new_lines = []
    n_patched = 0
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(\"with filelock.\") and stripped.endswith(\":\"):
            indent = len(line) - len(line.lstrip())
            new_lines.append(\" \" * indent + \"with contextlib.nullcontext():\n\")
            n_patched += 1
        else:
            new_lines.append(line)
    if n_patched:
        with open(p, \"w\") as f:
            f.writelines(new_lines)
        print(f\"Patched {n_patched} filelock site(s) in {p}\")
    else:
        print(f\"WARNING: no filelock sites found in {p} — patch may already be applied or code changed\")
"

# CSCS SGLang scheduler-watchdog diagnostic: wire _cscs_watchdog_dump_info (staged above, see the
# batch-host heredoc for the full reasoning) into the sglang WatchdogRaw dump_info hook, which
# is None by default (sglang/srt/utils/watchdog.py never passes it). Best-effort: WARN and
# continue on any mismatch (sglang version drift) rather than aborting the run -- this is
# diagnostic-only, not load-bearing.
python3 -c "
import importlib.util
spec = importlib.util.find_spec(\"sglang.srt.utils.watchdog\")
if not spec:
    print(\"WARNING: sglang.srt.utils.watchdog not found -- skipping CSCS watchdog diagnostic patch\")
else:
    p = spec.origin
    with open(p) as f:
        src = f.read()
    if \"_cscs_watchdog_dump_info\" in src:
        print(\"CSCS watchdog diagnostic already patched in \" + p)
    else:
        anchor = \"class Watchdog:\"
        # soft=soft, appears twice in this file: once in the Watchdog.create() call to
        # _WatchdogReal(...) (which has no dump_info parameter -- injecting there would break
        # sglang startup with a TypeError), and once in the _WatchdogReal.__init__ call to
        # WatchdogRaw(...) (the real target, which does accept dump_info). Disambiguate by also
        # matching the closing paren: only the WatchdogRaw(...) call has soft=soft, immediately
        # followed by the closing paren; the _WatchdogReal(...) call has test_stuck_time=... in
        # between. Verified against the real sglang v0.5.16 source before wiring this in.
        call_line = \"            soft=soft,\n        )\"
        if anchor not in src or call_line not in src:
            print(\"WARNING: sglang watchdog.py anchor or call-site not found in \" + p + \" -- skipping (sglang version drift?)\")
        else:
            with open(\"${TRAINING_CONFIG}/cscs_watchdog_diag.py\") as f:
                diag_fn = f.read()
            src = src.replace(anchor, diag_fn + \"\n\n\" + anchor, 1)
            src = src.replace(
                call_line,
                \"            soft=soft,\n            dump_info=_cscs_watchdog_dump_info,\n        )\",
                1,
            )
            with open(p, \"w\") as f:
                f.write(src)
            print(\"Patched CSCS watchdog diagnostic hook into \" + p)
"

# DeepSeek-V3: make the sglang DeepSeek loader qkv_a fusion cache persist across load_weights
# calls (see the batch-host heredoc + patch header). LOAD-BEARING: without it the delta_sharded
# weight sync leaves a random subset of layers with dummy attention projections (run 3279810).
# Fatal if the sglang anchors are missing -- a silently unpatched rollout is exactly the failure
# mode this fixes. Must run before ray start (the SGLang servers import the module at startup).
python3 ${TRAINING_CONFIG}/sglang-deepseek-qkv-a-cache-fix.py \
    || { echo "FATAL: sglang DeepSeek qkv_a cache fix failed on $(hostname)"; exit 1; }

# Apply the upstream PR patches sbcast to ${TRAINING_CONFIG} before the srun. They are
# NOT in v0.9.0 (all three still apply cleanly to the tag) and must be applied *after*
# the checkout above — git checkout -f would discard them.
#
# A patch that neither applies nor is already present is fatal: a cluster where only
# some ranks carry a patch is worse than one that carries none (run 3124273).
#
# PR #7421: DSA (experimental_attention_variant=dsa) compatibility with mcore >= 0.16.2.
#   Upstream _run_core_attention no longer forwards the x/qr kwargs DSA requires; route
#   DSA instances through the verl patch_forward. Also injects a pure-PyTorch
#   Walsh-Hadamard fallback when fast_hadamard_transform is not installed.
#   DeepSeek-V3: NOT exercised (MLA, not DSA) -- kept so the patched verl tree is identical
#   to the validated GLM-5.1 recipe. Safe to drop once this recipe is validated on its own.
# PR #7422: Preserve load_format=dummy in disaggregated SGLang rollout.
#   At v0.9.0 async_sglang_server.py still overrides dummy -> auto for every non-hybrid
#   replica, which is exactly the standalone rollout of separate-async, so the rollout
#   nodes load the 700B weights from Lustre instead of receiving them over NCCL.
# PR #7423: Fix NCCL deadlock in async disaggregated weight sync.
#   update_actor (thread-pool) and update_weights (event loop) both submit ops to the
#   same PP/EP NCCL communicators. A threading.Lock serialises them; entry and exit
#   barriers ensure all actors complete the weight-sync collective before any resumes
#   training, preventing seq-number mismatches across EP ranks.
for pr in 7421 7422 7423; do
    p="${TRAINING_CONFIG}/${pr}.patch"
    if git -C /workspace/verl apply --check "$p" 2>/dev/null; then
        git -C /workspace/verl apply "$p" && echo "Applied PR #${pr} on $(hostname)"
    elif git -C /workspace/verl apply --reverse --check "$p" 2>/dev/null; then
        echo "PR #${pr} already present on $(hostname), skipping"
    else
        echo "FATAL: PR #${pr} neither applies nor is already present on $(hostname)"
        exit 1
    fi
done

# example/patches/v1-separate-async-fixes.patch: three local fixes for this recipe
# specifically (hybrid-rollout OOM, stale hybrid weight-sync call, a shape-equality
# bug in TensorDict construction) — see the header of the patch file, and Known
# hazards / Run log in CLAUDE.md, for the full story. Same apply-or-fail discipline
# as the PR patches above.
p="${TRAINING_CONFIG}/v1-separate-async-fixes.patch"
if git -C /workspace/verl apply --check "$p" 2>/dev/null; then
    git -C /workspace/verl apply "$p" && echo "Applied v1-separate-async-fixes.patch on $(hostname)"
elif git -C /workspace/verl apply --reverse --check "$p" 2>/dev/null; then
    echo "v1-separate-async-fixes.patch already present on $(hostname), skipping"
else
    echo "FATAL: v1-separate-async-fixes.patch neither applies nor is already present on $(hostname)"
    exit 1
fi

# example/patches/r3-sglang-routed-experts-import-fix.patch: fixes R3 rollout-side
# capture to import from the path this image sglang 0.5.16 actually has
# (sglang.srt.state_capturer.routed_experts), not the sglang.srt.layers.moe.
# routed_experts_capturer path verl v0.9.0 ships by default. See Known hazards / the
# Configuration audit entry / Run log for run 3199623 and probe job 3199799 in CLAUDE.md.
# Same apply-or-fail discipline as the PR patches and v1-separate-async-fixes.patch above.
p="${TRAINING_CONFIG}/r3-sglang-routed-experts-import-fix.patch"
if git -C /workspace/verl apply --check "$p" 2>/dev/null; then
    git -C /workspace/verl apply "$p" && echo "Applied r3-sglang-routed-experts-import-fix.patch on $(hostname)"
elif git -C /workspace/verl apply --reverse --check "$p" 2>/dev/null; then
    echo "r3-sglang-routed-experts-import-fix.patch already present on $(hostname), skipping"
else
    echo "FATAL: r3-sglang-routed-experts-import-fix.patch neither applies nor is already present on $(hostname)"
    exit 1
fi

# example/patches/wsync-debug-progress-log.patch: per-tensor progress logging for the trainer to
# rollout weight sync (diagnostic) + the seed-sync WORLD-barrier that bounds rank drift in
# stream_weights_megatron_to_hf to zero, fixing the ~1-in-2 megatron-bridge collective desync
# (runs 3141801/3207923/3219811/3240762; see its header and CLAUDE.md). Same apply-or-fail
# discipline as the patches above.
p="${TRAINING_CONFIG}/wsync-debug-progress-log.patch"
if git -C /workspace/verl apply --check "$p" 2>/dev/null; then
    git -C /workspace/verl apply "$p" && echo "Applied wsync-debug-progress-log.patch on $(hostname)"
elif git -C /workspace/verl apply --reverse --check "$p" 2>/dev/null; then
    echo "wsync-debug-progress-log.patch already present on $(hostname), skipping"
else
    echo "FATAL: wsync-debug-progress-log.patch neither applies nor is already present on $(hostname)"
    exit 1
fi

# example/patches/delta-sharded-localserializedtensor-import-fix.patch: re-points the
# delta_sharded rollout weight-loader import at the sglang 0.5.16 path (moved). Required by the
# checkpoint_engine.backend: delta_sharded switch. Same apply-or-fail discipline.
p="${TRAINING_CONFIG}/delta-sharded-localserializedtensor-import-fix.patch"
if git -C /workspace/verl apply --check "$p" 2>/dev/null; then
    git -C /workspace/verl apply "$p" && echo "Applied delta-sharded-localserializedtensor-import-fix.patch on $(hostname)"
elif git -C /workspace/verl apply --reverse --check "$p" 2>/dev/null; then
    echo "delta-sharded-localserializedtensor-import-fix.patch already present on $(hostname), skipping"
else
    echo "FATAL: delta-sharded-localserializedtensor-import-fix.patch neither applies nor is already present on $(hostname)"
    exit 1
fi

# example/patches/step1-oom-memdump.patch: DIAGNOSTIC ONLY -- see its header and CLAUDE.md
# runs 3241496 / 3243323 / 3244653. Same apply-or-fail discipline.
p="${TRAINING_CONFIG}/step1-oom-memdump.patch"
if git -C /workspace/verl apply --check "$p" 2>/dev/null; then
    git -C /workspace/verl apply "$p" && echo "Applied step1-oom-memdump.patch on $(hostname)"
elif git -C /workspace/verl apply --reverse --check "$p" 2>/dev/null; then
    echo "step1-oom-memdump.patch already present on $(hostname), skipping"
else
    echo "FATAL: step1-oom-memdump.patch neither applies nor is already present on $(hostname)"
    exit 1
fi

# example/patches/delta-sharded-gather-round-size.patch (DeepSeek-V3, run 3289294): sizes the
# steady delta-sync gather rounds independently of the rollout flush bucket -- see its header.
# LOAD-BEARING together with engine_kwargs.delta_sharded.gather_round_megabytes in the YAML.
# Same apply-or-fail discipline.
p="${TRAINING_CONFIG}/delta-sharded-gather-round-size.patch"
if git -C /workspace/verl apply --check "$p" 2>/dev/null; then
    git -C /workspace/verl apply "$p" && echo "Applied delta-sharded-gather-round-size.patch on $(hostname)"
elif git -C /workspace/verl apply --reverse --check "$p" 2>/dev/null; then
    echo "delta-sharded-gather-round-size.patch already present on $(hostname), skipping"
else
    echo "FATAL: delta-sharded-gather-round-size.patch neither applies nor is already present on $(hostname)"
    exit 1
fi

# example/patches/delta-sharded-p2p-gather.patch (DeepSeek-V3, run 3293605): the steady delta-sync
# hang fix -- targeted P2P instead of the padded 480-way gather-to-one. LOAD-BEARING. Touches
# only sparse_gather.py (no ordering constraint). Same apply-or-fail discipline.
p="${TRAINING_CONFIG}/delta-sharded-p2p-gather.patch"
if git -C /workspace/verl apply --check "$p" 2>/dev/null; then
    git -C /workspace/verl apply "$p" && echo "Applied delta-sharded-p2p-gather.patch on $(hostname)"
elif git -C /workspace/verl apply --reverse --check "$p" 2>/dev/null; then
    echo "delta-sharded-p2p-gather.patch already present on $(hostname), skipping"
else
    echo "FATAL: delta-sharded-p2p-gather.patch neither applies nor is already present on $(hostname)"
    exit 1
fi


# Mirror model config files to local tmpfs to avoid Lustre metadata contention.
# All training workers calling AutoConfig.from_pretrained() simultaneously causes
# ENOLCK / ESTALE on the Lustre MDS. Only local rank 0 does the copy; others wait.
export MODEL_LOCAL=/tmp/dsv3_model_${SLURM_JOB_ID}
if [ $SLURM_LOCALID -eq 0 ]; then
    mkdir -p $MODEL_LOCAL
    # Copy small config/tokenizer files locally
    find ${TRAINING_HOME}/models/${MODEL_NAME} -maxdepth 1 -not -name "*.safetensors" -type f \
        -exec cp {} $MODEL_LOCAL/ \; 2>/dev/null || true
    # Symlink safetensors back to Lustre so megatron-bridge can still load weights
    for f in ${TRAINING_HOME}/models/${MODEL_NAME}/*.safetensors; do
        ln -sf "$f" "$MODEL_LOCAL/$(basename "$f")"
    done 2>/dev/null || true
    # DeepSeek-V3: sanitize the MIRRORED config.json (the Lustre checkpoint is untouched), per
    # the upstream verl DeepSeek-V3 example ("remove quantization_config from config.json and
    # set num_nextn_predict_layers=0"):
    #   quantization_config     -> removed. The checkpoint is FP8 block-quantized; the trainer
    #                              side does not care (DeepSeekV3Bridge dequantizes by tensor
    #                              dtype), but the SGLang standalone rollout would otherwise
    #                              build FP8 weights expecting *_scale_inv while
    #                              load_format=dummy + the bf16 weight sync feed it bf16.
    #   num_nextn_predict_layers -> 0. verl already zeroes this on the trainer side when
    #                              model.mtp.enable is False; mirrored here so SGLang and the
    #                              weight-sync export see the same MTP-free model.
    #   auto_map                 -> removed. trust_remote_code is False; the bundled
    #                              transformers-4.33-era remote code must never be picked up.
    # Fatal on any failure: a half-sanitized config would surface much later as an opaque
    # dtype / shape error in SGLang model load. NOTE: this python -c lives inside the outer
    # single-quoted bash -c body -- double quotes only, no apostrophes (run-3149339 hazard).
    python3 -c "
import json
p = \"${MODEL_LOCAL}/config.json\"
with open(p) as f:
    cfg = json.load(f)
mt = cfg.get(\"model_type\")
assert mt == \"deepseek_v3\", \"unexpected model_type %r in %s\" % (mt, p)
changed = []
for k in (\"quantization_config\", \"auto_map\"):
    if k in cfg:
        cfg.pop(k)
        changed.append(\"-\" + k)
if cfg.get(\"num_nextn_predict_layers\", 0) != 0:
    cfg[\"num_nextn_predict_layers\"] = 0
    changed.append(\"num_nextn_predict_layers=0\")
with open(p, \"w\") as f:
    json.dump(cfg, f, indent=2)
print(\"Sanitized %s for bf16 / no-MTP serving: %s (num_hidden_layers=%s, n_routed_experts=%s)\" % (p, changed or [\"already clean\"], cfg.get(\"num_hidden_layers\"), cfg.get(\"n_routed_experts\")), flush=True)
" || { echo "FATAL: could not sanitize ${MODEL_LOCAL}/config.json on $(hostname)"; exit 1; }
    touch $MODEL_LOCAL/.ready
fi
until [ -f $MODEL_LOCAL/.ready ]; do sleep 1; done

# Patch the YAML on the head node to point at the local model dir
if [ $SLURM_PROCID -eq 0 ]; then
    sed -i "s|${TRAINING_HOME}/models/${MODEL_NAME}|${MODEL_LOCAL}|g" ${TRAINING_CONFIG}/grpo_gsm8k.yaml
fi

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

# NCCL flight recorder — so the NEXT weight-sync hang (or any collective timeout) dumps a
# per-rank collective trace instead of the useless "last enqueued NCCL work: -1" seen in run
# 3219811 (that -1 means the ring buffer was OFF). TRACE_BUFFER_SIZE enables it; DUMP_ON_TIMEOUT
# writes it on watchdog fire; TEMP_FILE lands one file per rank under /tmp so it can be pulled
# from the failed nodes. Pinpoints which rank stalled on which collective seq (i.e. which tensor).
export TORCH_NCCL_TRACE_BUFFER_SIZE=20000
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE=/tmp/nccl_flightrecorder_${SLURM_JOB_ID}_rank

# NCCL_NVLS_ENABLE=0 (2026-08-31): disable NVLink-multicast (NVLS / NVLink SHARP). On GH200,
# with the many communicators here -- TP=4, PP=3, EP=8, DP=5, world, CE -- each NVLS-enabled
# group reserves multicast buffers on device 0, part of the ~12.3 GiB non-PyTorch memory on
# trainer local-GPU-0 that leaves the step-1 fused_adam optimizer-state alloc 12-24 MiB short
# (runs 3241496 / 3243323). Pure perf knob -- falls back to ring/tree all-reduce -- no
# correctness or collective-stability impact. Paired with the precision-aware optimizer above.
export NCCL_NVLS_ENABLE=0

# NOTE: do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True here. Tried in run 3219305 as
# an OOM mitigation, but this srun body is shared by the SGLang standalone-rollout processes too,
# and SGLang torch_memory_saver -- called unconditionally by load_model_with_memory_saver even
# with free_cache_engine set to false -- hard-raises "TorchMemorySaver is disabled ...
# expandable_segments is not supported yet", killing every rollout TP rank before the first
# training step. The step-21 OOM mitigation is the ppo_max_token_len_per_gpu 16384 to 12288 cut
# in grpo_gsm8k.yaml instead.

# The hybrid-rollout OOM fallback, stale weight-sync skip, and
# list_of_dict_to_tensordict fix all now live in
# example/patches/v1-separate-async-fixes.patch (applied via git apply above,
# alongside the upstream PR patches) rather than a sitecustomize.py runtime
# monkeypatch — see the header of that patch file and Known hazards / Run log in
# CLAUDE.md for why. Confirmed working end-to-end in run 3149736.

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

    HYDRA_FULL_ERROR=1 python -m verl.trainer.main_ppo \
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
