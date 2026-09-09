#!/bin/bash

#SBATCH --nodes=40
#SBATCH --account=csstaff
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=4:00:00

# ─────────────────────────────────────────────────────────────────────────────
# Copy of rl-bench-apertus-v1.5-70B-sglang-megatron-async.sh migrated to the
# same image and verl trainer as
# example/train-gsm8k-glm5.1-700B-v1-separate-async-megatron.sh:
#
#   * image   : the verl-cuda image with baked-in verl v0.9.0 + updated deps
#               (TransferQueue 0.1.7, megatron-core 0.19.0, megatron-bridge 0.6.1,
#               flashinfer 0.6.14) -- no runtime verl checkout needed.
#   * trainer : verl.experimental.fully_async_policy.fully_async_main
#               -> verl.trainer.main_ppo, V1 trainer in separate-async mode
#               (trainer.use_v1=True, trainer.v1.trainer_mode=separate_async).
#
# Config changes carried over from the GLM v1-separate migration:
#   * async_training: block  -> trainer.v1.separate_async.* / trainer.v1.sampler.*
#   * top-level rollout: block gone; the standalone rollout pool is declared in
#     actor_rollout_ref.rollout.{nnodes,n_gpus_per_node}
#   * data.train_batch_size: 0 -> parameter_sync_step * ppo_mini_batch_size
#   * TransferQueue: forced on by main_ppo; set explicitly + size storage units
#   * old_log_probs: rollout.calculate_log_probs + algorithm.rollout_correction
#     (actor.use_rollout_log_probs is unused by V1)
#   * lr schedule: V1 sets actor.optim.total_training_steps before the workers
#     are created, so the fully-async lr_decay_steps workaround is dropped
#   * apertus-benchmarks/patches/v1-separate-async-fixes.patch (2-hunk) + upstream
#     PR #7421/#7422/#7423/#7661 (#7661 = the 3rd v1-separate-async fix, now upstream)
#     are applied (needed by the V1 separate-async standalone-rollout path)
#
# The fully-async recipe's reset to theely/verl Fix-fsdp-model-loading-on-async is
# dropped: it is a single-file FSDP2-only change (unused by the Megatron trainer)
# and a hard reset would clobber the baked v0.9.0 verl tree.
#
# CAVEAT -- the V1 separate-async trainer is not a pure disaggregated setup:
# PPOTrainer._setup() always builds hybrid rollout replicas on top of the training
# worker group (trainer world / rollout world = 48/16 = 3 replicas here) *in
# addition to* the standalone rollout. actor_rollout_ref.hybrid_engine is not
# consulted in the V1 path; v1-separate-async-fixes.patch no-ops those replicas
# (they would otherwise squat gpu_memory_utilization=0.75 on every training GPU
# at init and OOM the trainer's weight-sync staging).
#
# Apertus-1.5 support:
#   Group 1 (swiss-ai transformers wheel, SGLang PR #32979 + local fixes,
#   vision_model=False patch) -- unchanged from the fully-async recipe.
#   Group 2 (Megatron) -- applied at runtime as two GitHub `compare` diffs
#   (upstream release tag ... theely fork branch) onto the STOCK image
#   megatron-core 0.19.0 / megatron-bridge 0.6.1 (NO fork wheels, NO runtime
#   clone, NO wheel build):
#     theely/Megatron-LM     : mlp.py module activation_func + finalize_model_grads
#                              xIELU TP grad-sum (2 hunks off NVIDIA:core_v0.19.0)
#     theely/Megatron-Bridge : models/apertus{,1p5}/ (Apertus1p5Bridge) +
#                              registration + safe_config_loader flock fix
#                              (off NVIDIA-NeMo:v0.6.0)
#   Regenerate the fork branches with patches/build-apertus1p5-megatron-forks.sh;
#   pin MEGATRON_{LM,BRIDGE}_FORK_REF in Group 2 below. The apertus code is
#   byte-identical to what runs 3240861/3243271/3243467 already built and ran
#   against stock 0.19.0/0.6.1 -- they trained degenerately only because the
#   reward function demanded <answer> tags the model never emits (CLAUDE.md
#   "The zero-reward chain"), since fixed and validated (run 3251587, fork wheels).
# ─────────────────────────────────────────────────────────────────────────────

export VERL_IMAGE="jfrog.svc.cscs.ch/docker-group-csstaff/alps-images/verl-cuda:alps7-dev-af30d3905eedb02f" #verl-cuda image: baked verl v0.9.0 + updated deps (matches train-gsm8k-glm5.1-700B-v1-separate-async-megatron.sh)

export MODEL_NAME="Apertus-v1.5-70B"
export MODEL_REPO="swiss-ai"

# Optional: load the actor/rollout/ref model from an existing local checkpoint
# directory (e.g. an SFT checkpoint) instead of downloading MODEL_REPO/MODEL_NAME
# from HF. Leave empty for the default HF-download behavior. When set, this must
# be a real HF-format snapshot dir (config.json, tokenizer files, safetensors) --
# same shape as what `hf download` produces -- and the HF download step below is
# skipped entirely.
# Example: /capstor/store/cscs/swissai/infra01/users/xyixuan/rl-bench/models/ap1p5-70b-sft-262k-2700_corr
# IMPORTANT: do not include a trailing slash in the path.
export MODEL_CHECKPOINT_PATH="/capstor/store/cscs/swissai/infra01/RL_Infra/models/ap1p5-70b-sft-262k-2700_corr"


# Passed through to data.apply_chat_template_kwargs.enable_thinking (verl's
# RLHFDataset forwards this straight into tokenizer.apply_chat_template) --
# controls whether the chat template opens a <think>/reasoning turn. Harmless
# to flip on for a template that doesn't branch on this kwarg -- unused
# template variables are silently ignored.
export ENABLE_THINKING="${ENABLE_THINKING:-False}"   # env-overridable (True|False)
# Default final-answer marker for the TRAINING rows (gsm8k / dapo_math), read by reward.py AND
# dataset_prepare.py (SYSTEM_PROMPT) from the container env: bracket = [[[N]]] (the benchmark
# original), boxed = \boxed{N}. The AIME-2024 eval rows (data_source aime_2024) are ALWAYS
# prompted for and scored with \boxed{} regardless of this switch (reward.py routes on
# data_source). Same convention as the NeMo-RL side of the benchmark.
export ANSWER_MARKER="${ANSWER_MARKER:-bracket}"     # env-overridable (bracket|boxed)

# Response-length budget is tied to ENABLE_THINKING:
#   thinking OFF (short, direct answers): 500 -> 2048 tokens,
#     ppo_max_token_len_per_gpu left at 16384 -- UNTESTED at 2048, but very low
#     risk: 500/16384 was VALIDATED (run 3283187: 46/46 steps, peak mem
#     73.9-76.3/95 GiB, no OOM) with 15872 tokens of headroom over the floor
#     (512+500=1012); 2048 only narrows that to 13824 (floor 512+2048=2560),
#     still far more headroom than either validated 8192 or 12288 pairing.
#   thinking ON (needs room for a <think> turn): 12288 tokens,
#     ppo_max_token_len_per_gpu 16384 -- UNTESTED. 8192/12288 was validated
#     (run 3284608: 92/92 steps, peak mem 71.76/95 GiB, no OOM); 16384/20480
#     reliably OOMs by step 3 as real response length grows toward the cap
#     (runs 3282747, 3289274) -- the OOM is driven by response length growing
#     during training, not a fixed one-time cost, so 12288 is a real memory
#     risk between those two known points, not a safe extrapolation. 16384 for
#     ppo_max_token_len_per_gpu keeps the same ~3584-token headroom margin as
#     every previously-validated pairing (floor = 512 + 12288 = 12800).
#     Watch closely for the same OOM signature on the first run at this value.
# reward.py reads ENABLE_THINKING back out of the container env (passed via
# env.toml below) to pick matching length-penalty thresholds -- keep the two
# in sync if either changes.
if [ "${ENABLE_THINKING}" = "True" ]; then
    export MAX_RESPONSE_LENGTH=12288
    # 16384 -> 14336 (2026-09-07, after run 3314817): exactly max_prompt(2048) + max_response, so
    # dynamic batching cannot pack two long samples into one 16k-token micro-batch; caps the
    # backward-pass activation peak at one longest sample. Not a length limit.
    export PPO_MAX_TOKEN_LEN_PER_GPU=14336
else
    export MAX_RESPONSE_LENGTH=2048
    export PPO_MAX_TOKEN_LEN_PER_GPU=16384
fi

# Benchmark dataset (2026-09-07): gsm8k (default) or dapo-math. Both use the same [[[N]]] answer
# convention and reward.py; dataset_prepare.py builds ${TRAINING_HOME}/data/${BENCHMARK}/.
#   gsm8k     : openai/gsm8k train (7473) / test (1319), greedy pass@1 validation.
#   dapo-math : BytedTsinghua-SIA/DAPO-Math-17k train (~17k unique problems, de-duplicated from
#               the replicated Hub parquet) / AIME-2024 test (30 problems), validated DAPO-style
#               as avg@32 (val_kwargs n=32, temperature 1.0, top_p 0.7) -> the metric to read is
#               val-core/aime_2024/acc/mean@32 (the per-sample "acc" from reward.py). Problems
#               are longer and need real reasoning: run it with ENABLE_THINKING=True (12288-token
#               responses); the 2048-token prompt budget below fits the longest DAPO prompts and
#               keeps ppo_max_token_len_per_gpu (16384) above max_prompt + max_response (14336).
#               92 steps x 48 prompts = 4416 prompts = ~1/4 of the 17k train set.
# Step budget (env-overridable). dapo-math with thinking on: run 3314817 (16 nodes: 12 train /
# 4 rollout) measured 26 min/step (21 min generation, mean response 8401 tokens, 44% at the
# 12288 cap) and ~25 min per AIME avg@32 validation -- size TOTAL_TRAINING_STEPS to the
# allocation (e.g. ~20 steps in 4 h on the 40-node 24/16 split below).
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-92}"
# Minor actor hyper-parameters, env-overridable for A/B benchmark runs (defaults = verl v0.9.0
# stock values used by every run up to 3330187): ACTOR_LR (actor.optim.lr, 1e-6) and
# CLIP_RATIO_HIGH (actor.clip_ratio_high, 0.2 -- DAPO "clip-higher" uses 0.28; clip_ratio_low
# stays 0.2).
export ACTOR_LR="${ACTOR_LR:-1e-6}"
export CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-0.2}"
export BENCHMARK="${BENCHMARK:-gsm8k}"
case "${BENCHMARK}" in
    # Training rollout sampling (ROLLOUT_*: GRPO group size n, temperature, top_p; top_k is
    # always disabled) and validation sampling (VAL_*). gsm8k: n=16 (matches 3184869), greedy
    # val. dapo-math (2026-09-08, benchmark spec): train n=16 T=1.0 top_p=1.0; AIME-2024 val
    # n=32 T=0.6 top_p=0.95 (read val-core/aime_2024/acc/mean@32).
    gsm8k)     export MAX_PROMPT_LENGTH=512;  export ROLLOUT_N=16; export ROLLOUT_TEMPERATURE=1.0; export ROLLOUT_TOP_P=1.0
               export VAL_N=1;  export VAL_TEMPERATURE=0;   export VAL_TOP_P=1.0;  export VAL_DO_SAMPLE=False ;;
    dapo-math) export MAX_PROMPT_LENGTH=2048; export ROLLOUT_N=16; export ROLLOUT_TEMPERATURE=1.0; export ROLLOUT_TOP_P=1.0
               export VAL_N=32; export VAL_TEMPERATURE=0.6; export VAL_TOP_P=0.95; export VAL_DO_SAMPLE=True ;;
    *) echo "FATAL: unknown BENCHMARK=${BENCHMARK} (gsm8k | dapo-math)"; exit 1 ;;
esac
export PROJECT_NAME="apertus-benchmarks"
export EXPERIMENT_NAME="${MODEL_NAME}-${BENCHMARK}-verl-sglang-megatron-v1-separate-async-${SLURM_JOB_NUM_NODES}n"
export RUN_NAME="${EXPERIMENT_NAME}-run-${SLURM_JOB_ID}"
export TRAINING_HOME=/capstor/scratch/cscs/${USER}/RL/${MODEL_NAME}
export TRAINING_CONFIG=/tmp
export CHECKPOINT_HOME=${TRAINING_HOME}/checkpoints/${EXPERIMENT_NAME}-run-${SLURM_JOB_ID} #remove "run-${SLURM_JOB_ID}" to enable checkpoint resuming
export MODEL_LOAD_PATH="${MODEL_CHECKPOINT_PATH:-${TRAINING_HOME}/models/${MODEL_NAME}}"


mkdir -p $TRAINING_HOME
cd $TRAINING_HOME



# Node split (2026-09-07, user decision after run 3314817 where generation was ~80% of a 26 min
# step on a 12/4 split): 40 nodes = 24 training (96 GPUs, TP=8 -> DP=12; 768 rows / 12 = 64 per
# rank) + 16 rollout (64 GPUs -> 16 SGLang TP=4 replicas, 4x the generation throughput).
# ROLLOUT_NNODES is env-overridable; the old default was ceil(0.25 x nodes).
export ROLLOUT_NNODES="${ROLLOUT_NNODES:-16}"
export TRAINING_NNODES=$(( SLURM_JOB_NUM_NODES - ROLLOUT_NNODES ))

# V1 separate-async batching contract:
#   data.train_batch_size == trainer.v1.separate_async.parameter_sync_step * actor.ppo_mini_batch_size
# parameter_sync_step is the number of actor updates between two weight syncs to
# the standalone rollout (the fully-async recipe called this trigger_parameter_sync_step).
# train_batch_size and ppo_mini_batch_size are PROMPT counts; ppo_mini_batch_size is
# multiplied by rollout.n internally for the DP-parallel actor mini-batch, and that
# product must be >= 2x dp_size (= DP x EP = 6 x 1 = 6: 12 training nodes x 4 GPUs
# / TP=8 = 6 DP replicas, dense so EP=1) -- 48 * 16 = 768 >> 12.
#
# ppo_mini_batch_size=48 matches the fully-async recipe / the sibling
# rl-bench-apertus-v1.5-70B-sglang-megatron-async.sh (job 3184869, which learned
# to reward ~0.99). The V1-separate migration originally shrank it to 6 (copied
# from the GLM-5.1 700B v1-separate recipe, where 6 is forced by that model being
# memory-bound at PP=3) -- that cut perf/throughput ~3x (fixed per-step cost:
# ~9.7s trainer->rollout weight sync + optimizer + TransferQueue round-trip,
# amortized over 8x fewer tokens; run 3247540 = ~37 tok/s vs 3184869 = ~120).
# Apertus-70B at TP=8 + full offload has ~34 GiB free (61/95 GiB peak at mb=6);
# dynamic-bsz caps each MICRObatch at ppo_max_token_len_per_gpu=16384 so the
# activation peak is per-microbatch, not per-mini-batch -- 8x more rows just
# means more sequential microbatches, same peak.
# ROLLOUT_N (responses per prompt) is set per benchmark in the BENCHMARK case above.
# ppo_mini_batch_size kept at 48 (matches 3184869; this is the actor mini-batch
# granularity, and dynamic-bsz already caps per-microbatch activation memory
# regardless of this value -- see the block comment above). To still get
# data.train_batch_size 96 -> 48 prompts/step (and trainer.total_training_steps
# 46 -> 92 below -- same total 4416 prompts across the run, just delivered in
# smaller/more frequent steps), PARAMETER_SYNC_STEP is lowered 2 -> 1 instead:
# weight sync to the standalone rollout now happens every actor update instead
# of every 2nd, i.e. fresher (less off-policy) rollout weights, at the cost of
# more frequent ~9.7s NCCL weight-sync overhead.
export PPO_MINI_BATCH_SIZE=48        # prompts; x ROLLOUT_N (16) = 768 rows
export PARAMETER_SYNC_STEP=1         # was 2 -- see comment above
export TRAIN_BATCH_SIZE=$(( PARAMETER_SYNC_STEP * PPO_MINI_BATCH_SIZE ))  # 48

cat > "${TRAINING_CONFIG}/env.toml" <<- EOF
image = "${VERL_IMAGE}"
mounts = ["/capstor", "/iopsstor", "/users","/tmp"]
workdir = "/workspace/verl"
writable = true
entrypoint = true
[env]
PMIX_MCA_psec = "native"
HF_TOKEN = "$(cat ~/HF_TOKEN)"
ENABLE_THINKING = "${ENABLE_THINKING}"  # chat-template thinking switch (reward.py no longer reads it)
ANSWER_MARKER = "${ANSWER_MARKER}"  # reward.py + dataset_prepare.py read this (bracket | boxed)
BENCHMARK = "${BENCHMARK}"  # dataset_prepare.py reads this (gsm8k | dapo-math)
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
  train_files: ${TRAINING_HOME}/data/${BENCHMARK}/train.parquet
  val_files:   ${TRAINING_HOME}/data/${BENCHMARK}/test.parquet
  max_prompt_length: ${MAX_PROMPT_LENGTH}  # 512 gsm8k / 2048 dapo-math (verl truncation=error: a longer prompt is fatal)
  train_batch_size: ${TRAIN_BATCH_SIZE}   # == parameter_sync_step * ppo_mini_batch_size
  gen_batch_size: 1      # prompts are submitted to the rollout one at a time
  return_raw_chat: True
  # max_response_length also sets rollout.response_length (oc.select default) --
  # the SGLang sampling_params.max_new_tokens cap. Tied to ENABLE_THINKING --
  # see MAX_RESPONSE_LENGTH above for the validated 500/8192 values and why.
  max_response_length: ${MAX_RESPONSE_LENGTH}
  apply_chat_template_kwargs:
    enable_thinking: ${ENABLE_THINKING}

actor_rollout_ref:
  model:
    path: ${MODEL_LOAD_PATH}
    use_remove_padding: True  # Megatron THD layout requires sequence packing
    # Fused linear cross-entropy (2026-09-07, after run 3314817's step-2 backward OOM at the 12288
    # thinking cap): computes log-probs + entropy without materializing the [tokens x 131072-vocab]
    # fp32 logits and their backward copies (~8.6 GB per copy at 16k tokens, the largest
    # length-dependent activation). Prereqs (THD, actor, no MTP, uniform temperature) all met;
    # verl auto-disables with a warning otherwise. Numerically equivalent; validated on DeepSeek-V3.
    use_fused_kernels: True
    # eager, not flash_attention_2/sdpa: the vision tokenizer submodule
    # (instantiated even for this text-only benchmark) supports neither;
    # eager is the one implementation every model supports.
    override_config:
      attn_implementation: eager
    use_shm: false

  actor:
    # V1 sets actor.optim.total_training_steps in _init_dataloader (before the
    # workers are created), so the fully-async lr_decay_steps workaround is gone.
    checkpoint:
      # Apertus-v1.5 is multimodal; this recipe deliberately does not map the
      # vision_tokenizer / audio_tokenizer towers (text-only GSM8K). The strict
      # HF-checkpoint export then fails ("473 tensors ... not written", run
      # 3240861). strict: False saves the LM-only partial checkpoint instead.
      strict: False
    optim:
      lr: ${ACTOR_LR}  # default 1e-6 (verl stock); env-overridable
    clip_ratio_low: 0.2
    clip_ratio_high: ${CLIP_RATIO_HIGH}  # default 0.2 (verl stock); 0.28 = DAPO clip-higher; env-overridable
    ppo_mini_batch_size: ${PPO_MINI_BATCH_SIZE}
    ppo_micro_batch_size_per_gpu: 1
    # Must exceed max_prompt_length(512) + max_response_length above, or a
    # single near-max-length sequence can't be packed into one microbatch.
    # Tied to ENABLE_THINKING -- see PPO_MAX_TOKEN_LEN_PER_GPU above.
    ppo_max_token_len_per_gpu: ${PPO_MAX_TOKEN_LEN_PER_GPU}
    use_dynamic_bsz: True
    # 12 training nodes x 4 GPUs = 48 GPUs; TP=8, PP=1, EP=1 -> 8 GPUs/replica,
    # 6 DP replicas. Apertus-v1.5-70B is dense (no MoE), so EP must be 1. TP=8
    # (raised from 4 in the fully-async recipe -- run 3171564: CUDA OOM in
    # Megatron's DDP grad-buffer allocation with TP=4's ~17.5B-param-per-GPU
    # shard) roughly halves the per-GPU parameter/gradient shard.
    megatron:
      tensor_model_parallel_size: 8
      pipeline_model_parallel_size: 1
      expert_model_parallel_size: 1
      # sequence_parallel defaults True whenever TP>1; it shards activations
      # along the packed (THD) sequence dim and reduce-scatters them, requiring
      # the packed micro-batch token count to be a multiple of TP -- which the
      # experimental async trainers never actually pad for (run 3173479:
      # AssertionError in megatron/core/tensor_parallel/mappings.py
      # _reduce_scatter_along_first_dim). Disabled; the only benefit is reduced
      # activation memory, which override_transformer_config.recompute_granularity:
      # full below already delivers.
      sequence_parallel: False
      param_offload: True
      grad_offload: True
      optimizer_offload: True
      vanilla_mbridge: False  # route through megatron-bridge AutoBridge -> the vendored Apertus1p5Bridge (Group 2 below)
      override_transformer_config:
        recompute_granularity: full
        recompute_method: uniform
        recompute_num_layers: 1
        use_cpu_initialization: True

  rollout:
    name: sglang
    mode: async
    load_format: dummy
    # Standalone (disaggregated) rollout resources -- V1 separate-async reads the
    # rollout pool size from here instead of a top-level rollout: block.
    nnodes: ${ROLLOUT_NNODES}
    n_gpus_per_node: 4
    temperature: ${ROLLOUT_TEMPERATURE}
    top_p: ${ROLLOUT_TOP_P}
    top_k: -1  # disabled
    n: ${ROLLOUT_N} # responses per prompt -- GRPO group size for the relative-advantage baseline (per benchmark)
    # Validation sampling per benchmark (see BENCHMARK above): gsm8k = greedy pass@1 (verl defaults);
    # dapo-math = AIME-2024 avg@32 (n=32, T=0.6, top_p=0.95, top_k disabled).
    val_kwargs:
      n: ${VAL_N}
      temperature: ${VAL_TEMPERATURE}
      top_p: ${VAL_TOP_P}
      top_k: -1
      do_sample: ${VAL_DO_SAMPLE}
    tensor_model_parallel_size: 4
    # 0.75 -> 0.65 (2026-09-08, run 3317733): with 12288-token responses a rollout TP rank OOMed on a
    # 2 GiB dynamic allocation -- SGLang held 80.5 GB (71 GB static pool + dynamic) and the
    # co-resident verl CheckpointEngineWorker 14.2 GB, 377 MiB free. Smaller static pool = more room
    # for prefill/sampling buffers at long sequences; less KV per replica, but there are 16 replicas.
    gpu_memory_utilization: 0.65
    calculate_log_probs: True   # required: bypass_mode reads rollout_log_probs as old_log_probs
    log_prob_use_dynamic_bsz: True
    checkpoint_engine:
      # dense 70B via NCCL broadcast -- the MoE-specific delta_sharded backend the
      # GLM recipe switched to (for its ~6000-collective full-model gather hang)
      # does not apply here.
      backend: nccl
      # 2048 -> 512 (2026-09-08, run 3317733): the NCCL checkpoint-engine receive buffers live on
      # the rollout GPUs next to SGLang; 4x smaller buckets free ~3 GB there (same as DeepSeek-V3).
      update_weights_bucket_megabytes: 512

  ref:
    log_prob_use_dynamic_bsz: True
    log_prob_max_token_len_per_gpu: ${PPO_MAX_TOKEN_LEN_PER_GPU}  # see actor.ppo_max_token_len_per_gpu above
    megatron:
      param_offload: True  # keep ref params on CPU when not computing log probs
      tensor_model_parallel_size: 8
      sequence_parallel: False  # see actor.megatron.sequence_parallel above (run 3173479)
      vanilla_mbridge: False

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
    path: ${TRAINING_CONFIG}/reward.py
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
  # _balance_batch (trainer_base.py:1467, on by default) reads tag["seq_len"] on
  # every trajectory. When the separate-async sampler delivers fewer than
  # required_multiple = ppo_mini_batch_size(48) * rollout.n(16) = 768 valid
  # trajectories in a sync window, verl pads with synthetic samples that carry
  # no seq_len tag -> KeyError (run 3250502 died here at step 36). Disabling
  # DP-token-load balancing avoids that path entirely; use_dynamic_bsz already
  # bounds per-microbatch tokens so the imbalance cost is small. (At the old
  # ppo_mini_batch_size=6 the multiple was 96 and run 3247540 never came up
  # short in 46 steps, so this was dormant.)
  balance_batch: False
  # Fixed step cap (overrides the epoch-based count). 92 steps x TRAIN_BATCH_SIZE
  # (parameter_sync_step 1 x ppo_mini_batch_size 48 = 48 prompts) = 4416 prompts
  # -- same total as the prior 46 x 96, and the fully-async recipe's
  # total_rollout_steps: 4416 (job 3184869) -- just twice as many, half-size steps.
  total_training_steps: ${TOTAL_TRAINING_STEPS}  # env-overridable, default 92
  project_name: ${PROJECT_NAME}
  experiment_name: ${RUN_NAME}
  nnodes: ${TRAINING_NNODES}
  n_gpus_per_node: 4
  save_freq: 100  # > total_training_steps: no mid-run checkpoint on the shakedown (faster, cleaner signal). Lower for a real run (actor.checkpoint.strict is False so the multimodal-tower export gap is non-fatal).
  test_freq: ${TOTAL_TRAINING_STEPS}  # == total_training_steps: validate once, at the end
  val_before_train: true   # + a baseline validation before step 1
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

# Reward function: downloaded from this repo (raw) rather than embedded, so it
# can be iterated on without touching this launch script. Fetched once on the
# batch host (never per-node -- CLAUDE.md "Never fetch per-node from the
# internet inside the srun"), sanity-checked, sbcast below. It must define
# compute_reward(data_source, solution_str, ground_truth, ...) -- see
# actor_rollout_ref.reward_model / custom_reward_function in grpo_gsm8k.yaml.
# NOTE: reward.py and dataset_prepare.py's SYSTEM_PROMPT are coupled -- [[[N]]] for training
# rows (or \boxed{N} with ANSWER_MARKER=boxed), \boxed{} for the AIME-2024 eval rows; keep
# them in sync, and bump DATASET_PROMPT_VERSION on any prompt
# change so the cached parquet is rebuilt.
export REWARD_FN_URL="https://raw.githubusercontent.com/eth-cscs/alps-extended-images/refs/heads/Add-megatron-rl-recipes/Alps-Images/apps/verl/apertus-benchmarks/reward.py"
export DATASET_PREPARE_URL="https://raw.githubusercontent.com/eth-cscs/alps-extended-images/refs/heads/Add-megatron-rl-recipes/Alps-Images/apps/verl/apertus-benchmarks/dataset_prepare.py"
curl -sfL "${REWARD_FN_URL}" -o "${TRAINING_CONFIG}/reward.py" \
    || { echo "FATAL: could not download reward.py from ${REWARD_FN_URL}"; exit 1; }
grep -q "def compute_reward" "${TRAINING_CONFIG}/reward.py" \
    || { echo "FATAL: downloaded reward.py has no compute_reward()"; exit 1; }
curl -sfL "${DATASET_PREPARE_URL}" -o "${TRAINING_CONFIG}/dataset_prepare.py" \
    || { echo "FATAL: could not download dataset_prepare.py from ${DATASET_PREPARE_URL}"; exit 1; }
grep -q "SYSTEM_PROMPT" "${TRAINING_CONFIG}/dataset_prepare.py" \
    || { echo "FATAL: downloaded dataset_prepare.py has no SYSTEM_PROMPT"; exit 1; }

sbcast -f ${TRAINING_CONFIG}/reward.py ${TRAINING_CONFIG}/reward.py
sbcast -f ${TRAINING_CONFIG}/dataset_prepare.py ${TRAINING_CONFIG}/dataset_prepare.py

# Content of apertus-benchmarks/patches/v1-separate-async-fixes.patch, embedded
# here rather than read via a script-relative path: under sbatch, BASH_SOURCE[0]
# resolves to the spool-staged copy of this script, not its checkout location,
# so a script-relative read silently fails on every node (this is the exact
# failure mode that lost the old sitecustomize.py-based fallback patch in run
# 3129805 -- see Known hazards in CLAUDE.md). Applied via git apply on the baked
# v0.9.0 verl tree below, alongside the upstream PR patches -- two real, stable
# fixes for the V1 separate-async trainer (hybrid-rollout OOM, stale hybrid
# weight-sync call). This is the 2-hunk Apertus variant: the third fix
# (list_of_dict_to_tensordict shape-equality bug) is upstream verl PR #7661,
# fetched by the PR-patch loop below. The GLM v1-separate recipe still carries
# the 3-hunk example/patches/ version -- keep the shared hunks 1+2 in sync.
cat > "${TRAINING_CONFIG}/v1-separate-async-fixes.patch" <<- 'EOF'
# Local fixes for verl v0.9.0's V1 separate-async trainer, discovered debugging
# train-gsm8k-glm5.1-700B-v1-separate-async-megatron.sh and
# rl-bench-apertus-v1.5-70B-sglang-megatron-v1-separate-async.sh (see Known
# hazards and the Run log in CLAUDE.md for the full incident history). Re-diff
# against a newer verl ref if this stops applying.
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
# 2 & 3. PPOTrainerSeparateAsync.on_init_end and .switch_to_rollout
#    (verl/trainer/ppo/v1/trainer_separate_async.py): both drop their
#    self.checkpoint_manager.update_weights(...) call. That manager's backend is
#    forced to "naive" (trainer_base.py), whose update_weights() bypasses the
#    (correctly empty, thanks to fix 1) hybrid replica list entirely and instead
#    calls ray.get(actor_wg.update_weights(mode="naive")) on every actor rank --
#    each of which unconditionally does `await self.rollout.resume(tags=["weights"])`
#    when free_cache_engine is True (engine_workers.py). This recipe's real weight
#    sync is self.standalone_checkpoint_manager (nccl backend, on_step_end) and
#    never touches SGLang memory-occupation tags at all, so a resume of "weights"
#    here has no matching prior release -- SGLang's resume_memory_occupation does
#    a bare `offload_tags.remove("weights")`, and KeyError: 'weights' crashes every
#    standalone rollout TP rank.
#    - on_init_end: confirmed fixed in run 3134772 (no OOM, training reached step 0).
#    - switch_to_rollout: this call site is NOT exercised by on_init_end's fix --
#      current_mode starts at ROLLOUT (so on_validate_begin's `if current_mode ==
#      TRAINER` guard skips switch_to_rollout for any validation before the first
#      training sample), then flips to TRAINER after the first sample and NEVER
#      flips back during training (should_switch_to_rollout() is a hardcoded
#      `return False` stub in this verl version) -- so switch_to_rollout only ever
#      runs the first time a validation happens *after* training has started, i.e.
#      the first genuinely mid-run test_freq validation. Run 3261338 (2026-09-02):
#      45/46 steps completed cleanly (val_before_train's baseline eval worked fine,
#      since it never touches this code path), then crashed exactly here on the
#      step-46 (test_freq=46) validation switch -- traced via the real v0.9.0
#      source (base.py:CheckpointEngineManager.update_weights, engine_workers.py:
#      WorkerDict.update_weights, sglang_rollout.py:ServerAdapter.resume,
#      async_sglang_server.py's tags semantics) to confirm this exact mechanism,
#      not guessed. self.checkpoint_manager.resume_generation_replicas() and
#      add_replicas_to_balancer() right below are left untouched: both operate on
#      the (correctly empty) replica list via asyncio.gather(*[... for r in []]),
#      confirmed genuinely inert (unlike update_weights's naive-backend branch,
#      which bypasses that list). switch_to_trainer() needs no equivalent fix --
#      its two calls (abort_replicas/sleep_replicas) are replica-list-only, and it
#      already ran once (after the first sample) with no crash across every run.
#
# (A fourth fix that used to live here -- list_of_dict_to_tensordict's
# shape-equality heuristic in verl/utils/tensordict_utils.py, which silently
# stacked ragged per-sample tensors into a dense Tensor and broke downstream
# .offsets() calls -- is now upstream verl PR #7661, fetched + git-apply'd by the
# PR-patch loop below, so it is no longer carried here.)
#
# Originally shipped as sitecustomize.py runtime monkeypatches (fast to iterate on
# mid-debugging -- no hand-crafted diff needed while the exact fix was still
# changing), converted to this source patch once run 3149736 confirmed the first
# two were correct and stable: same reasoning as the Apertus benchmark's
# equivalent conversion (apertus-benchmarks/patches/sglang-apertus1p5-local-fixes.patch)
# -- a runtime monkeypatch is great for fast iteration but not the form a stable
# fix should end up in. A plain source patch is one less moving part (no
# sys.meta_path machinery, no PYTHONPATH staging, no import-timing dependency)
# and the actual behavior is just readable in the file.
diff --git a/verl/trainer/ppo/v1/trainer_separate_async.py b/verl/trainer/ppo/v1/trainer_separate_async.py
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
@@ -179,7 +185,16 @@ class PPOTrainerSeparateAsync(PPOTrainer):

     def switch_to_rollout(self):
         # TODO: disable auto offload in config and offload according to the switch strategy
-        self.checkpoint_manager.update_weights(self.global_steps)
+        # self.checkpoint_manager.update_weights(...) is skipped for the same reason
+        # as on_init_end above: backend="naive" bypasses the (empty) hybrid replica
+        # list and pushes mode="naive" onto every actor rank, which unconditionally
+        # calls rollout.resume(tags=["weights"]) -- resuming a tag never released,
+        # since this recipe's real weight sync (standalone_checkpoint_manager, nccl
+        # backend, on_step_end) never touches memory-occupation tags at all. Left
+        # unpatched, this crashes SGLang's resume_memory_occupation with
+        # KeyError: 'weights' the first time switch_to_rollout runs -- which, since
+        # should_switch_to_rollout() is a hardcoded False stub, is only ever at
+        # validation (run 3261338: 45/46 steps clean, crashed here at the final val).
         self.checkpoint_manager.resume_generation_replicas()
         self.add_replicas_to_balancer()
         self.current_mode = HybridEngineMode.ROLLOUT
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

# apertus-benchmarks/patches/grad-sync-empty-cache.patch, embedded (same reason). Fix for the
# batch-dependent "NCCL Error 1: unhandled cuda error" in the DP gradient reduce-scatter at the
# 12288-token thinking cap (runs 3316163 / 3326537): empty_cache() before finalize_model_grads so
# NCCL can allocate its collective buffers outside the PyTorch cache. See the patch header.
cat > "${TRAINING_CONFIG}/grad-sync-empty-cache.patch" <<- 'EOF'
# [CSCS, 2026-09-08] Return PyTorch's cached-but-free CUDA blocks to the driver right before the
# Megatron DP gradient reduce-scatter (finalize_model_grads), by wrapping the hook verl installs
# in verl/utils/megatron_utils.py:register_megatron_training_hooks.
#
# Found on rl-bench-apertus-v1.5-70B-sglang-megatron-v1-separate-async.sh with DAPO-Math at the
# 12288-token thinking cap (runs 3316163 and 3326537): step 1 dies with
#   RuntimeError: NCCL Error 1: unhandled cuda error
#   finalize_model_grads -> finish_grad_sync -> start_grad_sync -> _coalescing_manager
# while an identical configuration (run 3317733) completed 2 steps -- a batch-dependent, marginal
# condition. At the step peak the trainer had ~60 GB allocated but ~86 GB RESERVED by the caching
# allocator; after the backward pass the activations are freed but the cache keeps holding the
# high-water mark, and NCCL's own CUDA allocations for the reduce-scatter happen outside that
# cache -> cudaMalloc fails on a 95 GB GPU even though the live tensors fit. Same mechanism and
# same remedy as the DeepSeek-V3 recipe's unconditional empty_cache() before optimizer.step().
# Numerically a no-op; cost is one empty_cache() per optimizer step (tens of ms).
#
# Generated from a real git worktree at verl v0.9.0; git apply --check / py_compile /
# --reverse --check verified. Touches only verl/utils/megatron_utils.py (no overlap with the other
# patches this recipe applies).
diff --git a/verl/utils/megatron_utils.py b/verl/utils/megatron_utils.py
index 0358497c..016e8f2d 100644
--- a/verl/utils/megatron_utils.py
+++ b/verl/utils/megatron_utils.py
@@ -1689,7 +1689,18 @@ def register_megatron_training_hooks(model: list[torch.nn.Module], optimizer):
     for one_model in model:
         config = get_model_config(one_model)
         config.grad_scale_func = optimizer.scale_loss
-        config.finalize_model_grads_func = finalize_model_grads
+
+        def _finalize_model_grads_with_empty_cache(*args, **kwargs):
+            # Return PyTorch's cached-but-free blocks to CUDA before the DP gradient
+            # reduce-scatter: after the backward pass the caching allocator still holds
+            # the activation high-water mark, and NCCL's own CUDA allocations for the
+            # collective happen outside that cache. On long-sequence RL steps this
+            # otherwise fails with "NCCL Error 1: unhandled cuda error" even though the
+            # step's live tensors fit. Numerically a no-op.
+            get_torch_device().empty_cache()
+            return finalize_model_grads(*args, **kwargs)
+
+        config.finalize_model_grads_func = _finalize_model_grads_with_empty_cache
 
         overlap_param_gather = getattr(optimizer.config, "overlap_param_gather", False)
         overlap_grad_reduce = getattr(one_model.ddp_config, "overlap_grad_reduce", False)
EOF
sbcast -f ${TRAINING_CONFIG}/grad-sync-empty-cache.patch ${TRAINING_CONFIG}/grad-sync-empty-cache.patch

# ══════════════════════════════════════════════════════════════════════════
# Add Apertus 1.5 support.
# ══════════════════════════════════════════════════════════════════════════

# Transformers==5.8.1 (pinned in the image) has no apertus1p5
# support. The swiss-ai fork does, but it has to be built from source.
export SWISS_AI_TRANSFORMERS_SHA=986d6dfa97cc6675a65f6d052e41ab316b0649eb
export SWISS_AI_WHEEL_DIR=${TRAINING_HOME}/wheels
mkdir -p ${SWISS_AI_WHEEL_DIR}
if ! ls ${SWISS_AI_WHEEL_DIR}/transformers-*.whl >/dev/null 2>&1 \
    || ! ls ${SWISS_AI_WHEEL_DIR}/safetensors-*.whl >/dev/null 2>&1 \
    || [ "$(cat ${SWISS_AI_WHEEL_DIR}/transformers.sha 2>/dev/null)" != "${SWISS_AI_TRANSFORMERS_SHA}" ]; then
    echo "Building swiss-ai/transformers@${SWISS_AI_TRANSFORMERS_SHA} + safetensors wheels..."
    rm -f ${SWISS_AI_WHEEL_DIR}/transformers-*.whl
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
    echo "${SWISS_AI_TRANSFORMERS_SHA}" > ${SWISS_AI_WHEEL_DIR}/transformers.sha
else
    echo "swiss-ai/transformers and safetensors wheels already present for ${SWISS_AI_TRANSFORMERS_SHA}, skipping build."
fi

# SGLang has no native apertus1p5 model
export SGLANG_APERTUS_PATCH_URL="https://github.com/sgl-project/sglang/pull/32979.diff"
curl -sfL "${SGLANG_APERTUS_PATCH_URL}" -o ${TRAINING_CONFIG}/sglang-apertus1p5-full.diff \
    || { echo "FATAL: could not download sglang PR #32979 diff"; exit 1; }
[ -s ${TRAINING_CONFIG}/sglang-apertus1p5-full.diff ] \
    || { echo "FATAL: sglang PR #32979 diff is empty"; exit 1; }
# This is an open, unmerged, still-actively-evolving PR (see CLAUDE.md) -- its
# diff is re-fetched fresh on every submission, not pinned to a commit, so its
# file list grows over time as the PR is updated upstream. Run 3152782: between
# the previous submission and this one the PR grew from 6 files to 10 (adding
# docs/test files plus a new hunk in detokenizer_manager.py, a tool-call-parser
# trim for a "apertus2509" parser this benchmark never enables), and that new
# hunk failed to apply against the image's pinned SGLang version (the exact
# same "context-sensitive against whatever SGLang version the image happens to
# have" issue already known for base_processor.py) -- a fatal-by-default
# blocklist (keep everything under python/sglang/ except a fixed set of known
# problem files) breaks every time the PR grows a new such file. Flipped to an
# allowlist instead: only the files actually confirmed load-bearing for the
# apertus1p5 fix (point 3 in CLAUDE.md) are fatal-if-they-fail; everything else
# the PR touches (test/docs excluded entirely; any other python/sglang/ file,
# currently base_processor.py and detokenizer_manager.py) is best-effort --
# self-adapting to future PR drift instead of needing another manual fix each
# time the PR changes shape.
awk '
    /^diff --git a\// { keep = 0 }
    /^diff --git a\/python\/sglang\/srt\/configs\/qwen3_asr\.py/ { keep = 1 }
    /^diff --git a\/python\/sglang\/srt\/models\/apertus\.py/ { keep = 1 }
    /^diff --git a\/python\/sglang\/srt\/models\/apertus_mm\.py/ { keep = 1 }
    /^diff --git a\/python\/sglang\/srt\/multimodal\/processors\/apertus_mm\.py/ { keep = 1 }
    keep { print }
' ${TRAINING_CONFIG}/sglang-apertus1p5-full.diff > ${TRAINING_CONFIG}/sglang-apertus1p5.diff
[ -s ${TRAINING_CONFIG}/sglang-apertus1p5.diff ] \
    || { echo "FATAL: filtered sglang PR #32979 diff is empty"; exit 1; }
sbcast -f ${TRAINING_CONFIG}/sglang-apertus1p5.diff ${TRAINING_CONFIG}/sglang-apertus1p5.diff

awk '
    /^diff --git a\// { keep = 1 }
    /^diff --git a\/python\/sglang\/srt\/configs\/qwen3_asr\.py/ { keep = 0 }
    /^diff --git a\/python\/sglang\/srt\/models\/apertus\.py/ { keep = 0 }
    /^diff --git a\/python\/sglang\/srt\/models\/apertus_mm\.py/ { keep = 0 }
    /^diff --git a\/python\/sglang\/srt\/multimodal\/processors\/apertus_mm\.py/ { keep = 0 }
    /^diff --git a\/test\// { keep = 0 }
    /^diff --git a\/docs\// { keep = 0 }
    /^diff --git a\/docs_new\// { keep = 0 }
    keep { print }
' ${TRAINING_CONFIG}/sglang-apertus1p5-full.diff > ${TRAINING_CONFIG}/sglang-apertus1p5-best-effort.diff
sbcast -f ${TRAINING_CONFIG}/sglang-apertus1p5-best-effort.diff ${TRAINING_CONFIG}/sglang-apertus1p5-best-effort.diff

# Fixes
cat > "${TRAINING_CONFIG}/sglang-apertus1p5-local-fixes.diff" <<- 'EOF'
# Local fixes on top of sgl-project/sglang#32979 (still open/unmerged).
# Apply *after* that PR's patch — this diff is against the new file it adds
# (python/sglang/srt/models/apertus_mm.py), not upstream SGLang.
#
# 1. _init_component_model calls component_config.to_dict() unconditionally,
#    assuming a PreTrainedConfig-like object; against the transformers
#    commit this script installs it's already a plain dict. Fixed to accept
#    either shape.
# 2. load_component_weight raises when a checkpoint key claims to be a
#    vision/audio tensor but the tower model has no matching parameter —
#    the checkpoint's vision-tokenizer weight names don't line up with the
#    structure this PR builds. Harmless to skip: this benchmark is
#    text-only GSM8K and never calls get_image_feature/get_audio_feature,
#    so vision/audio tower weights never need to be numerically correct,
#    only present so the model instantiates. The language-model weight path
#    (everything load_component_weight returns False for) is untouched.
#
# Re-diff against the PR's current head if it moves and this stops applying.
--- a/python/sglang/srt/models/apertus_mm.py
+++ b/python/sglang/srt/models/apertus_mm.py
@@ -50,7 +50,11 @@
     component_config: Any,
     model_cls: type[nn.Module] | None = None,
 ) -> nn.Module:
-    config_dict = component_config.to_dict()
+    config_dict = (
+        component_config.to_dict()
+        if hasattr(component_config, "to_dict")
+        else dict(component_config)
+    )
     config = AutoConfig.for_model(config_dict.pop("model_type"), **config_dict)
     return AutoModel.from_config(config) if model_cls is None else model_cls(config)

@@ -301,9 +305,16 @@

             component_tensor = component_tensors.get(name)
             if component_tensor is None:
-                raise ValueError(
-                    f"No vision/audio tensor matches checkpoint key: {name}"
+                # Vision/audio tower weights never need to be numerically
+                # correct for a text-only benchmark that never calls
+                # get_image_feature/get_audio_feature -- skip instead of
+                # aborting the whole load.
+                print(
+                    "[apertus1p5-local-fixes] WARNING: skipping unmatched "
+                    f"vision/audio checkpoint key {name}",
+                    flush=True,
                 )
+                return True
             weight_loader = getattr(
                 component_tensor, "weight_loader", default_weight_loader
             )
EOF
sbcast -f ${TRAINING_CONFIG}/sglang-apertus1p5-local-fixes.diff ${TRAINING_CONFIG}/sglang-apertus1p5-local-fixes.diff

# ══════════════════════════════════════════════════════════════════════════
# Group 2: Megatron support for Apertus 1.5, applied at runtime as two GitHub
# `compare` diffs (upstream release tag ... theely fork branch) against the STOCK
# image megatron-core 0.19.0 / megatron-bridge 0.6.1 -- no fork wheels, no
# runtime clone, no wheel build. The apertus code is identical to what runs
# 3240861/3243271/3243467 already built and ran against stock 0.19.0/0.6.1 (they
# trained degenerately only because of the reward function, since fixed -- see
# CLAUDE.md "The zero-reward chain").
#
# `<tag>...<branch>.diff` is GitHub's compare endpoint: `...` gives the diff from
# the merge-base, i.e. exactly the fork branch's delta over the release tag. The
# fork branch is a single commit off the tag; no PR / PR-number / base branch is
# needed. Regenerate it with patches/build-apertus1p5-megatron-forks.sh (which
# also prints the head SHAs -- pin those instead of the branch name for
# reproducibility if the branch is ever force-pushed).
#
#   Megatron-LM  (NVIDIA:core_v0.19.0 5be9626709af): 2 hunks -- transformer/mlp.py
#     module activation_func, distributed/finalize_model_grads.py xIELU TP grad-sum
#   Megatron-Bridge (NVIDIA-NeMo:v0.6.0 51885cf132b2 -- no v0.6.1 tag exists,
#     0.6.0->0.6.1 is a patch release with no models/ API change): add
#     models/apertus/ + models/apertus1p5/ (Apertus1p5Bridge) + models/__init__.py
#     registration + safe_config_loader.py flock -> contextlib.nullcontext
export MEGATRON_LM_FORK_REF="9898eb2be164641600f197368f765a904ad73812"      # theely/Megatron-LM apertus1p5-support head
export MEGATRON_BRIDGE_FORK_REF="53bea08c8417627d5e83af6c9c16163a47e3a8b1"  # theely/Megatron-Bridge apertus1p5-support head
# Fetch both diffs once on the batch host (never per-node -- "Never fetch
# per-node from the internet inside the srun", CLAUDE.md); -f so an HTTP error
# page is a failure not a no-op patch; assert non-empty; sbcast to every node.
curl -sfL "https://github.com/theely/Megatron-LM/compare/NVIDIA:core_v0.19.0...theely:${MEGATRON_LM_FORK_REF}.diff" \
    -o ${TRAINING_CONFIG}/megatron-lm-apertus1p5.diff \
    || { echo "FATAL: could not download theely/Megatron-LM ${MEGATRON_LM_FORK_REF} compare diff"; exit 1; }
[ -s ${TRAINING_CONFIG}/megatron-lm-apertus1p5.diff ] \
    || { echo "FATAL: theely/Megatron-LM compare diff is empty"; exit 1; }
curl -sfL "https://github.com/theely/Megatron-Bridge/compare/NVIDIA-NeMo:v0.6.0...theely:${MEGATRON_BRIDGE_FORK_REF}.diff" \
    -o ${TRAINING_CONFIG}/megatron-bridge-apertus1p5.diff \
    || { echo "FATAL: could not download theely/Megatron-Bridge ${MEGATRON_BRIDGE_FORK_REF} compare diff"; exit 1; }
[ -s ${TRAINING_CONFIG}/megatron-bridge-apertus1p5.diff ] \
    || { echo "FATAL: theely/Megatron-Bridge compare diff is empty"; exit 1; }
sbcast -f ${TRAINING_CONFIG}/megatron-lm-apertus1p5.diff     ${TRAINING_CONFIG}/megatron-lm-apertus1p5.diff
sbcast -f ${TRAINING_CONFIG}/megatron-bridge-apertus1p5.diff ${TRAINING_CONFIG}/megatron-bridge-apertus1p5.diff
echo "Fetched theely/Megatron-{LM,Bridge} apertus1p5 compare diffs (${MEGATRON_LM_FORK_REF} / ${MEGATRON_BRIDGE_FORK_REF})."

# apertus_bridge.py's MCoreXIELU (reused as-is by Apertus1p5Bridge) wraps xielu
# and unconditionally requires the optional CUDA xielu extension
# (github.com/rubber-duck-debug/xielu), raising
# "RuntimeError: CUDA xIELU is required. Install rubber-duck-debug/xielu." at
# TransformerLayer construction time if it is not importable (run 3171309). Not
# in any megatron package or the image -- built once here and installed on every
# node, gated by its own commit-SHA marker.
export XIELU_SHA=2a55f6b9efa64954bf173e63297a2b2f99741b69
export XIELU_WHEEL_DIR=${TRAINING_HOME}/wheels
mkdir -p ${XIELU_WHEEL_DIR}
if ! ls ${XIELU_WHEEL_DIR}/xielu-*.whl >/dev/null 2>&1 \
    || [ "$(cat ${XIELU_WHEEL_DIR}/xielu.sha 2>/dev/null)" != "${XIELU_SHA}" ]; then
    echo "Building rubber-duck-debug/xielu@${XIELU_SHA} wheel..."
    srun --mpi=pmix --network=disable_rdzv_get -N 1 --ntasks=1 -u \
        --environment="${TRAINING_CONFIG}/env.toml" \
        --container-writable bash -c '
        set -e
        export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
        rm -rf /tmp/xielu-src
        curl -sfL "https://github.com/rubber-duck-debug/xielu/archive/${XIELU_SHA}.tar.gz" \
            -o /tmp/xielu-src.tar.gz
        [ -s /tmp/xielu-src.tar.gz ]
        mkdir -p /tmp/xielu-src
        tar xzf /tmp/xielu-src.tar.gz -C /tmp/xielu-src --strip-components=1
        rm -f ${XIELU_WHEEL_DIR}/xielu-*.whl
        pip wheel --no-build-isolation --no-deps -w /tmp/xielu-wheel /tmp/xielu-src
        cp /tmp/xielu-wheel/xielu-*.whl ${XIELU_WHEEL_DIR}/
    '
    ls ${XIELU_WHEEL_DIR}/xielu-*.whl >/dev/null 2>&1 \
        || { echo "FATAL: rubber-duck-debug/xielu wheel build failed"; exit 1; }
    echo "${XIELU_SHA}" > ${XIELU_WHEEL_DIR}/xielu.sha
else
    echo "rubber-duck-debug/xielu wheel already present and up to date (${XIELU_SHA}), skipping build."
fi


# Download model (skip if already present, or if MODEL_CHECKPOINT_PATH points at
# an existing local checkpoint to load from instead -- e.g. an SFT checkpoint).
if [ -n "${MODEL_CHECKPOINT_PATH}" ]; then
    if [ ! -d "${MODEL_CHECKPOINT_PATH}" ]; then
        echo "FATAL: MODEL_CHECKPOINT_PATH=${MODEL_CHECKPOINT_PATH} does not exist"
        exit 1
    fi
    echo "Using model checkpoint at ${MODEL_CHECKPOINT_PATH}, skipping HF download."
elif [ ! -d "${TRAINING_HOME}/models/${MODEL_NAME}" ]; then
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

# Prepare the ${BENCHMARK} dataset. ${TRAINING_HOME}/data/${BENCHMARK} persists on Lustre across
# submissions; a .prompt.version marker forces a rebuild whenever dataset_prepare.py's
# SYSTEM_PROMPT changes (bump DATASET_PROMPT_VERSION in the same edit as any prompt change --
# the old plain "file exists" guard silently served a stale parquet).
case "${BENCHMARK}-${ANSWER_MARKER}" in
    gsm8k-bracket)     export DATASET_PROMPT_VERSION="v2-triple-bracket" ;;   # unchanged: cached gsm8k parquets stay valid
    gsm8k-boxed)       export DATASET_PROMPT_VERSION="v3-boxed" ;;
    dapo-math-bracket) export DATASET_PROMPT_VERSION="v3-bracket-train-boxed-aime" ;;  # 2026-09-08: AIME rows now use the boxed prompt -> rebuild
    dapo-math-boxed)   export DATASET_PROMPT_VERSION="v3-boxed-train-boxed-aime" ;;
    *) echo "FATAL: unsupported BENCHMARK/ANSWER_MARKER combination ${BENCHMARK}/${ANSWER_MARKER}"; exit 1 ;;
esac
export DATASET_DIR="${TRAINING_HOME}/data/${BENCHMARK}"
if [ ! -f "${DATASET_DIR}/train.parquet" ] \
    || [ "$(cat ${DATASET_DIR}/.prompt.version 2>/dev/null)" != "${DATASET_PROMPT_VERSION}" ]; then
    echo "Preparing ${BENCHMARK} dataset (SYSTEM_PROMPT ${DATASET_PROMPT_VERSION})..."
    srun --mpi=pmix --network=disable_rdzv_get -N 1 --ntasks=1 -u \
        --environment="${TRAINING_CONFIG}/env.toml" \
        --container-writable bash -c '
        set -e
        rm -f ${DATASET_DIR}/train.parquet ${DATASET_DIR}/test.parquet
        # BENCHMARK selects the source (gsm8k | dapo-math); cached raw downloads are reused when present
        python ${TRAINING_CONFIG}/dataset_prepare.py
        echo "${DATASET_PROMPT_VERSION}" > ${DATASET_DIR}/.prompt.version
    '
    [ -f "${DATASET_DIR}/train.parquet" ] \
        || { echo "FATAL: ${BENCHMARK} dataset preparation failed"; exit 1; }
else
    echo "Dataset ${BENCHMARK} already present for SYSTEM_PROMPT ${DATASET_PROMPT_VERSION}, skipping preparation."
fi


export MASTER_NODE=$(hostname)
export MASTER_NODE_IP=$(hostname -i)
export PORT=6382
export RAY_ADDRESS="${MASTER_NODE_IP}:${PORT}"

export WANDB_API_KEY=$(cat /users/${USER}/.wandb_api_key)
export WANDB_SILENT=true # Suppress WandB logs

export RAY_memory_usage_threshold=0.99

# Fetch the upstream PR patches once here and sbcast them to every node, instead
# of curl-ing them from inside the srun (run 3124273: per-node fetches gave a
# mixed cluster where only ~45/80 nodes carried a patch). -f makes an HTTP error
# page a hard failure instead of a no-op "patch".
#   PR #7421: DSA / mcore >= 0.16.2 compat (harmless here -- apertus1p5 is not DSA).
#   PR #7422: preserve load_format=dummy in the disaggregated SGLang rollout --
#             load-bearing for separate-async (v0.9.0 async_sglang_server.py flips
#             dummy -> auto for every non-hybrid replica, i.e. the standalone rollout).
#   PR #7423: fix the NCCL deadlock in async disaggregated weight sync.
#   PR #7661: list_of_dict_to_tensordict -- build nested tensors unconditionally
#             (the 3rd v1-separate-async fix, now upstream; applies clean to the
#             baked v0.9.0 tree). Superseded the tensordict hunk of
#             v1-separate-async-fixes.patch, which no longer carries it.
for pr in 7421 7422 7423 7661; do
    curl -sfL "https://github.com/verl-project/verl/pull/${pr}.patch" -o "${TRAINING_CONFIG}/${pr}.patch" \
        || { echo "FATAL: could not download PR #${pr}"; exit 1; }
    [ -s "${TRAINING_CONFIG}/${pr}.patch" ] \
        || { echo "FATAL: PR #${pr} patch is empty"; exit 1; }
    sbcast -f "${TRAINING_CONFIG}/${pr}.patch" "${TRAINING_CONFIG}/${pr}.patch"
done



srun --mpi=pmix --network=disable_rdzv_get -N ${SLURM_JOB_NUM_NODES} --ntasks-per-node=1 -u \
    --environment="${TRAINING_CONFIG}/env.toml" \
    --container-writable bash -c '


# verl is baked into the image at v0.9.0 (Containerfile VERL_REF=v0.9.0),
# editable-installed from /workspace/verl -- no runtime checkout. The fully-async
# recipe reset to theely/verl Fix-fsdp-model-loading-on-async is dropped: that
# fork branch is a single-file FSDP2-only change (unused by the Megatron trainer)
# and a hard reset would clobber the baked v0.9.0 tree.
git -C /workspace/verl --no-pager log --oneline -1 || true

# Image-version smoke test (diagnostic only, non-fatal -- a wrong image tag shows
# here in the first ~30 s instead of via a downstream crash). NOTE: the megatron
# versions printed here are the stock image ones (0.19.0 / 0.6.1); Group 2 below
# patches Apertus1p5Bridge into them via the two PR diffs, not a version change.
python3 -c "
import importlib.metadata as _m
for _p in (\"verl\", \"TransferQueue\", \"megatron-core\", \"megatron-bridge\", \"sglang\", \"transformers\", \"flashinfer-python\"):
    try:
        print(f\"  {_p}: {_m.version(_p)}\")
    except Exception as _e:
        print(f\"  {_p}: <not installed> ({_e})\")
"

# Apply the upstream PR patches + the V1 separate-async local fixes on the baked
# v0.9.0 tree (sbcast to ${TRAINING_CONFIG} before the srun). None are in v0.9.0;
# all apply cleanly to the tag. A patch that neither applies nor is already
# present is fatal -- a cluster where only some ranks carry a patch is worse than
# one that carries none (run 3124273).
for p in "${TRAINING_CONFIG}/7421.patch" "${TRAINING_CONFIG}/7422.patch" "${TRAINING_CONFIG}/7423.patch" "${TRAINING_CONFIG}/7661.patch" "${TRAINING_CONFIG}/v1-separate-async-fixes.patch" "${TRAINING_CONFIG}/grad-sync-empty-cache.patch"; do
    if git -C /workspace/verl apply --check "$p" 2>/dev/null; then
        git -C /workspace/verl apply "$p" && echo "Applied $(basename "$p") on $(hostname)"
    elif git -C /workspace/verl apply --reverse --check "$p" 2>/dev/null; then
        echo "$(basename "$p") already present on $(hostname), skipping"
    else
        echo "FATAL: $(basename "$p") neither applies nor is already present on $(hostname)"
        exit 1
    fi
done

# Fix: Apertus1p5 text-only forward corruption (root cause of the
# split_with_sizes crash chased through runs 3174079/3183472; see CLAUDE.md).
# verl/workers/engine/megatron/transformer_impl.py always passes
# vision_model=hasattr(self.model_config.hf_config, "vision_config") into
# gptmodel_forward_model_engine (two call sites). Apertus1p5Config always
# carries a vision_config attribute (it is a genuinely multimodal
# architecture), so this is True on every forward call regardless of whether
# the current batch has any image content. When vision_model=True,
# verl/models/mcore/model_forward.py unconditionally overwrites the correctly
# THD-packed input_ids_rmpad with build_vlm_attn_mask_thd(input_ids,
# pad_token_id)s output -- despite the "_thd" in its name, that function calls
# input_ids.to_padded_tensor(...), producing a dense [batch_size, max_seqlen]
# tensor -- while packed_seq_params (built moments earlier from the correctly
# packed data) is left unchanged and still describes the true packed cu_seqlens
# totals. That mismatched pair (BSHD-shaped tensor + THD-format
# packed_seq_params) survives all the way to the rotary-embedding call, where
# megatron-core tries to torch.split() the small BSHD tensor using cu_seqlens
# sized for the full packed batch -- RuntimeError: split_with_sizes expects
# split_sizes to sum exactly to <N> ... but got split_sizes=[...summing to the
# full packed total]. Confirmed via run 3183472 [DIAG-ROPE] diagnostic (since
# removed): the crashing tensor shape (e.g. (623, 43, 8192)) has batch dim 43
# exactly equal to the sequence count from cu_seqlens and seq dim 623 matching
# a plausible padded max-sequence-length -- exactly what
# word_embeddings([43, 623]).transpose(0, 1) produces, confirming input_ids
# really arrived batched/padded rather than packed. This benchmark is
# text-only GSM8K -- multi_modal_inputs is always empty here -- so forcing
# vision_model=False is correct for this recipe specifically (a real image
# input would need the opposite fix: keeping packed_seq_params in sync with
# whatever build_vlm_attn_mask_thd produces -- out of scope here). Two call
# sites confirmed identical text in the verl-project v0.9.0 tag (baked into this
# image) -- applied after the git-apply patch loop above so it stacks on top.
# Located via importlib.util.find_spec rather than a hardcoded path, same
# convention as the megatron-bridge filelock patch below.
python3 -c "
import importlib.util
spec = importlib.util.find_spec(\"verl.workers.engine.megatron.transformer_impl\")
if not spec:
    raise SystemExit(\"FATAL: verl.workers.engine.megatron.transformer_impl not found\")
p = spec.origin
with open(p) as f:
    lines = f.readlines()
marker = \"forced off for this text-only Apertus1p5 recipe\"
n_marker = sum(1 for l in lines if marker in l)
if n_marker == 2:
    print(f\"vision_model already patched (2 markers found) in {p}, skipping\")
elif n_marker != 0:
    raise SystemExit(f\"FATAL: expected 0 or 2 already-patched vision_model markers in {p}, found {n_marker}\")
else:
    q = chr(34)
    target = \"vision_model=hasattr(self.model_config.hf_config, \" + q + \"vision_config\" + q + \"),\"
    new_lines = []
    n_patched = 0
    for line in lines:
        if line.strip() == target:
            indent = len(line) - len(line.lstrip())
            new_lines.append(\" \" * indent + \"vision_model=False,  # \" + marker + \", see CLAUDE.md run 3183472\n\")
            n_patched += 1
        else:
            new_lines.append(line)
    if n_patched != 2:
        raise SystemExit(f\"FATAL: expected 2 vision_model=hasattr(...) sites in {p}, found {n_patched}\")
    with open(p, \"w\") as f:
        f.writelines(new_lines)
    print(f\"Patched {n_patched} vision_model site(s) in {p} (forced False)\")
"

# Install Transformers and Safetensors from the swiss-ai wheels
pip install --no-deps ${SWISS_AI_WHEEL_DIR}/transformers-*.whl ${SWISS_AI_WHEEL_DIR}/safetensors-*.whl
python3 -c "import transformers, safetensors; print(\"transformers:\", transformers.__version__, transformers.__file__); print(\"safetensors:\", safetensors.__version__)"

# Install the rubber-duck-debug/xielu CUDA extension (built once above) --
# required by MCoreXIELU (megatron.bridge.models.apertus.apertus_bridge,
# reused by the new Apertus1p5Bridge) or actor_init_model() raises
# "RuntimeError: CUDA xIELU is required" the moment it builds a TransformerLayer.
pip install --no-deps ${XIELU_WHEEL_DIR}/xielu-*.whl \
    || { echo "FATAL: rubber-duck-debug/xielu wheel install failed"; exit 1; }
python3 -c "import xielu.ops; print(\"xielu:\", xielu.__file__)"

# Apply the two Apertus-1.5 Megatron support compare diffs onto the STOCK image
# megatron-core (0.19.0) and megatron-bridge (0.6.1), in site-packages. Every
# main-srun node gets a fresh container from the image (pyxis --container-writable
# edits are per-job, discarded on exit), so this always patches pristine
# site-packages -- no shared-checkout race (the run-3149726 hazard the fork-wheel
# build existed to avoid), same "fresh + apply-or-FATAL" as the sglang patches.
#   Megatron-LM diff      -> megatron/core/*      (repo path == installed path, -p1)
#   Megatron-Bridge diff  -> src/megatron/bridge/* (installed drops src/, -p2)
# --fuzz=3 on the bridge diff absorbs the harmless v0.6.0 (compare base) -> 0.6.1
# (installed) context drift; the new-file hunks (models/apertus*, ...) are
# version-independent.
# (megatron is a namespace package with no __file__; derive site-packages from
# the real megatron.core __init__.py instead.)
SITE_PKG=$(python3 -c "import os, megatron.core as m; print(os.path.dirname(os.path.dirname(os.path.dirname(m.__file__))))")
patch -p1 -d "${SITE_PKG}" < ${TRAINING_CONFIG}/megatron-lm-apertus1p5.diff \
    || { echo "FATAL: theely/Megatron-LM apertus1p5 compare diff failed to apply"; exit 1; }
patch -p2 -d "${SITE_PKG}" --fuzz=3 < ${TRAINING_CONFIG}/megatron-bridge-apertus1p5.diff \
    || { echo "FATAL: theely/Megatron-Bridge apertus1p5 compare diff failed to apply"; exit 1; }
python3 -c "import megatron.core, megatron.bridge; from megatron.bridge.models import Apertus1p5Bridge; print(\"megatron.core:\", megatron.core.__file__, \"| Apertus1p5Bridge OK\")" \
    || { echo "FATAL: Apertus1p5Bridge not importable after applying the Megatron compare diffs"; exit 1; }

# Apply SGLang partch for apertus1p5 model.
patch -p2 -d /usr/local/lib/python3.12/dist-packages < ${TRAINING_CONFIG}/sglang-apertus1p5.diff \
    || { echo "FATAL: sglang PR #32979 patch failed to apply"; exit 1; }
patch -p2 -d /usr/local/lib/python3.12/dist-packages < ${TRAINING_CONFIG}/sglang-apertus1p5-best-effort.diff \
    || echo "WARNING: one or more best-effort sglang PR #32979 hunks did not apply (context-sensitive against the image SGLang version, or a feature this benchmark never enables, e.g. base_processor.py audio-input path or detokenizer_manager.py apertus2509 tool-call-parser trim) — continuing"
patch -p2 -d /usr/local/lib/python3.12/dist-packages < ${TRAINING_CONFIG}/sglang-apertus1p5-local-fixes.diff \
    || { echo "FATAL: local apertus_mm.py fixes failed to apply"; exit 1; }


# Redirect all JIT/kernel caches to local tmpfs — Lustre does not support file locking
export FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer_${SLURM_JOB_ID}
mkdir -p $FLASHINFER_WORKSPACE_BASE
# Triton JIT cache off Lustre (2026-09-07, run 3317266): use_fused_kernels compiles the verl fused
# linear-cross-entropy Triton kernel at the first actor forward; the default ~/.triton cache on
# Lustre fails with OSError Errno 116 Stale file handle (the same flock hazard as flashinfer).
# Same redirect the DeepSeek-V3 / GLM-5.1 recipes use.
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

# Required for Megatron communication/computation overlapping.
export CUDA_DEVICE_MAX_CONNECTIONS=1

# NCCL flight recorder -- so a weight-sync collective timeout (the V1
# separate-async trainer->rollout sync is the prime hang suspect) dumps a
# per-rank trace instead of a black box. One file per rank under /tmp.
export TORCH_NCCL_TRACE_BUFFER_SIZE=20000
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE=/tmp/nccl_flightrecorder_${SLURM_JOB_ID}_rank


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
