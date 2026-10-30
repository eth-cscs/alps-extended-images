#!/bin/bash

#SBATCH --nodes=128
#SBATCH --account=csstaff
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=5:00:00

# ─────────────────────────────────────────────────────────────────────────────
# GLM-5.1 (~744B MoE, DSA attention) GRPO on GSM8K -- verl V1 trainer in separate_async mode,
# Megatron actor (TP=4 / PP=3 / EP=8 / DP=5 on 120 nodes) + one standalone SGLang TP=32
# rollout replica (8 nodes), delta_sharded weight sync, R3 router replay. The DeepSeek-V3
# recipe (train-gsm8k-deepseek-v3-671B-v1-separate-async-megatron.sh) is derived from this
# one; the two share everything except the model-specific knobs (marked "GLM-5.1" /
# "DeepSeek-V3").
#
# Status: the last full GLM-5.1 validation (20 clean R3+THD steps, run 3217439) predates the
# delta_sharded / CPU-offload-optimizer settings below; the delta_sharded steady sync was only
# ever completed on DeepSeek-V3 (run 3309430) after the verl PR #7777/#7778 fixes, which are
# applied here too (2026-09-07) but not yet re-validated on GLM-5.1. History: the "Debugging
# train-gsm8k-glm5.1-700B-v1-separate-async-megatron.sh" section of .claude/CLAUDE.md.
#
# GLM-5.1 specifics, in one place:
#   * model_type glm_moe_dsa: DSA attention (verl PR #7421 is load-bearing), custom tokenizer
#     (trust_remote_code: True), 78 layers -> PP=3 is 26 layers per stage, EP=8 (EP=4 leaves
#     experts 64-127 unmapped in megatron-bridge). Per-rank params 9.24B.
#   * megatron-core 0.19.0 / megatron-bridge 0.6.1 / sglang 0.5.16 / flashinfer 0.6.14 are baked
#     into the image (older flashinfer hung the TP=32 MLA/DSA decode plan()).
#   * Checkpoint saving disabled (save_freq: -1): the HF export gathers the full model through
#     rank-0 HOST RAM (302 GB on DeepSeek-V3, run 3279810; GLM-5.1 is bigger). Use
#     actor.megatron.use_dist_checkpointing: True (untested) for a run that needs checkpoints.
#
# verl V1 separate_async notes (vs the fully-async recipe this descends from):
#   * data.train_batch_size == parameter_sync_step * ppo_mini_batch_size (asserted by verl).
#   * old_log_probs come from rollout.calculate_log_probs + algorithm.rollout_correction.bypass_mode.
#   * PPOTrainer._setup() also builds HYBRID rollout replicas on the training GPUs (not
#     configurable) -- v1-separate-async-fixes.patch below no-ops them; without it init OOMs.
# ─────────────────────────────────────────────────────────────────────────────

export VERL_IMAGE="jfrog.svc.cscs.ch/docker-group-csstaff/alps-images/verl-cuda:alps7-dev-621fa40275c4f036" #alps7-dev-a9f9e56471c0574e image with update dependencies #alps7-dev-0f334b540ccc7034 image with megatron

export MODEL_NAME="${MODEL_NAME:-GLM-5.1}"
export MODEL_REPO="${MODEL_REPO:-zai-org}"

export PROJECT_NAME="async-grpo-gsm8k"
export EXPERIMENT_NAME="${MODEL_NAME}-verl-sglang-megatron-v1-separate-async-${SLURM_JOB_NUM_NODES}n"
export RUN_NAME="${EXPERIMENT_NAME}-${SLURM_JOB_ID}"
export TRAINING_HOME=/capstor/scratch/cscs/${USER}/RL/${MODEL_NAME}
export TRAINING_CONFIG=/tmp
export CHECKPOINT_HOME=${TRAINING_HOME}/checkpoints/${EXPERIMENT_NAME}-run-${SLURM_JOB_ID} #remove "run-${SLURM_JOB_ID}" to enable checkpoint resuming


mkdir -p $TRAINING_HOME
cd $TRAINING_HOME


# 8 rollout nodes = one SGLang TP=32 replica (the model needs all 32 GPUs). The remaining 120
# nodes (480 GPUs) train at TP=4 x PP=3 x EP=8 -> DP=5. The node count grew 80 -> 104 -> 128 while
# chasing the step-1 optimizer-state OOM on GPU 0; the CPU-offload optimizer below is what fixed
# it (see CLAUDE.md), so fewer nodes should work now -- untested.
export ROLLOUT_NNODES=8
export TRAINING_NNODES=$(( SLURM_JOB_NUM_NODES - ROLLOUT_NNODES ))

# Batching contract (verl asserts it): train_batch_size == parameter_sync_step * ppo_mini_batch_size.
# Both are PROMPT counts; verl multiplies ppo_mini_batch_size by rollout.n for the row count and
# checks it against dp_size (DP=5 x EP=8 = 40). Keep rows >= 2x dp_size so every DP rank gets
# >= 2 rows per mini-batch (cheap insurance against length-1 batch edge cases).
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
  max_response_length: 256  # GSM8K answers fit; 1024 made generation the bottleneck at TP=32

actor_rollout_ref:
  model:
    path: ${TRAINING_HOME}/models/${MODEL_NAME}
    # THD packed sequences: required by R3 router replay and by use_fused_kernels. Works with DSA
    # on megatron-core 0.19.0 + megatron-bridge 0.6.1 (the image); it did not on older versions.
    use_remove_padding: True
    use_shm: false
    trust_remote_code: True  # GLM-5.1 uses a custom TokenizersBackend tokenizer
    # Fused linear cross-entropy: never materializes the [tokens, vocab] logits tensor (several GiB
    # of activation relief). verl auto-disables it (with a warning) if any prerequisite is unmet.
    use_fused_kernels: True

  actor:
    # 480 training GPUs: TP=4 x PP=3 x EP=8 = 96 -> DP=5. GLM-5.1: PP=3 -> 78 / 3 = 26 layers per
    # stage; EP=8 (EP=4 leaves experts 64-127 unmapped in megatron-bridge).
    ppo_mini_batch_size: ${PPO_MINI_BATCH_SIZE}
    ppo_micro_batch_size_per_gpu: 1
    # Token budget per micro-batch: 4096 keeps the fwd/bwd reserved high-water low ahead of the
    # optimizer step (memory history in CLAUDE.md); with ~2 rows per DP rank this is 1 micro-batch.
    ppo_max_token_len_per_gpu: 4096
    use_dynamic_bsz: True
    megatron:
      tensor_model_parallel_size: 4
      pipeline_model_parallel_size: 3
      expert_model_parallel_size: 8
      # Params stay on GPU (bf16, ~17 GB/rank): offloading them tipped trainer-node host RAM over
      # 450 GB together with the CPU-offloaded optimizer state and the delta-sync snapshot.
      param_offload: False
      grad_offload: True
      optimizer_offload: True
      vanilla_mbridge: False  # GLM-5.1 (glm_moe_dsa) needs megatron-bridge
      # R3 (Rollout Router Replay): replays the rollout's expert routing in the trainer -- verl's
      # recommended alignment for large MoE. Needs use_remove_padding + rollout.enable_rollout_routing_replay
      # + the r3 sglang import patch below. Config path confirmed against v0.9.0 (actor.megatron.*,
      # not the top-level actor.router_replay).
      router_replay:
        mode: R3
      override_transformer_config:
        recompute_granularity: full
        recompute_method: uniform
        recompute_num_layers: 1
        use_cpu_initialization: True
        moe_grouped_gemm: True
        moe_permute_fusion: True
    optim:
      # Megatron HybridDeviceOptimizer (CPU-streamed optimizer state) -- verl's own large-MoE
      # megatron-async recipes set exactly these keys. Removes the ~40 GB optimizer-state transient
      # that OOMed GPU 0 at DP<=5 (6 runs; CLAUDE.md). main_grads_dtype bf16 halves the DDP grad
      # buffer (~37 -> ~18 GB); safe here because the 4096-token budget means no grad accumulation.
      override_optimizer_config:
        optimizer_cpu_offload: True
        optimizer_offload_fraction: 0.7  # 30% of the optimizer state stays on GPU; frees ~7 GB host/rank
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
    # One TP=32 replica on 32 GPUs: ~744B in bf16 needs all of them.
    tensor_model_parallel_size: 32
    gpu_memory_utilization: 0.75
    free_cache_engine: false  # keep KV cache alive across weight syncs — avoids engine rebuild + CUDA graph re-capture across TP=32 (8-node deadlock)
    calculate_log_probs: True   # required: bypass_mode reads rollout_log_probs as old_log_probs
    log_prob_use_dynamic_bsz: True
    enable_rollout_routing_replay: True  # R3 companion flag (see actor.megatron.router_replay)
    # delta_sharded: each rank byte-diffs its local shard and ships only changed (position, value)
    # pairs; the nccl backend re-streamed the full model via megatron-bridge every sync and hung
    # ~1 run in 2. Its two scale bugs (rank-0 padded gather OOM / 480-peer gather hang) are fixed
    # by verl PRs #7778 / #7777, applied below.
    checkpoint_engine:
      backend: delta_sharded  # separate-async rejects "naive"; delta_sharded extends the nccl engine
      update_weights_bucket_megabytes: 512  # flush bucket to the rollout (2048 cost host RAM on the trainer)
      engine_kwargs:
        delta_sharded:
          rebuild_group: false
          # Per-rank byte budget of one steady-sync gather round (verl PR #7778). Under PP>1 every
          # param merges over the 480-rank WORLD group, so this bounds what rank 0 receives per round;
          # with PR #7777 (targeted P2P) that is real data from <= 32 ranks, i.e. a few GB at most.
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
      vanilla_mbridge: False  # GLM-5.1 (glm_moe_dsa) needs megatron-bridge

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
  # Fixed step count (overrides epochs; also sizes the LR schedule). 40 steps ~ 1 h wall here.
  total_training_steps: 40
  project_name: ${PROJECT_NAME}
  experiment_name: ${RUN_NAME}
  nnodes: ${TRAINING_NNODES}
  n_gpus_per_node: 4
  # Checkpoint saving OFF (2026-09-07): with the default use_dist_checkpointing: False the model save
  # is an HF export gathered through rank-0 HOST RAM (302 GB -> node OOM on DeepSeek-V3, run 3279810;
  # this model is larger). For checkpoints, try actor.megatron.use_dist_checkpointing: True (untested).
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


_DUMP_LEFT = [6]  # print a few raw rollouts per reward process (verl does not log generations)


def _maybe_dump(solution_str: str, ground_truth, model_ans) -> None:
    # Cheap visibility into what the rollout actually generates ([REWARD-DUMP] lines in the log):
    # the fastest way to tell a broken rollout from a wrong answer format.
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

    # Smooth length penalty starting at 1000 words, max -0.2 at 2000 words (runaway verbosity only).
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

# Local verl source patches (example/patches/*.patch), embedded as heredocs: under sbatch the
# script runs from Slurm's spool copy, so script-relative paths do not resolve on the nodes. Each
# heredoc must stay byte-identical to its checked-in file. Staged on the batch host, sbcast to
# every node, applied in the srun (apply-or-fail, so no node can silently run unpatched code).
#   v1-separate-async-fixes.patch ....... no-op the hybrid rollout replicas + the stale hybrid
#                                          weight-sync call; nested-tensor fix in TensorDict assembly
#   r3-sglang-routed-experts-import-fix .. R3 rollout capture vs sglang 0.5.16 (moved module,
#                                          base64 routed_experts)
#   wsync-debug-progress-log.patch ....... per-tensor weight-sync progress log + the seed-sync
#                                          WORLD barrier (megatron-bridge collective desync)
#   delta-sharded-localserializedtensor-import-fix .. sglang 0.5.16 import path for the delta path
#   step1-oom-memdump.patch .............. unconditional empty_cache() before optimizer.step()
#                                          (+ a one-shot [MEMDUMP] memory report per rank)
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


# SGLang scheduler-watchdog diagnostic hook: on a TP=32 rollout watchdog timeout (3 distinct call
# sites so far, see the "TP=32 SGLang" hazard in CLAUDE.md) also print nvidia-smi + local CUDA
# state next to the stock py-spy dump. Wired into sglang's WatchdogRaw dump_info in the srun.
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


# Upstream verl PRs applied at runtime (v0.9.0 does not have them). Fetched ONCE here and sbcast:
# per-node downloads inside the srun left half the cluster unpatched (run 3124273). -f turns an
# HTTP error page into a hard failure. What each one is: see the apply loop in the srun.
for pr in 7421 7422 7423 7777 7778; do
    curl -sfL "https://github.com/verl-project/verl/pull/${pr}.patch" -o "${TRAINING_CONFIG}/${pr}.patch" \
        || { echo "FATAL: could not download PR #${pr}"; exit 1; }
    [ -s "${TRAINING_CONFIG}/${pr}.patch" ] \
        || { echo "FATAL: PR #${pr} patch is empty"; exit 1; }
    sbcast -f "${TRAINING_CONFIG}/${pr}.patch" "${TRAINING_CONFIG}/${pr}.patch"
done

# sglang PR #38298 (found on the DeepSeek-V3 recipe, 2026-09-07): the sglang DeepSeek-family loader
# fused q_a_proj / kv_a_proj_with_mqa only when both halves arrived in the same load_weights call;
# the chunked delta weight sync split pairs across calls and left layers with dummy-init attention.
# A no-op for models without q_lora_rank; applied here for parity with the DeepSeek-V3 recipe
# (patch -p2 against the installed sglang 0.5.16 in the srun).
export SGLANG_FIX_QKV_A_CACHE="38298"
curl -sfL "https://github.com/sgl-project/sglang/pull/${SGLANG_FIX_QKV_A_CACHE}.patch" -o "${TRAINING_CONFIG}/sglang-qkv-a-cache.patch" \
    || { echo "FATAL: could not download sglang PR #${SGLANG_FIX_QKV_A_CACHE}"; exit 1; }
grep -q "^diff --git a/python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py" "${TRAINING_CONFIG}/sglang-qkv-a-cache.patch" \
    || { echo "FATAL: sglang qkv_a cache patch has no loader diff (bad SHA or GitHub error page)"; exit 1; }
sbcast -f "${TRAINING_CONFIG}/sglang-qkv-a-cache.patch" "${TRAINING_CONFIG}/sglang-qkv-a-cache.patch"


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


# verl is baked into the image at v0.9.0 (editable install at /workspace/verl); the patches below go on top.
git -C /workspace/verl --no-pager log --oneline -1 || true


# Redirect pip cache to local tmpfs — ~/.cache/pip is on Lustre which causes
# "Stale file handle" (ESTALE) errors during package downloads.
export PIP_CACHE_DIR=/tmp/pip_cache_${SLURM_JOB_ID}
export TMPDIR=/tmp
mkdir -p $PIP_CACHE_DIR

# Image-version + import smoke test (non-fatal): makes a wrong image tag obvious in the first 30 s.
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


# megatron-bridge safe_config_loader: drop its filelock (fcntl.flock is unsupported in-container on
# CSCS Lustre/tmpfs). Safe: the config files are written by local rank 0 before any reader starts.
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

# Wire the watchdog diagnostic (cscs_watchdog_diag.py) into the sglang WatchdogRaw dump_info hook.
# Best-effort: WARN and continue on sglang drift (diagnostic only, not load-bearing).
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
        # soft=soft, appears twice: the WatchdogRaw(...) call (the target) is the one immediately
        # followed by the closing paren.
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

# sglang PR #38298 (see the batch-host block): LOAD-BEARING for the delta weight sync. patch -p2
# against the installed sglang (diff paths are python/sglang/...); presence of the new attribute
# is both the idempotency test and the post-check. Must run before ray start.
SGL_LOADER=$(python3 -c "import sglang.srt.models.deepseek_common.deepseek_weight_loader as m; print(m.__file__)")
[ -n "$SGL_LOADER" ] || { echo "FATAL: sglang deepseek_weight_loader not found on $(hostname)"; exit 1; }
SGL_PKG_ROOT=${SGL_LOADER%/sglang/srt/models/deepseek_common/deepseek_weight_loader.py}
if grep -q "_pending_fused_a_proj" "$SGL_LOADER"; then
    echo "sglang qkv_a cache fix already present on $(hostname), skipping"
elif patch -p2 -d "$SGL_PKG_ROOT" -s < "${TRAINING_CONFIG}/sglang-qkv-a-cache.patch" && grep -q "_pending_fused_a_proj" "$SGL_LOADER"; then
    echo "Applied sglang PR #${SGLANG_FIX_QKV_A_CACHE} (qkv_a cache fix) on $(hostname)"
else
    echo "FATAL: sglang qkv_a cache fix did not apply on $(hostname)"
    exit 1
fi

# Upstream verl PRs (fetched on the batch host). apply-or-already-present-or-FATAL: a cluster where
# only some ranks carry a patch is worse than one that carries none (run 3124273).
#   #7421 DSA attention compat with mcore >= 0.16.2 (load-bearing: GLM-5.1 uses DSA)
#   #7422 keep load_format=dummy for the standalone rollout (else it loads the full model from Lustre)
#   #7423 lock + barriers around the async weight sync (NCCL deadlock)
#   #7777 delta_sharded steady sync: targeted P2P instead of the padded 480-way gather-to-rank-0
#         (found on the DeepSeek-V3 recipe; run 3263683 here hung on the same collective).
#   #7778 gather_round_megabytes kwarg (same origin) -- applied separately below with fuzz.
for pr in 7421 7422 7423 7777; do
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

# PR #7778: delta_checkpoint_engine.py moved between v0.9.0 and main, so its hunk needs patch
# --fuzz=3. Idempotency = a presence test on the installed file (NOT patch --dry-run -R, which with
# fuzz reports "already applied" on an unpatched tree), re-asserted after applying.
p="${TRAINING_CONFIG}/7778.patch"
_has_round() { grep -q "gather_round_megabytes" /workspace/verl/verl/checkpoint_engine/delta_checkpoint_engine.py; }
if _has_round; then
    echo "PR #7778 already present on $(hostname), skipping"
elif patch -p1 -d /workspace/verl --fuzz=3 -s < "$p" && _has_round; then
    echo "Applied PR #7778 (patch --fuzz=3) on $(hostname)"
else
    echo "FATAL: PR #7778 did not apply on $(hostname)"
    exit 1
fi

# Local patches (staged on the batch host, see there for what each does). Same apply-or-fail
# discipline. Order matters: step1-oom-memdump.patch is diffed on top of wsync-debug-progress-log.patch.
for lp in v1-separate-async-fixes r3-sglang-routed-experts-import-fix wsync-debug-progress-log \
          delta-sharded-localserializedtensor-import-fix step1-oom-memdump; do
    p="${TRAINING_CONFIG}/${lp}.patch"
    if git -C /workspace/verl apply --check "$p" 2>/dev/null; then
        git -C /workspace/verl apply "$p" && echo "Applied ${lp}.patch on $(hostname)"
    elif git -C /workspace/verl apply --reverse --check "$p" 2>/dev/null; then
        echo "${lp}.patch already present on $(hostname), skipping"
    else
        echo "FATAL: ${lp}.patch neither applies nor is already present on $(hostname)"
        exit 1
    fi
done


# Mirror model config files to local tmpfs to avoid Lustre metadata contention.
# All training workers calling AutoConfig.from_pretrained() simultaneously causes
# ENOLCK / ESTALE on the Lustre MDS. Only local rank 0 does the copy; others wait.
export MODEL_LOCAL=/tmp/glm_model_${SLURM_JOB_ID}
if [ $SLURM_LOCALID -eq 0 ]; then
    mkdir -p $MODEL_LOCAL
    # Copy small config/tokenizer files locally
    find ${TRAINING_HOME}/models/${MODEL_NAME} -maxdepth 1 -not -name "*.safetensors" -type f \
        -exec cp {} $MODEL_LOCAL/ \; 2>/dev/null || true
    # Symlink safetensors back to Lustre so megatron-bridge can still load weights
    for f in ${TRAINING_HOME}/models/${MODEL_NAME}/*.safetensors; do
        ln -sf "$f" "$MODEL_LOCAL/$(basename "$f")"
    done 2>/dev/null || true
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

# NCCL flight recorder: on a collective timeout, dump per-rank collective traces (one file per rank
# under /tmp) instead of the useless "last enqueued NCCL work: -1".
export TORCH_NCCL_TRACE_BUFFER_SIZE=20000
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE=/tmp/nccl_flightrecorder_${SLURM_JOB_ID}_rank

# No NVLink multicast: with this many communicators each NVLS group reserves buffers on device 0
# (~1.4 GB back on the tight trainer GPU 0). Pure perf knob, no correctness impact.
export NCCL_NVLS_ENABLE=0

# Do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True here: this srun body is shared by the
# SGLang rollout processes, whose torch_memory_saver hard-raises on it (run 3219305).


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
