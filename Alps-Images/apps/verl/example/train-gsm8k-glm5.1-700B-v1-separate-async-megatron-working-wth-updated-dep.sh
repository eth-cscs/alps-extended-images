#!/bin/bash

#SBATCH --nodes=80
#SBATCH --account=csstaff
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=5:00:00

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

export VERL_IMAGE="jfrog.svc.cscs.ch/docker-group-csstaff/alps-images/verl:alps7-dev-a9f9e56471c0574e" #alps7-dev-a9f9e56471c0574e image with update dependencies #alps7-dev-0f334b540ccc7034 image with megatron

export MODEL_NAME="GLM-5.1"
export MODEL_REPO="zai-org"

export PROJECT_NAME="async-grpo-gsm8k"
export EXPERIMENT_NAME="${MODEL_NAME}-verl-sglang-megatron-v1-separate-async-${SLURM_JOB_NUM_NODES}n"
export RUN_NAME="${EXPERIMENT_NAME}-${SLURM_JOB_ID}"
export TRAINING_HOME=/capstor/scratch/cscs/${USER}/RL/${MODEL_NAME}
export TRAINING_CONFIG=/tmp
export CHECKPOINT_HOME=${TRAINING_HOME}/checkpoints/${EXPERIMENT_NAME}-run-${SLURM_JOB_ID} #remove "run-${SLURM_JOB_ID}" to enable checkpoint resuming


mkdir -p $TRAINING_HOME
cd $TRAINING_HOME



# Standalone rollout needs exactly 8 nodes for TP=32 (8 nodes × 4 GPUs = 32 GPUs, 1 replica).
# Training gets the remaining 72 nodes (288 GPUs) for TP=4, PP=3, EP=8, DP=3.
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
export PPO_MINI_BATCH_SIZE=6         # prompts; x ROLLOUT_N = 48 rows = 2x dp_size (DP=3 x EP=8 = 24)
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
  max_response_length: 256  # reduced from 1024 — at ~10 tok/s for 744B TP=32, 1024 tokens takes 83min for 48 samples

actor_rollout_ref:
  model:
    path: ${TRAINING_HOME}/models/${MODEL_NAME}
    # DSA attention + THD (packed-sequence): re-enabled 2026-08-28, third attempt. Run 3201189
    # (2026-08-27) found megatron-bridge 0.6.1 alone breaks core Megatron model init unconditionally
    # — it needs megatron-core ~0.19.0, not the 0.18.2 this image ships. This script now upgrades
    # both together; validated on a cheap 1-node probe (job 3207095) before this re-enable: both
    # wheels install cleanly, and every import smoke test passes, including the exact chain that
    # crashed 3201189 and the GLM-5.1 bridge module itself. See CLAUDE.md's Configuration audit +
    # Run log entries for the full trace. Still unverified end-to-end on the real 80-node recipe —
    # the probe only exercised imports, not actual model construction/training.
    use_remove_padding: True
    use_shm: false
    trust_remote_code: True  # GLM-5.1 uses a custom TokenizersBackend tokenizer

  actor:
    # 72 training nodes x 4 GPUs = 288 GPUs; TP=4, PP=3, EP=8 -> 4x3x8=96, DP=3.
    # Memory at optimizer step: M*(2 + 8/DP) + 11 GiB NCCL = 14.5*(2+2.67)+11 = 78.7 GiB < 95 GiB.
    # DP=2 was insufficient: 6M+11 = 98 GiB > 95 GiB (4 optimizer tensors + param + grad).
    # EP=8: EP=4 causes megatron-bridge to fail for experts 64-127 (unmapped).
    # PP=3: 78 layers / 3 = 26 layers/stage.
    ppo_mini_batch_size: ${PPO_MINI_BATCH_SIZE}
    ppo_micro_batch_size_per_gpu: 1
    # 16384 -> 12288 (2026-08-29): run 3217439 ran 20 clean steps then CUDA-OOM'd by 24 MiB in the
    # Megatron optimizer on GPU 0 (the weight-sync coordinator rank) -- torch peak was dead-flat
    # 77.9 GiB the whole run, the tip-over was ~17 GiB of non-torch NCCL/checkpoint-engine memory
    # creeping up over the weight syncs. Lowering the dynamic-bsz token budget cuts the actor
    # fwd/bwd activation peak to open real headroom (paired with expandable_segments in the srun).
    ppo_max_token_len_per_gpu: 12288
    use_dynamic_bsz: True
    megatron:
      tensor_model_parallel_size: 4
      pipeline_model_parallel_size: 3
      expert_model_parallel_size: 8
      param_offload: True
      grad_offload: True
      optimizer_offload: True
      vanilla_mbridge: False  # GLM-5.1 model_type=glm_moe_dsa requires Megatron-Bridge
      # R3 (Rollout Router Replay) — verl's MoE routing-alignment mechanism, appropriate for this
      # model per verl's own docs (GLM-5 named as an adopter). Re-enabled 2026-08-28, third
      # attempt — see the model.use_remove_padding comment above and this script's Configuration
      # audit / Run log entries in CLAUDE.md for the full trace of what broke it twice before and
      # what changed (megatron-core now upgraded alongside megatron-bridge, validated on a cheap
      # probe first). The correct config path, confirmed against v0.9.0 source, is
      # router_replay/mode: R3 right here under actor.megatron — not the top-level
      # actor.router_replay.mode field. Still unverified on the real 80-node recipe.
      router_replay:
        mode: R3
      override_transformer_config:
        recompute_granularity: full
        recompute_method: uniform
        recompute_num_layers: 1
        use_cpu_initialization: True
        moe_grouped_gemm: True
        moe_permute_fusion: True

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
    # 8 rollout nodes × 4 GPUs = 32 GPUs; TP=32 (one replica) — 700B needs all 32 GPUs to fit
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
      engine_kwargs:
        delta_sharded:
          rebuild_group: false
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
      vanilla_mbridge: False  # GLM-5.1 model_type=glm_moe_dsa requires Megatron-Bridge

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
  # 3149736): 40 * 360s = 4h worst case + ~20min setup overhead, comfortably inside the 5h
  # limit. Enough to see whether training is moving at all (KL-from-reference nonzero, reward
  # mean not degenerate) but not enough for a trustworthy reward trend, which needs closer to
  # 50-100 steps on GSM8K's noisy per-step reward signal.
  total_training_steps: 40
  project_name: ${PROJECT_NAME}
  experiment_name: ${RUN_NAME}
  nnodes: ${TRAINING_NNODES}
  n_gpus_per_node: 4
  save_freq: 20
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
# reason as the patches above). Diagnostic-only: per-tensor progress logging for the
# trainer->rollout weight sync so the next megatron-bridge collective hang shows where each
# rank stalled. Applied after the verl checkout, alongside the other verl-source patches.
cat > "${TRAINING_CONFIG}/wsync-debug-progress-log.patch" <<- 'EOF'
# [CSCS debug, 2026-08-29] Per-tensor progress logging for the trainer->rollout weight sync, so
# the next megatron-bridge weight-sync collective hang (see CLAUDE.md: gather_from_ep_ranks NCCL
# ALLGATHER timeout, hit in runs 3141801 / 3207923 / 3219811, ~1 run in 2) shows WHERE each rank
# stalled instead of a 30-minute black box. Wraps the two weight-export generators in the
# Megatron engine (get_per_tensor_param = the full seed export; get_per_tensor_param_delta_shard
# = the delta_sharded steady export) with a pass-through that prints `[WSYNC-DBG] rank=N ...
# tensor#K` every 1000 tensors and on generator exit. Pairs with TORCH_NCCL_TRACE_BUFFER_SIZE /
# TORCH_NCCL_DUMP_ON_TIMEOUT (set in the srun env) which give the per-rank NCCL flight-recorder
# trace on timeout. Diagnostic only -- remove once the hang is root-caused.
#
# Generated from a real edited git worktree at the verl v0.9.0 tag; verified against a fresh
# checkout: git apply --check clean, py_compile clean, git apply --reverse --check detects
# "already applied".
diff --git a/verl/workers/engine/megatron/transformer_impl.py b/verl/workers/engine/megatron/transformer_impl.py
index e8a6c56..0f171d2 100644
--- a/verl/workers/engine/megatron/transformer_impl.py
+++ b/verl/workers/engine/megatron/transformer_impl.py
@@ -87,6 +87,30 @@ logger = logging.getLogger(__file__)
 logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))
 
 
+def _wsync_progress_log(gen, tag):
+    """[weight-sync debug, CSCS] Wrap a per-tensor weight-export generator so a collective hang
+    during weight sync shows, per rank, how far the streaming got (which tensor / how long).
+    Pairs with TORCH_NCCL_TRACE_BUFFER_SIZE. Cheap: one print per 1000 tensors. Remove once the
+    megatron-bridge gather_from_ep_ranks hang (CLAUDE.md) is understood."""
+    import time as _t
+
+    _rank = os.environ.get("RANK") or os.environ.get("SLURM_PROCID") or "?"
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
+    finally:
+        print(f"[WSYNC-DBG] rank={_rank} {tag} generator exited after {_n} tensors, {_t.time() - _t0:.0f}s", flush=True)
+
+
 def _resolve_fused_temperature(temperature: float | torch.Tensor) -> float:
     """Return the scalar temperature required by fused linear cross entropy."""
     values = torch.as_tensor(temperature).detach().flatten()
@@ -1042,7 +1066,7 @@ class MegatronEngine(BaseEngine):
 
             per_tensor_param = export_qat_weights(per_tensor_param, self.module, self._qat_config.mode, self.bridge)
 
-        return per_tensor_param, peft_config
+        return _wsync_progress_log(per_tensor_param, "seed/full"), peft_config
 
     def _mcore_export_index(self):
         """Build (once) the per-parameter delta export index: geometry specs and
@@ -1104,7 +1128,9 @@ class MegatronEngine(BaseEngine):
 
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

# Upgrade TransferQueue 0.1.6 (pinned in Alps-Images/apps/verl/Containerfile at image
# build time) to 0.1.7 (the version verl v0.9.0 itself declares in its own
# requirements.txt — this closes a real image/runtime version gap, it is not a
# workaround). This script's runtime `git checkout` above only bumps verl's own
# source, never its pip dependencies, so the image's pinned 0.1.6 would otherwise
# never move. 0.1.6's AsyncSimpleStorageManager._pack_field_values
# (transfer_queue/storage/managers/simple_backend_manager.py) has
# `if all(v.shape == values[0].shape for v in values): return torch.stack(values)` —
# the same "stack if shapes coincidentally match" bug as verl's own
# list_of_dict_to_tensordict (patched below), just in a different package — and it is
# the actual, sole cause of every `AttributeError: 'Tensor' object has no attribute
# 'offsets'` seen in runs 3134772/3136766/3137775/3139371/3144665 (which call site
# trips first — engine_workers.py:294 or padding.py:119 — just depends on which field
# happens to hit the coincidence in a given run). Fixed upstream in 0.1.7's rewritten
# `simple_storage_manager.py` (renamed file, same class/method names), which always
# tries `torch.nested.as_nested_tensor(..., layout=jagged)` first. Fetched once here
# and sbcast, same "fetch once, distribute, apply-or-fail" discipline as the PR
# patches above — pip installing independently on 80 nodes risks the same
# mixed-cluster hazard as run 3124273's per-node curl. sha256-verified against the
# published PyPI digest since this is a binary wheel, not a text patch a `git apply
# --check` could validate.
curl -sfL "https://files.pythonhosted.org/packages/16/fe/bc9c75492b15ad90d0c9e97eb8b3ad643c8b2a023e7361c1ff99948a2f96/transferqueue-0.1.7-py3-none-any.whl" \
    -o "${TRAINING_CONFIG}/transferqueue-0.1.7-py3-none-any.whl" \
    || { echo "FATAL: could not download TransferQueue 0.1.7 wheel"; exit 1; }
echo "d0657b107f668a431989a35b555d47425f398655e876f7b12984924f29d6dba2  ${TRAINING_CONFIG}/transferqueue-0.1.7-py3-none-any.whl" | sha256sum -c - \
    || { echo "FATAL: TransferQueue 0.1.7 wheel failed sha256 verification"; exit 1; }
sbcast -f "${TRAINING_CONFIG}/transferqueue-0.1.7-py3-none-any.whl" "${TRAINING_CONFIG}/transferqueue-0.1.7-py3-none-any.whl"

# Upgrade megatron-core 0.18.2 -> 0.19.0 AND megatron-bridge 0.5.1 -> 0.6.1 together (2026-08-28,
# third R3 attempt). Run 3201189 confirmed a megatron-bridge-only upgrade breaks core Megatron
# model init unconditionally: megatron-bridge 0.6.x is built against megatron-core ~0.19.0 (its
# own pinned Megatron-LM submodule commit at tag v0.6.0 is a 0.19.0-tagged commit, confirmed
# directly from the real GitHub repo), not this image's 0.18.2. Validated on a cheap 1-node probe
# (job 3207095) before touching this script again: both wheels install cleanly with --no-deps
# --force-reinstall, and every import smoke test passes, including the exact chain that crashed
# 3201189 (megatron.bridge.training.config.DistributedDataParallelConfig ->
# megatron.training.models.gpt) and the GLM-5.1 bridge module (megatron.bridge.models.glm_moe_dsa)
# -- see CLAUDE.md's Configuration audit + Run log entries for the full trace and citations.
# megatron-core is a real bdist_wheel for this exact platform (cp312, aarch64/manylinux) with only
# one compiled extension (a CPU-only dataset-helpers .so, no CUDA/ABI-sensitive code) -- much lower
# risk than a generic "compiled package" swap would suggest. Fetched once here and sbcast, same
# discipline as the TransferQueue wheel above. megatron-core installed first (matching the
# validated probe's install order), megatron-bridge second.
curl -sfL "https://files.pythonhosted.org/packages/d1/75/621dc2772a5aba566a828e10f4772f71ed33d2fa99ad47b2dcc763be7d10/megatron_core-0.19.0-cp312-cp312-manylinux_2_24_aarch64.manylinux_2_28_aarch64.whl" \
    -o "${TRAINING_CONFIG}/megatron_core-0.19.0-cp312-cp312-manylinux_2_24_aarch64.manylinux_2_28_aarch64.whl" \
    || { echo "FATAL: could not download megatron-core 0.19.0 wheel"; exit 1; }
echo "25618d4ba1fbed1fd7b00a210905065c1d0479894c1a5f8626c3627844e072fa  ${TRAINING_CONFIG}/megatron_core-0.19.0-cp312-cp312-manylinux_2_24_aarch64.manylinux_2_28_aarch64.whl" | sha256sum -c - \
    || { echo "FATAL: megatron-core 0.19.0 wheel failed sha256 verification"; exit 1; }
sbcast -f "${TRAINING_CONFIG}/megatron_core-0.19.0-cp312-cp312-manylinux_2_24_aarch64.manylinux_2_28_aarch64.whl" "${TRAINING_CONFIG}/megatron_core-0.19.0-cp312-cp312-manylinux_2_24_aarch64.manylinux_2_28_aarch64.whl"

curl -sfL "https://files.pythonhosted.org/packages/1d/d0/90c877735309154b26b650f14f17040fb066d7229edb342d6ef24264718d/megatron_bridge-0.6.1-py3-none-any.whl" \
    -o "${TRAINING_CONFIG}/megatron_bridge-0.6.1-py3-none-any.whl" \
    || { echo "FATAL: could not download megatron-bridge 0.6.1 wheel"; exit 1; }
echo "a3ef69c679a786e353f14ecef34329a819598edfaeb51aec02a519ace4c33c89  ${TRAINING_CONFIG}/megatron_bridge-0.6.1-py3-none-any.whl" | sha256sum -c - \
    || { echo "FATAL: megatron-bridge 0.6.1 wheel failed sha256 verification"; exit 1; }
sbcast -f "${TRAINING_CONFIG}/megatron_bridge-0.6.1-py3-none-any.whl" "${TRAINING_CONFIG}/megatron_bridge-0.6.1-py3-none-any.whl"

# Upgrade flashinfer 0.6.12 -> 0.6.14, MATCHED python + cubin pair (2026-08-29). The Containerfile
# pins flashinfer 0.6.12 (added 2026-07-27), but sglang 0.5.16 -- what the image sglang[all] now
# resolves to -- requires flashinfer_python[cu13]==0.6.14. Run 3209484 hung at TP=32 (~step 13)
# inside sglang MLA/DSA decode plan() (flashinfer/mla/_core.py:839), the code path this version
# skew affects. flashinfer 0.6.14 hard-raises at import unless flashinfer_cubin is the *exact*
# same version (run 3214410). flashinfer_cubin is NOT fully published to PyPI (PyPI stops at
# 0.6.13) -- its real distribution channel is https://flashinfer.ai/whl, which serves the GitHub
# release assets (per flashinfer's own install docs; see GitHub issue #2133 for the PyPI gap).
# flashinfer_cubin-0.6.14 exists there -> install the matched 0.6.14 pair, clean AOT, no
# version-check bypass needed.
#
# Both wheels staged on shared Lustre (${FLASHINFER_WHEEL_DIR}), NOT sbcast: runs 3211735 and
# 3213313 died with `Bus error (core dumped)` from the flashinfer `sbcast` call -- on the 458 MB
# cubin AND on the 14.6 MB python wheel whose sha256 verified fine moments earlier -- so it is not
# wheel size, the Nth-consecutive-sbcast / batch-host /tmp pressure is the problem. Every node
# reads the wheels directly from Lustre instead; a plain static-file read has none of the
# flock/write hazards that make Lustre unsafe for other things.
FLASHINFER_WHEEL_DIR="${TRAINING_HOME}/wheels"
mkdir -p "${FLASHINFER_WHEEL_DIR}"
curl -sfL "https://github.com/flashinfer-ai/flashinfer/releases/download/v0.6.14/flashinfer_python-0.6.14-py3-none-any.whl" \
    -o "${FLASHINFER_WHEEL_DIR}/flashinfer_python-0.6.14-py3-none-any.whl" \
    || { echo "FATAL: could not download flashinfer_python 0.6.14 wheel"; exit 1; }
echo "d124369346a3d48eac67e31c42f7a3c813bcc0abc10e2e36db413b7b3dfd97df  ${FLASHINFER_WHEEL_DIR}/flashinfer_python-0.6.14-py3-none-any.whl" | sha256sum -c - \
    || { echo "FATAL: flashinfer_python 0.6.14 wheel failed sha256 verification"; exit 1; }
curl -sfL "https://github.com/flashinfer-ai/flashinfer/releases/download/v0.6.14/flashinfer_cubin-0.6.14-py3-none-any.whl" \
    -o "${FLASHINFER_WHEEL_DIR}/flashinfer_cubin-0.6.14-py3-none-any.whl" \
    || { echo "FATAL: could not download flashinfer_cubin 0.6.14 wheel"; exit 1; }
echo "7bbed9f3851b59f3f6f6cb344810775bbce8a1c012ecf9a502bebd91cfb6433e  ${FLASHINFER_WHEEL_DIR}/flashinfer_cubin-0.6.14-py3-none-any.whl" | sha256sum -c - \
    || { echo "FATAL: flashinfer_cubin 0.6.14 wheel failed sha256 verification"; exit 1; }


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


# Upgrade Verl to v0.9.0.
# The image clones with --branch ${VERL_REF} --depth 1, so no other ref is present
# locally and the tag has to be fetched explicitly before it can be checked out.
# verl is installed editable (pip install -e) from /workspace/verl, so the checkout
# takes effect without reinstalling; -f discards any dirty state in the clone.
export VERL_REF=v0.9.0
git -C /workspace/verl fetch --depth 1 origin +refs/tags/${VERL_REF}:refs/tags/${VERL_REF} \
    && git -C /workspace/verl checkout -f ${VERL_REF} \
    || { echo "FATAL: could not check out verl ${VERL_REF}"; exit 1; }
git -C /workspace/verl --no-pager log --oneline -1


# Redirect pip cache to local tmpfs — ~/.cache/pip is on Lustre which causes
# "Stale file handle" (ESTALE) errors during package downloads.
export PIP_CACHE_DIR=/tmp/pip_cache_${SLURM_JOB_ID}
export TMPDIR=/tmp
mkdir -p $PIP_CACHE_DIR

# Upgrade TransferQueue 0.1.6 (baked into the image at build time) to 0.1.7 (the
# version verl v0.9.0 itself declares in its own requirements.txt) — see the wheel
# fetch/sbcast above and Known hazards in CLAUDE.md for why: 0.1.6 AsyncSimpleStorageManager
# _pack_field_values has a genuine upstream bug, already fixed in 0.1.7, that is the actual
# cause of every offsets AttributeError seen in this recipe so far. --no-deps avoids pulling
# in an unwanted transitive bump; --force-reinstall makes the upgrade deterministic
# regardless of what pip thinks 0.1.6 already satisfies.
pip install --no-deps --force-reinstall "${TRAINING_CONFIG}/transferqueue-0.1.7-py3-none-any.whl" \
    || { echo "FATAL: could not install TransferQueue 0.1.7 wheel"; exit 1; }
python3 -c "import importlib.metadata; print(\"TransferQueue version:\", importlib.metadata.version(\"TransferQueue\"))"


# Upgrade megatron-core 0.18.2 -> 0.19.0, then megatron-bridge 0.5.1 -> 0.6.1 (see the wheel
# fetch/sbcast above for the full reasoning and the 1-node probe job 3207095 that validated this
# exact combination). Order matches the validated probe. MUST both run before the
# safe_config_loader filelock patch below -- --force-reinstall rewrites every file in the
# megatron-bridge package, so patching first would just get silently wiped here.
pip install --no-deps --force-reinstall "${TRAINING_CONFIG}/megatron_core-0.19.0-cp312-cp312-manylinux_2_24_aarch64.manylinux_2_28_aarch64.whl" \
    || { echo "FATAL: could not install megatron-core 0.19.0 wheel"; exit 1; }
pip install --no-deps --force-reinstall "${TRAINING_CONFIG}/megatron_bridge-0.6.1-py3-none-any.whl" \
    || { echo "FATAL: could not install megatron-bridge 0.6.1 wheel"; exit 1; }
python3 -c "import importlib.metadata; print(\"megatron-core version:\", importlib.metadata.version(\"megatron-core\"))"
python3 -c "import importlib.metadata; print(\"megatron-bridge version:\", importlib.metadata.version(\"megatron-bridge\"))"


# Upgrade flashinfer to the matched 0.6.14 python + cubin pair (see the wheel fetch above for the
# full reasoning). sglang 0.5.16 requires flashinfer_python[cu13]==0.6.14; the image stale 0.6.12
# pin (predates the sglang[all] bump) is the leading suspect for run 3209484 TP=32 MLA plan() hang
# -- stuck frame inside flashinfer/mla/_core.py:839 plan(). flashinfer 0.6.14 hard-requires an
# exact python/cubin version match at import (run 3214410), and flashinfer_cubin-0.6.14 is only on
# https://flashinfer.ai/whl (GitHub release assets), not PyPI -- both wheels are fetched from
# there above. --no-deps keeps the existing nvidia-cutlass-dsl (already >=4.5.0, what the cu13
# extra wants); --force-reinstall makes the swap deterministic. Read from shared Lustre (not
# sbcast -- runs 3211735/3213313 both bus-errored on the flashinfer sbcast); a plain static-file
# read has none of the flock/write hazards that make Lustre unsafe elsewhere. cubin first (larger
# payload), then python.
pip install --no-deps --force-reinstall "${TRAINING_HOME}/wheels/flashinfer_cubin-0.6.14-py3-none-any.whl" \
    || { echo "FATAL: could not install flashinfer_cubin 0.6.14 wheel"; exit 1; }
pip install --no-deps --force-reinstall "${TRAINING_HOME}/wheels/flashinfer_python-0.6.14-py3-none-any.whl" \
    || { echo "FATAL: could not install flashinfer_python 0.6.14 wheel"; exit 1; }
python3 -c "import importlib.metadata; print(\"flashinfer-python version:\", importlib.metadata.version(\"flashinfer-python\")); print(\"flashinfer-cubin version:\", importlib.metadata.version(\"flashinfer-cubin\"))"


# R3 router replay prerequisite checks (diagnostic only, not fatal — see the Configuration
# audit entry for this script in CLAUDE.md). Confirms both upgrades above actually took, confirms
# the exact import chain that crashed run 3201189 now resolves (same check the validating probe
# job 3207095 ran), and confirms whether sglang has the routed-experts-capture feature at the
# path the r3-sglang-routed-experts-import-fix.patch below expects (applied after the verl
# checkout, further down) — probe job 3199799 found it at sglang.srt.state_capturer.routed_experts
# on sglang==0.5.16, but sglang[all] has no version pin in the Containerfile, so that is not
# guaranteed stable across image rebuilds. If any check below fails, the real step will still fail
# loudly later with a clear exception rather than silently, so it is safe to let the job continue
# and read the real error.
python3 -c "
import importlib.metadata
from packaging import version
v = importlib.metadata.version(\"megatron-bridge\")
print(\"megatron-bridge version:\", v)
if version.parse(v) < version.parse(\"0.6.0\"):
    print(\"WARNING: megatron-bridge < 0.6.0 — GLM THD-packed fused DSA support (needed for \"
          \"use_remove_padding=True + R3) was only confirmed added in 0.6.0; this build may not have it.\")
mc = importlib.metadata.version(\"megatron-core\")
print(\"megatron-core version:\", mc)
if version.parse(mc) < version.parse(\"0.19.0\"):
    print(\"WARNING: megatron-core < 0.19.0 — megatron-bridge 0.6.x needs this; the \"
          \"ModuleNotFoundError from run 3201189 will likely recur.\")
try:
    from megatron.training.models.gpt import GPTModelBuilder, GPTModelConfig, mtp_block_spec
    print(\"megatron.training.models.gpt: OK (the module missing in run 3201189)\")
except ImportError as e:
    print(f\"WARNING: megatron.training.models.gpt still not importable ({e}) — the \"
          \"megatron-core upgrade did not take as expected\")
"
python3 -c "
# Checks the path the r3-sglang-routed-experts-import-fix.patch (applied after the verl
# checkout, further down) points verl at — sglang.srt.state_capturer.routed_experts, confirmed
# correct for sglang==0.5.16 via probe job 3199799. Also checks the old
# sglang.srt.layers.moe.routed_experts_capturer path verl v0.9.0 ships by default, purely as a
# signal: if THAT one succeeds instead, the deployed sglang changed again and the patch itself
# (not just this diagnostic) needs re-pointing.
try:
    from sglang.srt.state_capturer.routed_experts import extract_routed_experts_from_meta_info
    print(\"sglang routed_experts capture: OK at sglang.srt.state_capturer.routed_experts \"
          \"(the path the source patch below points verl at)\")
except ImportError as e:
    print(f\"WARNING: sglang.srt.state_capturer.routed_experts not importable ({e}) — the \"
          \"r3-sglang-routed-experts-import-fix.patch below points at a path that no longer \"
          \"exists on this image; R3 will fail once rollout.enable_rollout_routing_replay=True \"
          \"actually tries to capture routed experts\")
try:
    from sglang.srt.layers.moe.routed_experts_capturer import extract_routed_experts_from_meta_info as _old_path_check
    print(\"NOTE: the old sglang.srt.layers.moe.routed_experts_capturer path also imports \"
          \"successfully on this image — unexpected, since probe job 3199799 found it removed; \"
          \"the source patch below may now be unnecessary or may need reconciling with both paths.\")
except ImportError:
    pass
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

# example/patches/wsync-debug-progress-log.patch: diagnostic-only per-tensor progress logging for
# the trainer to rollout weight sync (see its header, and the weight-sync-hang investigation in
# CLAUDE.md). Same apply-or-fail discipline as the patches above.
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

# NCCL flight recorder — so the NEXT weight-sync hang (or any collective timeout) dumps a
# per-rank collective trace instead of the useless "last enqueued NCCL work: -1" seen in run
# 3219811 (that -1 means the ring buffer was OFF). TRACE_BUFFER_SIZE enables it; DUMP_ON_TIMEOUT
# writes it on watchdog fire; TEMP_FILE lands one file per rank under /tmp so it can be pulled
# from the failed nodes. Pinpoints which rank stalled on which collective seq (i.e. which tensor).
export TORCH_NCCL_TRACE_BUFFER_SIZE=20000
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE=/tmp/nccl_flightrecorder_${SLURM_JOB_ID}_rank

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
