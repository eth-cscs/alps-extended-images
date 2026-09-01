#!/bin/bash

#SBATCH --nodes=16
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
#   * example/patches/v1-separate-async-fixes.patch + upstream PR #7421/#7422/#7423
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
# Apertus-1.5 support (Group 1 + Group 2): UNCHANGED from the fully-async recipe
# -- swiss-ai transformers wheel, SGLang PR #32979 + local fixes, and the
# wqwqazwsxedc Megatron-LM / Megatron-Bridge apertus fork wheels (installed over
# the image's stock megatron-core/bridge) + the vendored Apertus1p5Bridge + xielu.
#
# NOTE: a first pass tried to keep the image's stock megatron-core 0.19.0 /
# megatron-bridge 0.6.1 and add only the apertus delta as line-patches + vendored
# bridge modules. It ran end-to-end but trained degenerately (runs 3240861 /
# 3243271 / 3243467: fluent generations, zero format compliance, no GRPO signal)
# -- a forward-pass discrepancy in the 0.18->0.19 / 0.5->0.6.1 jump that the
# fork's apertus code silently depends on, not visible in weight stats. Reverted
# to the fork wheels; only the image and the trainer are migrated.
# ─────────────────────────────────────────────────────────────────────────────

export VERL_IMAGE="jfrog.svc.cscs.ch/docker-group-csstaff/alps-images/verl-cuda:alps7-dev-621fa40275c4f036" #verl-cuda image: baked verl v0.9.0 + updated deps (matches train-gsm8k-glm5.1-700B-v1-separate-async-megatron.sh)

export MODEL_NAME="Apertus-v1.5-70B"
export MODEL_REPO="swiss-ai"

export PROJECT_NAME="apertus-benchmarks"
export EXPERIMENT_NAME="${MODEL_NAME}-verl-sglang-megatron-v1-separate-async-${SLURM_JOB_NUM_NODES}n"
export RUN_NAME="${EXPERIMENT_NAME}-run-${SLURM_JOB_ID}"
export TRAINING_HOME=/capstor/scratch/cscs/${USER}/RL/${MODEL_NAME}
export TRAINING_CONFIG=/tmp
export CHECKPOINT_HOME=${TRAINING_HOME}/checkpoints/${EXPERIMENT_NAME}-run-${SLURM_JOB_ID} #remove "run-${SLURM_JOB_ID}" to enable checkpoint resuming


mkdir -p $TRAINING_HOME
cd $TRAINING_HOME



export ROLLOUT_NNODES=$(python3 -c "import math; print(max(1, math.ceil($SLURM_JOB_NUM_NODES * 0.25)))")
export TRAINING_NNODES=$(( SLURM_JOB_NUM_NODES - ROLLOUT_NNODES ))

# V1 separate-async batching contract:
#   data.train_batch_size == trainer.v1.separate_async.parameter_sync_step * actor.ppo_mini_batch_size
# parameter_sync_step is the number of actor updates between two weight syncs to
# the standalone rollout (the fully-async recipe called this trigger_parameter_sync_step).
# train_batch_size and ppo_mini_batch_size are PROMPT counts; ppo_mini_batch_size is
# multiplied by rollout.n internally for the DP-parallel actor mini-batch, and that
# product must be >= 2x dp_size. Here dp_size = DP x EP = 6 x 1 = 6 (12 training
# nodes x 4 GPUs / TP=8 = 6 DP replicas; dense model so EP=1). 6 * 16 = 96 >> 12.
export ROLLOUT_N=16                  # responses per prompt (unchanged from the fully-async recipe)
export PPO_MINI_BATCH_SIZE=6         # prompts; x ROLLOUT_N = 96 rows
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
HF_TOKEN = "$(cat ~/HF_TOKEN)"
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

actor_rollout_ref:
  model:
    path: ${TRAINING_HOME}/models/${MODEL_NAME}
    use_remove_padding: True  # Megatron THD layout requires sequence packing
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
    ppo_mini_batch_size: ${PPO_MINI_BATCH_SIZE}
    ppo_micro_batch_size_per_gpu: 1
    ppo_max_token_len_per_gpu: 16384
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
    temperature: 1.0
    n: ${ROLLOUT_N} # responses per prompt -- GRPO group size for the relative-advantage baseline
    tensor_model_parallel_size: 4
    gpu_memory_utilization: 0.75
    calculate_log_probs: True   # required: bypass_mode reads rollout_log_probs as old_log_probs
    log_prob_use_dynamic_bsz: True
    checkpoint_engine:
      # dense 70B via NCCL broadcast -- the MoE-specific delta_sharded backend the
      # GLM recipe switched to (for its ~6000-collective full-model gather hang)
      # does not apply here.
      backend: nccl

  ref:
    log_prob_use_dynamic_bsz: True
    log_prob_max_token_len_per_gpu: 16384
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
  # Fixed step cap (overrides the epoch-based count) -- short benchmark run to see
  # whether the loss/reward trend moves, same intent as the fully-async recipe's
  # total_rollout_steps: 4416 (~46 training steps).
  total_training_steps: 46
  project_name: ${PROJECT_NAME}
  experiment_name: ${RUN_NAME}
  nnodes: ${TRAINING_NNODES}
  n_gpus_per_node: 4
  save_freq: 100  # > total_training_steps: no mid-run checkpoint on the shakedown (faster, cleaner signal). Lower for a real run (actor.checkpoint.strict is False so the multimodal-tower export gap is non-fatal).
  test_freq: -1   # disable validation
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
#
# Robust answer extraction. Run 3246622's [REWARD-DUMP] blocks showed
# Apertus-v1.5-70B solves GSM8K fluently and correctly but NEVER emits the
# <answer>...</answer> tags this recipe's original prompt asked for -- it ends
# with "Final answer: N", "\boxed{N}", a bare trailing number, or "N" right
# after the <|inner_suffix|> reasoning delimiter. The old reward only credited
# <answer> tags, so every rollout scored 0 -> no GRPO signal. This version
# parses all of those forms so outcome reward drives training. (The dataset /
# system prompt are unchanged -- no regen needed; RL just rewards correctness.)
import re
import math
from typing import Optional

_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_FINAL = re.compile(
    r"(?:final answer|the answer is|answer is|answer:)\D{0,24}(-?\d[\d,]*(?:\.\d+)?)",
    re.IGNORECASE,
)


def _norm(raw: str) -> Optional[str]:
    raw = raw.strip().replace(chr(36), "").replace(",", "").rstrip(".").strip()
    if not raw:
        return None
    try:
        val = float(raw)
    except ValueError:
        return raw
    if not math.isfinite(val):
        return str(val)
    return str(int(val)) if val == int(val) else str(val)


def _boxed(s: str) -> Optional[str]:
    i = s.rfind("boxed")
    if i < 0:
        return None
    j = s.find("{", i)
    k = s.find("}", j) if j >= 0 else -1
    return s[j + 1:k] if 0 <= j < k else None


def extract_model_answer(response: str) -> Optional[str]:
    # Apertus-v1.5 puts its final answer after the last <|inner_suffix|> delimiter.
    marker = "<|inner_suffix|>"
    tail = response.split(marker)[-1] if marker in response else response

    for scope in (tail, response):
        m = re.findall(r"<answer>(.*?)</answer>", scope, re.DOTALL)
        if m:
            return _norm(m[-1])
    for scope in (tail, response):
        b = _boxed(scope)
        if b is not None:
            return _norm(b)
    for scope in (tail, response):
        m = _FINAL.findall(scope)
        if m:
            return _norm(m[-1])
    for scope in (tail, response):
        nums = _NUM.findall(scope)
        if nums:
            return _norm(nums[-1])
    return None


def compute_reward(
    data_source, solution_str, ground_truth, extra_info=None, **kwargs
) -> float:
    # Reasoning opened but never closed (truncated) -> neutral 0.0, not a big
    # negative, to avoid extreme GRPO advantages that cause gradient spikes.
    if "<think>" in solution_str and "</think>" not in solution_str:
        return 0.0

    model_ans = extract_model_answer(solution_str)
    gt = _norm(str(ground_truth))
    outcome_reward = 1.0 if (model_ans is not None and gt is not None and model_ans == gt) else 0.0

    # Small shaping bonus for a clearly-delimited final answer (any of the forms
    # the model actually uses).
    has_delim = (
        "boxed{" in solution_str
        or ("<answer>" in solution_str and "</answer>" in solution_str)
        or bool(_FINAL.search(solution_str))
    )
    format_reward = 0.1 if has_delim else 0.0

    # Smooth length penalty starting at 350 words, max -0.2 at 700 words.
    words = len(solution_str.split())
    length_penalty = -0.2 * min(1.0, max(0.0, (words - 350) / 350))

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
# 3129805 -- see Known hazards in CLAUDE.md). Applied via git apply on the baked
# v0.9.0 verl tree below, alongside the upstream PR patches -- three real, stable
# fixes for the V1 separate-async trainer (hybrid-rollout OOM, stale hybrid
# weight-sync call, a shape-equality bug in TensorDict construction).
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
# Group 2: Megatron support. Apertus 1.5 needs forked Megatron-LM/
# Megatron-Bridge (wqwqazwsxedc, apertus branch) — the image's stock
# megatron-bridge>=0.5.1/megatron-core>=0.17.0 (see Containerfile) have no
# apertus1p5 support, same story as transformers/SGLang above. Clone +
# checkout once on a single node into Lustre (never per-node inside the main
# srun — see "Never fetch per-node from the internet inside the srun" in
# CLAUDE.md).
export MEGATRON_LM_DIR=${TRAINING_HOME}/Megatron-LM
export MEGATRON_BRIDGE_DIR=${MEGATRON_LM_DIR}/Megatron-Bridge
if [ ! -d "${MEGATRON_LM_DIR}/.git" ]; then
    echo "Cloning wqwqazwsxedc/Megatron-LM + Megatron-Bridge (apertus branches)..."
    srun --mpi=pmix --network=disable_rdzv_get -N 1 --ntasks=1 -u \
        --environment="${TRAINING_CONFIG}/env.toml" \
        --container-writable bash -c '
        set -e
        rm -rf ${MEGATRON_LM_DIR}
        git clone https://github.com/wqwqazwsxedc/Megatron-LM.git ${MEGATRON_LM_DIR}
        git -C ${MEGATRON_LM_DIR} checkout apertus
        git clone https://github.com/wqwqazwsxedc/Megatron-Bridge.git ${MEGATRON_BRIDGE_DIR}
        git -C ${MEGATRON_BRIDGE_DIR} checkout apertus
    '
    [ -d "${MEGATRON_LM_DIR}/.git" ] && [ -d "${MEGATRON_BRIDGE_DIR}/.git" ] \
        || { echo "FATAL: Megatron-LM/Megatron-Bridge clone failed"; exit 1; }
else
    echo "Megatron-LM/Megatron-Bridge already cloned, skipping."
fi

# Megatron-Bridge's own vendored qwen3_asr HF code (a local copy it keeps to
# avoid a transformers/torch version conflict with the official qwen3-asr
# package) unconditionally does AutoConfig.register/AutoModel.register/
# AutoProcessor.register for "qwen3_asr" with the default exist_ok=False. The
# swiss-ai transformers fork installed above already registers "qwen3_asr"
# itself (AutoModel/AutoProcessor's own _LazyAutoMapping.register matches by
# config class __name__, not object identity, so a same-named-but-distinct
# Qwen3ASRConfig class still collides) -- every Megatron worker rank raises
# ValueError: 'qwen3_asr'/'<Qwen3ASRConfig>' is already used by a Transformers
# config/model at actor_init_model(). Run 3141796 hit the AutoConfig.register
# collision; fixing only that one still left AutoModel.register colliding
# next (run 3144663) -- all three calls needed the fix, not just the first.
# This is a different vendored copy from the one SGLang PR #32979 already
# fixes for the rollout side (see CLAUDE.md) -- that fix does not cover this
# one. Patched once on the batch host since this is a single Lustre checkout
# shared by every node via `pip install -e`, not a per-node dist-packages
# copy; grep-guarded so reruns against an already-cloned checkout don't need
# to re-patch.
QWEN3_ASR_INIT="${MEGATRON_BRIDGE_DIR}/src/megatron/bridge/models/qwen3_asr/hf_qwen3_asr/__init__.py"
[ -f "${QWEN3_ASR_INIT}" ] \
    || { echo "FATAL: expected qwen3_asr __init__.py not found at ${QWEN3_ASR_INIT}"; exit 1; }
# Run the sed unconditionally instead of skipping on a single-line "already
# patched" check (run 3149242: a stale Lustre checkout left over from an
# earlier run had only the AutoConfig.register line patched -- from the
# 3141796 fix, before the 3144663 fix extended it to all three calls -- so the
# old guard's single grep matched, the block was skipped entirely, and the
# still-unpatched AutoModel.register call crashed actor_init_model() on every
# rank again). Each sed pattern only matches its call's un-patched
# (exist_ok=True-less) form, so this is idempotent whether zero, one, two, or
# all three calls were already patched.
sed -i \
    -e 's/AutoConfig\.register("qwen3_asr", Qwen3ASRConfig)$/AutoConfig.register("qwen3_asr", Qwen3ASRConfig, exist_ok=True)/' \
    -e 's/AutoModel\.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)$/AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration, exist_ok=True)/' \
    -e 's/AutoProcessor\.register(Qwen3ASRConfig, Qwen3ASRProcessor)$/AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor, exist_ok=True)/' \
    "${QWEN3_ASR_INIT}"
grep -q 'AutoConfig.register("qwen3_asr", Qwen3ASRConfig, exist_ok=True)' "${QWEN3_ASR_INIT}" \
    && grep -q 'AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration, exist_ok=True)' "${QWEN3_ASR_INIT}" \
    && grep -q 'AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor, exist_ok=True)' "${QWEN3_ASR_INIT}" \
    || { echo "FATAL: one or more qwen3_asr Auto*.register patches failed to apply"; exit 1; }
echo "Patched (or confirmed already-patched) qwen3_asr AutoConfig/AutoModel/AutoProcessor.register exist_ok=True in Megatron-Bridge checkout."

# ══════════════════════════════════════════════════════════════════════════
# Add Apertus 1.5 support to Megatron-Bridge itself. wqwqazwsxedc/Megatron-Bridge
# (apertus branch) never implemented a bridge for
# Apertus1p5ForConditionalGeneration -- confirmed against its git history: the
# whole branch is stock NVIDIA r0.5.0 plus exactly one commit ("add apertus")
# that only adds a bridge for the older, text-only ApertusForCausalLM.
# AutoBridge.from_hf_pretrained() raises "Model architecture
# 'Apertus1p5ForConditionalGeneration' is not yet supported" (run 3149776)
# without this. New bridge module, adapted from apertus/apertus_bridge.py --
# see the file's own docstring below for the full architectural comparison and
# the vision/audio-weights and pruned-LM-head caveats. Checked-in copy:
# apertus-benchmarks/patches/apertus1p5_bridge.py -- the two must be kept in
# sync by hand (same convention as sitecustomize-verl-v0.9.0.py in the GLM
# script; diff them before trusting either). Written once on the batch host,
# before the wheel-build step, so the new file is what gets packaged.
mkdir -p "${MEGATRON_BRIDGE_DIR}/src/megatron/bridge/models/apertus1p5"
touch "${MEGATRON_BRIDGE_DIR}/src/megatron/bridge/models/apertus1p5/__init__.py"
cat > "${MEGATRON_BRIDGE_DIR}/src/megatron/bridge/models/apertus1p5/apertus1p5_bridge.py" <<- 'APERTUS1P5_BRIDGE_EOF'
# Megatron-Bridge bridge for Apertus 1.5 (`Apertus1p5ForConditionalGeneration`).
#
# wqwqazwsxedc/Megatron-Bridge (apertus branch) only ever added a bridge for the
# older, text-only `ApertusForCausalLM` (see apertus/apertus_bridge.py) -- there is
# no apertus1p5 bridge anywhere in that fork (confirmed against its git history:
# the entire branch is stock NVIDIA r0.5.0 plus one commit, "add apertus", which
# touches only apertus/{__init__.py,apertus_bridge.py} and a functional test).
# AutoBridge.from_hf_pretrained() therefore raises "Model architecture
# 'Apertus1p5ForConditionalGeneration' is not yet supported" (run 3149776) -- this
# is a real, total gap, not a registration bug fixable with a one-line patch.
#
# Feasibility: swiss-ai/transformers' Apertus1p5TextModel (the actual language
# backbone) is architecturally identical to plain Apertus -- same xielu MLP
# (up_proj/down_proj only, no gate), same q_norm/k_norm RMSNorm on attention,
# same attention_bias=False, same llama3-style rope_parameters dict shape
# (rope_type/rope_theta/factor/original_max_position_embeddings/low_freq_factor/
# high_freq_factor) -- confirmed by reading modeling_apertus1p5.py at the exact
# commit this script pins (SWISS_AI_TRANSFORMERS_SHA). It only differs in two
# structural ways:
#   1. Hyperparameters live under `config.text_config` (an Apertus1p5TextConfig),
#      not the top-level Apertus1p5Config -- the top level also carries
#      vision_config/audio_config for the multimodal tokenizers this benchmark
#      never uses.
#   2. The decoder stack sits one level deeper in the HF module tree:
#      Apertus1p5ForConditionalGeneration.model.language_model.{embed_tokens,
#      layers,norm} instead of plain Apertus's Apertus...ForCausalLM.model.*.
#      `lm_head` itself stays at the top level, tied to
#      model.language_model.embed_tokens.weight (per _tied_weights_keys) --
#      exactly analogous to plain Apertus's own lm_head/output_layer tie.
#
# This bridge is therefore adapted directly from apertus/apertus_bridge.py: same
# MCoreXIELU activation and get_apertus_decoder_block_spec (imported, not
# duplicated, so behavior stays identical to the one upstream commit that
# actually exists), same rope/attention_bias/hidden_act validation, only the HF
# config source (text_config instead of the top-level config) and the HF-side
# weight key prefix (model.language_model.* instead of model.*) change.
#
# Deliberately NOT mapped: vision_tower/audio_tower/vision_tokenizer/
# audio_tokenizer weights. Megatron's target here is plain GPTModel (text-only --
# see "want to take advantage of Megatron parallelism" in the dispatching
# conversation, not multimodal fidelity), which has no parameter slots for them
# anyway, and this benchmark (text-only GSM8K GRPO) never calls
# get_image_feature/get_audio_feature -- same reasoning already established for
# the SGLang side of this script (see apertus-benchmarks/patches/
# sglang-apertus1p5-local-fixes.patch point 6 in CLAUDE.md: those weights never
# need to be numerically correct for this benchmark, only absent-is-fine). No
# strict/leftover-key check was found anywhere in model_bridge.py's weight-load
# path, so simply omitting these mappings is sufficient -- there is nothing to
# suppress.
#
# Pruned LM head: confirmed (run 3171151) that swiss-ai/Apertus-v1.5-70B does
# use a pruned head -- output_vocab_size=131072 vs the extended
# vocab_size=266752 that also covers the visual/audio token ranges. An
# earlier version of this bridge raised rather than risk mismapping shapes;
# now handled properly since this benchmark (text-only GSM8K GRPO) never
# produces or consumes a token id outside [0, output_vocab_size) anyway --
# `output_vocab_size`'s own docstring in configuration_apertus1p5.py confirms
# ids `0..output_vocab_size - 1` are exactly the retained/generatable
# (non-multimodal) ones. Megatron-core's GPTModelProvider only has one
# `vocab_size` sizing both tables, so provider_bridge sets it to
# `output_vocab_size` (not the full extended vocab_size) whenever the two
# differ, and mapping_registry truncates the *input* embedding table
# (`model.language_model.embed_tokens.weight`, shape (266752, hidden) in the
# checkpoint) down to its first `output_vocab_size` rows to match -- the
# checkpoint's `lm_head.weight` is already exactly (131072, hidden) since it
# really is a physically pruned Linear layer (see
# Apertus1p5ForConditionalGeneration.__init__), so the output-side mapping
# needs no such transform. `_TruncatedVocabEmbeddingMapping` below also
# best-effort zero-pads back out to the full vocab_size on the reverse
# (megatron_to_hf) direction, for whatever HF-format-export path might call
# it -- those padded rows carry no real multimodal embedding, same caveat as
# megatron_to_hf_config below.
#
# megatron_to_hf_config below (used for full HF-format checkpoint export, not
# for the live NCCL weight sync this benchmark actually exercises during
# training) is a best-effort adaptation nesting the generic flat CONFIG_MAPPING
# output under "text_config" to match Apertus1p5Config's real shape -- lower
# confidence than the forward (provider_bridge/mapping_registry) path, since it
# is not needed to get past this benchmark's first training step and so was not
# a priority to verify.

from __future__ import annotations

import torch
from megatron.bridge.models.apertus.apertus_bridge import MCoreXIELU, get_apertus_decoder_block_spec
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ColumnParallelMapping,
    QKVMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.conversion.utils import unwrap_model
from megatron.core.models.gpt.gpt_model import GPTModel
from transformers import Apertus1p5ForConditionalGeneration


class _TruncatedVocabEmbeddingMapping(AutoMapping):
    """AutoMapping variant for a pruned-LM-head checkpoint's input embedding table.

    HF's `embed_tokens.weight` covers the full extended vocab_size (text plus
    multimodal token ids); Megatron-core's single `vocab_size` here is set to
    the narrower `output_vocab_size` (see provider_bridge and the module
    docstring's "Pruned LM head" section) so it matches the checkpoint's
    already-pruned `lm_head.weight`. hf_to_megatron truncates the source
    tensor to the first output_vocab_size rows before handing off to the
    normal VocabParallelEmbedding sharding logic; megatron_to_hf best-effort
    zero-pads back out to the full width (those rows carry no real multimodal
    embedding -- acceptable since this bridge's HF-export path is already
    lower-priority, see megatron_to_hf_config below).
    """

    def __init__(self, megatron_param, hf_param, output_vocab_size, full_vocab_size, permute_dims=None):
        super().__init__(megatron_param, hf_param, permute_dims)
        self._output_vocab_size = output_vocab_size
        self._full_vocab_size = full_vocab_size

    def hf_to_megatron(self, hf_weights, megatron_module):
        if hf_weights.shape[0] > self._output_vocab_size:
            hf_weights = hf_weights[: self._output_vocab_size].contiguous()
        return super().hf_to_megatron(hf_weights, megatron_module)

    def megatron_to_hf(self, megatron_weights, megatron_module):
        result = super().megatron_to_hf(megatron_weights, megatron_module)
        if not result:
            return result
        key = next(iter(result))
        value = result[key]
        if value.shape[0] < self._full_vocab_size:
            pad = value.new_zeros((self._full_vocab_size - value.shape[0], *value.shape[1:]))
            value = torch.cat([value, pad], dim=0)
        return {key: value}

    def resolve(self, captures):
        resolved_megatron_param, resolved_hf_param = self._resolve_names(captures)
        return type(self)(
            resolved_megatron_param,
            resolved_hf_param,
            self._output_vocab_size,
            self._full_vocab_size,
            self.permute_dims,
        )

_ROPE_DEFAULTS = {
    "rope_type": "llama3",
    "original_max_position_embeddings": 8192,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
}


class _TextConfigOnlyShim:
    """Minimal stand-in for an hf_pretrained wrapper, exposing only `.config`.

    MegatronModelBridge.provider_bridge() (the base implementation we delegate
    to below) only ever reads `hf_pretrained.config` -- this lets us hand it
    `Apertus1p5Config.text_config` directly instead of the top-level config
    that also carries the unrelated vision_config/audio_config blocks.
    """

    def __init__(self, config):
        self.config = config


@MegatronModelBridge.register_bridge(source=Apertus1p5ForConditionalGeneration, target=GPTModel, model_type="apertus1p5")
class Apertus1p5Bridge(MegatronModelBridge):
    @classmethod
    def hf_to_megatron_activation(cls, hidden_act: str):
        if hidden_act != "xielu":
            return super().hf_to_megatron_activation(hidden_act)
        return lambda _: (_ for _ in ()).throw(RuntimeError("expected MCoreXIELU"))

    def provider_bridge(self, hf_pretrained):
        text_config = hf_pretrained.config.text_config
        provider = super().provider_bridge(_TextConfigOnlyShim(text_config))

        output_vocab_size = getattr(text_config, "output_vocab_size", None)
        if output_vocab_size is not None and output_vocab_size != text_config.vocab_size:
            # Pruned LM head: size Megatron's single vocab_size to the narrower,
            # physically-real output_vocab_size (matching the checkpoint's actual
            # lm_head.weight shape) rather than the full extended vocab_size --
            # see the module docstring's "Pruned LM head" section. mapping_registry
            # truncates the embedding table to match.
            provider.vocab_size = output_vocab_size
            provider.make_vocab_size_divisible_by = self.make_vocab_size_divisible_by(output_vocab_size)

        rope = {
            **(getattr(text_config, "rope_scaling", None) or {}),
            **(getattr(text_config, "rope_parameters", None) or {}),
        }
        rope_type = rope.get("rope_type", rope.get("type", "llama3"))
        factor = float(rope.get("factor", 1.0))
        theta = float(rope.get("rope_theta", getattr(text_config, "rope_theta", 10000.0)))
        if text_config.hidden_act != "xielu":
            raise ValueError(f"Expected hidden_act='xielu', got {text_config.hidden_act!r}")
        if text_config.attention_bias:
            raise ValueError("Apertus1p5 attention_bias=True is unsupported")
        if rope_type != "llama3":
            raise ValueError(f"Unsupported Apertus1p5 RoPE type: {rope_type!r}")

        provider.apertus_rope_scaling = {
            "rope_type": rope_type,
            "type": rope_type,
            "factor": factor,
            "original_max_position_embeddings": int(rope.get("original_max_position_embeddings", 8192)),
            "low_freq_factor": float(rope.get("low_freq_factor", 1.0)),
            "high_freq_factor": float(rope.get("high_freq_factor", 4.0)),
        }
        provider.normalization = "RMSNorm"
        provider.qk_layernorm = True
        provider.gated_linear_unit = False
        provider.use_te_activation_func = False
        provider.bias_activation_fusion = False
        provider.add_bias_linear = False
        provider.add_qkv_bias = False
        provider.hidden_dropout = 0.0
        provider.rotary_interleaved = False
        provider.position_embedding_type = "rope"
        provider.rotary_base = theta
        provider.rope_scaling = True
        provider.rope_scaling_factor = factor
        provider.transformer_layer_spec = get_apertus_decoder_block_spec
        return provider

    def load_weights_hf_to_megatron(self, hf_pretrained, megatron_model, allowed_mismatched_params=None):
        models = super().load_weights_hf_to_megatron(
            hf_pretrained, megatron_model, allowed_mismatched_params=allowed_mismatched_params
        )
        [
            m._sync_runtime_scalars()
            for model in unwrap_model(models)
            for m in model.modules()
            if isinstance(m, MCoreXIELU)
        ]
        return models

    @classmethod
    def megatron_to_hf_config(cls, provider) -> dict:
        text_config = super().megatron_to_hf_config(provider)
        theta = float(provider.rotary_base)
        rope = {
            **(getattr(provider, "apertus_rope_scaling", None) or _ROPE_DEFAULTS),
            "factor": float(provider.rope_scaling_factor),
        }
        rope["rope_type"] = rope["type"] = rope.get("rope_type", rope.get("type", "llama3"))
        text_config.update(
            hidden_act="xielu",
            attention_bias=False,
            rope_theta=theta,
            rope_scaling=rope,
            rope_parameters={**rope, "rope_theta": theta},
        )
        return {
            "model_type": "apertus1p5",
            "architectures": ["Apertus1p5ForConditionalGeneration"],
            "text_config": text_config,
        }

    def mapping_registry(self) -> MegatronMappingRegistry:
        text_config = self.hf_config.text_config
        output_vocab_size = getattr(text_config, "output_vocab_size", None)
        full_vocab_size = text_config.vocab_size

        L, H = "decoder.layers.*", "model.language_model.layers.*"
        if output_vocab_size is not None and output_vocab_size != full_vocab_size:
            embedding_mapping = _TruncatedVocabEmbeddingMapping(
                "embedding.word_embeddings.weight",
                "model.language_model.embed_tokens.weight",
                output_vocab_size,
                full_vocab_size,
            )
        else:
            embedding_mapping = AutoMapping(
                "embedding.word_embeddings.weight", "model.language_model.embed_tokens.weight"
            )

        auto = {
            "output_layer.weight": "lm_head.weight",
            "decoder.final_layernorm.weight": "model.language_model.norm.weight",
            f"{L}.self_attention.linear_proj.weight": f"{H}.self_attn.o_proj.weight",
            f"{L}.self_attention.q_layernorm.weight": f"{H}.self_attn.q_norm.weight",
            f"{L}.self_attention.k_layernorm.weight": f"{H}.self_attn.k_norm.weight",
            f"{L}.mlp.linear_fc2.weight": f"{H}.mlp.down_proj.weight",
        }
        # Pre-attn / pre-MLP RMSNorm weights + the learnable xIELU scalars, mapped
        # replicated (not TP-sharded). This recipe's use_transformer_engine spec
        # produces the fused `linear_qkv.layer_norm_weight` / `linear_fc1.layer_norm_weight`
        # names, so those direct mappings are what load. Validated end-to-end in run
        # 3247540 (46 steps, healthy GRPO curve) -- and confirmed against fork
        # megatron-core 0.18 / megatron-bridge 0.5.0; switching these to AutoMapping
        # on stock megatron-bridge 0.6.1 loaded them as ~0 instead (run 3243271).
        repl = {
            f"{L}.self_attention.linear_qkv.layer_norm_weight": f"{H}.attention_layernorm.weight",
            f"{L}.mlp.linear_fc1.layer_norm_weight": f"{H}.feedforward_layernorm.weight",
            **{f"{L}.mlp.activation_func.{n}": f"{H}.mlp.act_fn.{n}" for n in ("alpha_p", "alpha_n", "beta", "eps")},
        }
        qkv = QKVMapping(
            f"{L}.self_attention.linear_qkv.weight",
            q=f"{H}.self_attn.q_proj.weight",
            k=f"{H}.self_attn.k_proj.weight",
            v=f"{H}.self_attn.v_proj.weight",
        )
        qkv._tp_mapping = ColumnParallelMapping(qkv.megatron_param, qkv.megatron_param)
        return MegatronMappingRegistry(
            embedding_mapping,
            *(AutoMapping(m, h) for m, h in auto.items()),
            *(ReplicatedMapping(m, h) for m, h in repl.items()),
            qkv,
            ColumnParallelMapping(f"{L}.mlp.linear_fc1.weight", f"{H}.mlp.up_proj.weight"),
        )
APERTUS1P5_BRIDGE_EOF
MEGATRON_BRIDGE_MODELS_INIT="${MEGATRON_BRIDGE_DIR}/src/megatron/bridge/models/__init__.py"
[ -f "${MEGATRON_BRIDGE_MODELS_INIT}" ] \
    || { echo "FATAL: expected Megatron-Bridge models/__init__.py not found at ${MEGATRON_BRIDGE_MODELS_INIT}"; exit 1; }
grep -q "from megatron.bridge.models.apertus1p5.apertus1p5_bridge import Apertus1p5Bridge" "${MEGATRON_BRIDGE_MODELS_INIT}" \
    || echo "from megatron.bridge.models.apertus1p5.apertus1p5_bridge import Apertus1p5Bridge" >> "${MEGATRON_BRIDGE_MODELS_INIT}"
grep -q "from megatron.bridge.models.apertus1p5.apertus1p5_bridge import Apertus1p5Bridge" "${MEGATRON_BRIDGE_MODELS_INIT}" \
    || { echo "FATAL: failed to register Apertus1p5Bridge import in Megatron-Bridge models/__init__.py"; exit 1; }
echo "Added Apertus1p5Bridge (new bridge for Apertus1p5ForConditionalGeneration) to Megatron-Bridge checkout."

# Build megatron-core + megatron-bridge into wheels once, on a single node,
# rather than having each of the 16 main-srun nodes run `pip install -e`
# directly against the same shared Lustre checkout. megatron-core's setup.py
# compiles a C++ extension (the `datasets` helpers) in place inside the
# source tree; 16 nodes doing that concurrently against one shared directory
# raced on the build/copy step (run 3149726: "error: [Errno 2] No such file
# or directory" copying helpers_cpp.*.so -- one node's build stepping on
# another's temp/output files -- which failed the install on at least one
# node and took the other 15 down with it). All 16 nodes are identical GH200
# hardware, so one compile is enough for all of them -- same "build once,
# install everywhere" treatment as the swiss-ai transformers wheel above.
# Built after the qwen3_asr source patch above so the patched source is what
# gets packaged. Package names confirmed from each fork's pyproject.toml:
# "megatron-core" (Megatron-LM) and "megatron-bridge" (Megatron-Bridge).
#
# MEGATRON_WHEEL_BUILD_VERSION + marker file: TRAINING_HOME/wheels persists
# across submissions on Lustre, and a plain "does the wheel file already
# exist" check has already bitten this exact script once (run 3149242: a
# stale, only-partially-patched checkout was skipped instead of re-patched).
# Without a version gate here, adding the Apertus1p5Bridge module above to the
# source tree would silently do nothing on a rerun that reuses wheels built
# before that source change existed -- bump this string whenever the
# packaged source changes in a way that needs a fresh wheel.
export MEGATRON_WHEEL_DIR=${TRAINING_HOME}/wheels
export MEGATRON_WHEEL_BUILD_VERSION="v5-apertus1p5-bridge-v1sep"
mkdir -p ${MEGATRON_WHEEL_DIR}
if ! ls ${MEGATRON_WHEEL_DIR}/megatron_core-*.whl >/dev/null 2>&1 \
    || ! ls ${MEGATRON_WHEEL_DIR}/megatron_bridge-*.whl >/dev/null 2>&1 \
    || [ "$(cat ${MEGATRON_WHEEL_DIR}/build.version 2>/dev/null)" != "${MEGATRON_WHEEL_BUILD_VERSION}" ]; then
    echo "Building megatron-core + megatron-bridge wheels (apertus forks, build version ${MEGATRON_WHEEL_BUILD_VERSION})..."
    srun --mpi=pmix --network=disable_rdzv_get -N 1 --ntasks=1 -u \
        --environment="${TRAINING_CONFIG}/env.toml" \
        --container-writable bash -c '
        set -e
        export PIP_CACHE_DIR=/tmp/pip-cache-wheelbuild-${SLURM_JOB_ID}
        export TMPDIR=/tmp/pip-tmp-wheelbuild-${SLURM_JOB_ID}
        mkdir -p $PIP_CACHE_DIR $TMPDIR /tmp/megatron-wheel
        rm -f ${MEGATRON_WHEEL_DIR}/megatron_core-*.whl ${MEGATRON_WHEEL_DIR}/megatron_bridge-*.whl
        pip wheel --no-build-isolation --no-deps -w /tmp/megatron-wheel ${MEGATRON_LM_DIR}
        # megatron-bridge is built against an importable megatron-core, in case
        # its build backend introspects it -- install the just-built wheel
        # locally on this same node before building the second wheel.
        pip install --no-deps /tmp/megatron-wheel/megatron_core-*.whl
        pip wheel --no-build-isolation --no-deps -w /tmp/megatron-wheel ${MEGATRON_BRIDGE_DIR}
        cp /tmp/megatron-wheel/megatron_core-*.whl /tmp/megatron-wheel/megatron_bridge-*.whl ${MEGATRON_WHEEL_DIR}/
    '
    ls ${MEGATRON_WHEEL_DIR}/megatron_core-*.whl >/dev/null 2>&1 \
        && ls ${MEGATRON_WHEEL_DIR}/megatron_bridge-*.whl >/dev/null 2>&1 \
        || { echo "FATAL: megatron-core/megatron-bridge wheel build failed"; exit 1; }
    echo "${MEGATRON_WHEEL_BUILD_VERSION}" > ${MEGATRON_WHEEL_DIR}/build.version
else
    echo "megatron-core/megatron-bridge wheels already present and up to date (${MEGATRON_WHEEL_BUILD_VERSION}), skipping build."
fi

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


# Download model (skip if already present)
if [ ! -d "${TRAINING_HOME}/models/${MODEL_NAME}" ]; then
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

# Fetch the upstream PR patches once here and sbcast them to every node, instead
# of curl-ing them from inside the srun (run 3124273: per-node fetches gave a
# mixed cluster where only ~45/80 nodes carried a patch). -f makes an HTTP error
# page a hard failure instead of a no-op "patch".
#   PR #7421: DSA / mcore >= 0.16.2 compat (harmless here -- apertus1p5 is not DSA).
#   PR #7422: preserve load_format=dummy in the disaggregated SGLang rollout --
#             load-bearing for separate-async (v0.9.0 async_sglang_server.py flips
#             dummy -> auto for every non-hybrid replica, i.e. the standalone rollout).
#   PR #7423: fix the NCCL deadlock in async disaggregated weight sync.
for pr in 7421 7422 7423; do
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
# versions printed here are the image stock ones; Group 2 below installs the
# apertus forks over them.
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
for p in "${TRAINING_CONFIG}/7421.patch" "${TRAINING_CONFIG}/7422.patch" "${TRAINING_CONFIG}/7423.patch" "${TRAINING_CONFIG}/v1-separate-async-fixes.patch"; do
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

# Install the apertus forks of Megatron-LM and Megatron-Bridge from the
# wheels built once above, over the images stock megatron-bridge/megatron-core.
# Same "Megatron-LM pulls its own deps, Megatron-Bridge deliberately does not"
# split the fork itself documents (--no-deps on Megatron-Bridge only — its
# deps are already satisfied by Megatron-LM + the image).
#
# A prebuilt wheel install is metadata-copy-only, no compilation -- this is
# deliberately NOT `pip install -e <shared-Lustre-dir>` on every node: run
# 3149726 had 16 nodes editable-installing (and therefore each recompiling
# the megatron-core C++ extension) against the same shared checkout
# concurrently, and the build step raced across nodes ("No such file or
# directory" copying the compiled .so, taking the whole job down). See the
# wheel-build step above.
#
# A node-local PIP_CACHE_DIR/TMPDIR: even a wheel install still needs pip
# dependency resolution (for Megatron-LM own deps) off Lustre -- the default
# cache under $HOME hit the same ENOLCK/ESTALE file-locking hazard as
# AutoConfig.from_pretrained elsewhere in this repo when 16 nodes shared it
# concurrently.
export PIP_CACHE_DIR=/tmp/pip-cache-${SLURM_JOB_ID}
export TMPDIR=/tmp/pip-tmp-${SLURM_JOB_ID}
mkdir -p $PIP_CACHE_DIR $TMPDIR
pip install ${MEGATRON_WHEEL_DIR}/megatron_core-*.whl \
    || { echo "FATAL: Megatron-LM (megatron-core) wheel install failed"; exit 1; }
pip install --no-deps ${MEGATRON_WHEEL_DIR}/megatron_bridge-*.whl \
    || { echo "FATAL: Megatron-Bridge wheel install failed"; exit 1; }
python3 -c "import megatron.core; print(\"megatron.core:\", megatron.core.__file__)"

# Patch megatron-bridge safe_config_loader to skip filelock -- same hazard,
# same fix, as the GLM scripts Known hazards entry (see CLAUDE.md): /dev/shm
# and /tmp on CSCS Alps do not support fcntl.flock in the container
# (ENOLCK/ESTALE on every attempt). Confirmed hit here too (run 3171930):
# AutoBridge.from_hf_pretrained -> safe_config_loader raised
# ValueError: Failed to load configuration ... Stale file handle, chained from
# filelock/_unix.py -- a different call site than any already-patched one in
# this script (the qwen3_asr/pruned-vocab/xielu fixes are all upstream of
# this), but the exact same module the GLM script already has a working patch
# for. The lock is unnecessary because the config files are written by
# localid=0 before any reader starts (purely read-only after that). Matches
# every unrelated function/class in the target file -- kept as a
# line-by-line "with filelock." match rather than a regex that tries to
# parse the argument, since FileLock() arguments often contain nested parens.
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
