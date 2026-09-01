#!/bin/bash

#SBATCH --nodes=16
#SBATCH --account=csstaff
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=2:00:00

export VERL_IMAGE="jfrog.svc.cscs.ch/docker-group-csstaff/alps-images/verl:alps7-dev-0f334b540ccc7034" #alps7-dev-0f334b540ccc7034 image with megatron

export MODEL_NAME="Apertus-v1.5-70B"
export MODEL_REPO="swiss-ai"

export PROJECT_NAME="apertus-benchmarks"
export EXPERIMENT_NAME="${MODEL_NAME}-verl-sglang-fsdp2-async-${SLURM_JOB_NUM_NODES}n"
export RUN_NAME="${EXPERIMENT_NAME}-run-${SLURM_JOB_ID}"
export TRAINING_HOME=/capstor/scratch/cscs/${USER}/RL/${MODEL_NAME}
export TRAINING_CONFIG=/tmp
export CHECKPOINT_HOME=${TRAINING_HOME}/checkpoints/${EXPERIMENT_NAME}-run-${SLURM_JOB_ID} #remove "run-${SLURM_JOB_ID}" to enable checkpoint resuming


mkdir -p $TRAINING_HOME
cd $TRAINING_HOME



export ROLLOUT_NNODES=$(python3 -c "import math; print(max(1, math.ceil($SLURM_JOB_NUM_NODES * 0.25)))")
export TRAINING_NNODES=$(( SLURM_JOB_NUM_NODES - ROLLOUT_NNODES ))

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
  - ppo_trainer
  - override rollout@actor_rollout_ref.rollout: rollout
  - override actor@actor_rollout_ref.actor: dp_actor
  - override data@data: legacy_data
  - _self_

# ── Required by fully_async_main ──────────────────────────────────────────────
async_training:
  
  # On Policy Settings
  # staleness_threshold: 0
  # trigger_parameter_sync_step: 1
  
  # Stream Pipeline Settings
  staleness_threshold: 0.1 
  trigger_parameter_sync_step: 2

  require_batches: 1
  partial_rollout: False
  use_trainer_do_validate: False

# Top-level rollout block — fully_async_main copies .nnodes/.n_gpus_per_node
# into actor_rollout_ref.rollout, so keep these in sync with the rollout block below.
rollout:
  nnodes: ${ROLLOUT_NNODES}
  n_gpus_per_node: 4
  total_rollout_steps: 22419
  test_freq: 10
# ──────────────────────────────────────────────────────────────────────────────

data:
  train_files: ${TRAINING_HOME}/data/gsm8k/train.parquet
  val_files:   ${TRAINING_HOME}/data/gsm8k/test.parquet
  train_batch_size: 0    # must be 0 in fully-async mode
  gen_batch_size: 1      # must be 1 in fully-async mode
  return_raw_chat: True

actor_rollout_ref:
  hybrid_engine: False

  model:
    path: ${TRAINING_HOME}/models/${MODEL_NAME}
    # eager, not flash_attention_2/sdpa: the vision tokenizer submodule
    # (instantiated even for this text-only benchmark) supports neither;
    # eager is the one implementation every model supports.
    override_config:
      attn_implementation: eager
    use_shm: false

  actor:
    strategy: fsdp2
    ppo_mini_batch_size: 48 #must be divisible by (rollout.n_gpus_per_node * rollout.nnodes)
    ppo_micro_batch_size_per_gpu: 1
    use_rollout_log_probs: True   # required for fully-async log prob correctness
    use_dynamic_bsz: True
    # verl's default ppo_max_token_len_per_gpu (16384) is sized for
    # flash-attention's O(n) memory; the eager attention above (O(n^2)) OOMs
    # a 71.94B dense model without a much smaller budget and FSDP offload.
    ppo_max_token_len_per_gpu: 4096
    fsdp_config:
      param_offload: True
      optimizer_offload: True
      grad_offload: True

  rollout:
    name: sglang
    mode: async
    load_format: dummy
    n_gpus_per_node: 4
    temperature: 1.0
    n: 16 #num responses per prompt 
    tensor_model_parallel_size: 4
    gpu_memory_utilization: 0.75
    log_prob_use_dynamic_bsz: True
    checkpoint_engine:
      backend: nccl # weight sync via NCCL broadcast

  ref:
    log_prob_use_dynamic_bsz: True

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
  total_epochs: 3
  project_name: ${PROJECT_NAME}
  experiment_name: ${RUN_NAME}
  nnodes: ${TRAINING_NNODES}
  n_gpus_per_node: 4
  save_freq: 50
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

    # Smooth length penalty starting at 350 words (~455 tokens), max -0.2 at 700 words.
    # Keeps thinking chains well below the 1024-token hard cap so truncation stays rare.
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
# file list grows over time as the PR is updated upstream. Run 3152782 (sibling
# Megatron script): between one submission and the next the PR grew from 6 files
# to 10 (adding docs/test files plus a new hunk in detokenizer_manager.py, a
# tool-call-parser trim for a "apertus2509" parser this benchmark never
# enables), and that new hunk failed to apply against the image's pinned SGLang
# version (the exact same "context-sensitive against whatever SGLang version
# the image happens to have" issue already known for base_processor.py) -- a
# fatal-by-default blocklist (keep everything under python/sglang/ except a
# fixed set of known problem files) breaks every time the PR grows a new such
# file. Run 3184895 reproduced this exact failure here too (this script had
# never been updated with the Megatron script's fix). Flipped to an allowlist
# instead: only the files actually confirmed load-bearing for the apertus1p5
# fix are fatal-if-they-fail; everything else the PR touches (test/docs
# excluded entirely; any other python/sglang/ file, currently
# base_processor.py and detokenizer_manager.py) is best-effort -- self-adapting
# to future PR drift instead of needing another manual fix each time the PR
# changes shape.
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

# FSDP2 rank-0-load fix (verl/workers/engine/fsdp/transformer_impl.py): ported from
# theely/verl@Fix-fsdp-model-loading-on-async as a static patch instead of a live
# `git reset --hard` branch swap onto that fork's tip -- the branch swap ran whatever
# commit the fork happened to be on at submit time with no pin, and (unlike every other
# external fix in this script) was never actually diffed against v0.9.0 to confirm it
# was scoped to just this one change. Confirmed via GitHub's compare API that the fork
# branch differs from the real v0.9.0 tag in exactly this one file -- no hidden drift --
# see CLAUDE.md's "Megatron vs FSDP2" section. Same apply-or-already-present-or-fatal
# discipline as the PR patches in the sibling GLM/Megatron scripts. Embedded as a
# heredoc here (not read from a script-relative path) for the same reason as the other
# checked-in patches in this repo -- see CLAUDE.md run 3129805: sbatch stages the
# submitted script elsewhere, so BASH_SOURCE-relative paths don't resolve on the batch
# host. Content of apertus-benchmarks/patches/fsdp2-rank0-load-fix.patch, embedded here;
# the two copies must be kept in sync by hand.
cat > "${TRAINING_CONFIG}/fsdp2-rank0-load-fix.patch" <<- 'EOF'
# FSDP2 rank-0-load fix for verl v0.9.0's actor engine
# (verl/workers/engine/fsdp/transformer_impl.py), ported from
# theely/verl@Fix-fsdp-model-loading-on-async as a static diff instead of a live
# git reset --hard branch swap -- see CLAUDE.md's "Megatron vs FSDP2" section
# for why. Confirmed via GitHub's compare API (verl-project:v0.9.0 vs the fork
# branch tip) that this is the ONLY file that differs between the two: no hidden
# drift from other unrelated commits on the fork branch.
#
# Stock v0.9.0 has every FSDP2 rank independently call auto_class.from_pretrained,
# loading the full model into host RAM on every rank -- for a 70B-param model at
# 4 workers/node that's ~4 x 140 GB = 560 GB per node, an OOM risk this recipe
# cannot afford after the many init-OOM incidents already logged in CLAUDE.md.
# This patch makes only global rank 0 load real weights from disk (via
# from_pretrained); every other rank builds an empty/meta-tensor model via
# accelerate's init_empty_weights(), and fsdp2_load_full_state_dict(...,
# broadcast_from_rank0=True) -- a real, supported PyTorch/verl FSDP2 API, not a
# custom mechanism -- distributes the actual weights afterward.
#
# The rl-bench-apertus-v1.5-70B-sglang-fsdp2-async.sh recipe runs critic.enable:
# false, so the load_valuehead_model branch below (value-head / TRL model
# loading) is present for completeness but is dead code for this recipe -- only
# the plain "language_model" branch and the final full_state/broadcast hunk are
# actually exercised.
diff --git a/verl/workers/engine/fsdp/transformer_impl.py b/verl/workers/engine/fsdp/transformer_impl.py
index 322e78f8141..23bdd07cf1b 100644
--- a/verl/workers/engine/fsdp/transformer_impl.py
+++ b/verl/workers/engine/fsdp/transformer_impl.py
@@ -239,22 +239,42 @@ def _build_module(self):

         torch_dtype = PrecisionType.to_dtype(torch_dtype)

-        init_context = get_init_weight_context_manager(
-            use_meta_tensor=not self.model_config.hf_config.tie_word_embeddings, mesh=self.device_mesh
-        )
-
-        with init_context(), warnings.catch_warnings():
+        # For fsdp2, only rank 0 loads weights from disk; all others receive via
+        # broadcast_from_rank0 in fsdp2_load_full_state_dict. We bypass from_pretrained
+        # entirely for non-rank-0 because init_empty_weights cannot reliably prevent file
+        # I/O in custom models (e.g. ApertusForCausalLM) that override _load_pretrained_model.
+        is_fsdp2 = self.engine_config.strategy == "fsdp2"
+        if is_fsdp2:
+            _is_fsdp2_src = torch.distributed.get_rank() == 0
+
+        # For fsdp1: keep original behaviour (meta tensors when tie_word_embeddings is False)
+        use_meta_tensor = not is_fsdp2 and (not self.model_config.hf_config.tie_word_embeddings)
+        init_context = get_init_weight_context_manager(use_meta_tensor=use_meta_tensor, mesh=self.device_mesh)
+
+        with warnings.catch_warnings():
             warnings.simplefilter("ignore")
+            from accelerate import init_empty_weights

             if self.model_config.model_type == "language_model":
                 auto_class = get_hf_auto_model_class(hf_config=self.model_config.hf_config)

-                module = auto_class.from_pretrained(
-                    pretrained_model_name_or_path=self.model_config.local_path,
-                    torch_dtype=torch_dtype,
-                    config=self.model_config.hf_config,
-                    trust_remote_code=self.model_config.trust_remote_code,
-                )
+                if is_fsdp2 and not _is_fsdp2_src:
+                    # Non-rank-0 for fsdp2: create model structure only.
+                    # Weights are broadcast from rank 0 via fsdp2_load_full_state_dict.
+                    logger.info("fsdp2 non-rank-0: creating model structure from config (no file I/O)")
+                    with init_empty_weights():
+                        module = auto_class.from_config(
+                            self.model_config.hf_config,
+                            trust_remote_code=self.model_config.trust_remote_code,
+                        )
+                else:
+                    with init_context():
+                        module = auto_class.from_pretrained(
+                            pretrained_model_name_or_path=self.model_config.local_path,
+                            torch_dtype=torch_dtype,
+                            config=self.model_config.hf_config,
+                            trust_remote_code=self.model_config.trust_remote_code,
+                        )

                 # Strip sub-modules listed in _verl_strip_modules (e.g.
                 # talker / code2wav for Qwen3-Omni Thinker-only training).
@@ -273,12 +293,56 @@ def _build_module(self):
                 self.model_config.hf_config.classifier_dropout = 0.0
                 self.model_config.hf_config.hidden_dropout = "0"
                 self.model_config.hf_config.summary_dropout_prob = 0.0
-                module = load_valuehead_model(
-                    local_path=self.model_config.local_path,
-                    torch_dtype=torch_dtype,
-                    model_config=self.model_config.hf_config,
-                    trust_remote_code=self.model_config.trust_remote_code,
-                )
+
+                if is_fsdp2 and not _is_fsdp2_src:
+                    # Non-rank-0 for fsdp2: create model structure only.
+                    # Weights are broadcast from rank 0 via fsdp2_load_full_state_dict.
+                    # Receive a flag from rank 0 indicating which branch load_valuehead_model
+                    # took: 0 = AutoModelForTokenClassification, 1 = TRL AutoModelForCausalLMWithValueHead.
+                    logger.info("fsdp2 non-rank-0: creating value model structure from config (no file I/O)")
+                    use_trl = torch.tensor([0], dtype=torch.int32, device="cpu")
+                    torch.distributed.broadcast(use_trl, src=0)
+
+                    from transformers import AutoModelForTokenClassification
+
+                    if not use_trl.item():
+                        with init_empty_weights():
+                            module = AutoModelForTokenClassification.from_config(
+                                self.model_config.hf_config,
+                                trust_remote_code=self.model_config.trust_remote_code,
+                            )
+                    else:
+                        from transformers import AutoModelForCausalLM
+                        from trl import AutoModelForCausalLMWithValueHead
+
+                        from verl.utils.model import AutoModelForVision2Seq as _V2S
+                        from verl.utils.model import patch_valuehead_model
+
+                        base_cls = (
+                            _V2S
+                            if (_V2S is not None and type(self.model_config.hf_config) in _V2S._model_mapping.keys())
+                            else AutoModelForCausalLM
+                        )
+                        with init_empty_weights():
+                            ori_model = base_cls.from_config(
+                                self.model_config.hf_config,
+                                trust_remote_code=self.model_config.trust_remote_code,
+                            )
+                        module = AutoModelForCausalLMWithValueHead.from_pretrained(ori_model)
+                        patch_valuehead_model(module)
+                else:
+                    with init_context():
+                        module = load_valuehead_model(
+                            local_path=self.model_config.local_path,
+                            torch_dtype=torch_dtype,
+                            model_config=self.model_config.hf_config,
+                            trust_remote_code=self.model_config.trust_remote_code,
+                        )
+                    if is_fsdp2:
+                        # Rank 0: broadcast which branch load_valuehead_model took so that
+                        # non-rank-0 ranks can create the matching model structure.
+                        use_trl = torch.tensor([1 if hasattr(module, "v_head") else 0], dtype=torch.int32, device="cpu")
+                        torch.distributed.broadcast(use_trl, src=0)

             use_liger = self.model_config.use_liger
             # Apply Liger kernel; disable fused_linear_cross_entropy (conflicts with verl's forward patching)
@@ -444,7 +508,10 @@ def _build_fsdp_module(self, module):
                 "offload_policy": offload_policy,
                 "reshard_after_forward": self.engine_config.reshard_after_forward,
             }
-            full_state = module.state_dict()
+            # Only rank 0 holds the full state dict; fsdp2_load_full_state_dict
+            # broadcasts it to all other ranks via broadcast_from_rank0=True.
+            # Loading on every rank would OOM for large models (e.g. 4 workers × 140 GB = 560 GB).
+            full_state = module.state_dict() if torch.distributed.get_rank() == 0 else {}
             apply_fsdp2(module, fsdp_kwargs, self.engine_config)
             fsdp2_load_full_state_dict(module, full_state, fsdp_mesh, offload_policy)
         else:
EOF
sbcast -f ${TRAINING_CONFIG}/fsdp2-rank0-load-fix.patch ${TRAINING_CONFIG}/fsdp2-rank0-load-fix.patch

# One-off diagnostic (NOT a fix), applied on top of the rank0-load fix above: checksums
# a small parameter's real value on rank 0 before FSDP2 sharding, then re-gathers the
# same parameter's full value on every rank after fsdp2_load_full_state_dict returns and
# prints both hashes. This is the direct test for the leading hypothesis behind the
# reward-collapse comparison in CLAUDE.md's "Megatron vs FSDP2" section: that the
# rank-0-loads-then-broadcasts path is silently corrupting/mismatching weights on some
# ranks. Remove once the mechanism is confirmed one way or the other -- content of
# apertus-benchmarks/patches/fsdp2-broadcast-diagnostic.patch, embedded here for the
# same script-relative-path reason as the patch above; the two copies must be kept in
# sync by hand.
cat > "${TRAINING_CONFIG}/fsdp2-broadcast-diagnostic.patch" <<- 'EOF'
# One-off diagnostic, NOT a fix. Applies on top of fsdp2-rank0-load-fix.patch
# against verl v0.9.0's verl/workers/engine/fsdp/transformer_impl.py.
#
# Context (see CLAUDE.md's "Megatron vs FSDP2" section): job 3185943 (FSDP2) showed
# zero learning signal over 47 training steps while the Megatron sibling (job
# 3184869) converged cleanly over a comparable step count, with 17% of FSDP2 steps
# showing an exact-zero grad_norm (every sample in the batch got an identical
# GRPO-standardized reward). Leading hypothesis: fsdp2-rank0-load-fix.patch's
# rank-0-loads-then-broadcasts weight loading (only global rank 0 calls
# from_pretrained; every other rank receives its shard via
# fsdp2_load_full_state_dict(..., broadcast_from_rank0=True)) is silently
# corrupting or mismatching weights on some ranks, producing a degenerate policy
# whose rollout samples collapse to identical (hence zero-variance) rewards.
#
# This patch does not confirm or refute that hypothesis by itself -- it adds
# instrumentation to let the next run's log answer it directly: checksums a
# small, deterministically-chosen parameter's true value on rank 0 right before
# sharding (from the real state_dict loaded from disk), then re-gathers that same
# parameter's full value via DTensor.full_tensor() on every rank right after
# fsdp2_load_full_state_dict returns, and prints both SHA-256 hashes. If every
# rank's post-load hash matches rank 0's pre-shard hash, the broadcast is
# reproducing weights correctly and this hypothesis is refuted; if any rank's
# hash differs, that is direct, positive evidence the broadcast/load path is the
# root cause and should be pursued further, not the sampling/reward pipeline.
#
# Remove once the mechanism is confirmed one way or the other -- same
# instrument-once-then-remove discipline as every other diagnostic patch in this
# repo (see CLAUDE.md's GLM section, runs 3137775 through 3144665, for the
# precedent).
diff --git a/verl/workers/engine/fsdp/transformer_impl.py b/verl/workers/engine/fsdp/transformer_impl.py
--- a/verl/workers/engine/fsdp/transformer_impl.py
+++ b/verl/workers/engine/fsdp/transformer_impl.py
@@ -517,8 +517,46 @@
             # broadcasts it to all other ranks via broadcast_from_rank0=True.
             # Loading on every rank would OOM for large models (e.g. 4 workers × 140 GB = 560 GB).
             full_state = module.state_dict() if torch.distributed.get_rank() == 0 else {}
+
+            # [DIAG-FSDP2-BROADCAST] One-off diagnostic: confirm fsdp2_load_full_state_dict's
+            # broadcast_from_rank0 actually reproduces rank 0's real on-disk weights on every
+            # other rank, rather than assuming it from the API's name/docs. Checksum a small,
+            # deterministically-chosen parameter's true value on rank 0 before sharding, then
+            # re-gather the same parameter's full value on every rank after loading and print
+            # both hashes -- any rank whose post-load hash doesn't match rank 0's pre-shard hash
+            # has a corrupted/mismatched broadcast for this parameter.
+            import hashlib
+
+            _diag_param_names = sorted(dict(module.named_parameters()).keys())
+            _diag_param_name = min(
+                _diag_param_names, key=lambda n: dict(module.named_parameters())[n].numel()
+            )
+            if torch.distributed.get_rank() == 0:
+                _diag_pre = full_state[_diag_param_name].detach().float().cpu().contiguous()
+                _diag_pre_hash = hashlib.sha256(_diag_pre.numpy().tobytes()).hexdigest()[:16]
+                print(
+                    f"[DIAG-FSDP2-BROADCAST] rank0 pre-shard param={_diag_param_name} "
+                    f"shape={tuple(_diag_pre.shape)} sha256={_diag_pre_hash}",
+                    flush=True,
+                )
+
             apply_fsdp2(module, fsdp_kwargs, self.engine_config)
             fsdp2_load_full_state_dict(module, full_state, fsdp_mesh, offload_policy)
+
+            _diag_post_param = dict(module.named_parameters())[_diag_param_name]
+            _diag_post_full = (
+                _diag_post_param.full_tensor()
+                if hasattr(_diag_post_param, "full_tensor")
+                else _diag_post_param
+            )
+            _diag_post_full = _diag_post_full.detach().float().cpu().contiguous()
+            _diag_post_hash = hashlib.sha256(_diag_post_full.numpy().tobytes()).hexdigest()[:16]
+            print(
+                f"[DIAG-FSDP2-BROADCAST] rank{torch.distributed.get_rank()} post-load "
+                f"param={_diag_param_name} shape={tuple(_diag_post_full.shape)} "
+                f"sha256={_diag_post_hash}",
+                flush=True,
+            )
         else:
             raise NotImplementedError(f"Unknown strategy {self.engine_config.strategy}")
 
EOF
sbcast -f ${TRAINING_CONFIG}/fsdp2-broadcast-diagnostic.patch ${TRAINING_CONFIG}/fsdp2-broadcast-diagnostic.patch


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



srun --mpi=pmix --network=disable_rdzv_get -N ${SLURM_JOB_NUM_NODES} --ntasks-per-node=1 -u \
    --environment="${TRAINING_CONFIG}/env.toml" \
    --container-writable bash -c '


# Upgrade Verl to v0.9.0.
export VERL_REF=v0.9.0
git -C /workspace/verl fetch --depth 1 origin +refs/tags/${VERL_REF}:refs/tags/${VERL_REF} \
    && git -C /workspace/verl checkout -f ${VERL_REF} \
    || { echo "FATAL: could not check out verl ${VERL_REF}"; exit 1; }
git -C /workspace/verl --no-pager log --oneline -1

# Apply Verl fixes
p="${TRAINING_CONFIG}/fsdp2-rank0-load-fix.patch"
if git -C /workspace/verl apply --check "$p" 2>/dev/null; then
    git -C /workspace/verl apply "$p" && echo "Applied fsdp2-rank0-load-fix.patch on $(hostname)"
elif git -C /workspace/verl apply --reverse --check "$p" 2>/dev/null; then
    echo "fsdp2-rank0-load-fix.patch already present on $(hostname), skipping"
else
    echo "FATAL: fsdp2-rank0-load-fix.patch neither applies nor is already present on $(hostname)"
    exit 1
fi

# One-off diagnostic (NOT a fix), stacked on top of the fix above -- see CLAUDE.md
# "Megatron vs FSDP2" section. Same apply-or-already-present-or-fatal discipline.
p="${TRAINING_CONFIG}/fsdp2-broadcast-diagnostic.patch"
if git -C /workspace/verl apply --check "$p" 2>/dev/null; then
    git -C /workspace/verl apply "$p" && echo "Applied fsdp2-broadcast-diagnostic.patch on $(hostname)"
elif git -C /workspace/verl apply --reverse --check "$p" 2>/dev/null; then
    echo "fsdp2-broadcast-diagnostic.patch already present on $(hostname), skipping"
else
    echo "FATAL: fsdp2-broadcast-diagnostic.patch neither applies nor is already present on $(hostname)"
    exit 1
fi

# Install Transformers and Safetensors from the swiss-ai wheels
pip install --no-deps ${SWISS_AI_WHEEL_DIR}/transformers-*.whl ${SWISS_AI_WHEEL_DIR}/safetensors-*.whl
python3 -c "import transformers, safetensors; print(\"transformers:\", transformers.__version__, transformers.__file__); print(\"safetensors:\", safetensors.__version__)"

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

    HYDRA_FULL_ERROR=1 python -m verl.experimental.fully_async_policy.fully_async_main \
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
