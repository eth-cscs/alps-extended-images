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
export EXPERIMENT_NAME="${MODEL_NAME}-verl-sglang-megatron-async-${SLURM_JOB_NUM_NODES}n"
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
  - ppo_megatron_trainer
  - override model_engine: megatron
  - override rollout@actor_rollout_ref.rollout: rollout
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
    use_remove_padding: True  # Megatron THD layout requires sequence packing
    # eager, not flash_attention_2/sdpa: the vision tokenizer submodule
    # (instantiated even for this text-only benchmark) supports neither;
    # eager is the one implementation every model supports.
    override_config:
      attn_implementation: eager
    use_shm: false

  actor:
    ppo_mini_batch_size: 48 #must be divisible by (rollout.n_gpus_per_node * rollout.nnodes)
    ppo_micro_batch_size_per_gpu: 1
    ppo_max_token_len_per_gpu: 16384
    use_rollout_log_probs: True   # required for fully-async log prob correctness
    use_dynamic_bsz: True
    # 12 training nodes x 4 GPUs = 48 GPUs; TP=4, PP=1, EP=1 -> 4 GPUs/replica,
    # 12 DP replicas. Apertus-v1.5-70B is dense (no MoE), so EP must be 1.
    # Starting point mirrored from train-gsm8k-qwen-3B-full-async-megatron.sh
    # (same offload strategy that fixed the FSDP2 OOM in the sibling script)
    # — unverified for this model/node count, needs a run.
    megatron:
      tensor_model_parallel_size: 4
      pipeline_model_parallel_size: 1
      expert_model_parallel_size: 1
      param_offload: True
      grad_offload: True
      optimizer_offload: True
      vanilla_mbridge: False  # use the wqwqazwsxedc/Megatron-Bridge apertus fork (Group 2 below)
      override_transformer_config:
        recompute_granularity: full
        recompute_method: uniform
        recompute_num_layers: 1
        use_cpu_initialization: True

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
    log_prob_max_token_len_per_gpu: 16384
    megatron:
      param_offload: True  # keep ref params on CPU when not computing log probs
      tensor_model_parallel_size: 4
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
awk '
    /^diff --git a\/python\/sglang\// { keep = 1 }
    /^diff --git a\/python\/sglang\/srt\/multimodal\/processors\/base_processor\.py/ { keep = 0 }
    /^diff --git a\/test\// { keep = 0 }
    /^diff --git a\/docs_new\// { keep = 0 }
    keep { print }
' ${TRAINING_CONFIG}/sglang-apertus1p5-full.diff > ${TRAINING_CONFIG}/sglang-apertus1p5.diff
[ -s ${TRAINING_CONFIG}/sglang-apertus1p5.diff ] \
    || { echo "FATAL: filtered sglang PR #32979 diff is empty"; exit 1; }
sbcast -f ${TRAINING_CONFIG}/sglang-apertus1p5.diff ${TRAINING_CONFIG}/sglang-apertus1p5.diff

awk '
    /^diff --git a\/python\/sglang\/srt\/multimodal\/processors\/base_processor\.py/ { keep = 1 }
    /^diff --git a\/test\// { keep = 0 }
    keep { print }
' ${TRAINING_CONFIG}/sglang-apertus1p5-full.diff > ${TRAINING_CONFIG}/sglang-apertus1p5-base-processor.diff
sbcast -f ${TRAINING_CONFIG}/sglang-apertus1p5-base-processor.diff ${TRAINING_CONFIG}/sglang-apertus1p5-base-processor.diff

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
export MEGATRON_WHEEL_DIR=${TRAINING_HOME}/wheels
mkdir -p ${MEGATRON_WHEEL_DIR}
if ! ls ${MEGATRON_WHEEL_DIR}/megatron_core-*.whl >/dev/null 2>&1 \
    || ! ls ${MEGATRON_WHEEL_DIR}/megatron_bridge-*.whl >/dev/null 2>&1; then
    echo "Building megatron-core + megatron-bridge wheels (apertus forks)..."
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
else
    echo "megatron-core/megatron-bridge wheels already present, skipping build."
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
git remote add pr_origin https://github.com/theely/verl.git 2>/dev/null || true
git fetch pr_origin Fix-fsdp-model-loading-on-async
git reset --hard pr_origin/Fix-fsdp-model-loading-on-async

# Install Transformers and Safetensors from the swiss-ai wheels
pip install --no-deps ${SWISS_AI_WHEEL_DIR}/transformers-*.whl ${SWISS_AI_WHEEL_DIR}/safetensors-*.whl
python3 -c "import transformers, safetensors; print(\"transformers:\", transformers.__version__, transformers.__file__); print(\"safetensors:\", safetensors.__version__)"

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

# Apply SGLang partch for apertus1p5 model.
patch -p2 -d /usr/local/lib/python3.12/dist-packages < ${TRAINING_CONFIG}/sglang-apertus1p5.diff \
    || { echo "FATAL: sglang PR #32979 patch failed to apply"; exit 1; }
patch -p2 -d /usr/local/lib/python3.12/dist-packages < ${TRAINING_CONFIG}/sglang-apertus1p5-base-processor.diff \
    || echo "WARNING: sglang PR #32979 base_processor.py hunk did not apply (audio-input path, unused here) — continuing"
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
