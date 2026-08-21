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
    # eager, not flash_attention_2/sdpa: Apertus-v1.5 instantiates a vision
    # tokenizer submodule (Apertus1p5VisionTokenizerModel) even for this
    # text-only GSM8K benchmark, and that submodule declares neither FA2 (run
    # 3130014) nor SDPA (run 3130169) support — only the language-model
    # backbone does. transformers' own error for the SDPA case points at
    # eager as the fallback; it is the one implementation every model
    # supports (no _supports_eager gate).
    override_config:
      attn_implementation: eager
    use_shm: false

  actor:
    strategy: fsdp2
    ppo_mini_batch_size: 48 #must be divisible by (rollout.n_gpus_per_node * rollout.nnodes)
    ppo_micro_batch_size_per_gpu: 1
    use_rollout_log_probs: True   # required for fully-async log prob correctness
    use_dynamic_bsz: True
    # torch.OutOfMemoryError in actor backward() (run 3138546): 89 GiB already
    # resident on a 95 GiB GPU before an incremental 12.94 GiB allocation.
    # verl's own default ppo_max_token_len_per_gpu (16384, unset here before)
    # is sized for flash-attention's O(n) memory; this model is forced onto
    # eager attention (see the model.override_config comment above — the
    # vision tokenizer submodule supports neither FA2 nor SDPA), whose O(n^2)
    # attention matrices make that budget far too large for a 71.94B dense
    # model with no FSDP offloading. Lowered 4x and paired with param/grad/
    # optimizer offload to trade step latency for headroom; this is a
    # benchmark shakedown run, not a throughput-tuned one.
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

# The swiss-ai/transformers fork pinned below already registers some newer model
# types (e.g. "qwen3_asr") that stock transformers does not. SGLang's own
# compatibility shims (sglang/srt/configs/qwen3_asr.py) call
# AutoConfig.register("qwen3_asr", ...) unconditionally on import (exist_ok
# defaults to False), assuming the running transformers predates that type —
# against this fork that now collides: ValueError: 'qwen3_asr' is already used
# by a Transformers config, pick another name (run 3130211), raised inside the
# FullyAsyncRollouter actor the first time it imports SGLang. Same
# PYTHONPATH/sitecustomize.py mechanism as the GLM v1-separate-async script
# (see CLAUDE.md) — applies cluster-wide, including inside Ray actors, without
# touching verl or the fork's source.
cat > "${TRAINING_CONFIG}/sitecustomize-autoconfig-register.py" <<- 'EOF'
"""
Two independent lazy patches, staged as the process's actual "sitecustomize.py"
via PYTHONPATH (only one file can hold that name, hence both live here):

1. transformers.models.auto.configuration_auto._LazyConfigMapping.register:
   tolerate re-registration of a model type that already exists.

   Why: the swiss-ai/transformers fork used for Apertus-v1.5 already registers
   some newer model types (e.g. "qwen3_asr") that stock/older transformers
   releases do not. SGLang's compatibility shims for those same model types
   (e.g. sglang/srt/configs/qwen3_asr.py) call
   AutoConfig.register(model_type, config) with the default exist_ok=False,
   assuming the running transformers predates the type. Against this fork the
   type is already present, so the call raises ValueError instead of being a
   harmless no-op, crashing every process that imports SGLang (run 3130211,
   repeated in run 3133412 because the mechanism below never actually fired).
   (Also fixed upstream in sgl-project/sglang#32979's own qwen3_asr.py change,
   applied separately as a source patch — this stays as a defensive backstop,
   harmless if the source-level fix already prevents the collision.)

2. sglang.srt.models.apertus_mm._init_component_model (added by
   sgl-project/sglang#32979, applied as a source patch elsewhere in this
   script): unconditionally calls component_config.to_dict(), assuming
   component_config (config.vision_tokenizer_config / .audio_tokenizer_config)
   is a PreTrainedConfig-like object. Against the transformers commit this
   script installs, it is already a plain dict, so that call raises
   AttributeError: 'dict' object has no attribute 'to_dict' (run 3137554) —
   a version-alignment gap between the two independently evolving upstream
   PRs, not something either one's authors got wrong in isolation. Patched to
   accept either shape.

3. sglang.srt.models.apertus_mm.Apertus1p5ForConditionalGeneration.load_weights
   (same PR, same module): once (2) is fixed and the vision tower model
   actually builds, this raises ValueError("No vision/audio tensor matches
   checkpoint key: vision_tower.encoder.down.0.block.0.conv1.bias") — the
   checkpoint's vision-tokenizer weight names do not line up with the
   structure this still-unmerged PR builds (run 3137785), a third alignment
   gap between the checkpoint and this experimental code. Not worth chasing
   exactly: this benchmark is text-only GSM8K and never calls
   get_image_feature/get_audio_feature, so the vision/audio tower weights
   never need to be numerically correct, only present so the model
   instantiates — only the language model weights actually matter. Patched to
   skip an unmatched vision/audio checkpoint key with a warning instead of
   raising; the language-model weight path (everything load_component_weight
   returns False for) is untouched.

Patched via a sys.meta_path find_spec hook that wraps each target module's
real loader — each patch runs only once its target module is actually
imported by something that needs it, same as before this file existed. Two
earlier versions of this file both had real bugs:
  1. A sys.meta_path hook using find_module/load_module — that protocol is
     deprecated since Python 3.4, and its importlib compatibility shim no
     longer calls find_module/load_module at all on the Python 3.11/3.12 in
     this image (confirmed by a local repro where find_module was never
     invoked for an ordinary import). That version never patched anything,
     in any run, including 3133412 where the confirmation print never
     appeared and the qwen3_asr collision happened unchanged.
  2. Importing the target module eagerly, right at sitecustomize time, to
     work around (1). This does make the patch fire (confirmed — the print
     appeared, and the qwen3_asr collision genuinely stopped recurring), but
     it forces transformers/torch's whole import chain into every single
     Python process that inherits this PYTHONPATH — including `ray start`
     itself and Ray's own internal daemon processes, not just verl's actual
     worker/actor processes. Two consecutive runs using that version
     (3133616, 3134586) stalled identically: Ray reports "Connected to Ray
     cluster" and then the FullyAsyncTaskRunner driver actor never starts,
     with a completely frozen log and a Slurm allocation that scontrol shows
     as fully healthy (no NodeFail, all 16 nodes/64 GPUs present) — i.e. an
     application-level Ray actor-scheduling hang, not a cluster/allocation
     problem, and happening twice in a row on fresh node allocations rather
     than looking like one-off flakiness. No run using the eager-import
     patch has ever gotten past that point. This version restores the
     original lazy, import-on-demand behavior using the modern find_spec/
     Loader.exec_module hook (verified locally to actually fire, unlike
     find_module) instead of either of the previous two approaches, so
     neither target is forced into processes that were never going to
     import it anyway.
"""

import sys
import importlib.abc
import importlib.util


def _patch_configuration_auto(mod) -> None:
    _orig_register = mod._LazyConfigMapping.register

    def _register(self, key, value, exist_ok=False):
        return _orig_register(self, key, value, exist_ok=True)

    mod._LazyConfigMapping.register = _register
    print(
        "[sitecustomize-autoconfig-register] patched "
        "_LazyConfigMapping.register to tolerate duplicate model types",
        flush=True,
    )


def _patch_apertus_mm(mod) -> None:
    _orig_init_component_model = mod._init_component_model

    def _init_component_model(component_config, model_cls=None):
        if not hasattr(component_config, "to_dict"):
            config_dict = dict(component_config)
            config = mod.AutoConfig.for_model(
                config_dict.pop("model_type"), **config_dict
            )
            return (
                mod.AutoModel.from_config(config)
                if model_cls is None
                else model_cls(config)
            )
        return _orig_init_component_model(component_config, model_cls=model_cls)

    mod._init_component_model = _init_component_model
    print(
        "[sitecustomize-autoconfig-register] patched "
        "apertus_mm._init_component_model to accept a plain dict "
        "component_config, not just a PreTrainedConfig",
        flush=True,
    )

    # Second, independent bug in the same module (run 3137785): once the model
    # actually builds, load_weights raises ValueError("No vision/audio tensor
    # matches checkpoint key: vision_tower.encoder.down.0.block.0.conv1.bias")
    # — the checkpoint's vision-tokenizer weight names do not line up with the
    # structure this (still-unmerged) PR builds via model_cls(config), a third
    # alignment gap between the checkpoint and this experimental code, not
    # something worth chasing exactly: this benchmark is text-only GSM8K and
    # never calls get_image_feature/get_audio_feature, so the vision/audio
    # tower weights never need to be numerically correct — only the language
    # model weights do. Full copy of the original method (see
    # sgl-project/sglang#32979, python/sglang/srt/models/apertus_mm.py) with
    # the one `raise ValueError` on an unmatched vision/audio key changed to a
    # skip-with-warning instead.
    def load_weights(self, weights):
        component_tensors = {}
        if self.pp_group.is_first_rank:
            for component_name, component in (
                ("vision_tower", self.vision_tower),
                ("audio_tower", self.audio_tower),
            ):
                if component is not None:
                    component_tensors.update(
                        {
                            f"{component_name}.{name}": tensor
                            for name, tensor in component.named_parameters()
                        }
                    )
                    component_tensors.update(
                        {
                            f"{component_name}.{name}": tensor
                            for name, tensor in component.named_buffers()
                        }
                    )

        def remapped_weights():
            for name, weight in weights:
                if name.startswith("model.language_model."):
                    name = "model." + name[len("model.language_model.") :]
                elif name.startswith("model.vision_tokenizer."):
                    name = "vision_tower." + name[len("model.vision_tokenizer.") :]
                elif name.startswith("model.audio_tokenizer."):
                    name = "audio_tower." + name[len("model.audio_tokenizer.") :]
                yield name, weight

        def load_component_weight(name, weight):
            if not name.startswith(("vision_tower.", "audio_tower.")):
                return False
            if not self.pp_group.is_first_rank:
                return True
            component_tensor = component_tensors.get(name)
            if component_tensor is None:
                print(
                    "[sitecustomize-autoconfig-register] WARNING: skipping "
                    f"unmatched vision/audio checkpoint key {name} (vision/audio "
                    "unused by this text-only benchmark)",
                    flush=True,
                )
                return True
            weight_loader = getattr(
                component_tensor, "weight_loader", mod.default_weight_loader
            )
            weight_loader(component_tensor, weight)
            return True

        def filter_language_weights():
            for name, weight in remapped_weights():
                if not load_component_weight(name, weight):
                    yield name, weight

        return super(mod.Apertus1p5ForConditionalGeneration, self).load_weights(
            filter_language_weights()
        )

    mod.Apertus1p5ForConditionalGeneration.load_weights = load_weights
    print(
        "[sitecustomize-autoconfig-register] patched "
        "Apertus1p5ForConditionalGeneration.load_weights to skip unmatched "
        "vision/audio checkpoint keys instead of raising",
        flush=True,
    )


_TARGETS = {
    "transformers.models.auto.configuration_auto": _patch_configuration_auto,
    "sglang.srt.models.apertus_mm": _patch_apertus_mm,
}
_PATCHED: set = set()


class _PatchLoader(importlib.abc.Loader):
    def __init__(self, fullname, orig_loader):
        self._fullname = fullname
        self._orig_loader = orig_loader

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        self._orig_loader.exec_module(module)
        if self._fullname not in _PATCHED:
            _PATCHED.add(self._fullname)
            _TARGETS[self._fullname](module)


class _PatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname not in _TARGETS or fullname in _PATCHED:
            return None
        sys.meta_path.remove(self)
        try:
            real_spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if real_spec is None or real_spec.loader is None:
            return None
        real_spec.loader = _PatchLoader(fullname, real_spec.loader)
        return real_spec


sys.meta_path.insert(0, _PatchFinder())
EOF
sbcast -f ${TRAINING_CONFIG}/sitecustomize-autoconfig-register.py ${TRAINING_CONFIG}/sitecustomize-autoconfig-register.py

# The image's pinned transformers==5.8.1 does not recognize model_type
# "apertus1p5" (Apertus-v1.5's architecture is not merged into any mainline
# transformers release yet). Build the swiss-ai fork commit the model card
# points at into a wheel once, on a single node, and stash it on Lustre —
# every node then does a plain wheel install (unzip + copy, no building, no
# per-node network). Originally this sbcast the ~large source tarball to every
# node directly, but run 3129881 hit "Bus error (core dumped)" from sbcast on
# that tarball, leaving several nodes with a truncated archive
# ("tar: Unexpected EOF") and a broken transformers install
# ("No module named 'transformers.utils'"); nodes where tar did succeed still
# built their own wheel independently inside the main srun — non-deterministic
# (different sha256 per build) and at least one hit
# "OSError: [Errno 116] Stale file handle" mid-install, the same Lustre
# file-locking hazard as the model-config loader (see Known hazards in
# CLAUDE.md). A single ~5MB wheel, built once, sidesteps all three.
#
# Also fetch a safetensors wheel meeting the fork's floor (>=0.8.0,
# dependency_versions_table.py) alongside it: the image's pinned safetensors==0.7.0
# fails transformers/dependency_versions_check.py at import time
# ("ImportError: safetensors>=0.8.0 is required ... but found safetensors==0.7.0",
# run 3129970) once --no-deps stops transformers from pulling it in itself.
# safetensors has no torch/torchvision ABI coupling, so upgrading it doesn't risk
# the ABI --no-deps is otherwise protecting (see the install step below).
#
# SHA is the current head of huggingface/transformers#47662 ("Apertus 1.5: add
# model", branch swiss-ai/transformers@add-apertus1p5), not the commit the model
# card documented — that earlier commit built and installed fine but had a real
# bug (SGLang's generic multimodal loader misloaded a vision-tokenizer weight as
# the text vocab embedding, run 3134672); paired with the SGLang patch below per
# https://github.com/sgl-project/sglang/pull/32979 (adds a real apertus1p5 model
# implementation to SGLang instead of relying on its generic fallback loader).
# The built wheel's own version string is always the same placeholder
# ("5.14.0.dev0" — no git tags for setuptools_scm to key off in a shallow
# tarball checkout), so a plain `ls transformers-*.whl` cache check can't tell
# an old commit's cached wheel from a new one; a marker file recording the SHA
# actually built is checked alongside it so bumping this SHA invalidates the
# Lustre-cached wheel from a previous run instead of silently reusing it.
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

# SGLang has no native apertus1p5 model implementation (mainline or the
# swiss-ai/sglang fork — checked every branch, none has one) so it falls back
# to its generic "transformers" multimodal loader, which misloads a
# vision-tokenizer weight as the text vocab embedding:
# "AssertionError: self.org_vocab_size=266752 ... loaded_weight.shape[output_dim]=131072"
# (run 3134672). Fetch the still-open sgl-project/sglang#32979 ("model: support
# Apertus 1.5") once on the batch host, keep only the six files under
# python/sglang/... that touch production code (drop docs_new/ and test/,
# which do not exist under the installed dist-packages tree and would make
# `patch` fail on those specific hunks), sbcast the trimmed patch, and apply
# it with plain `patch` (not `git apply`) since dist-packages/sglang is not a
# git checkout. -p2 strips the diff's "a/python/..." / "b/python/..." prefix
# down to "sglang/...", matching pip's install layout. Every node's
# --container-writable overlay starts from the same pristine image for this
# job, so — unlike the verl PR-patch pattern elsewhere in this repo — there is
# no cross-run idempotency concern to guard: apply-or-fail is enough.
#
# Split into two diffs, applied with different failure handling: run 3136877
# showed the 5 files that actually matter (the new apertus_mm.py model +
# apertus.py's get_input_embeddings() addition are what fix the vocab_size
# crash) all apply cleanly, but base_processor.py's one-line hunk — adding
# "Apertus1p5Processor" to a set of processor names checked only for *audio*
# inputs — failed on a context mismatch against whatever unpinned sglang
# version the image actually has ("Hunk #1 FAILED at 513"). That hunk is
# unrelated to the crash (this benchmark is text-only, never hits an audio
# code path) and isn't worth making the whole job fatal over, so it is
# fetched/applied separately, best-effort: a warning on failure, not FATAL.
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
# The image clones with --branch ${VERL_REF} --depth 1, so no other ref is present
# locally and the tag has to be fetched explicitly before it can be checked out.
# verl is installed editable (pip install -e) from /workspace/verl, so the checkout
# takes effect without reinstalling; -f discards any dirty state in the clone.
export VERL_REF=v0.9.0
git -C /workspace/verl fetch --depth 1 origin +refs/tags/${VERL_REF}:refs/tags/${VERL_REF} \
    && git -C /workspace/verl checkout -f ${VERL_REF} \
    || { echo "FATAL: could not check out verl ${VERL_REF}"; exit 1; }
git -C /workspace/verl --no-pager log --oneline -1


# Apply Verl fixes
git remote add pr_origin https://github.com/theely/verl.git 2>/dev/null || true
git fetch pr_origin Fix-fsdp-model-loading-on-async
git reset --hard pr_origin/Fix-fsdp-model-loading-on-async


# Install the swiss-ai transformers + safetensors wheels built once above (adds
# the apertus1p5 architecture used by Apertus-v1.5; not yet in any mainline
# transformers release). Plain wheel installs straight off Lustre: no building,
# no per-node network, no isolated build env, so none of run 3129881 failure
# modes apply here. --no-deps: the image already carries a torch/torchvision
# pair whose ABI transformers pip installs are normally pinned against (see
# torch_constraints.txt in the Containerfile); pulling in the
# [torch,vision,audio] extras here could silently upgrade them. Those extras
# are for Apertus-v1.5 image/audio inputs, unused by this text-only GSM8K
# benchmark. safetensors is bumped alongside it to satisfy the forks
# safetensors>=0.8.0 floor (dependency_versions_check.py rejects the images
# pinned safetensors==0.7.0 at import time, run 3129970) — safetensors has no
# torch/torchvision ABI coupling, so this does not carry the same risk.
pip install --no-deps ${SWISS_AI_WHEEL_DIR}/transformers-*.whl ${SWISS_AI_WHEEL_DIR}/safetensors-*.whl
python3 -c "import transformers, safetensors; print(\"transformers:\", transformers.__version__, transformers.__file__); print(\"safetensors:\", safetensors.__version__)"

# Give SGLang a real apertus1p5 model implementation (sgl-project/sglang#32979,
# sbcast above) instead of its generic multimodal fallback loader, which
# misloaded a vision-tokenizer weight as the text vocab embedding (run
# 3134672: AssertionError: self.org_vocab_size=266752 ...
# loaded_weight.shape[output_dim]=131072). dist-packages/sglang is not a git
# checkout, so plain patch(1), not git apply; -p2 strips the diffs
# a/python/sglang/... and b/python/sglang/... prefix down to sglang/...,
# matching the installed layout. A context/version mismatch between this PR
# (based on current sglang main) and whatever unpinned sglang[all] version
# actually got installed at image-build time is possible and is fatal here
# rather than silently continuing on a half-patched, broken install — this is
# the set of files that actually fix the crash (confirmed applying cleanly in
# run 3136877).
patch -p2 -d /usr/local/lib/python3.12/dist-packages < ${TRAINING_CONFIG}/sglang-apertus1p5.diff \
    || { echo "FATAL: sglang PR #32979 patch failed to apply"; exit 1; }

# The base_processor.py hunk (run 3136877: "Hunk #1 FAILED at 513" — context
# mismatch against the installed sglang version) only adds
# "Apertus1p5Processor" to a set of processor names checked on an *audio*
# input code path this text-only GSM8K benchmark never reaches. Best-effort,
# not fatal — a warning here should not sink the whole job the way the patch
# above does.
patch -p2 -d /usr/local/lib/python3.12/dist-packages < ${TRAINING_CONFIG}/sglang-apertus1p5-base-processor.diff \
    || echo "WARNING: sglang PR #32979 base_processor.py hunk did not apply (audio-input path, unused here) — continuing"


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

# Stage the AutoConfig.register patch (sbcast to ${TRAINING_CONFIG} above) as
# this node sitecustomize.py — Python only auto-loads a module literally
# named "sitecustomize" found on sys.path. --ntasks-per-node=1 here, so there
# is exactly one task per node and no cross-task race to guard against.
export SITECUSTOMIZE_LOCAL=/tmp/verl_patches_${SLURM_JOB_ID}
mkdir -p $SITECUSTOMIZE_LOCAL
cp "${TRAINING_CONFIG}/sitecustomize-autoconfig-register.py" "$SITECUSTOMIZE_LOCAL/sitecustomize.py"
export PYTHONPATH="${SITECUSTOMIZE_LOCAL}:${PYTHONPATH:-}"


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
