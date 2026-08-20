#!/bin/bash

#SBATCH --nodes=80
#SBATCH --account=csstaff
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=12:00:00

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

export VERL_IMAGE="jfrog.svc.cscs.ch/docker-group-csstaff/alps-images/verl:alps7-dev-0f334b540ccc7034" #alps7-dev-0f334b540ccc7034 image with megatron

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
export PPO_MINI_BATCH_SIZE=24        # must be >= dp_size (DP=3 x EP=8 = 24)
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
    use_remove_padding: False  # DSA attention does not support THD packed-sequence format; use BSHD
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
    ppo_max_token_len_per_gpu: 16384
    use_dynamic_bsz: True
    megatron:
      tensor_model_parallel_size: 4
      pipeline_model_parallel_size: 3
      expert_model_parallel_size: 8
      param_offload: True
      grad_offload: True
      optimizer_offload: True
      vanilla_mbridge: False  # GLM-5.1 model_type=glm_moe_dsa requires Megatron-Bridge
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
    n: 1 #num responses per prompt — reduced from 16 for pipeline smoke test (GRPO advantages degenerate at n=1)
    # 8 rollout nodes × 4 GPUs = 32 GPUs; TP=32 (one replica) — 700B needs all 32 GPUs to fit
    tensor_model_parallel_size: 32
    gpu_memory_utilization: 0.75
    free_cache_engine: false  # keep KV cache alive across weight syncs — avoids engine rebuild + CUDA graph re-capture across TP=32 (8-node deadlock)
    calculate_log_probs: True   # required: bypass_mode reads rollout_log_probs as old_log_probs
    log_prob_use_dynamic_bsz: True
    checkpoint_engine:
      backend: nccl # weight sync via NCCL broadcast; separate-async rejects the "naive" backend
      engine_kwargs:
        nccl:
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
  project_name: ${PROJECT_NAME}
  experiment_name: ${RUN_NAME}
  nnodes: ${TRAINING_NNODES}
  n_gpus_per_node: 4
  save_freq: 50
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

# Content of example/patches/sitecustomize-verl-v0.9.0.py (see Known hazards in
# CLAUDE.md), embedded here rather than read via a script-relative path: under
# sbatch, BASH_SOURCE[0] resolves to the spool-staged copy of this script
# (/var/spool/slurmd/job<ID>/...) on every node, not its checkout location, so a
# `cp "$SCRIPT_DIR"/patches/...` silently failed on all 80 nodes in run 3129805 —
# the hybrid-rollout OOM fallback patch was never applied and the run OOMed exactly
# like runs 3121001/3125195. Generating it as a heredoc, like gsm8k_reward.py above,
# and sbcast-ing it removes the dependency on the script's own filesystem location.
cat > "${TRAINING_CONFIG}/sitecustomize-verl-v0.9.0.py" <<- 'EOF'
"""
Hybrid-rollout OOM fallback for verl v0.9.0, V1 separate-async trainer.

Injected at Python startup via PYTHONPATH — does not modify verl source files.
Distinct from sitecustomize.py (written against v0.8.0; class/module paths there
are not verified against v0.9.0) so that file is left alone. This file must be
staged as the process's actual "sitecustomize.py" on sys.path to be auto-loaded —
see the training script for how it's copied into place per node.

Why: PPOTrainer._setup() (verl/trainer/ppo/v1/trainer_base.py, v0.9.0) always
builds hybrid rollout replicas on top of the training worker group
(trainer_world_size / rollout_world_size replicas, e.g. 288/32 = 9 here) *in
addition to* the standalone rollout — actor_rollout_ref.hybrid_engine is not
consulted anywhere in the V1 path, so this cannot be disabled from config. Those
replicas are instantiated at gpu_memory_utilization=0.75 on the training GPUs
during trainer.init(), before the first on_sample_end() ever runs, and
free_cache_engine=false (required for TP=32 SGLang stability) makes their
sleep() a no-op — so they hold ~71 GiB per training GPU for the life of the run.
trainer.init() then needs ~15 GiB back on those same GPUs to stage Megatron
params for the NCCL export to the standalone rollout, and OOMs (run 3121001,
repeated unpatched in run 3125195).

separate-async never actually needs the hybrid engine: get_llm_client() is
overridden in PPOTrainerSeparateAsync to always route through the standalone
rollout, so the hybrid replicas exist only to be immediately put to sleep.

Patches applied:
  - LLMServerManager._initialize_llm_servers (verl.workers.rollout.llm_server):
    no-ops when called in hybrid mode (worker_group is not None) instead of
    launching replicas via init_hybrid(). Standalone-mode calls (worker_group
    is None) are unaffected.
  - PPOTrainerSeparateAsync.on_init_end (verl.trainer.ppo.v1.trainer_separate_async):
    drops the self.checkpoint_manager.update_weights(...) call. That manager's
    backend is forced to "naive" (trainer_base.py), which pushes weights into
    each worker's colocated hybrid engine directly rather than going through
    the (now empty) replica list — with hybrid replicas disabled above, that
    colocated engine is never created, so the call would push into nothing.
    self.standalone_checkpoint_manager.update_weights(...), the actual sync to
    the standalone rollout, is left untouched.
"""
import sys

_PATCHED: set = set()


class _HybridRolloutOomPatcher:
    """sys.meta_path hook: patches specific verl modules immediately after they load."""

    _TARGETS = frozenset({
        "verl.workers.rollout.llm_server",
        "verl.trainer.ppo.v1.trainer_separate_async",
    })

    def find_module(self, fullname, path=None):
        if fullname in self._TARGETS and fullname not in _PATCHED:
            return self
        return None

    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]
        # Remove self temporarily to avoid recursion during the real import.
        sys.meta_path[:] = [m for m in sys.meta_path if m is not self]
        try:
            import importlib
            mod = importlib.import_module(fullname)
        finally:
            sys.meta_path.append(self)
        _PATCHED.add(fullname)
        _apply_patches(fullname, mod)
        return mod


def _apply_patches(name: str, mod) -> None:
    # ── LLMServerManager: skip hybrid replica launch ────────────────────────────
    if name == "verl.workers.rollout.llm_server":
        cls = mod.LLMServerManager
        _orig_init_servers = cls._initialize_llm_servers

        async def _initialize_llm_servers(self, start_rank: int = None):
            if self.worker_group is not None:
                self.rollout_replicas = []
                self.server_handles = []
                self.server_addresses = []
                print(
                    "[sitecustomize-verl-v0.9.0] LLMServerManager: hybrid replicas "
                    "disabled (worker_group is set) — skipping init_hybrid()",
                    flush=True,
                )
                return
            await _orig_init_servers(self, start_rank=start_rank)

        cls._initialize_llm_servers = _initialize_llm_servers

    # ── PPOTrainerSeparateAsync: drop the hybrid-engine weight push at init ─────
    elif name == "verl.trainer.ppo.v1.trainer_separate_async":
        cls = mod.PPOTrainerSeparateAsync

        def _on_init_end(self):
            self.standalone_checkpoint_manager.update_weights(self.global_steps)
            print(
                "[sitecustomize-verl-v0.9.0] on_init_end: skipped "
                "self.checkpoint_manager.update_weights (naive backend, no hybrid "
                "replicas to push into)",
                flush=True,
            )

        cls.on_init_end = _on_init_end


sys.meta_path.append(_HybridRolloutOomPatcher())
EOF
sbcast -f ${TRAINING_CONFIG}/sitecustomize-verl-v0.9.0.py ${TRAINING_CONFIG}/sitecustomize-verl-v0.9.0.py

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

# Inject the v0.9.0 hybrid-rollout-OOM fallback patch (see Known hazards in
# CLAUDE.md — runs 3121001 and 3125195) via sitecustomize.py, executed at Python
# startup for every process including Ray workers, so it applies cluster-wide
# without modifying verl source.
#
# Python only auto-imports a module literally named "sitecustomize" found on
# sys.path — patches/sitecustomize.py (written against v0.8.0, not verified
# against v0.9.0) keeps that name and is left untouched. The v0.9.0-specific
# patch was sbcast to ${TRAINING_CONFIG} above (as sitecustomize-verl-v0.9.0.py)
# and has to be staged under the "sitecustomize.py" name here to be picked up,
# so copy it into a local tmpfs dir instead of pointing PYTHONPATH at
# ${TRAINING_CONFIG} directly. Local rank 0 does the copy; other local ranks
# wait — same pattern as the model-config mirror above.
# Remove this block (and its PYTHONPATH export) to disable the patch.
export SITECUSTOMIZE_LOCAL=/tmp/verl_patches_${SLURM_JOB_ID}
if [ $SLURM_LOCALID -eq 0 ]; then
    mkdir -p $SITECUSTOMIZE_LOCAL
    cp "${TRAINING_CONFIG}/sitecustomize-verl-v0.9.0.py" "$SITECUSTOMIZE_LOCAL/sitecustomize.py"
    touch $SITECUSTOMIZE_LOCAL/.ready
fi
until [ -f $SITECUSTOMIZE_LOCAL/.ready ]; do sleep 1; done
export PYTHONPATH="${SITECUSTOMIZE_LOCAL}:${PYTHONPATH:-}"

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
