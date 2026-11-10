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
#   * actor_rollout_ref.hybrid_engine: False keeps HYBRID rollout replicas off the training GPUs
#     (verl 0.9.1); without it init OOMs.
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
# /users is mounted: keep ~/.local site-packages from shadowing the image.
PYTHONNOUSERSITE = "1"
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
  hybrid_engine: False  # no colocated rollout replicas on the trainer GPUs
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


# Upstream verl PRs applied at runtime (v0.9.1 does not have them). Fetched ONCE here and sbcast:
# per-node downloads inside the srun left half the cluster unpatched (run 3124273). -f turns an
# HTTP error page into a hard failure. What each one is: see the apply loop in the srun.
# sha256 of each fetched patch: an upstream push to an open PR changes the content, and the job
# must stop rather than silently run different code. On a mismatch, review the new patch and
# update the hash here.
declare -A VERL_PR_SHA256=(
    [7421]=25c803f00a8912da87bb07d5dab1980d294ec5b38fdd2dbde32372c2476cf9fc
    [7661]=b22025a6f5387f0ef4abd24fcce330debc6eb40eb4c539a2b179454ccbc5bbc9
    [7777]=2fdc8dee0253e21cb215cbb82f8b67bd5c3ca541597d46e781c2f86d96737db4
    [7778]=ffd083e076f0105ff0f9ecd9c9ffd0734cb3b6c146e6e087bb152ddb0e94fb82
    [8493]=e6697106260c095a17d3c0a5263fd755c544ef04aa72f471c15c4bf55ae5cb25
)
for pr in 7421 7661 7777 7778 8493; do
    curl -sfL "https://github.com/verl-project/verl/pull/${pr}.patch" -o "${TRAINING_CONFIG}/${pr}.patch" \
        || { echo "FATAL: could not download PR #${pr}"; exit 1; }
    [ -s "${TRAINING_CONFIG}/${pr}.patch" ] \
        || { echo "FATAL: PR #${pr} patch is empty"; exit 1; }
    echo "${VERL_PR_SHA256[$pr]}  ${TRAINING_CONFIG}/${pr}.patch" | sha256sum -c --quiet - \
        || { echo "FATAL: PR #${pr} patch changed upstream (sha256 mismatch); review it and update VERL_PR_SHA256"; exit 1; }
    sbcast -f "${TRAINING_CONFIG}/${pr}.patch" "${TRAINING_CONFIG}/${pr}.patch"
done

# R3 fix: only upstream commit 8f016846 (verl #8287) has it, and the rest of that 60-file refactor
# does not apply to 0.9.1, so keep just its routed_experts hunk. Drop this once a tag contains it.
export VERL_R3_COMMIT="8f016846e3010e225072a947270dc0c1b98c1cd0"
curl -sfL "https://github.com/verl-project/verl/commit/${VERL_R3_COMMIT}.patch" -o "${TRAINING_CONFIG}/verl-r3-commit.patch" \
    || { echo "FATAL: could not download verl commit ${VERL_R3_COMMIT}"; exit 1; }
echo "6ea519373703cb3f7e6baf3e44292324ee7739ad97ff719cbf8b8a862b3eb448  ${TRAINING_CONFIG}/verl-r3-commit.patch" | sha256sum -c --quiet - \
    || { echo "FATAL: verl commit ${VERL_R3_COMMIT} patch sha256 mismatch"; exit 1; }
awk '
/^diff --git /{file=index($0,"rollout/sglang_rollout/async_sglang_server.py")>0; hunk=0; seen=0}
file && /^@@ /{seen=1; hunk=index($0,"@@ -662,22 ")==1}
file && (!seen || hunk)
' "${TRAINING_CONFIG}/verl-r3-commit.patch" > "${TRAINING_CONFIG}/r3-sglang-routed-experts-import-fix.patch"
grep -q "sglang.srt.state_capturer.routed_experts" "${TRAINING_CONFIG}/r3-sglang-routed-experts-import-fix.patch" \
    || { echo "FATAL: routed_experts hunk not found in verl commit ${VERL_R3_COMMIT}"; exit 1; }
sbcast -f ${TRAINING_CONFIG}/r3-sglang-routed-experts-import-fix.patch ${TRAINING_CONFIG}/r3-sglang-routed-experts-import-fix.patch

# sglang PR #38298 (found on the DeepSeek-V3 recipe, 2026-09-07): the sglang DeepSeek-family loader
# fused q_a_proj / kv_a_proj_with_mqa only when both halves arrived in the same load_weights call;
# the chunked delta weight sync split pairs across calls and left layers with dummy-init attention.
# A no-op for models without q_lora_rank; applied here for parity with the DeepSeek-V3 recipe
# (patch -p2 against the installed sglang 0.5.16 in the srun).
export SGLANG_FIX_QKV_A_CACHE="38298"
curl -sfL "https://github.com/sgl-project/sglang/pull/${SGLANG_FIX_QKV_A_CACHE}.patch" -o "${TRAINING_CONFIG}/sglang-qkv-a-cache.patch" \
    || { echo "FATAL: could not download sglang PR #${SGLANG_FIX_QKV_A_CACHE}"; exit 1; }
echo "1248c6f47a08e0b930ae4b0053948ee5592bc4f81f8bb42a96df14ac5e34c76e  ${TRAINING_CONFIG}/sglang-qkv-a-cache.patch" | sha256sum -c --quiet - \
    || { echo "FATAL: sglang PR #${SGLANG_FIX_QKV_A_CACHE} patch changed upstream (sha256 mismatch)"; exit 1; }
grep -q "^diff --git a/python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py" "${TRAINING_CONFIG}/sglang-qkv-a-cache.patch" \
    || { echo "FATAL: sglang qkv_a cache patch has no loader diff (bad SHA or GitHub error page)"; exit 1; }
sbcast -f "${TRAINING_CONFIG}/sglang-qkv-a-cache.patch" "${TRAINING_CONFIG}/sglang-qkv-a-cache.patch"


# Download model (skip if already present)
if [ ! -f "${TRAINING_HOME}/models/${MODEL_NAME}/config.json" ] \
    || ! ls "${TRAINING_HOME}/models/${MODEL_NAME}"/*.safetensors >/dev/null 2>&1; then
    # Check files, not the directory: the scratch purge leaves empty model dirs behind.
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

[ -r /users/${USER}/.wandb_api_key ] \
    || { echo "FATAL: /users/${USER}/.wandb_api_key not found (wandb logging needs the API key)"; exit 1; }
export WANDB_API_KEY=$(cat /users/${USER}/.wandb_api_key)
export WANDB_SILENT=true # Suppress WandB logs

export RAY_memory_usage_threshold=0.99


srun --mpi=pmix --network=disable_rdzv_get -N ${SLURM_JOB_NUM_NODES} --ntasks-per-node=1 -u \
    --environment="${TRAINING_CONFIG}/env.toml" \
    --container-writable bash -c '


# verl is baked into the image at v0.9.1 (editable install at /workspace/verl); the patches below go on top.
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
    print(f\"  WARNING sglang LocalSerializedTensor NOT at the patched path ({_e}) — verl 0.9.1 expects it there\")
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
#   #7661 list_of_dict_to_tensordict: always build nested tensors
#   #7777 delta_sharded steady sync: targeted P2P instead of the padded 480-way gather-to-rank-0
#         (found on the DeepSeek-V3 recipe; run 3263683 here hung on the same collective).
#   #7778 gather_round_megabytes kwarg (same origin) -- applied separately below with fuzz.
#   #8493 lockstep barrier on the Megatron HF export (the seed-sync collective desync).
# The patches below target verl 0.9.1: fail now rather than deep into an allocation on an older image.
VERL_INSTALLED=$(python3 -c "import importlib.metadata as m; print(m.version(\"verl\"))")
[ "$VERL_INSTALLED" = "0.9.1" ] || { echo "FATAL: this recipe needs verl 0.9.1, image has $VERL_INSTALLED (update VERL_IMAGE)"; exit 1; }
for pr in 7421 7661 7777 8493; do
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
# verl v0.9.1 passes print-style flush=True to logging.Logger.info on the hybrid_engine=False
# path, a TypeError at trainer init (PR 8493 validation jobs 3842126/3842147). Main rewrote it.
sed -i -E "/^[[:space:]]+flush=True,[[:space:]]*$/d" /workspace/verl/verl/trainer/ppo/v1/trainer_separate_async.py
sed -i -E "s/, flush=True\)$/)/" /workspace/verl/verl/workers/rollout/sglang_rollout/http_server_engine.py
grep -q "flush=True" /workspace/verl/verl/trainer/ppo/v1/trainer_separate_async.py \
    && { echo "FATAL: logger flush=True fix did not apply on $(hostname)"; exit 1; }

# Local patches (staged on the batch host, see there for what each does). Same apply-or-fail
# discipline.
for lp in r3-sglang-routed-experts-import-fix; do
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
        -exec cp {} $MODEL_LOCAL/ \; || { echo "FATAL: could not mirror model config to $MODEL_LOCAL"; exit 1; }
    # Symlink safetensors back to Lustre so megatron-bridge can still load weights
    for f in ${TRAINING_HOME}/models/${MODEL_NAME}/*.safetensors; do
        ln -sf "$f" "$MODEL_LOCAL/$(basename "$f")"
    done || { echo "FATAL: could not symlink safetensors into $MODEL_LOCAL"; exit 1; }
    touch $MODEL_LOCAL/.ready
fi
for _ in $(seq 600); do [ -f $MODEL_LOCAL/.ready ] && break; sleep 1; done
[ -f $MODEL_LOCAL/.ready ] || { echo "FATAL: local rank 0 did not stage $MODEL_LOCAL within 600s"; exit 1; }

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

    # Fail instead of idling the allocation when a node never joins (pyxis container-start
    # failure left run 3260837 waiting at 15/16 nodes).
    RAY_JOIN_DEADLINE=$(( SECONDS + ${RAY_JOIN_TIMEOUT:-1200} ))
    while true; do
            alive_nodes=$(ray status | awk "/Active:/{flag=1;next}/Pending:/{flag=0}flag" | grep "node_" | wc -l)
            if ! [[ "$alive_nodes" =~ ^[0-9]+$ ]]; then
                alive_nodes=0
            fi
            if [ "$alive_nodes" -ge "$SLURM_JOB_NUM_NODES" ]; then
                break
            fi
            if [ "$SECONDS" -ge "$RAY_JOIN_DEADLINE" ]; then
                echo "FATAL: only $alive_nodes/$SLURM_JOB_NUM_NODES nodes joined Ray within ${RAY_JOIN_TIMEOUT:-1200}s"
                exit 1
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
