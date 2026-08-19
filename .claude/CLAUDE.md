@../AGENTS.md

# Dependencies

Verl: You can find a local version of the verl repo at ../verl

# Debugging `train-gsm8k-glm5.1-700B-v1-separate-async-megatron.sh`

GLM-5.1 (700B) GRPO on GSM8K, CSCS Alps, 80 nodes × 4 GH200. Script:
`Alps-Images/apps/verl/example/train-gsm8k-glm5.1-700B-v1-separate-async-megatron.sh`.
Derived from `train-gsm8k-glm5.1-700B-full-async-megatron.sh` (verl fully-async recipe),
switched to the V1 trainer: `trainer.use_v1=True`, `trainer.v1.trainer_mode=separate_async`,
entrypoint `python -m verl.trainer.main_ppo`.

## Layout

- 8 nodes (32 GPUs) → standalone SGLang rollout, TP=32, 1 replica (`actor_rollout_ref.rollout.nnodes`).
- 72 nodes (288 GPUs) → Megatron trainer, TP=4 PP=3 EP=8 DP=3 (`trainer.nnodes`).
- Weight sync trainer → standalone rollout: NCCL checkpoint engine, every
  `trainer.v1.separate_async.parameter_sync_step` (=2) actor updates.
- Experience flows through TransferQueue (`transfer_queue.enable=True`).

## Config invariants (asserted by verl — violating them fails at init)

- `data.train_batch_size == trainer.v1.separate_async.parameter_sync_step * actor.ppo_mini_batch_size` (48 = 2 × 24).
- `actor.ppo_mini_batch_size >= dp_size` (DP=3 × EP=8 = 24).
- `actor_rollout_ref.rollout.{nnodes,n_gpus_per_node} > 0` and `checkpoint_engine.backend != naive`.
- `rollout.calculate_log_probs=True` is required by `algorithm.rollout_correction.bypass_mode=True`
  (old_log_probs are copied from rollout log probs).

## Known hazards (check these first)

- **Hybrid rollout on trainer GPUs.** `PPOTrainer._setup()` (`verl/trainer/ppo/v1/trainer_base.py:351`)
  unconditionally starts SGLang servers *inside* the Megatron worker processes —
  288/32 = 9 replicas at `gpu_memory_utilization=0.75` — in addition to the standalone rollout.
  `actor_rollout_ref.hybrid_engine` is not read anywhere in the V1 path; there is no config to
  disable this. They sleep after the first `on_sample_end()` and only wake for validation
  (disabled here), so the exposure is the init window. Prime suspect for init OOM.
  Fallback patch (via `example/patches/sitecustomize.py`, already on `PYTHONPATH`):
  (1) make `LLMServerManager._initialize_llm_servers` return no replicas when `worker_group is not None`;
  (2) drop the `self.checkpoint_manager.update_weights(...)` call from
  `PPOTrainerSeparateAsync.on_init_end` — its backend is forced to `naive`, which bypasses the
  replica list and pushes weights into a colocated engine that would no longer exist.
- **Lustre file locking.** `flock` fails (ENOLCK/ESTALE) in the container: hence the
  megatron-bridge filelock patch, the `/tmp` model-config mirror, and the Triton/FlashInfer
  cache redirects. New "stale file handle" errors usually mean a new cache path needs redirecting.
- **TP=32 SGLang.** CUDA graphs disabled and `free_cache_engine: false` — engine rebuild across
  8 nodes deadlocked previously. Re-enabling either is a regression risk.
- **Upstream PRs** #7421 (DSA/mcore), #7422 (`load_format=dummy` in standalone rollout),
  #7423 (NCCL deadlock in async weight sync) are applied at runtime. None of the three are in
  **v0.9.0** (all still apply cleanly to the tag), and they must run *after* the
  `git checkout`, which would otherwise discard them. #7422 is load-bearing here: at v0.9.0
  `async_sglang_server.py:179-181` still flips `load_format` dummy → auto for every non-hybrid
  replica, which is exactly the standalone rollout of separate-async.
- **Never fetch per-node from the internet inside the srun.** 80 nodes curl-ing github.com
  independently succeeded on roughly half of them (run 3124273), leaving different verl code on
  different ranks — far worse than an unpatched cluster. Download once on the batch host with
  `curl -sfL` (`-f` so an HTTP error page is a failure, not a no-op patch), assert non-empty,
  `sbcast` to every node, and make a patch that neither applies nor is already present fatal.
- **`free_cache_engine: false` disables sleep.** `SGLangHttpServer.sleep()` returns early when
  it is False (`async_sglang_server.py:485`), so `checkpoint_manager.sleep_replicas()` becomes a
  no-op and the hybrid replicas keep their share of every training GPU for the whole run.

## Run log

One sub-section per shared SLURM log, newest last. Template:

### Run `<SLURM_JOB_ID>` — `<YYYY-MM-DD>` — <one-line verdict>

- **Log**: <path or how it was shared>; nodes/image if they differ from the defaults above.
- **Symptom**: first real error and where it surfaced (rank, worker, phase).
- **Root cause**: what actually broke, with `file:line` in `../verl` or the script.
- **Fix**: change made or proposed; note if it is unverified.
- **Commit**: SHA + subject, or `not committed`.

### Run `3121001` — 2026-08-19 — init OOM: hybrid rollout squats on the training GPUs

- **Log**: `~/Downloads/slurm-3121001.out`. 80 nodes, image `alps7-dev-0f334b540ccc7034`,
  verl as shipped in the image (before the v0.9.0 bump).
- **Symptom**: `torch.OutOfMemoryError` in `WorkerDict.actor_rollout_update_weights()` on
  training node `172.28.45.200`, inside `trainer.init()` → `on_init_end` →
  `standalone_checkpoint_manager.update_weights()`. Tried to allocate 14.62 GiB with
  14.17 GiB free.
- **Root cause**: the hybrid `LLMServerManager` came up with 9 replicas spanning all 72
  training nodes (log line 2340 lists 9 addresses, all `RolloutMode.HYBRID`, including the
  OOMing node). The SGLang process on that GPU held **70.74 GiB** ≈ `0.75 × 95`. Megatron
  then needed 14.62 GiB to pull its offloaded params back on-GPU
  (`verl/utils/megatron_utils.py:641`, `load_megatron_model_to_gpu`) to export them over
  NCCL to the standalone rollout. Compounded by `free_cache_engine: false` making
  `sleep_replicas()` a no-op, so the hybrid replicas never gave the memory back.
- **Fix**: none applied. Two options: patch the hybrid replicas out via
  `example/patches/sitecustomize.py` (see Known hazards), or set `free_cache_engine: true`
  and accept kv-cache release/resume on every TP=32 weight sync. The small-model shakedown
  (`train-gsm8k-qwen-3B-v1-separate-async-megatron.sh`, `free_cache_engine` left at its
  default True) is meant to decide which.
- **Commit**: not committed.

### Run `3123689` — 2026-08-19 — model config load fails: Lustre mitigations were missing

- **Log**: `~/Downloads/slurm-3123689.out`. Same 80-node layout, verl bumped in-container to
  **v0.9.0** (`483b8a00`, confirmed on every node) — the bump itself worked.
- **Symptom**: `ValueError: Failed to load configuration from .../models/GLM-5.1 after 4
  attempts. Last error: [Errno 116] Stale file handle` (also `[Errno 37] No locks available`)
  from `megatron/bridge/models/hf_pretrained/safe_config_loader.py:134` via
  `filelock/_unix.py:63`, during `WorkerDict.actor_rollout_init_model()` — much earlier than
  3121001, so it says nothing about the hybrid-rollout OOM above.
- **Root cause**: not a v0.9.0 regression. The script had lost the filelock patch and the
  `/tmp` model-config mirror + YAML `sed`; the failing path in the error is still the Lustre
  one, confirming the rewrite never happened. `flock` is unsupported in-container on Lustre
  and 288 concurrent `AutoConfig.from_pretrained()` calls hammer the MDS.
- **Fix**: restored into the srun block, after the v0.9.0 checkout — filelock patch, `/tmp`
  mirror + `sed`, and the three PR patches (`git -C /workspace/verl apply`). Ordering matters:
  `git checkout -f` would discard patches applied before it. Also `git --no-pager log` so the
  ref no longer drags an alternate-screen pager dump into the slurm log. Unverified — needs a
  run.
- **Commit**: not committed.

### Run `3124273` — 2026-08-19 — SGLang load barrier fails: PR patches reached only half the nodes

- **Log**: `~/Downloads/slurm-3124273.out`. 80 nodes, verl v0.9.0 (`483b8a00` on all 80). The
  3123689 fix worked: `Patched 1 filelock site(s)` on every node and no config-load failure.
- **Symptom**: SGLang `ValueError: TP rank N could finish the model loading, but there are
  other ranks that didn't finish loading` from
  `sglang/.../load_model_utils.py:270 dist_barrier_after_load`, several Ray workers dying with
  `SYSTEM_ERROR ... connection error code 2`, and `trainer.init()` ending in `ActorDiedError`.
- **Root cause**: patch application was per-node and non-deterministic. Each of the 80 nodes
  curl-ed the three PRs from github.com inside the srun; applied/skipped came out at 49/31
  (#7421), 45/35 (#7422), 43/37 (#7423). On the nodes that missed #7422 the standalone rollout
  reverted to `load_format=auto` — log line 4122 is the proof:
  `WARNING: rollout mode is RolloutMode.STANDALONE, load_format is dummy, set to auto` (×3).
  Those TP ranks began loading the real 700B weights from Lustre while the patched ranks
  dummy-initialised, so the post-load barrier failed and the loading ranks were killed.
  Aggravating: `curl -sL` without `-f` writes an error page or an empty file, and `git apply`
  on an empty file *succeeds* — so an "Applied PR #…" line was never proof the patch landed.
- **Fix**: fetch the three patches once on the batch host with `curl -sfL`, assert non-empty,
  `sbcast` them to `${TRAINING_CONFIG}` on every node (same mechanism as the reward/prepare
  scripts), and inside the srun apply from the local file — apply if it applies, skip if
  `git apply --reverse --check` shows it is already present, otherwise `exit 1`. Unverified —
  needs a run.
- **Commit**: not committed.