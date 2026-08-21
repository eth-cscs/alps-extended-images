@../AGENTS.md

# Dependencies

Verl: You can find a local version of the verl repo at ../verl

# Launching and testing on HPC Cluster

To test a new Slurm Batch script you can schedule a job using the FirecREST API.
Find documentation here: https://api.svc.cscs.ch/ml/firecrest/v2/docs

IMPORTAN: Before submitting a new job ALWAYS ask permission!

Spawn a new agent to submit a new job and monitor the output.
If the job fails or completes download the output and proccess it.

You can find the Firecrest client and secret at:
~/F7T_Credentials
NEVER EVER SHARE, PRINT, or OUTPUT the Firecrest secret.

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

- `data.train_batch_size == trainer.v1.separate_async.parameter_sync_step * actor.ppo_mini_batch_size` (96 = 2 × 48).
- `actor.ppo_mini_batch_size >= dp_size` (DP=3 × EP=8 = 24). Raising this above the minimum
  was tried as a mitigation for the `list_of_dict_to_tensordict` hazard below (24 → 48) but
  turned out not to be load-bearing — that bug is fixed at its actual source now (sitecustomize
  patch), so any value satisfying this assertion is fine again.
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
  Fallback patch: `example/patches/sitecustomize-verl-v0.9.0.py` — (1) makes
  `LLMServerManager._initialize_llm_servers` return no replicas when `worker_group is not None`;
  (2) drops the `self.checkpoint_manager.update_weights(...)` call from
  `PPOTrainerSeparateAsync.on_init_end` — its backend is forced to `naive`, which bypasses the
  replica list and pushes weights into a colocated engine that would no longer exist. Written
  after run 3125195 confirmed the hazard was still live. Delivery: the script embeds this file's
  content as a heredoc and `sbcast`s it to every node before the srun (same treatment as
  `gsm8k_reward.py`), then stages it as each node's `sitecustomize.py` under a per-node `/tmp`
  dir (Python only auto-loads a module literally named `sitecustomize`) — a plain
  script-relative `cp` failed silently on all 80 nodes in run 3129805 because `sbatch` executes
  from a spool-staged copy of the script, not its checkout path, so `${BASH_SOURCE[0]}`-derived
  paths don't resolve inside the srun. **The checked-in file and the heredoc are two copies of
  the same content and must be kept in sync by hand** — diff them before trusting either.
  Deliberately a separate file from `example/patches/sitecustomize.py` — that one was written
  against v0.8.0 and its module/class paths are unverified against v0.9.0, so it's left untouched
  and unused by this script. Patches apply eagerly now (see the next bullet) — **still not yet
  verified end-to-end by a run.**
- **`sys.meta_path` hooks using `find_module`/`load_module` are dead on Python 3.11+.** That
  legacy finder/loader protocol (deprecated since 3.4) lost its importlib compatibility shim on
  the Python 3.11/3.12 shipped in this image — confirmed locally: a `find_module`-only meta_path
  entry is never even called for an ordinary import. Both `sitecustomize-verl-v0.9.0.py` (above)
  and the Apertus benchmark's `sitecustomize-autoconfig-register.py` (see
  `apertus-benchmarks/rl-bench-apertus-v1.5-70B-full-async.sh`) originally used this hook style
  and silently never patched anything in any run — including 3129805, where the file *did* reach
  the node correctly and still never fired, and 3133412 (Apertus), where the confirmation print
  never appeared in the log and the target bug reproduced unchanged.
  **Do not fix this by importing the target module eagerly, right at sitecustomize time** — that
  was tried next and does make the patch fire, but it forces the target's whole import chain
  (transformers → torch, or verl's worker/rollout modules → torch/ray/sglang) into *every* Python
  process that inherits the PYTHONPATH pointing at the sitecustomize file — including `ray start`
  itself and Ray's own internal daemon processes, not just the actual verl worker/actor processes
  that were ever going to import that module anyway. Two consecutive Apertus runs on the
  eager-import version (3133616, 3134586) both stalled identically and reproducibly: Ray logs
  "Connected to Ray cluster" and then the driver actor (`FullyAsyncTaskRunner`) never starts —
  frozen log, `scontrol show job` reporting the full node/GPU allocation as completely healthy
  (no `NodeFail`, nothing drained) — i.e. an application-level Ray actor-scheduling hang, not an
  allocation problem, on two different fresh 16-node allocations in a row. No run using the
  eager-import version ever got past that point, whereas every run *before* the qwen3_asr patch
  started actually executing (i.e. every run still on the dead no-op hook) reached much further
  (3133412 got to 81% weight loading). That correlation is the evidence, not proof of the exact
  mechanism — importing torch pre-fork inside Ray's own daemons is the leading suspect, but this
  hasn't been root-caused inside the container.
  The fix that avoids both failure modes: a `sys.meta_path` finder using the *current* import
  protocol, `find_spec` returning a `ModuleSpec` whose `loader.exec_module` first delegates to the
  real loader (so the target module's actual code still runs) and only then applies the patch —
  confirmed locally that `find_spec` (unlike `find_module`) is genuinely invoked by Python 3.11+,
  and confirmed the pattern patches correctly and lazily (only on-demand, once, no eager import,
  no recursion) with a local multi-target repro. Both `sitecustomize-verl-v0.9.0.py` and
  `sitecustomize-autoconfig-register.py` now use this pattern — a class implementing
  `find_spec(fullname, path, target=None)` that: returns `None` immediately unless `fullname` is
  a target and not yet patched; otherwise removes itself from `sys.meta_path`, calls
  `importlib.util.find_spec(fullname)` to get the *real* spec, restores itself, then returns that
  spec with `.loader` replaced by a wrapper whose `exec_module` calls the original loader's
  `exec_module` first and patches the now-executed module second. **Any new sitecustomize-style
  patch in this repo must use this `find_spec` pattern — not `find_module`/`load_module` (dead),
  and not an eager top-level `import`.** Unverified end-to-end by a run using this exact version
  yet — the eager-import version is what got tested (and found to stall) in 3133616/3134586.
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
- **`list_of_dict_to_tensordict` silently produces a dense `Tensor` instead of a jagged
  `NestedTensor` whenever items coincidentally share a shape.**
  `verl/utils/tensordict_utils.py:918` decides nested-vs-stacked per field purely by
  `all(item.shape == val_list[0].shape for item in val_list)` — trivially true for a length-1
  list (verl calls this once per rollout output, in `agent_loop_tq.py:224`, so
  `len(list_of_dicts) == 1` routinely) and also true whenever an undertrained model saturates
  `max_response_length` across a whole rollout group, so several same-length responses land in
  the same call. Any downstream code that assumes the field is nested (`.offsets()`) then raises
  `AttributeError: 'Tensor' object has no attribute 'offsets'`. Seen at two independent call
  sites in this chain: `engine_workers.py:294` (run 3134772) and `padding.py:119` (run 3136766,
  *after* `ppo_mini_batch_size` had already been raised from `dp_size` to `2 * dp_size` — proof
  that batch-size tuning only reduces the collision probability, it cannot eliminate it, since
  the corruption happens once at write time in `list_of_dict_to_tensordict`, per rollout output,
  not at mini-batch read time). Confirmed via the pinned `TransferQueue==0.1.7` wheel
  (`verl/requirements.txt`, pulled with `pip download TransferQueue==0.1.7 --no-deps` since it
  isn't vendored in `../verl`) that the **read** side already gets this right —
  `transfer_queue/storage/managers/simple_storage_manager.py`'s `_pack_field_values` always tries
  `torch.nested.as_nested_tensor` first for non-scalar tensor lists and never takes a
  shape-equality shortcut — and that `nested_tensor_from_tensor_list`, elsewhere in this very
  same verl file (used for chunking/dispatch), already builds nested tensors unconditionally too.
  `list_of_dict_to_tensordict` is the one outlier in the whole pipeline. Fix: sitecustomize patch
  (third target in `example/patches/sitecustomize-verl-v0.9.0.py`, targeting
  `verl.utils.tensordict_utils`) replaces it with a version that delegates non-scalar tensor
  fields straight to the module's own `nested_tensor_from_tensor_list`, matching both the read
  side and the rest of the file — no more shape-equality guessing. `PPO_MINI_BATCH_SIZE` stays at
  48 (harmless, just no longer load-bearing for this bug). Unverified end-to-end by a run yet.
  General lesson for this whole recipe (see the "no existing recipe" note below): GLM-5.1 is
  trained upstream with Zhipu's own `slime` framework (Megatron+SGLang), not verl, and verl's V1
  trainer + `separate_async` + TransferQueue combination has no reference recipe anywhere
  (`verl-recipe` has 40+ recipes, none for `separate_async`, TransferQueue, or any GLM model) —
  treat every new failure in this chain as plausibly a genuine unpatched upstream gap, not
  necessarily a mistake in this script.

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

### Run `3125195` — 2026-08-19 — init OOM again: hybrid-rollout fallback patch was never written

- **Log**: `~/Downloads/slurm-3125195 (1).out`. 80 nodes, verl v0.9.0. The 3124273 fix worked:
  `Applied PR #7421/#7422/#7423` on all 80/80 nodes and `Patched 1 filelock site(s)` on all 80 —
  no barrier failure, no config-load failure this time. (Unrelated, non-fatal: 15×
  `OSError: [Errno 37/116]` from `matmul_ext_update_autotune_table` atexit callbacks — a Triton
  autotune-cache lock/handle on Lustre, but it fires in `atexit` after the run already failed,
  so it's noise, not the cause here — worth a Lustre cache redirect if it ever shows up mid-run.)
- **Symptom**: identical failure to 3121001 — `torch.OutOfMemoryError` in
  `WorkerDict.actor_rollout_update_weights()` → `get_per_tensor_param()` →
  `load_megatron_model_to_gpu` (`verl/utils/megatron_utils.py:779`), during `trainer.init()` →
  `on_init_end()` → `standalone_checkpoint_manager.update_weights()`
  (`verl/trainer/ppo/v1/trainer_separate_async.py:135`). Tried to allocate 14.62 GiB, 14.61 GiB
  free, 70.74 GiB held by another process on the same GPU. Failing rank `172.28.39.104`, which
  the `LLMServerManager` log line (`... LLMServerManager: ['172.28.25.132:45923', ...
  '172.28.39.104:42725', ...]`) shows as one of 9 `RolloutMode.HYBRID` replicas.
- **Root cause**: the fallback patch documented under "Known hazards" — neutering
  `LLMServerManager._initialize_llm_servers` and dropping the `update_weights` call from
  `PPOTrainerSeparateAsync.on_init_end` — was never actually added to
  `example/patches/sitecustomize.py`. That file only contains the older fully-async debug
  instrumentation (`FullyAsyncRollouter`, `SGLangHttpServer.abort_all_requests`/
  `resume_generation`, `CheckpointEngineManager`); none of its three patch targets touch
  `LLMServerManager` or `PPOTrainerSeparateAsync`. So `PPOTrainer._setup()` built the usual 9
  hybrid replicas at `gpu_memory_utilization=0.75` on the training GPUs, `free_cache_engine:
  false` (line 164) keeps them from sleeping, and this run is a straight repeat of 3121001 —
  nothing about the hybrid-rollout hazard was fixed between the two.
- **Fix**: implemented the two-part fallback patch, checked against v0.9.0 source
  (`../verl` at tag `v0.9.0`, not the `v0.8.0-4-g933979db` checkout the repo happened to be on)
  since the old `sitecustomize.py` was written against v0.8.0 and its targets were never
  confirmed there. New file `example/patches/sitecustomize-verl-v0.9.0.py`: makes
  `LLMServerManager._initialize_llm_servers` (`verl/workers/rollout/llm_server.py`) a no-op when
  `worker_group is not None`, and drops the `self.checkpoint_manager.update_weights(...)` call
  from `PPOTrainerSeparateAsync.on_init_end` (`verl/trainer/ppo/v1/trainer_separate_async.py:136`).
  Traced the downstream call sites (`add_replicas_to_balancer`, `switch_to_trainer`/`_to_rollout`,
  `_init_global_load_balancer`) to confirm they degrade to no-ops on an empty replica list rather
  than erroring. Wired into the script: since Python only auto-imports a module literally named
  `sitecustomize`, the file is copied to `sitecustomize.py` under a per-node `/tmp` dir (local
  rank 0 copies, others wait) and `PYTHONPATH` points there instead of at `patches/` directly —
  see "Known hazards". Unverified — needs a run.
- **Commit**: not committed.

### Run `3129805` — 2026-08-20 — init OOM a third time: staging path assumed the wrong filesystem location

- **Log**: `~/Downloads/slurm-3129805.out`. 80 nodes, verl v0.9.0. PR/filelock patching: clean —
  80/80 on all four (`#7421`/`#7422`/`#7423`/filelock).
- **Symptom**: `cp: cannot stat '.../patches/sitecustomize-verl-v0.9.0.py': No such file or
  directory` on all 80 nodes (e.g. line 1960), then the identical
  `torch.OutOfMemoryError` as 3121001/3125195 in `on_init_end()` →
  `standalone_checkpoint_manager.update_weights()` → `load_megatron_model_to_gpu`: 14.62 GiB
  requested, 14.06 GiB free, 70.86 GiB held by another process on the same GPU. The
  `LLMServerManager` log line again lists 9 `RolloutMode.HYBRID` replicas (line 4187).
- **Root cause**: the sitecustomize-v0.9.0 fallback patch from 3125195 was correct but never
  reached the nodes. The script computed `SCRIPT_DIR` from `${BASH_SOURCE[0]}` at the top level
  and copied `${SCRIPT_DIR}/patches/sitecustomize-verl-v0.9.0.py` into `/tmp` inside the srun —
  but `sbatch` executes a batch script from a copy it stages in
  `/var/spool/slurmd/job<ID>/...`, so `BASH_SOURCE[0]` resolves to that spool path, not the
  script's checkout location under `/users/...`. `SCRIPT_DIR` was therefore wrong on every node
  (all 80 hit the identical spool path in the error), the `cp` failed silently (no `&&`/`set -e`
  gate), `sitecustomize.py` never landed in `$SITECUSTOMIZE_LOCAL`, the ready-file was still
  touched unconditionally so the barrier didn't catch it, and the run trained with an empty
  `PYTHONPATH` addition — i.e. no patch, hybrid replicas spawn as usual, same OOM as before.
- **Fix**: stop reading the patch from a script-relative path inside the srun at all. The patch
  content is now embedded as a heredoc directly in the script (same treatment as
  `gsm8k_reward.py`/`prepare_gsm8k.py`) and `sbcast` to `${TRAINING_CONFIG}` on the batch host
  before the srun, then copied from there to the per-node `sitecustomize.py` staging dir — no
  dependency on where Slurm happens to have staged the submitted script. The checked-in
  `example/patches/sitecustomize-verl-v0.9.0.py` and the heredoc are now two copies of the same
  content; they must be kept in sync by hand (added to Known hazards). Unverified — needs a run.
- **Commit**: not committed.

### Run `3134772` — 2026-08-21 — hybrid-rollout OOM finally fixed; new failure in first training step

- **Log**: `~/Downloads/slurm-3134772.out` (683,646 bytes). 80 nodes, verl v0.9.0, system
  `clariden`. PR/filelock patching: clean — 80/80 on all four (`#7421`/`#7422`/`#7423`/filelock),
  same as prior runs. Ran ~2442s (~40.7 min) of a 2:00:00 limit before failing.
- **Root cause of what changed**: the `sys.meta_path` hook in
  `example/patches/sitecustomize-verl-v0.9.0.py` was rewritten from the dead
  `find_module`/`load_module` protocol to a `find_spec`-based hook (see "Known hazards"). Both
  confirmation lines appeared exactly once each, at `on_init_end`:
  `[sitecustomize-verl-v0.9.0] LLMServerManager: hybrid replicas disabled (worker_group is set) —
  skipping init_hybrid()` and `[sitecustomize-verl-v0.9.0] on_init_end: skipped
  self.checkpoint_manager.update_weights (naive backend, no hybrid replicas to push into)` — i.e.
  the patch fired for the first time across this whole chain (3121001/3125195/3129805 never got
  a confirmation line). **The `torch.OutOfMemoryError` in `load_megatron_model_to_gpu` did not
  recur.** `trainer.init()` completed and training progress started
  (`Training Progress: 0/465`) — furthest any run in this chain has reached.
- **New symptom** (distinct from the hybrid-rollout hazard, first time seen): ~2m22s into the
  first `update_actor`/`train_mini_batch` call, on all 80 training ranks:
  `AttributeError: 'Tensor' object has no attribute 'offsets'` at
  `verl/workers/engine_workers.py:294`, in
  `global_token_num = mini_batch_td["input_ids"].offsets().diff().tolist()`. Code expects a
  nested/jagged tensor (has `.offsets()`) but received a plain `Tensor` for `input_ids` — looks
  like a `use_dynamic_bsz`/sequence-packing format mismatch between what the data pipeline
  produces and what this line assumes. Crashed all 80 nodes via Ray; `srun` force-terminated the
  step.
- **Root cause (found)**: `PPO_MINI_BATCH_SIZE=24` was set exactly equal to `dp_size` (DP=3 ×
  EP=8 = 24) — the minimum value the `ppo_mini_batch_size >= dp_size` assertion allows, giving
  `mini_batch_size_per_gpu = 1`, confirmed by the log line right before the crash:
  `Task update_actor (pid=54223) is getting len_samples=1`. See "Known hazards" —
  `list_of_dict_to_tensordict`-style shape-equality checks trivially pass for a length-1 list, so
  the batch gets stacked into a plain `Tensor` instead of a jagged `NestedTensor`, and
  `engine_workers.py:294`'s `.offsets()` call then fails.
- **Fix**: `PPO_MINI_BATCH_SIZE` 24 → 48 (2× `dp_size` instead of exactly `dp_size`);
  `TRAIN_BATCH_SIZE` auto-derives 48 → 96 (`PARAMETER_SYNC_STEP` unchanged at 2). Reduces the
  crash from certain (100% of steps, batch size 1) to a coincidence-dependent risk (needs two
  same-length sequences in a 2-sample batch) — not a full fix of the underlying verl/TransferQueue
  shape-equality heuristic, which is upstream code this repo doesn't vendor. Unverified — needs a
  run.
- **Commit**: not committed.

### Run `3136766` — 2026-08-21 — mini-batch-size mitigation confirmed insufficient; real root cause found and fixed

- **Log**: `~/Downloads/slurm-3136766.out` (768 KB). 80 nodes, verl v0.9.0, system `clariden`.
  Queued ~1h40m (`PENDING`, reason `Priority`) before allocation; ran to a Slurm-forced
  termination (exitCode 15) after reaching training. Sitecustomize hybrid-rollout fix: confirmed
  still working, no regression — both confirmation lines appeared, no OOM.
- **Symptom**: the same `AttributeError: 'Tensor' object has no attribute 'offsets'` as 3134772
  recurred, this time at a *different* call site — `verl/workers/utils/padding.py:119`
  (`no_padding_2_padding`, via `ppo_loss` in `verl/workers/utils/losses.py:59`), during the
  Megatron forward/backward pass, not the `engine_workers.py:294` TransferQueue-assembly site
  from 3134772. Training reached `Training Progress: 0/231` and sat there ~3 minutes before all
  80 ranks crashed — never advanced past step 0. The preceding `tqbridge` log line showed
  `len_samples=2` (not 1) and `PPO_MINI_BATCH_SIZE=48`/`TRAIN_BATCH_SIZE=96` were confirmed
  correctly substituted — i.e. the 3134772 mitigation worked exactly as intended (2 samples per
  DP rank instead of 1), and the bug still recurred anyway, exactly as flagged as a residual risk
  in that run's fix note.
- **Root cause (found, for real this time)**: `list_of_dict_to_tensordict`
  (`verl/utils/tensordict_utils.py:918`) is called once per rollout output in
  `agent_loop_tq.py:224` (`list_of_dict_to_tensordict(fields)`, `len(fields)` usually 1) — i.e.
  the corruption happens at **write** time into TransferQueue, per individual rollout output,
  not at mini-batch **read** time. Any rollout output whose "responses" tensor happens to share a
  shape with itself (trivially true for a 1-item list) or, for multi-output writes, whose
  responses in the group coincidentally share a length (common early in RL when an undertrained
  model saturates `max_response_length`) gets `torch.stack`ed into a dense `Tensor` instead of a
  jagged `NestedTensor` and stored that way permanently. `mini_batch_size_per_gpu` only changes
  how many *already-corrupted-or-not* per-sample records get read back together — it can reduce
  the chance two corrupted-shape records collide on read, but does nothing about the write-time
  corruption itself, which is why raising it from 1 to 2 samples/rank reduced but didn't
  eliminate the crash (and moved where it happened to bite, `input_ids` → `responses`). Verified
  by downloading the actual `TransferQueue==0.1.7` wheel (`pip download TransferQueue==0.1.7
  --no-deps`, per the pin in `../verl/requirements.txt` — the package isn't vendored in the local
  verl checkout) and confirming its own read-side reconstruction
  (`transfer_queue/storage/managers/simple_storage_manager.py:_pack_field_values`) always tries
  `torch.nested.as_nested_tensor` first, never taking verl's shape-equality shortcut. Also
  confirmed `nested_tensor_from_tensor_list`, elsewhere in the same verl file (used for
  chunking/dispatch), already builds nested tensors unconditionally — `list_of_dict_to_tensordict`
  is the one function in the whole pipeline that guesses from shape equality instead.
- **Fix**: third target added to `example/patches/sitecustomize-verl-v0.9.0.py`
  (`verl.utils.tensordict_utils` → `_patch_tensordict_utils`), replacing
  `list_of_dict_to_tensordict` with a version that delegates non-scalar tensor fields straight to
  the module's own `nested_tensor_from_tensor_list` — no more shape-equality guessing, matching
  both TransferQueue's own read path and the rest of the file. `PPO_MINI_BATCH_SIZE` left at 48
  (harmless, just no longer load-bearing). Checked-in patch file and the script's heredoc copy
  verified byte-identical. Unverified end-to-end by a run yet.

### Run `3137775` — 2026-08-21 — list_of_dict_to_tensordict patch fired everywhere but the offsets crash recurred unchanged

- **Log**: `~/Downloads/slurm-3137775.out` (682,892 bytes; saved from
  `/users/palmee/{{home_path}}/slurm-3137775.out` — the FirecREST API's `workingDirectory:
  "{{home_path}}"` template was not substituted server-side, so the log lands under a literal
  `{{home_path}}` directory, not directly in the home dir; worth checking on future submissions).
  80 nodes, verl v0.9.0, system `clariden`. Ran 08:40:34–09:19:14 UTC (~38.6 min) before Slurm
  force-terminated all 80 nodes (exitCode 15).
- **Symptom**: all three sitecustomize confirmation lines fired correctly, including the new
  third one (`tensordict_utils: patched list_of_dict_to_tensordict ...`, repeated across ~283 Ray
  workers) — no OOM. But the `AttributeError: 'Tensor' object has no attribute 'offsets'`
  recurred at the **exact same call site** as 3136766 (`padding.py:119`,
  `response_ids.offsets()`, inside `no_padding_2_padding`'s `if prompt_ids.is_nested:` branch —
  i.e. prompts *were* nested, responses were not). Training never advanced past
  `Training Progress: 0/231`, crashing ~3 minutes into the first step, identical to 3136766 in
  every respect except that the write-path patch had definitely fired this time and definitely
  didn't fix it.
- **Root cause**: not fully identified — deep-diving without spending another allocation. Ruled
  out empirically (via a local `pip install torch tensordict` + the downloaded
  `TransferQueue==0.1.7` wheel, no cluster needed) two leading theories:
  1. Shape-coincidence in TransferQueue's own read reconstruction — `torch.nested.as_nested_tensor`
     never silently densifies same-length inputs; confirmed directly (`nt.is_nested` stays `True`,
     `.offsets()` works, even for identical-length items).
  2. The leading-dim-of-1 write-time slicing quirk in
     `simple_storage_manager.py:_select_by_positions` (`field_data[pos:pos+1]` keeps a `(1, L)`
     shape for a dense single-element write, vs. the nested branch's proper `(L,)` unbind) — also
     confirmed harmless: `as_nested_tensor` on a list of `(1, L_i)` items still stays nested with
     working `.offsets()` (just an extra singleton dim).
  Found instead, empirically: `torch.nested.as_nested_tensor(items, layout=torch.jagged)` raises
  `RuntimeError` on a **dtype mismatch** across items (e.g. int64 vs int32) — and
  `_pack_field_values`'s own except-block catches exactly that and falls back to
  `layout=torch.strided`, whose result has `is_nested=True` and `type() == torch.Tensor` but
  **no `.offsets()` method** — reproducing the crash's exact error message character-for-character
  in a local repro (`'Tensor' object has no attribute 'offsets'`). `_pack_field_values` already
  logs a `logger.warning(...)` on this fallback, but that string does not appear in either
  3136766's or 3137775's log — inconclusive rather than exculpatory, since Python `logging` output
  is not reliably captured in Ray worker stdout the way `print(..., flush=True)` is (all of this
  repo's own sitecustomize confirmation lines deliberately use `print(flush=True)` for exactly
  this reason).
- **Fix**: none yet — this is a diagnostic-only run. Added a fourth sitecustomize target,
  `verl.workers.utils.padding` → `_patch_padding_diagnostics`, wrapping `no_padding_2_padding` to
  `print(flush=True)` the nested-state (proper jagged / degraded strided-with-no-offsets / plain
  dense) of `tensor`, `prompts`, `responses`, and `attention_mask` on every call, before it would
  crash. This should tell us definitively, from the very next run's log, which of the three states
  is actually occurring for `responses` — turning the next fix attempt from a guess into a
  targeted one. Remove this wrapper once the real mechanism is confirmed. Checked-in patch file
  and the script's heredoc copy verified byte-identical.
- **Commit**: not committed.
- **Commit**: not committed.

# Training Apertus v1.5 on verl (`rl-bench-apertus-v1.5-70B-full-async.sh`)

Apertus-v1.5-70B GRPO on GSM8K, CSCS Alps, 16 nodes × 4 GH200 (`clariden`). Script:
`Alps-Images/apps/verl/apertus-benchmarks/rl-bench-apertus-v1.5-70B-full-async.sh`. 8 nodes →
FSDP2 trainer (text-only, eager attention), remaining nodes → standalone SGLang rollout
(`rollout.tensor_model_parallel_size=4`), fully-async trainer, verl v0.9.0, `--time=2:00:00`.

**Apertus-v1.5 is a brand-new (2026-07-24), unreleased-upstream multimodal architecture**
(`model_type: apertus1p5` — text + discrete image/audio-token fusion extending the older
text-only `apertus`). Neither transformers nor SGLang has merged support for it yet; both
patches came from open, unmerged upstream PRs whose code is itself still actively evolving and
does not fully agree with either the released checkpoint or each other. Every fix below is
therefore a genuine gap in bleeding-edge upstream code, not a mistake in this script — expect
this list to shrink or become unnecessary once the two PRs merge and stabilize.

Confirmed working end-to-end in run `3139370` (2026-08-21): 4+ consecutive real GRPO training
steps (`actor/loss`, `actor/grad_norm`, `actor/lr` all sane; GSM8K validation ran with real
accuracy metrics), no crash, memory stable well under the 95 GiB/GPU ceiling. Getting there took
nine failed submissions (`3129881` → `3138546`) and eight distinct fixes, in order encountered:

1. **transformers has no `apertus1p5` support at all.** The image's pinned `transformers==5.8.1`
   raises `ValueError: ... does not recognize this architecture`. Fix: build a wheel from the
   swiss-ai fork (`huggingface/transformers#47662`, branch `swiss-ai/transformers@add-apertus1p5`)
   once on a single node, cache it on Lustre, `pip install --no-deps` on every node — never
   `sbcast` the full source tree directly (run `3129881`: `sbcast` hit "Bus error (core dumped)"
   on the large tarball, corrupting it on several nodes). Paired with a `safetensors>=0.8.0`
   wheel (`--no-deps` stops transformers pulling it in itself; the image's pinned
   `safetensors==0.7.0` fails `dependency_versions_check.py`, run `3129970`). A `.sha` marker
   file next to the cached wheel forces a rebuild when the pinned commit changes — the wheel's
   own version string is always the same dev placeholder, so a plain `ls transformers-*.whl`
   cache check can't tell an old commit's wheel from a new one.
2. **The vision-tokenizer submodule forces `attn_implementation: eager`.**
   `Apertus1p5VisionTokenizerModel` supports neither FlashAttention-2 (run `3130014`) nor SDPA
   (run `3130169`) — transformers' own SDPA error names `eager` as the fallback; it is the one
   implementation every model supports. Set in `actor_rollout_ref.model.override_config`.
3. **SGLang's generic multimodal fallback misloads a vision weight as the text embedding.**
   SGLang has no native `apertus1p5` model (mainline or the swiss-ai fork — checked every
   branch), so it falls back to `sglang/srt/models/transformers.py`'s generic loader, which
   raised `AssertionError: self.org_vocab_size=266752 ... loaded_weight.shape[output_dim]=131072`
   (run `3134672`) — 131072/131272 turned out to be vision/audio token-offset constants from the
   real fix, not a coincidence. Fixed by applying the still-open `sgl-project/sglang#32979`
   ("model: support Apertus 1.5", adds a real `apertus1p5` model) as a source patch: fetch the PR
   diff once, keep only the 6 files under `python/sglang/...` (drop `docs_new/`/`test/`, which
   don't exist under `dist-packages` and would make `patch` fail on those hunks), `sbcast`, apply
   with plain `patch -p2` (not `git apply` — `dist-packages/sglang` isn't a git checkout) against
   every node's `dist-packages/sglang`. One of the 6 files (`base_processor.py`, an
   audio-input-only hunk) failed on a context mismatch against the image's unpinned SGLang
   version (run `3136877`) — split into its own diff, applied best-effort (warn, don't abort);
   the other 5 (crucially the new `apertus_mm.py` model and `apertus.py`'s
   `get_input_embeddings()` addition) are fatal-if-they-fail, since those are what fix the crash.
4. **The `qwen3_asr` config-registration collision** (same root cause and fix as the GLM section's
   "sys.meta_path hooks" hazard above — the swiss-ai fork already registers types stock
   transformers doesn't, SGLang's own `AutoConfig.register(..., exist_ok=False)` collides) also
   hits here (run `3130211`, repeated in `3133412`/`3133616`/`3134586` while the patch mechanism
   was still broken). PR #32979 also fixes this upstream (`exist_ok=True` in its own
   `qwen3_asr.py`), so the sitecustomize patch is now a defensive backstop, not load-bearing.
5. **`sgl-project/sglang#32979`'s own `_init_component_model` assumes a config *object*.**
   `component_config.to_dict()` raised `AttributeError: 'dict' object has no attribute 'to_dict'`
   (run `3137554`) — against this transformers commit, `config.vision_tokenizer_config` /
   `.audio_tokenizer_config` are already plain dicts. A version-alignment gap between the two
   independently-evolving PRs, not a mistake by either in isolation. Fixed with a sitecustomize
   monkeypatch (not another source patch — see point 7) accepting either shape.
6. **The same PR's `load_weights` can't match the checkpoint's vision-tokenizer weight names.**
   `ValueError: No vision/audio tensor matches checkpoint key:
   vision_tower.encoder.down.0.block.0.conv1.bias` (run `3137785`) — structurally, the checkpoint
   key *does* match what `Apertus1p5VisionTokenizerModel` (verified against
   `../verl`-adjacent transformers source) should produce; a third checkpoint/PR alignment gap,
   not obviously wrong on either side. Since this benchmark is text-only GSM8K and never calls
   `get_image_feature`/`get_audio_feature`, the vision/audio tower weights never need to be
   numerically correct, only present so the model instantiates — patched `load_weights` to skip
   an unmatched vision/audio key with a warning instead of raising; the language-model weight
   path is untouched.
7. **`torch.OutOfMemoryError` in the FSDP actor's `backward()`** (run `3138546`) — 89 GiB
   resident on a 95 GiB GPU before an incremental 12.94 GiB allocation. This 71.94B-param *dense*
   model, forced onto eager attention by point 2, has no flash-attention memory savings, and no
   FSDP offloading was configured. Fixed by lowering `actor.ppo_max_token_len_per_gpu` from
   verl's default (16384, sized for flash-attention's O(n) memory) to 4096, and enabling
   `actor.fsdp_config.{param,optimizer,grad}_offload: True` — confirmed via `../verl`'s actual
   config schema (`verl/workers/config/actor.py`, `engine.py`) rather than guessing at field
   names or the default value.
8. **Points 4 and 5 depend on the `find_spec` sitecustomize mechanism, not the dead
   `find_module` one or an eager top-level import** — see the GLM section's "Known hazards" entry
   above for the full mechanism and evidence (it's shared code, and the eager-import
   Ray-actor-scheduling stall was actually first found *here*: runs `3133616`/`3134586`, both
   this Apertus script, both stalling identically right after "Connected to Ray cluster" with the
   driver actor never starting, before the GLM section's parallel investigation confirmed the
   same fix). Fixed the same way in both scripts: `find_spec` returning a `ModuleSpec` whose
   `loader.exec_module` delegates to the real loader first, patches second.

**Fix delivery mechanism**: everything above (transformers/safetensors wheels, the SGLang source
patch, the sitecustomize monkeypatches) is embedded/fetched inside
`rl-bench-apertus-v1.5-70B-full-async.sh` itself, following the same "fetch once on the batch
host or a single srun node, `sbcast`/cache on Lustre, apply/install identically on every node"
discipline as the GLM script (see "Never fetch per-node from the internet inside the srun" above)
— none of it lives only in this file.