@../AGENTS.md

# Cross-cutting gotchas (apply to every verl recipe in this repo)

- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is INCOMPATIBLE with SGLang.** SGLang's
  `torch_memory_saver` (invoked unconditionally by `load_model_with_memory_saver` during model
  load, *even when* `free_cache_engine: false`) runs a sanity check that hard-raises
  `RuntimeError: TorchMemorySaver is disabled for the current process because expandable_segments
  is not supported yet` — the two allocator mechanisms are mutually exclusive. Every rollout TP
  rank dies before the first training step (seen in run `3219305`). **Never set this as a global
  env var in a script whose srun body is shared by the SGLang rollout processes** (i.e. every
  recipe here — they all colocate or share the srun). There is no clean per-worker-type env
  scoping in verl, so if a Megatron trainer genuinely needs `expandable_segments` for an OOM, it
  has to come from a different lever (smaller `ppo_max_token_len_per_gpu`, grad-buffer bucket
  size, more offload) — not this env var.

- **verl v0.9.0 hardcodes several SGLang import paths that MOVED in SGLang 0.5.16** (this image's
  version). Each surfaces as `ImportError: cannot import name 'X' from '<old module>'` only when
  the relevant feature runs. Known so far, each with a checked-in one-line try/except patch under
  `Alps-Images/apps/verl/example/patches/`: `extract_routed_experts_from_meta_info`
  (`layers.moe.routed_experts_capturer` → `state_capturer.routed_experts`; R3 rollout capture;
  `r3-sglang-routed-experts-import-fix.patch`), `LocalSerializedTensor`
  (`model_executor.model_runner` → `model_executor.model_runner_components.weight_updater`;
  `delta_sharded` rollout weight apply; `delta-sharded-localserializedtensor-import-fix.patch`).
  When enabling any new verl rollout-side feature against this image, expect one more of these —
  cross-check every `from sglang.srt...import` in the new code path against SGLang v0.5.16's own
  source (its `weight_sync/utils.py` and `state_capturer/` are good references for the current
  paths).

# Dependencies

Verl: You can find a local version of the verl repo at ../verl

- **flashinfer packaging**: `flashinfer-cubin` / `flashinfer-jit-cache` are NOT fully on PyPI —
  their real index is `https://flashinfer.ai/whl` (serves GitHub release assets). flashinfer
  `>= ~0.6.14` hard-raises at import unless the installed `flashinfer-cubin` is the exact same
  version. See the "flashinfer packaging" section under the GLM script's Image bake-in for the
  full details, sub-index layout, and the matched-pair install used by the image + GLM script.

# Launching and testing on HPC Cluster

To test a new Slurm Batch script you can schedule a job using the FirecREST API.
Find documentation here: https://api.svc.cscs.ch/ml/firecrest/v2/docs

IMPORTAN: Before submitting a new job ALWAYS ask permission!

Spawn a new agent to submit a new job and monitor the output.
If the job fails or completes download the output and proccess it.

You can find the Firecrest client and secret at:
~/F7T_Credentials
NEVER EVER SHARE, PRINT, or OUTPUT the Firecrest secret.

# Configuration correctness audits (architecture-vs-config)

**Why this exists**: every "Known hazards"/"Run log" entry in this file was found by chasing a
crash, OOM, or assertion — a job log told us something was wrong. That process is blind to
config that is *silently wrong but still runs*: a knob that doesn't match the model's real
architecture, so training completes without erroring but is weaker or less aligned than it should
be. Case in point: `train-gsm8k-glm5.1-700B-v1-separate-async-megatron.sh` trains a MoE model
(GLM-5.1) with `actor_rollout_ref.actor.router_replay.mode` left at verl's default (`disabled`).
verl's own docs (`docs/ascend_tutorial/dev_guide/model_dev/transfer_to_npu_guide.md`) name R3
("Rollout Router Replay") as the mechanism that aligns MoE expert routing between the SGLang
rollout and the Megatron trainer, and explicitly list GLM-5 as one of the models that adopt it in
practice — a real, still-open gap in this script, found only by asking "does this config match
what this architecture needs," not by any run failing. No amount of crash-driven debugging would
ever have surfaced it, because nothing about its absence errors.

**The audit, for any script in this repo that trains a model**:

1. **Identify the model's real architecture class** from its actual HF `config.json` /
   `model_type` — not from its name or from what the script's author assumed. Distinguish at
   least: dense vs. MoE (routing/expert-parallelism knobs apply), text-only vs. multimodal
   (vision/audio-tower and token-pruning knobs apply — see the Apertus v1.5 sections' pruned-LM-head
   and `vision_model` bugs, both real architecture-vs-config mismatches found by crashes that this
   audit would ideally catch before a run), and any other structural trait (tied embeddings,
   custom attention, non-standard rope) that has a corresponding verl config knob.
2. **Cross-reference against verl's own docs and recipes**, not memory or assumption — `grep`
   `../verl/docs/` and `../verl/recipe/` (and, for a specific model family, `../verl/docs/algo/`
   and any Ascend/NPU tutorial docs, which tend to spell out per-architecture recommendations more
   explicitly than the main README) for the architecture class and, if one exists, the specific
   model family. List every algorithm-relevant flag those sources recommend or require, then diff
   it against what the script actually sets. Flags checked so far as load-bearing per architecture:
   `algorithm.rollout_correction.bypass_mode` (off-policy log-prob correction, all models),
   `actor_rollout_ref.actor.router_replay.mode` (MoE expert-routing alignment — `disabled`/`R2`/`R3`,
   see verl's `verl/workers/config/actor.py`), and the multimodal-specific gaps documented in the
   Apertus v1.5 sections below (pruned LM head sizing, `vision_model` flag forcing BSHD padding on
   a text-only batch). Extend this list as new architecture-specific knobs are found.
3. **Record the result** as its own dated entry (same format as a Run log entry) under that
   script's section: what was checked, what verl's docs actually recommend, whether the script
   matches, and if not, whether it was fixed or left as a documented, deliberate gap (e.g. "R3
   would need a `record_file` and hasn't been verified to work with the `separate_async`/
   TransferQueue V1 path — flagged, not yet enabled").
4. **When to run it**: at least once for every script in this repo that has reached a stable,
   real-training-step state (the point where its own Run log would otherwise stop growing) — do
   not let a script go undocumented on this axis just because it stopped crashing. Re-run it
   whenever verl is upgraded to a new pinned version/commit, since these are upstream
   recommendations that move with verl's own code, not something this repo controls.

This is a correctness-and-quality audit, not a crash hunt — a script can pass every item here and
still be worth a second look from a human who knows the intended training objective; it catches
"doesn't match documented best practice for this architecture," not "is definitely optimal."

**First full pass — 2026-08-26**, across every training script that has reached a stable,
real-step state in this repo:

- `train-gsm8k-glm5.1-700B-v1-separate-async-megatron.sh` (GLM-5.1, MoE): **gap found, tried
  twice, reverted twice.** R3 router replay is appropriate but was unset; enabling it also
  required THD (`use_remove_padding: True`), which this script had disabled for a
  since-plausibly-fixed reason. Run `3199623` (2026-08-27): confirmed two missing prerequisites
  (megatron-bridge 0.5.1, not the needed ≥0.6.0; SGLang missing the `routed_experts_capturer`
  module at the path verl expects) plus an unexplained CUDA crash. Fixed both prerequisites (a
  source patch for the sglang import path — genuinely correct, stayed in; a megatron-bridge
  0.5.1→0.6.1 runtime upgrade — turned out wrong) and retried as run `3201189`: got much further
  (real Megatron model construction, not just init), but the megatron-bridge upgrade itself
  turned out to need megatron-core ~0.19.0, not this image's 0.18.2, breaking core Megatron model
  init *unconditionally* — worse than not upgrading at all. Fully reverted, including the
  megatron-bridge upgrade this time (not just the three R3/THD flags). User explicitly authorized
  fixing megatron-core too (2026-08-28); validated the combined megatron-core 0.19.0 +
  megatron-bridge 0.6.1 upgrade on a cheap 1-node probe first (job `3207095`, after one
  probe-script bug caught and fixed in `3207054`) — every import smoke test passed, including the
  exact chain that crashed `3201189`. Wired into the real script and all three R3/THD flags
  re-enabled. Run `3207151` (2026-08-28) then cleared every prior blocker and hit one more, in
  verl's rollout-side routed-experts capture (stale `.numpy()` assumption vs sglang 0.5.16's
  base64 string); the expanded `r3-sglang-routed-experts-import-fix.patch` fixed it, and run
  `3207923` (2026-08-28) **validated the R3+THD fix — 5 clean training steps, the first this
  recipe has ever produced with R3 enabled, stable metrics** — before failing on a separate
  megatron-bridge weight-sync NCCL race (unrelated to R3; same class as run `3141801`, retry
  pending). Remaining: confirm that race is a one-off, and PR the verl fix upstream. Full
  findings in that script's own "Configuration audit" entry and Run log, below.
- `train-gsm8k-glm5.1-700B-full-async-megatron.sh` (GLM-5.1, MoE, the base this was derived
  from): same gap — router_replay unset here too. Not otherwise separately audited (this repo
  tracks it only as the V1 script's ancestor, with no dedicated section of its own); unlike the
  V1 script, it uses the classic non-TQ `agent_loop.py` path, which has confirmed full
  `routed_experts` plumbing — so R3 is plausibly *more* directly usable here than on the TQ path,
  for whoever picks this up next.
- `rl-bench-apertus-v1.5-70B-sglang-fsdp2-async.sh` and
  `rl-bench-apertus-v1.5-70B-sglang-megatron-async.sh` (Apertus-v1.5-70B, dense multimodal):
  router_replay/R3 doesn't apply (not MoE). Their real architecture-vs-config gaps (vision-tower
  attention implementation, SGLang multimodal loading, pruned-LM-head sizing, forced
  `vision_model` flag) were already found — by crashes, ahead of this process existing — and are
  fixed; see each script's own "Configuration audit" entry for the retroactive mapping.
- `train-gsm8k-apertus-8B.sh` / `train-gsm8k-apertus-8B-full-async.sh` /
  `train-gsm8k-apertus-70B-full-async.sh` (Apertus-8B/70B, the older text-only, non-v1.5,
  non-multimodal architecture): dense, no MoE, no vision/audio config surface — confirmed via
  `model_type` and the absence of any `vision_config`/expert-parallel settings. No gap found;
  `rollout_correction.bypass_mode` (the one algorithm-level flag that applies to every model in
  this repo) is set correctly in all three.
- `train-gsm8k-qwen-3B-v1-separate-async-megatron.sh` / `train-gsm8k-qwen-3B-full-async-megatron.sh`
  (Qwen2.5-3B): dense — confirmed by the script's own comment
  (`expert_model_parallel_size: 1  # Qwen2.5-3B is dense (no MoE); EP must be 1`). No gap found.
- `train-gsm8k-qwen-3B.sh`, `train-gsm8k-apertus-8B.sh` (the un-suffixed base variants) and the
  `*-working`/`*-unclean` snapshot scripts under `example/` and `apertus-benchmarks/`: not
  audited this pass — treated as superseded/backup copies of the scripts above rather than
  independently maintained recipes (none of them appear in this file's own tracked run logs).
  Worth a real look before anyone trains from one of them directly.

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
  Fix: `example/patches/v1-separate-async-fixes.patch` — (1) makes
  `LLMServerManager._initialize_llm_servers` return no replicas when `worker_group is not None`;
  (2) drops the `self.checkpoint_manager.update_weights(...)` call from
  `PPOTrainerSeparateAsync.on_init_end` — its backend is forced to `naive`, which bypasses the
  replica list and pushes weights into a colocated engine that would no longer exist. Applied via
  `git apply` against `/workspace/verl` after the v0.9.0 checkout, same discipline as the
  upstream PR patches (fetched/embedded once, `sbcast` to every node, apply-or-fail).
  **This started life as a `sitecustomize.py` runtime monkeypatch** (written after run 3125195
  confirmed the hazard was still live; see the `find_spec` entry below for why that mechanism
  exists and how it evolved) — fast to iterate on while still debugging, but converted to this
  plain source patch once run 3149736 confirmed the whole recipe works end-to-end with it, same
  reasoning as the Apertus benchmark's equivalent conversion
  (`apertus-benchmarks/patches/sglang-apertus1p5-local-fixes.patch`): a monkeypatch is one more
  moving part (`sys.meta_path` machinery, `PYTHONPATH` staging, import-timing dependence) that a
  stable fix doesn't need. `example/patches/sitecustomize-verl-v0.9.0.py` and the script's old
  heredoc copy of it are both gone now; `example/patches/sitecustomize.py` (a different, older
  file, written against v0.8.0 with module/class paths never verified against v0.9.0) was never
  used by this script and is untouched.
- **`sys.meta_path` hooks using `find_module`/`load_module` are dead on Python 3.11+.** That
  legacy finder/loader protocol (deprecated since 3.4) lost its importlib compatibility shim on
  the Python 3.11/3.12 shipped in this image — confirmed locally: a `find_module`-only meta_path
  entry is never even called for an ordinary import. Both this GLM script's `sitecustomize.py`
  (now retired — see above, converted to `example/patches/v1-separate-async-fixes.patch`) and the
  Apertus benchmark's `sitecustomize-autoconfig-register.py` (see
  `apertus-benchmarks/rl-bench-apertus-v1.5-70B-sglang-fsdp2-async.sh`, similarly retired once its
  own conversion landed) originally used this hook style and silently never patched anything in
  any run — including 3129805, where the file *did* reach the node correctly and still never
  fired, and 3133412 (Apertus), where the confirmation print never appeared in the log and the
  target bug reproduced unchanged.
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
  no recursion) with a local multi-target repro. `sitecustomize-autoconfig-register.py` (still
  actively used by the Apertus Megatron-trainer variant,
  `rl-bench-apertus-v1.5-70B-sglang-megatron-async.sh`, untested as of run 3141796) uses this
  pattern — a class implementing `find_spec(fullname, path, target=None)` that: returns `None`
  immediately unless `fullname` is a target and not yet patched; otherwise removes itself from
  `sys.meta_path`, calls `importlib.util.find_spec(fullname)` to get the *real* spec, restores
  itself, then returns that spec with `.loader` replaced by a wrapper whose `exec_module` calls
  the original loader's `exec_module` first and patches the now-executed module second. **Any new
  sitecustomize-style patch in this repo must use this `find_spec` pattern — not
  `find_module`/`load_module` (dead), and not an eager top-level `import`.** Confirmed working
  end-to-end for this exact pattern in this GLM script's own retired `sitecustomize.py` (all four
  of its confirmation lines fired correctly across every run from 3134772 through 3149736) before
  it was converted to a source patch — so the pattern itself is sound; convert to a stable source
  patch once a fix is confirmed, the same way this script and the Apertus FSDP2 benchmark both
  did, rather than leaving it as a permanent runtime monkeypatch.
- **Lustre file locking.** `flock` fails (ENOLCK/ESTALE) in the container: hence the
  megatron-bridge filelock patch, the `/tmp` model-config mirror, and the Triton/FlashInfer
  cache redirects. New "stale file handle" errors usually mean a new cache path needs redirecting.
- **TP=32 SGLang.** CUDA graphs disabled and `free_cache_engine: false` — engine rebuild across
  8 nodes deadlocked previously. Re-enabling either is a regression risk.
  A second, distinct TP=32 hang showed up once in run 3152802: ~9 minutes into ordinary
  steady-state rollout generation (well after weight sync had already completed cleanly), one
  SGLang TP rank fired its own 300s scheduler watchdog timeout, blocked inside
  `recv_requests → _broadcast_reqs_across_ranks → torch.distributed.broadcast` — a stuck
  NCCL/collective broadcast across the 8-node TP=32 group. Took down the whole SGLang replica
  (Ray `SYSTEM_ERROR`), and every subsequent rollout task then failed in an infinite
  `ActorDiedError` retry loop for the rest of the run (no recovery, no restart) until Slurm
  killed it at the `--time=2:00:00` limit. Retried unmodified as run 3171176: did not recur in
  that run — but 3171176 only reached 3 steps before monitoring stopped, nowhere near the
  step-13 mark below, so "one-off" was a weak verdict on a short run.
  **RECURRED, run `3209484` (2026-08-28) — new call site, ~step 13 of steady-state generation.**
  Same shape (standalone rollout replica, 300s scheduler watchdog fires, py-spy dump, replica
  killed, Ray `SYSTEM_ERROR`, `ActorDiedError` cascade, job FAILED, no recovery), but the stuck
  frame this time is inside the **flashinfer MLA/DSA decode-attention `plan()`**:
  `event_loop_overlap` -> `_execute_decode` -> `flashinfer_mla_backend.py:381 init_forward_metadata`
  -> `:749 call_begin_forward` -> `flashinfer/mla/_core.py:839 plan` (a tensor `.to()` copy that
  never returned). flashinfer 0.6.12; DSA backends `prefill=flashmla_sparse, decode=fa3`. NOT the
  `_broadcast_reqs_across_ranks` broadcast of 3152802 — a different collective in the DSA plan
  path, same lethal pattern. **This is the 3rd TP=32 SGLang scheduler hang across this recipe
  (3152802, 3209484), 2 distinct call sites — no longer dismissable as a single flake; it is a
  real intermittent fragility in the TP=32 SGLang + flashinfer + Slingshot DSA-rollout stack.**
  It is *not* caused by R3 or the megatron-core/bridge upgrade — this code path is unchanged from
  the pre-R3 baseline. **ROOT CAUSE FOUND AND FIXED — a stale flashinfer pin.** The Containerfile
  had `flashinfer_python==0.6.12` (+ `flashinfer_cubin==0.6.12`, added 2026-07-27) while sglang
  0.5.16 (what `sglang[all]` resolves to since the 2026-08-12 bump) requires
  `flashinfer_python[cu13]==0.6.14`; the hang was on exactly the flashinfer MLA `plan()` path.
  Upgrading to the matched `flashinfer_{python,cubin}==0.6.14` pair (from `flashinfer.ai/whl` —
  see the "flashinfer packaging" section) **eliminated the hang: run `3217439` (2026-08-29) ran
  20 clean R3+THD training steps with zero hang signatures**, well past the step-13 mark where
  `3209484` died. Both the script (runtime wheel install) and the Containerfile now carry the
  0.6.14 pair. The earlier "intermittent flake, you can get lucky" framing was wrong — the
  short clean runs (`3149736` 14 steps, `3171176` 3 steps) just hadn't run long enough to hit it
  reliably.
  If a TP=32 SGLang scheduler hang ever recurs *on flashinfer 0.6.14+*, it is a new bug — start
  from the watchdog py-spy dump; earlier fallback ideas (raise `watchdog_timeout`, different DSA
  decode backend, shrink the replica below TP=32, rollout-replica auto-restart) are on record but
  were never needed.
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
- **The offsets `AttributeError` chain (runs 3134772/3136766/3137775/3139371/3144665) — ROOT
  CAUSE FOUND: `TransferQueue==0.1.6` (pinned in `Alps-Images/apps/verl/Containerfile` at image
  build time) has a genuine upstream bug already fixed in `0.1.7` (the version `../verl`'s own
  `requirements.txt` declares), and this script's runtime `git checkout` of verl to v0.9.0 never
  touches verl's pip dependencies — so the image's 0.1.6 silently never moved, and every fix
  attempt against 0.1.7 source (which behaves correctly) was chasing the wrong version.**
  `transfer_queue/storage/managers/simple_backend_manager.py`'s (0.1.6; renamed to
  `simple_storage_manager.py` in 0.1.7) `AsyncSimpleStorageManager._pack_field_values` has:
  `if all(v.shape == values[0].shape for v in values): return torch.stack(values)` — the exact
  same "stack if shapes coincidentally match" anti-pattern verl's own (separately buggy, since
  fixed — see below) `list_of_dict_to_tensordict` had, just in a different package. True whenever
  a batch of stored per-sample tensors coincidentally share a shape — trivially for a length-1
  batch, and confirmed in run 3139371's diagnostic for a 2-sample micro-batch where both
  `responses` hit `max_response_length` (a routine GRPO scenario: an undertrained model rambling
  to the cap for multiple samples of the same prompt). Any downstream code assuming the field is
  nested (`.offsets()`) then raises `AttributeError: 'Tensor' object has no attribute 'offsets'`
  — non-deterministically at whichever call site first reads the corrupted field
  (`engine_workers.py:294` in 3134772/3144665, `padding.py:119` in 3136766/3137775/3139371).
  0.1.7's rewritten version always tries `torch.nested.as_nested_tensor(..., layout=jagged)`
  first, with no shape-equality shortcut — confirmed correct via a `pip download`ed 0.1.7 wheel
  tested locally, which is exactly what made every fix/diagnostic attempt against it (mini-batch
  size bump, a verl-side `list_of_dict_to_tensordict` patch, two diagnostic wrappers, a
  misdirected sitecustomize patch targeting the wrong 0.1.7-only module name) come up empty —
  none of them touched the actually-running 0.1.6 code at all. **Fix**: upgrade the container's
  TransferQueue from 0.1.6 to 0.1.7 at runtime — fetch the wheel once on the batch host,
  sha256-verify it, `sbcast` to every node (same "fetch once, distribute, apply-or-fail"
  discipline as the PR patches), `pip install --no-deps --force-reinstall` inside the srun, right
  after the verl checkout. This is not a workaround: 0.1.7 is the exact version verl v0.9.0
  itself already declares as its dependency, so this closes a real image/runtime version gap.
  Verified locally end-to-end (not just read) before writing this fix: downloaded both wheels,
  diffed their storage-manager modules, confirmed the buggy line's presence in 0.1.6 and absence
  in 0.1.7, and confirmed the sha256 embedded in the script matches the actual downloaded file.
  Unverified end-to-end by a run yet.
  Separately, verl's own `list_of_dict_to_tensordict` (`verl/utils/tensordict_utils.py:918`) had
  the identical anti-pattern and remains patched (third target in
  `example/patches/sitecustomize-verl-v0.9.0.py`, targeting `verl.utils.tensordict_utils`,
  delegating to the module's own `nested_tensor_from_tensor_list`) — a real, independent bug, just
  never the (sole) cause of the crashes in this chain, since TransferQueue's write-time storage
  layer (`_select_by_positions`) always unbinds a batch into individual per-sample raw tensors
  before storing regardless of dense-vs-nested, erasing whatever that function produces at the
  storage boundary either way. `PPO_MINI_BATCH_SIZE` stays at 48 (harmless, no longer load-bearing
  for either bug).
  General lesson for this whole recipe (see the "no existing recipe" note below): GLM-5.1 is
  trained upstream with Zhipu's own `slime` framework (Megatron+SGLang), not verl, and verl's V1
  trainer + `separate_async` + TransferQueue combination has no reference recipe anywhere
  (`verl-recipe` has 40+ recipes, none for `separate_async`, TransferQueue, or any GLM model) —
  treat every new failure in this chain as plausibly a genuine unpatched upstream gap, not
  necessarily a mistake in this script. This bug is a second confirmed instance: always check
  what a script's base image actually pins (`grep` the app's own `Containerfile`) before trusting
  a locally-checked-out dependency's own `requirements.txt` as ground truth for what's running —
  a runtime `git checkout` of one component (verl) does not upgrade that component's own pinned
  pip dependencies (TransferQueue) baked into the image at build time.
- **Gloo `all_gather_object` can hang for 30 min inside megatron-bridge's weight-sync, before
  training ever starts.** Run `3141801`: `trainer.init()` → `on_init_end()` →
  `standalone_checkpoint_manager.update_weights()` → `nccl_checkpoint_engine.py:246
  send_weights` → `megatron/bridge/models/conversion/model_bridge.py:1284
  stream_weights_megatron_to_hf` → `param_mapping.py:460 broadcast_obj_from_pp_rank` →
  `torch.distributed.all_gather_object(obj_flags, has_obj, group=self.pp_group)` timed out after
  exactly 1800000ms (`RuntimeError: ... gloo/transport/tcp/unbound_buffer.cc:78 ... Timed out
  waiting 1800000ms for recv operation`). `all_gather_object` always routes through Gloo
  regardless of the process group's primary backend, since it has to pickle a Python object
  (here, a `"detected_type"` flag) across PP ranks — this is NOT the NCCL tensor-broadcast path
  PRs #7421/#7422/#7423 touch, and it fired despite those three patches applying cleanly and no
  OOM. Init otherwise looked completely normal up to this point (dataset loaded, worker groups
  created, all sitecustomize patches installed). Reads as at least one PP rank never entering
  this specific `all_gather_object` call while others waited — a rank-count/collective-order
  mismatch somewhere inside megatron-bridge's per-tensor streaming HF conversion
  (`gather_from_ep_ranks`/`gather_from_tp_ranks`/`broadcast_from_pp_rank` all appear in the
  flight-recorder dump). Not yet root-caused or fixed — and it pre-empts the
  `list_of_dict_to_tensordict`/`_pack_field_values` diagnostic chain below entirely (that code
  only runs once training experience actually flows, which requires getting past this hang
  first).
  **Recurrence, run `3207923` (2026-08-28) — same family, different collective and phase.** This
  time it hit the **7th** trainer→rollout weight sync (after 5 clean training steps, not at
  init), and the stuck collective was an **NCCL `ALLGATHER`** on `EXPERT_MODEL_PARALLEL_GROUP` /
  `TENSOR_MODEL_PARALLEL_GROUP` in `param_mapping.py:806 gather_from_ep_ranks` (not the gloo
  `all_gather_object` on the PP group). `last completed work: 50565, last enqueued: 50634` — a
  peer never posted its matching collective. After the 1800000ms NCCL timeout,
  `torch.distributed.barrier()` (`engine_workers.py:782`) failed with gloo `Connection closed by
  peer` and the job died. Two differences from `3141801` worth noting: this run was on
  **megatron-bridge 0.6.1** (vs. 0.5.1 then), and the desync is on the **EP/TP** gather rather
  than the PP broadcast — so an EP-gather rank-count/order bug that's load- or version-dependent
  can't be ruled out. `3141801` did not reproduce on unmodified retry (`3144665`).
  **3rd occurrence, run `3219811` (2026-08-29) — now a confirmed recurring bug, not a flake.**
  Same signature exactly as `3207923`: NCCL `ALLGATHER` 30-min timeout in
  `param_mapping.py:806 gather_from_ep_ranks` → `stream_weights_megatron_to_hf` →
  `send_weights (nccl_checkpoint_engine.py:246)`, at weight-sync #5 (feeding `global_step` 10),
  rank 59 never entered the gather. Tally: **hit in `3141801` / `3207923` / `3219811`, did NOT
  hit in `3144665` / `3209484` (12+ syncs) / `3217439` (20 steps)** — ~1 in 2 runs, at a random
  sync, spanning megatron-bridge 0.5.1 and 0.6.1, present before R3 was enabled. It is the
  single biggest blocker to a full run.
  **4th occurrence, run `3240762` (2026-08-31) — mechanism found, fix applied.** First time it
  hit the **`delta_sharded` SEED sync** (`delta_checkpoint_engine.py:521 _send_full_seed` →
  `stream_weights_megatron_to_hf` → `broadcast_obj_from_pp_rank`'s gloo `all_gather_object` on
  the PP group; `gather_from_ep_ranks` ALLGATHER on others). Ranks 244/33 never entered;
  `last enqueued 172, last completed 109` — 63-collective drift. Root cause: `_send_full_seed`
  drives the full HF-export generator **asymmetrically** — rank 0 buckets + broadcasts each
  flush to the rollout CE group between pulls, non-master ranks discard and immediately pull the
  next — so non-master ranks race ahead in the per-tensor PP/EP/TP assembly-collective chain
  until a cross-group collective deadlocks. This is very likely the same asymmetry behind the
  earlier occurrences (the `nccl` backend's `_send_weights` has a similar rank-0-does-extra-work
  shape). **Fix applied** (`example/patches/wsync-debug-progress-log.patch`, expanded): for the
  `"seed/full"` export only, `torch.distributed.barrier()` on the trainer WORLD group every
  `VERL_WSYNC_SEED_BARRIER_EVERY` (default 1) HF tensors, after the consumer processed each —
  bounds the drift to zero. Safe (a barrier can't corrupt state); ~1–3 min added to the one-time
  seed. Unverified on cluster as of this entry — see run `3240762`'s Run-log entry. If the hang
  recurs *with the barrier applied* it's a deeper bug (a single collective inside one tensor's
  assembly, not cross-tensor drift) — then look at a megatron-bridge `stream_weights_megatron_to_hf`
  robustness knob or a verl sync-retry, per the earlier notes here.
  Meanwhile, before the barrier is validated, every long run still has a coin-flip chance of
  dying here.
- **Backticks (and bare `$(`/`$VAR`) inside this script's unquoted heredocs are live shell,
  not text.** `env.toml`, `grpo_gsm8k.yaml`, `gsm8k_reward.py`, and `prepare_gsm8k.py` are all
  generated via `cat > file <<- EOF` with an **unquoted** delimiter (`EOF`, not `'EOF'`) — this
  script relies on that for real, working `$((...))` arithmetic expansion inside
  `grpo_gsm8k.yaml` (e.g. `num_data_storage_units: $(( SLURM_JOB_NUM_NODES * 2 ))`), so it can't
  simply be quoted away. The cost: any backtick written into one of these heredocs — including in
  what looks like an inert YAML/Python comment — triggers real command substitution. Run
  `3201189` hit this twice in one session: a markdown-style `` `router_replay: {mode: R3}` ``
  in a YAML comment made bash try to *execute* `router_replay: {mode: R3}` as a command
  (`line 104: router_replay:: command not found`, the literal first line of that run's log), and
  a second backtick-quoted error message introduced while fixing the first had the identical
  effect. Confirmed reproducible in a local sandbox, not a cluster artifact (`bash -x` on the
  script's own first ~285 lines pinpoints the exact backtick pair via the xtrace `++` prefix) — so
  this is always worth checking locally before spending cluster time. **Any edit inside one of
  these four heredocs must be scanned for backticks (and any accidental bare `$(`/`$VAR`) the same
  routine way the single-quoted `srun bash -c '...'` body is already scanned for stray `'`** — see
  run 3149339's entry above for that established discipline; this is its counterpart for the
  unquoted heredocs.

## Configuration audit (architecture-vs-config) — 2026-08-26

Per the repo-wide "Configuration correctness audits" process above. GLM-5.1 is a MoE model
(`model_type=glm_moe_dsa`) — checked whether R3 (Rollout Router Replay), verl's mechanism for
aligning MoE expert routing between the SGLang rollout and the Megatron trainer, is (a)
appropriate here and (b) actually enabled.

- **(a) Appropriate**: yes. verl's own docs
  (`docs/ascend_tutorial/dev_guide/model_dev/transfer_to_npu_guide.md`) name R3 as the
  alignment mechanism for large MoE models and list GLM-5 by name as one of the models that
  adopt it in practice, alongside DeepSeek-V3.2 and MiMo-V2.
- **(b) Enabled**: no. Confirmed the script never sets `router_replay` anywhere, and verl's
  default (`verl/trainer/config/actor/actor.yaml:279`, `mode: disabled`) is off. Neither
  `router_replay.mode` nor `rollout.enable_rollout_routing_replay` (a separate, also-required
  flag — see below) is set.
- **The correct config path for this recipe is `actor_rollout_ref.actor.megatron.router_replay.mode`,
  not the top-level `actor_rollout_ref.actor.router_replay.mode`** — traced both fields to their
  actual consumers: `engine_workers.py:494` (`WorkerDict.__init__`, fetched from the real v0.9.0
  tag, since local `../verl` is v0.8.0 and has no `trainer/ppo/v1/` directory at all) reads
  `self.config.actor.megatron.router_replay.mode` to set `self.enable_routing_replay` — the
  top-level field is a distinct, separately-defined dataclass field that the V1/Megatron worker
  path never reads. verl's own Ascend tutorial doc is internally inconsistent about which of the
  two it means (states `actor.megatron.router_replay.mode` in prose, then shows
  `actor.router_replay.mode` in its CLI example) — trust the code, not the doc prose, on this
  specific point.
- **Second required flag, independent of `router_replay.mode`**: `rollout.enable_rollout_routing_replay`
  (`verl/workers/config/rollout.py:265`, default `False`) — read by
  `verl/workers/rollout/sglang_rollout/async_sglang_server.py` (the standalone-rollout server this
  recipe uses) to set SGLang's `enable_return_routed_experts` server arg and request
  `return_routed_experts` per generate call, and to populate `output.routed_experts` via
  `sglang.srt.layers.moe.routed_experts_capturer`. R3 needs *both* flags set, not just
  `router_replay.mode`; also needs the image's SGLang build to actually contain that
  `routed_experts_capturer` module (not checked — would need a cluster-side `python -c` probe).
- **Whether it would actually work with this recipe's `separate_async` + TransferQueue combo:
  plausible but unverified, not confirmed broken.** Traced the real propagation path in the
  fetched v0.9.0 source: `verl/trainer/ppo/v1/agent_loop_tq.py` (the file behind every
  `agent_loop_tq.py:224`-style reference elsewhere in this log) has zero direct mentions of
  `routed_experts`/`router_replay`, but its `_agent_loop_postprocess` calls the shared, inherited
  `AgentLoopOutput.as_dict()` (from `verl/experimental/agent_loop/agent_loop.py`, which *does*
  carry a `routed_experts` field end-to-end when the classic non-TQ agent loop is used) and stores
  whatever fields that produces into TransferQueue generically — so the mechanism for getting
  `routed_experts` into TransferQueue at all appears to exist without TQ-specific code, contingent
  on the inherited `_run_agent_loop` actually populating that field. What's genuinely untested:
  `routed_experts` is a `[length, layer_num, topk_num]` tensor, a shape/structure this whole
  recipe's TransferQueue debugging chain (runs 3134772–3149776, see above) never exercised — every
  fix in that chain (TransferQueue 0.1.6→0.1.7, `list_of_dict_to_tensordict`) was validated only
  for `input_ids`/`responses`/`attention_mask`-shaped fields, not this one.
- **Blocker found before enabling anything**: `align_r3_router_replay_data`
  (`verl/utils/megatron/router_replay_utils.py:341`) hard-requires nested/jagged `input_ids` —
  `if not layers_topk_idx.is_nested or not input_ids.is_nested: raise TypeError("R3 router replay
  requires jagged route targets and input_ids")`. This script had
  `actor_rollout_ref.model.use_remove_padding: False`, with an existing (unattributed, no run
  citation) comment: `# DSA attention does not support THD packed-sequence format; use BSHD`. As
  configured, turning on R3 would not have produced an untested-but-plausible feature — it would
  have produced a guaranteed `TypeError` on the first replay pass, since THD is exactly what
  `use_remove_padding=True` enables and R3 cannot run without it.
- **Researched whether this was already solved upstream** (web search, since local `../verl`
  cannot answer questions about upstream roadmap/fixes): yes, plausibly. verl's own
  ["Adding DeepSeek V4 support"](https://verl.readthedocs.io/en/latest/advance/deepseek_v4_integration.html)
  doc (updated 2026-07-12) documents R2/R3 working together with THD for a DSA-based MoE model in
  its "Model and kernel compatibility" section: *"The fused DSA kernel requires each local THD
  shard to contain at least one CSA window. Shorter local shards must be padded before the kernel
  call and unpadded afterward"* — a shard-size handling requirement, not a blanket
  THD-incompatibility. Separately, Megatron-Bridge's 0.6.0 release notes explicitly advertise
  **"GLM-5.2 with cuDNN fused DSA for 128K THD-packed context-parallel training"** — GLM's own DSA,
  by name. This script's `Containerfile` pins `megatron-bridge>=0.5.1` with no upper bound
  (`pip_install python "megatron-bridge>=0.5.1" ...`), so whatever image actually gets built very
  plausibly already resolves to ≥0.6.0. This reads as the blocking comment predating a since-landed
  upstream fix, not as a permanent architectural limit — but this is inference from public
  docs/release notes, not a run against this exact pinned build, so still genuinely unverified for
  GLM-5.1 specifically (the DeepSeek V4 doc and the 0.6.0 note are both about sibling/adjacent
  models, not this one).
- **Verdict — enabled, 2026-08-26 (user-approved after being shown the blocker and the research
  above)**: the script now sets, in order of dependency:
  1. `actor_rollout_ref.model.use_remove_padding: True` (was `False`) — unlocks THD, the
     prerequisite for both R3 and the fused DSA kernel per the research above.
  2. `actor_rollout_ref.actor.megatron.router_replay.mode: R3` — the correct, verified config
     path (not the top-level legacy field).
  3. `actor_rollout_ref.rollout.enable_rollout_routing_replay: True` — the SGLang-side companion
     flag, required independently of `router_replay.mode`.
  Also added two non-fatal diagnostic prints to the srun setup, right after the TransferQueue
  version check: the resolved `megatron-bridge` version (warns if `<0.6.0`) and whether
  `sglang.srt.layers.moe.routed_experts_capturer` actually imports (warns if not) — both of the
  open assumptions above, made visible in the very next run's log instead of discovered only via
  a downstream crash. Not fatal, since either failing will already produce a clear, specific
  exception later (a router-replay forward error or an `AttributeError` at first
  routed-experts-capturing generate call) rather than a silent wrong result — no need to guess
  which assumption was wrong from a generic early abort.
  **This is a bigger, coupled change than R3 alone** — it also flips the model's packed-sequence
  format, an axis with its own failure history elsewhere in verl (a known, separate
  `use_remove_padding=True` + context-parallelism grad_norm-explosion bug, irrelevant here since
  this recipe sets no context parallelism, but real evidence this axis isn't risk-free in
  general). The `routed_experts` TransferQueue round-trip risk from the previous version of this
  entry — a `[length, layer_num, topk_num]` tensor shape this recipe's TQ debugging chain never
  exercised — is also still completely untested. **Entirely unverified end-to-end — needs a
  dedicated cluster run before any of this is trusted**, and given how much moved at once
  (THD + R3 + two new diagnostics), a failure on the next run should not be assumed to be any one
  of these specifically until the log says which.

**Outcome, 2026-08-27 (run `3199623`, see Run log below for full detail)**: reverted. Both
open assumptions above turned out to be real, confirmed-missing prerequisites, not just
theoretical risk — the run's own new diagnostic prints showed `megatron-bridge version: 0.5.1`
(not the ≥0.6.0 the THD-DSA research pointed to) and `sglang routed_experts_capturer not
importable` (the module doesn't exist in this image's SGLang at all). The run also crashed with
an unexplained single-rank CUDA "unspecified launch failure" in Megatron's DDP grad-buffer
construction, immediately downstream of DSA-attention + router-replay-patch init — not
conclusively pinned on either new flag vs. a one-off flake. All three of `use_remove_padding`,
`router_replay.mode`, and `enable_rollout_routing_replay` are back to their pre-2026-08-26 values;
the two diagnostic prints stay in the script (harmless, non-fatal) so a future re-attempt gets
this signal for free. **Before trying this again**: megatron-bridge needs an actual runtime
upgrade to ≥0.6.0 (same wheel-install-at-runtime pattern already used for TransferQueue
0.1.6→0.1.7 in this same script — confirmed 0.6.0/0.6.1 are published on PyPI), and the SGLang
side needs real investigation this pass didn't do: the deployed image's actual SGLang version was
never probed (a gap in the diagnostics — should be added alongside the megatron-bridge check next
time), and `routed_experts_capturer`'s presence looked inconsistent across the SGLang release tags
spot-checked afterward (present in some, absent in others, plus evidence of the file having been
relocated within the SGLang tree at some point) — this needs a clean answer, not another guess,
before spending another allocation on it.

**Outcome, 2026-08-28 (runs `3201189` then `3207151`)**: prerequisites now all met (megatron-core
0.19.0 + megatron-bridge 0.6.1 runtime upgrade, validated on probe `3207095`; sglang
routed-experts capture confirmed at `sglang.srt.state_capturer.routed_experts` and patched in) —
and R3 **still fails**, now at the first-step rollout call: verl's `async_sglang_server.py`
`skip_tokenizer_init: True` branch does `captured.numpy()` on what sglang 0.5.16 returns as a
base64 **string**, not a tensor (`AttributeError: 'str' object has no attribute 'numpy'`,
cluster-wide). verl's routed-experts capture is stale against this sglang across the board
(v0.9.0 and `main` identical). This disproves this entry's earlier "plausible but unverified"
read that R3 would work with `separate_async`+TransferQueue once prerequisites were met — the
TransferQueue round-trip was never even reached. **Fix implemented + validated 2026-08-28**: the
`skip_tokenizer_init` branch of verl's routed-experts capture is dead code against any current
sglang — `r3-sglang-routed-experts-import-fix.patch` now rewrites the whole block to always
base64-decode + reshape (shape/layout confirmed from sglang 0.5.16's own capturer source).
**Run `3207923` cleared the whole pipeline and completed 5 clean training steps — the first this
recipe has ever produced with R3 enabled** (loss 0.013–0.054, grad_norm 0.32–0.61, stable,
memory in budget), plus 7 successful weight syncs. It then FAILED at ~1h40min on the 7th weight
sync — an NCCL `ALLGATHER` desync inside megatron-bridge's Megatron→HF weight streaming
(`gather_from_ep_ranks`), **not in R3 code** — same non-deterministic-race class as run `3141801`
(which didn't reproduce on retry). So: the R3+THD fix itself is validated; the recipe now has a
separate, likely-pre-existing megatron-bridge weight-sync stability question to settle (retry
pending). See run `3207151` (diagnosis) and `3207923` (validation + the new failure) Run log
entries.

**Outcome, 2026-08-29 (run `3217439`)**: R3+THD on GLM-5.1 is validated end-to-end — **20 clean
training steps with real GSM8K learning** (`critic/score/mean` 0.05 → ~0.15). The `3207923`
weight-sync hang was confirmed a one-off (`3209484` cleared 12+ syncs); the `3209484` TP=32
SGLang MLA hang was root-caused to the stale flashinfer 0.6.12 pin and **fixed** by the matched
`flashinfer_{python,cubin}==0.6.14` pair. `3217439` then hit a *new*, marginal (24 MiB) CUDA OOM
in the Megatron distributed optimizer at step 21 — non-PyTorch memory (NCCL / checkpoint-engine
buffers) on the coordinator rank, torch's own peak dead-flat across all 20 steps. Proposed fix:
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` + lower `ppo_max_token_len_per_gpu`.
Remaining open items: (a) land the step-21 OOM fix and get a longer run; (b) PR the
`async_sglang_server.py` R3 fix upstream to verl; (c) validate the CI-built image (all the
runtime upgrades are baked into the Containerfile now).

## Image bake-in — 2026-08-28

Once R3+THD was validated (run `3207923`), the runtime pip upgrades this script had been doing on
every run were moved into `Alps-Images/apps/verl/Containerfile` so the image ships them directly,
plus one stale-pin fix found while investigating run `3209484`:

- `flashinfer_cubin==0.6.12` / `flashinfer_python[cu13]==0.6.12` → matched
  `flashinfer_cubin==0.6.14` / `flashinfer_python[cu13]==0.6.14` from
  `--extra-index-url https://flashinfer.ai/whl` (committed `8e6aae1`; PyPI's cubin lags at 0.6.13
  so the matched pair must come from the flashinfer index). The 0.6.12 pin (2026-07-27) predates
  the `sglang[all]`→0.5.16 bump and sglang 0.5.16 requires `flashinfer_python==0.6.14`; the skew
  was the cause of run `3209484`'s TP=32 MLA `plan()` hang (fixed, validated run `3217439`).
- `TransferQueue==0.1.6` → `==0.1.7` (line ~90).
- `megatron-bridge>=0.5.1` (→0.5.1) / `megatron-core>=0.17.0` (→0.18.2) → pinned
  `megatron-bridge==0.6.1` / `megatron-core==0.19.0` (line ~110), in the same single
  `pip install -c /tmp/torch_constraints.txt ...` as before (the post-steps that restore
  `transformers==5.8.1` / `timm==1.0.16` and bump `opentelemetry-sdk` still run after it).
- **`sglang[all]` → `sglang[all]==0.5.16`** (2026-08-30, line ~37). Was unpinned, so a rebuild
  resolved whatever was latest (0.5.17 / 0.5.18 by late Aug). 0.5.16 is the only recent sglang
  whose own flashinfer requirement (`flashinfer_python[cu13]==0.6.14`) matches the Containerfile's
  flashinfer pin — 0.5.17 wants `0.6.15.post1`, 0.5.18 wants `0.6.17`, so an unpinned sglang
  silently reintroduces the flashinfer skew that caused run `3209484`'s MLA `plan()` hang. It is
  also the layout the two verl-vs-sglang import-drift source patches target. sglang + flashinfer
  must be bumped together, deliberately. (Note: sglang 0.5.16 declares `transformers==5.12.1` but
  the Containerfile forces `transformers==5.8.1` a few lines down — a pre-existing skew that has
  nonetheless worked for this recipe through many runs; left as-is, not part of this pin.)
  **The `transformers==5.8.1` pin's origin** (traced 2026-08-30, commit `c8fca4a`, the commit that
  first added megatron-bridge): `megatron-bridge` + `nvidia-modelopt[hf]` pull in a newer
  transformers as a transitive dep, and unconstrained that resolution *also* drags
  torch/torchvision to a different version than sglang compiled its extensions against → ABI break
  (same family as run `3235127`'s transformer-engine break). The explicit `transformers==` /
  `timm==` re-pin (now also `-c torch_constraints.txt`, added in `802a996`) restores what sglang
  wants without letting torch move. The *value* `5.8.1` is just what unpinned `sglang[all]`
  resolved to in July 2026 — now lagging sglang 0.5.16's declared `5.12.1`, hence pip's
  non-fatal incompatibility warning at build. **Follow-up (user decision, 2026-08-30): test
  `transformers==5.12.1` to match sglang 0.5.16, but only AFTER the current `delta_sharded`
  validation run completes — not folded into it, since it's a deliberate re-validation of its
  own.**
- `ARG VERL_REF` `"v0.8.0"` → `"v0.9.0"` (line ~25). Every actively-run verl script already does
  a runtime `git checkout -f v0.9.0` of `/workspace/verl` (both GLM v1-separate scripts, the
  qwen-3B v1-separate script, both Apertus v1.5 benchmarks), so this makes the image match what
  they already run and turns those runtime checkouts into no-ops. The V1 trainer
  (`verl/trainer/ppo/v1/`) only exists in v0.9.0. Scripts that rely on the stock image verl
  *without* a runtime checkout — `train-gsm8k-apertus-8B*.sh`, `train-gsm8k-qwen-3B.sh`,
  `train-gsm8k-glm5.1-700B-full-async-megatron.sh` — move v0.8.0→v0.9.0 with this bump, untested
  for those specific recipes. The CI image test (`tests/rl-async-benchmark.sh`) itself does a
  runtime `git reset --hard` to the theely fork, so it does not exercise the stock ref either
  way.

**Status (2026-08-30)**: image **built and tagged** — `jfrog.svc.cscs.ch/.../verl:alps7-dev-a9f9e56471c0574e`
("image with update dependencies"), from the current Containerfile. The user updated the
`VERL_IMAGE` tag in `train-gsm8k-glm5.1-700B-v1-separate-async-megatron.sh` and asked to strip
the now-redundant runtime upgrades. **Done** (2026-08-30): removed from that script — the
TransferQueue / megatron-core / megatron-bridge / flashinfer wheel fetch+sbcast blocks (batch
host) *and* the `pip install --no-deps --force-reinstall` blocks (srun), *and* the runtime
`git fetch/checkout v0.9.0` (image is v0.9.0). Replaced the scattered version-print / R3-prereq
diagnostics with one consolidated **non-fatal** image-version + import smoke test (`verl`,
`TransferQueue`, `megatron-core/bridge`, `flashinfer-python/cubin`, `sglang`, `transformers`
versions + the `megatron.training.models.gpt` / `sglang.srt.state_capturer.routed_experts` /
`sglang...model_runner_components.weight_updater:LocalSerializedTensor` import chains) so a wrong
image tag is obvious in ~30 s instead of via a downstream crash. Net −170 lines. **Still applied
at runtime** (NOT in the image): the verl *source* patches — PR #7421/#7422/#7423,
`v1-separate-async-fixes.patch`, `r3-sglang-routed-experts-import-fix.patch`,
`wsync-debug-progress-log.patch` (diagnostic), `delta-sharded-localserializedtensor-import-fix.patch`
— plus the megatron-bridge `safe_config_loader` filelock patch and the Lustre cache redirects.
**Open risk**: the Containerfile does not pin `sglang` (`sglang[all]`), so this rebuild could
have resolved a version other than 0.5.16 — which would break the two sglang-import source
patches. The new consolidated diagnostic block checks exactly this (WARNs if either patched
import path is absent). The other verl scripts (Apertus benchmarks, older GLM/qwen) still carry
their own runtime upgrades and are unaffected by this cleanup — it was done only for the GLM
v1-separate script the user pointed at the new tag.

### flashinfer packaging — KEY: `flashinfer-cubin` / `flashinfer-jit-cache` are NOT (fully) on PyPI

**Their real distribution channel is `https://flashinfer.ai/whl`** (a plain HTML index that serves
the GitHub release assets, `github.com/flashinfer-ai/flashinfer/releases/download/vX.Y.Z/...`).
flashinfer's own [install docs](https://docs.flashinfer.ai/installation.html) say to install them
with `pip install flashinfer-cubin --index-url https://flashinfer.ai/whl` (and
`flashinfer-jit-cache --index-url https://flashinfer.ai/whl/cu130` for a CUDA-specific JIT cache).

- **PyPI `flashinfer-python`**: complete, up to date.
- **PyPI `flashinfer-cubin`**: **incomplete / lagging** — as of 2026-08-29 it stops at `0.6.13`,
  while `flashinfer.ai/whl` has `0.6.14` … `0.6.18`. ([GitHub issue #2133](https://github.com/flashinfer-ai/flashinfer/issues/2133)
  is someone hitting the same PyPI gap at `0.5.3`.)
- **PyPI `flashinfer-jit-cache`**: does not exist on PyPI at all.
- **`flashinfer-python` >= ~0.6.14 hard-raises at import** if an installed `flashinfer-cubin` is
  not the *exact* same version (`RuntimeError: flashinfer-cubin version (X) does not match
  flashinfer version (Y)`). `FLASHINFER_DISABLE_VERSION_CHECK=1` bypasses it (per-kernel content-hash
  AOT lookup + JIT fallback still work), but the clean fix is a matched pair — which requires
  pulling the cubin from `flashinfer.ai/whl`, not PyPI.
- **Sub-index layout**: `flashinfer.ai/whl/flashinfer-cubin/` and `.../flashinfer-python/` (arch-
  independent `py3-none-any` wheels); `flashinfer.ai/whl/cu130/flashinfer-jit-cache/` etc. for the
  CUDA-tagged jit-cache (`0.6.14+cu130-cp39-abi3-manylinux_2_28_aarch64.whl`). Nightlies at
  `flashinfer.ai/whl/nightly/`.

Found while debugging run `3214410` (the GLM script bumped `flashinfer-python` to 0.6.14 — sglang
0.5.16's pin — and every worker then died on the cubin-mismatch `RuntimeError` because I had only
checked PyPI, concluded no 0.6.14 cubin existed, and reached for the bypass). The GLM script and
the Containerfile now install the matched `flashinfer_{python,cubin}==0.6.14` pair from
`flashinfer.ai/whl` (the script via `${TRAINING_HOME}/wheels/` Lustre staging because the 458 MB
cubin bus-errors on `sbcast`; the Containerfile via `--extra-index-url https://flashinfer.ai/whl`).

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

### Run `3139371` — 2026-08-21 — diagnostic confirms dense (not degraded), and disproves the list_of_dict_to_tensordict fix theory

- **Log**: `~/Downloads/slurm-3139371.out` (1,016,332 bytes). 80 nodes, verl v0.9.0, system
  `clariden`. Ran ~45 min before Slurm-forced termination (exitCode 15).
- **Symptom**: all four sitecustomize confirmation lines fired correctly, no regression. The
  `AttributeError` recurred at the same call site, same branch, same pattern as 3136766/3137775.
  The new diagnostic wrapper's output (the deliverable of this run) showed exactly two states,
  ~95 occurrences total via Ray's log dedup:
  - ~68×: `tensor_arg: OK nested(jagged) ... | prompts: OK nested(jagged) ... | responses: DENSE
    (not nested at all) type=Tensor dtype=torch.int64 shape=(2, 256)` — correlates with the crash.
  - ~28×: everything `OK nested(jagged)` — no crash.
  **No "DEGRADED nested(strided, NO .offsets()!)" state was ever observed**, for any field, in
  the whole log. This cleanly refutes the dtype-mismatch/strided-fallback hypothesis from 3137775
  — `responses` isn't a jagged-construction-then-fallback casualty, it's genuinely dense from the
  start. `shape=(2, 256)` — `256` is exactly `data.max_response_length`, i.e. both responses in
  this micro-batch of 2 (matching `PPO_MINI_BATCH_SIZE=48`/`dp_size=24` ⇒
  `mini_batch_size_per_gpu=2`) genuinely hit the length cap, the classic "undertrained model
  rambles to max_response_length" GRPO scenario.
- **Root cause — still not found, but the 3137775 fix theory is now disproven**: re-traced
  `simple_storage_manager.py`'s write-time slicing (`_select_by_positions`) and realized it
  always `.unbind()`s a batch into individual per-sample raw tensors before storing them,
  regardless of whether the TensorDict handed to it was dense or nested. So whatever
  `list_of_dict_to_tensordict` produces (nested, since 3137775's patch; dense, before it) gets
  erased at the storage boundary either way — each sample is always stored as its own raw 1D
  tensor. That patch is real, correct, and confirmed firing, but was **never structurally capable
  of fixing this crash** — the corruption has to be on the *read* side, in
  `AsyncSimpleStorageManager._pack_field_values`'s reconstruction of 2 stored per-sample
  `responses` values back into a batch. A local repro of the naive case (two same-length,
  contiguous, CPU, int64 1D tensors) through the *actual* `_pack_field_values` (installed the
  real `transferqueue-0.1.7-py3-none-any.whl` locally, patched it with the same find_spec
  mechanism, called the real function) stayed properly nested — so whatever differs about the
  *real* stored items (dtype, device, contiguity, or something in the multi-threaded
  per-storage-unit gather in `get_data`) has to be observed directly from production, not
  guessed again from a synthetic repro.
- **Fix**: none yet — another diagnostic-only run. Added a fifth sitecustomize target,
  `transfer_queue.storage.managers.simple_storage_manager` → `_patch_tq_storage_diagnostics`
  (confirming the same `find_spec` mechanism works identically well on the external pip-installed
  `transfer_queue` package as on verl's own modules — no reason it wouldn't, but verified rather
  than assumed). Wraps `AsyncSimpleStorageManager._pack_field_values` to print per-item
  dtype/device/contiguity/shape/is_nested for every tensor-field values-list it's about to pack,
  plus a side-probe of `torch.nested.as_nested_tensor(..., layout=jagged)` on the *exact* same
  input (does not alter the real return value) reporting success or the literal exception. This
  goes one level deeper than the `no_padding_2_padding` wrapper from 3137775 (kept active
  alongside it) — straight into the suspected function itself. Verified end-to-end locally against
  the real installed wheel before submitting (not just syntax-checked): the wrapper fires, prints
  correctly, and preserves the original nested-tensor return value. Checked-in patch file and the
  script's heredoc copy verified byte-identical.
- **Commit**: not committed.

### Run `3141801` — 2026-08-21 — new, unrelated hang: 30-min Gloo timeout in megatron-bridge's weight sync, before training started

- **Log**: `~/Downloads/slurm-3141801.out` (419 KB). 80 nodes, verl v0.9.0, system `clariden`.
  Ran 16:34:00–17:57:56 UTC (~84 min, under the 2h limit) before Slurm force-terminated all 80
  nodes (exitCode 15).
- **Symptom**: three of five sitecustomize confirmation lines fired (hybrid-rollout disable,
  `list_of_dict_to_tensordict` patch, `no_padding_2_padding` diagnostic wrapper) — no OOM, no
  regression there. The other two did **not** fire, for a benign reason each: `on_init_end`'s
  print only runs after `standalone_checkpoint_manager.update_weights()` returns, and that call
  is exactly what hung; the new `_pack_field_values` diagnostic's module is only imported once
  actual rollout experience flows through TransferQueue, which never happened. **No
  `DIAGNOSTIC _pack_field_values` or `DIAGNOSTIC no_padding_2_padding` lines and no offsets
  `AttributeError` appeared anywhere** — this run is inconclusive for the diagnostic chain it was
  meant to advance; see the new "Known hazards" entry above for the actual failure
  (`RuntimeError: ... gloo/transport/tcp/unbound_buffer.cc:78 ... Timed out waiting 1800000ms``
  in `param_mapping.py:460 broadcast_obj_from_pp_rank`'s `all_gather_object`, inside the very
  first `trainer.init()` → `on_init_end()` weight sync).
- **Root cause**: not yet found — first occurrence of this specific hang. Everything before it
  in the log looked completely normal (dataset loaded, worker groups created, patches installed
  cleanly).
- **Fix**: none yet. This pre-empts the offsets/padding diagnostic chain entirely — need to get
  past this hang before a future run can even reach the code the `_pack_field_values`/
  `no_padding_2_padding` diagnostics target.
- **Commit**: not committed.

### Run `3144665` — 2026-08-22 — Gloo hang was a one-off; offsets crash recurred at yet another site, and the 5th patch never installed

- **Log**: `~/Downloads/slurm-3144665.out`. 80 nodes, verl v0.9.0, system `clariden`. Ran 68 min
  (well within the 2h limit) — a real crash, not a timeout. Identical script to 3141801, retried
  unchanged on the theory the Gloo hang was a non-deterministic race.
- **Good news**: the Gloo `all_gather_object` hang from 3141801 **did not recur** —
  `on_init_end: skipped self.checkpoint_manager.update_weights` fired this time, confirming
  `update_weights()` succeeded. Treat 3141801 as a one-off race, not a repeatable blocker, unless
  it recurs again.
- **Symptom**: patches 1–3 confirmed firing normally (hybrid-rollout disable, update_weights
  skip, `list_of_dict_to_tensordict`). Patch 4 (`no_padding_2_padding` diagnostic) confirmed
  *installed* but never invoked — zero `DIAGNOSTIC no_padding_2_padding` lines — because the
  crash this run took a different path that doesn't call that function. **Patch 5
  (`_pack_field_values` diagnostic) never installed at all** — its confirmation line
  ("transfer_queue storage: installed diagnostic wrapper around
  AsyncSimpleStorageManager._pack_field_values") is completely absent from the full 4454-line
  log, meaning the `find_spec` hook for
  `transfer_queue.storage.managers.simple_storage_manager` was never triggered in any process.
  The offsets `AttributeError` recurred, but at the **original** 3134772 call site again —
  `verl/workers/engine_workers.py:294`, `mini_batch_td["input_ids"].offsets().diff().tolist()`
  — not `padding.py:119` (3136766/3137775/3139371's site). Three `WorkerDict` actors hit it
  simultaneously ~2m24s into step 0/231, killing the job. Neither diagnostic wrapper covers this
  call site's data, so no per-item state was captured for `input_ids` here either.
- **Investigated (free, no cluster cost) why patch 5 never installed**: confirmed the target
  module path is correct —
  `transfer_queue/storage/managers/simple_storage_manager.py:60` really does
  `@StorageManagerFactory.register("SimpleStorage")`, matching the script's
  `backend.storage_backend: SimpleStorage` config. Confirmed
  `transfer_queue/storage/managers/__init__.py` eagerly imports it
  (`from .simple_storage_manager import AsyncSimpleStorageManager`), and that `transfer_queue`
  storage was definitely in active use this run (the crash itself is data read out of it) — so
  the module must have been imported by *something*, somewhere, contradicting the total absence
  of the confirmation line. Root cause of the absence not yet found — candidates not yet
  checked: a Ray fork/prewarm worker pool where some process's `sys.modules` already contains
  this module from before our `sys.meta_path` hook installed (would need confirming Ray's actual
  worker-startup model); a stale/mismatched `transfer_queue` build in the container image
  differing from the `TransferQueue==0.1.7` wheel pulled fresh from PyPI for local testing; or
  the print firing but being lost to Ray's log deduplication/buffering in a way patches 1–3
  weren't (same `print(flush=True)` mechanism, so this would be surprising).
- **Fix**: none yet. This is now the third consecutive run (3137775, 3139371, 3144665 — plus the
  non-arriving 3141801) that failed to capture the specific diagnostic data needed, each for a
  different reason (crash moved to an uncovered call site; new unrelated hang; patch never
  installed). Also worth noting: the crash's call site is not deterministic across runs
  (`engine_workers.py:294` in 3134772 and now 3144665; `padding.py:119` in 3136766/3137775/
  3139371) — whatever the underlying corruption is, which field/site it first trips on varies
  run to run.
- **Commit**: not committed.

**Follow-up (same day, no cluster cost): root cause found.** The "candidates not yet checked"
list above turned out to have the answer in its second entry. `Alps-Images/apps/verl/Containerfile`
pins `TransferQueue==0.1.6` at image build time; `../verl`'s own `requirements.txt` (what every
diagnostic/fix in this chain was built and tested against) declares `0.1.7`. The script's runtime
verl checkout never upgrades verl's own pip dependencies, so the image's 0.1.6 was never touched.
0.1.6's actual module is `simple_backend_manager.py` (not `simple_storage_manager.py` — hence
patch 5 never matching anything), and its `_pack_field_values` has the exact shape-equality bug
this whole chain was hunting, already fixed in 0.1.7. See the rewritten "Known hazards" entry
above for the full story and the real fix, now in the script: upgrade the wheel at runtime rather
than patch a private method in an external package. `example/patches/sitecustomize-verl-v0.9.0.py`
had its (mis-targeted, dead) fifth patch target removed accordingly; the fourth
(`no_padding_2_padding` diagnostic) stays for one more run as confirmation, then can be removed.

### Run `3149339` — 2026-08-22 — the fix's own verification print broke the srun script before it could run

- **Log**: `~/Downloads/slurm-3149339.out`. 80 nodes, `clariden`. Failed in ~33s — batch step
  exit code 15, every node's task `Terminated`/exit 1-2 uniformly (not the partial/per-node
  pattern from run 3124273).
- **Symptom**: `` -c: line 31: unexpected EOF while looking for matching `"' `` plus raw script
  source text leaking into stdout — the whole per-node training script (the big
  `srun ... bash -c '...'` block, lines 759–970) never actually ran as intended.
- **Root cause**: this run's own new verification line,
  `python3 -c "import importlib.metadata; print('TransferQueue version:', ...)"`, used single
  quotes for its Python string literals — but it sits *inside* the outer `bash -c '...'` wrapper,
  which is itself single-quoted. Bash's outer single-quote parsing has zero escape mechanism and
  no concept of "nested" quotes for an inner interpreter: the very first `'` character anywhere
  inside that whole ~200-line block ends the outer single-quoted argument right there, at that
  exact byte, and everything after it becomes separate, unquoted tokens for the *outer* shell to
  parse directly. Two single quotes on one line keeps the outer script's raw quote-parity even, so
  `bash -n` reports no syntax error at all — this class of bug is silent until actual execution.
  The surrounding `python3 -c "..."` blocks (e.g. the filelock patch a few lines below) already
  used double-quoted Python literals for exactly this reason; this new line was the one
  inconsistent spot.
- **Fix**: escaped double quotes instead —
  `print(\"TransferQueue version:\", importlib.metadata.version(\"TransferQueue\"))` — verified
  by extracting the exact line into a local `bash -c '...'` wrapper and running it for real
  (prints `TransferQueue version: 0.1.7` correctly), and by grepping the full contents of all
  three single-quoted `srun ... bash -c '...'` blocks in the script for any other literal `'`
  character — none found. **The actual TransferQueue upgrade this run was meant to test remains
  unverified** — the job died in shell setup before `ray start` even ran, so nothing about the
  real fix (or the offsets bug) was confirmed or refuted here.
- **Commit**: not committed.

### Run `3149736` — 2026-08-22 — the TransferQueue 0.1.6→0.1.7 fix is confirmed correct and complete

- **Log**: `~/Downloads/slurm-3149736.out` (5941 lines). 80 nodes, verl v0.9.0, system `clariden`.
  Ran the full `--time=2:00:00` and was cleanly stopped by Slurm's time limit (`CANCELLED ...
  DUE TO TIME LIMIT`) — not a crash. Plus benign `Failed to destroy CXI Service ID 5: Device or
  resource busy` cleanup noise on kill, and the known-benign matmul_ext/Triton Lustre-lock
  `OSError` noise during init — neither fatal.
- **Result — this is the resolution of the entire 8-run chain
  (3134772/3136766/3137775/3139371/3141801/3144665/3149339/3149736)**:
  - `TransferQueue version: 0.1.7` printed identically on all 80/80 nodes — no version drift, no
    mixed-cluster hazard.
  - All four sitecustomize confirmation lines fired correctly (hybrid-rollout disable —
    `LLMServerManager` shows only the one standalone replica address, not 9 hybrid ones;
    `on_init_end` skip; `list_of_dict_to_tensordict` patch; `no_padding_2_padding` diagnostic
    wrapper). **No OOM** — a first for this recipe (runs 3121001/3125195/3129805 all OOM'd at
    this exact point). No Gloo hang (the 3141801 hang stays a one-off).
  - **Zero occurrences of `AttributeError: 'Tensor' object has no attribute 'offsets'` in the
    entire log.** Every `DIAGNOSTIC no_padding_2_padding` check showed `OK nested(jagged)` for
    tensor/prompts/responses/attention_mask — never DENSE, never degraded-strided.
  - Training completed **14 of 231 steps** with sane, stable metrics before the time limit hit
    (loss ranged ≈ -0.20 to -0.05, grad_norm ≈ 0.14 to 0.25 across all 14 steps — no NaN/Inf, no
    instability).
- **Conclusion**: the TransferQueue 0.1.6→0.1.7 upgrade (see "Known hazards" above) is the
  correct, complete fix for the offsets bug — confirmed, not just theorized. The only reason this
  run didn't finish is that 2 hours is far too short for 231 steps at ~350–360s/step (~23h
  projected total); this was a shakedown run, not intended to complete. **Next step for a real
  training run: bump `--time` substantially (or reduce total steps) — no further fix needed for
  the offsets/OOM/quoting issues this chain was chasing.**
- **Commit**: not committed.

**Follow-up cleanup (same day, no cluster cost)**: with the three real fixes (hybrid-rollout
disable, stale weight-sync skip, `list_of_dict_to_tensordict`) confirmed stable end-to-end,
converted them from the `sitecustomize.py` runtime monkeypatch to a plain source patch,
`example/patches/v1-separate-async-fixes.patch`, applied via `git apply` against
`/workspace/verl` alongside the upstream PR patches — same conversion, same reasoning, as the
Apertus FSDP2 benchmark's equivalent (`sglang-apertus1p5-local-fixes.patch`): a monkeypatch is
fine for fast iteration but not the form a stable fix should end up in. Verified locally before
committing: generated the diff from a real edited `git worktree` at the v0.9.0 tag (not
hand-written), confirmed it applies cleanly with `git apply --check` against a fresh v0.9.0
checkout, confirmed `git apply --reverse --check` correctly detects "already applied" (the same
idempotency check the PR-patch loop relies on), and confirmed all three edited files still
compile. The now-diagnostic-only `no_padding_2_padding` wrapper and the whole
`sitecustomize-verl-v0.9.0.py` mechanism (heredoc, `sbcast`, `PYTHONPATH`/`SITECUSTOMIZE_LOCAL`
staging) are removed from both the script and the repo — this script no longer runs any
`sitecustomize.py` at all. `example/patches/sitecustomize.py` (older, written against v0.8.0,
never used by this script) is untouched.

### Run `3152802` — 2026-08-22 — cleanup verified clean; new TP=32 SGLang broadcast hang during generation

- **Log**: `~/Downloads/slurm-3152802.out`. 80 nodes, verl v0.9.0, system `clariden`. Ran the
  full `--time=2:00:00` to `TIMEOUT` (exitCode 0) — not a fast crash, a slow one: dead in an
  infinite retry loop for the last ~70 minutes of the run.
- **Purpose**: verify the sitecustomize→source-patch cleanup (previous entry) introduced no
  regression versus the confirmed-good run 3149736.
- **Result — cleanup confirmed clean**: weight sync from trainer to standalone rollout
  completed successfully (346.80s, ~3.97 GB/s per rank, all 32 ranks). All three converted
  fixes verifiably took effect (no hybrid-rollout OOM, `v1-separate-async-fixes.patch` applied
  on 80/80 nodes, no stale-weight-sync error). `TransferQueue version: 0.1.7` confirmed on all
  80 nodes. Zero occurrences of the offsets `AttributeError` this whole chain was chasing.
- **New, unrelated symptom**: ~9 minutes into ordinary rollout generation (well after the
  successful weight sync), one SGLang TP rank (TP=32 across the 8-node standalone rollout)
  fired its own 300s scheduler watchdog timeout, its py-spy dump showing it blocked inside
  `recv_requests → _broadcast_reqs_across_ranks → torch.distributed.broadcast` — a stuck
  NCCL/collective broadcast. Took down the whole SGLang replica (Ray `SYSTEM_ERROR`), after
  which every `AgentLoopWorkerTQ` rollout task failed in an infinite `ActorDiedError` retry loop
  against the now-dead actor for the rest of the run — no recovery, no restart, no further
  training steps, just burning wall-clock until Slurm's `--time` limit killed it.
- **Root cause**: not investigated further pending a second data point (see run 3171176) — see
  the updated "TP=32 SGLang" entry in Known hazards above for the full mechanism and comparison
  against the already-documented engine-rebuild deadlock hazard (this one is distinct: happened
  during steady-state generation, not weight-sync/engine-rebuild timing).
- **Fix**: none — resubmitted unmodified as run 3171176 to determine one-off vs. repeatable.
- **Commit**: not committed.

### Run `3171176` — 2026-08-22 — retry confirms the SGLang hang was a one-off; recipe considered solid

- **Log**: `~/Downloads/slurm-3171176.out` (partial — monitoring stopped once the verification
  purpose was met, job left running). 80 nodes, verl v0.9.0, system `clariden`. Unmodified
  resubmission of the exact script from 3152802 — no code changes. Notably fast queue turnaround
  this time (~99s to RUNNING, vs. minutes-to-~1h40m in prior runs).
- **Result**: the SGLang TP=32 broadcast hang from 3152802 **did not recur**. Standalone rollout
  came up cleanly; training reached 3 consecutive completed steps (loss -0.06 → -0.12 → -0.09,
  grad_norm 0.17 → 0.24 → 0.18, both finite and stable; critic/reward mean 0.08 → 0.15 → 0.12,
  noisy-but-expected for early GRPO) spanning ~15–29 minutes after rollout startup — comfortably
  past the ~9-minute mark where 3152802 died. Zero occurrences anywhere in the log of the
  watchdog-timeout/`SYSTEM_ERROR`/`ActorDiedError`/stuck-broadcast signature. `TransferQueue
  version: 0.1.7` and all four patches (PR #7421/#7422/#7423 + `v1-separate-async-fixes.patch`)
  confirmed applied cleanly on 80/80 nodes again. Only noise: the already-documented benign
  Triton-autotune-cache `Stale file handle` atexit errors.
- **Conclusion**: 3152802's hang is a one-off infra flake (most likely Slingshot/NCCL over the
  8-node TP=32 group), not a repeatable bug — no code differs between the two runs, and the
  second run cleared the exact window where the first one died. **The recipe end-to-end
  (hybrid-rollout OOM fix, `list_of_dict_to_tensordict` fix, TransferQueue 0.1.7 upgrade, and
  the sitecustomize→source-patch cleanup) is now considered solid.** If the TP=32
  watchdog/broadcast signature recurs in any future run, treat it as a real, repeatable hazard
  worth investigating from the py-spy dump — but on the evidence so far it isn't one.
- **Commit**: not committed.

### Run `3199623` — 2026-08-27 — R3 + THD test: both new prerequisites confirmed missing, new unexplained CUDA crash, reverted

- **Log**: `~/Downloads/slurm-3199623.out` (416,701 bytes / 5,198 lines). 80 nodes, verl v0.9.0,
  system `clariden`. FAILED after ~275s (~4.6 min) — exitCode 15, never reached weight loading or
  SGLang rollout startup.
- **Purpose**: first real test of the 2026-08-26 Configuration audit change — R3 router replay
  plus the `use_remove_padding: True` it depends on, both entirely new to this recipe.
- **The two new diagnostic prints (the run's other purpose) both surfaced real, confirmed
  gaps**: `megatron-bridge version: 0.5.1` (not the ≥0.6.0 the THD-DSA research pointed to —
  its accompanying `WARNING` fired on all 80 nodes) and `WARNING: sglang routed_experts_capturer
  not importable (No module named 'sglang.srt.layers.moe.routed_experts_capturer')` (also all 80
  nodes) — this image's `sglang[all]` build has no such module at all, so R3's rollout-side
  capture cannot function regardless of anything on the Megatron side.
- **Symptom**: never reached SGLang or weight loading (zero SGLang log lines, no `Loading
  weights`). All 287/288 trainer ranks got as far as: `enable_routing_replay in MegatronEngine:
  True` → `Applying Router Replay Patch...` → (non-fatal) `Only support config type of [...], but
  got glm_moe_dsa. MFU will always be zero` (R3's model-type allowlist doesn't include
  `glm_moe_dsa` — cosmetic) → (non-fatal, already-known DSA hazard) `fast_hadamard_transform is
  unavailable; falling back to a pure-torch Walsh-Hadamard transform` → per-rank parameter counts.
  Then one single worker (`pid=252380, ip=172.28.44.120`) crashed:
  ```
  File ".../megatron/core/distributed/param_and_grad_buffer.py", line 1216, in __init__
      param.data.detach().copy_(old_param_data)
  torch.AcceleratorError: CUDA error: unspecified launch failure
  ```
  Only one occurrence of any CUDA/NCCL error in the whole log; every other rank was killed
  afterward via Ray, not independently faulting.
- **Root cause — not conclusively found.** "unspecified launch failure" is CUDA's generic
  async-reported symptom of an earlier bad kernel launch (the traceback itself warns the reported
  stack may be misleading). It hit in Megatron's DDP grad-buffer construction, immediately
  downstream of DSA-attention + router-replay-patch init — the two code paths this run newly
  exercises. Best-supported explanation: a bad/unsupported CUDA kernel invocation somewhere in
  DSA's THD-jagged handling or the router-replay patch on this specific (too-old, per the
  diagnostic) megatron-bridge version, corrupting the CUDA context, surfacing on an unrelated
  buffer copy a few calls later. Does not match the signature or phase of any previously
  documented hazard for this script (OOM, Gloo hang, sequence_parallel assertion, lr_decay_steps,
  filelock, xielu, TP=32 hang) — genuinely new. Caveat: only 1 of 288 ranks hit it, so a one-off
  hardware/driver flake on that single GPU can't be fully ruled out either; unlike the TP=32 hang
  precedent (3152802/3171176), this was never retried unmodified to check.
- **Note on job lifecycle**: a cancellation was attempted mid-investigation (before the failure
  was confirmed) — `DELETE`/`scancel` returned HTTP 500 / `Invalid job id specified`, because the
  job had already gone terminal in Slurm before the cancel reached it. No effect either way; the
  job's actual failure is what's documented here, not a cancellation.
- **Fix**: reverted. `actor_rollout_ref.model.use_remove_padding` back to `False`,
  `actor_rollout_ref.actor.megatron.router_replay` removed (back to disabled),
  `actor_rollout_ref.rollout.enable_rollout_routing_replay` removed (back to unset/`False`) — see
  this script's Configuration audit entry above for the full reasoning and what a real re-attempt
  needs (megatron-bridge upgraded to ≥0.6.0 at runtime, and a real, unguessed answer on the
  deployed image's actual SGLang version and where `routed_experts_capturer` actually lives in
  it). The two diagnostic prints themselves stay in the script — harmless, non-fatal, and useful
  signal for whenever this is retried.
- **Commit**: not committed.

### Probe job `3199799` — 2026-08-27 — cheap 1-node check answers both open questions from 3199623

- **Log**: `~/Downloads/slurm-3199799.out`. 1 node, `clariden`, ran 61s, COMPLETED. Not a training
  run — a minimal probe (no verl checkout, no config) that starts the same image and just prints
  installed package versions and searches the installed `sglang` tree for the router-replay
  capture feature. Submitted specifically to avoid spending another 80-node allocation on
  diagnostics alone.
- **Installed versions**: `sglang: 0.5.16`, `megatron-bridge: 0.5.1`, `megatron-core: 0.18.2`,
  `transformers: 5.8.1`, `torch: 2.11.0`, `TransferQueue: 0.1.6`. Confirms 3199623's
  `megatron-bridge` finding exactly; `TransferQueue: 0.1.6` matches the Containerfile's hard pin
  (expected — this script's own runtime upgrade to 0.1.7 only takes effect inside the real GLM
  script's srun, not this bare probe).
  `sglang[all]` has no version pin anywhere in `Alps-Images/apps/verl/Containerfile` (confirmed
  by reading it directly) — 0.5.16 is simply whatever PyPI resolved to at image-build time, and
  this probe is now the only record of what that actually is.
- **`routed_experts_capturer` — found, not missing, just relocated.** Not importable at
  `sglang.srt.layers.moe.routed_experts_capturer` (the path verl v0.9.0's
  `async_sglang_server.py` imports from) — but a full filename+content search of the installed
  `sglang` tree found it at `sglang/srt/state_capturer/routed_experts.py` (module renamed too,
  `routed_experts_capturer.py` → `routed_experts.py`), with `enable_return_routed_experts`
  present in both `srt/server_args.py` and that same new file. **This is verl v0.9.0 shipping a
  stale import path against sglang 0.5.16's current layout, not a genuinely missing feature** —
  changes the fix from "wait for an upstream feature" to "patch one import path," much smaller in
  scope than previously assumed.
- **Two independent, now precisely-scoped fixes needed before retrying R3 + THD**:
  1. Patch verl's `async_sglang_server.py` to import from `sglang.srt.state_capturer.routed_experts`
     instead of `sglang.srt.layers.moe.routed_experts_capturer` — small, source-level, same
     `find_spec`/source-patch discipline already used elsewhere in this script.
  2. Upgrade `megatron-bridge` from 0.5.1 to ≥0.6.0 at runtime — same wheel-fetch/sbcast/install
     pattern already proven for `TransferQueue` 0.1.6→0.1.7 in this exact script.
  Neither has been implemented yet as of this entry.
- **Commit**: not committed.

**Implementation, 2026-08-27 (same day)**: both fixes above are now in the script, neither yet
tested on a cluster.

1. **SGLang import-path fix** — new checked-in patch,
   `example/patches/r3-sglang-routed-experts-import-fix.patch`, a one-line change to
   `verl/workers/rollout/sglang_rollout/async_sglang_server.py` re-pointing the lazy import from
   `sglang.srt.layers.moe.routed_experts_capturer` to `sglang.srt.state_capturer.routed_experts`.
   Verified before wiring in, same discipline as every other patch in this file: generated from a
   real edited `git worktree` at the v0.9.0 tag (not hand-written), confirmed `extract_routed_
   experts_from_meta_info` exists with the same name/signature at the new path by fetching the
   real `sglang` `v0.5.16` tag source directly, confirmed the patch applies cleanly
   (`git apply --check`) and the patched file compiles (`python3 -m py_compile`) against a fresh
   v0.9.0 checkout, and confirmed `git apply --reverse --check` correctly detects "already
   applied." Embedded into the script as a heredoc (same `BASH_SOURCE[0]`-under-`sbatch` reasoning
   as `v1-separate-async-fixes.patch`), `sbcast`, and applied via the same
   apply-or-already-present-or-fatal `git apply` loop as the other verl-source patches, right
   after `v1-separate-async-fixes.patch`. Caught and fixed one real bug while wiring this in: the
   round-trip risk this repo has hit before (Write/Edit silently trimming a trailing
   whitespace-only line, per the Apertus FSDP2 broadcast-diagnostic incident) was avoided by
   building the checked-in patch file from a real `git diff` output via direct file
   concatenation rather than retyping it, and verified byte-identical between the checked-in file
   and what actually landed in the script's heredoc by extracting it back out and diffing.
2. **megatron-bridge upgrade** — 0.5.1 → 0.6.1 (the latest published release; 0.6.0 also exists
   but 0.6.1 is newer by version number despite an earlier PyPI upload timestamp, an oddity not
   investigated further). Both were published to PyPI on 2026-08-19/20 — about a week before this
   image's probe found it resolved to 0.5.1, consistent with the earlier theory that the image
   simply predates 0.6.0's release rather than something pinning it down. Fetched the real wheel
   (`megatron_bridge-0.6.1-py3-none-any.whl`), verified its sha256 locally before embedding the
   hash in the script, and wired it in with the same fetch-once/sbcast/`pip install --no-deps
   --force-reinstall`/verify-version-by-printing-it pattern already proven for TransferQueue
   0.1.6→0.1.7 in this exact script. Placed the install **before** the existing
   `safe_config_loader` filelock patch (which targets a file inside the `megatron-bridge`
   package) — ordering matters, since `--force-reinstall` rewrites every file in the package and
   would silently undo a patch applied to it beforehand. One genuinely unverified residual risk,
   stated plainly in the script's own comment: `megatron-bridge` 0.6.1's PyPI metadata declares an
   unversioned `megatron-core[dev,mlm]` dependency (no explicit floor), so leaving the image's
   `megatron-core` at 0.18.2 via `--no-deps` is plausible but not confirmed compatible with
   0.6.1's actual code — first real signal on this is whether the next run gets past model
   construction, not something checkable without a run.

Also updated the existing `routed_experts_capturer` diagnostic print to check the new, confirmed
path instead of the old one (and to separately flag if the *old* path unexpectedly starts working
too, which would mean the deployed sglang changed again and the patch itself needs
re-pointing) — otherwise the diagnostic would keep reporting the exact problem the new patch just
fixed. **While editing that diagnostic's comment text, introduced and then caught (via the
established stray-single-quote scan of the whole `srun bash -c '...'` body, not by a failed run)
two literal apostrophes — "the image's sglang" and "that isn't guaranteed" — inside the outer
single-quoted wrapper: the exact run-3149339 quoting hazard.** Both rephrased to avoid
apostrophes entirely and the full body re-scanned clean before considering this done. Every
change in this implementation pass is entirely unverified end-to-end by an actual cluster run.

### Run `3201189` — 2026-08-27 — third R3+THD attempt: a new quoting bug, then the real megatron-core blocker

- **Log**: `~/Downloads/slurm-3201189.out` (also fetched directly to
  `scratchpad/slurm-3201189-check.out` while diagnosing). 80 nodes, verl v0.9.0, system
  `clariden`. FAILED, exitCode 15, elapsed 4142s (~69 min) — the longest of the three R3 attempts
  by far, and the first to actually reach real model construction.
- **Submission itself hit two layers of friction, both resolved, neither a real infra problem**:
  (1) background `train-launcher` dispatches were blocked twice by the Claude Code auto-mode
  permission classifier before ever reaching FirecREST — resolved by submitting directly from the
  foreground coordinator session instead, where a live human could approve; (2) that direct
  submission then hit two separate brief CSCS-wide `clariden` outages (scheduler/SSH/filesystems
  unhealthy, confirmed via `/status/systems`) — one during submission (HTTP 503, cleared in
  ~1 minute) and one mid-run (~21 minutes, cleared on its own). Both self-resolved; neither related
  to this recipe.
- **Bug 1 (self-inflicted, found and fixed same day): a backtick inside an unquoted heredoc.**
  The very first line of the log was `slurm_script: line 104: router_replay:: command not found` —
  bash trying to *execute* `router_replay: {mode: R3}` as a command. Root cause: the
  `grpo_gsm8k.yaml` heredoc uses an **unquoted** delimiter (`<<- EOF`, not `<<- 'EOF'`), so
  backticks inside it trigger real command substitution — and the 2026-08-26 audit-entry comment
  had written `` `router_replay: {mode: R3}` `` in backticks (markdown-style code formatting) as
  plain English prose, not realizing this heredoc evaluates its content as shell. Confirmed by
  reproducing locally (`bash -x` on the script's own first ~285 lines reproduced the identical
  error and pinpointed the exact backtick pair via the `++` xtrace prefix) — **not a cluster
  artifact, deterministic in a local sandbox too.** Verified the actual functional YAML was
  unaffected (the real `router_replay:`/`mode: R3` keys are on separate lines a few rows down,
  untouched — only the one comment line's text was garbled, replaced with the backtick command's
  empty stdout), so R3's config was not the reason this run ultimately failed. Fixed by removing
  the backticks; **found and fixed a second, self-introduced instance of the identical bug while
  editing this same comment block during the subsequent revert** (a
  `` `ModuleNotFoundError: ... 'megatron.training.models.gpt'` `` backtick-quoted error message,
  same heredoc) — this class of hazard needs checking on every edit to these heredocs, the same
  way the single-quote scan is already routine for the `srun bash -c '...'` body. **New standing
  rule, added to Known hazards below: any edit inside an unquoted heredoc (`env.toml`,
  `grpo_gsm8k.yaml`, `gsm8k_reward.py`, `prepare_gsm8k.py`) must be scanned for backticks and bare
  `$(`/`$VAR`, the same discipline already applied to the single-quoted `srun` body for stray `'`.**
- **Bug 2 (the real blocker): megatron-bridge 0.6.1 requires megatron-core ~0.19.0, not this
  image's 0.18.2.** After the backtick fix, the run got further than either prior R3 attempt —
  past setup, patch application (`r3-sglang-routed-experts-import-fix.patch` and all others
  applied cleanly on all reporting nodes), and into real `actor_rollout_init_model()` — but every
  worker hit `ModuleNotFoundError: No module named 'megatron.training.models.gpt'`, traced through
  `megatron/bridge/training/config.py` → `megatron/bridge/models/gpt/gpt_builder.py` →
  `from megatron.training.models.gpt import GPTModelBuilder, GPTModelConfig, mtp_block_spec`, from
  inside `create_ddp_config` — called unconditionally during Megatron model setup, **not gated on
  R3 or THD at all**. Root-caused (no cluster cost — pure research after the log came back):
  megatron-bridge's PyPI metadata declares an unversioned `megatron-core[dev,mlm]` dependency (no
  floor), but its actual GitHub repo pins an exact `3rdparty/Megatron-LM` git submodule commit —
  fetched that commit's `megatron/core/package_info.py` directly from GitHub for the `v0.6.0` tag
  and confirmed it declares `MAJOR = 0, MINOR = 19, PATCH = 0` — i.e. megatron-bridge 0.6.x is
  built and tested against Megatron-Core **0.19.0**, a real, non-adjacent jump from this image's
  0.18.2. A benign side-effect confirming the mismatch further: ~40+ repeated (one per worker)
  `UserWarning: Failed to import modelopt megatron.bridge plugin due to: ImportError("cannot
  import name 'MegatronMappingRegistry' from partially initialized module ...")` earlier in the
  same log — non-fatal on its own, but the same underlying version skew.
- **Fix**: reverted, more thoroughly than the 3199623 revert. All three R3/THD flags back to
  disabled (same as before), **and the megatron-bridge 0.6.1 upgrade itself fully removed** (wheel
  fetch/sbcast on the batch host, and the `pip install --no-deps --force-reinstall` in the srun) —
  this one doesn't just gate an unused feature when reverted, it was actively breaking core
  Megatron model construction for *any* config, so leaving it in place at all was strictly worse
  than the pre-this-session baseline. The `r3-sglang-routed-experts-import-fix.patch` stays
  applied (harmless, independently correct, just unexercised while R3 is off). The two diagnostic
  prints (megatron-bridge version, sglang routed-experts-capture path) stay too — the
  megatron-bridge one will now correctly report `0.5.1` with its `<0.6.0` warning again, which is
  accurate. Verified locally before considering this done: reproduced the script's first 282 lines
  in a local sandbox after all fixes and got a clean run with no unexpected errors; `bash -n` and
  full backtick/stray-quote sweeps (both the single-quoted `srun` body and all four unquoted
  heredocs) pass clean.
- **What a real next attempt needs**: this is now confirmed to require upgrading megatron-core
  alongside megatron-bridge, not megatron-bridge alone — and unlike megatron-bridge (a pure-Python
  wheel), megatron-core has real CUDA/compiled-extension dependencies (see the Containerfile's own
  `fast-hadamard-transform` stub workaround for `megatron-core[dev,mlm]`), so a naive
  `--no-deps` version bump carries meaningfully more risk of a *new*, different breakage (ABI/CUDA
  mismatch) than anything fixed so far in this chain. Given this is now the second consecutive
  real 80-node failure for this feature (3199623, 3201189), a third attempt should not be another
  blind version-bump guess — it needs either (a) real investigation into whether megatron-core
  0.19.x installs cleanly and stays ABI-compatible with this image's torch/CUDA stack before ever
  touching the training recipe again, or (b) treating this as an image-level fix (bumping both
  pins together in the Containerfile, rebuilt and tested through the normal CI pipeline) rather
  than a runtime patch. Not attempted in this session — flagged for explicit user decision before
  any further cluster spend on this feature.
- **Commit**: not committed.

**Follow-up, 2026-08-28 (user explicitly authorized): investigated and fixed the megatron-core
gap properly, validated cheaply, then wired into the real script for a third attempt.**

Research (no cluster cost): megatron-core 0.19.0 publishes a real `bdist_wheel` for this exact
platform — `cp312` (matches the image's Python 3.12), `manylinux_2_24_aarch64.manylinux_2_28_aarch64`
(matches GH200/aarch64) — confirmed by inspecting the wheel directly: only **one** compiled
extension inside (`megatron/core/datasets/helpers_cpp...so`, a CPU-only dataset-helper, no
CUDA/torch-ABI-sensitive code), and it does contain the previously-missing
`megatron/training/models/gpt.py`. Much lower risk than a generic "compiled package" swap would
suggest — most of Megatron-Core's actual CUDA-heavy code lives in separate packages
(`transformer-engine`, `apex`, `flash-attn`) that this wheel imports at runtime but doesn't bundle.

### Probe job `3207054` — 2026-08-28 — first validation attempt: a probe-script bug, not a real problem

- **Log**: `~/Downloads/slurm-3207054.out`. 1 node, `clariden`. FAILED in 34s.
- **Symptom**: `ERROR: Invalid wheel filename (wrong number of parts):
  'megatron_core-0.19.0-cp312-aarch64'` — the probe script's own `curl -o` saved the downloaded
  wheel under a shortened local filename that dropped required parts of the real 5-hyphen-part
  wheel filename (`megatron_core-0.19.0-cp312-cp312-manylinux_2_24_aarch64.manylinux_2_28_aarch64.whl`).
  pip parses metadata from a local wheel's filename directly and rejects anything that doesn't
  match the exact format, so the install (and every import smoke test after it) never ran.
- **Confirmed unaffected**: both wheels (megatron-bridge 0.6.1, megatron-core 0.19.0) downloaded
  and sha256-verified cleanly; baseline versions matched the existing write-up exactly
  (`megatron-bridge: 0.5.1`, `megatron-core: 0.18.2`).
- **Fix**: corrected the probe script to preserve the real full wheel filename throughout
  (download target, sha256 check, and the `pip install` call) — a scratch-script bug, not
  something that touched the real training script. Resubmitted immediately as `3207095`.

### Probe job `3207095` — 2026-08-28 — megatron-bridge 0.6.1 + megatron-core 0.19.0 confirmed compatible

- **Log**: `~/Downloads/slurm-3207095.out`. 1 node, `clariden`. **COMPLETED** in ~150s.
- **Result**: both wheels installed cleanly (`pip install --no-deps --force-reinstall`, megatron-core
  first then megatron-bridge, matching how the real script now orders them). Post-upgrade versions
  confirmed via `importlib.metadata`: `megatron-bridge: 0.6.1`, `megatron-core: 0.19.0`.
- **Every smoke test passed**: the exact import chain that crashed real run `3201189`
  (`megatron.bridge.training.config.DistributedDataParallelConfig`) — OK. The specific
  previously-missing module (`megatron.training.models.gpt` — `GPTModelBuilder`, `GPTModelConfig`,
  `mtp_block_spec`) — OK. A broader sweep of 9 other `megatron.core`/`megatron.bridge` import
  paths verl itself uses — 9/9 OK. The GLM-5.1-specific bridge module
  (`megatron.bridge.models.glm_moe_dsa`) — OK. The previously-observed benign `modelopt`
  circular-import `UserWarning` still fires (explicitly marked ignorable by modelopt itself,
  unchanged from before the upgrade) — no new warnings or errors introduced.
- **Caveat, stated plainly**: this validates that both packages *import* cleanly together and that
  the specific broken chain now resolves. It does not validate actual model construction, forward/
  backward passes, or anything GPU-execution-level — a 1-node probe with no real model/weights
  can't test that. The real signal on those still requires the 80-node recipe itself.
- **Fix wired into the real script** (`train-gsm8k-glm5.1-700B-v1-separate-async-megatron.sh`),
  same day: replaced the removed megatron-bridge-only upgrade with a combined
  megatron-core-then-megatron-bridge upgrade (same fetch-once/sbcast/`--no-deps
  --force-reinstall` pattern as TransferQueue and the original megatron-bridge attempt), added a
  megatron-core version + `megatron.training.models.gpt` import check to the existing diagnostic
  block, and re-enabled all three R3/THD flags (`use_remove_padding`, `router_replay.mode: R3`,
  `enable_rollout_routing_replay`) for a third real attempt. Verified locally before considering
  this done: `bash -n`, the full stray-single-quote scan of the `srun bash -c '...'` body, and the
  full backtick scan of all four unquoted heredocs (per the new standing rule from run `3201189`'s
  entry above) all pass clean; reproduced the script's own YAML-generation portion in a local
  sandbox with no unexpected output; confirmed the megatron-core wheel filename is byte-identical
  across all 5 places it appears in the script (learned from probe `3207054`'s bug). **Entirely
  unverified end-to-end on the real 80-node recipe — this is the third real attempt, needs a run.**

### Run `3207151` — 2026-08-28 — fourth R3+THD attempt: both prior blockers cleared, first-ever R3 rollout, new failure in verl's routed-experts capture (stale vs sglang 0.5.16)

- **Log**: `~/Downloads/slurm-3207151.out` (9660 lines / 953 KB, intact; backup at
  `scratchpad/goodlog.out`). 80 nodes, verl v0.9.0, system `clariden`. Submitted PENDING, ran
  ~39 min, then **user-authorized `scancel`** (FirecREST `DELETE /compute/clariden/jobs/3207151`
  → HTTP 204, state `CANCELLED`) once the failure was confirmed unrecoverable — it was in a
  cluster-wide infinite rollout-retry loop that would otherwise have burned the full 5h wall
  limit (the driver never incremented its own `failure` counter — `running: 24, finished: 0,
  failure: 0` held for ~7 min — because `AgentLoopWorkerTQ` retries internally forever).
- **Purpose**: first real test of the combined megatron-core 0.19.0 + megatron-bridge 0.6.1
  runtime upgrade (validated on probe `3207095`) with all three R3/THD flags re-enabled.
- **Furthest any R3+THD attempt has ever reached — both historical blockers are genuinely
  fixed**, confirmed by this run's own diagnostic prints and progress:
  - `megatron-core version: 0.19.0`, `megatron-bridge version: 0.6.1`,
    `megatron.training.models.gpt: OK` (the exact `ModuleNotFoundError` that killed `3201189`),
    `sglang routed_experts capture: OK at sglang.srt.state_capturer.routed_experts`,
    `TransferQueue version: 0.1.7` — every check green.
  - Passed `DistributedDataParallel contains 9.24B parameters` cleanly — the exact
    `_ParamAndGradBuffer.__init__` line where `3199623` died with a CUDA "unspecified launch
    failure". No CUDA error this run.
  - GLM5Bridge HF→Megatron weight conversion completed (`Loading from /tmp/glm_model_3207151 ...
    100% (6201/6201)`).
  - First trainer→rollout NCCL weight sync completed (`Rank 0 send weights done, 307.09s` /
    `Rank 25 receive weights done, total_params: 59079, 309.02s, 4.48 GB/s`, world_size 33).
  - SGLang standalone rollout serving (8-node replica, `HTTP server started`, DSA backends
    `flashmla_sparse`/`fa3`).
  - R3 router-replay patch initialized on all 288 trainer ranks
    (`enable_routing_replay in MegatronEngine: True`, `Applying Router Replay Patch...`,
    `routing replay layers: 23`/`26` per PP rank).
  - `Training Progress: 0/40` — training loop entered; TransferQueue live (first
    PUT_DATA/KV_RETRIEVE ops). Then failed at **first-step rollout generation** — no step
    metrics line ever produced.
- **Symptom**: every rollout call crashed, cluster-wide, first occurrence ~logline 6968:
  ```
  File ".../verl/workers/rollout/sglang_rollout/async_sglang_server.py", line 660, in generate
      routed_experts = captured.numpy() if captured is not None else None
  AttributeError: 'str' object has no attribute 'numpy'
  ```
  Path: `single_turn_agent_loop.run` → `server_manager.generate` → `llm_server.py:274`
  `server.generate.remote` → `SGLangHttpServer.generate`. Surfaced to `AgentLoopWorkerTQ` as
  `RayTaskError(AttributeError)`. Only one root cause in the entire log.
- **Root cause — verl's routed-experts capture is stale against sglang 0.5.16, one layer below
  the import-path patch.** `async_sglang_server.py`'s `enable_rollout_routing_replay` block has
  two branches (verl v0.9.0 *and* current `main`, identical):
  - `skip_tokenizer_init: True` (**this recipe** — confirmed `'skip_tokenizer_init': True` in the
    log's rollout config dump): `captured = output["meta_info"].get("routed_experts");
    routed_experts = captured.numpy() if captured is not None else None` — assumes `captured` is
    a torch tensor (an **old** sglang behavior).
  - `skip_tokenizer_init: False`: `from sglang.srt.layers.moe.routed_experts_capturer import
    extract_routed_experts_from_meta_info` then
    `extract_routed_experts_from_meta_info(output).reshape(-1, num_hidden_layers,
    num_experts_per_tok)`.
  `example/patches/r3-sglang-routed-experts-import-fix.patch` re-pointed the import **in the
  `else` branch only** — but this recipe's TP=32 standalone rollout takes the **first** branch,
  which never uses that import. In sglang 0.5.16, `output["meta_info"]["routed_experts"]` is a
  **base64-encoded string** (confirmed by reading `sglang/srt/state_capturer/routed_experts.py`
  at the `v0.5.16` tag: `extract_routed_experts_from_meta_info` does
  `np.frombuffer(pybase64.b64decode(routed_experts_base64.encode("utf-8")), dtype=np.int32)` and
  returns a **flat** `np.int32` array). So `.numpy()` on a `str` → `AttributeError`. The
  `else`-branch path — decode the base64, then `.reshape(-1, num_hidden_layers,
  num_experts_per_tok)` — is the one that actually matches sglang 0.5.16; the `skip_tokenizer_init`
  shortcut is simply obsolete.
- **Fix — implemented 2026-08-28, unverified on cluster.** Expanded
  `example/patches/r3-sglang-routed-experts-import-fix.patch` (checked-in file + the script's
  heredoc copy, verified byte-identical) from the earlier import-path-only one-liner to a
  full rewrite of the `enable_rollout_routing_replay` block in `async_sglang_server.py`: it
  **drops the dead `skip_tokenizer_init` tensor branch entirely** and always goes through
  `extract_routed_experts_from_meta_info(output).reshape(-1, num_hidden_layers,
  num_experts_per_tok)` (imported from `sglang.srt.state_capturer.routed_experts`, with an
  `ImportError` fallback to the old `sglang.srt.layers.moe.routed_experts_capturer` path so it's
  also valid against older sglang / as an upstream PR). Keeps the `captured is not None` guard
  and the `hf_config` `hasattr` check.
  - **The layer-count-reshape worry from the earlier draft of this entry is resolved** — read
    sglang 0.5.16's `state_capturer/routed_experts.py` directly: `RoutedExpertsCapturer`
    allocates its buffer `num_layers = hf_text_config.num_hidden_layers` wide (dense-layer slots
    stay zero), so reshaping by `num_hidden_layers` is exactly right. The trainer's MoE-only
    recording (`routing replay layers: 23`/`26`) is an *internal* detail with its own
    `index_by_layer` handling in `router_replay_utils.py:453` — it does not have to match the
    rollout-captured tensor's layer dimension.
  - `glm_moe_dsa` exposing `num_hidden_layers`/`num_experts_per_tok` as flat top-level config
    fields is confirmed indirectly: sglang's own capturer reads the same two fields off
    `hf_text_config` and constructed successfully this run (the rollout served and produced the
    base64 output). transformers' `glm4_moe` config has both as flat fields; `glm_moe_dsa` is a
    flat extension of it. If a future checkpoint nests them, the `hasattr` guard raises a clear
    error rather than mis-shaping.
  - **Verified locally before wiring in**: patch generated from a real edited `git worktree` at
    the v0.9.0 tag (not hand-written), applies cleanly (`git apply --check`) + compiles
    (`py_compile`) + `git apply --reverse --check` detects already-applied against a fresh
    checkout; heredoc byte-identical to the checked-in file (spliced in programmatically, not
    retyped, to avoid the trailing-whitespace-trim hazard); `bash -n` + the quoted-heredoc tab
    check pass; and a standalone Python simulation of the full decode path (base64 int32 buffer →
    `np.frombuffer` → `.reshape` → the `agent_loop.py:800-804` read-only-array unpack) round-trips
    correctly and produces `(num_tokens, num_hidden_layers, num_experts_per_tok)`. Not testable
    locally: actual `hf_config` field presence on the deployed checkpoint, and the trainer-side R3
    alignment end-to-end with this tensor.
- **Plan (user decision, 2026-08-28): (1) validate this fix on a cluster run, then (2) PR it
  upstream** — verl's `skip_tokenizer_init` routed-experts branch is dead code against any
  current sglang (v0.9.0 *and* `main` are identical), so this is a genuine upstream bug worth
  fixing there. This is the fourth consecutive R3+THD failure (3199623, 3201189, 3207151, plus
  the 2026-08-26 config-audit revert), each one layer deeper — but unlike the prior three, this
  fix is against a fully-understood mechanism with the shape/layout confirmed from sglang's own
  source, not a version-bump guess. Still needs a real 80-node run before it's trusted (per
  memory `hpc_job_fix_then_ask`, ask before resubmitting).
- **Outcome: validated — run `3207923` (2026-08-28) completed the first training steps this
  recipe has ever produced with R3 enabled.** See that run's entry below. Step (2) — the upstream
  verl PR — is the remaining open item.
- **Config-audit note**: this run *disproves* the 2026-08-26 audit entry's tentative "plausible
  but unverified" read that R3 would work with `separate_async` + TransferQueue given the
  prerequisites — the prerequisites are all now met and it still fails, at the sglang-capture API
  boundary rather than anywhere in the TQ round-trip. The TQ path was never reached.
- **Infra note**: one ~6-min clariden SSH-service outage mid-run (CSCS-side, `/status/systems`
  confirmed, self-recovered) plus a few transient FirecREST job-status timeouts — none affected
  the job. A monitor-script bug briefly overwrote the local log with a 119-byte error stub during
  one timeout; restored from backup, final log verified intact.
- **Fix**: expanded `r3-sglang-routed-experts-import-fix.patch` to rewrite the whole
  `enable_rollout_routing_replay` block (drop the dead `skip_tokenizer_init` `.numpy()` branch;
  always base64-decode + reshape). All three R3/THD flags and the megatron-core 0.19.0 +
  megatron-bridge 0.6.1 upgrade stay enabled. See the "Fix — implemented 2026-08-28" bullet
  above for the full detail and local verification. Validated on run `3207923` (next entry).
- **Commit**: not committed.

### Run `3207923` — 2026-08-28 — R3+THD fix VALIDATED (5 clean steps, first ever with R3); later FAILED on a megatron-bridge weight-sync NCCL desync unrelated to R3

- **Log**: `~/Downloads/slurm-3207923.out`. 80 nodes, verl v0.9.0, system `clariden`. ~43 min
  queued (FirecREST/Slurm-controller was intermittently timing out cluster-wide during the wait —
  scheduler overload, same transient pattern as run 3201189; job submission itself also hit one
  transient HTTP 500 "Channel not open" and succeeded on retry, verified no duplicate job).
  Started 11:27:38, RUNNING.
- **Purpose**: validate the expanded `r3-sglang-routed-experts-import-fix.patch` (the
  base64-decode fix from run `3207151`'s entry) end-to-end.
- **Result — the `'str' object has no attribute 'numpy'` bug is fixed.** Zero occurrences of that
  `AttributeError` / any `routed_experts` / `RayTaskError` / `ActorDiedError` / retry-loop
  signature anywhere in the log. Full pipeline cleared for the first time with R3 on:
  - All 5 diagnostic version prints green (`megatron-core 0.19.0`, `megatron-bridge 0.6.1`,
    `megatron.training.models.gpt: OK`, `sglang routed_experts capture: OK at
    sglang.srt.state_capturer.routed_experts`, `TransferQueue 0.1.7`).
  - All patches applied cluster-wide (PR #7421/#7422/#7423, `v1-separate-async-fixes.patch`, the
    expanded `r3-sglang-routed-experts-import-fix.patch`, megatron-bridge filelock patch).
  - DDP grad-buffer construction (`DistributedDataParallel contains 9.24B parameters` — past the
    3199623 crash point), GLM5Bridge weight conversion (6201/6201), first trainer→rollout weight
    sync, SGLang rollout serving (HTTP up 11:37:36), rollout generation, R3 router replay active
    (`routing replay layers: 26`).
- **Training steps (the success bar — never reached before with R3)**:
  - Step 1: `actor/loss` 0.0269, `actor/grad_norm` 0.420, `ppo_kl` 0.421, `critic/score` mean
    0.092 (max 1.1), peak mem 77.9 GB, step time 518s (gen 120s / update_actor 86s /
    update_weights 309s).
  - Step 2: `actor/loss` 0.0128, `actor/grad_norm` 0.318, `ppo_kl` 0.350, `critic/score` mean
    0.023, step time 344s, `trajectory_staleness` 1 (expected for separate-async).
  - Steps 3–4: `actor/loss` 0.032 / 0.054, `actor/grad_norm` 0.422 / 0.605. Close watch stopped
    at step 4/40, elapsed ~67 min — well past the failure window that killed every prior R3
    attempt; a low-noise watch stays on for the terminal state only.
  - Health: losses/grad-norms finite and stable across all 4 steps, no NaN/inf, no grad-norm
    blow-up, peak GPU mem 77.9/95 GB, step time ~350–520s (`update_weights` ~310s dominates),
    reward signal computing (near-zero, normal for early GRPO). `ppo_kl` ~0.35–0.42 is on the
    high side but not unstable this early — worth watching over a longer run.
- **Terminal state: FAILED (exit 15), elapsed ~1h40min, on the 7th trainer→rollout weight
  sync — NOT in R3 code.** 5 clean training steps + 7 weight syncs (~309s each, ~4.47 GB/s)
  completed first. A subset of trainer ranks hung in an NCCL `ALLGATHER` inside megatron-bridge's
  Megatron→HF weight streaming:
  ```
  WorkNCCL(SeqNum=50566, OpType=ALLGATHER) ran for 1800006 ms  [EXPERT_MODEL_PARALLEL_GROUP / TENSOR_MODEL_PARALLEL_GROUP]
    gather_from_ep_ranks    megatron/bridge/models/conversion/param_mapping.py:806
    megatron_to_hf          param_mapping.py:2722
    stream_weights_megatron_to_hf   model_bridge.py:1353
    send_weights            verl/checkpoint_engine/nccl_checkpoint_engine.py:246
  ```
  `last completed work: 50565, last enqueued: 50634` — a peer never posted its matching
  collective (rank-count/order mismatch in the EP/TP gather). After the 30-min NCCL timeout,
  `torch.distributed.barrier()` (`engine_workers.py:782`) failed with gloo `Connection closed by
  peer` and the job cascaded to death.
  - **Same class as run `3141801`** (30-min megatron-bridge weight-sync collective hang,
    `gather_from_ep_ranks`/`gather_from_tp_ranks` in the flight recorder) — which this repo saw
    once and could **not** reproduce on unmodified retry (`3144665`), i.e. a non-deterministic
    race in shared weight-streaming code that runs on every sync regardless of R3. 7 syncs
    succeeded here before the 8th... actually the 7th hung. Caveat: this is megatron-bridge
    **0.6.1** (new this attempt vs. 0.5.1 when `3141801` happened), and an EP-gather desync could
    be load- or version-dependent, so R3-independence is likely but not certain.
- **Not a full training run anyway**: 40 steps × ~350–520s ≈ 4–6h against a 5h `--time`, so even
  without this failure it would have TIMEOUT'd around step ~35. A real training run needs more
  `--time` / fewer steps.
- **Fix**: none applied. The fix under test — `r3-sglang-routed-experts-import-fix.patch`
  (expanded form) — **passed**: 5 clean R3 training steps, 0 `numpy`-bug occurrences, weight
  syncs working. The failure is a separate, pre-existing-class megatron-bridge weight-streaming
  race. Next move (user decision, per `hpc_job_fix_then_ask`): most likely an unmodified retry
  to check one-off vs. repeatable (matching the `3141801`→`3144665` precedent), with an eye on
  whether megatron-bridge 0.6.1 makes this EP-gather hang more systematic than 0.5.1 did.
- **Remaining open item**: step (2) of the plan — PR the `async_sglang_server.py` fix upstream to
  verl. The `skip_tokenizer_init` routed-experts branch is dead code against any current sglang
  (v0.9.0 and `main` identical), so this is a genuine upstream bug. The patch already carries the
  `ImportError` fallback to the old `sglang.srt.layers.moe.routed_experts_capturer` path, so it's
  in PR-ready shape (works against both old and new sglang layouts).
- **Commit**: not committed.

### Run `3209484` — 2026-08-28 — unmodified retry: 3207923's weight-sync hang was a one-off (12+ clean syncs); ~step 13, a DIFFERENT hang — SGLang rollout scheduler stuck in flashinfer MLA plan (the TP=32 SGLang hazard, new call site)

- **Log**: `~/Downloads/slurm-3209484.out` (1.8 MB). 80 nodes, `clariden`. Unmodified resubmit of
  `3207923` (same script, R3+THD + megatron-core 0.19.0 / megatron-bridge 0.6.1). ~48 min queued
  (FirecREST/Slurm-controller intermittently timing out during the wait again). Ran ~2h40min,
  then FAILED (exit 15).
- **Purpose**: determine whether `3207923`'s megatron-bridge `gather_from_ep_ranks` NCCL ALLGATHER
  hang (7th weight sync) was a one-off race or repeatable.
- **Answer: one-off.** All **12+** trainer→rollout weight syncs completed cleanly (~308–311s
  each, ~4.47 GB/s), well past the 7th where `3207923` hung. No `gather_from_ep_ranks` / NCCL
  ALLGATHER timeout / gloo "Connection closed by peer" anywhere. `3207923`'s hang stays a
  non-deterministic race (matching the `3141801`→`3144665` precedent).
- **R3+THD fix: validated again, harder.** ~13 clean training steps (up from `3207923`'s 5),
  loss 0.011–0.088, grad_norm 0.29–0.63, `critic/score/mean` climbing 0.05→0.24 (policy
  learning), memory steady 77.9/95 GB. **Zero** `'str' object has no attribute 'numpy'` /
  `routed_experts` / rollout `RayTaskError`. R3 active throughout (`routing replay layers`).
  `r3-sglang-routed-experts-import-fix.patch` (expanded) is solid.
- **New failure — the "TP=32 SGLang" Known-hazard, at a new call site.** At ~step 13 (~16:19),
  the standalone SGLang rollout replica's scheduler process (`sglang_server_0_0`, pid 162557)
  hung. py-spy dump (triggered by SGLang's own 300s `watchdog_timeout`, which this script sets
  explicitly) shows the scheduler stuck in:
  ```
  plan (flashinfer/mla/_core.py:839)               # a tensor .to() copy
  call_begin_forward (flashinfer_mla_backend.py:749)
  init_forward_metadata (flashinfer_mla_backend.py:381)
  _execute_decode (runner/eager_runner.py:234)
  run_batch (scheduler.py:3351) → event_loop_overlap (scheduler.py:1601)
  ```
  i.e. hung inside the flashinfer **MLA/DSA decode-attention `plan()`** step during ordinary
  generation (DSA backends: `prefill=flashmla_sparse, decode=fa3`; flashinfer 0.6.12). Watchdog
  killed the replica → Ray `SYSTEM_ERROR` ("connection error code 2") → every `AgentLoopWorkerTQ`
  rollout task → `ActorDiedError` → `TaskRunnerV1` died → job FAILED. No recovery path for a
  dead standalone replica.
- **This is the third TP=32 SGLang scheduler hang in this recipe's history, second distinct call
  site**: `3152802` (`_broadcast_reqs_across_ranks → torch.distributed.broadcast`), now `3209484`
  (`flashinfer MLA plan`). `3152802`'s retry `3171176` "cleared" it — but `3171176` only ran 3
  steps before monitoring stopped, nowhere near the step-13 mark, so that "one-off" verdict was
  on a short run. The pre-R3 baseline `3149736` did 14 clean steps (to TIMEOUT) with no hang — so
  the fragility is real but intermittent, and R3 is not obviously the cause (this hang is in the
  base DSA rollout path, unchanged by R3's capture or the megatron upgrade). See the updated
  "TP=32 SGLang" Known-hazards entry.
- **Fix**: none applied. The R3 fix passed; the failure is the pre-existing TP=32 SGLang
  collective fragility.
- **Investigation, 2026-08-28 (no cluster cost) — root-cause candidate found: a stale flashinfer
  version pin.** The stuck py-spy frames: one TP rank (TP2) ACTIVE in
  `run_batch → _execute_decode → flashinfer_mla_backend.py:381 init_forward_metadata → :749
  call_begin_forward → flashinfer/mla/_core.py:839 plan` (a `.to()` → `cuMemcpyDtoHAsync_v2` that
  never returned — flashinfer's dense-MLA `plan()`; the batch's ~350-token seqs are below the
  `index_topk=2048` DSA dense/sparse threshold, so it took the flashinfer dense-MLA path), while
  another rank (TP3) is idle in `_broadcast_reqs_across_ranks` waiting for the *next* batch — a
  single-rank stall → whole-TP-group deadlock → 300s watchdog kill. **`Alps-Images/apps/verl/Containerfile`
  pins `flashinfer_python[cu13]==0.6.12` + `flashinfer_cubin==0.6.12` (added 2026-07-27, commit
  `2d514a64`), but sglang 0.5.16's own `pyproject.toml` requires `flashinfer_python[cu13]==0.6.14`
  ("keep it aligned with jit-cache version in Dockerfile").** The `sglang[all]` line was changed
  2026-08-12 (commit `8bf10427`) and now resolves to 0.5.16, but the flashinfer pin was never
  updated to match — so the image runs sglang 0.5.16's MLA-backend code (written/tested against
  flashinfer 0.6.14's `wrapper.plan()`) against a downgraded 0.6.12. flashinfer has a documented
  history of MLA/FMHA `plan`-path hangs and IMAs fixed version-to-version (flashinfer issue #2236
  "BatchMLAPagedAttentionWrapper IMA", the 0.6.8 "revert Blackwell-ultra opt that caused FMHA
  deadlocks"). Not proven to be *the* cause (would need a run on the fixed image), but it is a
  concrete version-skew bug on exactly the code path that hung.
- **Fix — Containerfile**: `flashinfer_python[cu13]` `0.6.12` → `0.6.14` (matching sglang 0.5.16's
  pin), `flashinfer_cubin` `0.6.12` → `0.6.13` (flashinfer-cubin has no 0.6.14 published; the
  1-patch python/cubin gap just JIT-compiles a few kernels). Part of the same image rebuild as
  `VERL_REF` / megatron / TransferQueue.
- **Fix — runtime (this script)**: **flashinfer_python 0.6.14 ONLY**, via the same
  fetch-once/sbcast/`--no-deps --force-reinstall` pattern as the TransferQueue/megatron wheels
  (`flashinfer_python-0.6.14-py3-none-any.whl`, 14.6 MB, sha256 `d124369...`). The cubin is
  **deliberately not bumped at runtime**: run `3211735` (2026-08-28) tried to `sbcast` the
  ~458 MB `flashinfer_cubin-0.6.13` wheel and hit `Bus error (core dumped)` on both flashinfer
  wheels — the documented "sbcast bus-errors on large files" hazard (Apertus section, from the
  transformers-fork tarball). The failed broadcast (identical src/dst path) also truncated the
  staged wheel, so `pip install` then rejected it as invalid and the fatal guard killed all
  80 nodes at 43s. The hang is in `flashinfer/mla/_core.py plan()` — **flashinfer_python code** —
  so bumping just the small pure-Python wheel is the targeted fix; flashinfer_python 0.6.14
  JIT-compiles any kernels it can't load from the image's older 0.6.12 cubin (JIT cache already
  redirected off Lustre, plus a pre-warm step). A resubmit followed.
- **Still on the table if the flashinfer bump doesn't fix it**: (a) raise `watchdog_timeout`
  300→~1200; (b) different DSA decode backend; (c) shrink the rollout replica below TP=32;
  (d) rollout-replica auto-restart. Independent of all of this, the R3 code fix is ready to
  bake/upstream.
- **Commit**: not committed.

### Run `3211735` — 2026-08-28 — flashinfer fix attempt died in setup: sbcast bus-error on the 458 MB flashinfer_cubin wheel

- **Log**: `~/Downloads/slurm-3211735.out` (147 KB). 80 nodes, `clariden`. FAILED (exit 15) at
  **43s** — per-node setup, never reached Ray/training.
- **Symptom**: `line 723: ... Bus error (core dumped) sbcast -f .../flashinfer_python-0.6.14...`
  and `line 730: ... Bus error (core dumped) sbcast -f .../flashinfer_cubin-0.6.13...`. The
  cubin sbcast left a truncated wheel on the nodes →
  `ERROR: Wheel 'flashinfer-cubin' located at /tmp/... is invalid` →
  `FATAL: could not install flashinfer_cubin 0.6.13 wheel` → fatal guard exit 1 on tasks
  16/23/27/65/… → srun killed all 80.
- **Root cause**: the first flashinfer-fix attempt added an sbcast of the ~458 MB
  `flashinfer_cubin` wheel. `sbcast` bus-errors on wheels/tarballs that large — already
  documented in the Apertus FSDP2 section ("`sbcast` hit Bus error (core dumped) on the large
  tarball, corrupting it on several nodes"). The 4 small wheels (TransferQueue, megatron-core,
  megatron-bridge, and now flashinfer_python at 14.6 MB) sbcast fine; the cubin does not. Should
  have been caught before submitting — the hazard is in this same file's own notes.
- **Diagnostics reached first**: `TransferQueue version: 0.1.7`, `megatron-core version: 0.19.0`,
  `megatron-bridge version: 0.6.1` all correct. Nothing downstream.
- **Fix (attempt 1)**: dropped the cubin from the runtime script — flashinfer_python 0.6.14 alone
  (14.6 MB), still via sbcast. **Did not work** — see run `3213313`.
- **Commit**: not committed.

### Run `3213313` — 2026-08-28 — flashinfer_python-only sbcast ALSO bus-errored: the flashinfer `sbcast` call itself is the problem, not wheel size

- **Log**: `~/Downloads/slurm-3213313.out` (142 KB). 80 nodes, `clariden`. FAILED (exit 15) at
  **40s** — setup only.
- **Symptom**: `slurm_script: line 725: 196596 Bus error (core dumped) sbcast -f
  ".../flashinfer_python-0.6.14-py3-none-any.whl" ...` — on the **14.6 MB** python wheel this
  time, whose `sha256sum -c` printed `OK` on the line immediately above (`/tmp/flashinfer_python-0.6.14-py3-none-any.whl: OK`).
  The 3 preceding wheel sbcasts (`transferqueue`, `megatron_core`, `megatron_bridge`) all
  succeeded and their `pip install`s ran fine later in the log. Corrupt wheel on the nodes →
  `ERROR: Wheel 'flashinfer-python' ... is invalid` → `FATAL` → all 80 killed.
- **Root cause — narrowed**: it is **not** a wheel-size issue (14.6 MB failed; 458 MB failed in
  `3211735`). The flashinfer `sbcast` call specifically fails while the earlier 3 wheel sbcasts
  in the same block succeed — so it is either an Nth-consecutive-`sbcast` limit or batch-host
  `/tmp` pressure by the time the ~8th–10th `sbcast` of the run (PR patches + source patches +
  3 wheels + flashinfer) is reached. Not root-caused precisely; a `Bus error` in the `sbcast`
  process on the submit host points at an mmap fault on the source file (batch-host `/tmp`),
  despite the full-content sha256 passing.
- **Fix (attempt 2)**: **stage the flashinfer wheel on shared Lustre (`${TRAINING_HOME}/wheels/`),
  no `sbcast` at all.** Every node `pip install`s it directly from that path — a plain static-file
  read, which has none of the flock/write hazards that make Lustre unsafe for locks/caches. Same
  pattern the Apertus Megatron script uses for its own wheels. `${TRAINING_HOME}` (`/capstor/...`)
  is mounted in the container per `env.toml`. **Worked — run `3214410` cleared setup.**
- **Commit**: not committed.

### Run `3214410` — 2026-08-29 — setup finally cleared (Lustre wheel), but flashinfer 0.6.14 hard-raises on the cubin version mismatch at import

- **Log**: `~/Downloads/slurm-3214410.out` (15,660 lines). 80 nodes, `clariden`. Ran overnight,
  FAILED (`srun: Force Terminated`) — got much further than `3211735`/`3213313` but still never
  reached a training step.
- **Setup cleared**: the Lustre-staged flashinfer wheel installed with no `Bus error` — `sbcast`
  fix confirmed. `flashinfer-python version: 0.6.14` printed on all ranks.
- **Symptom**: on every worker, at flashinfer import time:
  `RuntimeError: flashinfer-cubin version (0.6.12) does not match flashinfer version (0.6.14).
  Please install the same version of both packages. Set FLASHINFER_DISABLE_VERSION_CHECK=1 to
  bypass this check.` Traced through `TaskRunnerV1.run()` → `run_ppo` → (flashinfer import). All
  80 nodes died on it; job force-terminated.
- **Root cause**: flashinfer 0.6.14 added a **hard** python-vs-cubin version gate at import — it
  does NOT silently JIT-fallback on a mismatched cubin package (my earlier assumption was wrong).
  The image has `flashinfer_cubin 0.6.12` (from the old Containerfile pin); bumping only
  `flashinfer_python` to 0.6.14 trips the gate.
- **Fix (attempt 3, wrong) — then corrected**: first reached for
  `export FLASHINFER_DISABLE_VERSION_CHECK=1` (job `3217384`, cancelled before it ran) because I
  concluded no `flashinfer_cubin 0.6.14` existed — **only true on PyPI**. `flashinfer-cubin` is
  distributed via `https://flashinfer.ai/whl` (GitHub release assets), where `0.6.14` and up all
  exist; see the new **"flashinfer packaging"** section above (the KEY takeaway from this whole
  sub-chain). **Corrected fix (attempt 4)**: install the matched `flashinfer_{python,cubin}==0.6.14`
  pair — script fetches both wheels (python 14.6 MB sha256 `d124369…`, cubin 458 MB sha256
  `7bbed9f3…`) from the GitHub `v0.6.14` release into `${TRAINING_HOME}/wheels/` (Lustre, not
  `sbcast`) and `pip install`s them from there; Containerfile uses
  `pip install --extra-index-url https://flashinfer.ai/whl "flashinfer_cubin==0.6.14"
  "flashinfer_python[cu13]==0.6.14"`. `FLASHINFER_DISABLE_VERSION_CHECK` removed from both — the
  matched pair passes the check naturally, clean AOT, no JIT-contention risk.
- **Commit**: not committed.

### Job `3217384` — 2026-08-29 — cancelled while PENDING (superseded by the matched-pair fix)

- Submitted with the `FLASHINFER_DISABLE_VERSION_CHECK=1` bypass approach; user opted for the
  cleaner matched-0.6.14-pair fix (after the `flashinfer.ai/whl` discovery) before it allocated.
  `DELETE /compute/clariden/jobs/3217384` → HTTP 204, `CANCELLED`, never ran, no allocation
  consumed, no log. Resubmitted as the next job with the corrected script.

### Run `3217439` — 2026-08-29 — flashinfer 0.6.14 FIXES the TP=32 MLA hang: 20 clean R3+THD steps, healthy learning; new failure is a marginal CUDA OOM in the Megatron optimizer at step 21

- **Log**: `~/Downloads/slurm-3217439.out` (938 KB). 80 nodes, `clariden`. Queued ~44 min, ran
  **~2h43m** (elapsed 9787s), FAILED (exit 15) at step 21.
- **THE KEY RESULT — the whole R3+THD chain is now working end-to-end through 20 training steps.**
  - **Setup fully clean**: matched `flashinfer-python 0.6.14` **+ `flashinfer-cubin 0.6.14`**
    (Lustre-staged from `flashinfer.ai/whl`, no `sbcast`), no version-check `RuntimeError`,
    `megatron-core 0.19.0` / `megatron-bridge 0.6.1` / `megatron.training.models.gpt: OK` /
    `routed_experts capture: OK` / `TransferQueue 0.1.7`. No `Bus error` / sbcast / wheel / FATAL.
  - **The TP=32 SGLang MLA/DSA `plan()` hang did NOT recur.** Cleared step 13–15 (where `3209484`
    hung) and ran to step 20 with zero hang signatures anywhere — no SGLang scheduler watchdog,
    no py-spy dump, no `flashinfer/mla/_core.py`, no `_broadcast_reqs_across_ranks`, no
    `SYSTEM_ERROR`, no `ActorDiedError`. **Confirms the stale flashinfer 0.6.12 pin was the
    cause.** (This resolves the "TP=32 SGLang" Known-hazards entry for this recipe — see it.)
  - **R3+THD training is healthy**: 20 completed steps, `actor/loss` ~0.02–0.06, `actor/grad_norm`
    0.27–0.56 (all finite, stable), `critic/score/mean` rising 0.048 → ~0.15 by step 12–14 (real
    GSM8K learning), response length drifting down 255 → 244. Weight syncs completed cleanly
    throughout (last: 316s, 4.38 GB/s).
- **New failure — CUDA OOM in the Megatron distributed optimizer, step 21, single rank
  (`pid 294017`, GPU 0 — the GLM5Bridge / checkpoint-engine coordinator rank).** Traceback:
  `actor_rollout_update_actor → train_mini_batch → distrib_optimizer.py:2789
  _copy_model_grads_to_main_grads → shard_main_param.grad = shard_model_grad.float()`.
  `torch.OutOfMemoryError: Tried to allocate 24.00 MiB. GPU 0 ... 21.38 MiB free ... this
  process has 85.29 GiB in use (68.29 GiB allocated by PyTorch, 131.81 MiB reserved-but-unallocated).`
  - **Analysis**: per-step `max_memory_allocated_gb` was **dead flat at 77.92 GiB across all 20
    steps** — torch's own high-water mark is completely stable, no leak on the PyTorch side. At
    OOM only 68.29 GiB was torch-allocated and just 132 MiB was reserved-but-unallocated (so
    *not* primarily fragmentation, despite the error's generic `expandable_segments` hint). The
    tip-over is the **~17 GiB of non-PyTorch memory** on GPU 0 (NCCL comm buffers + checkpoint-engine
    weight-sync buffers + CUDA context) — plausibly creeping up over the ~10 weight syncs, on the
    coordinator rank specifically, until a routine 24 MiB optimizer allocation had nowhere to go.
    Razor-thin (24 MiB short at step 21 of 40).
  - **Distinct from every prior failure in this chain** (hang, sbcast, version check,
    megatron-bridge `gather_from_ep_ranks`, offsets, Gloo). First OOM this recipe has hit at a
    real training step (the earlier OOMs were all at init / hybrid-rollout).
- **Fix (attempt 1 — one half failed)**: (a) `export
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in the srun env block — **reverted, does not
  work here** (run `3219305`, next entry): the srun body is shared by the SGLang standalone-rollout
  processes, and SGLang's `torch_memory_saver` (called unconditionally by
  `load_model_with_memory_saver`, even with `free_cache_engine: false`) hard-raises
  `TorchMemorySaver is disabled ... expandable_segments is not supported yet` on every rollout TP
  rank. Removed; a `# do NOT set this` note left in its place. (b) `actor.ppo_max_token_len_per_gpu`
  16384 → 12288 in `grpo_gsm8k.yaml` — **kept**, this is the real lever (cuts the actor fwd/bwd
  activation peak; the model already has full param/grad/optimizer offload + full recompute).
- **Commit**: not committed.

### Run `3219305` — 2026-08-29 — OOM-mitigation attempt 1: `expandable_segments` global export is incompatible with SGLang, dies in rollout bring-up

- **Log**: `~/Downloads/slurm-3219305.out` (638 KB). 80 nodes, `clariden`. Queued ~58 min, ran
  ~24 min, FAILED (exit 15) — **before the first training step**.
- **Trainer side got further than ever**: GLM5Bridge conversion 100% (6201/6201),
  `DistributedDataParallel contains 9.24B parameters`, **no OOM at the DDP grad-buffer**. Setup
  fully clean (flashinfer 0.6.14 py+cubin, megatron 0.19.0/0.6.1, gpt module OK, routed_experts
  OK, TransferQueue 0.1.7) — no regression on any previously-fixed issue.
- **Symptom**: all 32 SGLang standalone-rollout TP ranks, in `Scheduler.__init__ →
  init_tp_model_worker → ModelRunner.load_model → load_model_with_memory_saver`:
  `RuntimeError: TorchMemorySaver is disabled for the current process because expandable_segments
  is not supported yet` (`torch_memory_saver/entrypoint.py:140 _sanity_checks`). Every rank →
  replica down → `ActorDiedError` → Ray `SYSTEM_ERROR` → all 80 tasks killed.
- **Root cause**: `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (OOM mitigation #1)
  was a global env var in the shared srun body, so it also applied to the SGLang rollout
  processes. `torch_memory_saver` and `expandable_segments` are mutually exclusive allocator
  mechanisms; SGLang invokes the former unconditionally.
- **Fix**: dropped mitigation #1 entirely (no clean way to scope an allocator env var to only the
  Megatron trainer workers without fragile per-worker-env plumbing). Mitigation #2
  (`ppo_max_token_len_per_gpu` 12288) stays and is now the sole step-21 OOM mitigation — untested,
  since `3219305` never reached a training step. Resubmit needed.
- **Commit**: not committed.

### Run `3219811` — 2026-08-29 — 9 clean steps (best learning signal yet), then the megatron-bridge weight-sync hang AGAIN (3rd occurrence) — did not reach the step-21 OOM test

- **Log**: `~/Downloads/slurm-3219811.out` (9529 lines). 80 nodes, `clariden`. Queued ~42 min,
  ran ~2h, FAILED (exit 15) at weight-sync #5 (feeding `global_step` 10).
- **Everything through step 9 worked**: setup all green (flashinfer 0.6.14 py+cubin, megatron
  0.19.0/0.6.1, gpt module OK, routed_experts OK, TransferQueue 0.1.7 — `PYTORCH_CUDA_ALLOC_CONF`
  removed, no `TorchMemorySaver` error this time), first weight sync + SGLang rollout serving + 4
  more clean weight syncs (~328s each). **9 completed training steps, and the model is clearly
  learning**: `critic/score/mean` 0.10 → 0.24 (step 6) → **0.32 (step 8)** → 0.19 (step 9) — the
  strongest GSM8K learning signal across all runs. loss 0.009–0.089, grad_norm 0.27–0.59, all
  finite. **No OOM anywhere.**
- **Symptom**: `Watchdog caught collective operation timeout: WorkNCCL(... OpType=ALLGATHER ...)
  ran for 1800099 ms` in
  `send_weights (nccl_checkpoint_engine.py:246) → stream_weights_megatron_to_hf
  (model_bridge.py:1353) → megatron_to_hf → gather_from_ep_ranks (param_mapping.py:806)`. Rank 59
  never entered the EP/TP/PP gather; peers waited the full 30-min NCCL timeout; Slurm killed all
  80.
- **This is the 3rd occurrence of the megatron-bridge weight-sync collective hang** — after
  `3141801` (at the very first sync, megatron-bridge 0.5.1, pre-R3) and `3207923` (7th sync,
  0.6.1). It did NOT recur in `3144665`, `3209484` (12+ syncs) or `3217439` (~10 syncs, 20 steps).
  So it is a **real, recurring, non-deterministic bug in this recipe's weight-streaming path** —
  roughly 1 in 2 runs, at a random sync — spanning megatron-bridge 0.5.1 and 0.6.1, present
  before R3 was enabled. **Not** caused by the `ppo_max_token_len_per_gpu` cut (that only touches
  actor activation memory, not weight-sync collectives). It is now the single biggest blocker to
  a full run — bigger than the step-21 OOM (which has only ever been reached once).
- **The step-21 OOM question is still unanswered** (never reached it). `max_memory_allocated_gb`
  stayed pinned at 77.92 GiB steps 1–9, identical to `3217439` — the token-budget cut is
  confirmed not to move that metric.
- **Investigation, 2026-08-29 (no cluster cost)** — what the `3219811` log + upstream source
  show about the weight-sync hang:
  - **Timeline**: ~11 weight syncs completed cleanly (~325s each, one roughly every step —
    `parameter_sync_step: 2` but syncs land ~every step in the log; each streams the FULL 700B
    model, `total_params: 59079` tensors, `Converting to HuggingFace 6201/6201`, ~1.4 TB at
    ~4.25 GB/s). The 12th hung: a rank in EP-group GUID 1404 (one of 288/8 = 36 expert-parallel
    groups) never posted allgather `SeqNum=79314` in `param_mapping.py:806 gather_from_ep_ranks`.
    `last enqueued 79382, last completed 79313` — that rank had raced *ahead* enqueuing 69 more
    async collectives, then one earlier one never matched. Every EP/TP/PP group that rank touches
    then cascaded (`gather_from_ep_ranks` ALLGATHER, `gather_from_tp_ranks` ALLGATHER,
    `broadcast_from_pp_rank` BROADCAST all time out). **Rank 0 (the CE sender) completed its own
    view of that sync at 18:04** while other ranks were already stuck since ~17:59 — because rank
    0 is in a *different* EP group; the stuck group's 8 members are all CE non-senders running a
    tight `for name,weight in weights: pass` over the same bridge generator.
  - **No preceding exception on any rank.** NCCL's own diagnostic: "wrong sizes / order not same
    / the scheduled collective didn't run ... GIL deadlock ... network errors ... bugs in NCCL".
    The bridge task list is built lockstep (`sorted_global_param_names_all_pp_ranks`, all-gathered,
    `None`-padded for non-owning PP ranks), so a *structural* divergence is unlikely — this reads
    as a **transient**: one rank momentarily stalled (GIL / a slow CPU copy / tqdm render / verl's
    async loop) or a Slingshot/NCCL message glitch during the ~5-min, ~6000-back-to-back-collective
    streaming window, and verl has **no retry** on a failed weight sync — a stuck collective =
    full 30-min NCCL timeout = job death.
  - **Not a version regression** (megatron-bridge 0.5.1 *and* 0.6.1), **not R3/THD-specific**
    (shared code, on every sync, `3141801` predates R3). Upstream: verl issue #6691 confirms the
    Megatron→HF→SGLang MoE weight-sync path is problematic and under active work; issue #3704
    recommends `NCCL_TIMEOUT` for *slightly-exceeded* timeouts — doesn't apply here (ours runs the
    full 30 min = true deadlock).
  - **No clean config-only fix found.** A real fix = patch verl's `send_weights`/`update_weights`
    to catch a sync-PG timeout and retry the sync (needs a short dedicated timeout on the sync PG
    + re-init) — real work, not attempted.
- **Deeper investigation, 2026-08-29 (user asked for online search + a verl retry/debug fix):**
  - **Online: this is a known, unresolved upstream class of bug.** verl #2197 (NCCL timeout,
    Megatron TP>1) — closed by an inactivity bot, never fixed, multiple "same problem" replies.
    verl #2325 (Megatron+SGLang NCCL timeout, "some process getting stuck") — open, 15 comments,
    no fix. verl #3704 (Megatron NCCL broadcast timeout) — open; maintainer asks for exactly the
    per-PG debug info we lack. verl #6691 (MoE megatron-bridge weight-sync OOM) — open, active
    work on the MoE sync path. **No upstream fix for "one rank silently stops posting a collective
    during weight sync."**
  - **An in-process retry of THIS hang is not really feasible.** It is stuck in *Megatron's own*
    EP/TP/PP process groups (not verl's 33-rank checkpoint-engine group), and torch has no
    "retry this collective" primitive — recovering a hung NCCL collective there needs a full
    `destroy_process_group` + `parallel_state.initialize_model_parallel(...)` re-init, i.e. ≈ a
    job restart. The realistic "retry" is job-level (monitor detects the hang → requeue).
  - **The actual fix verl provides: `checkpoint_engine.backend: delta_sharded`
    (`DeltaShardedCheckpointEngine`).** verl's newer disaggregated-async weight-sync engine,
    built specifically to avoid this fragility. Read of its source (`delta_checkpoint_engine.py`,
    `workers/engine/megatron/transformer_impl.py:1015-1110`): steady-state syncs export **each
    rank's LOCAL mcore shard** (`get_per_tensor_param_shard` — `yield rec.megatron_name,
    local.reshape(-1), rec.spec`, **no cross-rank gather in the export**; params owned by another
    PP stage yield "zero-count lockstep rows"), run the bridge param-mapping logic **comm-stubbed**
    (`_hf_delta_entry`: "real group sizes, gathers synthesized locally"), and ship only the
    **changed `(position, value)` pairs** via `_GatherQueue` with **count-only flush triggers**
    (their own comment: "a per-rank byte trigger desyncs the gathers" — so they made it count-based
    on purpose). This replaces the ~6000-back-to-back-`gather_from_ep_ranks`-collectives storm
    with a small, count-lockstepped sparse gather. Only the **one-time seed sync (#1)** still uses
    the old full `get_per_tensor_param()` streaming path — our hangs were always at sync 5/7/12,
    never #1, so if the seed passes every later sync is on the robust path. **Bonus: much faster**
    (ships deltas, not the full ~1.4 TB every 2 steps).
    - **Config cost is likely one line**: `checkpoint_engine.backend: nccl → delta_sharded`. verl
      auto-appends the SGLang `delta_loader` when the backend is `delta_sharded`
      (`async_sglang_server.py:263-269` — no SGLang fork/patch), and `engine_workers.py:751`
      routes the delta engine's own seed/steady state machine + snapshot priming.
    - **Requirements we already meet**: `vanilla_mbridge: False` (asserted), no LoRA (asserted).
    - **Risks**: not in verl's README "Supported Backends" table (newer / less battle-tested) —
      could carry its own bugs; the seed sync still exercises the fragile path once.
- **Implemented, 2026-08-29 (user-approved), unverified on cluster:**
  1. **`checkpoint_engine.backend: nccl → delta_sharded`** (`grpo_gsm8k.yaml`;
     `engine_kwargs.nccl → engine_kwargs.delta_sharded`). `bucket_size` comes from
     `update_weights_bucket_megabytes` not `engine_kwargs`, so no other config needed; verl
     auto-wires the SGLang `delta_loader`.
  2. **`example/patches/wsync-debug-progress-log.patch`** (new checked-in patch + script heredoc,
     verified byte-identical + applies clean + compiles against fresh v0.9.0) — wraps both weight-
     export generators in `MegatronEngine` (`get_per_tensor_param` = seed; `get_per_tensor_param_delta_shard`
     = steady) with a pass-through that prints `[WSYNC-DBG] rank=N <tag> tensor#K name=… t+Ns`
     every 1000 tensors and on generator exit. Applied via the same `git apply` apply-or-fail
     loop as the other verl-source patches.
  3. **NCCL flight-recorder env** in the srun block: `TORCH_NCCL_TRACE_BUFFER_SIZE=20000`
     (the missing piece — run `3219811`'s dump showed `last enqueued NCCL work: -1`, i.e. the
     ring buffer was OFF), `TORCH_NCCL_DUMP_ON_TIMEOUT=1`,
     `TORCH_NCCL_DEBUG_INFO_TEMP_FILE=/tmp/nccl_flightrecorder_<jobid>_rank`.
  4. **Node count / mini-batch: left at 80 / `PPO_MINI_BATCH_SIZE=6`**, NOT bumped to 104 / 8 as
     the pre-approved plan said — because `delta_sharded` removes the full-model HF-export buffers
     that rank 0 staged every sync on the `nccl` backend (`prepare()` even skips the parent's
     `2 × bucket_size` fixed buffers), which is very plausibly the bulk of the ~17 GiB non-torch
     memory that caused `3217439`'s step-21 OOM. Testing `delta_sharded` at 80 nodes first tells
     us whether the OOM is also gone; 104 nodes + `PPO_MINI_BATCH_SIZE=8` (DP 3→4) is the
     immediate fallback if the step-21 OOM recurs. `PARAMETER_SYNC_STEP 2→4` is a further fallback.
  `ppo_max_token_len_per_gpu` stays at 12288 (harmless).
- **Commit**: not committed.

### Run `3223205` — 2026-08-29/30 — `delta_sharded` FIXES the weight-sync hang (trainer side: zero `gather_from_ep_ranks`, seed export in ~3s), then one more verl-vs-sglang import drift on the rollout side

- **Log**: `~/Downloads/slurm-3223205.out` (8694 lines). 80 nodes, `clariden`. ~15h queued, ran
  ~24 min, FAILED (exit 15) — **during the first (seed) weight sync**, before any training step.
- **The `delta_sharded` trainer side works — and it kills the hang.** The `[WSYNC-DBG]` patch fired
  on all 383 workers: `seed/full tensor#2001 ... t+3s` — the full-model HF export that took ~326s
  and hung ~1 run in 2 on the `nccl` backend completed in **~3 seconds** with **zero
  `gather_from_ep_ranks`, zero NCCL ALLGATHER, zero watchdog, zero SYSTEM_ERROR**. Setup all green
  (`Applied wsync-debug-progress-log.patch` 80/80, flashinfer 0.6.14 py+cubin, megatron
  0.19.0/0.6.1, gpt OK, routed_experts OK, TransferQueue 0.1.7), DDP grad-buffer no OOM, SGLang
  rollout HTTP server up.
- **Symptom** — the seed sync's *rollout-side* delta ingestion:
  ```
  verl/workers/rollout/sglang_rollout/sglang_rollout.py:404  _update_weights_delta_flush
    from sglang.srt.model_executor.model_runner import LocalSerializedTensor
  ImportError: cannot import name 'LocalSerializedTensor' from 'sglang.srt.model_executor.model_runner'
  ```
  (`on_init_end` → `standalone_checkpoint_manager.update_weights` → `checkpoint_engine/base.py:337`
  → `sglang_rollout.py:317 update_weights` → `_update_weights_delta` → `_update_weights_delta_flush`.)
- **Root cause**: exactly the same class as `r3-sglang-routed-experts-import-fix.patch` — verl
  v0.9.0's `delta_sharded` rollout path hardcodes `from sglang.srt.model_executor.model_runner
  import LocalSerializedTensor`, but sglang 0.5.16 moved that class to
  `sglang.srt.model_executor.model_runner_components.weight_updater` (confirmed: sglang v0.5.16's
  own `weight_sync/utils.py:10-12` imports it from the new path, class def at
  `model_runner_components/weight_updater.py:370`). The `nccl` backend (runs `3217439` etc.)
  never hit this because its rollout-side apply goes through sglang's *own*
  `weight_sync.utils.update_weights`, which has the correct import; verl's `delta` path inlines a
  stale copy. The other 3 sglang imports in that verl function were checked and are still valid.
- **Fix**: new checked-in patch `example/patches/delta-sharded-localserializedtensor-import-fix.patch`
  (+ script heredoc/sbcast/apply, verified byte-identical, applies clean + compiles + reverse-checks
  against fresh v0.9.0) — one-line: try the new path, `except ImportError` fall back to the old.
  Wired into the same apply-or-fail loop as the other verl-source patches, right after
  `wsync-debug-progress-log.patch`.
- **Still unanswered** (never reached): the step-21 OOM (no `OutOfMemoryError` anywhere in this
  log), the delta *steady* path, and whether steps are faster.
- **Commit**: not committed.

### Runs `3234169` / `3235127` — 2026-08-30 — the CI-built image is broken (transformer-engine ↔ torch 2.11 ABI); the new consolidated diagnostic caught it in 60 s

- **Context**: user built the image from the current Containerfile (VERL_REF v0.9.0 + TransferQueue
  0.1.7 + megatron 0.19.0/0.6.1 + flashinfer 0.6.14) and pointed the GLM script at it, and asked
  to strip the now-redundant runtime pip upgrades (done — see the Image bake-in section).
- **`3234169`** — FAILED at 43 s: pyxis 404, wrong image path. Corrected to
  `.../alps-images/verl-cuda:alps7-dev-a9f9e56471c0574e` (CI appends the `-cuda` variant suffix).
- **`3235127`** — FAILED at ~4.6 min, phase 1 partial. Container pulled fine.
  - **The new consolidated image-diagnostic block worked exactly as designed** — printed
    `verl 0.9.0 · TransferQueue 0.1.7 · megatron-core 0.19.0 · megatron-bridge 0.6.1 ·
    flashinfer-python 0.6.14 · flashinfer-cubin 0.6.14 · sglang 0.5.18 · transformers 5.8.1` and
    ran the 3 import smoke tests, in the first ~60 s. **sglang resolved to 0.5.18, not 0.5.16**
    (unpinned build — the Containerfile `sglang[all]==0.5.16` pin was uncommitted at build time).
    Both sglang-import source patches still import OK on 0.5.18 (`state_capturer.routed_experts`
    and `model_runner_components.weight_updater:LocalSerializedTensor` both present) — so the
    sglang drift was NOT the failure.
  - **Root cause**: `transformer_engine_torch.cpython-312-aarch64-linux-gnu.so: undefined symbol:
    _ZN3c104impl3cow23materialize_cow_storageERNS_11StorageImplE` (`c10::impl::cow::materialize_cow_storage`).
    The NGC base image (`pytorch-cuda:26.02-py3`) ships transformer-engine built against its own
    torch; `sglang[all]` then upgrades torch to its pinned **2.11.0** (PyPI wheel), and nothing
    rebuilds TE against it. `import megatron.core` fails on all 80 nodes → verl's `EngineRegistry`
    has no `megatron` backend → fatal `AssertionError: Unknown backend: megatron` at
    `engine_workers.py:137`. torchao `_C*.so` also fails to load (same ABI break; secondary).
    This never bit the runs on the *old* image because the removed runtime upgrades used
    `pip install --no-deps --force-reinstall` — `--no-deps` never disturbed the base image's
    (then-matched) TE.
  - All 5 source patches applied cleanly 80/80. `delta_sharded` still untested.
- **Fix (Containerfile, 2026-08-30, needs rebuild)**: two changes.
  1. **`sglang[all]` → `sglang[all]==0.5.16`** (line ~37) — the only recent sglang whose own
     flashinfer pin (`0.6.14`) matches the Containerfile's; see the Image bake-in bullet.
  2. **A final `transformer-engine[core-cu13,pytorch]` `--force-reinstall --no-build-isolation`
     step** placed *after every torch-moving install* (after flashinfer), so TE's torch bindings
     (`transformer-engine-torch`, sdist-only → source build, ~20-40 min) are compiled against the
     final torch 2.11.0. Committed `72f6472`.
- **Build attempt 1 (commit `72f6472`) FAILED at the verification step**: the TE source rebuild
  itself succeeded (STEP 36), but the build-gate `RUN python -c "import ... transformer_engine.pytorch,
  megatron.core"` died with `OSError: libcuda.so.1: cannot open shared object file` — `import
  transformer_engine` dlopens the CUDA *driver*, which is host-provided at container *runtime* and
  absent during `docker build`. My gate was too strict.
- **Fix (commit `8dc2a9b`)**: build-safe gate — `import torch` (fine at build) + assert
  `transformer-engine` and `transformer-engine-torch` are installed at the **same** version (what
  `--force-reinstall` can get wrong); NO `import transformer_engine` / `megatron.core` at build.
  The real runtime import is already checked by the GLM script's own diagnostic block on the GPU
  nodes (which is exactly what caught `3235127`).
  - **Residual risk (REALIZED in run `3236353`, now fixed)**: `transformer-engine[core-cu13,pytorch]`
     was unpinned → resolved to 2.18.0, whose cutlass flash-attn backend imports `block_copy` from
     a newer `nvidia-cutlass-dsl` than the NGC base ships. See run `3236353` below.
- **Commit**: Containerfile committed (`72f6472`, `8dc2a9b`); script + patches not committed.

### Run `3236353` — 2026-08-30 — first run on the CI image: TE unpinned → 2.18.0 → cutlass `block_copy` ImportError → `Unknown backend: megatron` (same shape as 3235127); TE pinned to 2.12.0

- **Log**: `~/Downloads/slurm-3236353.out` (16,584 lines). 80 nodes, `clariden`,
  image `verl-cuda:alps7-dev-b9c322b732cca289` (built from Containerfile `8dc2a9b`). ~34 min
  queued, ran ~3m52s, FAILED (exit 15) at the **start of phase 3** — `actor_rollout_init_model()`,
  before DDP grad-buffer construction. 0 training steps.
- **What checked out**: all baked-in versions correct (verl 0.9.0, TransferQueue 0.1.7,
  megatron-core 0.19.0, megatron-bridge 0.6.1, flashinfer-python/cubin 0.6.14, sglang 0.5.16,
  transformers 5.8.1). **Both sglang import-drift patches validated by the diagnostic smoke tests**
  — `sglang routed_experts capture: OK` and `sglang LocalSerializedTensor: OK at
  model_runner_components.weight_updater` (the latter is run `3223205`'s blocker — smoke test
  passes, though the actual delta weight-sync path was never reached). All 6 source patches
  applied cleanly on 80/80.
- **Symptom**: the consolidated diagnostic block's third check printed, on all 80 nodes:
  `WARNING megatron.training.models.gpt NOT importable` — traceback:
  ```
  transformer_engine/pytorch/attention/dot_product_attention/backends.py:169
      from cutlass.utils import LayoutEnum, block_copy
  ImportError: cannot import name 'block_copy' from 'cutlass.utils'
      (.../nvidia_cutlass_dsl/python_packages/cutlass/utils/__init__.py)
  ```
  Then the fatal crash: `import megatron.core` fails through
  `megatron.core.extensions.transformer_engine` → verl never registers the `megatron` backend →
  `AssertionError: Unknown backend: megatron` (`verl/workers/engine/base.py:399`, via
  `engine_workers.py:137`). Identical *shape* to run `3235127`.
- **Root cause**: the `8dc2a9b` Containerfile change added a `--force-reinstall` of
  `transformer-engine[core-cu13,pytorch]` **unpinned** — it resolved to **2.18.0**. The NGC base
  `pytorch-cuda:26.02-py3` bundles **TE 2.12.0** matched to its own `nvidia-cutlass-dsl` /
  `nvidia-cudnn-frontend` / flash-attn. TE 2.18.0's cutlass flash-attn backend does
  `from cutlass.utils import block_copy` — a symbol the base image's older cutlass-dsl lacks. This
  is a **known, documented TE issue** (TE 2.18 release notes: *"known compatibility issue between
  the FlashAttention v4, CuTeDSL and CUDNN Frontend pip packages, which could produce ...
  `ImportError: cannot import name 'block_copy' from 'cutlass.utils'`"*; the documented stable
  combo is `flash-attn-4==4.0.0b11` + `nvidia-cutlass-dsl[cu13]==4.4.2` + `nvidia-cudnn-frontend==1.26.0`).
  The `8dc2a9b` gate only asserted `transformer-engine == transformer-engine-torch` version — it
  did not pin the version or check cutlass-dsl.
- **Fix (Containerfile, needs rebuild)**: pin TE to **2.12.0** (the NGC 26.02 base's own version)
  and name all three packages explicitly with `--no-deps` so nothing else moves:
  ```
  RUN pip install -c /tmp/torch_constraints.txt --no-cache-dir --no-build-isolation --force-reinstall --no-deps \
          "transformer-engine==2.12.0" "transformer_engine_cu13==2.12.0" "transformer_engine_torch==2.12.0"
  ```
  Only `transformer_engine_torch` (sdist) rebuilds against the final stable torch 2.11.0 —
  `transformer_engine_cu13` (prebuilt wheel) is re-laid at 2.12.0, and `nvidia-cutlass-dsl` /
  `nvidia-cudnn-frontend` / flash-attn stay exactly as the base image ships them (matched to TE
  2.12.0). TE 2.12.0's Python code does NOT import `block_copy`. Build gate updated to assert
  `te == tt == '2.12.0'`. Also added `transformer-engine`, `transformer-engine-torch`,
  `nvidia-cutlass-dsl` to the GLM script's runtime diagnostic package list, and expanded the
  `megatron.training.models.gpt` WARNING text to name this failure mode.
- **What this run was meant to test — all still untested**: `delta_sharded` steady-state syncs
  (fast? no `gather_from_ep_ranks` hang?), the `LocalSerializedTensor` runtime path (smoke test
  only), the step-21 OOM, the TP=32 SGLang MLA hang. None reached.
- **Commit**: Containerfile fix committed (`e3f4d98`); script diagnostic tweaks not committed.

### Run `3240384` — 2026-08-31 — TE 2.12.0 pin WORKS (image fixed); furthest run yet; died mid-seed-sync on a hardware node failure — infra, not code

- **Log**: `~/Downloads/slurm-3240384.out`. 80 nodes, `clariden`,
  image `verl-cuda:alps7-dev-621fa40275c4f036` (Containerfile `e3f4d98`, TE pinned to 2.12.0).
  ~18 min queued, ran ~41 min, FAILED — **a dead compute node, not a code/config bug**.
- **Cause**: compute node `172.28.27.124` stopped responding mid-run — Ray raylet
  `"marked dead because the detector has missed too many heartbeats"`; GCS then went
  cluster-wide-unavailable and Ray tore down. Driver surfaced it as
  `trainer_separate_async.py:135 on_init_end → standalone_checkpoint_manager.update_weights →
  checkpoint_engine/base.py:515 → ray.get(...) → ActorDiedError` (that node's WorkerDict died
  with the node). Nothing in the script or config is implicated.
- **The TE 2.12.0 pin (`e3f4d98`) is validated — the CI image works.** Diagnostic block, all 80
  nodes: `transformer-engine: 2.12.0` / `transformer-engine-torch: 2.12.0` /
  `nvidia-cutlass-dsl: 4.5.0` (TE 2.12's code doesn't reference `block_copy`, so the newer
  cutlass-dsl is fine), and all 3 import smoke tests green — `megatron.training.models.gpt: OK`
  (the exact import that killed `3235127` and `3236353`), `sglang routed_experts capture: OK`,
  `sglang LocalSerializedTensor: OK`. 6 source patches applied 80/80, no FATAL.
- **Furthest any run on a CI-built image has reached** — cleared every historical crash point
  before the node died:
  - Phase 2: `DistributedDataParallel contains 9.24B parameters` (past `3199623`'s CUDA crash).
  - Phase 3: GLM5Bridge HF→Megatron conversion `100% (6201/6201)`.
  - Phase 4: SGLang TP=32 standalone rollout `HTTP server started`.
  - **Seed weight sync (`delta_sharded`), trainer-side export: ran clean** — `[WSYNC-DBG]
    seed/full generator exited after 59079 tensors, 78–80s` on the ranks that finished (others
    were still streaming ~tensor #38001 at t+91s when the node dropped). **Zero
    `gather_from_ep_ranks`, zero NCCL ALLGATHER** — the trainer side of the weight-sync-hang
    concern looks good, but this was the one-time seed path, not a steady-state delta sync.
- **Still not validated end-to-end** (job died before reaching them): `delta_sharded` rollout-side
  ingestion / `_update_weights_delta_flush` / the `LocalSerializedTensor` runtime call, the
  steady-state delta sync (speed + no hang), the step-21 OOM, the TP=32 SGLang MLA hang, any
  training step.
- **Fix**: none — hardware node failure. Clean case for an unmodified resubmit (same as this
  repo's infra-flake precedent, `3152802`→`3171176` / `3207923`→`3209484`). Per memory
  `hpc_job_fix_then_ask`, asked the user before resubmitting.
- **Commit**: not committed.

### Run `3240762` — 2026-08-31 — unmodified retry of 3240384: cleared the node-failure, then hit the megatron-bridge weight-sync desync (4th occurrence) — this time on the `delta_sharded` SEED sync. Seed-sync lockstep-barrier fix added.

- **Log**: `~/Downloads/slurm-3240762.out` (460 KB, complete). 80 nodes, `clariden`, image
  `verl-cuda:alps7-dev-621fa40275c4f036`. ~5 min queued, ran ~50 min, FAILED (exit 15) — **the
  megatron-bridge weight-sync collective desync, NOT a node failure** (no NODE_FAIL, no raylet
  death, no OOM).
- **Everything green up to the seed sync** — same as `3240384`: diagnostic block (TE 2.12.0 /
  TE-torch 2.12.0, all 3 smoke tests OK), 6 patches 80/80, DDP grad-buffer
  (`9.24B parameters`), GLM5Bridge conversion (6201/6201), SGLang TP=32 rollout HTTP up (no
  flashinfer MLA hang — that fix holds), `checkpoint_engine.backend: delta_sharded` active.
- **Symptom**: the seed weight sync hung. ~96 trainer ranks finished the seed export cleanly
  in 77–132s (`[WSYNC-DBG] seed/full generator exited after 59079 tensors`); the rest stalled
  after **545 / ~1116 tensors** and ran the full 1800000ms (30-min) NCCL/gloo timeout.
  ```
  delta_checkpoint_engine.py:521  _send_full_seed  →  for name, tensor in weights
  model_bridge.py:1353            stream_weights_megatron_to_hf
  param_mapping.py:1581           megatron_to_hf
  param_mapping.py:465            broadcast_obj_from_pp_rank
                                 torch.distributed.all_gather_object(..., group=self.pp_group)
  RuntimeError: gloo ... Timed out waiting 1800000ms for recv operation
  ```
  Plus `gather_from_ep_ranks` (`param_mapping.py:806`) NCCL ALLGATHER timeouts on other ranks.
  `last enqueued work: 172, last completed work: 109` (PP group) — a rank raced ~63 collectives
  ahead, one never matched. Ranks 244 and 33 flagged as the peers that never entered.
- **Root cause — 4th occurrence** (`3141801` / `3207923` / `3219811` / `3240762`) of the
  megatron-bridge weight-streaming collective desync, and the **first on the `delta_sharded`
  seed sync**. `delta_sharded` fixed the STEADY-STATE syncs (validated: `3223205` trainer-side
  seed export was clean, no `gather_from_ep_ranks`) — but its one-time seed sync
  (`_send_full_seed`) still streams the full `get_per_tensor_param()` HF export, which runs a
  chain of PP/EP/TP assembly collectives per HF tensor. `_send_full_seed` drives that generator
  **asymmetrically**: rank 0 buckets and broadcasts each flush to the rollout CE group between
  pulls, while every non-master rank just discards its tensor and immediately pulls the next —
  so non-master ranks race ahead in the per-tensor collective chain (the observed 63-tensor
  drift) until a later cross-group collective deadlocks. This is the mechanism the earlier
  "one rank momentarily stalled" investigation (run `3219811` entry) suspected but couldn't
  pin — the `delta_sharded` seed path makes the asymmetry explicit and reproducible.
- **Fix — seed-sync lockstep barrier** (`example/patches/wsync-debug-progress-log.patch`,
  expanded from the diagnostic-only version; checked-in file + script heredoc verified
  byte-identical; `git apply --check` / `py_compile` / `--reverse --check` all clean against a
  fresh v0.9.0 worktree). `_wsync_progress_log` now, for the `"seed/full"` export only, calls
  `torch.distributed.barrier()` on the trainer WORLD group every
  `VERL_WSYNC_SEED_BARRIER_EVERY` (default **1**) HF tensors, **after** the consumer processed
  each — no rank can start tensor K+1's assembly collectives until every rank finished K, so the
  drift is bounded to `VERL_WSYNC_SEED_BARRIER_EVERY`. One-time cost on the seed only
  (~1–3 ms/barrier × ~59079 items ≈ 1–3 min added to the ~80–130s seed; the env var lets us
  raise it to 8/16 without a rebuild if that's too slow). The `delta-steady` path is already
  count-lockstepped by design and is **not** barriered. Not a monkeypatch of megatron-bridge —
  it's a WORLD barrier in verl's own generator wrapper, which is strictly safe (a barrier
  cannot corrupt state; worst case it doesn't prevent this specific race and costs a few
  minutes).
- **Still not validated end-to-end** (job died in the seed sync): `delta_sharded` rollout-side
  ingestion / `_update_weights_delta_flush` / `LocalSerializedTensor` runtime call, steady-state
  delta sync speed, step-21 OOM, any training step.
- **Fix**: `wsync-debug-progress-log.patch` seed-barrier (above), unverified on cluster. Per
  `hpc_job_fix_then_ask`, asking the user before resubmitting.
- **Commit**: not committed.

### Run `3241496` — 2026-08-31 — seed-sync barrier fix WORKS (288/288 ranks, no hang) + `LocalSerializedTensor` runtime path WORKS; new blocker: fused_adam optimizer-state OOM at STEP 1 on local-GPU-0. 104-node / DP=4 fix.

- **Log**: `~/Downloads/slurm-3241496.out` (728 KB, complete). 80 nodes, `clariden`, image
  `verl-cuda:alps7-dev-621fa40275c4f036`, **seed-sync barrier applied** (`wsync-debug-progress-log.patch`
  expanded). ~47 min queued, ran ~27.7 min, FAILED (exit 15) — 24× CUDA OOM over ~3.5 min in the
  first optimizer step, then Slurm force-terminated. Phase 7 (training loop) entered, **0 steps**.
- **THE SEED-SYNC BARRIER FIX WORKS** — the 4-occurrence megatron-bridge desync is resolved:
  - `[WSYNC-DBG] rank=N seed/full generator exited after 59079 tensors, 139s` on **all 288
    trainer ranks** (`[repeated 287x across cluster]`). **Zero** `gather_from_ep_ranks` /
    `all_gather_object` / `OpType=ALLGATHER` / `1800000ms` anywhere in the log.
  - `delta-sharded FULL-SEED v=0 done in 139.6s (flushes=687 wire=1385.6GB)`, then
    `cupy staging pool after seed send: held 3.54GB; device free 57.10->60.65GB on release` —
    staging cleanly returned, **60.65 GB free on GPU 0 right after the seed**.
  - Seed took ~140s vs ~80–130s unbarriered — the predicted modest slowdown, no hang.
- **`LocalSerializedTensor` runtime path WORKS** (run `3223205`'s blocker): rollout-side seed
  ingestion `delta apply v=0 flushes=687 (in-place via sglang loader)` on all 32 CE workers, no
  `ImportError`. `delta-sharded-localserializedtensor-import-fix.patch` validated end-to-end.
- **Also confirmed**: TE 2.12.0 pin holds, DDP grad-buffer (`9.24B parameters`), GLM5Bridge
  6201/6201, SGLang TP=32 rollout up (no flashinfer MLA hang).
- **New blocker — CUDA OOM in `fused_adam._initialize_state`** (lazy `exp_avg`/`exp_avg_sq`
  alloc on the first `optimizer.step()`), traceback
  `engine_workers.py:710 update_actor → transformer_impl.py:736 optimizer_step →
  megatron/core/optimizer/distrib_optimizer.py:3163 → transformer_engine/pytorch/optimizers/fused_adam.py:381
  torch.empty_like`. Hit **local-GPU-0 on ~9+ distinct trainer nodes** (every OOM says "GPU 0";
  the failing actor ip/pid varies across 9 IPs). Tried 12–24 MiB, `8.75 MiB free`. Numbers
  thrash across the 24 retries (PyTorch-allocated 58–78 GiB, process total 71–91 GiB) but the
  non-PyTorch slice is a **consistent ~12.3 GiB** (NCCL comm buffers + CUDA context on each
  node's device 0), and **"reserved but unallocated" is only 35–95 MiB — NOT fragmentation**, so
  `expandable_segments` would not help (and breaks SGLang, run `3219305`).
- **Same OOM class as run `3217439`** (nccl backend, step 21, `fused_adam` / GPU 0 / ~24 MiB
  short / ~17 GiB non-torch). `delta_sharded` moved it **earlier** (step 21 → step 1): its seed
  sync runs at `on_init_end`, so the CE NCCL group + residual buffers are already resident when
  step 1's optimizer runs, whereas the nccl backend's first sync is at step 2. Not a regression
  from `delta_sharded` per se — the underlying "local-GPU-0 carries ~12 GiB extra, workload
  sized to just fit a normal GPU" was always there.
- **Fix — 104 nodes / DP 3→4 + token cut** (user-approved, 2026-08-31): `#SBATCH --nodes` 80→104
  (`TRAINING_NNODES` 72→96 → 384 trainer GPUs → DP = 384/(TP4·PP3·EP8) = 4);
  `PPO_MINI_BATCH_SIZE` 6→8 (× `ROLLOUT_N` 8 = 64 rows = 2× the new `dp_size` DP4·EP8 = 32 —
  `TRAIN_BATCH_SIZE` auto-derives 12→16 prompts); `ppo_max_token_len_per_gpu` 12288→8192.
  DP 3→4 shrinks the per-GPU grad-buffer / optimizer working set / activation footprint ~25%
  (multi-GiB, vastly more than the 12 MiB shortfall); the token cut is paired insurance on the
  activation peak. Rejected: `expandable_segments` (not fragmentation, breaks SGLang), NCCL-env
  trims (gamble, touches the just-stabilized collective stack). Unverified on cluster.
- **Still not validated**: `delta_sharded` steady-state delta sync speed (seed done, steady never
  reached), any training step, step-21 OOM.
- **Commit**: not committed (script: 104n/DP4/token; `wsync-debug-progress-log.patch` seed-barrier).

### Run `3243323` — 2026-08-31 — 104 nodes / DP=4: seed-barrier + LocalSerializedTensor re-validated; step-1 `fused_adam` OOM RECURRED (DP=4 + token cut cut ~3–6 GiB, still 12–24 MiB short). Deep-dived the memory; ~17.7 GiB unattributed.

- **Log**: `~/Downloads/slurm-3243323.out` (700 KB, complete). **104 nodes** (96 trainer → DP 3→4),
  `PPO_MINI_BATCH_SIZE` 8, `ppo_max_token_len_per_gpu` 8192, `clariden`, image
  `alps7-dev-621fa40275c4f036`. ~53 min queued, ran ~29 min, FAILED (exit 15) — 24× CUDA OOM in
  step 1's optimizer, then Slurm force-terminated. 0 steps.
- **Re-validated (3rd time now)**: seed-sync WORLD-barrier — `seed/full generator exited after
  59079 tensors, ~140s` on all **384** ranks, zero `gather_from_ep_ranks` / hang. `LocalSerializedTensor`
  runtime path — `delta apply v=0 flushes=687` on all 32 CE workers. TE 2.12.0, R3 active
  (`routing replay layers: 23/26`), SGLang rollout up (no MLA hang).
- **Symptom — same as `3241496`**: `fused_adam.py:381 _initialize_state` → `torch.empty_like` for
  `exp_avg`/`exp_avg_sq` on the first `optimizer.step()`, OOM by 12–24 MiB on trainer-node
  local-GPU-0 (nodes `172.28.51.179` pids 171645–648, `172.28.42.20`, and ~5 more). DP 3→4 +
  token 12288→8192 cut PyTorch-allocated ~3–6 GiB (3241496 ~64–68 GiB → this run ~61–66 GiB) —
  **not enough**.
- **Memory forensics** (from the saved log, per OOM message):
  `GPU 0: 95.00 total, ~7.5 MiB free, this process ~77.3 GiB (PyTorch ~65 GiB + non-PyTorch
  ~12.3 GiB), "reserved but unallocated" 35–95 MiB` → **~17.7 GiB on GPU 0 unaccounted by
  PyTorch's "this process"** (95 − 77.3 − 0.007). Could NOT definitively attribute it:
  - **Not oversubscription** — each OOM node has exactly 4 `WorkerDict` pids (1/GPU).
  - **Not a co-resident SGLang / CE worker** — the standalone rollout (8 nodes) and the 32 CE
    workers are on disjoint IPs from every OOM node.
  - **Not a ref model** — `need_reference_policy` = `use_kl_in_reward OR actor.use_kl_loss`, both
    `False` here, so verl builds `Role.ActorRollout` (no ref); the `"actor and ref model engine
    initialized"` log line is unconditional and does not imply a ref was created.
  - **Not `critic`** (`enable: False`), not a `SimpleStorageUnit` requirement (node `.42.20`
    OOM'd with no StorageUnit co-resident).
  - **Leading candidates, unconfirmed**: (a) onloaded bf16 params (`9.24B × 2 ≈ 18.5 GiB`,
    `param_offload: True` onloads them during `train_mode`) mis-attributed by PyTorch's accounting
    when `load_megatron_model_to_gpu` bypasses the caching allocator; (b) C-extension memory
    PyTorch under-counts — TE fused-adam workspace, megatron-core distrib-optimizer scratch, NCCL
    NVLS/registered buffers, R3/cupy routed-experts buffers; (c) same-GPU orphan process from
    node reuse (this repo's run `3185487` precedent) — but 5+ nodes is a lot for random orphans.
  - **Same OOM class as `3217439`** (nccl, step 21). `delta_sharded` moved it to step 1 (seed
    sync at `on_init_end` leaves buffers resident before step 1; nccl's first sync is step 2).
- **Fix — NOT applied, needs user decision.** DP scaling gives only ~2–3 GiB per bump. Higher-
  leverage, in order:
  1. **`actor.optim.use_precision_aware_optimizer: True` + `exp_avg_dtype: bf16` +
     `exp_avg_sq_dtype: bf16`** (`verl/workers/config/optimizer.py:201-204`; currently all
     default fp32) — halves the two biggest optimizer tensors, ~7 GiB saved. Standard megatron-core
     feature; slight numerical change (bf16 2nd moment); untested here with `optimizer_offload` +
     TE fused-adam.
  2. **`NCCL_NVLS_ENABLE=0`** in the srun env — frees NVLink-multicast buffers (part of the
     ~12.3 GiB NCCL footprint); pure perf knob, low risk, ~2–4 GiB.
  3. More trainer nodes → DP=5 (128 total) — guaranteed but +24 nodes on top of 104.
  Recommendation: **1 + 2 together, keep 104 nodes** (or drop to 80 to test whether #1 alone
  suffices).
- **IMPORTANT framing correction (user, 2026-08-31)**: the "12–24 MiB short" throughout this
  entry and `3241496`'s is **misleading** — it is only the size of the *single* `torch.empty_like`
  that happened to fail, mid-way through `fused_adam._initialize_state` iterating over the whole
  sharded param set. The optimizer-state init is a **multi-GiB cumulative allocation** (this
  rank's DP-shard of fp32 master + `exp_avg` + `exp_avg_sq` ≈ ~28 GiB fp32 total at DP=4), and it
  ran out partway. The true deficit is **unknown** and could be several GiB — the PyTorch-allocated
  number bouncing 58→78 GiB across the 24 retries is consistent with a large, still-incomplete
  allocation, not a hairline miss. So a fix must free GiB, not MiB.
- **Fix applied (user-approved, 2026-08-31), unverified on cluster**: **1 + 2, kept 104 nodes.**
  Added to the script: `actor_rollout_ref.actor.optim.{use_precision_aware_optimizer: True,
  exp_avg_dtype: bf16, exp_avg_sq_dtype: bf16}` (new `optim:` block, sibling of `actor.megatron`;
  `main_grads_dtype` left `fp32`) — halves `exp_avg`/`exp_avg_sq` (the exact tensors that fail),
  ~28 GiB fp32 optimizer state → ~18.5 GiB, **~9 GiB saved**. Plus `export NCCL_NVLS_ENABLE=0` in
  the srun env block (right after the `TORCH_NCCL_*` flight-recorder exports) — frees
  NVLink-multicast buffers, part of the ~12.3 GiB non-PyTorch footprint, ~2–4 GiB. Caught + fixed
  one stray apostrophe ("recipe's") in the new NVLS comment inside the single-quoted `srun bash
  -c '...'` body before submitting (the run-3149339 hazard — `bash -n` flagged it). `bash -n` +
  full stray-quote / backtick sweeps clean.
  **This run will actually MEASURE the deficit** — the OOM message's this-process / PyTorch-allocated
  numbers (and how far into the param loop it gets) will show whether ~11–13 GiB of new headroom
  is enough or whether DP=5/6 / TP=8 / a smaller config is needed. bf16 2nd moments also carry a
  slight numerical risk (watch grad_norm / NaN over the first steps).
- **Commit**: not committed.

### Run `3244653` — 2026-08-31 — precision-aware bf16 moments barely helped (~1.4 GiB, all from NVLS-off); step-1 OOM recurred. Memory math points at ~18.5 GiB of onloaded bf16 params PyTorch doesn't count.

- **Log**: `~/Downloads/slurm-3244653.out` (584 KB). 104 nodes, `use_precision_aware_optimizer:
  True` + `exp_avg{,_sq}_dtype: bf16`, `NCCL_NVLS_ENABLE=0`, `ppo_max_token_len_per_gpu` 8192.
  ~29 min run, FAILED — step-1 `fused_adam` OOM again (only 3 OOMs this time, then `srun` task 0
  exited → cascade).
- **Re-validated (4th time)**: seed sync clean on all 384 ranks, 137s, zero hang (NVLS-off did
  not slow it). LocalSerializedTensor, TE 2.12.0, R3, SGLang rollout — all fine.
- **Symptom**: `fused_adam.py:568 step → :324 get_unscaled_state → unscaled =
  state[state_name].float()` — TE fused-adam **upcasts each bf16 moment back to fp32 on GPU** at
  step time. `Tried 24.00 MiB. GPU 0: 30.75 MiB free. this process 75.63 GiB (PyTorch 63.07 +
  non-PyTorch 12.2), reserved-but-unallocated 405 MiB.`
- **Delta vs `3243323`** (~65 GiB PyTorch / ~77 GiB process): PyTorch-allocated **65 → 63.07 GiB**
  (−~2), process **77 → 75.63 GiB** (−~1.4). **Almost all the ~1.4 GiB gain is from
  `NCCL_NVLS_ENABLE=0`.**
- **Why precision-aware bf16 moments did ~nothing for GPU**: `optimizer_offload: True` already
  keeps the fp32 master + moments on **CPU**, so bf16 saves *CPU* memory, not GPU. And TE
  fused-adam's `get_unscaled_state` does an explicit `.float()` upcast of each moment **on GPU 0**
  during the step — so bf16 actually *adds* a transient GPU alloc on the tight GPU. Net ≈ neutral
  (slightly negative). **→ revert `use_precision_aware_optimizer`.**
- **The ~19 GiB "another process" — best current theory**: `95.00 − 75.63 (this process) − 0.03
  (free) = 19.3 GiB` used by *something else* on trainer GPU 0. Ruled out (runs `3243323`/`3244653`):
  oversubscription (exactly 4 `WorkerDict` pids/node), co-resident SGLang/CE (disjoint IPs), ref
  model (`need_reference_policy` False — no ref built), critic (disabled), `SimpleStorageUnit`
  requirement (a node OOM'd with none co-resident). **Leading unconfirmed theory: it is the
  actor's own onloaded bf16 params** — `9.24B × 2 bytes ≈ 18.5 GiB`, matches ~19 GiB — brought
  on-GPU by verl's `param_offload` onload (`load_megatron_model_to_gpu`) via a path PyTorch's
  caching allocator / process-footprint accounting doesn't track, so it shows as "another
  process" rather than part of "this process 75.63 GiB". If true, the real per-GPU need is
  ~75.6 + 18.5 ≈ 94 GiB, and the fp32 upcast (24 MiB) is the final straw. **Not confirmed — the
  log has NO memory instrumentation; a diagnostic run (dump `torch.cuda.memory_summary()` +
  `nvidia-smi --query-compute-apps` on the OOMing rank right before `optimizer.step()`) would
  settle it.**
- **Fix — NOT applied, needs user decision.** Levers, now better understood:
  1. **Revert `use_precision_aware_optimizer`** (net-negative here) — do regardless.
  2. **`ppo_max_token_len_per_gpu` 8192 → 4096** — real cut to the step-1 activation contribution
     to the 63 GiB PyTorch number (~3–6 GiB).
  3. **TP 4 → 8** (keep 96 trainer nodes, PP=3, EP=8, DP=2): halves the per-GPU param shard
     (→ ~9.2 GiB) *and* optimizer state — directly halves the suspected ~18.5 GiB. Cost: TP=8
     spans 2 GH200 nodes → half the per-layer TP collective goes over Slingshot, not NVLink — a
     real per-step slowdown. `DP=2` was flagged insufficient at TP=4 but the arithmetic
     (`3M + 11 GiB` at TP=8 vs `6M + 11` at TP=4) says it fits with room at TP=8.
  4. **DP 4 → 5** (128 total nodes) — +24 nodes again; ~2–3 GiB/bump, may still not be enough
     if the ~18.5 GiB param theory is right (DP doesn't shard params).
  5. **Diagnostic run first** — instrument the OOMing rank, confirm the ~19 GiB, then fix
     precisely. Costs one run.
- **Decision (user, 2026-08-31): #5, diagnostic run.** Reverted `use_precision_aware_optimizer`
  (removed the `optim:` block, left a NOTE explaining why); kept `NCCL_NVLS_ENABLE=0` (~1.4 GiB,
  safe); kept 104 nodes and `ppo_max_token_len_per_gpu: 8192` unchanged so the memory picture is
  directly comparable to `3244653`. Added **`example/patches/step1-oom-memdump.patch`** (new,
  diagnostic-only): `_step1_oom_memdump()` fires once per rank right before the first
  `optimizer.step()` — every rank prints its own torch CUDA accounting
  (`free/total/gpu_used/torch_alloc/torch_reserved`), and local rank 0 (the GPU that OOMs) also
  shells out to `nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory` +
  `--query-gpu=...` + `torch.cuda.memory_summary()`. `[MEMDUMP]` lines land in the main slurm log
  just before the crash → identifies the ~19 GiB by pid/process. **Must apply after
  `wsync-debug-progress-log.patch`** (both touch `transformer_impl.py`; the memdump diff's context
  assumes `_wsync_progress_log` is present) — wired into the apply loop last, after
  `delta-sharded-localserializedtensor-import-fix.patch`. Verified: fresh v0.9.0 → wsync → memdump
  applies clean + compiles + reverse-checks; heredoc byte-identical to the checked-in file;
  `bash -n` + stray-quote/backtick sweeps clean.
- **Commit**: not committed.

### Run `3246683` — 2026-08-31 — step-1 OOM diagnostic: `[MEMDUMP]` fired, mystery SOLVED — there is NO phantom process; it is a genuine ~12–24 MiB hairline miss on the fused_adam optimizer-state transient on GPU 0.

- **Log**: `~/Downloads/slurm-3246683.out` (6842 lines). 104 nodes, `NCCL_NVLS_ENABLE=0`,
  `use_precision_aware_optimizer` reverted, `ppo_max_token_len_per_gpu: 8192`,
  `step1-oom-memdump.patch` + `wsync-debug-progress-log.patch` applied. ~28 min, FAILED — step-1
  `fused_adam` OOM as expected. All 8 patches applied 104/104, seed sync clean (131s, all 384
  ranks, no hang), SGLang rollout up — everything through step 1 works.
- **`[MEMDUMP]` fired on every rank right before the first `optimizer.step()`.** lrank=0 header
  (the GPU that OOMs), representative: `free≈35.5G total=95.00G gpu_used≈59.4G torch_alloc≈45.7G
  torch_reserved≈48.9G non_torch_this_proc≈10.6–12.1G`.
  **`nvidia-smi --query-compute-apps` for the OOMing GPU — THE KEY DATA:**
  ```
  GPU-c51c…, 181272, ray::WorkerDict.actor_rollout_update_actor, 61384 MiB
  GPU-ddcf…, 181273, ray::WorkerDict.actor_rollout_update_actor, 61512 MiB
  GPU-209a…, 181274, ray::WorkerDict.actor_rollout_update_actor, 61512 MiB
  GPU-1038…, 181275, ray::WorkerDict.actor_rollout_update_actor, 61512 MiB
  ```
  **Exactly ONE process per GPU — the trainer `WorkerDict` itself.** No SGLang, no
  CheckpointEngineWorker, no ref/critic, no second CUDA context. `nvidia-smi --query-gpu` shows
  GPU total **97871 MiB (≈95.6 GiB)**, not the 95.00 PyTorch reports.
- **THE "~19 GiB another process" FROM RUNS 3241496/3243323/3244653 WAS A MISATTRIBUTION.** It
  came from reading PyTorch's OOM message (`95.00 total − 75.6 this process = 19.4 "other"`)
  literally — but "95.00" is PyTorch rounding down the real ~95.6 GiB, and its "this process has
  X" figure is a partial estimate that misses some driver/context bytes, so the subtraction
  manufactured a phantom. The `[MEMDUMP]` `nvidia-smi` proves there is no other process.
- **Real mechanism**: right before the optimizer step GPU 0 has ~35 GiB free (torch reserved
  ~48 GiB + non-PyTorch ~12.3 GiB). `fused_adam.initialize_state` then allocates the full
  distributed-optimizer shard on GPU 0 — fp32 master-remainders + `exp_avg` + `exp_avg_sq`,
  ~35 GiB for the DP=4 shard — and misses by **12–24 MiB**. `optimizer_offload: True` only helps
  *between* steps; during the step the whole shard is resident on GPU. **Genuinely a hairline
  miss** — deficit is MiB-to-maybe-a-few-hundred-MiB, not GiB (contra the earlier "could be much
  more" worry; the `[MEMDUMP]` free-before / OOM-after bracket confirms it).
- **This run's OOM numbers ≈ identical to `3244653`** (`63.08 GiB alloc / 75.31 GiB process`).
  NVLS-off did not reduce the ~12.3 GiB non-PyTorch. DP 3→4, token 8192, precision-aware revert —
  all moved the step-1 picture by ~0.
- **Fix directions (not applied)** — now a confirmed hairline miss, so cheap/targeted first:
  1. **`torch.cuda.empty_cache()` immediately before `optimizer.step()`** — `[MEMDUMP]` shows
     torch_reserved (~48.9) > torch_alloc (~45.7): ~3 GiB of reserved-but-unallocated cache.
     Returning it to the driver before the optimizer's large contiguous allocs could hand it the
     few hundred MiB it needs. Tiny, safe. (`optimizer_step` already has a *conditional*
     `empty_cache()` for the distillation-topk path — make it unconditional.)
  2. **`ppo_max_token_len_per_gpu` 8192 → 4096** — lowers the fwd/bwd reserved high-water
     (peak reserved ~59 GiB per `memory_summary`) carried into the optimizer step.
  3. **DP 4 → 5** (128 nodes) — shrinks the optimizer shard ~35 → ~28 GiB. Guaranteed, +24 nodes.
  4. **megatron-core CPU-streaming optimizer** (`use_layer_wise_distributed_optimizer`, currently
     `False`; or `HybridDeviceOptimizer` / `optimizer_cpu_offload`) — streams optimizer state
     per-bucket instead of whole-shard-on-GPU, structurally cutting the ~35 GiB. Needs research.
  Recommendation: **1 + 2 together, keep 104 nodes**; DP=5 (#3) fallback; #4 the "proper" fix.
- **Fix applied (user-approved, 2026-08-31), unverified**: **1 + 2.** `step1-oom-memdump.patch`
  expanded (same filename, header updated to "FIX + diagnostic"): `optimizer_step()`'s
  `get_torch_device().empty_cache()` made **unconditional** (was gated on the
  `_distillation_use_topk_active` path, unused here) — returns the ~3 GiB of PyTorch
  reserved-but-unallocated cache to the driver right before `self.optimizer.step()`, so
  `fused_adam.initialize_state` gets a clean arena. `_step1_oom_memdump()` kept (confirms
  whether the fix cleared it + by how much). And `ppo_max_token_len_per_gpu` **8192 → 4096**.
  Kept: 104 nodes, `NCCL_NVLS_ENABLE=0`, `use_precision_aware_optimizer` reverted. Verified:
  fresh v0.9.0 → wsync → step1 patch applies + compiles + reverse-checks; heredoc byte-identical;
  `bash -n` + sweeps clean.
- **Commit**: not committed.

### Run `3247517` — 2026-08-31 — `empty_cache()` fix WORKED (reserved−alloc gap 3 GiB → 0.15 GiB) but step-1 OOM still recurred (5th). The optimizer-state init genuinely needs ~38 GiB and GPU 0 has ~38 GiB — zero margin. Needs DP scaling or CPU-streaming optimizer.

- **Log**: `~/Downloads/slurm-3247517.out` (742 KB). 104 nodes, unconditional `empty_cache()`,
  `ppo_max_token_len_per_gpu: 4096`, `NCCL_NVLS_ENABLE=0`. ~28 min, FAILED — step-1 `fused_adam`
  OOM (5th consecutive). All 8 patches 104/104, seed sync clean (132s, all 384 ranks), SGLang
  rollout up.
- **The `empty_cache()` fix did exactly what it should**: `[MEMDUMP]` right before the optimizer
  step shows `torch_reserved − torch_alloc = 0.15 GiB` (was ~3 GiB in `3246683`). Reserved-but-
  unallocated at OOM: 16–52 MiB — nothing left to reclaim. The cache is not the problem.
- **`ppo_max_token_len_per_gpu` 8192→4096 had zero visible effect** — expected: the OOM is in
  **fixed-size optimizer-state init** (fp32 master-remainders + `exp_avg` + `exp_avg_sq` for the
  DP=4 shard), not token-dependent activations.
- **The real number**: `[MEMDUMP]` shows **~38 GiB free on GPU 0 right before `optimizer.step()`**
  (`torch_alloc ≈44.3, torch_reserved ≈44.5, non_torch ≈12.1, gpu_used ≈56.6, free ≈38.4`).
  `fused_adam.initialize_state` then allocates the full optimizer-state shard — which needs
  **~38 GiB** — and misses by 12–24 MiB. **Genuine hairline miss, but the thing that has to fit
  is ~38 GiB, so there is essentially zero margin and every run lands on OOM.** The user's
  "could be much more" caution was right: the *deficit* is MiB, but the *allocation that must
  fit* is ~38 GiB, so a fix must free multiple GiB.
- **`use_precision_aware_optimizer` cannot help** (tried `3244653`): with `optimizer_offload:
  True` the fp32 master+moments live on CPU between steps, and TE's `get_unscaled_state` upcasts
  to fp32 on GPU anyway during the step.
- **Fix — NOT applied. Needs user decision. Levers that actually free GiB:**
  1. **DP 4 → 5** (128 nodes: 120 trainer, 480 GPUs / (TP4·PP3·EP8) = 5) — shrinks the optimizer
     shard `~38 → ~30 GiB` (~7.6 GiB margin, overwhelming for a MiB miss). Config-only, no new
     patches, no risk. `dp_size` = 5·8 = 40 → `PPO_MINI_BATCH_SIZE` 8 → 10 (10·8 = 80 = 2×),
     `TRAIN_BATCH_SIZE` auto 16→20. +24 nodes (128 = 60% over the original 80).
  2. **DP 4 → 6** (152 nodes) — `~38 → ~25 GiB`, ~12.7 GiB margin. `PPO_MINI_BATCH_SIZE` → 12.
     +48 nodes (90% over 80).
  3. **megatron-core CPU-streaming optimizer** — `use_layer_wise_distributed_optimizer` (config
     field, currently `False`) or `HybridDeviceOptimizer` / per-bucket `optimizer_cpu_offload`:
     stream the optimizer state from CPU per-bucket during the step instead of whole-shard-on-GPU.
     Structurally removes the ~38 GiB (could even allow going back toward 80 nodes). Needs a
     research pass — confirm it composes with verl's V1 Megatron path + TE fused-adam + the
     `distrib_optimizer.py:3163` code that OOMs — and likely its own debugging.
  Recommendation: **DP=5 (#1)** — cheapest guaranteed fix, we've validated everything else works,
  the miss is tiny. #3 is the "stop adding nodes" play but carries its own risk after 5 clean
  failures on the same line.
- **Fix applied (user-approved, 2026-09-01)**: **DP=5.** `#SBATCH --nodes` 104→128,
  `PPO_MINI_BATCH_SIZE` 8→10.

### Run `3250425` — 2026-09-01 — DP=5 / 128 nodes: step-1 OOM RECURRED (6th). DP scaling has hit diminishing returns — the optimizer.step() transient (~40 GiB) is dominated by the full-param all-gather buffer, which does NOT shrink with DP.

- **Log**: `~/Downloads/slurm-3250425.out` (665 KB). 128 nodes (DP=5), unconditional
  `empty_cache()`, `ppo_max_token_len_per_gpu: 4096`. ~32 min, FAILED — step-1 `fused_adam` OOM
  (6th consecutive). All 8 patches 128/128, **seed sync clean (136s, all 480 ranks, zero
  `gather_from_ep_ranks`)**, delta ingestion worked, SGLang rollout up.
- **`[MEMDUMP]` free-before-step: 38.9 / 39.9 / 42.3 GiB** — vs `3247517`'s ~38 GiB. **DP 4→5
  bought only ~2 GiB, not the projected ~8.** `torch_reserved − torch_alloc` = 0.15–0.22 GiB
  (empty_cache working).
- **OOM**: `fused_adam.py:381 _initialize_state → torch.empty_like` for `exp_avg_sq`.
  `Tried 24 MiB · 4.81 MiB free · process 73.6 GiB · PyTorch-allocated 61.4 GiB ·
  reserved-unalloc 45 MiB` (not fragmentation). **~19 MiB short — trajectory essentially
  unchanged from `3247517`.**
- **Why DP scaling stalled** (analysis): at MEMDUMP time GPU 0 holds ~40 GiB (bf16 params 18.5,
  which DP does NOT shrink; grad buffer ~7–9; fp32 master ~7–9; ~12 non-PyTorch). DP 4→5 shrinks
  only the grad-buffer + fp32-master shards (~3.6 GiB total). Then `optimizer.step()` allocates
  its ~40 GiB transient: `exp_avg` + `exp_avg_sq` (`2 × 9.24B·4/DP` ≈ 14.8 GiB at DP=5, DOES
  shrink) **plus the distributed-optimizer's full-param all-gather buffer (~18.5 GiB, full param
  size, does NOT shrink with DP)** plus grad-norm + `.float()` upcast scratch. Against ~40 GiB
  free, it misses by MiB every time. **DP=6/7 would keep giving ~2–4 GiB each — the all-gather
  buffer and the bf16 params are the fixed floor.**
- **Fix — NOT applied. The real lever is the optimizer's all-gather buffer / whole-step
  transient, not DP.** Options:
  1. **`use_layer_wise_distributed_optimizer: True`** (`verl/workers/config/optimizer.py`, seen
     `False` in every config dump) — Megatron-core processes the optimizer update **layer-by-
     layer**, keeping only one layer's optimizer state + all-gather buffer on GPU at a time.
     Cuts the ~40 GiB transient to single-digit GiB. Could allow going back to 104 or even 80
     nodes. **Best structural fix**; needs a research pass (does it compose with verl V1 Megatron
     + TE fused-adam + `optimizer_offload` + the `distrib_optimizer.py:3163` path?) and probably
     its own debugging — but every other lever is exhausted.
  2. **`overlap_param_gather: False`** / a smaller `distributed_optimizer` all-gather bucket — do
     the param all-gather in chunks instead of one ~18.5 GiB buffer. Smaller change than #1.
  3. **DP=6 (152 nodes)** — ~2–4 GiB more. Brute force, diminishing, +48 nodes (90% over 80).
  Recommendation: **#1 (layer-wise optimizer)** — DP scaling is out of runway.
- **Fix applied (user-approved, 2026-09-01) — CPU-streaming (HybridDevice) optimizer.** Research
  found the real answer: `use_layer_wise_distributed_optimizer` is **Muon-only** in verl v0.9.0
  (`verl/utils/megatron/optimizer.py:33 is_muon_layer_wise_config`), doesn't apply to Adam. BUT
  `actor.optim.override_optimizer_config` (a real `McoreOptimizerConfig` field,
  `verl/workers/config/optimizer.py:113`) is forwarded **verbatim** to Megatron-core's
  `OptimizerConfig(**optim_args)` (`optimizer.py:210`), and **verl's own canonical large-MoE
  Megatron-async recipes use exactly this** — `verl/experimental/fully_async_policy/shell/
  grpo_30b_a3b_base_math_megatron_96_32_mis.sh` (Qwen 30B-A3B MoE, 96+32 nodes) and every other
  30B/35B MoE megatron-async example script set the same four keys:
  ```yaml
  actor.optim.override_optimizer_config:
    optimizer_cpu_offload: True          # -> Megatron HybridDeviceOptimizer: optimizer state
    optimizer_offload_fraction: 1.0      #    lives on CPU, streamed bucket-by-bucket during
    overlap_cpu_optimizer_d2h_h2d: True  #    step(); GPU never holds the full moments
    use_precision_aware_optimizer: True  #    (required by that path; moments stay fp32)
  ```
  `actor.megatron.{param,grad,optimizer}_offload: True` stays on alongside (verl's
  `offload_megatron_optimizer` already handles a HybridDeviceOptimizer via
  `_move_new_state_to_right_device`, and the example scripts keep `megatron.optimizer_offload`
  on too). This directly removes the ~40 GiB GPU optimizer transient that DP scaling couldn't
  touch. **Kept 128 nodes / DP=5 / `PPO_MINI_BATCH_SIZE=10` for this run to isolate the
  optimizer change** — if it works, node count can come back down in a follow-up. Config-only,
  no new patches, same image. `bash -n` + YAML-indent + backtick sweeps clean.
- **Commit**: not committed.

# Training Apertus v1.5 on verl (`rl-bench-apertus-v1.5-70B-sglang-fsdp2-async.sh`)

Apertus-v1.5-70B GRPO on GSM8K, CSCS Alps, 16 nodes × 4 GH200 (`clariden`). Script:
`Alps-Images/apps/verl/apertus-benchmarks/rl-bench-apertus-v1.5-70B-sglang-fsdp2-async.sh`. 8 nodes →
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
4. **The `qwen3_asr` config-registration collision** (same root cause as the GLM section's
   "sys.meta_path hooks" hazard above — the swiss-ai fork already registers types stock
   transformers doesn't, SGLang's own `AutoConfig.register(..., exist_ok=False)` collides) also
   hits here (run `3130211`, repeated in `3133412`/`3133616`/`3134586` while the fix mechanism was
   still broken — see point 8). Fixed for good by PR #32979 itself (`exist_ok=True` in its own
   `qwen3_asr.py`, part of the source patch in point 3) — no separate fix needed once that PR is
   applied, so unlike points 5/6 there is nothing left over from this one.
5. **`sgl-project/sglang#32979`'s own `_init_component_model` assumes a config *object*.**
   `component_config.to_dict()` raised `AttributeError: 'dict' object has no attribute 'to_dict'`
   (run `3137554`) — against this transformers commit, `config.vision_tokenizer_config` /
   `.audio_tokenizer_config` are already plain dicts. A version-alignment gap between the two
   independently-evolving PRs, not a mistake by either in isolation. Fixed to accept either shape.
6. **The same PR's `load_weights` can't match the checkpoint's vision-tokenizer weight names.**
   `ValueError: No vision/audio tensor matches checkpoint key:
   vision_tower.encoder.down.0.block.0.conv1.bias` (run `3137785`) — structurally, the checkpoint
   key *does* match what `Apertus1p5VisionTokenizerModel` (verified against
   `../verl`-adjacent transformers source) should produce; a third checkpoint/PR alignment gap,
   not obviously wrong on either side. Since this benchmark is text-only GSM8K and never calls
   `get_image_feature`/`get_audio_feature`, the vision/audio tower weights never need to be
   numerically correct, only present so the model instantiates — patched to skip an unmatched
   vision/audio key with a warning instead of raising; the language-model weight path is
   untouched.

   Points 5 and 6 were built and verified as `sitecustomize.py` runtime monkeypatches first (fast
   to iterate on — no need to hand-craft a diff against a file mid-debugging), then, once run
   `3139370` confirmed both were correct and stable, converted to a real unified-diff patch file,
   `apertus-benchmarks/patches/sglang-apertus1p5-local-fixes.patch`, applied with `patch -p2`
   immediately after point 3's PR patch (same `dist-packages/sglang`, same node, same srun) —
   fatal if it fails to apply, since (unlike point 3's `base_processor.py` hunk, which is
   context-sensitive against whatever SGLang version the image happens to have) this diff is
   against a file *this script itself* just patched into place two steps earlier, so a failure
   here means point 3's PR moved out from under the pinned SHA, which is worth stopping for. Two
   copies of this content exist by the same convention as the GLM section's
   `sitecustomize-verl-v0.9.0.py` — the checked-in `patches/` file and the script's heredoc copy —
   and must be kept in sync by hand; diff them before trusting either. The whole point of this
   conversion: a runtime monkeypatch is great for fast iteration but not the form a stable fix
   should end up in — a plain source patch is one less moving part (no `sys.meta_path` machinery,
   no PYTHONPATH staging, no import-timing dependency) and the actual behavior is just readable
   in the file.
7. **`torch.OutOfMemoryError` in the FSDP actor's `backward()`** (run `3138546`) — 89 GiB
   resident on a 95 GiB GPU before an incremental 12.94 GiB allocation. This 71.94B-param *dense*
   model, forced onto eager attention by point 2, has no flash-attention memory savings, and no
   FSDP offloading was configured. Fixed by lowering `actor.ppo_max_token_len_per_gpu` from
   verl's default (16384, sized for flash-attention's O(n) memory) to 4096, and enabling
   `actor.fsdp_config.{param,optimizer,grad}_offload: True` — confirmed via `../verl`'s actual
   config schema (`verl/workers/config/actor.py`, `engine.py`) rather than guessing at field
   names or the default value.
8. **Points 4–6 depended on the `find_spec` sitecustomize mechanism while they were still runtime
   monkeypatches, not the dead `find_module` one or an eager top-level import** — see the GLM
   section's "Known hazards" entry above for the full mechanism and evidence (it's shared code,
   and the eager-import Ray-actor-scheduling stall was actually first found *here*: runs
   `3133616`/`3134586`, both this Apertus script, both stalling identically right after "Connected
   to Ray cluster" with the driver actor never starting, before the GLM section's parallel
   investigation confirmed the same fix). This script no longer runs any sitecustomize.py at all
   as of the point-5/6 source-patch conversion — this point is kept for the history and because
   the GLM script's `sitecustomize-verl-v0.9.0.py` still depends on the same mechanism.

**Patch groups**: everything above is organized into one labeled group in the script, "Group 1:
add Apertus 1.5 support" — 1a transformers (point 1), 1b SGLang critical files (point 3, fatal),
1c SGLang `base_processor.py` (point 3, best-effort), 1d local fixes (points 5/6, fatal). All
fetched/built once (batch host or a single srun node), cached/`sbcast`'d, and applied identically
on every node, the same "fetch once, distribute, apply-or-fail" discipline as the GLM script (see
"Never fetch per-node from the internet inside the srun" above) — none of it lives only in this
file.

## Configuration audit (architecture-vs-config) — 2026-08-26

Per the repo-wide "Configuration correctness audits" process above.

- **Architecture**: Apertus-v1.5-70B is a **dense**, text+vision+audio multimodal model
  (`model_type: apertus1p5`) — confirmed dense (71.94B parameters, no expert/EP config anywhere
  in this script) in run `3189642`'s successful weight load. Not MoE, so `router_replay`/R3 (see
  the GLM section's audit entry) does not apply here.
- **Multimodal-specific config, already found and fixed — but found by crashes, not by this
  audit process** (this recipe's whole "Known hazards" numbered list above *is* the
  architecture-vs-config audit for this script, just discovered the hard way): the vision-tower
  forced `eager` attention (point 2), the SGLang generic-multimodal-loader misload (point 3), and
  `attn_implementation`/checkpoint-key mismatches (points 5–6) are all instances of "config didn't
  match what this multimodal architecture actually needs." Flagging retroactively so future
  scripts get this checked *before* a crash forces it, not instead of the existing fixes.
- **FSDP2-specific check**: this script loads the model via stock HF
  `Apertus1p5ForConditionalGeneration.from_pretrained` (no custom bridge, unlike the Megatron
  variant) — confirmed by `grep`ping the script for any `output_vocab_size`/`lm_head`/pruned-vocab
  handling: none exists, and none is needed here. HF's own modeling code already sizes `lm_head`
  to the checkpoint's real (pruned, 131072-wide) weight; there is no separate `vocab_size` field a
  bridge has to get right, unlike Megatron's single-`vocab_size` `GPTModelProvider` (see that
  section's pruned-head fix, run `3171151`). No gap found on this axis for FSDP2.
- **Not checked this pass**: whether `actor.fsdp_config.{param,optimizer,grad}_offload: True` and
  `ppo_max_token_len_per_gpu: 4096` (point 7's OOM fix) are still the *right* values now that the
  hybrid-rollout-adjacent OOM chain (run 3185487) and the FSDP2-vs-Megatron reward-collapse
  investigation are both still open — revisit once those resolve, since a memory-budget fix and a
  correctness fix can interact.

## Run log

### Run `3184895` — 2026-08-25 — SGLang PR #32979 drift breaks the patch step here too; the Megatron script's allowlist fix was never ported

- **Log**: `~/Downloads/slurm-3184895.out` (226,053 bytes). 16 nodes, `clariden`. FAILED in ~35-43s
  — died during per-node setup, well before Ray/training started.
- **Symptom**: `patch -p2 -d /usr/local/lib/python3.12/dist-packages <
  ${TRAINING_CONFIG}/sglang-apertus1p5-full.diff` (the fatal, non-best-effort patch step) failed:
  `patching file sglang/srt/managers/detokenizer_manager.py` / `Hunk #1 FAILED at 143` / `FATAL:
  sglang PR #32979 patch failed to apply`. `srun` force-terminated all 16 nodes.
- **Root cause**: identical class of bug already fixed once in the sibling Megatron script (see
  its own Run log, `3152782`): `sgl-project/sglang#32979` is open/unmerged and this script
  re-fetches its diff fresh every run with no commit pin, so its file set drifts. This script's
  own patch-categorization was still a fatal-by-default blocklist (only `base_processor.py`
  special-cased as best-effort) — the Megatron script's allowlist fix for this exact drift was
  never ported here.
- **Fix**: ported the Megatron script's allowlist verbatim — only the 4 files confirmed
  load-bearing for apertus1p5 support (`qwen3_asr.py`, `apertus.py`, both `apertus_mm.py`) are
  fatal-if-they-fail; everything else the PR touches (currently `base_processor.py` and
  `detokenizer_manager.py`) is merged into one best-effort diff applied with a `WARNING` instead
  of aborting. Verified the new comment text sits outside the single-quoted `srun bash -c '...'`
  block (the run-3149339 quoting hazard) before submitting.
- **Commit**: not committed.

### Run `3185487` — 2026-08-25 — allowlist fix confirmed working; new, undocumented OOM: rank 0 alone materializes the full un-sharded model before FSDP shards it

- **Log**: `~/Downloads/slurm-3185487.out` (417,000 bytes). 16 nodes, `clariden`. Ran 14m49s
  (889s), FAILED (exit code 15) — furthest this exact patch-categorization test needed to go.
- **Confirmed working**: the 3184895 allowlist fix worked — the 4 load-bearing files patched
  cleanly on every node; the best-effort bundle hit the same context-sensitive
  `detokenizer_manager.py`/`base_processor.py` hunk failures as before, but now only logged
  `WARNING: one or more best-effort sglang PR #32979 hunks did not apply ... — continuing` instead
  of aborting the job. The job proceeded through config load, Ray init, and into FSDP2 trainer
  model loading — through **100% weight loading** (1224/1224 shards, ~9m44s) before crashing.
  This confirms the patch-categorization fix (this run's actual purpose) is correct and complete.
- **Symptom (new, unrelated to the patch fix)**: `torch.OutOfMemoryError: CUDA out of memory.
  Tried to allocate 8.14 GiB. GPU 0 has a total capacity of 95.00 GiB of which 2.48 GiB is free.
  Including non-PyTorch memory, this process has 524.00 MiB memory in use`, in
  `WorkerDict.actor_init_model()`, on `pid=133569, ip=172.28.25.100` — the same rank that had just
  printed `Loading weights: 100%|██████████| 1224/1224` moments earlier. Immediately preceded by
  `Before FSDP, memory allocated (GB): 0.00, memory reserved (GB): 0.00, device memory used/total
  (GB): 92.52/95.00` (verl's `log_gpu_memory_usage`, only ever printed by global rank 0). Full
  traceback: `_build_fsdp_module` → `apply_fsdp2` → `torch.distributed._composable.fully_shard` →
  `_get_modules_and_states` → `_move_states_to_device` → `tensor.to(device)`.
- **Not caused by the patch-categorization fix**: confirmed by scope — this session's only
  uncommitted change to this file before submitting was the two `awk` filter blocks and the apply
  step (patch-drift fix above), none of which touch model loading, FSDP, or memory config. The
  `actor.fsdp_config.{param,optimizer,grad}_offload` / `ppo_max_token_len_per_gpu: 4096` settings
  from the run-3138546 OOM fix (see the numbered list above) are unchanged and intact.
- **Root cause — not fully resolved, but narrowed with fetched fork source**: the running FSDP2
  loading code comes from `theely/verl`'s `Fix-fsdp-model-loading-on-async` branch (`git reset
  --hard` onto this after the `v0.9.0` checkout — see the srun body), fetched fresh from GitHub
  and inspected directly (not the local `../verl`, which is `v0.8.0` and doesn't have this fork's
  code at all). By design (`transformer_impl.py`, comment: "For fsdp2, only rank 0 loads weights
  from disk; all others receive via broadcast_from_rank0"), **global rank 0 alone** calls
  `auto_class.from_pretrained(...)` with real (non-meta) tensors — every other rank builds an
  empty/meta-tensor model structure only. `log_gpu_memory_usage`'s `device memory used/total`
  reads `torch.cuda.mem_get_info()` — physical-GPU-wide usage, not this process's own allocator
  stats (which is why `memory allocated`/`memory reserved` read 0.00 while `device ... used` reads
  92.52 — two different measurements). So rank 0's physical GPU already had ~92.52 GiB in use from
  *something*, before rank 0's own FSDP-triggered `_move_states_to_device` call (its first GPU
  write) even started, at a point where nothing in this trace has yet asked to place the model on
  GPU. Checked whether the fork branch itself regressed since the last confirmed-working run
  (3139370, 2026-08-21): via the GitHub API, the branch's most recent fsdp2-loading-relevant
  commits are `9708019a` (2026-07-10) and `0acefce7` (2026-06-25) — both well before 3139370, so
  the branch has not changed; this rules out a fork-branch regression as the explanation. Two
  remaining, undistinguished hypotheses: (a) something else (most plausibly a standalone SGLang
  rollout replica, given `gpu_memory_utilization: 0.75` × 95 GiB ≈ 71 GiB plus engine/cudagraph
  overhead is in the right range) landed on the same physical node/GPU as trainer global-rank-0 —
  an oversubscription/placement bug in this script's `fully_async` node split, analogous to the
  GLM/Megatron "hybrid rollout squats on trainer GPUs" hazard but never previously documented for
  this FSDP2 recipe; or (b) a leftover/orphaned process from an earlier job left GPU memory
  unreleased on this specific (possibly reused) physical node — a cluster-side flake unrelated to
  this script. Not distinguished from the log alone; no SGLang-server-startup log line appears
  anywhere before the crash, which is some (not conclusive) evidence against (a), since it implies
  the rollout replicas hadn't started serving yet — though placement/GPU reservation could still
  precede the log lines that would show that.
- **Why more nodes is not a safe blind next step**: `ppo_mini_batch_size: 48` is currently exactly
  equal to `TRAINING_NNODES(12) * 4 = 48` training GPUs (1 sample/GPU, the documented minimum,
  same shape as the Megatron/GLM chain's `ppo_mini_batch_size >= dp_size` invariant). Raising
  total node count without recomputing this would make `ppo_mini_batch_size` smaller than the new
  GPU count (e.g. 24 nodes → 18 training nodes → 72 GPUs > 48) and hit that assertion immediately
  — trading this OOM for a new, avoidable failure. Also: a single-GPU pre-FSDP squat on rank 0
  isn't obviously fixed by adding *more* GPUs elsewhere in the cluster either way, since the squat
  (whatever it is) is local to rank 0's own physical GPU.
- **Fix**: none yet — this needs one more data point before deciding on a fix. Next run:
  resubmit this exact, unmodified 16-node config to determine whether the OOM recurs identically
  (systematic — start pointing at the placement/oversubscription hypothesis in earnest, e.g. by
  checking Ray's actual node/GPU assignment for the rollout replicas vs. rank 0) or does not
  recur (one-off node/GPU squat, matching this repo's own precedent for exactly this kind of call
  — see the Megatron/GLM section's `3152802`/`3171176` TP=32 SGLang hang, retried unmodified and
  confirmed a one-off).
- **Commit**: not committed.

# Training Apertus v1.5 on verl, Megatron trainer (`rl-bench-apertus-v1.5-70B-sglang-megatron-async.sh`)

Apertus-v1.5-70B GRPO on GSM8K, CSCS Alps, 16 nodes × 4 GH200 (`clariden`). Script:
`Alps-Images/apps/verl/apertus-benchmarks/rl-bench-apertus-v1.5-70B-sglang-megatron-async.sh`.
Sibling of the FSDP2 recipe above — reuses its "Group 1: add Apertus 1.5 support" patches
verbatim (transformers wheel, SGLang PR #32979 + local fixes) — but swaps the FSDP2 trainer for
Megatron, which needs its own fork: `wqwqazwsxedc/Megatron-LM`/`Megatron-Bridge` (branch
`apertus`), cloned once into Lustre and `pip install -e`'d on every node ("Group 2: Megatron
support" in the script). Untested as of run `3141796`.

## Configuration audit (architecture-vs-config) — 2026-08-26

Per the repo-wide "Configuration correctness audits" process above. Same dense (not MoE)
architecture as the FSDP2 sibling, so `router_replay`/R3 doesn't apply here either. Two
architecture-vs-config gaps specific to routing this dense multimodal model through a Megatron
`GPTModel` (rather than HF's own class, as the FSDP2 sibling does) were already found — both by
crashes, both now fixed, both retroactively exactly what this audit category exists to catch
earlier next time:

- **Pruned LM head** (run `3171151`/`3171218`): `Apertus1p5TextConfig`'s `output_vocab_size`
  (131072) is narrower than the embedding `vocab_size` (266752) — a real architectural trait of
  this checkpoint — but Megatron-core's `GPTModelProvider` has only one `vocab_size` for both
  input embedding and output head. `Apertus1p5Bridge.provider_bridge()` now sizes Megatron's
  `vocab_size` to `output_vocab_size` and truncates the input embedding table to match
  (`_TruncatedVocabEmbeddingMapping` in `apertus1p5_bridge.py`) — see that run's entry for the
  full reasoning and the acknowledged remaining risk (raises rather than silently mismapping if a
  future checkpoint isn't actually pruned the way this one is).
- **`vision_model=True` forced on every forward regardless of batch content** (found and fixed
  same day as run `3183472`): `verl/workers/engine/megatron/transformer_impl.py` derives
  `vision_model` from `hasattr(hf_config, "vision_config")` — true for any vision-capable
  architecture, not from whether the current batch has image content — which corrupted THD
  packed-sequence tensors for this text-only GSM8K recipe. Patched to force `vision_model=False`
  for this specifically-text-only workload; a genuinely multimodal use of this model would need
  the opposite fix. See that run's entry for the full trace.

Nothing new found this pass beyond what's already fixed above; both fixes remain unverified by an
actual cluster run as of the last entry in this section's run log.

### Run `3141796` — 2026-08-21 — same qwen3_asr collision as the SGLang side, now on Megatron-Bridge

- **Log**: `~/Downloads/slurm-3141796.out`. 16 nodes, `clariden`. RUNNING → FAILED in ~5m18s
  (exit code 15), died during init — never reached trainer weight-loading (phase 1 only).
- **Symptom**: `ValueError: 'qwen3_asr' is already used by a Transformers config, pick another
  name.` from `transformers/models/auto/configuration_auto.py:140`, on every Megatron worker
  rank, inside `WorkerDict.actor_init_model()` → `engine.initialize()` → `_build_tf_config()` →
  `verl.models.mcore.bridge.AutoBridge` → the `megatron.bridge` import chain →
  `megatron.bridge.models.qwen3_asr.hf_qwen3_asr.__init__` →
  `AutoConfig.register("qwen3_asr", Qwen3ASRConfig)`.
- **Root cause**: same class of bug as the FSDP2/SGLang section's point 4 — the swiss-ai
  transformers fork already registers `qwen3_asr`, and this vendored copy calls
  `AutoConfig.register` with the default `exist_ok=False`. A different vendored copy this time:
  `wqwqazwsxedc/Megatron-Bridge`'s own
  `src/megatron/bridge/models/qwen3_asr/hf_qwen3_asr/__init__.py` (confirmed on GitHub, `apertus`
  branch) — unrelated to SGLang PR #32979's own copy, whose `exist_ok=True` fix (point 4 above)
  does not cover this one.
- **Fix**: patch that one line to `exist_ok=True` directly in the shared Megatron-Bridge clone
  (`${MEGATRON_BRIDGE_DIR}`) right after it's cloned, on the batch host — it's a single Lustre
  checkout shared by every node via `pip install -e`, so (unlike the SGLang `dist-packages` copy)
  one `sed` on the batch host is enough; no `sbcast`/per-node apply needed. `grep`-guarded so
  reruns against an already-cloned, already-patched checkout skip re-patching. Unverified — needs
  a run.
- **Commit**: not committed.

### Run `3144663` — 2026-08-21 — same file, next line: `AutoModel.register` collides too

- **Log**: `~/Downloads/slurm-3144663.out` (664,670 bytes). 16 nodes, `clariden`. RUNNING → FAILED
  in ~4m12s (exit code 15) — essentially the same fast-failure window as 3141796. Phase 1 only,
  crashed in `actor_init_model()` before trainer weight loading.
- **What worked**: confirms the "Group 2: Megatron support" wiring is sound as a first attempt —
  Megatron-LM/Megatron-Bridge apertus-fork clone + `pip install -e` succeeded on all 16 nodes
  (`megatron.core.__file__` confirmed pointing at the forked checkout); swiss-ai transformers
  wheel + safetensors install clean; SGLang PR #32979 + local fixes applied cleanly (only the
  already-known non-fatal `base_processor.py` hunk skipped, as expected); and the 3141796
  `AutoConfig.register` fix worked — that exact error did not recur.
- **Symptom**: `ValueError: '<class '...Qwen3ASRConfig'>' is already used by a Transformers
  model.` — the *next* registration call in the same file,
  `AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)`, one line below the
  `AutoConfig.register` call that 3141796 fixed.
- **Root cause**: the 3141796 fix only patched the first of three back-to-back `Auto*.register`
  calls in `hf_qwen3_asr/__init__.py`; the other two (`AutoModel.register`, then presumably
  `AutoProcessor.register` next) have the identical default-`exist_ok=False` collision against the
  swiss-ai transformers fork's own registration. Confirmed from transformers source
  (`models/auto/auto_factory.py`'s `_LazyAutoMapping.register`) that the collision check matches
  by `config_class.__name__` string, not object identity — so Megatron-Bridge's own distinct
  `Qwen3ASRConfig` class object still collides with the swiss-ai fork's differently-instantiated
  same-named class.
- **Fix**: extended the same `sed` block to patch all three `Auto*.register` calls
  (`AutoConfig`, `AutoModel`, `AutoProcessor`) to `exist_ok=True` in one pass, all
  grep-verified individually before continuing. Unverified — needs a run.
- **Commit**: not committed.

### Run `3149242` — 2026-08-22 — the three-call fix was correct but the idempotency guard undid it on a stale checkout

- **Log**: `~/Downloads/slurm-3149242.out` (3543 lines). 16 nodes, `clariden`. RUNNING → FAILED at
  ~242-311s (exit code 15). Got further than 3141796/3144663: cleared the verl v0.9.0 checkout,
  swiss-ai transformers/safetensors wheel install, and SGLang PR #32979 + local-fixes patching,
  reaching phase 1 (`[ASYNC MAIN] Initializing model and tokenizer...`,
  `FullyAsyncTrainer.init_workers()` started) before crashing in `WorkerDict.actor_init_model()`.
- **Symptom**: the *same* `AutoModel.register` collision as 3144663 —
  `ValueError: '<class '...Qwen3ASRConfig'>' is already used by a Transformers model.` — recurred
  even though the 3144663 fix (patching all three `Auto*.register` calls) had already landed in
  the script before this run.
- **Root cause**: `TRAINING_HOME` is persistent Lustre storage, reused across all three attempts,
  and `MEGATRON_BRIDGE_DIR` is only cloned once ("already cloned, skipping" — see the clone
  block). 3141796 left behind a checkout with only `AutoConfig.register` patched. The
  idempotency guard added in that same run only ever checked *that one* line
  (`if grep -q '...AutoConfig.register(...exist_ok=True)...'; then skip; else <patch all three>
  fi`) to decide whether to skip the whole block — so on this run it saw the pre-existing
  `AutoConfig` patch from 3141796's checkout, concluded "already patched," and skipped patching
  `AutoModel`/`AutoProcessor` entirely, even though the 3144663 fix for those two had never
  actually been applied to this specific on-disk checkout (only ever to a Megatron-Bridge clone
  that had since been re-cloned or was checked against a different one). Net effect: a
  short-circuiting all-or-nothing guard silently re-introduced an already-fixed bug.
- **Fix**: dropped the single-line short-circuit guard entirely. The `sed` block now always runs;
  each of its three patterns is anchored on the un-patched (no `exist_ok=True`) form of its own
  call, so a pattern simply doesn't match (and is a no-op) if that specific call was already
  patched — correct whether zero, one, two, or all three calls were already fixed on a given
  checkout, with no shared guard to go stale. Still grep-verifies all three afterward and is
  fatal if any didn't apply. General lesson: an idempotency check spanning multiple independent
  edits must confirm *each* edit individually, not use one edit as a proxy for all of them.
  Unverified — needs a run.
- **Commit**: not committed.

### Run `3149726` — 2026-08-22 — qwen3_asr fix confirmed working; new failure, much earlier: concurrent editable-install race

- **Log**: `~/Downloads/slurm-3149726.out` (320,339 bytes). 16 nodes, `clariden`. RUNNING → FAILED
  in **~53s** — far earlier than any prior attempt (all four before this died 4-5 min in, during
  `actor_init_model()`). Never reached phase 1 (config load) — died during environment setup,
  before Python/Ray even started.
- **Good news**: the qwen3_asr `Auto*.register` fix from 3144663/3149242 fired correctly this
  time — `Patched (or confirmed already-patched) qwen3_asr AutoConfig/AutoModel/AutoProcessor.
  register` appears in the log. That bug is closed; this is a new, earlier-stage problem.
- **Symptom**: `pip install --no-build-isolation -e ${MEGATRON_LM_DIR}` failed on at least one
  node (task 6, `nid007302`, exit 1):
  ```
  copying /tmp/pip-tmp-3149726/.../helpers_cpp.cpython-312-aarch64-linux-gnu.so -> megatron/core/datasets
  error: [Errno 2] No such file or directory
  ...
  FATAL: Megatron-LM install failed
  ```
  The script's own `|| exit 1` guard tripped on that node, and `srun` then force-terminated the
  other 15 tasks.
- **Root cause**: `${MEGATRON_LM_DIR}` is a single shared Lustre checkout (cloned once), and all
  16 nodes ran `pip install --no-build-isolation -e ${MEGATRON_LM_DIR}` concurrently against it.
  Unlike a plain metadata install, this compiles megatron-core's C++ `datasets` helpers extension
  in place inside the shared source tree — 16 concurrent compiles racing on the same build/output
  paths collided (one node's build step deleted/moved a file another node's copy step still
  needed).
- **Fix**: build `megatron-core` and `megatron-bridge` into wheels once, on a single node (same
  "Group 2" clone-once srun, run after the qwen3_asr source patch so the patched source gets
  packaged), cache them under `${TRAINING_HOME}/wheels`. The main 16-node srun now does a plain
  `pip install <wheel>` (metadata-copy only, no compilation) instead of `pip install -e
  <shared-dir>` — same "build once, install everywhere" treatment already used for the swiss-ai
  transformers wheel. Package names (`megatron-core`, `megatron-bridge`) confirmed from each
  fork's `pyproject.toml` rather than guessed. Unverified — needs a run.
- **Commit**: not committed.

### Run `3149776` — 2026-08-22 — wheel-race fix confirmed working; furthest ever reached; new failure is a real upstream gap, not a quick patch

- **Log**: `~/Downloads/slurm-3149776.out`. 16 nodes, `clariden`. RUNNING → FAILED in ~6 min —
  furthest this script variant has ever reached.
- **Confirmed working**: the 3149726 wheel-build fix worked cleanly —
  `megatron-core-0.18.0+60b5c9588` and `megatron-bridge-0.5.0+032c0740` both built successfully
  in the single-node build step (no compile race), and the main 16-node srun's plain wheel
  installs succeeded on every node (past the exact point 3149726 died at). `transformers:
  5.15.0.dev0` and `megatron.core.__file__` both printed cleanly. SGLang PR #32979 + local fixes
  applied clean (only the expected best-effort `base_processor.py` warning). **Phase 1 reached**:
  Ray cluster connected, `FullyAsyncTaskRunner` started, worker mapping and trainer creation
  began — no eager-import Ray stall, no OOM. All four earlier bug classes (qwen3_asr ×3,
  wheel-build race) are now closed for this script.
- **Symptom**: during `trainer.init_workers()` → `actor_init_model()` →
  `megatron.bridge.models.conversion.auto_bridge.AutoBridge.from_hf_pretrained` → `_validate_config`:
  ```
  ValueError:
  ✗ Model architecture 'Apertus1p5ForConditionalGeneration' is not yet supported
  Currently supported architectures: ApertusForCausalLM, BailingMoeV2ForCausalLM, ... (no Apertus1p5 variant)
  ```
- **Root cause — genuine gap, not a registration bug**: confirmed by reading
  `wqwqazwsxedc/Megatron-Bridge`'s `apertus_bridge.py` (GitHub, `apertus` branch) directly: its
  `ApertusBridge` is registered with `@MegatronModelBridge.register_bridge(source=
  ApertusForCausalLM, ...)` — the older, text-only, non-wrapped architecture — and there is no
  `apertus1p5` bridge anywhere in the fork's `models/` directory (only `apertus/`). Unlike the
  qwen3_asr bugs (an existing registration call just missing `exist_ok=True`), this fork has
  simply never implemented Apertus1p5 support at all — this is the Megatron-Bridge-side
  equivalent of the FSDP2 section's "no reference recipe" situation (see that section's header
  note), except there the gap was fillable with a source patch against an existing upstream PR;
  here there is no upstream PR to patch, no existing bridge to extend.
  Checked feasibility of writing one: fetched swiss-ai/transformers'
  `modeling_apertus1p5.py` (same commit pinned by `SWISS_AI_TRANSFORMERS_SHA` in this script) and
  confirmed `Apertus1p5TextModel`/`Apertus1p5TextDecoderLayer`/`Apertus1p5TextAttention`/
  `Apertus1p5TextMLP` are architecturally identical to plain Apertus (xielu MLP activation, q/k
  RMSNorm, no attention bias, same rope handling) — so a bridge is plausible to write by adapting
  `ApertusBridge`: source class `Apertus1p5ForConditionalGeneration`, hyperparameters read from
  `hf_pretrained.config.text_config` instead of the top-level config (Apertus1p5 nests them one
  level down), and every decoder-layer/embedding HF key path re-prefixed from `model.*` to
  `model.language_model.*` (confirmed via the class hierarchy:
  `Apertus1p5ForConditionalGeneration.model` is an `Apertus1p5Model`, whose `.language_model` is
  the actual `Apertus1p5TextModel`) — `lm_head.weight` itself stays unprefixed and tied to
  `model.language_model.embed_tokens.weight` per `_tied_weights_keys`, same as plain Apertus's
  `output_layer.weight`/`lm_head.weight` mapping. Not yet attempted — this is real new-code
  implementation work (a full `MegatronModelBridge` subclass), not a one-line fix, and unlike the
  qwen3_asr/wheel fixes it cannot be smoke-tested locally (needs a real Megatron init to validate
  the mapping registry), so it was intentionally not attempted without checking in first.
- **Fix**: user asked to write the bridge (wants Megatron parallelism, not just the working FSDP2
  fallback). New file `apertus-benchmarks/patches/apertus1p5_bridge.py` (checked-in copy) defines
  `Apertus1p5Bridge`, adapted directly from `apertus_bridge.py`'s `ApertusBridge` — reuses its
  `MCoreXIELU`/`get_apertus_decoder_block_spec` by import (not duplicated), same
  rope/attention_bias/hidden_act validation, only two things change: `provider_bridge` reads
  `hf_pretrained.config.text_config` (via a tiny local `_TextConfigOnlyShim` exposing just
  `.config`, since `Apertus1p5Config` nests all LM hyperparameters one level down, alongside
  unrelated `vision_config`/`audio_config`) instead of the top-level config, and every HF-side
  weight key in `mapping_registry` is re-prefixed `model.language_model.*` instead of `model.*`
  (confirmed against `modeling_apertus1p5.py`: `Apertus1p5ForConditionalGeneration.model` is an
  `Apertus1p5Model` whose `.language_model` is the actual `Apertus1p5TextModel` holding
  `embed_tokens`/`layers`/`norm`; `lm_head` itself stays unprefixed at the top level, tied to
  `model.language_model.embed_tokens.weight` per `_tied_weights_keys`, mirroring plain Apertus's
  own `output_layer.weight`/`lm_head.weight` tie). Vision/audio tower weights are deliberately
  left unmapped — target is plain `GPTModel` (no parameter slots for them anyway), this benchmark
  is text-only GSM8K and never needs them numerically correct (same reasoning as the FSDP2/SGLang
  section's point 6), and no strict/leftover-key check exists anywhere in `model_bridge.py`'s
  weight-load path to object. One real unresolved risk, documented in the file's own docstring
  and guarded with an explicit `raise` rather than silently mismapping: `Apertus1p5TextConfig`
  supports a *pruned* LM head (`output_vocab_size` narrower than the embedding `vocab_size`), which
  Megatron-core's standard `GPTModelProvider` cannot represent (one `vocab_size` sizes both
  tables) — not verified against the real checkpoint since `swiss-ai/Apertus-v1.5-70B`'s
  `config.json` sits behind a gated HF repo unreachable from where this was authored; if the
  checkpoint does turn out pruned, `provider_bridge` raises a clear error instead of training on
  wrong shapes. Wired into the script right after the qwen3_asr source patch (same shared Lustre
  checkout, single batch-host write, same two-copies-kept-in-sync-by-hand convention) and *before*
  the wheel-build step so the new file gets packaged — also added a `MEGATRON_WHEEL_BUILD_VERSION`
  marker gate to that step (`build.version` file next to the cached wheels) so this source change
  actually forces a rebuild, rather than repeating the exact "stale cache masks a real source
  change" mistake from run 3149242's qwen3_asr guard. Entirely unverified — no `provider_bridge`/
  `mapping_registry` implementation in this bridge library can be meaningfully checked outside a
  real Megatron init, so this needs a cluster run.
- **Commit**: not committed.

### Run `3152782` — 2026-08-22 — new regression, unrelated to the bridge: the open SGLang PR grew a context-sensitive hunk

- **Log**: `~/Downloads/slurm-3152782.out` (1510 lines). 16 nodes, `clariden`. RUNNING → FAILED in
  ~101s — died on all 16 nodes during the SGLang PR #32979 patch-apply step, before ever reaching
  the new `Apertus1p5Bridge` (never got to test point 3, `AutoBridge.from_hf_pretrained`).
- **Good news, unrelated to the failure**: both new setup steps for the bridge fix confirmed
  working — "Added Apertus1p5Bridge ... to Megatron-Bridge checkout" printed, and the wheel-build
  step logged "build version v2-apertus1p5-bridge" (a real rebuild, not a stale-cache skip) — the
  `MEGATRON_WHEEL_BUILD_VERSION` marker gate from the 3149776 fix write-up worked as intended.
- **Symptom**: `patch -p2 -d /usr/local/lib/python3.12/dist-packages < sglang-apertus1p5.diff`
  failed identically on all 16 nodes:
  ```
  patching file sglang/srt/managers/detokenizer_manager.py
  Hunk #1 FAILED at 143.
  1 out of 2 hunks FAILED -- saving rejects to file sglang/srt/managers/detokenizer_manager.py.rej
  ...
  FATAL: sglang PR #32979 patch failed to apply
  ```
- **Root cause**: `sgl-project/sglang#32979` is an open, unmerged PR that is still being actively
  pushed to upstream (confirmed via the GitHub API: `updated_at` for the PR was essentially
  concurrent with this run), and the script always re-fetches its diff fresh on every submission
  (`curl .../pull/32979.diff`, no commit pin, no caching) — deliberate, since pinning an unmerged
  PR to a stale SHA would just trade this failure mode for silently missing upstream fixes, but it
  does mean the file set is not stable between runs. Between the 3149776 submission and this one
  the PR grew from 6 changed files to 10: two new test files, a `docs/docs/...` file (not
  `docs_new/`, so unaffected by the existing docs exclusion), and — the actual break — a new hunk
  in `python/sglang/srt/managers/detokenizer_manager.py` adding an `apertus2509` tool-call-parser
  output trim. That hunk is context-sensitive against whatever SGLang version the image happens to
  ship, exactly the same class of issue already known and already tolerated for
  `base_processor.py` — but the script's filtering was a *blocklist* (`keep` everything under
  `python/sglang/` except `base_processor.py`+`test/`+`docs_new/`), so this brand-new file landed
  in the fatal-if-it-fails bucket by default and took the whole job down. Confirmed the new hunk
  itself is inert for this benchmark regardless: it only fires when
  `server_args.tool_call_parser == "apertus2509"`, which this GSM8K recipe never sets — same "safe
  to skip" situation as `base_processor.py`'s audio-only hunk.
- **Fix**: flipped the categorization from a blocklist to an allowlist. Only the files actually
  confirmed load-bearing for the apertus1p5 fix (`qwen3_asr.py`, `apertus.py`, and both
  `apertus_mm.py` files — point 3 above) are fatal-if-they-fail; every other file the PR touches
  under `python/sglang/` (currently `base_processor.py` and `detokenizer_manager.py`) is
  extracted into one combined best-effort diff and applied with a warning instead of aborting;
  `test/` and `docs/` (both the old `docs_new/` guess and the actual `docs/docs/...` path this PR
  uses) stay excluded entirely. This is meant to self-adapt to future drift in this still-evolving
  PR — a new unrelated file added upstream now defaults to best-effort instead of fatal, so this
  exact failure mode (one new file breaks the whole fatal bucket) should not recur even as the PR
  keeps changing shape. Unverified — needs a run; the new `Apertus1p5Bridge` from the previous
  entry is still completely untested since this failure pre-empted it.
- **Commit**: not committed.

### Run `3171151` — 2026-08-24 — SGLang allowlist fix confirmed working; new bridge registered correctly and hit exactly the anticipated pruned-head gap

- **Log**: `~/Downloads/slurm-3171151.out` (514,854 bytes). 16 nodes, `clariden`. RUNNING → FAILED
  after ~3.8 min.
- **Test 1 passed**: the allowlist fix from 3152782 worked as designed. All four load-bearing
  files (`qwen3_asr.py`, `apertus.py`, both `apertus_mm.py`) patched cleanly; the best-effort
  bundle hit non-fatal warnings for `detokenizer_manager.py` and `base_processor.py` (both
  context-sensitive against the image's SGLang version, both inert for this benchmark) without
  aborting the job. Confirms this categorization is now resilient to the PR's continued drift.
- **Test 2 — real signal, not a registration bug**: `AutoBridge.from_hf_pretrained` no longer
  raised "not yet supported" — the new `Apertus1p5Bridge` registration and dispatch worked
  correctly. It got as far as its own `provider_bridge()` and hit exactly the deliberate guard
  written into that method:
  ```
  ValueError: Apertus1p5 checkpoint uses a pruned LM head (output_vocab_size=131072 != vocab_size=266752).
  ```
  This confirms empirically (for the first time — the real `config.json` was never reachable
  during authoring, see the previous entry) that `swiss-ai/Apertus-v1.5-70B` really does use a
  pruned head: 131072 real/generatable text ids out of a 266752-wide extended vocabulary that
  also covers the visual/audio token ranges.
- **Fix**: implemented proper pruned-head support instead of raising. Since this benchmark
  (text-only GSM8K GRPO) never produces or consumes a token id outside `[0, output_vocab_size)`
  — confirmed via `output_vocab_size`'s own docstring in `configuration_apertus1p5.py`, which
  states the retained ids are exactly `0..output_vocab_size - 1` — it's safe to size Megatron's
  single `vocab_size` to `output_vocab_size` (131072) rather than the full extended `vocab_size`.
  `provider_bridge` now sets `provider.vocab_size = output_vocab_size` and recomputes
  `provider.make_vocab_size_divisible_by` from that narrower value (this factor is itself derived
  from the vocab size, per `MegatronModelBridge.make_vocab_size_divisible_by`'s docstring, so
  reusing the value computed for the full vocab_size would have been wrong). The checkpoint's
  `lm_head.weight` is already physically `(131072, hidden)` — a real pruned `nn.Linear`, not a
  view — so `output_layer.weight` needs no transform. Only the *input* embedding table,
  `model.language_model.embed_tokens.weight` (physically `(266752, hidden)`, since input ids can
  span the full extended range even when output can't), needs to shrink to match; new
  `_TruncatedVocabEmbeddingMapping(AutoMapping)` in `apertus1p5_bridge.py` truncates it to its
  first `output_vocab_size` rows on `hf_to_megatron` before delegating to the normal
  `VocabParallelEmbedding` sharding logic, and best-effort zero-pads back out on the reverse
  direction (for whatever HF-export path might exercise it — those padded rows have no real
  multimodal embedding, same acknowledged caveat as `megatron_to_hf_config`). Needed
  `mapping_registry(self)` to read `self.hf_config.text_config` directly (it takes no
  `hf_pretrained` argument, unlike `provider_bridge`) — confirmed from `auto_bridge.py` that
  `bridge.hf_config` is always populated by the framework's own dispatch machinery
  (`_get_model_bridge_impl`) before `mapping_registry()` can run, and that this is the ordinary,
  only way any bridge's argument-less `mapping_registry()` can access config context, not a
  private workaround. Checked-in patch file and the script's heredoc copy verified byte-identical.
  Unverified — needs a run.
- **Commit**: not committed.

### Run `3171218` — 2026-08-24 — pruned-head fix never actually ran: forgot to bump the wheel cache version

- **Log**: `~/Downloads/slurm-3171218.out` (551,369 bytes). 16 nodes, `clariden`. Reached phase 1
  cleanly, then failed with the *exact same* `ValueError: Apertus1p5 checkpoint uses a pruned LM
  head (output_vocab_size=131072 != vocab_size=266752)...` as 3171151 — i.e. the fix from that
  entry appeared not to have any effect at all.
- **Root cause**: self-inflicted repeat of the exact class of bug the `MEGATRON_WHEEL_BUILD_VERSION`
  marker exists to prevent (see run 3149726/3149776's entries). The pruned-head fix was added to
  `apertus1p5_bridge.py` without bumping `MEGATRON_WHEEL_BUILD_VERSION` (left at
  `"v2-apertus1p5-bridge"`, unchanged from the *previous* fix). Lustre's `${TRAINING_HOME}/wheels`
  still had a `build.version` file reading `v2-apertus1p5-bridge` from run 3171151's build — since
  the marker matched, the wheel-build step's cache check saw "already present and up to date" and
  skipped rebuilding, installing the stale pre-fix wheel again. The new embedding-truncation code
  never actually executed on a cluster.
- **Fix**: bumped `MEGATRON_WHEEL_BUILD_VERSION` to `"v3-apertus1p5-bridge-pruned-vocab"`. General
  lesson (already stated once in this log and now demonstrated again by ignoring it): every source
  change to anything packaged into these cached wheels must be paired with a version bump in the
  same edit, not treated as a separate step to remember later.
- **Commit**: not committed.

### Run `3171309` — 2026-08-24 — pruned-head fix confirmed working; new failure, one layer deeper: missing CUDA xielu extension

- **Log**: `~/Downloads/slurm-3171309.out` (550,073 bytes). 16 nodes, `clariden`. Failed after
  ~4.5 min of actual execution — furthest point reached yet for this script.
- **Confirmed working**: the wheel rebuild fired correctly this time ("build version
  v3-apertus1p5-bridge-pruned-vocab", not a skip), and — the actual point of this run —
  **no repeat of the pruned-LM-head `ValueError` or the "not yet supported" error.** The
  `Apertus1p5Bridge.provider_bridge()` vocab-size fix from run 3171151 is confirmed correct.
  `actor_init_model()` proceeded past `AutoBridge.from_hf_pretrained` entirely, into actual
  Megatron `TransformerLayer` construction — new territory.
- **Symptom**: `RuntimeError: CUDA xIELU is required. Install rubber-duck-debug/xielu.`, raised
  from `megatron/bridge/models/apertus/apertus_bridge.py:33`, inside `MCoreXIELU.__init__`
  (imported and reused as-is by the new `Apertus1p5Bridge` — see that file's own docstring),
  while Megatron builds a `TransformerLayer`'s MLP.
- **Root cause**: `MCoreXIELU` (Megatron-Bridge's own wrapper, distinct from
  `transformers.activations.XIELUActivation`, which the FSDP2 script uses directly and which has
  a pure-PyTorch fallback when no CUDA kernel is available) hard-requires the CUDA xielu op and
  has no such fallback. The optional `xielu` PyPI/GitHub package
  (`github.com/rubber-duck-debug/xielu`, a fast CUDA implementation of the activation from
  arXiv:2411.13010) is not vendored in the image and nothing in this script installed it — every
  earlier run died upstream of this line (qwen3_asr collisions, then the pruned-head check), so
  this is the first run to ever reach the code path that needs it.
- **Fix**: build-once-install-everywhere, same pattern as the other CUDA-extension wheels in this
  script. Confirmed via the package's own `setup.py`/README: `pip install . --no-build-isolation
  --no-deps` with `CUDA_HOME` set, using CMake (a real compiled build, not a pure wheel download)
  producing an arch/os-tagged-but-python-version-independent wheel (`universal_wheel` override in
  its `setup.py`). Built once on a single node (fetch commit `2a55f6b9`'s tarball, `pip wheel
  --no-build-isolation --no-deps`), cached under the same `${TRAINING_HOME}/wheels`, installed on
  all 16 nodes with a plain `pip install --no-deps`. Deliberately given its own commit-SHA cache
  marker (`xielu.sha`), independent of `MEGATRON_WHEEL_BUILD_VERSION` — direct lesson from run
  3171218: coupling unrelated changes to one shared cache-version string is exactly what caused
  that repeat. Unverified — needs a run.
- **Commit**: not committed.

### Run `3171564` — 2026-08-24 — xielu fix confirmed working; furthest point yet; new failure: CUDA OOM in Megatron's DDP grad buffer

- **Log**: `~/Downloads/slurm-3171564.out` (536 KB). 16 nodes, `clariden`. Failed after ~7.6 min —
  furthest point this script has ever reached.
- **Confirmed working**: the xielu wheel built and installed cleanly on all 16 nodes, and was
  actually exercised — `[transformers] Using experimental xIELU CUDA. Enabled torch._dynamo for
  xIELU CUDA.` fired with no `CUDA xIELU is required` error. `actor_init_model()` proceeded past
  `AutoBridge`/bridge dispatch, past the pruned-head vocab handling, into real Megatron
  `TransformerLayer` construction — every previously-diagnosed bug in this chain (qwen3_asr ×3,
  wheel-build race, pruned LM head, missing xielu) is now closed.
- **Symptom**:
  ```
  torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 32.88 GiB. GPU 0 has a total
  capacity of 95.00 GiB of which 28.56 GiB is free. Including non-PyTorch memory, this process
  has 66.43 GiB memory in use.
  ```
  on all 16 nodes, inside `megatron.core.distributed.distributed_data_parallel
  .DistributedDataParallel.__init__` → `_ParamAndGradBuffer.__init__` → `self.grad_data =
  torch.zeros(...)` (`param_and_grad_buffer.py:1122`), called from
  `WorkerDict.actor_init_model()` → `engine_workers.py:590 init_model` →
  `transformer_impl.py:399 initialize()` → `_build_megatron_module` →
  `megatron.bridge.models.model_provider.provide_distributed_model` → `get_model` → `_ddp_wrap`.
- **Root cause**: at `tensor_model_parallel_size: 4`, each GPU holds roughly a 70B/4 ≈
  17.5B-parameter shard — already 66.43 GiB resident (model weights plus whatever `param_offload`/
  `grad_offload`/`optimizer_offload` haven't moved to CPU yet at this point in init) before the
  DDP grad buffer's own allocation is even attempted, leaving only 28.56 GiB free for a 32.88 GiB
  buffer. Not related to xielu or any of the earlier bugs — this is the first run to ever reach
  real Megatron model construction at all.
- **Fix**: raised `tensor_model_parallel_size` from 4 to 8 in both `actor.megatron` and
  `ref.megatron` (rollout's own `tensor_model_parallel_size: 4`, a separate SGLang-side setting,
  left untouched) — roughly halves the per-GPU parameter/gradient shard. Deliberately did *not*
  also increase node count this run despite being invited to: 12 training nodes × 4 GPUs = 48
  GPUs, and 48/8 = 6 DP replicas divides `ppo_mini_batch_size: 48` cleanly (48/6=8) — verified
  arithmetic. Increasing node count changes `ROLLOUT_NNODES`/`TRAINING_NNODES` via the script's
  0.25 split and would need re-deriving this same divisibility from scratch (e.g. 24 nodes gives
  18 training nodes → 72 GPUs → 9 DP replicas, and 48 is *not* divisible by 9 — would need
  `ppo_mini_batch_size` changed too). Isolated this run to testing only the TP fix rather than
  guessing at a new node-count/mini-batch combination blind; node count is an easy follow-up once
  TP=8 is confirmed sufficient (or insufficient). Unverified — needs a run.
- **Commit**: not committed.

### Run `3171930` — 2026-08-24 — new failure upstream of the OOM fix: same Lustre filelock hazard as the GLM script, different call site

- **Log**: `~/Downloads/slurm-3171930.out` (519,163 bytes). 16 nodes, `clariden`. Failed after
  239s (~4 min) — during `actor_init_model()`, but *before* the point 3171564 reached, so the
  TP=8 fix for that run's DDP grad-buffer OOM was never actually exercised here.
- **Symptom**: `ValueError: Failed to load configuration from
  .../models/Apertus-v1.5-70B after 4 attempts ... Last error: [Errno 116] Stale file handle`,
  raised in `megatron/bridge/models/hf_pretrained/safe_config_loader.py:134`, chained from
  `filelock/_unix.py:63` (`fcntl.flock(..., LOCK_NB)` → `OSError: [Errno 116]`). Call path:
  `WorkerDict.actor_init_model()` → `engine.initialize()` → `_build_tf_config()` →
  `AutoBridge.from_hf_pretrained()`.
- **Root cause**: the exact same "Lustre does not support `flock` in-container" hazard already
  documented and already fixed once in this repo — see the GLM script's Known hazards entry
  ("Lustre file locking") — but at a *different* call site: megatron-bridge's own
  `safe_config_loader`, not the `AutoConfig.from_pretrained`/filelock site the GLM script's
  existing filelock patch targets (that patch was never ported to this Apertus/Megatron script at
  all — this is the first run to reach a code path in `megatron.bridge` that hits it). Every bug
  fixed so far in this script's chain (qwen3_asr ×3, wheel-build race, pruned LM head, missing
  xielu) is upstream of this call site too, which is why it took 11 attempts to surface.
- **Fix**: ported the GLM script's proven `safe_config_loader` filelock patch verbatim — a
  `python3 -c` snippet that finds the module via `importlib.util.find_spec`, then replaces every
  `with filelock....:` line with `with contextlib.nullcontext():` (line-by-line prefix match, not
  a regex on the `FileLock(...)` argument, since those often contain nested parens). Safe because
  the config files are written once by local rank 0 before any reader starts — purely read-only
  after that, so the lock was never load-bearing here, only unsupported by the filesystem. Wired
  in right after the megatron-bridge wheel install in the main srun, before the SGLang patches.
  Unverified — needs a run; the TP=8 OOM fix from 3171564 also still needs its first real test.
- **Commit**: not committed.

### Run `3172177` — 2026-08-24 — filelock fix confirmed working, ran 33 min (vs. 4 min); new failure: LR scheduler needs total-steps set statically

- **Log**: `~/Downloads/slurm-3172177.out` (743 KB). 16 nodes, `clariden`. Failed after ~1976s
  (~33 min) — by far the longest this recipe has ever run before failing, though still inside
  `actor_init_model()`, before any weight-loading log line. Inconclusive on the TP=8 OOM fix from
  3171564: no OOM occurred, but the run never reached that far either.
- **Confirmed working**: the filelock patch fired (`Patched 1 filelock site(s)`), no recurrence of
  the "Stale file handle" config-load crash from 3171930.
- **Symptom**: `AssertionError: assert self.lr_decay_steps > 0`, raised in
  `megatron/core/optimizer_param_scheduler.py:159`, via `verl/utils/megatron/optimizer.py:131`
  (`get_megatron_optimizer_param_scheduler`) → `verl/workers/engine/megatron/transformer_impl.py:428`
  (`_build_lr_scheduler`) → `WorkerDict.actor_init_model()`, identically on all 18 worker ranks.
  The printed config dump showed `lr_decay_steps: None`, `total_training_steps: -1`/`None`.
- **Root cause**: an ordering bug in `verl`'s own `fully_async_main.py` (not specific to this
  script or to Megatron) — `_initialize_components()` calls `self._create_trainer(config)`, which
  calls `trainer.init_workers()` (building the Megatron LR scheduler, hence the assertion) several
  steps *before* it ever calls `trainer.set_total_train_steps.remote(total_train_steps)` (only
  reached later, after the rollouter is also created). So the runtime path that is supposed to
  populate `actor.optim.total_training_steps`/`lr_decay_steps` structurally cannot run in time for
  worker init in this entrypoint — `total_training_steps` must be set statically in the YAML
  config instead. Confirmed this is a known, already-solved gap: the sibling
  `train-gsm8k-qwen-3B-full-async-megatron.sh` script (which this recipe's `actor` block otherwise
  mirrors) already carries the exact fix with a comment stating this exact mechanism almost
  verbatim — this script just never copied that one line over.
- **Fix**: added `actor_rollout_ref.actor.optim.lr_decay_steps: 22419`, matching
  `rollout.total_rollout_steps: 22419` already set above it (so the LR schedule spans the full
  intended run) — same value, same fix, as the qwen-3B sibling script. Unverified — needs a run;
  this is the third run in a row where the TP=8 OOM fix from 3171564 still has not actually been
  exercised (each prior attempt died earlier for an unrelated reason first).
- **Commit**: not committed.

### Run `3173479` — 2026-08-24 — furthest run ever: cleared every prior blocker, reached the first real training step, new failure in sequence-parallel packing

- **Log**: `~/Downloads/slurm-3173479.out` (946,738 bytes). 16 nodes, `clariden`. Ran 2893s
  (~48.2 min) — by far the longest and furthest this recipe has ever gotten. Included a ~25-minute
  quiet stretch during DDP/optimizer construction (no crash, no log growth) that resolved on its
  own — not a hang, just slow setup for a 71B-param model across 16 nodes; flagged as a stall
  mid-run but turned out to be legitimate.
- **Everything from 3171564 through 3172177 confirmed fixed, in one run**: no `lr_decay_steps`
  assertion; **the TP=8 OOM fix from 3171564, untested for three straight runs, finally exercised
  and passed** (`DistributedDataParallel contains 8.83B parameters`, no
  `torch.OutOfMemoryError`); full `FullyAsyncTrainer` init; all 4 SGLang rollout replicas
  completed weight loading, CUDA graph capture, and HTTP server startup (first time ever reached);
  first NCCL trainer→rollout weight sync completed; initial GSM8K validation ran with real
  (near-zero, expected pre-training) accuracy metrics.
- **Symptom**: the first real training step (`update_actor`/`train_mini_batch`) crashed on all
  trainer ranks:
  ```
  AssertionError: First dimension of the tensor should be divisible by tensor parallel size
  ```
  in `megatron/core/tensor_parallel/mappings.py:173` (`_reduce_scatter_along_first_dim`).
- **Root cause**: `sequence_parallel` defaults to `True` in verl whenever `tensor_model_parallel_size
  > 1` (`verl/workers/config/engine.py`), which shards activations along the packed-sequence
  (THD) dimension and requires that dimension's length be a multiple of TP size. verl does have a
  TP-aware padding helper for exactly this
  (`verl/utils/megatron/sequence_parallel.py:pad_to_sequence_parallel`, computed dynamically from
  `mpu.get_tensor_model_parallel_world_size()`, not hardcoded) — but tracing its only two call
  sites showed it is invoked solely from the pipeline-parallel *shape-hint* computation
  (`verl/utils/megatron/pipeline_parallel.py:compute_transformers_input_shapes`), not from
  wherever the experimental `fully_async_policy` trainer actually builds the real packed input
  tensor fed to the model. So the shape metadata assumes TP-aligned padding while the real data
  tensor is never actually padded to match — a latent gap in `verl`'s experimental async-Megatron
  combination that no prior run had ever reached (every earlier bug in this whole 13-attempt chain
  was upstream of the first real training step).
- **Fix**: disabled `sequence_parallel` outright (`actor.megatron.sequence_parallel: False`,
  `ref.megatron.sequence_parallel: False`) rather than patching the padding gap in verl's
  experimental code blind — that would need cluster-cost-expensive iteration to get right, and
  this recipe's `override_transformer_config.recompute_granularity: full` (already set) already
  captures sequence_parallel's main benefit here (reduced activation memory, via full activation
  recomputation instead), so little is actually given up by turning it off. Unverified — needs a
  run; this would be the first real training step ever completed for this whole recipe if it
  clears.
- **Commit**: not committed.

### Run `3174079` — 2026-08-24 — sequence_parallel fix confirmed working; new failure one step deeper: cu_seqlens/tensor-size mismatch inside THD rotary embedding

- **Log**: `~/Downloads/slurm-3174079.out` (1,125,817 bytes). 16 nodes, `clariden`. Ran
  16:04:18–16:53:22 (~49 min) — essentially the same duration/shape as 3173479.
- **Confirmed working**: the 3173479 `sequence_parallel: False` fix held — no recurrence of the
  `_reduce_scatter_along_first_dim` divisibility assertion. Training again reached
  `update_actor`/`train_mini_batch` on the first real step (`Training Progress: 0/233` printed,
  matching this run's shorter `total_rollout_steps`/mini-batch-size-derived step count vs.
  3173479's 231 — consistent, not a regression).
- **Symptom**: all trainer ranks crashed inside the model forward, in Megatron-core's unfused THD
  rotary-embedding path:
  ```
  RuntimeError: split_with_sizes expects split_sizes to sum exactly to 669 (input tensor's size
  at dimension 0), but got split_sizes=[240, 256, 296, ... 280, 240]   # 32 entries, sum=14752
  ```
  at `megatron/core/models/common/embeddings/rope_utils.py:235` (`_apply_rotary_pos_emb_thd`,
  called from `attention.py`'s `query = apply_rotary_pos_emb(query, ..., cu_seqlens=cu_seqlens_q,
  ...)`), reached via `transformer_block.py`'s `checkpointed_forward` →
  `tensor_parallel.checkpoint` → a `TransformerLayer.forward` call — i.e. inside the *original*
  forward pass of one of the per-layer activation-recompute checkpoints (`recompute_granularity:
  full`, `recompute_method: uniform`, `recompute_num_layers: 1`), not a backward-recompute replay.
  Recurred identically (varying only which layer/step it hit) 7 times across the log before Slurm
  killed the job; the target (tensor's real size) varied run-to-run (669, 659, 706) but the
  32-entry `split_sizes` list always summed to ~14,744–14,768 — i.e. the full per-DP-rank
  mini-batch (32 packed GRPO sequences, matching `ppo_mini_batch_size:48 / 6 DP replicas × n=4
  responses/prompt`).
- **Root cause — not yet found**: traced the whole pipeline that builds `packed_seq_params`
  (`../verl`'s `preprocess_thd_engine`, `verl/models/mcore/util.py:317`) and the THD rope code in
  the actual `wqwqazwsxedc/Megatron-LM` fork (fetched fresh from GitHub — `rope_utils.py`,
  `attention.py`, `recompute.py`, `transformer_config.py` — since `../verl`'s vendored megatron
  isn't this fork). Ruled out, with reasoning:
  - **Not a config gap**: `context_parallel_size` (default 1, unset anywhere in this script or
    the fork's `TransformerConfig`) and `distribute_saved_activations` (default `False`, unset)
    both confirmed at their safe defaults — neither CP-chunking nor the TP-activation-compression
    codepath in `tensor_parallel/random.py`'s `CheckpointFunction` should be active. Also, the
    crash is in `CheckpointFunction.forward`'s *first* `run_function(*args)` call
    (`random.py:581`), before any save/compress/gather logic runs — so `distribute_saved_activations`
    couldn't be the cause even if enabled.
  - **Not the known sequence_parallel padding gap** (3173479's fix) — that's already disabled and
    confirmed not recurring (different assertion, different file).
  - **Not `dynamic_context_parallel`** (verl's own hybrid-CP feature, `engine.py:191`, default
    `False`, unset here) — would need an explicit config flag neither this script nor the bridge
    sets.
  - **`prepare_micro_batches`/`rearrange_micro_batches`** (`verl/workers/engine/utils.py:58`)
    genuinely does token-budget-based dynamic batching (`ppo_max_token_len_per_gpu: 16384`, this
    rank's full 32-sequence/~14.75k-token mini-batch fits in one microbatch under that budget) —
    so `packed_seq_params` and the model's `input_ids_rmpad` should always be built from the same
    32-sequence batch inside a single `forward_step` call; no evidence found of stale/cross-microbatch
    reuse.
  - **`Apertus1p5Bridge`/`get_apertus_decoder_block_spec`** (Megatron-Bridge's `apertus_bridge.py`,
    reused by this script's bridge) uses the *standard* `get_gpt_decoder_block_spec(...,
    use_transformer_engine=True)` layer spec — only `q_layernorm`/`k_layernorm`/`activation_func`
    (xIELU) are swapped in; the self-attention module class itself is untouched, so this isn't an
    Apertus-specific attention reshape bug on its face.
  - The one concrete, unexplained fact: the crashing tensor's real size (669/659/706) sits close
    to a *single* sequence's padded length in each batch (e.g. run 1's list has entries 656/672
    near 669), while `cu_seqlens`/`split_sizes` always describes the *full* 32-sequence batch —
    i.e. whatever tensor reaches this rope call has collapsed to roughly one sequence's worth of
    tokens while the packed-sequence metadata paired with it still describes all 32. No contiguous
    run of the 32 sequence-length list sums exactly to the observed target in any of the three
    logged occurrences, so it isn't simply "wrong slice boundary" either. This is unresolved.
- **Fix**: none yet. Added a diagnostic-only patch (user approved this path over the
  config-bisection alternative): `${MEGATRON_LM_DIR}/megatron/core/recompute.py`'s `chunk_runner`
  now prints, right before every per-layer activation-recompute chunk runs,
  `hidden_states.shape` and `packed_seq_params.cu_seqlens_q_padded[-1]`/`num_seqs` (tagged
  `[DIAG-ROPE]`) — applied via a plain Python line-insert on the batch host (not `sed -i .../a\`,
  to sidestep the GNU/BSD portability gap that class of edit has; tested locally end-to-end
  against the real fetched `recompute.py`, including idempotency on a rerun) right after the
  `Apertus1p5Bridge` addition and before the megatron wheel build, with
  `MEGATRON_WHEEL_BUILD_VERSION` bumped to `v4-diag-rope-recompute` so the instrumented source
  actually gets packaged (the exact mistake from run 3171218, avoided this time). Since
  `recompute_num_layers: 1` + `recompute_method: uniform`, this fires once per transformer layer
  and should show either the tensor already wrong at layer 0 (verl-side preprocessing bug) or a
  specific later layer where it first diverges (a bug in that layer's forward, most likely the
  xIELU/q-k-RMSNorm/GQA-adjacent code the Apertus bridge swaps in). Remove once the mechanism is
  found, same instrument-once-then-remove discipline as the GLM script's diagnostics. Unverified
  end-to-end by a cluster run yet — needs one.
- **Commit**: not committed.

### Run `3183472` — 2026-08-25 — [DIAG-ROPE] data collected: corruption exists before layer 0 even runs

- **Log**: `~/Downloads/slurm-3183472.out` (991,922 bytes). 16 nodes, `clariden`. Queued ~33 min,
  ran ~19 min, then failed (exit code 15, elapsed 4565s incl. queue). Setup confirmed clean, no
  regression: qwen3_asr patch, `Apertus1p5Bridge` dispatch, xIELU wheel, Megatron wheels rebuilt
  at `v4-diag-rope-recompute` (real rebuild, not a stale-cache skip), Lustre filelock patch,
  SGLang PR #32979 + local fixes all applied correctly. Zero OOM. Rollout came up fully (4 SGLang
  replicas, weight loading, CUDA graph capture, HTTP servers) and training began
  (`Training Progress: 0/233`) before crashing on the first `update_actor` step — identical crash
  signature to 3174079 (`split_with_sizes` in `rope_utils.py:235`).
- **The `[DIAG-ROPE]` data (the actual point of this run)**: only ever printed at `chunk=0-1` —
  **the very first transformer layer** — on every rank that logged it; no later chunk index ever
  appeared, meaning the crash happens inside/around layer 0 itself, before a second diagnostic
  line could even fire. Two distinct lines survived Ray's log dedup:
  - `hidden_states.shape=(623, 43, 8192) cu_seqlens_last=14976 num_seqs=43`
  - `hidden_states.shape=(597, 43, 8192) cu_seqlens_last=15000 num_seqs=43` (repeated 47x —
    essentially every other trainer rank matched this pattern)
  Two of the run's six distinct `split_with_sizes ... sum exactly to <N>` crash values (**623**
  and **597**) exactly match the two captured `hidden_states.shape[0]` values — direct proof the
  undersized tensor exists **before layer 0's forward is even entered**, not something that
  shrinks partway through the layer stack. (The other four crash variants — 706, 629, 572, 580,
  presumably other DP replicas — never got a surviving `[DIAG-ROPE]` line in the log, lost to
  buffering/dedup before the crash killed those ranks, but the mechanism is presumably identical.)
- **New, much stronger root-cause theory** (not yet confirmed by a diagnostic in verl's own code,
  only by matching shapes): `hidden_states.shape = (623, 43, 8192)` has middle dim **43 == num_seqs
  exactly** and first dim 623 close to a plausible max-individual-sequence-length for a
  43-sequence batch — i.e. this is the classic Megatron post-embedding **`[s, b, h]` layout with a
  real batch dimension of 43**, not the packed THD `[total_tokens, 1, hidden]` layout
  `preprocess_thd_engine` (`../verl`'s `verl/models/mcore/util.py:317`) is supposed to produce.
  Standard Megatron embedding (`word_embeddings(input_ids)` on a `[b, s]` input, then
  `.transpose(0, 1)`) turns a `[43, 623]`-shaped `input_ids` into exactly `[623, 43, hidden]` — so
  the tensor actually reaching the model looks like a normal **padded, per-sequence-batched**
  input (43 sequences × 623-token padding), not the packed rmpad buffer, while
  `packed_seq_params` still carries `qkv_format='thd'` and THD-style `cu_seqlens` (confirmed,
  since reaching `_apply_rotary_pos_emb_thd` at all requires `packed_seq_params.qkv_format ==
  'thd'` to be true in `attention.py`) describing the true packed total (14976/15000 tokens).
  Traced one plausible mechanism — `verl/workers/engine/megatron/transformer_impl.py:910`:
  `data_format = "thd" if self.engine_config.use_remove_padding else "bshd"` — and found verl's
  own code carries an acknowledged, unresolved gap here: `EngineConfig.use_remove_padding`
  (`verl/workers/config/engine.py:112`, default `True`) is a **separate field** from
  `ModelConfig.use_remove_padding` (`verl/workers/config/model.py:120`, also default `True`, and
  what this script actually sets — `actor_rollout_ref.model.use_remove_padding: True`) — the
  dataclass has its own literal `# TODO (this may conflict with the one in model config)` comment
  on the engine-level field. `verl/workers/engine_workers.py` syncs engine↔model at several call
  sites via `self.engine_config.use_remove_padding = self.model_config.get("use_remove_padding",
  False)` (note the **default of `False`** in that `.get()`, mismatched against the dataclass's
  own default of `True`) — a real candidate for the sync silently not happening on this
  experimental `fully_async_policy` Megatron path and `data_format` ending up `"bshd"`. **This
  theory has a hole**: the `"bshd"` branch of `gptmodel_forward_model_engine`
  (`verl/models/mcore/model_forward.py`) should not itself construct a `qkv_format='thd'`
  `packed_seq_params` at all (that's built only in the `"thd"` branch, via
  `preprocess_thd_engine`) — yet the crash unambiguously reached THD-format rope, which requires
  exactly that. So either this `data_format` mechanism isn't the actual cause and the bug is a
  distinct defect inside `preprocess_thd_engine` itself (its `shape = list(input_ids.shape[1:])`
  computation, `util.py:378`, was the other candidate spot examined but not conclusively ruled in
  or out), or there's a code path not yet found where both branches' effects combine. Not
  resolved — needs one more targeted diagnostic, this time patching verl's own (git-checked-out
  v0.9.0) `preprocess_thd_engine`/`gptmodel_forward_model_engine` to print `data_format`,
  `input_ids.shape` (pre-preprocessing), and `input_ids_rmpad.shape` (post-preprocessing,
  pre-model-call) directly, rather than inferring from megatron-core's post-embedding
  `hidden_states` shape one level downstream.
- **Fix**: none yet — still diagnostic. Awaiting a decision on the next diagnostic patch before
  spending another cluster allocation.
- **Commit**: not committed.

**Follow-up (same day, no cluster cost): root cause found and fixed, no further diagnostic run
needed.** Fetched the real `verl-project/verl` **v0.9.0 tag** source directly from GitHub for
`verl/models/mcore/model_forward.py`, `verl/models/mcore/util.py`, and
`verl/workers/engine/megatron/transformer_impl.py` — the local `../verl` checkout used for all
prior analysis in this chain is actually **v0.8.0** (confirmed via `git describe`), and per this
repo's own established lesson ("checked against v0.9.0 source... not the v0.8.0-4-g933979db
checkout"), that gap mattered again here. Also fetched the same three files from
`theely/verl`'s `Fix-fsdp-model-loading-on-async` branch, since this script's srun does
`git checkout -f v0.9.0` **then** `git reset --hard pr_origin/Fix-fsdp-model-loading-on-async`
— the second command replaces the whole tree, so the fork branch's content is what actually
ships, not the bare v0.9.0 tag. Confirmed the relevant code is textually identical in both.

Root cause: `verl/workers/engine/megatron/transformer_impl.py` calls
`gptmodel_forward_model_engine(..., vision_model=hasattr(self.model_config.hf_config,
"vision_config"), ...)` at two call sites. `Apertus1p5Config` always carries a `vision_config`
attribute (it is a genuinely multimodal architecture, per its own class definition) — so this
evaluates `True` on every forward call, regardless of whether the current batch has any image
content. Inside `verl/models/mcore/model_forward.py`'s `gptmodel_forward_model_engine`, the
`"thd"` branch calls `preprocess_thd_engine` and gets a correctly packed `input_ids_rmpad`
(shape `[1, total_tokens]`) plus a matching THD `packed_seq_params` — but immediately after,
unconditionally: `if vision_model: input_ids_rmpad, attention_mask =
build_vlm_attn_mask_thd(input_ids, pad_token_id)`. Despite the `"_thd"` in its name,
`build_vlm_attn_mask_thd` (`verl/models/mcore/util.py`) calls
`input_ids.to_padded_tensor(pad_token_id)`, producing a **dense, per-sequence-padded** tensor of
shape `[batch_size, max_seqlen]` — this **overwrites** the correctly packed `input_ids_rmpad`,
while `packed_seq_params` (built moments earlier from the correctly packed data) is left
completely unchanged and still describes the true packed-THD `cu_seqlens`/token totals. That
mismatched pair — a BSHD-shaped tensor paired with THD-format `packed_seq_params` — survives
through the embedding layer (`word_embeddings([batch, maxseq]).transpose(0,1)` producing exactly
the `[maxseq, batch, hidden]` shape the `[DIAG-ROPE]` diagnostic captured,
e.g. `(623, 43, 8192)` with `43` exactly matching `num_seqs`) all the way to the rotary-embedding
call, where megatron-core tries to `torch.split()` the small BSHD-shaped tensor using
`cu_seqlens` sized for the full packed batch — exactly the `split_with_sizes` crash chased
through runs 3174079 and 3183472. Confirmed this is not a mistake specific to this script: it is
a genuine, reachable bug in verl's own `gptmodel_forward_model_engine` for any Megatron
`vision_model=True` + THD-packing combination — `vision_model` is derived purely from whether
the **architecture** supports vision (`hasattr(..., "vision_config")`), never from whether the
**current batch** actually has any image content, so a text-only workload on a vision-capable
architecture always hits this.

Fix: added a source patch (in the per-node main srun, right after the `git reset --hard
pr_origin/...` step so it survives that reset, using `importlib.util.find_spec` to locate the
file rather than a hardcoded path — same convention as the megatron-bridge filelock patch a few
lines below it) that forces `vision_model=False` at both call sites. Correct specifically for
this recipe: it is text-only GSM8K, `multi_modal_inputs` is always empty, so there is no real
image content `build_vlm_attn_mask_thd` needs to handle — a genuine multimodal workload on this
model would need the opposite fix (keeping `packed_seq_params` in sync with whatever
`build_vlm_attn_mask_thd` produces), out of scope here. The patch is idempotent (checks for its
own marker comment before re-patching, fatal if the expected un-patched text isn't found and no
marker is present either) and was verified end-to-end locally against both the real v0.9.0 tag
file and the theely-fork variant (patches correctly, produces valid Python, indentation
preserved, idempotent re-run skips cleanly) before being wired into the script. The now-obsolete
`[DIAG-ROPE]` instrumentation (previous entry) was removed and `MEGATRON_WHEEL_BUILD_VERSION`
reverted to `v3-apertus1p5-bridge-pruned-vocab` accordingly. Also caught and fixed, before
submitting: the new comment block's prose used two apostrophes (`3183472's`, `tensor's`) and the
python patch's `target` string used literal single quotes around `'vision_config'` — all of
which, unescaped, would have prematurely terminated the outer single-quoted `bash -c '...'`
wrapper this whole srun body lives in (the exact class of bug documented in run 3149339's
Known-hazards-worthy lesson). Caught by grepping the entire outer single-quoted body for any
literal `'` before submitting, not by a failed run. Unverified by an actual cluster run yet —
needs one.

# Training Apertus v1.5 on verl, Megatron + V1 separate-async trainer (`rl-bench-apertus-v1.5-70B-sglang-megatron-v1-separate-async.sh`)

Apertus-v1.5-70B GRPO on GSM8K, CSCS Alps, 16 nodes × 4 GH200 (`clariden`). Script:
`Alps-Images/apps/verl/apertus-benchmarks/rl-bench-apertus-v1.5-70B-sglang-megatron-v1-separate-async.sh`.
A copy of the fully-async Megatron sibling (`rl-bench-apertus-v1.5-70B-sglang-megatron-async.sh`),
migrated on two axes at the user's request (2026-08-31): the **image** and the **verl trainer**.

## What changed vs the fully-async sibling

- **Image**: `verl:alps7-dev-0f334b540ccc7034` → `verl-cuda:alps7-dev-621fa40275c4f036` (the tag
  `train-gsm8k-glm5.1-700B-v1-separate-async-megatron.sh` uses — baked verl v0.9.0 + TransferQueue
  0.1.7 / megatron-core 0.19.0 / megatron-bridge 0.6.1 / flashinfer 0.6.14 / sglang 0.5.16). No
  runtime verl checkout. The fully-async recipe's `git reset --hard theely/verl
  Fix-fsdp-model-loading-on-async` is dropped — single-file FSDP2-only change, unused by the
  Megatron trainer, and a hard reset would clobber the baked v0.9.0 tree.
- **Trainer**: `verl.experimental.fully_async_policy.fully_async_main` →
  `verl.trainer.main_ppo`, V1 trainer in `separate_async` mode. Same transformation as the GLM
  v1-separate migration: `async_training:` block → `trainer.v1.separate_async.*` /
  `trainer.v1.sampler.*`; top-level `rollout:` block gone (standalone-rollout pool declared in
  `actor_rollout_ref.rollout.{nnodes,n_gpus_per_node}`); `data.train_batch_size: 0` →
  `parameter_sync_step * ppo_mini_batch_size` (`ROLLOUT_N=16`, `PPO_MINI_BATCH_SIZE=6`,
  `PARAMETER_SYNC_STEP=2`, `TRAIN_BATCH_SIZE=12`); `transfer_queue:` block added; `old_log_probs`
  via `rollout.calculate_log_probs` + `algorithm.rollout_correction.bypass_mode` (the fully-async
  `actor.use_rollout_log_probs` and `actor.optim.lr_decay_steps` workaround are both dropped —
  V1 sets `total_training_steps` before workers are created).
- **PR patches + `v1-separate-async-fixes.patch`**: fetched once on the batch host, `sbcast`, and
  `git apply`'d on `/workspace/verl` — same as the GLM v1-separate script. #7422 (preserve
  `load_format=dummy` in the disaggregated rollout) and the hybrid-replica no-op (fix 1 of
  `v1-separate-async-fixes.patch`) are load-bearing for separate-async; without the latter the
  V1 trainer builds hybrid rollout replicas (trainer-world/rollout-world = 48/16 = 3 here) at
  `gpu_memory_utilization=0.75` on every training GPU and OOMs at init.
- **`checkpoint.strict: False`** + **`save_freq: 100`** (see hazards below).
- **Reward function rewritten** (see the "The zero-reward chain" section below).

## Group 1 / Group 2 (Apertus support) — UNCHANGED mechanism, kept from the sibling

Group 1 (swiss-ai transformers wheel, SGLang PR #32979 + local fixes) and Group 2 (wqwqazwsxedc
Megatron-LM / Megatron-Bridge **apertus fork wheels** built at runtime and installed over the
image's stock megatron-core/bridge + the vendored `Apertus1p5Bridge` +
`patches/apertus1p5_bridge.py` + the xielu CUDA wheel + the `safe_config_loader` filelock patch)
are all identical to the sibling. **A first pass tried to keep the image's stock megatron-core
0.19.0 / megatron-bridge 0.6.1 and add only the apertus delta as targeted line-patches
(`mlp.py` module-activation gate, `finalize_model_grads.py` TP grad-sum) + vendored bridge
modules — this was abandoned.** It built and ran end-to-end (runs `3240861`/`3243271`/`3243467`)
but trained degenerately for a reason later shown to be unrelated (the reward function — see
below), *and* the fork wheels on the new image trained identically (run `3244676`), so the
line-patch complexity bought nothing. The fork wheels (`megatron-core 0.18.0+60b5c9588`,
`megatron-bridge 0.5.0+032c0740`) install cleanly over the new image's torch 2.11 / TE and were
validated end-to-end in run `3247540`.

## Known hazards (this recipe specifically)

- **`checkpoint.strict: False` is required.** Apertus-v1.5 is multimodal; this recipe deliberately
  does not map the `vision_tokenizer` / `audio_tokenizer` towers (text-only GSM8K). The strict
  HF-checkpoint export then hard-fails at whatever `save_freq` triggers — `RuntimeError: 473
  tensors from the original checkpoint were not written` (all `model.audio_tokenizer.*` /
  `model.vision_tokenizer.*`), which killed run `3240861` at step 20. `strict: False` saves the
  LM-only partial checkpoint; `save_freq: 100` (> `total_training_steps: 46`) also keeps the
  shakedown from exercising checkpoint save at all.
- **`[a1p5-diag]` weight dump** (in `patches/apertus1p5_bridge.py`'s `load_weights_hf_to_megatron`)
  was a temporary diagnostic — removed after run `3247540` confirmed the conversion. It proved:
  every weight group loads with sane stats on BOTH fork-0.18/0.5 and stock-0.19/0.6.1
  (`attention_layernorm.weight` mean +0.0022 byte-identical to the HF checkpoint — Apertus's
  layer-0 norm gammas genuinely are ~0, not zero-centered; standard `w·rmsnorm(x)` applies
  correctly). If a future conversion regression is suspected, re-add it.
- **Do NOT switch the `apertus1p5_bridge.py` RMSNorm mappings from `ReplicatedMapping` to
  `AutoMapping`.** Tried on stock megatron-bridge 0.6.1 (run `3243271`) chasing an "Unrecognized
  mapping type" warning from `_add_separate_layernorm_mappings` — it loaded
  `attention_layernorm`/`feedforward_layernorm` as ~0 (wrong) and did not fix anything. The
  warning only skips an unused `input_layernorm.weight` alias; the fused-TE
  `linear_qkv.layer_norm_weight` direct mapping is what actually loads.
- **`<|inner_prefix|>` / `<|inner_suffix|>` chat-template leak.** On the new image's sglang 0.5.16
  + PR #32979, Apertus-v1.5's reasoning delimiters leak into the response text as literal tokens
  (`...candy.<|inner_suffix|>76`). The fork's older image handled them (the model followed the
  prompt's `<answer>` instruction there and RL reinforced it — that's how run `3184869` learned
  to reward 0.99). Not fixed here — the reward function parses around it (the final answer is the
  region after the last `<|inner_suffix|>`). A proper fix (verl tokenizer / sglang apertus_mm
  chat-template wiring) is an open follow-up if `<answer>`-format behavior is wanted.
- Everything in the sibling's hazards list (qwen3_asr triple-`exist_ok`, SGLang PR #32979 file-set
  drift → allowlist, pruned LM head, missing xielu, filelock, `sequence_parallel: False`,
  `vision_model=False`, sbcast bus-error on large wheels) applies verbatim — same Group 1/2 code.

## The zero-reward chain (runs 3240861 → 3247540) and its resolution

Five consecutive runs completed real training steps but with **`critic/score/max` flat at exactly
0.0 every step** — zero GRPO advantage variance, `grad_norm` 0.0 on ~half the steps, no learning.
Chased as a weight-conversion bug for four runs (stock line-patches vs fork wheels, `AutoMapping`
vs `ReplicatedMapping`, zero-centered-gamma, RoPE) — all dead ends; `[a1p5-diag]` proved the
weights load correctly and identically in every configuration, and the RoPE code is byte-identical
across megatron-core 0.18/0.19.

**Root cause (run `3246622`, found by adding a `[REWARD-DUMP]` print to the reward function):**
Apertus-v1.5-70B **solves GSM8K fluently and correctly** but ends its answers with
`Final answer: N` / `\boxed{N}` / a bare trailing number / `N` right after `<|inner_suffix|>` —
**never** the `<answer>...</answer>` tags the recipe's original `gsm8k_reward.py` (copied from the
GLM/qwen recipes) demanded. Every rollout scored 0 → no signal. Not a stack bug at all.

**Fix**: `gsm8k_reward.py` rewritten to parse `<answer>` OR `\boxed{}` OR `final answer: N` OR the
last number, each checked first in the post-`<|inner_suffix|>` region; outcome reward (correct = 1.0)
now drives training; small shaping bonus for any clear delimiter; length penalty kept. **No
dataset / system-prompt change** — GSM8K parquet is not regenerated. General lesson: when a
migrated recipe shows zero reward with visibly-working generations, dump the actual rollout text
(a `print` in the custom reward fn is the fastest way — verl does not log generations) BEFORE
assuming a model/weight bug.

## Configuration audit (architecture-vs-config) — 2026-08-31

Per the repo-wide process. Apertus-v1.5-70B is **dense** (8.83B params/rank at TP=8; no
expert/EP config), text+vision+audio multimodal, `model_type: apertus1p5`. `router_replay`/R3
does not apply (not MoE). `algorithm.rollout_correction.bypass_mode: True` set (the one
algorithm flag applying to every model in this repo). The multimodal-specific gaps (vision-tower
`eager` attention, SGLang generic-loader misload, pruned LM head, `vision_model=False` forced,
deliberately-unmapped vision/audio towers) are all inherited from the sibling's already-fixed
list — see that section's audit entry. No new architecture-vs-config gap this pass. The one
architecture-adjacent finding is the reward/prompt-format mismatch above, which is a
recipe-config issue, not a model-config one.

## Run log

### Runs `3240861` / `3243271` / `3243467` — 2026-08-31 — stock-0.19/0.6.1 line-patch approach: builds + runs, trains degenerately (later shown = reward bug)

- **Approach**: keep the image's stock megatron-core 0.19.0 / megatron-bridge 0.6.1; add apertus
  support as `mlp.py` + `finalize_model_grads.py` line-edits (the apertus-relevant delta of the
  fork's one megatron-core commit) + vendor `apertus_bridge.py` (fork verbatim) +
  `apertus1p5_bridge.py` into the installed package.
- **Result**: all setup clean (image smoke test green, both line-patches applied, bridges
  vendored, `DistributedDataParallel contains 8.83B parameters`, no OOM). `3243271` also carried
  an `AutoMapping` experiment for the two RMSNorm mappings (chasing an "Unrecognized mapping
  type" warning) — loaded them as ~0, reverted. `3240861` FAILED at step 20 on the strict
  HF-export `473 tensors not written` (→ `checkpoint.strict: False` added). All three trained
  with `critic/score/max` flat 0.0.
- **`[a1p5-diag]`** (added in `3243467`): every weight group loads sane; `config.rotary_base`
  4000000, `rope_scaling_factor` 32.0 (real 70B checkpoint values, correctly read);
  `layernorm_zero_centered_gamma=False`; HF `attention_layernorm` spot-check byte-identical to
  loaded. Ruled out weights / norms / rope as the cause.
- **Commit**: not committed.

### Run `3244676` — 2026-08-31 — reverted to fork megatron wheels on the new image: builds cleanly, completes 46 steps, STILL degenerate → megatron version is definitively not the cause

- Fork wheels (`megatron_core-0.18.0+60b5c9588`, `megatron_bridge-0.5.0+032c0740`) built from
  source at runtime and `pip install`'d over the new image's stock — **no ABI error, `import
  megatron.core` OK**, DDP grad-buffer fine. Completed all 46 steps.
- `critic/score/max` = 0.0 on every one of the 46 steps; `[a1p5-diag]` byte-identical to the
  stock-patch runs. **Conclusion: 0.18/0.5 vs 0.19/0.6.1 is not the variable.** Reverted the
  script to the fork-wheel Group 2 (the line-patch approach bought nothing).
- **Commit**: not committed.

### Run `3246622` — 2026-08-31 — `[REWARD-DUMP]` diagnostic: ROOT CAUSE FOUND — the model works, the reward function was wrong

- Added a rate-limited `print(solution_str)` to `compute_reward`. Generations are **fluent,
  correct, step-by-step GSM8K solutions** ending in `Final answer: 10.00`, `Final Answer:
  \boxed{20}`, bare `76`, `The final answer is $20,800.` — never `<answer>` tags. Also visible:
  `<|inner_prefix|>`/`<|inner_suffix|>` leaking as literal text.
- The reward only credited `<answer>...</answer>` → every rollout 0 → no GRPO variance. Completed
  46 steps, degenerate, no crash (`checkpoint.strict: False` held — the `473` export error was
  now non-fatal).
- **Fix**: rewrote `gsm8k_reward.py` — robust answer extraction (`<answer>` | `\boxed{}` |
  `final answer: N` | last number, post-`<|inner_suffix|>` first), outcome reward drives
  training. Verified locally against the actual dumped generations.
- **Commit**: not committed.

### Run `3247540` — 2026-08-31 — VALIDATED: 46/46 steps, healthy GRPO curve — the migration is complete

- Reward fix live. `[REWARD-DUMP]` `parsed=` field non-`None` on every surviving sample
  (`gt='9240' parsed='9240'`, etc.). **`critic/score/max` = 1.1 from step 1** (1.0 outcome +
  0.1 delimiter bonus). `critic/score/mean` ~0.94 → ~0.99 (modest — Apertus-v1.5-70B already
  near-saturates GSM8K); GRPO learning shows most clearly in **response-length compression 306 →
  225 tokens**. `grad_norm` nonzero every step (one benign zero-variance step 34); `ppo_kl`
  ~0.001 stable; `actor/loss` small/finite, no NaN. `update_weights` ~8 s / ~15 GB/s throughout.
  Slurm COMPLETED, exit 0, ~30 s/step, ~24 min of stepping. One non-fatal `473 tensors not
  written` at final HF export (the known multimodal gap; `strict: False` makes it a warning).
- **Diagnostics removed** (`[a1p5-diag]`, `[REWARD-DUMP]`); `MEGATRON_WHEEL_BUILD_VERSION` →
  `v5-apertus1p5-bridge-v1sep`; checked-in `patches/apertus_bridge.py` deleted (the fork wheels
  provide it). `patches/apertus1p5_bridge.py` kept (still vendored into the fork bridge
  checkout), comments updated.
- **Post-run tuning (2026-08-31; validated in runs `3250502`/`3251587` below)**: `perf/throughput` was ~37 tok/s vs job
  `3184869`'s ~120. Cause: the V1-separate migration had shrunk `ppo_mini_batch_size` 48 → 6
  (copied from the GLM-5.1 700B v1-separate recipe, where 6 is forced by PP=3 memory pressure).
  `perf/throughput = total_tokens_per_step / step_time`, and the ~9.7 s/step trainer→rollout
  weight sync + optimizer + TransferQueue round-trip is a fixed per-step cost, so an 8× smaller
  batch ≈ 1/3 the token rate (and ~3.5× worse wall-clock per prompt). MFU is 0 in both (verl
  can't compute it for `apertus1p5`), so `perf/throughput` is the only number and it is
  batch-size-sensitive. Restored `PPO_MINI_BATCH_SIZE=48` / `TRAIN_BATCH_SIZE=96` to match
  `3184869` exactly (all other actor params — `ppo_max_token_len_per_gpu` 16384,
  `ppo_micro_batch_size_per_gpu` 1, `use_dynamic_bsz`, TP=8/PP=1/EP=1, offloads, recompute —
  already matched). Safe on memory: dynamic-bsz caps each *microbatch* at 16384 tokens, so the
  activation peak (was 61/95 GiB at mb=6) is per-microbatch, unchanged by mb size. `46 steps ×
  96 prompts = 4416` = the fully-async recipe's `total_rollout_steps` exactly, so
  `total_training_steps: 46` is unchanged and processes the same total data.
- **Open follow-ups** (both non-blocking): the `<|inner_suffix|>` chat-template leak (a proper
  fix would let the model follow `<answer>` again); and GSM8K being near-saturated for this base
  model (a longer run / harder eval would show real learning headroom).
- **Commit**: not committed.

### Run `3250502` — 2026-09-01 — `ppo_mini_batch_size` 6 → 48 (match job 3184869): perf/memory/quality all validated; new crash at step 36 in verl's `_balance_batch`

- **Change**: `PPO_MINI_BATCH_SIZE` 6 → 48, `TRAIN_BATCH_SIZE` 12 → 96 — the only training param
  that differed from the fully-async sibling. Motivation: `perf/throughput` = `total_tokens /
  step_time`, and the ~9.7 s/step fixed weight-sync + optimizer + TQ cost was amortized over 8×
  fewer tokens → ~37 tok/s vs job `3184869`'s ~120.
- **Validated at mb=48**: peak `actor/perf/max_memory_allocated_gb` **72.6 GiB** / 95 (mb=6 was
  61 — the per-microbatch dynamic-bsz cap held, ~+11 GiB only, ~10 GiB headroom); **throughput
  ~160–180 tok/s** (beat the 100–130 estimate); `timing_s/step` ~49–53 s, `update_actor`
  ~38–41 s; `critic/score` healthy, identical shape to `3247540`. **NOT an OOM.**
- **Crash — step 36, verl bug, directly triggered by the batch change**:
  ```
  verl/trainer/ppo/v1/trainer_base.py:1467  _balance_batch
    torch.tensor([tag["seq_len"] for tag in batch.tags], ...)   KeyError: 'seq_len'
  ```
  `required_multiple = ppo_mini_batch_size(48) * rollout.n(16) = 768`. When the separate-async
  sampler delivers < 768 valid trajectories in a sync window, verl pads with synthetic samples
  carrying no `seq_len` tag; `_balance_batch` (`trainer.balance_batch: True` by default) reads
  `tag["seq_len"]` unconditionally → KeyError. Steps 34/35 logged `Upsampled batch from N to 768`
  and survived (intermittent — depends whether padded samples reach the tag list); step 36 didn't.
  At mb=6 the multiple was 96 and `3247540` never came up short in 46 steps → dormant.
- Also surfaced: a **stale `[DIAG-ROPE]` diagnostic** patched into `recompute.py` in the shared
  Lustre `Megatron-LM` checkout by an old sibling run (see that script's run `3174079` entry) —
  the checkout persists across submissions and the reuse branch never reset it, so every wheel
  build since packaged it. ~hundreds of thousands of log-spam lines.
- **Fix** (both in the script): `trainer.balance_batch: False` (skips `_balance_batch` entirely —
  `use_dynamic_bsz` already bounds per-microbatch tokens so the DP-imbalance cost is small); and
  the Megatron clone reuse branch now `git checkout -f apertus && git reset --hard` on both
  `${MEGATRON_LM_DIR}` and `${MEGATRON_BRIDGE_DIR}` (no `git clean` — Bridge is a nested untracked
  dir and the diagnostics only touch tracked files). `MEGATRON_WHEEL_BUILD_VERSION` →
  `v6-apertus1p5-bridge-pristine`.
- **Commit**: not committed.

### Run `3251587` — 2026-09-01 — FULLY VALIDATED: 46/46 steps at `ppo_mini_batch_size: 48`, healthy curve, throughput beats the original fully-async run

- Both `3250502` fixes live. **`git reset --hard` ran, `[DIAG-ROPE]` = 0 occurrences** (was
  hundreds of thousands). Wheels rebuilt at `v6`. **Zero `KeyError: 'seq_len'` / `_balance_batch`
  / `Upsampled batch`** — ran clean through step 36 to 46 + a checkpoint save at `global_step_46`.
- **Perf** (steady steps 2–45): `perf/throughput` **158–184 tok/s** (mb=6 was ~37; job `3184869`
  fully-async was ~120 — this now beats it), `timing_s/step` ~48–56 s, peak
  `max_memory_allocated_gb` **72.5 GiB** / 95, weight sync ~9.5–9.8 s.
- **Score**: `critic/score/max` 1.10 every step; `critic/score/mean` ~0.84–1.05 (~0.98 avg,
  trending up — near-saturated); `actor/grad_norm` 0.18–0.43, all nonzero, no NaN. Same shape as
  `3247540` / `3250502`.
- COMPLETED, exit 0, ~87 min wall. Only log "errors" are post-`Training Progress: 100%` teardown
  noise (DataLoader worker `Killed`, wandb atexit, Ray GCS spam).
- **The migration + tuning is done.** Recipe is stable at 16 nodes / mb=48 / 46 steps. Remaining
  open items are the two non-blocking follow-ups noted under `3247540` (chat-template
  `<|inner_suffix|>` leak; GSM8K near-saturation).
- **Commit**: not committed.

# Megatron vs FSDP2

Direct comparison of the two Apertus-v1.5-70B GRPO/GSM8K trainer variants
(`rl-bench-apertus-v1.5-70B-sglang-megatron-async.sh` vs
`rl-bench-apertus-v1.5-70B-sglang-fsdp2-async.sh`) once both had, for the first time, actually
reached sustained real training steps rather than crashing/OOMing during init (see each script's
own run log above for how many attempts that took).

## Job `3184869` (Megatron) vs job `3185943` (FSDP2) — 2026-08-25

- **Logs**: `~/Downloads/slurm-3184869.out` (Megatron, 1,138,195 bytes) and
  `~/Downloads/slurm-3185943.out` (FSDP2, 3,507,893 bytes). Both fetched fresh, both 16 nodes,
  `clariden`.
- **Final state**: both jobs hit **TIMEOUT** at the `--time=2:00:00` limit — neither finished the
  configured 233-step run, and neither crashed (no OOM/traceback in either tail).
- **Steps reached**: Megatron 38/233 (~16%), FSDP2 47/233 (~20%) — comparable, so the trend
  comparison below isn't an artifact of one run being too short to judge.
- **Reward/accuracy (`critic/score/mean`, the actual GSM8K learning signal — not the PPO-clip
  actor loss, which stays near-zero for GRPO regardless of whether the policy is improving)**:
  - **Megatron**: near-0 through step 10 (-0.0002 to +0.013 noise), then a clean breakout —
    step 11: 0.036 → step 17: 0.12 → step 19: 0.66 → step 21: 0.99 → holds ~0.93-1.02 through
    step 38. `grad_norm` stayed healthy (0.09-0.43) throughout; response length fell from ~330 to
    ~200 tokens as accuracy rose — a textbook converging GRPO curve.
  - **FSDP2**: flat at ~0 (-0.0003 to +0.00001) across all 47 steps. No breakout at any point.
    **8 of 47 steps (17%) had `grad_norm` exactly 0.0** — every sample in that batch received an
    identical GRPO-standardized advantage, i.e. zero training signal for the whole step. Megatron
    had zero such steps in 38.
- **Verdict**: not noise, not a "FSDP2 just learns slower" difference — FSDP2 shows no learning
  signal whatsoever over a step count comparable to where Megatron had already reached ~95%+
  accuracy. The recurring exact-zero-`grad_norm` steps point to a degenerate reward/advantage
  collapse (every one of the `rollout.n: 16` samples for a prompt getting an identical reward)
  specific to the FSDP2 run, not merely slower convergence.

## Investigation: config diff ruled out, fork-based weight loading is the leading suspect

- **Sampling config is identical between the two scripts** — checked directly, not assumed:
  both set `actor_rollout_ref.rollout.{temperature: 1.0, n: 16}` and neither sets `top_p`/`top_k`
  (both scripts source the same `rollout` config defaults). `async_training` block
  (`staleness_threshold: 0.1`, `trigger_parameter_sync_step: 2`, `require_batches: 1`,
  `partial_rollout: False`) is byte-identical too. So a rollout-sampling misconfiguration
  (e.g. accidentally-greedy decoding collapsing all `n=16` samples to the same output/reward) is
  ruled out as the cause — both scripts ask SGLang to sample the same way.
- **The one significant source-level delta between the two scripts, beyond FSDP2-vs-Megatron
  parallelism itself**: the FSDP2 script's srun does
  `git remote add pr_origin https://github.com/theely/verl.git` →
  `git fetch pr_origin Fix-fsdp-model-loading-on-async` →
  `git reset --hard pr_origin/Fix-fsdp-model-loading-on-async` after the v0.9.0 checkout — i.e.
  it runs a **forked, non-upstream verl tree** for its actor/FSDP2 weight-loading path. The
  Megatron script has no equivalent fork swap; it runs the plain v0.9.0 checkout (plus the
  documented PR/local patches). This fork's own design, per its code comment (see run
  `3185487`'s entry in the FSDP2 script's own run log): **only global rank 0 loads real weights
  from disk; every other rank builds an empty/meta-tensor model and receives its shard via a
  broadcast from rank 0.**
- **Hypothesis (not yet confirmed)**: if that rank-0-load-then-broadcast path is subtly broken —
  wrong shard boundaries, a race with FSDP2's own sharding, or corruption in the broadcast itself
  — some or all FSDP2 trainer ranks could end up with incorrect actor weights. Those weights are
  then pushed to the standalone SGLang rollout replicas via the NCCL checkpoint engine
  (`checkpoint_engine.backend: nccl`, same as Megatron's) every `trigger_parameter_sync_step: 2`
  actor updates. A broken/degenerate policy on the rollout side would produce near-identical
  (uniformly wrong, or uniformly the same low-entropy) responses for repeated samples of the same
  prompt — which is exactly what a GRPO zero-variance reward group looks like, and would explain
  both the flat reward curve and the recurring exact-zero `grad_norm` steps.
- **Not yet verified.** This is a plausible mechanism from reading the fork's own documented
  design and matching it against the observed symptom, not something confirmed by a targeted
  diagnostic. Next step, if pursued: log rollout output diversity (e.g. distinct response text
  count) per prompt group directly from the FSDP2 rollout replicas, and/or diff actor weight
  checksums across FSDP2 trainer ranks right after `actor_init_model()` and again after the first
  NCCL weight sync, to confirm or rule out corrupted/mismatched weights before assuming the fork
  is at fault. No code changes made yet — this section is diagnosis, not a fix.
- **Commit**: not committed.

## Hygiene fix: forked-branch swap converted to a pinned static patch (not yet cluster-tested)

Before pursuing the checksum diagnostic above, converted the FSDP2 script's dependency on
`theely/verl@Fix-fsdp-model-loading-on-async` from a live `git reset --hard` branch swap to a
static, checked-in patch — `apertus-benchmarks/patches/fsdp2-rank0-load-fix.patch`, applied via
`git -C /workspace/verl apply` with the same apply-or-already-present-or-fatal idempotency check
the GLM/Megatron scripts already use for their PR patches.

- **Why**: the branch swap ran whatever commit the fork happened to be at when each job
  submitted, with no pin — the exact "uncontrolled version drift" class of hazard already
  documented for SGLang PR #32979's live re-fetch (runs `3184895`/`3152782`). Unlike every other
  external fix in this script, it had never actually been diffed against the real v0.9.0 tag to
  confirm it was scoped to just the one intended change.
- **Verified before converting, not assumed**: fetched GitHub's compare between the real
  `verl-project:v0.9.0` tag and the fork branch tip. Despite a messy/diverged commit graph
  (`ahead_by: 4, behind_by: 159` — an artifact of history rewriting on one side or the other), the
  file-level tree comparison shows **exactly one file differs**:
  `verl/workers/engine/fsdp/transformer_impl.py`. No hidden drift from unrelated commits on the
  fork branch — the branch swap and this patch are functionally identical.
- **What the fix actually does** (for the record, since it was previously undocumented in this
  repo): stock v0.9.0 has every FSDP2 rank independently call `auto_class.from_pretrained(...)`,
  loading the full model into host RAM per rank — for this 70B model at 4 workers/node, ~4×140 GB
  = 560 GB per node. The patch makes only global rank 0 load real weights from disk; every other
  rank builds an empty/meta-tensor model via `accelerate.init_empty_weights()`, and
  `fsdp2_load_full_state_dict(..., broadcast_from_rank0=True)` — a real, supported PyTorch/verl
  FSDP2 API, not a custom mechanism — distributes the actual weights afterward. This recipe runs
  `critic.enable: false`, so the patch's value-head/TRL branch is dead code here; only the plain
  `language_model` branch and the final `full_state`/broadcast hunk are actually exercised.
- **Verified locally before wiring in** (mirroring this repo's own established discipline —
  see the GLM/Megatron sections' repeated "verify against the real v0.9.0 tag, not local `../verl`"
  lesson; local `../verl` is confirmed still `v0.8.0`, so it was not used for this check): checked
  out the actual `verl-project/verl` `v0.9.0` tag fresh from GitHub, confirmed the patch applies
  cleanly (`git apply --check`), confirmed the patched file compiles
  (`python3 -m py_compile`), and confirmed `git apply --reverse --check` correctly detects
  "already applied" — the same idempotency check the apply-or-fail loop relies on. Checked-in
  patch file and the script's heredoc copy verified byte-identical. `bash -n` and a grep of the
  entire outer single-quoted `srun bash -c '...'` body for stray literal `'` characters both pass
  (the run-3149339 quoting hazard).
- **This is a reproducibility/hygiene fix, not a fix for the reward-collapse symptom itself** —
  the resulting behavior should be identical to what runs 3184869/3185943 already ran, just now
  pinned and auditable instead of live-fetched. Does not replace the checksum/rollout-diversity
  diagnostic above, which still needs to run to actually confirm or rule out this code path as the
  collapse's cause. Unverified end-to-end by a cluster run yet — needs one before drawing any
  conclusion.
- **Commit**: not committed.

## Diagnostic added on top: weight-broadcast checksum (not yet cluster-tested)

Implemented the checksum diagnostic proposed above, rather than a separate short test-only job —
`apertus-benchmarks/patches/fsdp2-broadcast-diagnostic.patch`, applied via the same
`git apply`/`--reverse --check` idempotency pattern, stacked directly on top of
`fsdp2-rank0-load-fix.patch` (same file, `verl/workers/engine/fsdp/transformer_impl.py`).

- **What it does**: picks a small parameter deterministically (`min` by `numel()` over
  `module.named_parameters()`, so every rank picks the same one). On rank 0, right before FSDP2
  sharding, hashes that parameter's *real* value (from the `state_dict()` just loaded from disk)
  with SHA-256 and prints it. After `fsdp2_load_full_state_dict(...)` returns, every rank
  re-gathers that same parameter's full value via `DTensor.full_tensor()` (a collective, called
  unconditionally by all ranks) and prints its own SHA-256. If every rank's post-load hash matches
  rank 0's pre-shard hash, the broadcast reproduced weights correctly and the hypothesis is
  refuted; if any rank's hash differs, that's direct evidence the rank-0-load/broadcast path is
  actually corrupting weights, redirecting the investigation there instead of the reward/sampling
  pipeline.
- **Verified locally before wiring in** (same discipline as the fix above, and as this repo's
  established diagnostic-patch precedent — see the GLM section's runs 3137775–3144665): built the
  isolated diagnostic-only diff by diffing a rank0-fix-only checkout against a rank0-fix+diagnostic
  checkout (not by hand-editing a diff), confirmed it applies cleanly on top of
  `fsdp2-rank0-load-fix.patch` against the real `v0.9.0` tag, confirmed the combined result
  compiles, confirmed `--reverse --check` idempotency detection works, and confirmed the
  checked-in patch file, its embedded heredoc copy in the script, and the intended source content
  are all byte-identical (caught and fixed one real bug in this process: the first attempt to embed
  the patch as a heredoc silently dropped a trailing context line that was only a single space
  character — both `Write` and `Edit` trimmed it — which would have made `git apply` fail with
  "corrupt patch" on the cluster; rebuilt via direct file concatenation instead, and re-verified
  hunk line counts explicitly rather than trusting `git apply --check` alone the second time).
  `bash -n` and the single-quoted `srun bash -c '...'` stray-quote scan both pass (the diagnostic's
  Python uses double-quoted f-strings throughout, no literal `'` characters).
- **Scope check**: since `git checkout -f v0.9.0` always resets the verl tree fresh at the start
  of every run, both patches' forward-apply always runs against a pristine checkout in practice —
  the `--reverse --check` "already present" fallback (present for the same idempotency-convention
  reasons as every other patch in this repo) is not actually load-bearing here and was not tested
  against a real already-both-applied state beyond confirming it fails safely (script would report
  FATAL and stop rather than silently doing the wrong thing).
- **Remove once the mechanism is confirmed one way or the other** — diagnostic only, same
  instrument-once-then-remove discipline as every other diagnostic in this repo.
- **Commit**: not committed.

### Run `3189642` — 2026-08-25 — script bug: diagnostic patch was sbcast'd but never applied; new, unrelated OOM in the first backward pass

- **Log**: `~/Downloads/slurm-3189642.out`. 16 nodes, `clariden`. Submission hit an unrelated,
  system-wide CSCS scheduler/SSH/filesystem outage (`/status/systems` confirmed all unhealthy
  except S3) that delayed status confirmation by ~25 min — not specific to this job. Once CSCS
  recovered, ran ~26 min before **FAILED** (exit code 15).
- **Symptom 1 (script bug, self-inflicted)**: `fsdp2-rank0-load-fix.patch` applied cleanly on
  16/16 nodes, but grepping the full log for `DIAG-FSDP2-BROADCAST` or any related confirmation
  text returns **zero matches** — the diagnostic instrumentation never ran.
- **Root cause**: the "Apply Verl fixes" block in the srun body only ever had a `git apply` step
  for `fsdp2-rank0-load-fix.patch`. `fsdp2-broadcast-diagnostic.patch` was correctly embedded as
  a heredoc and `sbcast`'d to every node (confirmed present on disk), but no corresponding
  `git apply` call was ever added for it — the patch sat on every node's `/tmp` unused. A pure
  oversight when wiring the diagnostic in, not caught by any of the local verification done
  beforehand (patch-apply/compile/idempotency checks all validated the *patch files themselves*,
  not that the *script* actually invokes `git apply` on the second one).
- **Fix**: added the missing apply block, mirroring the first one exactly (same
  apply-or-already-present-or-fatal pattern), stacked immediately after
  `fsdp2-rank0-load-fix.patch`'s. Verified this time by re-extracting both heredocs from the
  actual current script content (not from scratch files) and simulating the full sequential
  apply against the real `v0.9.0` tag: both apply cleanly in order, the combined result compiles,
  and `grep -c DIAG-FSDP2-BROADCAST` on the patched file returns 3 (the tag appears 3 times in
  the instrumentation, as expected). `bash -n` and the stray-single-quote scan both still pass.
- **Symptom 2 (new, unrelated to the diagnostic)**: progress well past every prior init-time
  blocker — 100% weight loading (1224/1224 shards, ~4m32s), `Apertus1p5ForConditionalGeneration
  contains 71.94B parameters`, healthy post-FSDP memory (22.37/95.00 GB), SGLang rollout fully up
  (CUDA graphs captured, HTTP servers started), reached `Training Progress: 0/233` — i.e. the
  rank0-load-fix patch's actual job (preventing the FSDP2 init OOM) is confirmed working exactly
  as intended. Crashed instead on the very first `update_actor` call:
  `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.89 GiB ... 92.98 GiB memory in
  use` on a single rank (`WorkerDict pid=228086`, node `172.28.44.100`), which took down the
  whole job via Ray. **Zero training-step metric lines exist in this log** — no `actor/loss`,
  `grad_norm`, or `critic/score/mean` — so this run produced neither the weight-broadcast
  diagnostic data nor any reward-trend data, and the original reward-collapse hypothesis remains
  completely untested.
- **Not yet investigated**: whether this backward-pass OOM is a new regression from converting
  the fork branch swap to a static patch (functionally, the two should be identical — the earlier
  diff/compare-API check found only the one file differs, and that file's content is unchanged by
  the conversion), a resource-boundary flake specific to this node/allocation (92.98/95 GiB is
  extremely tight — consistent with this repo's other one-off node-squat OOMs, e.g. run 3185487's
  rank-0 pre-FSDP squat and run 3152802's TP=32 hang, both later confirmed non-repeating on
  retry), or something else entirely (e.g. a larger-than-usual dynamic-bsz micro-batch on this
  particular first mini-batch). Since run 3185943 — functionally the same rank0-load/broadcast
  mechanism, via the live fork branch instead of this patch — ran 47 steps with no OOM at all,
  a code-level regression from the patch conversion looks unlikely on its face, but hasn't been
  ruled out with certainty.
- **Fix**: none yet for the OOM. The diagnostic-apply bug fix above should be sufficient on its
  own to collect the `[DIAG-FSDP2-BROADCAST]` hash data on a retry, since that instrumentation
  runs during `actor_init_model()` — before this OOM's crash point in `update_actor` — so even an
  identical OOM recurrence wouldn't prevent capturing it. Whether to also investigate/mitigate the
  OOM before resubmitting, or resubmit unmodified first (same "retry once to check if it's a
  one-off" approach already validated elsewhere in this repo) is an open decision, not yet made.
- **Commit**: not committed.