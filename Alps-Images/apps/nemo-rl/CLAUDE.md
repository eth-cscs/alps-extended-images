# CLAUDE.md -- Alps-Images/apps/nemo-rl

Working notes for the NeMo-RL app image and the training recipes under `example/`. Loaded by
Claude Code when it works on files in this directory; repo-wide rules are in `/AGENTS.md`.
The verl recipes and their (much longer) history live on branch `Add-megatron-rl-recipes`
(`.claude/CLAUDE.md` there).

## Working rules (same as the verl recipes)

- **Always ask before submitting a job.** Submit/monitor through FirecREST v2
  (https://api.svc.cscs.ch/ml/firecrest/v2/docs, system `clariden`); credentials are in
  `~/F7T_Credentials` -- never print or output the secret. On failure: download the log, diagnose,
  fix, then ask again before resubmitting.
- One "Run log" entry per job, newest last: log path, symptom, root cause, fix, commit.
- Shell hazards in these launchers: the per-node body is a single-quoted `srun ... bash -c '...'`
  -- one apostrophe anywhere inside (comments included) ends it; the config heredocs are unquoted
  -- a backtick or bare `$(` inside them is live shell. Scan both after every edit, plus `bash -n`.
- `example/train-gsm8k-chonk.sh` (Apertus 2 "Chonk", swiss-ai/Nemo-RL) is the image author's own
  launcher; it also overlays Lens and soundfile on this image tag.

# Training Apertus v1.5 on NeMo-RL (`example/train-gsm8k-apertus-v1.5-70B.sh`)

Apertus-v1.5-70B GRPO on GSM8K with **NeMo-RL instead of verl**, CSCS Alps, 16 nodes x 4 GH200.
Script: `Alps-Images/apps/nemo-rl/example/train-gsm8k-apertus-v1.5-70B.sh`. **Created 2026-10-09; not yet
validated -- three shakedowns so far, all stopped in the rank-0 preflight (Run log below).**

- **Image**: `nemo-rl-cuda:alps7-dev-cfcfede4262bfb99f4659aad30992344`, built from
  `Alps-Images/apps/nemo-rl` in PR #64 (`andresnowak:app/nemo-rl`, an Apertus-2 image). It ships the
  runtime stack (ray 2.56.1, vLLM `0.28.0+apertus2` from `swiss-ai/vllm@d4d41485`, TE 2.17,
  transformers 5.17.0, flashinfer 0.6.16.post3) but **no NeMo-RL / Megatron-Bridge / Megatron-LM
  source** -- those are external checkouts on `PYTHONPATH`. Image defaults that matter:
  `NEMO_RL_PY_EXECUTABLES_SYSTEM=1` (Ray actors run the container interpreter, no per-worker uv
  venvs), `NRL_CONTAINER=1`, and an `LD_PRELOAD` of the TorchMemorySaver CUDA hook (the script
  clears it: unused by vLLM/Megatron, and the recipe runs Megatron with `expandable_segments`).
- **Source**: `Alvorecer721/Nemo-RL@ce1c6818` (the fork with the Apertus recipes), cloned once
  into `${TRAINING_HOME}/nemo-rl-src/<sha12>` with only two submodules: Megatron-Bridge
  (`Alvorecer721/Megatron-Bridge@5f63da5a`) and its nested Megatron-LM. The bridge
  `safe_config_loader` filelock is replaced with `nullcontext` at clone time (Lustre flock hazard).
- **Recipe**: the fork's `grpo-apertus1p5-70b-16n4g-tp2pp4-gsm8k-2k-windowed-optimized.yaml`
  (reference run job `3352055`) inherited via `defaults:`; the script overrides only node split,
  steps and logging. 12 trainer nodes (TP2/PP4 -> DP6) + 4 rollout nodes (vLLM TP=4), 48x16, 92
  steps, thinking off, `\boxed{}` answers (`bracket_math` env, composite reward), refit
  `nccl_reshard`, entrypoint `python3 -m examples.run_grpo_single_controller`.
- **No Apertus-1.5 patches needed** (unlike verl): the SFT checkpoint
  `ap1p5-70b-sft-262k-2700_corr` has `architectures: [ApertusForCausalLM]`, `model_type: apertus`
  (read from its `config.json`), which stock transformers, the image vLLM registry and the fork
  `ApertusBridge` (`source="ApertusForCausalLM"`) all handle. NeMo-RL's own fail-closed vLLM
  patches (`_patch_vllm_xielu_static_constants`, `_patch_vllm_apertus_static_xielu_loader`) were
  checked against the image's vLLM source and both anchors match.
- **Open risks for the first run**: (1) version skew -- the fork locks vLLM 0.29 / ray 2.58 /
  TE 2.18 / transformers 5.12.1 / torch 2.13, the image has 0.28+apertus2 / 2.56.1 / 2.17 / 5.17.0
  and the fork's other vLLM patches target 0.25-0.29 source shapes; (2) Megatron-LM comes from the
  bridge fork checkout against the image's TE 2.17; (3) no CUDA xIELU kernel in the image (bridge
  falls back to jit-fused Python); (4) Ray started verl-style without `ray.sub`'s
  `nvlink_domain_*`/`topo_rank` resources (`cluster.segment_size` is null, so not required);
  (5) the fork was never run with system-interpreter workers on this image. The script's rank-0
  preflight (versions, imports of the worker modules, `MasterConfig` validation) runs before
  `ray start` and is fatal, so (1)/(2) import-level breakage shows in the first minutes.

## Run log

### Run `3621638` — 2026-10-09 — first shakedown: checkout OK, preflight fails on `nemo.lens`

- **Log**: `~/Downloads/slurm-3621638.out`. 16 nodes, `clariden`, `TOTAL_TRAINING_STEPS=1`. ~13 min
  queued, FAILED after 170 s in the rank-0 preflight (before Ray).
- **Worked**: one-off checkout (NeMo-RL `ce1c6818`, Megatron-Bridge `5f63da5a`, Megatron-LM
  `9ae528ca`, filelock patch). Image versions as printed: torch `2.13.0a0+8145d630e8.nv26.6`,
  vllm `0.28.0+apertus2`, transformers `5.17.0`, ray `2.56.1`, transformer_engine `2.17.0`,
  flashinfer-python `0.6.16.post3`, openai `2.7.2`. `megatron.core`, `megatron.bridge`, `nemo_rl`
  and the Apertus bridge import from the Lustre checkout against the image.
- **Symptom**: `ModuleNotFoundError: No module named 'nemo'` at
  `nemo_rl/telemetry/instrumentation.py:47` (`from nemo.lens import ...`), reached through
  `nemo_rl.models.policy` -> `nemo_rl.data_plane` -> `nemo_rl.telemetry`.
- **Root cause**: the fork needs `nemo-lens>=0.2.0` (git rev `b0f977d4`); the image has no
  importable `nemo` package (PR #64's Containerfile installs nemo-lens 0.1.0, so this image tag
  either predates that layer or the install is not importable -- not probed).
- **Fix (unverified)**: nemo-lens is pure Python (`src/nemo/lens`, needs only opentelemetry, in the
  image) -- cloned at the fork's pinned rev next to the NeMo-RL checkout and put on `PYTHONPATH`
  for the driver/trainer and, via `vllm_cfg.env_vars.PYTHONPATH`, the vLLM workers (the recipe
  restricts those to the checkout only). Preflight now tries every worker module and reports all
  failing imports in one run.
- **Second bug fixed from the same log**: workers did `sleep 15; ray start --address` while the
  head was still in the preflight (`Failed to connect to GCS ... within 5 seconds` on 15 nodes);
  with `|| true` they would have exited and never joined. Workers now wait for a
  `${RUN_DIR}/ray_head_ready` marker the head touches after `ray start --head`.
- **Non-fatal warnings seen**: `triton_kernels.matmul_ogs` missing, `nixl_ep` import failure
  (`torch reports CUDA 13 but nixl-cu13 is not installed`), modelopt vLLM plugin `FusedMoE`
  AttributeError.
- **Commit**: not committed.

### Run `3621707` — 2026-10-09 — nemo-lens fix worked; preflight stops on a transformers version assert in the fork

- **Log**: `~/Downloads/slurm-3621707.out` (181 lines). ~11 min queued, failed in the rank-0 preflight.
- **Worked**: nemo-lens checkout (`lens-b0f977d414b2`), `import nemo.lens OK`; 8 of 12 preflight
  imports OK (megatron.core/bridge, nemo_rl, Apertus bridge, both vLLM workers, bracket_math env).
- **Symptom**: 4 imports fail with one cause --
  `nemo_rl/models/policy/__init__.py:53  assert PkgVersion(transformers.__version__) < PkgVersion("5.13.0")`
  -> `AssertionError: transformers 5.17.0 detected. The deepseek_v3 tokenizer-blocklist patch was
  verified against <5.13`.
- **Root cause**: a maintenance reminder in the fork guarding a DeepSeek-V3/Moonlight
  `AutoTokenizer.from_pretrained` workaround (try the normal load first, strip `deepseek_v3` from two
  registries only on failure). Irrelevant to Apertus; both registry symbols it imports still exist
  in transformers 5.17.0 (checked in the v5.17.0 source). The other `<5.13` gate
  (`nemo_rl/transformers_compat.py`) is a silent no-op above 5.13.
- **Fix (unverified)**: rank 0 rewrites that assert into a printed warning in the shared checkout
  before the preflight (idempotent, fatal if the anchor is missing).
- **Commit**: not committed.

### Run `3621888` — 2026-10-09 — assert fix worked; preflight stops on missing `soundfile`

- **Log**: `~/Downloads/slurm-3621888.out` (148 lines). ~8 min queued, FAILED after 105 s (preflight).
- **Worked**: `Relaxed the transformers<5.13 assert`; 10 of 12 preflight imports OK, including
  `megatron_policy_worker`, both vLLM workers and `sync_rollout_actor`.
- **Symptom**: `nemo_rl.algorithms.async_utils` and `examples.run_grpo_single_controller` fail with
  `ModuleNotFoundError: No module named 'soundfile'` (`nemo_rl.algorithms.grpo` ->
  `nemo_rl.data.datasets` -> `response_datasets/audiomcq.py:20`, imported unconditionally).
- **Root cause**: image tag `alps7-dev-cfcfede4...` predates PR #64 commit `4a59429` ("Add validated
  NeMo-RL runtime packages", 2026-10-08), which is what adds `soundfile`, `nemo-lens` and the rest of
  `sources/package-additions.in`. That also explains run `3621638`'s missing `nemo` package.
- **Fix (unverified)**: `EXTRA_PIP_PACKAGES="soundfile==0.14.0"` installed once, `--no-deps`, into
  a shared `--target` dir (`nemo-rl-src/extra-site-v1`) and appended to `PYTHONPATH` (driver, trainer
  and vLLM workers). A static scan of the pinned checkout gives 40 third-party top-level modules in
  the module-level import closure of the entrypoint + worker modules; compared against the image's
  install list only `nemo` and `soundfile` are absent. The preflight now `find_spec`s all of them
  (plus `cffi`, soundfile's one dependency) and prints the missing set.
- **Simpler alternative**: an image rebuilt from the current PR #64 head would need neither the
  nemo-lens checkout (if 0.1.0 satisfied the fork, which wants >=0.2.0 -- it may not) nor this.
- **Commit**: not committed.

### Image bump — 2026-10-09 — `alps7-dev-cfcfede4...` -> `alps7-dev-7eab7fc435699b52c9668a663afde4b3`

- Newest `nemo-rl-cuda` tag in the registry (created 2026-10-08, label
  `com.cscs.alps.git.commit=577502c`, i.e. the current image-affecting head of PR #64, with a
  `-tested-` marker). Listing: `https://jfrog.svc.cscs.ch/artifactory/api/storage/docker-group-csstaff/alps-images/nemo-rl-cuda`
  (anonymous read works; `?properties` on a tag's `manifest.json` shows the build commit).
- It includes `sources/package-additions.in` (soundfile 0.14.0, nemo-lens 0.1.0, ...), so
  `EXTRA_PIP_PACKAGES` now defaults to empty and the overlay step is skipped. The nemo-lens
  checkout stays: the fork needs >= 0.2.0 and the checkout is first on `PYTHONPATH`.
- Everything in "Recipe"/"Open risks" above that names versions was read from the old tag;
  re-read them from the next run's preflight.

### Run `3622335` — 2026-10-09 — new image: preflight, Ray, model load and vLLM all clean; first weight refit fails with NCCL `InvalidUsage (5)`

- **Log**: `~/Downloads/slurm-3622335.out` (first 3.6 MB only -- `ops/download` returned 504 once the
  log passed that size) and `~/Downloads/slurm-3622335.tail.out` (last 3000 lines, has the error).
  Image `alps7-dev-7eab7fc4...`. ~15 min queued, FAILED after 1165 s (~19.5 min).
- **Worked, all for the first time**: preflight (`third-party modules not importable: none`,
  12/12 imports, `config OK`), 16/16 nodes in Ray (the head-ready marker fix), 48/48
  `MegatronPolicyWorker`s initialised, 4 vLLM engines up on `v0.28.0+apertus2`
  (`Resolved architecture: ApertusForCausalLM`), checkpoint load from the HF snapshot through the
  bridge. No OOM, no Lustre lock errors.
- **Symptom**: at the first trainer->vLLM refit (`refit: watchdog armed, deadline 300.0s`), every
  vLLM TP worker raises in
  `nemo_rl/weight_sync/xferdtensor.py:116 _communicator_supports_device_api` ->
  `getattr(communicator, "device_api_support", False)` -> NCCL `InvalidUsage (5): invalid usage`;
  the trainer side then fails `MegatronPolicyWorker.nccl_reshard_refit()` with
  `RayTaskError(NCCLError)` and the driver exits 1. Every rank also logs
  `NCCL WARN NCCL library is too old. This application was compiled with NCCL version 2.31.2, but
  is running with NCCL library version 2.30.7`.
- **Root cause**: the image ships `nccl-extensions` 0.1.0, so `nccl.m2n.reshard` imports and the
  fork probes the refit communicator for device-API support; the nccl4py binding (built against
  NCCL 2.31.2) errors on the image's runtime NCCL 2.30.7 instead of returning "unsupported".
  `getattr(..., False)` only absorbs `AttributeError`, so the `NCCLError` propagates. On the older
  tag this would not have triggered (no `nccl.m2n`, the fork falls back by itself).
- **Fix (unverified)**: `export NRL_XFERDTENSOR_PYTHON=1` -- the fork's own switch for the Python
  exact-transfer reshard, the path it takes anyway when the native op is absent or the fabric has
  no GPU-initiated networking (Slingshot: `NET/OFI GIN only supports RDMA`).
- **Image-side note for PR #64**: nccl4py / nccl-extensions are built against a newer NCCL than the
  image runs; worth aligning, independent of this recipe.
- **Log volume**: `NCCL_DEBUG=WARN` (fork launcher default) makes every rank repeat the OFI/GIN
  warnings -- several MB in 15 min. Monitor with `ops/tail`, not `ops/download`.
- **Commit**: see `git log` (fix committed with this entry).

### Run `3622451` — 2026-10-09 — **FIRST COMPLETE STEP: `train step 1/1`, `SC run complete`** (1-step shakedown)

- **Log**: `~/Downloads/slurm-3622451.tail.out` (last 3000 lines; the full log is > 5 MB, on the
  cluster at `/users/palmee/slurm-3622451.out` and `<RUN_DIR>/run.log`). Image `alps7-dev-7eab7fc4...`,
  16 nodes, `TOTAL_TRAINING_STEPS=1`. ~5 min queued. Slurm state at time of writing: `COMPLETED 0 889`
  (state exitCode elapsed_s).
- **Result**: the `NRL_XFERDTENSOR_PYTHON=1` fix cleared the refit; rollout, training, weight sync
  and the step-1 data-plane checkpoint all completed.
- **Step-1 metrics**: `loss` 0.0813, `grad_norm` 0.157, **`gen_kl_error` 0.00066** (fork's healthy
  bar is < 0.002: generator and trainer agree after the refit), `policy_kl_error` 0.00067,
  `sampling_importance_ratio` 0.9999, `reward` mean 1.021 (max 1.1, min 0, median 1.077 -- composite
  reward, 1.0 correct + 0.1 format), 768 valid samples, 582k valid tokens, `lr` 1e-6.
- **Step-1 timing**: total 137.0 s = generation 67.0 s + policy training 60.6 s + checkpointing
  4.4 s + weight sync 3.6 s. (First step; includes warm-up.)
- **Teardown noise, not failures**: `Error in GPU monitoring collection loop ...` and a wandb
  `atexit` `ConnectionResetError: Connection lost` after `SC run complete`.
- **Still unvalidated**: more than one step (windowed async sampler across weight versions,
  repeated refits, memory over time), the full 92-step run, and throughput vs. the fork's own
  image (no CUDA xIELU kernel here).
- **Commit**: recorded with this entry.

# Megatron-generation variant (`example/train-gsm8k-apertus-v1.5-70B-megatron-gen.sh`)

Created 2026-10-09 as a copy of `train-gsm8k-apertus-v1.5-70B.sh` with Megatron-Core's inference
engine as the rollout backend instead of vLLM. **UNTESTED -- no run yet.** Keep the two scripts in
sync by hand: `diff` them from `export NEMORL_IMAGE` down; the only intended differences are the
experiment name, `GEN_TP`, the generation block of the YAML, the two `VLLM_*` exports and the
preflight module list.

- **Config delta** (child YAML): `policy.generation.backend: megatron`,
  `refit_transport: mcore`, `mcore_generation_config.{tensor_model_parallel_size: 4,
  pipeline_model_parallel_size: 1, refit_backend: nccl, kv_cache_management_mode: recompute}`,
  `async_rl.generation_fleet_health.refit_timeout_s: null` -- the same four generation keys as the
  fork's `grpo-qwen2.5-math-1.5b-instruct-1n4g-megatron_generation-noncolocated-single-controller-async-lag4.yaml`,
  plus the generator layout. Everything else (CUDA graphs `local`/`block`, `buffer_size_gb: 10`,
  `max_tokens: 16384`, `logprobs_mode: processed_logprobs`) is the fork default from
  `grpo_math_1B.yaml`. The generator's `megatron_cfg` is the trainer's merged with
  `mcore_generation_config` (`merged_inference_megatron_cfg`), activation checkpointing forced off.
- **What it does and does not remove** (checked in the fork source, not assumed):
  - removed: vLLM, NeMo-RL's runtime source patches to vLLM, the Megatron->HF-schema conversion
    on every refit (`mcore` transport = `megatron.core.resharding`, Megatron-to-Megatron).
  - NOT removed: **Megatron-Bridge** -- `megatron_worker.py` builds the generation model from a
    bridge provider (`megatron.bridge.models.gpt_provider`), and the HF checkpoint import on both
    sides is the bridge; **transformers** -- `megatron_generation.py` imports it at module level
    (tokenizer, processor), and tokenizer / chat template / HF config come from it everywhere.
    Dropping them would need a pre-converted Megatron checkpoint plus a non-HF tokenizer path,
    which this fork does not offer.
- **Open risks**: no Apertus Megatron-generation recipe exists in the fork (Qwen, Llama,
  Nemotron only). `ApertusModelProvider` sets its own `transformer_layer_spec`
  (`apertus_layer_spec`, xIELU as a module activation), so `_configure_inference_optimized_layer_spec`
  leaves it alone and the generator runs regular TE layers, not `inference_optimized` -- which also
  rules out `refit_transport: nccl_reshard` for this backend (it requires `inference_optimized`).
  Generation CUDA graphs with the xIELU module, native refit TP2/PP4 -> TP4/PP1, and rollout
  throughput vs vLLM (KV buffer is only 10 GB by default) are all unexercised.

## No source edits to Megatron-Bridge / Megatron-LM — 2026-10-09 (unverified)

- Both launchers used to rewrite the bridge's `safe_config_loader.py` (`filelock` ->
  `nullcontext`) at clone time. Replaced by the bridge's own knob:
  `MEGATRON_CONFIG_LOCK_DIR=/tmp/mbridge_locks_<jobid>` (its default lock dir is
  `~/.cache/huggingface`, i.e. Lustre home, where flock fails in-container; `/tmp` is node-local).
- The checkout dir is now `nemo-rl-src/<sha12>-clean`, so the next job makes a fresh clone with the
  bridge and Megatron-LM untouched. The old `nemo-rl-src/<sha12>` (with the filelock edit) is what
  runs `3621638`-`3622703` used; it can be deleted once nothing is running from it.
- Still edited at runtime, in the NeMo-RL checkout only: the `transformers < 5.13` assert in
  `nemo_rl/models/policy/__init__.py`.
- What cannot be avoided is using a *fork* of the bridge: Apertus support (provider, bridge, xIELU)
  exists only in `Alvorecer721/Megatron-Bridge`; upstream `NVIDIA-NeMo/Megatron-Bridge` `main` has no
  `models/apertus*` (checked 2026-10-09). That ends only when the Apertus bridge is upstreamed.

# Chonk (Apertus 2) with Megatron generation on near-vanilla libraries (`example/train-gsm8k-chonk-megatron-gen.sh`)

Created 2026-10-09. **UNTESTED on cluster.** Goal (user): train Chonk without vLLM and with as few
library changes as possible -- vanilla NeMo-RL and Megatron-Bridge plus only what is strictly
needed; the Megatron-LM fork is accepted as unavoidable.

## What Chonk is (from the checkpoint `config.json`, read 2026-10-09)

`architectures: [Apertus2KDAForCausalLM]`, `model_type: apertus2`, config class shipped as remote
code (`configuration_apertus2.py`). 42 layers mixing `linear_attention` (KDA, Kimi Delta Attention)
and `full_attention`; MoE from layer 3 on, 256 routed + 1 shared experts, 8 active, latent MoE
(`moe_latent_size` 1792), quantile-balancing router; `hidden_act: sssglu`, sandwich norm, QK norm,
embedding/residual multipliers; hidden 3584, vocab 200064. Roughly 70B parameters, ~69B of them
experts (my arithmetic from the config, not measured).

## Sources and the delta from vanilla

| Library | Base (vanilla) | Added |
|---|---|---|
| NeMo-RL | `NVIDIA-NeMo/RL@f196ef0a` (the commit the author's example pins; `swiss-ai/Nemo-RL` is 0 commits ahead of upstream) | `patches/nemo-rl-chonk-megatron-gen-minimal.patch`: 2 files, +17/-1 |
| Megatron-Bridge | `NVIDIA-NeMo/Megatron-Bridge@07722453` (merge-base of the swiss-ai fork with upstream) | `models/apertus2/` (6 new files) copied verbatim from `swiss-ai/Megatron-Bridge@f3bc8e7e`; `patches/bridge-apertus2-shared-minimal.patch`: 5 files, +91/-46 |
| Megatron-LM | -- | `swiss-ai/Megatron-LM-MoE@a9e0c497` (head of its PR #88), untouched |
| nemo-lens | `NVIDIA-NeMo/Lens@b85578fc` (NeMo-RL's pin) | -- |

Both patches were generated from the forks' own commits with git (not hand-written), carry a
header listing each hunk and its reason, apply cleanly to the vanilla bases, and are embedded in
the launcher byte-identically (verified). The launcher's checkout step was run locally end to end.

**Bridge hunks kept, and why** (each checked against the source):
- `models/__init__.py`: import/registration of the Apertus2 bridge (the `register_bridge`
  decorator only fires on import).
- `models/deepseek/__init__.py`, `models/nemotron_omni/__init__.py` (+ their guards in
  `models/__init__.py`): the vanilla bridge imports `megatron.core.transformer.mla_qk_norm_config`
  and the dynamic-resolution context-parallel helpers at package import; the Megatron-LM fork has
  neither (verified absent at `a9e0c497`), so `import megatron.bridge.models` would fail.
- `models/model_provider.py`: keep the quantile-balancing buffer `qb_beta` in fp32 under the
  mixed-precision wrapper.
- `training/optim.py` (first hunk only): dispatch `optimizer: md_decoupling` to the fork's
  optimizer factory. Needed only because the recipe keeps the author's optimizer; drop it if
  training with Adam.

**Bridge changes of the fork deliberately left out**:
- `conversion/auto_bridge.py`, `training/model_load_save.py`, `training/mlm_compat/arguments.py`,
  `training/checkpointing.py`: loading native / legacy Megatron checkpoints (no `run_config.yaml`,
  legacy offloading schema). Not needed when starting from the HF checkpoint.
- `conversion/utils.py`, `conversion/model_bridge.py` (incl. swiss-ai bridge PR #4): per-expert
  weight views for `OffloadingExpertsMLP`, and moving CPU master weights to CUDA before NCCL
  collectives in the Megatron->HF export. Needed only with expert offloading and/or the vLLM
  refit; this recipe has neither.
- formatting-only hunks in `optim.py`, docs, scripts, tests.

**NeMo-RL hunks kept**: `megatron_worker.py` from swiss-ai PR #1 -- pass
`KDAInferenceStateConfig.from_model(...)` to Megatron-Core's `InferenceConfig` (vanilla only wires
the Mamba state config, so KDA layers would have no per-request state), and give the data-parallel
inference coordinator the node IP instead of the container hostname; plus the local one-line
`transformers < 5.13` assert relaxation (image has 5.17).
**Left out**: upstream PR #2517 (vLLM DP), swiss-ai PR #5 (checkpoint load under `no_grad`), the
tokenizer rework and `max_requests` key of PR #1.

## Why Megatron generation is feasible here (checked, not assumed)

- Every `megatron.core` symbol NeMo-RL's `models/generation/megatron/*` imports resolves in the
  fork (40/40, static check).
- The fork's `megatron/core/ssm/kimi_delta_attention.py` implements `_dynamic_inference` /
  `_dynamic_inference_decode` (needs `fla.ops.kda.fused_recurrent_kda_fwd`; the image ships
  flash-linear-attention) and raises `NotImplementedError` for anything but the dynamic engine;
  `InferenceConfig.kda_inference_state_config` and `DynamicInferenceContext.has_kda` exist.
- The image was described by its author as "Megatron policy + Megatron generation backend".

## Recipe = the author's `grpo-chonk-vllm.yaml` with these differences

- `policy.generation`: `backend: megatron`, `refit_transport: mcore`,
  `mcore_generation_config: {expert_model_parallel_size: 8, tensor/pipeline 1, refit_backend: nccl,
  cuda_graph_impl: none, inference_cuda_graph_scope: none, num_cuda_graphs: null,
  use_cuda_graphs_for_non_decode_steps: false, kv_cache_management_mode: recompute,
  logprobs_mode: raw_logprobs}`; `vllm_cfg` / `vllm_kwargs` removed.
- `megatron_cfg.model_overrides` (coarse-grained expert offloading, in-place FP8 params) removed.
- Everything else (24 nodes = 8 train EP16 (EP32 since run `3640152`) + 16 generation EP8, 32x8 rollouts, 2 steps,
  md_decoupling optimizer, sync GRPO via `examples/run_grpo.py`) unchanged.

## Open risks, in the order a first run would meet them

1. Preflight: the patched vanilla bridge importing against the Megatron-LM fork, and
   `AutoBridge.from_hf_config` resolving the checkpoint to the Apertus2 provider.
2. Trainer memory without expert offloading (the author enabled it; if this OOMs, the next step is
   the offloading patch group: bridge PR #4 + the two `conversion/` files).
3. Whether `md_decoupling` needs anything else from the fork bridge (the author's job 5003401
   reached generation with the full fork, not with this subset).
4. KDA + latent MoE under the dynamic inference engine, eager.
5. Native `mcore` refit EP16 -> EP8 for the fork's custom modules.
6. If swiss-ai PR #5 (`no_grad` checkpoint load) turns out to be load-bearing.

## Optimizer switched to Adam — 2026-10-09 (user decision, TEMPORARY)

**Superseded 2026-10-15**: md_decoupling is back (see "what must be restored", item 2). The bridge
delta is again the minimal patch + `bridge-md-decoupling-optimizer.patch`.

- **TODO: change back to `md_decoupling`.** Chonk was pretrained with the Megatron-LM fork's
  `md_decoupling` optimizer (Muon-style orthogonal updates, hypersphere-constrained weights with
  learned gains); Adam does not maintain those constraints, so this is for bringing the RL loop up,
  not the intended training setup.
- Recipe: `optimizer: adam` with the author's `lr` 2e-6 / `min_lr` 2e-7 / betas 0.9, 0.95 /
  `weight_decay` 0 and the author's scheduler; everything else is NeMo-RL's stock Adam config
  (`use_distributed_optimizer: true`, `use_precision_aware_optimizer: true`).
- Consequence for the library delta: the `training/optim.py` hunk is no longer applied. The bridge
  patch is now **4 files, +77/-45**; the hunk lives on as
  `example/patches/bridge-md-decoupling-optimizer.patch` (not applied by any launcher, verified to
  apply on top of the minimal patch). `SRC_VERSION` bumped to `v2` so the shared checkout is rebuilt.
- To switch back: restore the optimizer block from `grpo-chonk-vllm.yaml`, embed/apply the
  optimizer patch after the minimal one, bump `SRC_VERSION`. Everything above this entry that says
  "5 files" or lists `md_decoupling` as kept describes the state before this change.

## Chonk run log

### Run `3622848` — 2026-10-09 — first Chonk attempt: checkouts, patches and preflight all pass; stops on a recipe validation error

- **Log**: `/users/palmee/slurm-3622848.out` on clariden (tail fetched via `ops/tail`). 24 nodes,
  image `alps7-dev-7eab7fc4...`, Adam, Megatron generation. ~15 min queued, failed ~4 min in.
- **Worked**: the one-node checkout step on the cluster (`Source checkouts ready at
  .../src-v2-f196ef0a-07722453-a9e0c497`, both patches applied); the preflight -- every import of
  the patched vanilla bridge + vanilla NeMo-RL against the Megatron-LM fork, and
  `architectures: ['Apertus2KDAForCausalLM'] -> provider Apertus2ModelProvider`; `config OK`.
  So the 4-file bridge patch + the `apertus2` package + the 2-file NeMo-RL patch are sufficient at
  import / registration level.
- **Symptom**: `ValueError: grpo.async_grpo.recompute_kv_cache_after_weight_updates=False conflicts
  with policy.generation.mcore_generation_config.kv_cache_management_mode='recompute'`
  (`nemo_rl/algorithms/grpo.py`, Megatron-generation consistency check at setup).
- **Root cause**: my recipe set `kv_cache_management_mode: recompute` (copied from the fork's
  single-controller Qwen recipe, where the matching loop flag is on); this recipe runs synchronous
  GRPO with the flag at its default `false`.
- **Fix**: override removed, default `persist` (no request is in flight across a refit in sync
  GRPO). Unverified.
- **Not reached**: Ray actors, model construction, HF import, memory, generation, refit.

### Run `3622931` — 2026-10-09 — trainer model builds and imports the HF checkpoint without offloading; OOM when the reference model is added

- **Log**: `~/Downloads/slurm-3622931.tail.out` (last 6000 lines) + first 2.6 MB in
  `~/Downloads/slurm-3622931.out`. 24 nodes, FAILED after 967 s. Reused the `src-v2` checkout.
- **Worked**: past the setup validation; Ray 24/24; all 32 `MegatronPolicyWorker`s built the
  Apertus2 model with Adam and imported the HF checkpoint through the vanilla bridge + `apertus2`
  package (`[GPU_DIAG] after_hf_import` and `after_model_setup` on every rank).
  **Trainer memory without expert offloading, per GPU at `after_model_setup`: NVML used ~68.5 GB
  of 97.9 GB, PyTorch allocated ~49 GB / reserved ~64 GB** (EP16, DP2, Adam distributed optimizer).
- **Symptom**: `torch.OutOfMemoryError: Tried to allocate 12.00 GiB ... 79.63 GiB allocated by
  PyTorch` in `megatron_policy_worker.py:759 self.move_model(self.model, "cuda")` ->
  `param_and_grad_buffer.py reload_from_cpu` (`grad_data` resize), on 3 ranks; the rest died with
  `ActorDiedError`.
- **Root cause**: with `loss_fn.reference_policy_kl_penalty > 0` the worker moves the policy's
  param/grad buffers to CPU, builds a full second (reference) model on the GPU to load the
  checkpoint, copies its state dict to CPU and then moves the policy back
  (`megatron_policy_worker.py` step 5 / `setup_reference_model_state`). The reference model's GPU
  memory is not explicitly released before the policy comes back, and the optimizer state never
  left the GPU, so the two overlap: ~80 GB allocated when the policy's 12 GiB grad buffer is
  re-created. The author's recipe avoids this by keeping the experts off the GPU (offloading).
- **Fix (user to confirm)**: `REFERENCE_KL_PENALTY` knob, default `0.0` -> no reference model.
  TEMPORARY, changes the objective (author: 0.01). Alternatives not taken: a small NeMo-RL patch
  freeing the reference model (`del` + `gc.collect()` + `empty_cache()`) before the policy is moved
  back -- plausible but unverified that nothing else holds it; or the expert-offloading patch group.
- **Not reached**: generation workers' model build, KDA inference, refit, any step.

## Chonk: what must be restored before the final run (user decisions, 2026-10-09)

Both are disabled only to bring the RL loop up; neither is the intended training setup.

1. **KL penalty against the starting policy** -- `REFERENCE_KL_PENALTY` is `0.0`; the final run
   needs the author's `0.01`. DONE in `3cd896f`, verified in run `3655036` (fits at EP32).
   **Back to `0.0` (user, 2026-10-14)** to free the reference model's ~32 GB GPU copy for the
   native refit (trainer OOM in `3659594`); the native refit is kept because the larger models
   to come need it. Restoring 0.01 needs the reference copy freed (NeMo-RL patch) or more room.
   **Kept at `0.0` (user, 2026-10-15)**: "we might don't need it"; no longer a final-run blocker
   unless the user brings it back.
2. **`md_decoupling` optimizer** -- RESTORED in the launcher (2026-10-15, `SRC_VERSION` `v7`): the
   author's optimizer block verbatim, `example/patches/bridge-md-decoupling-optimizer.patch`
   embedded and applied after the minimal bridge patch, `emerging_optimizers` and
   `megatron.core.optimizer.md_decoupling` added to the preflight imports. Checked offline: both
   bridge patches apply to `07722453`; every key of the merged optimizer block (base config + recipe
   + NeMo-RL's added fp8/overlap keys) is a field of the fork's `OptimizerConfig`; the Adam-only
   `exp_avg_dtype: bf16` lines are gone (the fork asserts fp32 moments without the precision-aware
   optimizer); NeMo-RL's `move_optimizer` handles the `LayerWiseDistributedOptimizer` (a
   `ChainedOptimizer`); the learned gains live in optimizer state, so the parameters the refit
   ships are the effective weights. Not yet run.

Also wanted back (user, 2026-10-09), not strictly a final-run blocker:

3. **Native Megatron-to-Megatron refit** -- `refit_transport: mcore` with
   `mcore_generation_config.refit_backend: nccl`. Currently `null` (NeMo-RL's default collective
   through the bridge mapping) because vanilla NeMo-RL `f196ef0a` passes `execution_batch_bytes` to
   `megatron.core.resharding.refit.{prepare_swap_model_weights,swap_model_weights}` and the
   Megatron-LM fork `a9e0c497` does not accept it (run `3623177`). Enabled again with
   `example/patches/mlm-refit-execution-batch-bytes.patch` on the fork (decision 2026-10-14, see
   below); untested.

### Run `3623177` — 2026-10-09 — KL penalty 0: trainer and generator both built; native `mcore` refit API mismatch at synchronizer init

- **Log**: `~/Downloads/slurm-3623177.tail.out` (last 8000 lines). 24 nodes, FAILED after 513 s
  (faster than `3622931`: the converted Megatron checkpoint was already cached).
- **Worked**: with `REFERENCE_KL_PENALTY=0.0` no reference model is built and the trainer OOM is
  gone. `grpo.setup` got past creating both the policy and the generation worker groups (the
  failure is in `init_megatron_weight_synchronizer`, which runs after them), i.e. the dedicated
  Megatron generation model was constructed on the 16 generation nodes.
- **Symptom**: `TypeError: prepare_swap_model_weights() got an unexpected keyword argument
  'execution_batch_bytes'` in `megatron_worker.py:1410 init_collective_mcore_generation`.
- **Root cause**: version drift between vanilla NeMo-RL and the Megatron-LM fork on the native
  refit API. NeMo-RL `f196ef0a` passes `execution_batch_bytes` to
  `megatron.core.resharding.refit.prepare_swap_model_weights` / `swap_model_weights` (3 call
  sites); the fork's functions do not take it. The author's recipe never exercises this path
  (`refit_transport: null`).
- **Fix (unverified)**: `refit_transport: null` (NeMo-RL's default collective through the bridge
  mapping -- same trainer-side path as the author's vLLM recipe), `refit_backend` override removed
  (only valid with `mcore`), and the author's `NRL_REFIT_BUFFER_MEMORY_RATIO=0.005` /
  `NRL_REFIT_NUM_BUFFERS=1` restored (env + `megatron_cfg.env_vars`). Zero library change.
  Alternative not taken: drop the kwarg at NeMo-RL's three call sites to keep the native refit --
  a further NeMo-RL patch with unknown further drift behind it.
- **Consequence**: the per-step Megatron -> HF-schema -> Megatron conversion is back in the refit
  (the thing the native transport would have avoided); vLLM is still gone.

### Run `3623722` — 2026-10-09 — default refit works, first step entered; eager generation crashes idle expert-parallel engines

- **Log**: `~/Downloads/slurm-3623722.tail.out`. 24 nodes. Reached `Step 1/2` ~13 min in, then
  hung (log idle; engines dying) -- state RUNNING when this was written.
- **Worked, all for the first time on Chonk**: weight-synchronizer init with
  `refit_transport: null`; the initial trainer -> generator refit (`refit: watchdog NOT armed`,
  then every generation rank `prepare_for_generation: engine already serving`); the 64 Megatron
  inference engines registered with the data-parallel coordinator; `Step 1/2` banner.
- **Symptom**: on 10 generation ranks,
  `megatron/core/inference/engines/dynamic_engine.py run_engine_with_coordinator ->
  _run_ep_dummy_step -> controller.dummy_forward -> ... -> kimi_delta_attention.py
  _dynamic_inference -> self.in_proj(hidden_states)` (TE `LayerNormLinear`) ->
  `RuntimeError: transformer_engine.cpp:255 CheckOutputTensor: Assertion failed: t.has_data() ||
  t.has_columnwise_data(). Output (rsigma) is not allocated!`; the coordinator logs
  `removed engine ... (now 54 engines)` and generation stalls.
- **Root cause**: my recipe forced eager generation (`cuda_graph_impl: none`). An expert-parallel
  rank with no request of its own runs a dummy step to stay in the EP collectives; without a
  matching CUDA graph `DynamicInferenceContext.initialize_attention_state` returns early ("the
  controller will directly call the model forward pass with a single token") and the fork's KDA
  dynamic-inference path does not survive that batch (TE 2.17 layernorm output unallocated). With
  CUDA graphs the dummy step is padded to a captured graph's dimensions instead.
- **Fix (unverified)**: eager overrides removed -> NeMo-RL defaults (`cuda_graph_impl: local`,
  `inference_cuda_graph_scope: block`, 4 graphs); `moe_pad_experts_for_cuda_graph_inference: true`
  added (required for EP > 1 with inference CUDA graphs unless `inference_optimized`).
  Open: whether graph capture works with the KDA + latent-MoE layers at all.

## Apertus 1.5 run log (continued)

### Run `3622703` — 2026-10-09 — **FULL RUN VALIDATED: 92/92 steps, COMPLETED exit 0, 2 h 04 min** (`train-gsm8k-apertus-v1.5-70B.sh`, vLLM rollout)

- **Log**: `~/Downloads/slurm-3622703.tail.out` (last 4000 lines); full log on the cluster at
  `/users/palmee/slurm-3622703.out`, run dir
  `.../RL/Apertus-v1.5-70B/nemo-rl-runs/Apertus-v1.5-70B-gsm8k-nemorl-vllm-megatron-async-16n-run-3622703`
  (`run.log`, `tb/`). 16 nodes, image `alps7-dev-7eab7fc4...`, script as committed at `1eacbe4`
  (old checkout with the bridge filelock edit; the `-clean` checkout + `MEGATRON_CONFIG_LOCK_DIR`
  variant is still untested). ~5 min queued, 7447 s elapsed.
- **Result**: `SC run complete: {'train_steps': 92, 'trainer_version': 92}`. No OOM, no NCCL
  timeout, no refit failure across 92 refits.
- **Sampled metrics** (poller, every ~10 steps):

  | step | step time | loss | grad_norm | gen_kl_error | reward mean |
  |---|---|---|---|---|---|
  | 1 | 126 s | 0.075 | 0.185 | 0.00068 | 0.975 |
  | 11 | 76 s | 0.012 | 0.097 | 0.00068 | 1.051 |
  | 20 | 78 s | 0.047 | 0.118 | 0.00074 | 1.001 |
  | 31 | 71 s | 0.064 | 0.132 | 0.00074 | 0.993 |
  | 40 | 69 s | 0.034 | 0.096 | 0.00072 | 1.025 |
  | 50 | 66 s | 0.020 | 0.092 | 0.00073 | 1.073 |
  | 60 | 67 s | 0.032 | 0.109 | 0.00068 | 1.049 |
  | 71 | 67 s | 0.014 | 0.104 | 0.00072 | 1.067 |
  | 81 | 69 s | 0.012 | 0.072 | 0.00072 | 1.066 |
  | 90 | 66 s | 0.014 | 0.100 | 0.00075 | 1.044 |
  | 92 | 72 s | 0.031 | 0.100 | 0.00080 | 1.067 |

  Reward is the composite one (1.0 correct + 0.1 format, max 1.1); GSM8K is near-saturated for
  this checkpoint from step 1, so the curve says "stable", not "learning". `gen_kl_error` stays
  ~0.0007 throughout (fork's healthy bar < 0.002): trainer and vLLM agree after every refit.
- **Timing** (last step): generation 24.6 s, policy training 39.0 s, weight sync 3.4 s. About
  67-70 s/step in steady state, i.e. ~2 h for 92 steps of 48x16 on 12 trainer + 4 rollout nodes.
- **Not captured**: per-step metrics for all 92 steps (only the samples above; the rest are in
  `run.log` / TensorBoard on the cluster); a held-out evaluation (the recipe has `val_period: 0`).

### Run `3624750` — 2026-10-09 — CUDA graphs on: graph capture works, same KDA crash in the idle-rank dummy step during prefill

- **Log**: `~/Downloads/slurm-3624750.tail.out` (last 14000 lines). 24 nodes. Reached `Step 1/2`
  ~11 min in, `Generating responses for batch of size 256...`, then 13 engines crashed and the job
  hung (log idle; state RUNNING when written). Job `3623722` before it was cancelled by the user's
  OK after hanging the same way.
- **Worked**: inference CUDA-graph warmup on the generation ranks with the KDA + latent-MoE layers
  (`cuda graph warmup - [152]: 0 P + 152 D ... 4/4`); `Initialized persistent inference engine`.
- **Symptom**: identical to `3623722` -- `_run_ep_dummy_step -> dummy_forward -> ... ->
  kimi_delta_attention.py:1422 _dynamic_inference: projected, _ = self.in_proj(hidden_states)` ->
  TE `CheckOutputTensor ... Output (rsigma) is not allocated`; 13 of 64 engines removed.
- **Root cause (refined)**: only DECODE graphs are captured -- the four warmup shapes are
  `0 P + {1, 2, 76, 152} D`, none with prefill requests. Generation starts with prefill, and
  `generation_batch_size: 32` hands requests to at most 32 of the 64 engines at a time, so idle
  expert-parallel ranks run the dummy step while a peer is in prefill. No captured graph matches,
  `initialize_attention_state` returns early with `token_count=0`, the dummy forward gets an empty
  batch, and the Megatron-LM fork's KDA layer feeds it to a TE `LayerNormLinear`, which asserts on
  the empty output. KDA's dummy-state handling (`kda_dummy_state_idx`) only covers dummy slots
  inside a captured decode graph. Enabling CUDA graphs therefore did not help; this is a gap in
  the fork's KDA inference path, independent of NeMo-RL and the bridge.
- **Options (none applied yet)**:
  1. config only -- `INFERENCE_REQUESTS=256` so all 64 engines get requests at once and none is
     idle during prefill; racy by construction (an engine can start a step before its peers have
     received theirs) and idle ranks will still appear whenever a prefill overlaps an empty engine;
  2. a small patch to the Megatron-LM fork: make `KimiDeltaAttention._dynamic_inference` return an
     empty output for an empty batch instead of calling `in_proj` (the MoE layers after it still
     run their collectives, which is the point of the dummy step) -- unverified, and belongs
     upstream in `swiss-ai/Megatron-LM-MoE`;
  3. get prefill graphs captured (`cuda_graph_max_tokens`, `num_cuda_graphs`) -- unknown whether
     KDA prefill is graph-capturable.

## Megatron-LM fork pin moved to `b0844260` — 2026-10-09 (fix for the KDA dummy-step crash, unverified)

- Job `3624750` cancelled with the user's OK (hung, 24 nodes). User chose "option 2, and if the fix
  already exists apply it".
- **The fix already exists in the fork**: `andresnowak/Megatron-LM-MoE` branch
  `feat/nemo-rl-integration-v2` (= `swiss-ai/Megatron-LM-MoE` PR #88) gained one commit on
  2026-10-09 16:05 CEST, directly on top of the previously pinned `a9e0c497`:
  `b0844260b2a0888c711dca230b88ec13b20f3229` "Complete nonempty EP dummy forwards for PR #4258"
  (adapted from NVIDIA/Megatron-LM PR #4258). It prepares a non-empty dummy batch (one decode
  request) for an idle expert-parallel rank *before* CUDA-graph matching, so both graph execution
  and the eager fallback get valid input, and removes the old empty/single-token dummy path --
  exactly the failure of runs `3623722` / `3624750`. Files: `inference/contexts/dynamic_context.py`,
  `inference/model_inference_wrappers/abstract_model_inference_wrapper.py`, `models/gpt/gpt_model.py`,
  `models/hybrid/hybrid_model.py` (+ tests). The KDA layer itself is unchanged.
- **Applied by moving the pin** (`MLM_REF` -> `b0844260`, `SRC_VERSION` -> `v3`, new shared
  checkout) rather than as an embedded patch on `a9e0c497`: the result is byte-identical, and the
  Megatron-LM fork stays "used as-is at a commit of its own PR".
- **Compatibility checked statically**: all 126 module-level `megatron.core` imports of vanilla
  NeMo-RL `f196ef0a` still resolve at `b0844260`; the commit removes `GPTModel.forward`'s
  `is_spec_decode` argument, which NeMo-RL only forwards when a caller sets it
  (`distributed/model_utils.py`) and no caller does.
- Note: the vLLM example `train-gsm8k-chonk.sh` (author's) still pins `a9e0c497`.

### Run `3628358` — 2026-10-09 — **Megatron generation of Chonk WORKS** (256 rollouts, no engine crash); trainer OOM in the first training forward

- **Log**: `~/Downloads/slurm-3628358.tail.out` (last 6000 lines). 24 nodes, fork pin `b0844260`,
  fresh `src-v3` checkout. ~50 min queued, FAILED after 827 s.
- **Worked, first time**: the whole generation side. `Generating responses for batch of size 256`
  -> every rank `finishing generation` (`mcore async scheduling steps (cumul): 93`), zero
  `CheckOutputTensor`, zero `removed engine`: the fork commit `b0844260` (non-empty EP dummy
  forwards) fixes runs `3623722` / `3624750`, and KDA + latent-MoE generation under Megatron-Core's
  dynamic engine with CUDA graphs works. Generation rank memory after the rollout: ~46.3 GB
  allocated / 56.2 GB device. Then rewards, advantages and the trainer's logprob pass
  (`logprob_batch_size: 1`) all completed.
- **Symptom**: `torch.OutOfMemoryError: Tried to allocate 2.52 GiB ... 86.28 GiB allocated by
  PyTorch` on trainer rank 17 in `MegatronPolicyWorker.train()` ->
  `megatron/core/pipeline_parallel/schedules.py forward_step_calc_loss` ->
  `nemo_rl/algorithms/loss/loss_input.py prepare_loss_input` ->
  `distributed/model_utils.py from_parallel_logits_to_logprobs` -> `all_to_all_vp2sq`
  (`torch.empty_like` of the logits).
- **Root cause**: training micro-batch 4 x 1280 tokens x vocabulary 200064 logits (several GiB per
  copy, more than one copy in the loss) plus the forward activations of 42 layers without activation
  checkpointing, on top of the ~49 GB the trainer holds at rest without expert offloading. The
  author's recipe uses micro-batch 4 only together with offloading (and notes the historical run
  used 1).
- **Fix (unverified)**: `TRAIN_MICRO_BATCH` default 4 -> 1, `TRAIN_GLOBAL_BATCH` decoupled and kept
  at 128 (4 accumulated micro-batches per rank; both env-overridable, divisibility checked).
  Further levers if needed: `activation_checkpointing: true`, expert offloading.
- **Side observation (relevant to the R3 question)**: during the trainer's logprob pass every rank
  logs `N/M valid tokens have -inf in prev_logprobs (policy top-k/top-p mismatch). Masking out`
  with N = 1-2 of 35-241 tokens -- i.e. about 1-2 % of sampled tokens fall outside the trainer's
  top-p 0.9 nucleus. Generator and trainer are therefore close but not identical at the nucleus
  boundary. No generation-KL number was printed (the run died before the step metrics).

### Run `3639449` — 2026-10-11 — micro-batch 1 clears the trainer OOM; first backward pass dies on TileLang's kernel cache on Lustre

- **Log**: `~/Downloads/slurm-3639449.tail.out`. 24 nodes, no queue wait, FAILED after 1188 s
  (checkpoint load was ~5 min slower than `3628358`; no error there).
- **Worked**: generation (again), rewards, the logprob pass, and the training FORWARD at
  `TRAIN_MICRO_BATCH=1` -- no `OutOfMemoryError` anywhere. Training reached the backward pass.
- **Symptom**: `MegatronPolicyWorker.train()` on rank 0:
  `fla/ops/kda/backends/tilelang/chunk_bwd_dqkg.py:455 chunk_kda_bwd_wy_dqkg_fused_tilelang ->
  tilelang/jit ... -> tilelang/cache/cuda_binary_cache.py:132 load -> f.read()` ->
  `RuntimeError: [Errno 116] Stale file handle` (surfaced through TVM FFI, hence the extra
  `ffi.Error ... ToJSONGraph` noise and Ray's "original cause isn't serializable").
- **Root cause**: the KDA backward kernels (flash-linear-attention's TileLang backend) are
  JIT-compiled on the first training step. TileLang 0.1.12 caches the compiled CUDA binaries under
  `TILELANG_CACHE_DIR`, default `~/.tilelang/cache` -- Lustre home, shared by all 32 trainer ranks
  compiling the same kernels at once. Same class as the Triton / FlashInfer / bridge-lock cache
  hazards. Generation did not hit it (its KDA path uses the fused recurrent kernel).
- **Fix (unverified)**: `export TILELANG_CACHE_DIR=/tmp/tilelang_<jobid>` (node-local) with the
  other cache redirects. Cost: every job recompiles these kernels (~30 s per kernel per the log).
- **New cache-redirect rule for this image**: anything KDA / TileLang needs `TILELANG_CACHE_DIR`
  off Lustre. `~/.tilelang/cache` may now hold partial files from this run; harmless once the
  variable is set.

### Run `3640152` — 2026-10-11 — TileLang fix confirmed, first backward pass completes; trainer OOM in a later micro-batch forward

- **Log**: `~/Downloads/slurm-3640152.out` (15 MB; fetched with `ops/view` in 1 MB chunks, since
  `ops/download` fails above ~5 MB and the S3 transfer job sat PENDING). ~65 min queue wait
  (`QOSGrpNodeLimit`), FAILED after 884 s.
- **Worked**: everything up to and including the first backward pass. TileLang compiled its
  kernels from node-local /tmp in ~30 s (`TileLang completes to compile kernel`) and
  `run_backward` ran -- the `TILELANG_CACHE_DIR` fix of run `3639449` is verified.
- **Symptom**: `MegatronPolicyWorker.train()` rank 23, `forward_backward_no_pipelining ->
  forward_step -> experts.py:653 -> TE grouped_linear` -> `torch.OutOfMemoryError`: 93.66 GiB in
  use by the process, 80.56 GiB allocated by PyTorch, 1.34 GiB reserved-unallocated (so ~12 GiB
  outside PyTorch: CUDA context, NCCL, TE workspaces, TileLang kernels).
- **Budget at EP16** (11.66 B parameters per trainer rank; generator at EP8 has 18.1 B, so
  ~5.2 B non-expert + ~103 B expert parameters in total): bf16 weights 23 GB + gradient buffer
  23 GB = the ~49 GB seen at build, ~31 GB of activations at micro-batch 1 without activation
  checkpointing, ~12 GB outside PyTorch. The Adam state (TE FusedAdam, precision-aware: fp32
  moments and master weights for the rank's shard) is allocated lazily at the first optimizer
  step and had not been allocated yet -- it would not have fitted either.
- **Fix (unverified, config only)**: trainer `expert_model_parallel_size: 32` (8 experts per
  GPU, ~8.4 B parameters per rank), `activation_checkpointing: true`, Adam moments in bf16
  (`exp_avg_dtype` / `exp_avg_sq_dtype`). Estimate: ~34 GB weights + gradients, ~20 GB optimizer
  state, ~12 GB outside PyTorch, leaving ~29 GB for (now checkpointed) activations. The
  optimizer-state estimate assumes the shard reasoning above and has not been measured.
- **Untested side effects**: default refit now gathers EP32 -> EP8; expert data-parallel size
  is 1; activation checkpointing recomputes the KDA forward during backward (slower steps).
- **Underlying trade-off**: the author's recipe fits because of coarse-grained expert offloading
  with in-place FP8 parameters, which needs swiss-ai Megatron-Bridge PR #4. Staying on the
  vanilla bridge is what makes the trainer memory this tight.

### Run `3642870` — 2026-10-12 — FIRST COMPLETE RUN (2/2 steps, COMPLETED exit 0, 967 s); generator degrades after the first refit

- **Log**: `~/Downloads/slurm-3642870.tail.out` (last 6 MB of a 16.5 MB log). No queue wait.
- **Memory fix verified**: trainer EP32 + activation checkpointing + bf16 Adam moments ran both
  steps with no `OutOfMemoryError`. Config validated, EP32 -> EP8 default refit ran (7.1 s).
- **Metrics**:

  | | Step 1 | Step 2 |
  |---|---|---|
  | Avg reward | 0.6016 | 0.2070 |
  | Generation KL error | 0.0555 | 0.5533 |
  | Loss | 0.0930 | 0.1869 |
  | Step time | 345 s (TileLang compile included) | 71 s |

  Step 2 breakdown: generation 30 s, policy_training 18 s, refit 7 s, logprobs 6 s.
- **Open correctness problem (not diagnosed)**: after one optimizer step at warm-up learning
  rate (~1e-7) the reward drops by two thirds and gen-KL rises tenfold. Step 1 has no refit
  (both models load the same checkpoint); the first trainer -> generator refit is just before
  step 2, so the refit is the prime suspect (EP32 -> EP8 through the vanilla bridge mapping:
  expert ordering, fork-only parameters / buffers such as `qb_beta`, KDA state). Second
  candidate: the Adam step itself on hypersphere-constrained weights. No refit warnings in the log.
- **Also off**: gen-KL 0.0555 at step 1, before any update (Apertus 1.5 run: ~0.0007) --
  generator (inference engine, KDA inference state, EP8) and trainer (EP32) already disagree.
- **Next**: separate refit from optimizer (learning rate 0 for one step, or compare generator
  and trainer weights after the refit).

### Run `3648142` — 2026-10-12 — learning-rate-0 diagnostic: generator still degrades after the refit, so the optimizer is not the cause

- **Log**: `~/Downloads/slurm-3648142.tail.out`. Submitted copy only: `lr`, `min_lr`,
  `lr_warmup_init` = 0 and `#SBATCH --partition=preemptable` (repo script unchanged).
  COMPLETED exit 0, 1037 s.
- **Queue**: the same job on `normal` (`3647237`) sat ~3 h on `QOSGrpNodeLimit` and was
  cancelled; on `preemptable` the pending reason became `Resources` and it started within
  ~10 min. `preemptable` bypasses the `csstaff` group node limit.
- **Metrics**:

  | | Step 1 | Step 2 |
  |---|---|---|
  | Avg reward | 0.5703 | 0.2461 |
  | Generation KL error | 0.0455 | 0.3678 |
  | Loss | 0.0729 | 0.0469 |

- **Conclusion**: with a zero update the trainer weights cannot have moved, yet the generator
  degrades exactly as in `3642870`. The fault is in the trainer -> generator refit (default
  path: bridge `export_hf_weights` on EP32, `stream_weights_hf_to_megatron` per task on EP8,
  then `finalize_hf_import`), or in a training-mode side effect on non-gradient state.
- **Checked by reading code, not the cause as far as it goes**: the bridge keeps `A_log`,
  `dt_bias` and `qb_beta` in fp32 on export; the recipe sets
  `moe_router_quantile_balancing_freeze: true` (not confirmed to reach the trainer config).
- **Not yet checked**: expert ordering of the TE grouped-MLP `weight*` mappings across
  EP32 -> EP8, the KDA custom mappings' round trip, `finalize_hf_import` for Apertus 2.
- **Separate, still open**: gen-KL ~0.05 at step 1 before any refit.

### Run `3652972` — 2026-10-13 — refit probe: the refit is faithful; the trainer rewrites `qb_beta` because the freeze flag never reached the model

- **Log**: `~/Downloads/slurm-3652972.out`. Learning rate 0, `preemptable`, separate checkout
  `src-v3diag-...` with a probe in `megatron_worker.py::_write_generation_refit_weight` that
  compares each destination tensor with the value about to be written (probe script:
  scratchpad `refit_diag.py`; not in the repo). Step 1 reward 0.59 / gen-KL 0.036, step 2
  reward 0.21 / gen-KL 0.31.
- **Finding 1**: there IS an initial refit before step 1 -- all 3230 tasks change, with large
  differences. The generator is built without loading the checkpoint and receives its weights
  from the trainer, so the bridge export/import round trip (EP32 -> EP8, KDA mappings, expert
  gather) is what produces the step-1 generator, and that generator scores ~0.6.
- **Finding 2**: at the refit before step 2, on all 64 generator ranks, only
  `decoder.layers.N.mlp.router.qb_beta` changes: 39/39 MoE layers, every one of the 256
  entries, max abs diff ~1.0 (plus denormal-level noise, ~1e-38, on 2 latent projections).
- **Root cause**: `moe_router_quantile_balancing_freeze: true` was a top-level `megatron_cfg`
  key, as in the author's `grpo-chonk-vllm.yaml`. Vanilla NeMo-RL's `_apply_moe_config` does
  not forward it (unknown `megatron_cfg` keys are dropped silently), so the fork router's
  default (`freeze=False`, EMA 0.0) replaced `qb_beta` with a quantile estimate from each
  128-sequence GSM8K batch. Every other recipe key in that block is forwarded (checked).
- **Fix (verified in `3653929`)**: `megatron_cfg.model_overrides: {moe_router_quantile_balancing_freeze:
  true}` -- vanilla NeMo-RL's route to model-config fields (`_merge_model_overrides`); the field
  comes from the fork's `TransformerConfig`. No library change.
- **Lesson**: a recipe key that vanilla NeMo-RL does not know is not an error. When porting a
  key from the author's fork recipe, grep `nemo_rl/models/megatron/setup.py` for it.
- **Still open**: gen-KL 0.036-0.055 at step 1 with identical weights on both sides (Apertus
  1.5: ~0.0007).

### Run `3653929` — 2026-10-13 — `qb_beta` freeze via `model_overrides`: generator no longer degrades after the refit

- **Log**: `~/Downloads/slurm-3653929.out`. Repo script at `f782e78` (normal learning rate,
  Adam, KL penalty 0, default refit, shared `src-v3` checkout), `preemptable` set in the
  submitted copy only. COMPLETED, exit 0, 967 s; started ~25 min after submission.
- **Result**: step 1 reward 0.6055 / gen-KL 0.0679 / loss -0.0300 / mean length 77.2;
  step 2 reward 0.6133 / gen-KL 0.0803 / loss -0.0467 / mean length 83.8. Before the fix,
  step 2 was reward 0.21-0.25 / gen-KL 0.31-0.55.
- **Confirmed**: `model_overrides` accepts `moe_router_quantile_balancing_freeze` for
  `Apertus2ModelProvider` (it appears in the resolved `MasterConfig`); no library change.
- **Still open**: gen-KL 0.04-0.08 on both steps (Apertus 1.5: ~0.0007). Two steps say
  nothing about learning; a longer run is the next check. Adam, KL penalty 0 and the default
  refit are still the temporary settings noted in the script.
- **Gen-KL analysis** (offline, from `nemo-rl-logs/exp_001/train_data_step1.jsonl`, 19775
  generated tokens): the 0.068 is not a Chonk model mismatch. Two effects, both from
  `top_p: 0.9`:
  1. 25 tokens (0.13%) carry 97% of the KL. The generator sampled them inside its nucleus,
     the trainer places them just outside its own (flat tails, bf16 noise moves the cutoff).
     Their trainer log-probability is `-inf`, `mask_out_neg_inf_logprobs`
     (`nemo_rl/models/megatron/train.py`) writes 0 instead, and the gen-KL metric still counts
     the position (trainer 0 vs generator down to -6.2). The loss masks these positions, so
     this inflates the metric, not the gradient. Without them the KL is 0.0018.
  2. The rest is a logprob-mode mismatch: the generator reported `raw_logprobs`, the trainer
     renormalises over the nucleus. 62% of trainer values are exactly 0 (single-token
     nucleus) and trainer - generator is in [0, -ln 0.9] for 95% of tokens. Fixed in the
     script: `logprobs_mode: processed_logprobs` (verified in `3654364`). The Apertus 1.5 Megatron-gen
     script never set `top_p`, which is why it showed ~0.0007.
- **Lesson**: judge gen-KL from the per-token dump, not the printed mean; with `top_p < 1` a
  handful of out-of-nucleus tokens dominates it. `top_p: 1.0` removes both effects but
  departs from the author's recipe.

### Run `3654364` — 2026-10-13 — `processed_logprobs`: generator and trainer agree; the printed gen-KL is only the out-of-nucleus artifact

- **Log**: `~/Downloads/slurm-3654364.out`. Repo script at `353667d` (`top_p: 0.9` kept, as in
  the author's recipe), `preemptable` in the submitted copy only. COMPLETED, exit 0, 996 s.
- **Printed**: step 1 reward 0.5117 / gen-KL 0.0741 / loss -0.0277; step 2 reward 0.5703 /
  gen-KL 0.0495 / loss 0.0115.
- **Per-token recomputation** (`train_data_step{1,2}.jsonl`): excluding the out-of-nucleus
  tokens (16 of 20149 and 19 of 22154, 0.08-0.09%, >99% of the printed KL) the gen-KL is
  0.00033 on both steps, below the Apertus 1.5 figure (~0.0007). Mean trainer - generator
  log-probability -0.0001 to -0.0003 (was +0.042 with `raw_logprobs`), median |diff| 0.
- **Open**: step-1 reward 0.51 is below the 0.57-0.61 of the four earlier runs (256 samples,
  standard error ~0.03). `logprobs_mode` should not change sampling; watch it on the next run.
- **Not fixable by config**: the printed gen-KL stays at 0.05-0.07 while `top_p < 1`, because
  NeMo-RL writes 0 for a `-inf` trainer log-probability and the metric keeps the position.

### Run `3654710` — 2026-10-13 — first 20-step run: reward drifts up from ~0.62 to ~0.72, loop stable

- **Log**: `~/Downloads/slurm-3654710.out`. Repo script at `353667d`, `MAX_NUM_STEPS` default
  set to 20 and `preemptable` in the submitted copy only. COMPLETED, exit 0, 2040 s
  (step 1 352 s, later steps 50-74 s including the refit).
- **Reward per step**: 0.57 0.56 0.61 0.69 0.69 0.63 0.64 0.66 0.59 0.75 0.70 0.66 0.66 0.77
  0.69 0.76 0.64 0.74 0.77 0.66. Mean of steps 1-5 0.623, 6-15 0.676, 16-20 0.716; first ten
  0.639, last ten 0.706. Single steps are noisy (256 samples, standard error ~0.03).
- **Gen-KL**: printed 0.02-0.06 with spikes at step 6 (0.387) and step 7 (0.125). Per-token
  dumps for steps 1, 6, 7, 20: excluding out-of-nucleus tokens (15-30 per step, >99% of the
  printed value) it is 0.00033 on every step. The spikes are single tokens with generator
  log-probability down to -8.9; nothing drifts over 20 refits.
- **Caveats**: the learning rate was still in warm-up (60 iterations, ~40 done); Adam, KL
  penalty 0 and the default refit are still the temporary settings; no validation set, so the
  reward is on training prompts.

### Run `3655036` — 2026-10-13 — KL penalty 0.01 restored: reference model fits at EP32, same reward trend

- **Log**: `~/Downloads/slurm-3655036.out`. Repo script at `3cd896f` (`REFERENCE_KL_PENALTY`
  default 0.01), 20 steps and `preemptable` in the submitted copy only. COMPLETED, exit 0,
  2377 s (step 1 430 s, later steps 60-78 s; `reference_logprobs` ~10 s per step, 88 s on
  step 1).
- **Memory**: `[GPU_DIAG] after_ref_model` on the trainer: PyTorch allocated 64.4 GB, reserved
  76.6 GB, NVML used up to 80.9 GB of 97.9 GB. The reference model's GPU copy is not freed by
  `setup_reference_model_state` (allocated doubles from 32.2 GB), as suspected in `3622931`;
  it fits at EP32 with ~17 GB headroom. All 20 training steps ran without OOM.
- **Reward per step**: 0.59 0.61 0.57 0.70 0.68 0.65 0.67 0.65 0.57 0.74 0.71 0.71 0.73 0.75
  0.68 0.75 0.60 0.77 0.78 0.68. Mean of steps 1-5 0.631, 6-15 0.685, 16-20 0.715 (run
  `3654710` without the penalty: 0.623 / 0.676 / 0.716).
- **Gen-KL** excluding out-of-nucleus tokens: 0.00031 (step 1), 0.00030 (step 20).
- **Not measured**: KL to the reference policy. Neither the console metrics nor
  `train_data_step*.jsonl` include reference log-probabilities in this NeMo-RL version.

### Native refit: Megatron-LM API shim instead of an upstream port (2026-10-14)

- Upstream NVIDIA/Megatron-LM `main` added `execution_batch_bytes` to
  `prepare_swap_model_weights` / `swap_model_weights` in `5eb0744c7` (#6878, 2026-08-28); `dev`
  and the fork (`b0844260`) do not have it. It is the only keyword NeMo-RL `f196ef0a` passes that
  the fork lacks.
- NeMo-RL always passes a value: `resolve_refit_execution_batch_bytes` turns `null` into its
  packed-tensor size. (An earlier note here said `null` reaches Megatron-LM; it does not.)
- A faithful port (rebasing is not realistic: 475 fork commits on a 2026-03-31 merge base,
  upstream 1610 ahead) means cherry-picking #5872, #5187, #6826, #6133, #6878; trial picks on
  `b0844260` conflict in 6, 2, 0, 3 and 11 files, mostly `execution.py` / `refit.py`, where the
  fork has its own reconciled refit port (`7dbf53902`). Left to the fork's maintainers.
- Chosen instead (user): `mlm-refit-execution-batch-bytes.patch`, upstream's signature only. With
  `nccl_m2n` it sets the M2N grouped-submission limit as upstream does; other backends (we use
  `nccl`) keep the fork's model-wide submission. Recipe: `refit_transport: mcore`,
  `mcore_generation_config.refit_backend: nccl`, `SRC_VERSION` v4.

### Run `3659197` — 2026-10-14 — native refit: plan builds, first transfer deadlocks in NCCL (600 s timeout on all 96 ranks)

- **Log**: `~/Downloads/slurm-3659197.out`. Repo script at `4e4a661` (`refit_transport: mcore`,
  `refit_backend: nccl`, `src-v4`, W&B on), `preemptable` in the submitted copy only. FAILED, exit
  15, 1225 s. `3659128` (same, no W&B) was cancelled while pending.
- **Worked**: `src-v4` checkout with the Megatron-LM patch; config check (`refit mcore`,
  `wandb True`); trainer with the reference model (64.4 GB allocated); generator; the refit
  process group (96 ranks = 32 trainer + 64 generator) and the reshard plan, so the plan builder
  handles KDA, latent MoE and `qb_beta`.
- **Symptom**: no output for 10 min after the refit started, then on every rank
  `Watchdog caught collective operation timeout: WorkNCCL(SeqNum=1, OpType=COALESCED, ...
  Timeout(ms)=600000)` in `batch_isend_irecv` <- `nccl_copy_service.py:76 run` <-
  `execution.py:218 execute_reshard_plan` <- `refit.py swap_model_weights`.
- **Root cause**: the fork's `NCCLCopyService.run` sends every P2P op of the plan in one
  `batch_isend_irecv` (one ncclGroup with the whole model). NCCL splits a group that large into
  kernel plans at rank-dependent points, and a send and its matching receive can end up in
  different plans on the two ranks: a device-side deadlock (pytorch/pytorch#174288). Upstream
  fixed exactly this in NVIDIA/Megatron-LM #7189 (`325c9936b`).
- **Fix (unverified)**: the Megatron-LM patch now also ports #7189's `run()`-side layer: P2P ops
  go out in windows of 256 global task ids, which sender and receiver share. Upstream decides to
  window from the plan's global task count; the fork's plans carry none, so the port windows
  whenever every op has a task id (always, for planner-built plans), the same decision on every
  rank. #7189's planner-side batching is not ported. `SRC_VERSION` v5.

### Run `3659387` — 2026-10-14 — NCCL windowing fix gets the native refit through setup; preempted in step 1

- **Log**: `~/Downloads/slurm-3659387.out`. Repo script at `c17e381` (`src-v5`), `preemptable` in
  the submitted copy only. PREEMPTED after 928 s, during step-1 generation.
- **Worked**: the initial native refit (EP32 -> EP8 over the 96-rank refit group) completed with
  no NCCL timeout: `SETUP COMPLETE`, total setup 497.7 s, then `Step 1/2` and all 64 generation
  ranks serving. The windowing fix removes the `3659197` deadlock.
- **Not yet known**: refit-before-step-2 time, step-2 reward, gen-KL. The `task-id windows` log
  line is at Megatron INFO level and did not reach the console.
- **First preemption** on `preemptable` in this series: with checkpointing off the run is lost.

### Run `3659594` — 2026-10-14 — native refit works for both refits; trainer OOM on one rank in step-2 training

- **Log**: `~/Downloads/slurm-3659594.out`. Same as `3659387` (`c17e381`, `src-v5`, W&B on),
  `preemptable` in the submitted copy only. FAILED, exit 15, 1206 s.
- **Worked**: initial native refit, step 1 (reward 0.6133, printed gen-KL 0.0373, mean length
  81.2; step time 369 s = generation 164 s + logprobs incl. reference 90 s + training 113 s),
  then the native refit with updated weights before step 2 and step-2 generation. Both native
  refits completed; no NCCL timeout.
- **Symptom**: step-2 training, trainer rank 14 only:
  `torch.OutOfMemoryError: ... Tried to allocate 588.00 MiB. GPU 2 has a total capacity of
  95.00 GiB of which 592.31 MiB is free. Including non-PyTorch memory, this process has 68.69 GiB
  memory in use. Of the allocated memory 57.71 GiB is allocated by PyTorch` in the top-p filter
  of the loss (`logits_sampling_utils.py:171 _apply_top_k_top_p_fn` <- `DistributedLogprobWithSampling`).
- **Reading**: the GPU is full (94.4 GiB used) while this process accounts for 68.7 GiB, so about
  26 GiB on that GPU sit outside the process's own accounting. At `init_complete` the process
  accounted for all used memory (79.2 of 79.8 GB). Most likely holder (unverified): NCCL buffers
  for the refit's point-to-point connections (each trainer rank exchanges with many generator
  ranks); the default refit path uses broadcasts and ran 20 steps with the same trainer config
  (`3655036`). The trainer also still holds the reference model's GPU copy (~32 GB).
- **Fix (user decision, unverified)**: keep the native refit, `REFERENCE_KL_PENALTY` default back
  to `0.0` (no reference model, ~32 GB less allocated per trainer GPU). Does not explain the
  ~26 GiB outside the process's accounting; if that grows across refits, the OOM will return
  later in the run.

### Run `3667575` — 2026-10-15 — native refit, KL penalty 0: 20/20 steps, no OOM; refit takes ~92 s per step

- **Log**: `~/Downloads/slurm-3667575.out` (20.5 MB, `ops/view` chunks). Repo script at `535f051`
  (`refit_transport: mcore`, `REFERENCE_KL_PENALTY` 0.0, `src-v5`), 20 steps and `preemptable` in
  the submitted copy only. COMPLETED, exit 0, 3792 s; ~8 min queued.
- **Memory**: no `OutOfMemoryError` over 19 native refits with updated weights, so dropping the
  reference model was enough. Per-step memory is not logged (`[GPU_DIAG]` only at init), so
  whether the ~26 GiB seen in `3659594` grows per refit is still unknown.
- **Reward per step**: 0.64 0.58 0.59 0.74 0.65 0.64 0.66 0.68 0.58 0.73 0.69 0.70 0.70 0.78
  0.71 0.79 0.59 0.74 0.79 0.62. Mean of steps 1-5 0.640, 6-15 0.688, 16-20 0.704 (default
  refit, `3654710`: 0.623 / 0.676 / 0.716). Printed gen-KL 0.019-0.11, same range as the default
  refit runs (per-token check not done).
- **Refit cost**: `prepare_for_generation/transfer_and_update_weights` 91.4-102.4 s on every step
  (default refit: ~7 s, `3642870`). Step time ~140 s instead of 50-74 s; the refit is ~65% of it.
  Likely causes (unverified): the P2P plan sent in serialized windows of 256 tasks
  (`MEGATRON_NCCL_COPY_TASK_WINDOW`), and plain `nccl` P2P instead of upstream's `nccl_m2n`
  grouped submission.
- **Root cause of the slow refit (found 2026-10-15)**: my port of #7189's task-id windows. The
  fork's planner numbers task ids destination rank by destination rank, so each window of 256
  holds one generator rank's transfers and the windows run one after another: the 64 generator
  ranks are filled one at a time, at about one GPU's network bandwidth (~2.3 TB in ~92 s,
  ~25 GB/s). Upstream uses those windows only as a fallback; its main path splits the plan into
  32-parameter batches that span all ranks, which the fork does not have.
- **Fix (unverified)**: the Megatron-LM patch now numbers task ids round-robin over destination
  ranks (planner `_iter_global_transfer_ops`): same transfers, same per-pair order, but each
  window spans all destination ranks. Offline check on a synthetic 64-receiver plan: ids are a
  permutation, median 64 receivers per window (was 1), at most ~25 ops per rank per window.
  `SRC_VERSION` v6.

### Run `3669084` — 2026-10-15 — round-robin task ids: refit fast; trainer OOM again in step-2 training

- **Log**: `~/Downloads/slurm-3669084.out`. Repo script at `261eef9` (`src-v6`, KL penalty 0), 4
  steps and `preemptable` in the submitted copy only. FAILED, exit 15, 1005 s.
- **Refit**: step 1 ended ~15:48:00 and the OOM is logged at 15:49:06, so the step-2 refit,
  generation (~30 s) and logprobs together took about a minute (the refit alone was 92 s in
  `3667575`). The exact refit time was not printed (the step did not finish).
- **Symptom**: trainer rank 15, step-2 training, same place and numbers as `3659594`: tried
  588 MiB, 652 MiB free of 95 GiB, this process 68.5 GiB, 57.4 GiB allocated by PyTorch.
- **Reading**: PyTorch allocation is the same with and without the reference model (57.4 vs
  57.7 GiB), so the reference copy was not what filled the GPU in `3659594`. Again ~26 GiB on
  the GPU is outside the process's NVML usage; at `init_complete` there was none
  (`GPU procs: [PID=63255 (self): 46790 MB]`, used 47422 MB). `NCCL_PXN_DISABLE=1` is already
  set in `env.toml`, so PXN relay buffers are not the explanation. `3667575` (same memory
  picture, rank-major windows) survived 20 steps, so the margin is small either way.
- **Next**: diagnostic run with `log_gpu_memory_diagnostics` (per-process NVML) before/after each
  native refit and at `train()` start, submitted copy only (`src-v6memdiag`, patch in scratchpad
  `nemo-rl-mem-diag.patch`).

### Run `3669266` — 2026-10-15 — memory diagnostic: COMPLETED 4/4; native refit 16 s; the refit allocates nothing

- **Log**: `~/Downloads/slurm-3669266.out`. Repo script at `1330502` plus, in the submitted copy
  only, `nemo-rl-mem-diag.patch` (scratchpad; `log_gpu_memory_diagnostics` before/after
  `swap_weights_via_reshard` and at `train()` start), `src-v6memdiag`, 4 steps, `preemptable`.
  COMPLETED, exit 0, 1253 s.
- **Refit**: `transfer_and_update_weights` 16.9 / 15.9 / 16.0 s (was 91-102 s in `3667575`):
  the round-robin task ids work. Step time 71-80 s (default refit runs: 50-74 s).
  Rewards 0.57 0.61 0.60 0.70; printed gen-KL 0.02-0.14 (per-token check not done).
- **Memory, per trainer GPU** (NVML used / this process / PyTorch allocated, MB): before and
  after every native refit identical to within ~0.5 GB, and no other process on any trainer GPU.
  Typical rank: step-1 `train_start` 41.9k / 41.3k / 32.5k; from step 2 on ~66k / 65k / 54k
  (the extra ~22 GB is the Adam state, allocated at the first optimizer step).
- **The unowned memory, localised**: on 3 of 32 trainer GPUs (ranks 12, 15, 25) NVML "used"
  rises 8-24 GB above the process's own usage, with no other process listed. It appears
  between the step-2 refit (`after_native_refit`: used = self + 0.6 GB) and step-2
  `train_start` (rank 25: 90.2k used vs 65.0k self), i.e. during the step-2 logprob pass, and
  then stays flat. Not the refit, and not the reference model. Same size as the ~26 GiB in the
  two OOMs. Cause unknown; candidate (unverified): GH200 memory not attributed to any CUDA
  context (e.g. system-allocated host memory migrated into HBM).

### Run `3669588` — 2026-10-15 — NUMA diagnostic: the "unowned" GPU memory is host pages placed in GPU HBM; OOM in step-2 training

- **Log**: `~/Downloads/slurm-3669588.out`. As `3669266` plus NUMA readings in every
  `[GPU_DIAG]` line (`numa_balancing`, per-node `MemUsed`, this process's resident pages per node
  from `/proc/self/numa_maps`) and readings at `get_logprobs` start/end; scratchpad
  `nemo-rl-mem-diag2.patch`, `src-v6memdiag2`. FAILED, exit 15, 1023 s: trainer rank 6, step-2
  training, tried 588 MiB with 46 MiB free.
- **Layout**: CPU NUMA nodes 0-3 (~119 GB each); GPU HBM is NUMA nodes 4, 12, 20, 28 (95 GB each).
  `numa_balancing=0`.
- **Finding**: on nid006586 all four trainer processes had ordinary host pages on node 20
  (GPU 2): rank 6 13.0 GB, ranks 4, 5, 7 ~5.0 GB each, ~28 GB together = the memory NVML counts
  as used on GPU 2 but no process owns (94.3 GB used, rank 6 itself 65.0 GB). The pages appear
  during step 2 (rank 6 at step-2 `logprobs_start`: 13.0 GB on node 20; before the refit: 0.04).
  No other node had more than ~0.1 GB of host pages on a GPU node; generator ranks none. So the
  OOMs of `3659594` / `3669084` / `3669588` are host memory placement, not the refit and not CUDA.
- **Fix (unverified)**: the launcher starts `ray start` (head and workers) and the driver under
  `numactl --membind=<NUMA nodes with CPUs>` (inherited by the raylet and every worker);
  `NUMA_BIND_HOST_MEMORY=0` disables it. Open question: why the kernel placed the pages on a GPU
  node with CPU nodes far from full (not answered by these readings).

### Run `3669709` — 2026-10-15 — NUMA binding verified: 6/6 steps, no host pages in GPU HBM, ~30 GB headroom

- **Log**: `~/Downloads/slurm-3669709.out`. Repo script at `379f4a0` (`numactl --membind=0,1,2,3`
  on every node, printed by the launcher) plus the `nemo-rl-mem-diag2.patch` diagnostics,
  `src-v6memdiag2`, 6 steps, `preemptable`, submitted copy only. COMPLETED, exit 0, 1388 s.
- **Binding is safe for CUDA**: model setup, refits, generation and training all ran.
- **Memory** (2208 diagnostic lines): host pages of any process on a GPU NUMA node at most
  0.28 GB (was 13 GB), NVML used minus own usage at most 0.83 GB (was ~28 GB). Peak NVML used per
  trainer GPU 64.3 GB at `train_start` and 65.3 GB after a refit (was up to 94.3 GB), so ~30 GB
  of the 95 GB free at those points. Generator GPUs at most 55.8 GB.
- **Steps**: reward 0.61 0.55 0.61 0.68 0.70 0.63; printed gen-KL 0.03-0.09. Native refit
  16.0-17.0 s every step; step time 67-78 s after step 1 (412 s).
- **Next**: md_decoupling (KL penalty kept at 0 by user decision, 2026-10-15).

### Run `3670987` — 2026-10-15 — md_decoupling restored: preflight and setup pass; trainer OOM in the first optimizer step

- **Setup**: 4 steps, `src-v7memdiag2` (repo `d490c75` + memory diagnostics), KL penalty 0, native
  refit, NUMA binding. Both bridge patches applied; `megatron.core.optimizer.md_decoupling` imports;
  config OK with `optimizer md_decoupling`.
- **Failure**: rank 15 only, `torch.OutOfMemoryError` in `md_decoupling.py:631` (`_param_step`,
  AdamW branch, `exp_avg_sq.sqrt()`), allocating 2.67 GiB with 82.75 GiB allocated by PyTorch
  (93.45 GiB in use). The other ranks were waiting in the step.
- **Memory vs Adam (`3669709`), step 1, PyTorch allocated**: `logprobs_end` 31.6 vs 16.1 GB,
  `train_start` 47.2 vs 31.7 GB. So ~15.5 GB more is resident even while optimizer state is
  offloaded: the fp32 master copy of the bf16 parameters, which the fp16/bf16 optimizer wrapper keeps
  outside `optimizer.state` (NeMo-RL's `move_optimizer` does not move it).
- **Cause**: at EP32 on 32 trainer GPUs the expert data-parallel group has size 1, so
  `LayerWiseDistributedOptimizer.shard_params` cannot shard expert parameters: every rank keeps
  fp32 masters, fp32 momentum and (with `grad_reduce_in_fp32: false`) an fp32 copy of every
  gradient (`_copy_model_grads_to_main_grads`, made for all parameters right before the step).
  Dense parameters are sharded over DP 32 by numel; the rank that gets the largest one
  (embedding/output, 2.67 GiB in fp32, AdamW branch with `exp_avg_sq` and full-size temporaries) runs
  out first. The author fitted this with coarse-grained expert offloading, which is not in this delta.
- **Fix (user decision, unverified)**: 32 nodes, 16 trainer (64 GPUs, EP32, expert data-parallel 2)
  + 16 generation; global batch pinned to the author's 128 (2 micro-batches per rank).
  `grad_reduce_in_fp32` stays at the author's `false` (considered: `true` would make the pre-step
  copy a no-op, net ~-7.8 GB, but with little margin). Fallback: the author's expert offloading.

### Run `3671334` — 2026-10-15 — md_decoupling on 32 nodes (16 trainer, expert DP 2): 4/4 steps, no OOM

- **Setup**: repo `fe2ae00` + memory diagnostics (`src-v7memdiag2`), 32 nodes (16 trainer EP32 =
  expert data-parallel 2, 16 generation EP8), global batch 128, md_decoupling (author's block,
  `grad_reduce_in_fp32: false`), KL penalty 0, native refit, NUMA binding. COMPLETED, exit 0, 1135 s.
- **Memory, trainer, PyTorch allocated (max over 64 ranks)**: log-prob pass 36.5 GB (Adam at 32
  GPUs: 16-18 GB; the fp32 masters stay resident), `train_start` 63.3 GB from step 2 on (Adam:
  52.9 GB), NVML used at most 75.6 GB of 95 at the logged points. The optimizer-step peak itself is
  not logged; it fit on every rank in 4 steps.
- **Steps**: reward 0.62 0.58 0.64 0.67 (Adam run `3669709`: 0.61 0.55 0.61 0.68); printed gen-KL
  0.022-0.095; loss 0.017 -0.090 -0.072 0.041; mean generation length 83-86 tokens. Native refit
  16.8-17.1 s (same as with 32 trainer GPUs); step time 67-74 s after step 1 (360 s).

### Run `3673498` — 2026-10-16 — md_decoupling, 32 nodes, repo script: 20/20 steps

- **Setup**: repo `fe2ae00` unchanged (+ `preemptable`, 20 steps), own checkout `src-v7`. Ran
  alongside `3673499`.
- **Steps 1-20**: reward 0.58 0.60 0.63 0.68 0.66 0.60 0.64 0.65 0.53 0.71 0.64 0.66 0.59 0.70 0.62
  0.72 0.63 0.67 0.66 0.64 (Adam run `3667575`, 20 steps: 0.58-0.79, mean of the last 10 ~0.70).
  Loss within +-0.11; mean generation length 77-101 tokens. Native refit 16.0-17.3 s; step time
  55-76 s after step 1 (362 s).
- **Printed gen-KL**: 0.02-0.09 on 17 of 20 steps, with one-step spikes at step 6 (0.227), 11 (0.175)
  and 19 (1.227), each back to baseline the next step (Adam 20-step run: 0.019-0.11, no spike above
  0.11). Not cumulative, so not a stale or missing refit weight (that would grow step over step).
  Likely the out-of-nucleus term: the metric averages k3 over ~22k tokens, a sampled token outside
  the trainer's nucleus counts with trainer log-prob 0, contributing ~exp(|generator log-prob|);
  one token at -10 (near-flat distribution) adds ~1.0. Masked from the loss (loss and reward
  normal on those steps). Unverified: the per-token values are only in wandb.
