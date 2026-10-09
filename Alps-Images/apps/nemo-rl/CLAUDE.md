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
- Everything else (24 nodes = 8 train EP16 + 16 generation EP8, 32x8 rollouts, 2 steps,
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
   needs the author's `0.01`. Blocked on trainer GPU memory for the reference model (run `3622931`):
   needs either expert offloading (bridge PR #4 patch group) or a NeMo-RL change that frees the
   reference model before the policy returns to the GPU.
2. **`md_decoupling` optimizer** -- currently Adam; restore the author's optimizer block and apply
   `example/patches/bridge-md-decoupling-optimizer.patch`.

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
