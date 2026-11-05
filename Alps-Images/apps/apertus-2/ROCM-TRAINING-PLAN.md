# Apertus 2 ROCm training: living plan

## Current checkpoint: 2026-11-05

- Candidate image acceptance and tiny PP2/EP2 checkpoint/restart (`716590`) passed; cleanup warnings remain documented separately.
- Recipe/harness-only implementation is approved, and the eventual 16-node native baseline is explicitly resource-authorized without another approval question.
- Exact reference gradient fusion remains disabled after Apex JIT failure; a host std::max fix based on existing ROCm compatibility work is syntax-validated on copied sources, but not installed/linked/runtime-tested.
- Named no-gradient-fusion recipe validation passed all eight ranks in two-node job `716600`: six updates, checkpoint at 3 and fresh-process resume through 6 (8m49s).
- Memory-backed warmup/reconciliation/restart `716602` and independent persistent read-only hydration `716605` passed all eight ranks with 280 GPU artifacts / 2240 files per measured cache.
- Exact tiny-model initialization-state comparison and six-update/restart acceptance passed on all eight ranks in `716607`.
- The first full-geometry sixteen-node/64-rank capacity job `716613` failed in first-warmup TE grouped-GEMM weight gradients (`HIPBLASLT Error: 6`); no update completed.
- Both two-node sequence-8192 controls failed with hipBLASLt status 6: native `716678` in grouped wgrad, Triton `716676` in non-grouped linear wgrad.
- Earlier standalone cases `716827` passed; logged training replay `716828` identified an NT BF16 2560/1280-feature, reduction-8192 call.
- Exact replay `716844` failed only for long BF16/default heuristic; FP32 output and requested heuristic one passed.
- Two-node native-grouped/heuristic-one training and restart control `716847` passed all eight ranks with full-job exit zero, checkpoints at 2/4 and fresh-process resume from 2.
- Reservation-blocked `716850` was cancelled while pending; unreserved full-shape retry `716859` failed after 36m12s in collective watchdog/store timeouts, with no completed warmup update or checkpoint.
- EP4/EDP2 stack control `716927` failed early with native SIGSEGV during the first timer dump; timer-disabled control `716928` passed two updates/checkpoint on all eight ranks.
- Combined PP2/EP2/EDP2 long-proxy `716931` passed all eight ranks, two updates and checkpoint at 2 in 8m36s.
- PP1/EP2/EDP4 control `716983` passed all eight ranks, two updates and checkpoint with cooperative Python sampling.
- Four-node PP2/EP2/cross-node-EDP4 control `716995` passed all sixteen ranks, two updates and checkpoint at 2 in 8m33s.
- Reduced PP2/cross-node EDP4 with 36 microbatches/update `717038` passed all sixteen ranks, two updates/checkpoint.
- Reduced four-layer/real-vocabulary/36-microbatch control `717068` passed all sixteen ranks, both updates and checkpoint validation.
- Eight-layer expert-footprint control `717080` passed all sixteen ranks, both updates and checkpoint validation.
- Eight-node PP1/EP8/DP32/EDP4, sixteen-layer control `717115` passed all 32 ranks, both updates and checkpoint validation.
- Four-layer full64-rank topology control `717130` passed both updates and checkpoint validation; frozen global acceptance finalized after the session restart.
- Full32-layer test `721544` completed zero updates and was cancelled at the user's request after 25m13s with a persistent post-backward wait.
- Tiny two-node router-communicator priming diagnostic `721576` passed both updates/checkpoint and actual all-rank priming validation.
- Full-depth priming comparison `721592` completed zero updates and failed at the 1500-second supervisor cutoff (25m21s Slurm elapsed); successful all-rank priming did not eliminate the post-backward wait.
- Clean reference checkout on native libraries passed small training/save/restart (`721759`) and traced small restart (`721782`); all-rank/stage actual-source and native critical-reduction completion evidence passed.
- Full-depth clean-source/traced comparison `721795` logged one finite update, then failed during update2 with expert SSSGLU OOM on ranks0/16 (16m51s); checkpoint2 is absent and global acceptance fails.
- All64 ranks passed the formerly blocked critical reductions in this source+trace variant; severe subsequent EP-rank concentration (524288 rows versus65536 balanced share) is the next numerical/memory lead, not an established root cause.
- Forced-routing small save/restart `722139` passed; full-depth forced trial `722153` observed1.013× peak expert-rank load but failed before its first completed update with CXI mapping-write/RCCL internal error on rank40 (10m04s), not an observed OOM.
- Second-update routing/OOM prevention remains untested because no optimizer update completed; investigate exact lower-stack metric-communicator error before another full-size trial.
- Interactive721530 remains preserved; no diagnostic job is active and no replacement run is submitted.
- The EDP1 full-vocabulary proxy remains prepared, unsubmitted and lower priority than reproducing the expert-DP stall.
- Earlier cache retry `716601` was agent-cancelled because its snapshot filter could omit every GPU object; empty/incomplete snapshots now fail explicitly.
- The CPU harness suite passes 166 tests; frozen failures and separate result records are preserved.
- The user’s current interactive allocation `721530` remains untouched; diagnostic jobs use sequential admission, including the full-depth comparison.
- Full mid-4b capacity/performance remains unvalidated; exact warmup/measured initialization equality is established at tiny geometry, and the first full-shape attempt failed before any completed update.

## Objective and scope

Make the native `megatron-rocm` image support reproducible Apertus 2 pretraining on Beverin MI300A over Slingshot/CXI without depending on Imanol’s scratch space.
Reproduce mid-4b first, then establish separate Chonk and Megachonk configurations.
Keep correctness, startup performance, steady-state throughput, restart reliability and convergence as separate acceptance criteria.
Use the current canonical Alps ROCm base rather than replacing it wholesale with Primus.

## Working agreement

- Check in with the user at meaningful milestones and when encountering a blocker or material design decision.
- The user initially authorized sequential Slurm experiments, then explicitly allowed short one/two-node submissions concurrently with the running job, keeping total requested/allocated agent resources around eight nodes maximum.
  Announce purpose, node count and duration; count pending jobs as well as running jobs against the budget.
  The later explicit approval for an eventual 16-node test supersedes the earlier eight-node limit for that baseline; use sequential admission, account for the existing interactive node and do not ask again solely about sixteen nodes.
  Earlier per-job and sequential approvals remain recorded below as historical evidence.
- Approved experiments use account `csstaff`, partition `mi300`, and exclusive nodes where communication requires them.
- The user clarified that overlapping diagnostic `srun` steps within the existing allocation are supported when using `--uenv-passthrough=ignore --account=csstaff --partition=mi300 --reservation=cscs`; this is an overlapping step rather than another job, and the later bounded-concurrency approval governs separate jobs.
- Do not modify Imanol’s scratch or discard historical experiments or existing user work.
- Keep image construction here and clean experiment orchestration in `amd-vs-gh200-moe-training`.
- Update this document with evidence, completed work, pending decisions and next actions.
- Do not infer GPU validation from static checks or short-run convergence from finite losses.

## Locations and reference evidence

| Item | Location or identity |
| --- | --- |
| Image project | `Alps-Images/apps/apertus-2/` in this repository |
| Benchmark project | `/iopsstor/scratch/cscs/boeschf/amd-vs-gh200-moe-training` |
| Clean harness | `experiments/apertus2/` in the benchmark project |
| Original scratch, read-only | `/iopsstor/scratch/cscs/schlag/beverin` |
| Reference Megatron | Benchmark `bench2/b2/mlm`, HEAD `d1277d86e` |
| Image Megatron pin | `0789b8e429553c2f17c350afa8619c69d01bec25` |
| Reference result | `bench2/b2/logs/b2-run22-custom-708188.out` |
| Reference geometry | mid-4b, 46.4B total / 3.88B active, 16 nodes / 64 GPUs, PP2 EP8, seq 8192, GBS 1152 |
| Reference steady window | Steps 10–26: mean 24.085 s, median 23.290 s, 6,122 tokens/s/GPU from mean |

The reference completed 26 iterations with finite logged losses and no reported skipped/NaN iterations.
This is short-run functional and throughput evidence, not production convergence.
The reference branch adds approximately 4,800 lines across 40 files relative to `41370f365`.
Patch files overlap and include malformed paths and rejected experiments; preserve the actual checkout and history as well as the patches.

## Implementation sequence

### 1. Freeze the reference

- [x] Capture Git identities, dirty-state evidence, Python-source hashes and source archives for the actual executed Megatron snapshot and optimizer checkout; non-Python/runtime completeness remains explicitly limited.
- [x] Snapshot launcher, EDF, tuning tables, Python hooks, relevant documentation and reference measurements without changing historical inputs.
- [x] Inventory active scratch dependencies and fingerprint binaries; make expensive reference-image hashing an explicit operation.
- [x] Mark unavailable or unhashed artifacts honestly instead of claiming a complete reproducible freeze.
- [x] Classify dependencies as already provided, required for reproduction, optional optimization, model tuning, or rejected/diagnostic in benchmark `experiments/apertus2/reference/DEPENDENCIES.md`.

### 2. Build a DRY experiment harness

- [x] Add one configuration resolver with validated precedence: platform, model where applicable, then experiment overrides.
- [x] Add one submission path, one batch runner and one per-rank launcher using argument arrays rather than command-string evaluation.
- [x] Keep model geometry, platform policy and experiment deltas separate; preserve historical launchers.
- [ ] Record resolved configuration, exact command, batch script, EDF, source identity, image identity and allowlisted post-hook environment.
- [ ] Record actual loaded library paths, node/GPU mapping, affinity, node health, cache state, warmup boundary and exit status.
- [x] Validate probe configuration and prepare frozen runs without executing Slurm; require explicit submission after approval.

### 3. Standardize image preparation and EDF execution

Local Podman images must first become a SquashFS on storage visible to every allocated node:

```bash
enroot import -x mount -o <image_name.sqsh> podman://<image:tag>
```

Record the Podman image identity and SquashFS checksum.
JFrog images may be used directly; resolve and record their immutable registry digest.
All Slurm container workloads use a rendered EDF with `writable = true`, `entrypoint = true`, `PMIX_MCA_psec = "native"`, and `com.hooks.cxi.enabled = "false"`.
Do not enable the `aws_ofi_nccl` injection hook for Alps Extended Images.
Use `srun --environment=run.edf.toml --mpi=pmix --network=disable_rdzv_get`, not `sbatch --environment`.
Verify actual environment after entrypoint/hooks; intended external overrides are not sufficient.
The legacy Primus reference EDF uses host injections and is evidence only, not a template for the Alps image.

### 4. Require memory-backed runtime storage

- [x] Implement filesystem-backing checks resolving symlinks and mount points, including nested shared mounts and noexec/read-only rejection; container validation passed on executable `/tmp` tmpfs in job `716120`; `/dev/shm` was correctly rejected as noexec in `716076`.
- [x] Use a private job/node/rank runtime tree on validated tmpfs and fail clearly when unavailable.
- [ ] Redirect AITER, Triton, TorchInductor, Torch extensions, TE/CK JIT, MIOpen, Python bytecode, application caches, compiler temporary files and profiler scratch.
- [ ] Keep datasets, checkpoints and final results durable on shared storage; flush diagnostics at bounded intervals and on orderly termination.
- [ ] Bound compiler concurrency, cache growth and profiling footprint; CPU, tmpfs and GPU allocations all consume MI300A HBM.
- [ ] Initially support cold runs plus image-baked AITER seeds; add optional immutable persistent snapshots with `off`, `read` and `readwrite` modes separately.
- [ ] Key reusable snapshots by image digest, Python ABI, Torch/Triton/HIP, AITER/TE/Turbo, GPU identity and custom kernel sources.
- [ ] Exclude unsafe rank-local mutable autotuning state; never put compiler locks or live writes on Lustre/NFS.

### 5. Reuse AITER infrastructure

- [ ] Factor genuinely reusable ROCm installation logic from `apps/vllm/sources/install-aiter.sh`; keep pins and patch selection app-owned and hashed.
- [ ] Retain MI300A’s 228-CU/six-XCD correction where required.
- [ ] Build core seed artifacts without a GPU against the Apertus image’s own Torch/ROCm; preserve inherited dependencies.
- [ ] Copy seeds into writable tmpfs before AITER import and verify reuse without unintended compilation.
- [ ] Warm training-specific attention kernels separately and check forward/backward numerics at real shapes.
- [ ] Do not automatically port GLM-5.3 FP8 inference tuning or copy old ABI-linked binaries from scratch.

### 6. Resolve foundational regressions

#### RCCL P2P

Treat `NCCL_P2P_DISABLE=1` as a workaround to reproduce and potentially narrow, not a hardware limitation or an established kernel diagnosis.
Git history `95c93c1` records inconclusive Sarus results.
Compare rccl-tests, PyTorch collectives and real Megatron communication using one and two nodes, P2P on/off, single/all-GPU visibility, fresh/reused buffers, multiple communicators and pipeline send/recv.
Use Enroot as the primary runtime and Podman only as a diagnostic comparison where useful.
Record actual RCCL/HIP/OFI paths so test executables and Torch cannot silently exercise different stacks.
Require repeated fresh-process initialization, numerical correctness, intended transport selection and clean teardown before removing or narrowing the default.

#### BLAS regression

Use the ticket’s plain, replicated-DTensor and sharded-DTensor BF16 matrix-product reproducer.
Compare the default backend with `torch.backends.cuda.preferred_blas_library("cublas")`, which selects rocBLAS on ROCm.
Isolate gathered operands as ordinary tensors and inspect shapes, strides, storage offsets, allocation history, native CPU stacks and GPU timelines.
Do not conflate host ATen timings with GPU-network timings or force rocBLAS globally without evidence.
PyTorch preference does not automatically switch TE or Triton kernels.
The ticket’s HIP 7.2 image differs from the current ROCm 10 image.

#### NUMA fragmentation

Record inexpensive boot-time, buddy-memory and NUMA observations per run.
Only run invasive THP-capacity screening on explicitly approved idle nodes.
Use same-allocation A/B comparisons and track tmpfs/host/GPU footprint and placement.
Reboots and boot-time memory policy are administrator actions, not automatic launcher workarounds.

### 7. Establish native hybrid-model correctness

- [x] Add a small end-to-end KDA/global-attention + MoE + MuonMD workload with layer-wise distributed optimization and selected recomputation.
- [x] Save a checkpoint, exit, resume in a fresh process and verify continued progress plus optimizer/scheduler/RNG state on the small archived-image workload.
- [ ] Exercise ordinary all-to-all and existing CXI UCCL separately.
- [ ] Check finite losses/gradients, numerical operations and clean teardown; existing component and dense-GPT tests are insufficient alone.

### 8. Port measured optimizations incrementally

Prioritize pinned Primus-Turbo grouped GEMMs, pointer access, FP32 gradient accumulation and tile selection.
Test uneven routing, empty experts, large weight spans, forward, input gradients and accumulated weight gradients against the existing TE path.
Private Turbo APIs require a compatible source pin or an explicit port.
Keep UCCL, T6 hierarchical rail-aligned exchange and plain all-to-all as alternative dispatchers.
Validate dispatch/combine forward/backward, routing weights, FP8 where enabled, memory and teardown.
Only evaluate the older OFI plugin/API shim if end-to-end measurements justify it, and rebuild against the current stack.
Curate a dependency-ordered patch series for compatibility, expert GEMMs, fused SSSGLU/LM head, attention/gates, routing/CE/permutation, Muon workspaces/batching, side streams and startup prewarming.
Check upstream equivalents, repair incomplete patches, test numerics and compare on the same nodes before enabling each group.
Do not apply every patch alphabetically or enable rejected experiments by default.

### 9. Apply Primus lessons selectively

Use Turbo operators, projection/tuning, preflight checks and profiling methods without replacing Swiss AI’s framework or Slurm orchestration.
Measure grouped GEMMs, synchronization across the whole MoE path, overlap, pipeline balance, recomputation and memory together.
Check documented options against pinned sources.
AMD MI300X/MI355X layouts and throughput claims are guidance, not validated defaults for four-APU MI300A Slingshot nodes.

### 10. Retune and scale

- [ ] Reproduce scratch-independent mid-4b before larger-model sweeps.
- [ ] Confirm exact Chonk/Megachonk definitions; establish PP/EP/VPP, batching, recomputation and total shared-HBM budgets separately.
- [ ] Retune dense hipBLASLt solutions, grouped-GEMM tiles and relevant FLA/attention configurations for each software/model combination.
- [ ] Do not assume old hipBLASLt solution indices or Triton 3.6 configurations remain valid.
- [ ] Validate real data, checkpoint save/resume and longer training-quality behavior.
- [ ] Keep CPU expert offload excluded as a capacity solution on unified-memory MI300A.

### 11. Measurement protocol

Progress from GPU-free checks to single GPU/node, two nodes, small multi-node proxies, 16-node mid-4b and then larger models.
Keep nodes, model, data and measurement windows controlled; use repeated/interleaved same-allocation A/B comparisons.
Separate compilation, warmup, steady state and profiling; report distributions and outliers rather than the best step.
Require clean process exit, including after the final training step.
Short benchmarks do not establish convergence.

### 12. Image and CI integration

Keep app-specific sources, patches and image-local tests under `Alps-Images/apps/apertus-2/`.
Keep base RCCL/plugin changes in the ROCm family; share only genuinely reusable helpers.
Update pins, constraints, Containerfile boundaries, explicit source/patch hash coverage, tests and documentation together.
Preserve canonical base refs, reproducible hashes, generated pipelines, tested markers and promotion policy.
Run shell syntax checks, ShellCheck where available, relevant unit tests and ref/hash/manual-build generation smoke checks.
Use real container builds to verify cache reuse and metadata refresh where build layout changes.
Never describe static/cache-layout checks as full GPU workload validation.

## Acceptance milestones

| ID | Acceptance | State |
| --- | --- | --- |
| M0 | Auditable reference inventory and clearly marked missing artifacts | Captured; later current-image checksum and 320 package metadata entries verified; historical identity/full binary provenance gaps explicit |
| M1 | One EDF-based launcher with provenance and verified tmpfs runtime | Validated on current-image single-node and two-node all-to-all hybrid runs |
| M2 | BLAS regression characterized; P2P policy supported by repeated evidence | BLAS ticket regression not reproduced in 716130; P2P matrix pending |
| M3 | Native hybrid KDA/MoE/MuonMD training saves/resumes cleanly | New candidate two-node/EP8 all-to-all restart passed in 716423; component UCCL passed in 716422, full-model UCCL remains unresolved; shutdown warnings retained |
| M4 | Scratch-independent mid-4b result | New candidate real-shape proxy restart passed with image-default router/no run-local repair in 716427; full mid-4b remains unvalidated |
| M5 | Curated optimizations improve end-to-end throughput without numerical regressions | Pending |
| M6 | Separately validated Chonk/Megachonk configurations and restart behavior | Pending |

## Decisions and open questions

- Retain the P2P workaround until reproducible evidence supports a narrower policy.
- Do not force a global BLAS backend or downgrade the base OFI plugin based on a different image’s result.
- Baseline candidate: archived app image `566e660d00b5556297d5a719ddcdd209`, SHA-256 `b3c19f91a0ea790dcf862451b6e51a6dd494bab5573d5ee9777d4f02bc0f17dd`, verified by a new full-file hash matching the saved checksum.
  Its build labels identify repository revision `a6e4e4a3e26000a55d1a79acea482955ee7dcb77`; app ROCm sources/patches/tests have no Git differences from that revision to current HEAD, but base SDK/build changes do differ.
  Treat it as an archived diagnostic baseline, not proof of a current-HEAD image build.
- Decide model-specific full-size resource requests from small-proxy memory evidence, not assumed capacity.
- Persistent cross-job cache publication follows the initial cold/image-seeded runtime implementation.

## Activity log

### 2026-11-03: execution started

Recorded the agreed full plan and execution boundaries.
Both repositories were inspected before edits; the image repository was clean and the benchmark repository contained pre-existing untracked checkouts, caches, results, core dumps and a local EDF.
Those existing files are being preserved.
First work package: auditable reference capture, clean experiment preparation, tmpfs runtime setup and standalone regression probes with GPU-free unit tests.
No Slurm commands or GPU workloads have been run for this work package.

### 2026-11-03: initial implementation and GPU-free verification

The clean harness now exists at benchmark `experiments/apertus2/`, with platform/experiment configs, one EDF preparation/submission path, frozen source/config checksums, a tmpfs runtime supervisor and standalone preflight/BLAS/collective probes.
Preparation defaults to no Slurm calls; explicit submission is separate.
The runtime applies reviewed experiment overrides after image initialization and captures allowlisted provenance plus actual loaded library paths in the probe process.
It seeds image-local AITER module binaries into private tmpfs, publishes bounded diagnostics periodically and catches nonzero process exit even after successful numerical results.
Cross-job persistent cache snapshots, hard cache quotas, full training/model configuration and the complete P2P test matrix are not implemented yet.

Reference capture: benchmark `experiments/apertus2/reference/run-708188/manifest.json` and source archives, approximately 3.3 MB total.
All selected inputs were accessible; the large Primus image checksum is explicitly pending and image versions remain documentary evidence.
The collector recomputed 17 steady samples: mean 24.0845176471 s, median 23.2896 s, 6122.439409 tokens/s/GPU.

Verification: 14 GPU-free unittest cases pass, Python AST syntax checks pass, and generated batch scripts pass `bash -n` and ShellCheck through the tests.
No GPU work, Slurm commands, image-content changes or P2P-default changes have been performed.

Additional archived evidence: single-node rccl-tests/MPI job `692473` passed with P2P unset, whereas two-node job `692455` failed with HIP `invalid device pointer` on 2026-10-23.
These are historical rccl-tests observations, not a new PyTorch reproduction or proof of the root cause.
The existing runtime/P2P workaround remains unchanged.

Next gate: request approval for one exclusive MI300 node, four preflight ranks, ten minutes, account `csstaff`, partition `mi300`, using the archived app SquashFS via EDF.
The prepared run is benchmark `experiments/apertus2/runs/preflight-566e660-v3/`.
This first test inventories the actual image and validates runtime backing, seeds and provenance; it does not run training, P2P overrides or invasive memory screening.

### 2026-11-03: first Slurm preflight approved

The user approved one exclusive MI300 node, four ranks and ten minutes for the preflight and suggested reservation `cscs`.
A reservation query confirmed `cscs` is active on partition `mi300` for group `csstaff`.
The prepared v3 preflight was submitted with `--reservation=cscs` and `--wait`; the wrapper records the exact command, approval, job ID and completion status alongside the run.
No additional training, P2P experiment or invasive capacity test is authorized by this approval.
Job `716076` completed unsuccessfully before Torch import; see the following outcome.

### 2026-11-03: preflight 716076 found noexec shared memory

The first approved preflight ran on `nid002412` and failed safely before importing Torch.
Inside Enroot, `/dev/shm` was tmpfs mounted `rw,nosuid,nodev,noexec,relatime`; the runtime correctly rejected it because JIT shared libraries require executable mappings.
All four ranks recorded failure status and no GPU workload or P2P experiment ran.
The versionable summary and artifact checksums are benchmark `experiments/apertus2/results/preflight-716076.json`.

The harness now records mount candidates/full mountinfo and allowlisted bootstrap provenance before enforcing storage policy.
The next candidate is `/tmp`, but it must pass the same memory-backed, writable, executable checks inside Enroot; there is no fallback to shared storage.
Reservation selection is now an explicit validated CLI/configuration option and is rendered into the frozen batch script.
Fifteen GPU-free unit tests pass, including reservation rendering, nested-mount/noexec rejection, post-hook overrides, exit-time failures, batch syntax and ShellCheck.
A fresh immutable run was prepared at benchmark `experiments/apertus2/runs/preflight-tmp-001/`.
The user explicitly approved the second one-node/ten-minute preflight, submitted as job `716120` using the frozen reservation-aware batch script.
Job `716120` passed on all four ranks with exit status zero and empty stderr; see the recorded outcome below.
Sixteen GPU-free tests now pass, including a regression check that storage-guard failures retain bootstrap diagnostics without launching a probe.

### 2026-11-03: preflight 716120 passed; execution resumed

The approved retry passed on all four ranks with no stderr and exit status zero.
Container `/tmp` passed the tmpfs/writable/executable checks; `/dev/shm` remains unsuitable for JIT shared libraries under this Enroot setup.
The versionable result and artifact checksums are benchmark `experiments/apertus2/results/preflight-716120.json`.
The native image identifies AMD Instinct MI300A `gfx942`, 228 CUs, Torch 2.13 and its actual HIP runtime; package versions and loaded libraries are retained per rank.
This validates the EDF launch, storage guard, Torch/GPU discovery, provenance and supervisor cleanup path, not BLAS, collectives, training or convergence.
The image does not contain the planned AITER training addition; missing seed state is reported rather than silently substituted.
Detailed environment review subsequently found the networking defaults had not been initialized; this pass must not be treated as Alps network-stack validation.

There was an avoidable pause waiting for the background completion notification; the user requested status and execution resumed with result inspection.
Next work: prepare a same-allocation, fresh-process default-versus-rocBLAS comparison on GPUs 2/3 and request approval before its Slurm submission.
Offline implementation and evidence updates can continue while approved jobs run.

### 2026-11-03: explicit ROCm environment bootstrap and paired BLAS probe

Inspection of job `716120` found `BASH_ENV` was present but `NCCL_NET`, `FI_PROVIDER` and the Alps initialization marker were absent.
A direct Python command does not source `BASH_ENV`, and this ROCm base does not have the NVIDIA entrypoint hook used by CUDA images.
The harness now launches a small Bash bootstrap with inherited `BASH_ENV` disabled, explicitly sources `/opt/alps/env/source-alps-env.sh`, and verifies the initialized Alps marker/AWS Libfabric/CXI defaults before applying experiment overrides.
This fixes the launcher rather than silently using TCP or changing image transport policy.
The batch script also records the Slurm network setting explicitly.

A paired BLAS experiment now runs default/rocBLAS/default/rocBLAS in fresh Python/RCCL processes on the same allocation with separate rendezvous ports.
It uses two ranks on GPUs 2/3, keeps P2P explicitly disabled, and adds ordinary gathered-tensor/contiguous-clone controls at matched global shapes.
Twenty GPU-free tests pass, including explicit environment bootstrap, direct-Python rejection, fresh-process/port ordering and exit-time failure propagation.
The next prepared run is benchmark `experiments/apertus2/runs/blas-compare-001/`: one exclusive node, two ranks, fifteen minutes, `csstaff`/`mi300`, reservation `cscs`.
This BLAS/bootstrap validation requires user approval before Slurm submission; no training, P2P enablement or invasive fills are included.

### 2026-11-03: paired BLAS 716130 passed; actual hybrid training is next

The user approved the paired BLAS/bootstrap comparison, submitted as job `716130` on one node/two ranks using GPUs 2/3 and reservation `cscs`.
All four fresh-process cases per rank passed numerical checks and clean exit.
The explicit bootstrap produced the required Alps initialized marker, `NCCL_NET=AWS Libfabric`, `FI_PROVIDER=cxi` and `NCCL_P2P_DISABLE=1`.
The ticket’s roughly 500 ms sharded product did not reproduce: the default backend took about 0.43–0.46 ms versus 0.38–0.40 ms for rocBLAS at these tested shapes.
This is not a global BLAS recommendation, a TE/Triton comparison, an inter-node network test or proof of pretraining.
Versionable evidence: benchmark `experiments/apertus2/results/blas-716130.json`.

The user asked whether pretraining had actually been established; it has not.
Prioritize the real hybrid training/restart milestone now rather than waiting for an exhaustive P2P matrix.
A small model definition and shared command builder now target four layers (3:1 KDA/global), hidden size 256, eight routed experts, top-2, EP2, sequence 128 and global batch 8.
The test runs actual Megatron/MuonMD forward/backward/updates to iteration 2, saves, then starts fresh processes to resume through iteration 4 with model/optimizer/scheduler/RNG checkpoint checks.
The cross-rank summary accounts for loss and resume messages being emitted on different ranks.
Prepared candidate: benchmark `experiments/apertus2/runs/train-hybrid-002/`, one exclusive node/four ranks/thirty minutes, `csstaff`/`mi300`, reservation `cscs`, P2P disabled.
The user approved this training test and then granted standing approval for sequential Slurm experiments with no concurrent jobs.
The prepared training run has been submitted using the unified `--prepared --submit --wait` path; completion evidence is pending.
Twenty-seven GPU-free tests pass, and the unified `--prepared --submit --wait` path now records completion without ad hoc submission wrappers or duplicate submissions.

### 2026-11-03: hybrid attempt 716177 failed in mock-data initialization

Job `716177` (`train-hybrid-002`) initialized the four-rank hybrid model and MuonMD optimizer but failed before the first training update.
Megatron writes cached mock-data indexes only on global rank 0, while the launcher incorrectly supplied a different `--data-cache-path` for every rank.
Nonzero ranks raised `FileNotFoundError` for their private document-index files; no checkpoint or pretraining success is claimed.
The corrected mock-data command omits the cache path, selecting deterministic in-memory index construction on every rank without shared-filesystem scratch.
JIT caches and bounded diagnostic logs remain in verified executable tmpfs.
The command-builder regression test now rejects a cache path for this mock workload; all 27 GPU-free tests pass.
Versionable failure evidence: benchmark `experiments/apertus2/results/train-hybrid-716177.json`.
Preparing immutable retry `train-hybrid-003` with the same one-node/four-rank/30-minute resources and reservation `cscs`.

The user clarified that `/tmp` and `/dev` are node-shared among container ranks.
The defect was distinct `mkdtemp` directory names, not mount isolation; shared mutable state must use an explicit common job/node path, with separate multi-node handling where needed.
Mode-0700 rank directories remain intentional for mutable JIT caches and logs.
The corrected retry was submitted with `--wait`; the harness holds a shared advisory submission lock and checks unfinished sibling submissions.
Twenty-nine GPU-free tests now pass, including lock contention across different runs and rejection of unfinished jobs.

### 2026-11-03: AITER reuse review while the hybrid retry runs

The vLLM installer already provides pinned source verification, app-owned patch application, wheel packaging with the image Triton retained, and GPU-free `module_aiter_core` seeding.
Do not reuse its unconditional `pip install -r requirements.txt` in Apertus: the selected AITER revision pins pandas 3.0.3, einops 0.8.2, pybind11 3.0.4 and ninja 1.13.0, conflicting with Apertus’s existing lock.
Upstream `setup.py` at commit `50da036acdedec2dd596f93188d6c615e2561672` has looser runtime requirements for those packages; its non-Triton-only mode additionally requires `flydsl==0.3.4.1`.
A reusable build helper should separate source/build/core seeding from app-owned dependency installation, preserve system Torch/Triton/NumPy under constraints, and explicitly supply setuptools-scm/vcs-versioning build requirements rather than permitting implicit setup-time downloads.
Keep the MI300A 228-CU/six-XCD patch, omit inference tuning, hash helper and patch inputs, and verify the installed package plus training forward/backward kernels rather than treating a core seed as complete attention precompilation.
No image-content changes or image builds have been made during this review; end-to-end hybrid functionality remains the immediate gate.

### 2026-11-03: native hybrid training/restart passed in 716179

Corrected archived-image run `train-hybrid-003` passed with zero job exit, all four rank supervisors/probes passed, and the frozen cross-rank summary passed.
Actual KDA/global-attention + MoE + MuonMD training reported losses 7.558259, 7.532846, 7.397711 and 7.328704 at iterations 1–4, with finite gradient norms 1.598, 2.029, 1.680 and 1.568 and no skipped/NaN iterations.
The first processes saved/exited at iteration 2; fresh processes explicitly loaded iteration 2, executed iterations 3–4 and saved/exited at iteration 4.
Both checkpoints include model, optimizer, scheduler and RNG metadata with expected consumed samples, and all four native layer-wise optimizer sidecars were found and fingerprinted for each checkpoint.
Versionable evidence: benchmark `experiments/apertus2/results/train-hybrid-716179.json`.
This establishes small synthetic hybrid functionality/restart only: full-size capacity, multi-node operation, performance and convergence remain unvalidated.
The first iterations include substantial first-use compilation; native logging retains the first iteration in the next log interval before resetting, so this four-update restart test is not a steady-state throughput estimate.

The user supplied the current-branch image `jfrog.svc.cscs.ch/docker-group-csstaff/alps-images/apertus-2-megatron-rocm:alps7-dev-40f69e8a90de66aa12b1b4e81bba68ce`.
The registry returned manifest digest `sha256:e0177f0ddbde95ae4abba2b0c951403e48828c35ecb4a9b5a9d7e09e3d63661a` through the standard bearer-token challenge flow; no image layers were pulled for digest resolution.
Prepared/submitted repeat `train-current-001` references that immutable digest directly in EDF, one exclusive node/four ranks/30 minutes with reservation `cscs`, after job 716179 completed.
The new snapshot also captures per-stage actual Megatron-child library provenance and validates the separate per-DP-rank layer-wise optimizer checkpoint files used by native resume.
Thirty-one GPU-free tests pass.

### 2026-11-03: current image and sequential cross-node preparation

Current-image one-node job `716181` is submitted directly from its immutable JFrog reference through EDF; no user-side pull/export or SquashFS conversion was performed.
Registry config digest `sha256:68b6914713c122ce71f7b7657a3560b13ee8b02d3a47a383f8db4af7e341b791` identifies build revision `0882e52228ae9b40c83c6e120e38000d43e51079`.
Comparing that revision with branch HEAD `d77367619d11516b424bc11779a453297686460e` found no differences in ROCm/common/app ROCm software inputs; differences under the inspected paths were CUDA Containerfile/documentation only.
Versionable identity: benchmark `experiments/apertus2/results/current-image-identity.json`.

Prepared, not submitted: `train-current-2node-001`, two exclusive nodes/eight ranks/EP8/30 minutes on the same digest.
The configuration resolver now supports repeated ordered `--experiment` files, so a minimal `configs/overlays/two-node-ep8.json` reuses the existing training environment instead of duplicating it.
EP8 with four ranks per node forces actual expert exchange across nodes; this is a small cross-node functionality/restart gate before full-size performance work, not an exhaustive microbenchmark matrix.
A separate `mid4b-shape-proxy.json` preserves captured mid-4b projection/expert dimensions while reducing depth, sequence and vocabulary; it is unvalidated and has not been submitted.
Newly prepared runtime snapshots record allocator peaks only for already-initialized Torch contexts plus process/NUMA memory observations, without importing Torch or initializing a context for diagnostics.
Those GPU allocator values explicitly exclude CPU/tmpfs/other-library shared-HBM allocations.
Thirty-nine GPU-free tests pass, including ordered overlays, per-rank optimizer sidecars, child-provenance exit preservation, registry digest-byte checks and noninitializing memory observation.

The archived training logs include TE’s embedded `[aiter]` forward/backward kernel messages.
These do not demonstrate that the standalone `amd-aiter` Python package is installed; TE’s working backend is the baseline and direct AITER attention remains a separate, measured optimization candidate.

### 2026-11-03: source-port audit and next-model boundaries

A separate temporary bare Git repository compared exact native `0789b8e` and executed reference `d1277d86e` without changing or fetching into the historical checkout.
Forty-five file differences and 52 selected reference environment flags are inventoried with per-file hashes in benchmark `experiments/apertus2/reference/native-port-inventory.json`; `NATIVE-PORT.md` records candidate ordering.
The native layer-wise optimizer and checkpoint implementation already match the executed reference.
The raw difference also deletes native tranche-end code/documentation/tests; those deletions are excluded from optimization porting.
A3/T7 diagnostic/experimental code is not a required dependency merely because it exists in the saved source.
No optimization patches or image-content changes have been applied from this audit.

Earlier documentary model research was found in benchmark `bench2/APERTUS2_LADDER.md` and captured under `reference/model-notes/`, with structured non-runnable evidence in `reference/model-targets-documentary.json`.
It distinguishes trained KDA Chonk (42 layers, hidden 3584, expert FFN 1920) from the earlier SWA Chonk and documents MegaChonk’s 61 layers/hidden 8192/3072 routed vs 6144 shared width.
These are secondary notes, not new primary-source verification; MegaChonk dense FFN remains discrepant (14336 in recorded W&B vs 30720 in the size file).
The current smoke/proxy builder intentionally does not represent those full recipes yet: it needs arbitrary-depth/final-global attention patterns, three dense layers, independent shared width, PP/VPP/checkpoint support and measured shared-HBM capacity without CPU expert offload.

New snapshots capture bounded compiler/training process activity from the existing supervisor’s `/proc` view, avoiding extra diagnostic Slurm steps.
Historical `CHONK.md` records that incorrectly configured overlapping sampler steps once disrupted VNI setup; this is not a general prohibition on overlapping diagnostic steps.
Prepared successor `train-current-2node-002` includes these observations and supersedes only the intended use of the earlier unsubmitted `-001`; both frozen directories are preserved.
Forty GPU-free tests pass.

The pinned native timing/logger implementation was inspected: its first-step report intentionally does not reset accumulators, so the second report combines the first two updates before normal resets begin.
Recorded smoke losses are finite logged window values, not necessarily independent per-update losses, and smoke timing is not reported as optimized throughput.

### 2026-11-03: current-branch hybrid training/restart passed in 716187

The first current-image attempt `716181` was cancelled by the agent after prolonged execution with no visible iteration/checkpoint evidence.
Its underlying cause is unresolved; the initial inference of an import stall from the missing argument banner was not justified.
Versionable cancellation record: benchmark `experiments/apertus2/results/train-current-716181.json`.
SSH diagnostic attempts were blocked by local configuration/host verification, and an attempted `srun --network=none` failed interconnect setup; neither result diagnoses the training/image issue.
The user supplied the correct overlapping-srun flags; read-only `ps` and image-source inspection succeeded with `--jobid=716187 --overlap --uenv-passthrough=ignore --account=csstaff --partition=mi300 --reservation=cscs`, the run’s EDF, `--mpi=none` and the normal `disable_rdzv_get` network option.
No SSH configuration, host-key checks or user credentials were modified.

The bounded retry `train-current-diagnostic-001` (`716187`) used import timing, two-minute in-process faulthandler dumps, process activity snapshots and a ten-minute probe budget inside a fifteen-minute job.
The first stack dump showed actual forward execution in TE router GEMM, not an import deadlock.
Both stages completed successfully: roughly 291 seconds for initial training/checkpoint and 73 seconds for fresh-process resume, including startup/JIT/diagnostics.
All four rank supervisors/probes and the frozen cross-rank summary passed; losses/gradient norms were finite and skip/NaN counters zero.
Logged loss windows: 7.558144, 7.532505, 7.397187 and 7.329075; finite gradient norms 1.598, 2.029, 1.668 and 1.568.
The run saved/exited at 2, explicitly loaded 2 in fresh processes, executed 3–4 and saved/exited at 4.
Native resume’s separate layer-wise optimizer sidecars were inspected on every rank for both checkpoints.
Actual Megatron-child library provenance confirms the image SDK’s Torch/HIP/hipBLASLt/rocBLAS/RCCL, TE and causal-conv1d libraries plus `/usr/lib/libcxi.so.1.5.0`, `/usr/lib/libfabric.so.1.32.0` and `/usr/lib/librccl-net.so`.
Image Megatron HEAD was directly checked as `0789b8e429553c2f17c350afa8619c69d01bec25`.
Versionable result: benchmark `experiments/apertus2/results/train-current-716187.json`.
This establishes small synthetic hybrid training/restart on the supplied immutable current-branch JFrog image, not full-size capacity, convergence, optimized throughput or repeated startup reliability.

Submitted after 716187 completed: `train-current-2node-003`, two exclusive nodes/eight ranks/EP8/up to thirty minutes on the same digest, reservation `cscs`, P2P disabled, direct JFrog EDF reference and periodic in-process stack dumps.
Earlier unsubmitted frozen two-node candidates are preserved.
Forty-two GPU-free tests pass, including process-group termination and durable evidence publication on a diagnostic timeout.

### 2026-11-03: real-dimension proxy prepared while cross-node training runs

Prepared, not submitted: benchmark `train-mid4b-shape-001`, two nodes/eight ranks/EP8 on the same current image, retaining mid-4b projection/expert dimensions but only four layers, sequence 512 and vocabulary 1024.
This is a native kernel-shape/functionality/checkpoint candidate, not the complete mid-4b architecture or a reference-performance reproduction.
New checkpoint inspection loads each rank’s actual layer-wise optimizer sidecar and colocated model shard, while still checking the complete expected sidecar inventory.
This preserves cross-rank coverage without multiplying shared-storage checkpoint reads and CPU/HBM occupancy by the number of ranks.
Existing frozen runs remain unchanged.
Forty-three GPU-free tests pass, all harness Python files parse, and both repositories pass `git diff --check`.
The app README now documents direct registry EDF references, explicit native Bash environment initialization, validated tmpfs scratch and the scope of the successful external hybrid test.
No image-content edits, image rebuilds, commits or pushes have been performed in this work package.

### 2026-11-03: cross-node 716191 failed; diagnostic interaction control submitted

Two-node/eight-rank/EP8 job `716191` reached actual hybrid forward execution and CK attention backward compilation, then failed before a reported completed iteration.
Training children on ranks 0 and 4 returned `-11` (SIGSEGV); subsequent TCPStore disconnects and CXI cleanup-busy messages are secondary observations, not proof of a transport root cause.
The logs stop during the second two-minute faulthandler dump with Triton background threads active.
That temporal correlation motivated an isolation control, not a claim that faulthandler is the cause.
Versionable failure evidence: benchmark `experiments/apertus2/results/train-current-2node-716191.json`.

Submitted sequentially after completion: `train-current-2node-004`, same image/model/eight-rank EP8/all-to-all/network policy and two-node thirty-minute resources, but periodic stack dumps disabled while fatal-crash tracebacks remain enabled.
Do not submit the real-dimension proxy until cross-node functionality is established.
Newly rendered training `srun` commands explicitly include `--uenv-passthrough=ignore`, and one shared read-only activity-command builder now carries the user-specified account/partition/reservation/EDF/network policy.

The user specified the actual UCCL EP environment, excluding vLLM-only settings: `FI_PROVIDER=cxi`, `UCCL_EP_TRANSPORT=cxi`, `FI_CXI_ENABLE_WRITEDATA=1`, `UCCL_EP_CPU_TIMEOUT_SECS=600`.
These are recorded in benchmark `configs/overlays/uccl-env.json`, accepted by configuration validation and retained by runtime provenance; `VLLM_DEEPEP_BUFFER_SIZE_MB` is not included or accepted.
The overlay configures only environment and does not secretly change dispatcher selection; the active retry remains ordinary all-to-all.
Forty-five GPU-free tests pass.

### 2026-11-03: UCCL training candidate wired with explicit environment validation

The harness now has an explicit `dispatcher=uccl` option selecting native Megatron `--moe-token-dispatcher-type flex --moe-flex-dispatcher-backend deepep`, supplied by this image’s UCCL-backed DeepEP wrapper.
Configuration validation rejects this selection unless all four user-required CXI/WRITEDATA/CPU-timeout values are present; vLLM settings remain excluded.
Prepared, not submitted: `train-current-uccl-001`, two nodes/eight ranks/EP8, BF16 dispatch, no periodic stack timer, the same immutable current JFrog digest and reservation `cscs`.
It is an alternative candidate, not a passing UCCL training result, and cannot be submitted until the active ordinary-all-to-all retry has finished.
Forty-six GPU-free tests pass.

### 2026-11-03: two-node all-to-all passed; UCCL configuration assertion isolated

Job `716196` (`train-current-2node-004`) passed all eight rank supervisors/probes and the frozen cross-rank summary with empty Slurm stderr.
It completed iterations 1–2, checkpointed/exited, explicitly resumed from iteration 2 in fresh processes, completed 3–4 and checkpointed/exited at 4.
Logged loss windows were 7.555894, 7.535254, 7.391277 and 7.322845 with finite gradient norms 1.601, 2.087, 1.719 and 1.573 and zero skip/NaN counters.
EP8 spans two four-APU nodes, establishing small synthetic cross-node native hybrid functionality/restart using ordinary all-to-all on the current JFrog digest, with P2P disabled.
Versionable evidence: benchmark `results/train-current-2node-716196.json`.
This replay passed without periodic stack dumps; the preceding SIGSEGV cause is still not established by one control.
Baseline training now leaves periodic dumps off and fatal-crash tracebacks on; `startup-diagnostics.json` explicitly opts into the timer.

Submitted after 716196 completed: UCCL `716202` (`train-current-uccl-001`) with the four user-required environment settings.
The proxies initialized CXI domains correctly, but the first internode combine failed `num_max_rdma_chunked_send_tokens >= num_warps_per_forwarder`.
The upstream wrapper’s EP8 combine default is `Config(Buffer.num_sms,4,256,6,128)`, while the image-local high-throughput numerical test explicitly supplies `Config(64,8,512,16,512)`.
The pinned wrapper already supports environment-selected dispatch/combine configurations and matching-SM validation; no image patch/rebuild is needed to test those settings.
Versionable failure record: benchmark `results/train-current-uccl-716202.json`.

Current active retry: job `716204`, `train-current-uccl-002`, two nodes/eight ranks/EP8/thirty minutes, same immutable JFrog EDF image/account/partition/reservation, required CXI settings plus `UCCL_EP_DISPATCH_CONFIG=64,8,512,16,512` and `UCCL_EP_COMBINE_CONFIG=64,8,512,16,512`.
The frozen configuration was rechecked after an interrupted response; no duplicate submission was made.
The harness validates paired five-positive-integer configs, even/equal SM counts, and records both overrides in versionable child-provenance reports.
Forty-eight GPU-free tests pass.

Prepared successor proxy `train-mid4b-shape-002` includes rank-local checkpoint verification and no periodic dump timer; earlier frozen proxy/candidate directories remain preserved and unsubmitted.
The shared `summarize.py --record` path now creates versionable success reports without ad hoc per-run formatting or overwriting a different existing record.

### 2026-11-03: UCCL numerical control passed; training and real-shape failures recorded

UCCL job `716204` passed the previous configuration assertion with both `64,8,512,16,512` overrides, then multiple training children exited with SIGSEGV in native `internode_combine` before a completed update.
The fatal Python stacks identify `deep_ep/buffer.py:1885` called from Megatron `FusedCombine.forward` without combine routing weights.
Periodic stack dumps were disabled, so this is not the earlier timer correlation.
Versionable failure record: benchmark `results/train-current-uccl-716204.json`.

Ordinary-all-to-all real-dimension proxy `716209` initialized approximately 800,172,160 local parameters but failed before its first completed update.
Rank 4 exited with SIGBUS in TransformerEngine `general_gemm` through `router_gating_linear`; its sampled child RSS was about 4.8 GB, with no reported OOM.
This does not establish a capacity limit or validate real-shape training.
The inspected pinned TE ROCm GEMM and bindings contain neither a rocBLAS backend nor the historical `NVTE_USE_HIPBLASLT`/`NVTE_USE_ROCBLAS` switches; those old knobs are not an appropriate retry.
Versionable failure record: benchmark `results/train-mid4b-shape-716209.json`.

Job `716216` passed the image-local numerical UCCL test on all eight ranks across two nodes, with exact dispatch/combine values, explicit buffer destruction and subsequent RCCL work; Slurm stderr was empty and frozen acceptance passed.
The harness invokes the existing image test rather than copying its implementation.
This test uses width 512, supplies combine routing weights, reserves a 20% buffer margin and explicitly configures buffer construction, unlike the failing Megatron path.
It is a component correctness result, not UCCL training validation.
Versionable evidence: benchmark `results/uccl-component-716216.json`.

Active width-isolation run: job `716219`, `train-current-uccl-width512-001`, two exclusive nodes/eight ranks/EP8/thirty minutes on `csstaff`/`mi300`, reservation `cscs`, same immutable current JFrog digest and required CXI settings/paired overrides.
Its model differs from the tiny baseline only in name and `moe_latent_size=512` instead of 128; hidden size remains 256, deliberately allowing an expanded diagnostic projection rather than a production architecture.
Require fresh-process checkpoint/resume and all-rank clean exit before claiming success.
No image-content changes, rebuilds or dependency installations have been made; fifty GPU-free tests pass.
Next: evaluate the width control, then isolate buffer construction and optional-weight handling if necessary; obtain native diagnostics for the separate TE router SIGBUS before retrying full shapes.

### 2026-11-03: bounded concurrency approved; native TE replay submitted

The user explicitly broadened approval to short one/two-node concurrent submissions while keeping total resources around eight nodes.
The live launcher retains sequential behavior by default and now accepts explicit `--node-budget 8` for approved short concurrent jobs.
Admission is serialized only until a submission record/job ID exists, counts unfinished/pending runs conservatively and rejects over-budget or larger/longer candidates.
The legacy sequential lock remains compatible with the already running width-control launcher; normal callers still cannot silently opt into concurrency.
Fifty-three GPU-free tests pass, including node-budget enforcement, pending-job accounting and admission-lock release before waiting for completion.

Submitted job `716226`, `train-mid4b-shape-gdb-002`: two exclusive nodes/eight ranks, fifteen-minute allocation, `csstaff`/`mi300`, reservation `cscs`, same immutable JFrog digest and ordinary-all-to-all real-shape proxy.
It wraps each training child in image-local GDB, stopping at the first SIGBUS/SIGSEGV and recording twelve native frames per thread, loaded-library paths and process mappings, with no core dump and a 600-second supervisor budget.
Host GDB controls verified clean exit 0, explicit child exit 7 and trapped SIGBUS returning 255; this is debugger-wrapper validation, not GPU validation.
An earlier unsubmitted `train-mid4b-shape-gdb-001` is preserved; host controls caught and corrected its post-exit diagnostic handling before the submitted snapshot was prepared.
Together with width-control job `716219`, the two requested runs total four nodes, not eight.

A read-only overlapping EDF step in 716219 found all four training children on its first node waiting on `g++`/`ld` for `megatron/core/datasets/libbfd_pack.so`, before the first training update.
The exact native `gpt_dataset.py` invokes `_load_bfd_c_library()` unconditionally at import, can compile directly into its source directory and has no cross-process compile lock; this startup behavior requires separate investigation and is not evidence about UCCL combine width.
No running model or frozen input was changed.

### 2026-11-03: width control 716219 inconclusive at startup

Job `716219` ended before distributed initialization, not inside UCCL dispatch/combine.
Rank 7 exited 1 after repeated TCPStore connection timeouts to rank 0 on `nid002412:36220`.
Rank 0’s last durable activity snapshot at 587 seconds still showed the import-time BFD helper compiler/linker chain, with no completed training update.
An earlier overlapping EDF process inspection identified the exact `g++ -O3 -shared -fPIC -o /opt/megatron-lm/megatron/core/datasets/libbfd_pack.so .../bfd_pack.cpp` command on all four first-node ranks.
Whether those identical pathnames resolve to shared writable backing across rank containers was not established.
Slurm ultimately reported `FAILED 9:0` during peer-failure cleanup; there is no OOM report, and exit 137 is not a capacity result.
The width-128 versus width-512 UCCL question remains unanswered.
Versionable evidence: benchmark `results/train-current-uccl-width512-716219.json`.
The TE native-debugger replay `716226` is the only remaining active agent job and has passed the BFD import stage on rank 0.

### 2026-11-03: native UCCL width replay submitted within concurrency budget

Submitted job `716244`, `train-current-uccl-width512-gdb-001`, two exclusive nodes/eight ranks/EP8, fifteen minutes, `csstaff`/`mi300`, reservation `cscs` and the same immutable current JFrog digest.
It repeats the diagnostic width-512 model with required CXI settings and paired conservative overrides, adding native GDB capture and a 600-second supervisor budget.
Both model width and native-debugger effects remain explicit; this is not a steady-state performance comparison.
Job `716219` has finished, so the outstanding TE replay `716226` plus this run total four requested nodes under the explicit eight-node budget.
The image installer currently runs the dataset Makefile but does not explicitly build `libbfd_pack.so`; no installer/image change has been made on that observation alone.
Next evidence should distinguish startup failures from the native TE/UCCL faults rather than treating all nonzero Slurm exits as the same regression.

### 2026-11-03: native launch crash captured; optional combine weights ruled out in isolation

Width-512 UCCL training replay `716244` reached the first combine and trapped SIGSEGV on all eight ranks.
The main native stack is `libamdhip64.so` through `hipLaunchKernelExC`, `uccl::internode::combine` and `Buffer::internode_combine` (`uccl_ep_hip.cc:1214`).
Increasing communication width from 128 to 512 alone did not fix training.
This identifies the native failure path, not yet a root cause in HIP or UCCL.
Versionable evidence: benchmark `results/train-current-uccl-width512-gdb-716244.json`.

New matched numerical contract controls passed all eight ranks and frozen acceptance in `716249` (combine weights absent) and `716250` (weights present), with exact dispatch/combine and post-destroy RCCL checks.
Both use width 512, 32 tokens, deterministic routing to both nodes, the explicit constructor and 20% buffer margin from the passing component test, with paired reviewed configs and required CXI settings.
Missing combine weights alone does not reproduce the training failure; do not add dummy weights to training as an assumed fix.
Versionable evidence: benchmark `results/uccl-contract-no-weights-716249.json` and `results/uccl-contract-weights-716250.json`.

The TE native replay `716226` did not reproduce the earlier SIGBUS.
It completed two real-shape updates with logged losses 7.551108 and 7.040235, gradient norms 4.079 and 4.100 and zero skip/NaN counters, then remained in iteration-2 checkpoint saving until the 600-second supervisor deadline.
The final Slurm SIGKILL was cleanup, not an observed OOM.
This is partial real-shape execution evidence, not a passing checkpoint/restart result or a throughput measurement.
A checkpoint inventory attempt also hit a host-side filesystem-operation deadline; checkpoint completeness is not established.
Versionable evidence: benchmark `results/train-mid4b-shape-gdb-716226.json`.

Submitted two short, separate UCCL numerical controls: `uccl-contract-zero-margin-001` removes only the 20% sizing margin, while `uccl-contract-qps24-001` changes only the constructor QP count from 64 to Megatron’s default 24; combine weights remain absent in both.
Each requests two exclusive nodes for ten minutes with the required CXI settings and conservative paired configs.
Submitted `train-mid4b-shape-gdb-003`: same real-shape model/two-node EP8/all-to-all, fifteen-minute allocation, with GDB stopping at 480 seconds to emit native stacks before the 600-second supervisor deadline.
This one-shot native stop is distinct from periodic Python faulthandler dumps, which remain disabled.
Host GDB controls verified timed SIGINT capture returning 255 and an early clean exit returning 0 without waiting for the timer; fifty-five GPU-free tests pass.
At most six nodes are requested across these three runs, below the explicit eight-node budget.

### 2026-11-03: buffer controls passed; allocator interaction isolated next

Frozen acceptance passed on all eight ranks in UCCL contract `716295` (no buffer margin) and `716296` (24 rather than 64 QPs), both without combine weights.
Removing the margin or changing the QP count alone does not reproduce the training crash.
These results do not establish that every combination of buffer settings is safe in training.
Versionable evidence: benchmark `results/uccl-contract-zero-margin-716295.json` and `results/uccl-contract-qps24-716296.json`.
The common reporter now retains complete standalone-probe numerical/runtime evidence and all explicitly requested experiment environment keys, including allocator policy; older versionable records remain untouched.
Fifty-six GPU-free tests pass.

Saved post-hook environments show another material distinction: failing training enables `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`, while the passing numerical controls do not set it.
This is a hypothesis, not yet an allocator diagnosis.
Submitted complementary controls: `716299`, `uccl-contract-expandable-001`, adds only expandable segments to the passing weight-free numerical control; `716300`, `train-current-uccl-noexpand-gdb-001`, disables expandable segments in the failing width-512 GDB training configuration.
Both retain the immutable JFrog digest and required CXI/paired UCCL configs, use exclusive two-node allocations on `csstaff`/`mi300` with reservation `cscs`, and preserve explicit frozen configuration/provenance.
The component requests ten minutes; training requests fifteen minutes with a 600-second supervisor budget.
The no-expandable overlay is an experiment only; baseline/image defaults are unchanged.

Checkpoint-stack replay `716298`, `train-mid4b-shape-gdb-003`, remains submitted with a one-shot native stop at 480 seconds.
Together these three unfinished submissions request at most six nodes, below the explicit eight-node budget.
No app image patch, build, commit or push has been made during these diagnostic controls.

### 2026-11-03: expandable-segment component control passed

Job `716299` passed frozen acceptance on all eight ranks with empty Slurm stderr.
Its actual probe-child provenance confirms `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`; dispatch, weight-free combine and post-destroy RCCL checks were exact.
Expandable segments alone therefore do not reproduce the training crash in this small standalone workload; an interaction under actual training is not ruled out by this pass.
Versionable evidence: benchmark `results/uccl-contract-expandable-716299.json`.
The complementary real-training allocator-off replay `716300` and native checkpoint-stack replay `716298` are still outstanding, accounting for at most four requested nodes.
No baseline or image allocator policy was changed.

### 2026-11-03: allocator-off training still crashes; reduced Megatron API controls submitted

Job `716300` trapped SIGSEGV on all eight ranks with `PYTORCH_HIP_ALLOC_CONF=expandable_segments:False`, confirmed in post-bootstrap provenance.
Its native stacks show the same HIP launch address/path through `hipLaunchKernelExC` and UCCL combine as `716244`.
Disabling expandable segments does not fix training; allocator defaults remain unchanged.
Versionable evidence: benchmark `results/train-current-uccl-noexpand-gdb-716300.json`.

Submitted two model-free, GDB-instrumented numerical API controls, each on two exclusive nodes/eight ranks for ten minutes with a 360-second supervisor budget, same image and required CXI/paired configs.
`uccl-contract-buffer-preload-001` retains the passing direct Buffer path but imports the native Megatron fused-dispatch stack first, isolating import presence from API use.
`uccl-contract-megatron-001` uses native Megatron `get_buffer`, fused dispatch/combine and autograd, checking exact identity outputs, input gradients and zero probability gradients.
The latter changes only buffer destruction ownership to `explicitly_destroy=True` so teardown and subsequent RCCL can be verified; native sizing, QP and inferred-topology defaults are otherwise unchanged.
The native fused source path/hash and this lifetime exception are recorded explicitly, not hidden as a production patch.
Neither control executes the full hybrid model or validates training/restart.

Native debugger command construction is now shared by training and standalone probes rather than copied; fifty-seven GPU-free tests pass.
With the outstanding checkpoint-stack replay `716298`, at most six nodes are requested under the explicit eight-node budget.
No image-content change or baseline-policy change was made.

### 2026-11-03: real-shape fresh-load crash localized; fused API gradient check exposed a separate mismatch

Real-shape replay `716298` completed stage-0 updates/checkpoint, and all eight stage-0 training children exited normally.
Its stage-1 rank-0 log confirms a fresh-process checkpoint load at iteration 2, followed by SIGBUS in `libhipblaslt.so.1` CPU-side `TensileLite::Predicates::And`/solution selection before a reported resumed update.
Ranks 1–3 trapped SIGABRT during CUDAGuard/device handling; their causal relationship to the rank-0 fault is not established.
The planned diagnostic SIGINT did not trigger, so these are genuine trapped native faults, not timer-induced stops.
This improves checkpoint/fresh-load evidence but does not pass resumed training or full-size acceptance.
The earlier Python-only TE router trace and this native hipBLASLt trace motivate a small shape-matched router-GEMM reproducer; the twelve-frame native stack does not by itself identify every higher-level caller.
Versionable evidence: benchmark `results/train-mid4b-shape-gdb-716298.json`.

Megatron-import/direct-Buffer control `716312` passed all eight ranks and frozen acceptance.
Native fused-API/autograd control `716313` did not trap a native signal and reached exact forward dispatch/combine and the exact input-gradient check, but failed the exact-zero probability-gradient check with tiny nonzero values, up to about 4.74e-38 in rank 0’s captured log.
Do not relax the tolerance to conceal this mismatch or claim it explains the full-training HIP-launch SIGSEGV.
Versionable evidence: benchmark `results/uccl-contract-buffer-preload-716312.json` and `results/uccl-contract-megatron-716313.json`.

Submitted model-free controls `716315` (`te-router-smoke-001`) and `716316` (`te-router-mid4b-001`), one exclusive node/four independent GPU ranks each, ten minutes and 360-second supervisor budgets.
They invoke the actual native Megatron `router_gating_linear`/TE wrapper at BF16 input/weight and FP32 router output, with FP32 ATen forward/backward references and recorded tolerances/errors, without a process group or the full hybrid model.
Both use the training environment, including expandable segments, and native GDB capture now records `$_siginfo` as well as stacks/mappings.
Submitted `716317`, `uccl-contract-megatron-grad1-001`, two exclusive nodes/eight ranks/ten minutes, changing the explicit probability-gradient reference from zero to one and retaining strict equality.
Gradient observations are now recorded before assertions so a failure preserves actual/expected values rather than only an exception string.
All three retain the immutable JFrog digest, account `csstaff`, partition `mi300` and reservation `cscs`; total requested resources are four nodes.
Fifty-eight GPU-free tests pass; no image patch, rebuild, commit or push has occurred.

### 2026-11-03: isolated router shapes passed; expanded-gradient edge case confirmed

Isolated native TE router probes `716315` (smoke) and `716316` (real projection dimensions) passed both forward/backward calls on all four GPU ranks per job and frozen acceptance.
Real shapes alone do not reproduce the full-model/fresh-load hipBLASLt SIGBUS.
Versionable evidence: benchmark `results/te-router-smoke-716315.json` and `results/te-router-mid4b-716316.json`.

Unmasked fused-API unit-gradient experiment `716317` failed, but its interpretation needs two explicit corrections: summing all received weight slots includes unused routing entries, and a sum’s expanded gradient may have zero strides.
Do not interpret that unmasked exact-one reference as a valid proof of arbitrary dense probability-gradient corruption.
The pinned native Megatron backward only applies `.float()` to probability gradients, and the pinned UCCL bridge forwards raw pointers without carrying their strides.
Versionable evidence with this limitation: benchmark `results/uccl-contract-megatron-grad1-716317.json`.

Matched follow-ups passed frozen acceptance on all eight ranks: `716324` made the zero probability gradient contiguous, while `716325` masked unused routing slots for the exact-one reference.
The hook recorded the expanded zero-gradient input as shape `[64,2]`, stride `[0,0]` and only four backing bytes on every rank; making it contiguous restored exact input/probability gradients and post-destroy RCCL.
The masked unit-gradient input naturally had stride `[2,1]`, 512 backing bytes and exact gradients without forced copying.
This isolates a non-contiguous-gradient boundary hazard; it is not evidence that the earlier forward-combine HIP-launch SIGSEGV is fixed.
No tolerance was relaxed and no native source/image patch was made.
Versionable evidence: benchmark `results/uccl-contract-megatron-grad0-contiguous-716324.json` and `results/uccl-contract-megatron-grad1-masked-716325.json`.

Submitted `train-current-uccl-boundary-gdb-001`, two exclusive nodes/eight ranks/EP8, fifteen minutes, same width-512 model/image/CXI/configs and original expandable-segment policy.
Its opt-in wrapper records the first two dispatch/combine tensor shapes, strides, backing/logical sizes, devices and compute streams without copying tensors, synchronizing streams or changing call arguments; GDB records native signal details.
Loading the diagnostic wrapper imports the UCCL module earlier, an explicit instrumentation difference rather than an image change.
The wrapper is bounded/restorable and remains off by default; fifty-nine GPU-free tests pass.
The preceding controls have completed, so this is the only outstanding two-node agent submission.

### 2026-11-03: boundary tracer import-order correction

Boundary tracing run `716326` failed before pretraining because the new diagnostic wrapper imported `deep_ep` before Torch had loaded `libc10.so`.
This is an agent-introduced instrumentation bootstrap error, not a new training or transport result.
Versionable evidence: benchmark `results/train-current-uccl-boundary-gdb-716326.json`.
The live wrapper now imports Torch first; a GPU-free test mocks native imports and verifies the order, bringing the suite to sixty passing tests.
Frozen failed inputs remain unchanged.
Submitted corrected retry `716328`, `train-current-uccl-boundary-gdb-002`, with the same two-node/eight-rank/EP8/fifteen-minute resources, image/model/required environment and conservative configs.
It is the only outstanding agent submission, accounting for two nodes.
A future native hardening candidate for the independently confirmed expanded-gradient hazard is to materialize probability gradients before the stride-blind transport; no such image patch has yet been applied, and it would not establish a fix for forward-combine launch failure.

### 2026-11-03: training boundary identified asynchronous allocation mode

Corrected trace run `716328` reached the first forward combine and reproduced SIGSEGV on all eight ranks.
Native signal information identifies `SEGV_MAPERR` at host address `0x18` inside the same HIP launch path.
Captured dispatch and combine tensors were contiguous BF16 with matching logical/backing sizes, zero storage offsets and correct local GPU devices; this observed forward failure is not the expanded-gradient layout edge case.
Crucially, actual training passed `async_finish=True` and `allocate_on_comm_stream=True` for both dispatch and combine, unlike every preceding passing standalone control’s synchronous defaults.
The caller’s compute stream was the default stream (`0`) on each rank; receive counts varied naturally with actual routing.
Versionable evidence: benchmark `results/train-current-uccl-boundary-gdb-716328.json`.

Submitted two matched model-free native Megatron API controls, retaining the known-correct masked unit-gradient reference and exact forward/backward/teardown checks.
`uccl-contract-megatron-async-001` enables asynchronous completion but retains compute-stream allocation.
`uccl-contract-megatron-async-comm-alloc-001` enables both asynchronous completion and communication-stream allocation, matching the two actual training flags.
Both use two exclusive nodes/eight ranks for ten minutes, GDB signal details, the immutable current JFrog image, required CXI/paired configs and `csstaff`/`mi300`/reservation `cscs`.
These are the only outstanding submissions, at most four requested nodes under the explicit eight-node budget.
The new flags are explicit/typed and default off for older component controls; no image or training default was changed.

### 2026-11-03: asynchronous API controls passed; direct training execution controls submitted

Both asynchronous native API controls passed all eight ranks, exact masked forward/backward checks, explicit teardown and frozen acceptance: `716331` with compute-stream allocation and `716330` with communication-stream allocation.
The two flags observed in training are therefore not sufficient alone to reproduce the full-model launch crash in this small deterministic workload.
Versionable evidence: benchmark `results/uccl-contract-megatron-async-716331.json` and `results/uccl-contract-megatron-async-comm-alloc-716330.json`.

Submitted two explicit run-local training controls, each on two exclusive nodes/eight ranks for fifteen minutes with GDB and a 600-second supervisor budget, same immutable JFrog image and `csstaff`/`mi300`/reservation `cscs`.
`716335`, `train-current-uccl-synchronous-gdb-001`, forces the actual native fused dispatch/combine API arguments to synchronous completion/compute-stream allocation on the failing width-512 model; bounded boundary tracing records whether the requested flags reached Buffer.
The wrapper preserves tensor/group/handle arguments and changes both public fused API references used by token dispatch, rather than returning a mismatched event contract from a low-level Buffer hook.
`716336`, `train-mid4b-router-aten-gdb-001`, keeps ordinary all-to-all and the real-shape model but selects the router function’s existing FP32 ATen fallback by disabling only the local `moe_utils.te_general_gemm` alias.
Other TE operations and Torch’s global BLAS preference remain untouched; this is not an obsolete NVTE backend switch or a global rocBLAS policy.
Both controls are explicit typed configuration, restricted to the instrumented entrypoint, restored after execution and retained in actual training-child provenance as well as the frozen run configuration.
They are diagnostic controls, not established production fixes or performance results.
Sixty-two GPU-free tests pass, including argument preservation and router-only/restorable behavior.
The previous controls have finished, so at most four requested nodes remain outstanding under the explicit eight-node budget.
No image content or baseline policy was changed.

### 2026-11-03: real-shape restart passed; candidate image patches and manual base build started

Job `716336` passed frozen acceptance on all eight ranks: real-shape ordinary-all-to-all updates 1–2, checkpoint, fresh-process load at 2, updates 3–4 and checkpoint/clean child exit at 4.
Actual child provenance records router-only `aten` in both stages; other TE operations and Torch’s global BLAS preference were unchanged.
Logged losses were 7.551388, 7.040029, 5.527560 and 5.057086, with finite gradient norms and zero skip/NaN counters.
This is a four-layer synthetic proxy with real projection/expert dimensions, not full mid-4b capacity, performance or convergence.
Versionable evidence: benchmark `results/train-mid4b-router-aten-gdb-716336.json`.

The user explicitly authorized the repository manual build tools and asked that work continue.
Candidate ROCm-only Megatron patches now materialize dispatch gradients before event capture and default only router GEMM to the existing ATen path, with an explicit `APERTUS_ROCM_ROUTER_GEMM_BACKEND=te` diagnostic opt-in.
A separate late finalizer applies/audits these patches and builds the CPU BFD DSO; it avoids changing the earlier shared installer and invalidating expensive UCCL/TE layers unnecessarily.
The new helper is declared in the existing ROCm source-file hash list, the complete existing patch tree remains covered and both CUDA target hashes stayed unchanged in the ref smoke test.
Three GPU-free applied-source/BFD regression tests pass against the exact pinned checkout; tracked-shell syntax checks, explicit new-script syntax/ShellCheck and Python AST checks pass.
Image-local metadata checks verify patch checksums/application and CPU BFD numerics; GPU tests now cover router dimensions and the actual patched fused-autograd edge case without a repairing hook.
These changes are not yet accepted in a built image, and do not claim to fix the separate UCCL first-forward-combine launch crash.

The exact reference and native `RouterGatingLinearFunction`, `FusedDispatch` and `_load_bfd_c_library` are AST-identical: these are native-stack compatibility/hardening changes, not a missing reference source transplant.

Manual app/base build scripts were generated under `/tmp/opencode/` using the production clone URL and JFrog prefix.
The current helper-selected canonical base tag is `rocm10.0-ubuntu24.04-py3.12-torch2.13-alps7-dev-8780fe9102fd5a190ba8252cf3c40ac0`, which returned registry 404.
The original tested app used the older base tag ending `907db4d29b8e4d3f1e17cf0bda358e36`; do not relabel that image as the new canonical base.
The local Podman store is empty, so this first manual build cannot reuse expensive pre-existing layers despite preserving the correct recipe boundaries.
Started the generated canonical base build on the current interactive allocation `715971`/`nid002666`, with Podman graph/run storage on `/dev/shm`, no extra node request and memory/storage free-space guards.
Its log/status are `/tmp/opencode/build-apertus2-rocm-base-candidate.log` and `build-apertus2-rocm-base-candidate-status.json`; it is pulling the declared Docker Hub upstream normally.
Nothing has been pushed, promoted or committed.
Next: finish the canonical base, regenerate the app script after all source/test edits, build the candidate, export via the user-prescribed Enroot command and rerun image-local plus hybrid/proxy restart acceptance.

### 2026-11-03: manual build moved to native host Podman with confined state

The in-session Podman wrapper pulled the upstream base but failed the base APT step because its single-ID user namespace cannot chown `man-db` paths to their package owners.
The package helper’s APT sandbox fallback was not the missing fix; `dpkg` failed assigning ownership to `/var/cache/man.dpkg-new`.
Do not change base-image package policy or relabel this failed build as a completed canonical base.
Failure log/status: `/tmp/opencode/build-apertus2-rocm-base-candidate.log` and `build-apertus2-rocm-base-candidate-status.json`.

The user explicitly authorized `srun --overlap` outside the bwrapped session, with no files written outside the directories already in use.
The native host Podman exposes the normal full sub-UID/sub-GID mappings, unlike the wrapper; the native OCI runtime is `/usr/bin/runc`.
A read-only initial native check preceded the user’s directory restriction; all subsequent builder HOME/config/cache/tmp/auth/runtime/graph paths are explicitly confined rather than using default home-directory storage.
The agent-private `/tmp/opencode` is not visible from the host Slurm step, so shared builder scripts/configuration live under `/dev/shm/podman-22404-state/native-rootless-apertus2` and graph storage under `/dev/shm/podman-22404/native-rootless-apertus2`.
The native builder’s HOME is its own `home` subdirectory, its auth file is empty anonymous-pull state and no registry login/push is performed.
An overlapping sixteen-CPU step in existing allocation `715971` is now running the generated base script with memory/storage guards; no additional node was requested.
Native build log/status: `/tmp/opencode/build-apertus2-rocm-base-native.log` and `build-apertus2-rocm-base-native-status.json`.

Repository CI-helper tests passed: 264 passed, one skipped, using an isolated tool environment/cache under `/tmp/opencode`.
The three applied-source/CPU-BFD regressions, shell syntax/ShellCheck, AST validation and family-isolation ref smoke checks also passed.
The candidate app script is regenerated as `/tmp/opencode/build-apertus2-megatron-rocm-ready.sh` after source/test changes, but the app build still awaits its canonical base.

### 2026-11-03: real-engine cache smoke passed; export and EDF acceptance prepared

The real host Podman cache-layout smoke passed: metadata-only changes refreshed the revision label while preserving software layers and not reexecuting the marker software step; changing a copied component input reexecuted that step and changed the layers.
This is a marker-based real-engine cache-layout test, not a full image build or GPU acceptance.
Logs, image inspection JSON, recipe and result are retained under benchmark `runs/image-build-candidate-001/cache-smoke/`.

The canonical base remains active through the existing sixteen-CPU overlapping step, with app handoff already queued; do not launch a duplicate app build.
At the latest status read, available memory was about 403 GiB and graph storage free space about 279 GiB.
The candidate remains unbuilt/unvalidated at this point.

Prepared a guarded host export driver under the existing shared builder state and copied it to the durable build record.
It uses the requested `enroot import -x mount -o ... podman://...` form, confines Enroot runtime/cache/config/data/tmp state to the existing builder tree, limits compression to eight workers/4 GiB workspace, refuses to overwrite output and records the actual local image ID plus SquashFS size/SHA-256.
The export driver is syntax-checked and passes ShellCheck with its sourced builder environment; it has not yet been executed.

Added the harness `stack-component` probe for the image-local metadata/TE/router/KDA/convolution/Muon suite rather than duplicating its numerical tests.
It requires both the exact success marker and a clean child exit, rejects any router-policy environment override (including empty), and runs under the existing verified-tmpfs supervisor with fatal-only Python tracebacks and child-process provenance.
The harness now records the narrowly allowlisted `APERTUS_ROCM_ROUTER_GEMM_BACKEND`; it does not broaden capture to arbitrary application secrets.
All 66 GPU-free harness tests pass, including four new acceptance-safety tests.

The durable `validation-plan.json` records commands/resources for sequential image-local stack checks, the UCCL numerical/gradient regression, two-node small all-to-all restart, and the real-shape proxy without a run-local router override.
These are plans only: immutable EDF preparation awaits the exported image, and no new GPU job was submitted.
UCCL full-model training remains unresolved, and nothing has been pushed, promoted or committed.

### 2026-11-03: canonical base completed; exact reference/native stack differences recorded

The canonical base build completed successfully in 1855 seconds through the existing host overlap step.
Actual local image ID is `d94606f0f9b25eb9df664497286a0dd9bd81333aecd19326a286513db7ddec92`; full Podman inspection and completed base log/status are now durable in benchmark `runs/image-build-candidate-001/`.
The pre-existing handoff started the app build automatically, verified the canonical base ID and passed its current content-hash guard; no duplicate app build was launched.
App supervisor PID is `59253` in allocation `715971`, still sixteen requested step CPUs and no extra node request.

At the user’s request, `STACK-DIFFERENCES.md` now separates the executed reference, actually measured older native image and declared unvalidated candidate.
Read-only inspection of the current Primus image extracted 320 package metadata entries plus retained source/version evidence, without executing or modifying the image.
The full current-image SHA-256 is `7db644ae56cd4b076cb15d87fa3d47f0fc03221f49ccef35388c1a460d4e56ef`; this does not retroactively prove job-time byte identity.
Directly verified reference identities include Torch `2.10.0+git94c6e04`, HIP `7.2.53211`, TE `2.12.0.dev0+40434cf6`, AITER `0.1.1.dev1611+gf299f579a.d20260512` / HEAD `f299f579a63b8de4f008e19b6f8867dbfa9eaf39`, Turbo `0.2.0+3cd482d` / HEAD `3cd482def8bb4814b1799fe481555559fff040b8`, NumPy `2.4.4` and causal-conv1d `1.5.0.post8`.

The exact pinned Megatron diff has 45 files (19 added, 23 modified, three reference deletions); 43 compared reference Python files match the captured execution archive.
The two candidate Megatron patches are recorded separately, making 46 candidate/reference differing files.
The captured FLA KDA gate is byte-identical to native FLA pin `1820dba7e15fb927294779f23ac5097eb3927b25`; complete FLA equivalence is not claimed.
Reference optimizer Python source (`0.3.1`, `1f29ffd2`) differs from native `0.2.0`/`cc1385e` in 51 Python files including 26 package files, with an added batched Newton–Schulz API and different supported modes; source differences are not measured regressions.
The unpatched router/fused-dispatch/BFD-loader AST matches and byte-identical layer-wise optimizer/checkpoint/BFD C++ sources are reconfirmed.

The selected reference T6 FP8 dispatcher is distinct from the ordinary all-to-all name printed by its CLI, because its modified MoE layer selects T6 by environment flag.
The reference’s environment print also precedes unsetting three NVTE attention variables and changing CXI TX size/GDR level; the audit records those later changes instead of mislabeling the print as final environment.
Exact argv comparison identifies remaining recipe differences beyond shape: schedule/LRs, PP, generic/CE fusions, explicit router top-k scale, init/rotary settings and checkpoint behavior.
The native proxy is not a full mid-4b reproduction, and omitted arguments are not assumed equal to explicit reference values.

All audit data, full source patches, package/source metadata, selected native recipe snapshots and input checksums are retained under benchmark `reference/stack-comparison-2026-11-03/`.
No historical captures/runs were overwritten, no reference optimizations were installed during the comparison, and no GPU job was submitted for the audit.
Next: finish the active app build, record actual identity/provenance, export with the prepared Enroot driver, then image-local and ordinary-all-to-all restart acceptance without run-local router repair.

### 2026-11-03: focused runtime-patch/UCCL review recorded; implementation paused by user

The user clarified that Megatron source/runtime patches, Primus-Turbo atop our base, optimizer relevance and the exact A2-versus-native UCCL differences are the priority, and explicitly requested no implementation changes.
The subsequent request authorizes documenting these findings, not implementing the proposals.
`STACK-DIFFERENCES.md` and benchmark `reference/stack-comparison-2026-11-03/runtime-and-uccl-followup/review.json` now record all of them, with 32 checksummed source/hook/build-evidence snapshots.
The existing build inputs/workflow are unchanged; no new dependency installation, reference patch application or GPU validation job was launched.
Earlier proposed export/validation/optimization steps remain pending rather than permission to expand implementation during this review pause.

Recorded runtime behavior includes the B2 Triton autotuner exception guard, Turbo tile-selector replacements, Origami-ranking exception fallback, share-storage extension substitution, exact selected prewarm flags, diagnostic hook chaining and the TE workspace-signature guard.
Patched checkout, environment-selected paths and runtime replacements are distinct audit layers; package installation alone cannot reproduce the missing Megatron integrations.
Turbo native source build at reference pin `3cd482def8bb4814b1799fe481555559fff040b8` and modern Turbo adoption are separate proposals, with explicit dependency/private-API/compiled-extension/topology risks.

The optimizer follow-up distinguishes Megatron NS workspaces/K7’s own batched routine from the newer library batched helper.
Ordinary NS/SYRK functions and simple/polar_express/cans coefficient sets match by AST; upgrading the library alone does not reproduce K7.

The A2 UCCL saved build is upstream `0578e0b819831f35c4ad17008e65e2159746ef80` plus `src.diff`, byte-identical to the later committed delta ending at `6f2a0084f564dd796e5a90bcb0d2d862abaac80c`.
It is distinct from the successful reference’s T6 dispatcher and from native `18b588a048a3a3fe3d4bd60aa5678c66e284df31` plus our two patches.
His explicit uncached HIP+CXI atomic signal-buffer allocation is missing from native: the earlier CXI ordinary-allocation branch wins over the later generic HIP uncached branch, and our ownership patch changes stop/destruction, not allocation.
A2 also adds endpoint sharing/retries, posting/bookkeeping and optional mode/diagnostic changes not carried wholesale by native; some A2 modes were unsuccessful.
Conversely, native device-resident D2H handles and Torch-owned allocation/stream lifetimes must not be discarded when reconciling A2.
The atomic-allocation gap is a concrete review candidate, not an established explanation or fix for the host-side combine-launch crash.
A2 component results, teardown/hard-exit caveats, tuning differences and A1’s later correction of allocator-tail measurements are explicitly retained.

### 2026-11-03: existing candidate build/export continuation authorized

After requesting current build status, the user asked to continue.
Continue the existing candidate build/export workflow without implementing the proposed Turbo, reference Megatron, optimizer or additional UCCL changes.
No image inputs were edited and no automatic GPU job submission was added.

The app reached step 19/30, Megatron editable installation/finalization, after completing UCCL, Transformer Engine, Emerging-Optimizers, FLA and causal-conv1d installation steps.
The latest app guard snapshot showed about 408 GiB available memory and 286 GiB free graph storage.
A guarded export handoff now waits on app supervisor PID `58577` via pidfd, then requires its published successful completion status and rechecks all frozen build-input hashes before invoking the previously prepared Enroot export driver.
This waits on the supervisor rather than only `srun` PID `59253`, avoiding a final-status publication race.
The export uses an eight-CPU overlapping step in existing allocation `715971`, with no extra node request and the existing memory/storage guards.
It refuses duplicate status/output creation and will not export after a failed or ambiguous build.
The supervisor is `/tmp/opencode/apertus2-export-supervisor.py`; status is `/tmp/opencode/apertus2-export-handoff-status.json`; export log will be `/tmp/opencode/export-apertus2-megatron-rocm-native.log`.
Copies of the orchestration and completed build/export evidence are retained in benchmark `runs/image-build-candidate-001/`.
Five mocked GPU-free handoff checks passed: failed/ambiguous app refusal, duplicate-status preservation, build-input drift refusal and successful eight-CPU export orchestration; these are not actual export/GPU acceptance.
No source-port changes or full mid-4b claims are implied by completing this earlier candidate.

### 2026-11-03: candidate app built successfully; SquashFS export started

The existing candidate app build completed successfully in 2404 seconds without adding any of the newly reviewed Turbo/reference/optimizer/UCCL proposals.
Actual local Podman image ID is `1de9bd00e3bd6f8c3cee5d59b25024fc6ea3d04643517fe93461c213911425d3`, with local manifest digest `sha256:af5a66322d7e30b1a2bc344898793e937279833becd3ea7c8105995b532629ab`.
These are local store identities, not evidence that the image was pushed to JFrog.
The canonical app tag remains `alps7-dev-64f5ff51e3f53ac123c1a0cec5cfd6a5`.

The built image passed all three CPU regressions and image-local metadata/source/patch/BFD checks; 27 ELF libraries were inspected without CUDA dependencies.
Build log/status, full image inspection and `build-acceptance.json` are retained under benchmark `runs/image-build-candidate-001/`.
This establishes build/GPU-free acceptance only, not HIP numerical or training acceptance.

The pre-existing guarded handoff rechecked frozen build inputs and started the prepared `enroot import -x mount` export in an eight-CPU overlap step of existing allocation `715971` (srun PID `60452`), with no extra node request.
Export status/log are `/tmp/opencode/apertus2-export-handoff-status.json` and `/tmp/opencode/export-apertus2-megatron-rocm-native.log`.
The final SquashFS checksum awaits successful export; no GPU jobs are automatically submitted, and nothing has been pushed, promoted or committed.

### 2026-11-03: export artifact verified after Enroot cleanup failure; first EDF check prepared only

The requested `enroot import -x mount` wrote a complete SquashFS but returned nonzero during cleanup; preserve the original return code 1 and log rather than relabeling the command as clean success.
The host Enroot implementation removes the Podman container normally, then its EXIT trap hardcodes another `docker rm -f -v` under `set -e`.
The host `docker` command is Podman emulation and a read-only inspection confirms no such already-removed export container; a minimal errexit/failed-EXIT-trap reproduction returns 1.
No Enroot/Podman installation or image source was changed, no container was removed manually and the existing SquashFS was not overwritten/re-exported.

The artifact independently passed a full metadata/file-data read through `unsquashfs -pf -`, with output discarded rather than another 31 GiB filesystem extracted into HBM.
A full SHA-256 and before/after file identity check passed, and post-export Podman inspection confirms the candidate image ID did not change.
Artifact: benchmark `runs/image-build-candidate-001/apertus2-megatron-rocm-candidate.sqsh`, 31,528,255,488 bytes, SHA-256 `b4f5300c8ecbb69a442e44c0044007a09e56742422b4a867b34eac716f51ea9d`.
The original failed export status remains intact; separate `export-artifact-verification.json`, cleanup diagnostics and `export-record.json` document verified-artifact recovery without claiming a zero exporter exit.

Read 27 selected provenance/test/source files from the exported artifact, verified Megatron patch checksums and frozen test copies, and retained them with a checksum manifest in the build record.
Exported base pins confirm Torch `2.13.0+rocm10.0.0`, Triton `3.8.0+git4cff872c.rocm10.0.0`, NumPy `2.5.2`, torchvision `0.28.0+rocm10.0.0` and torchaudio `2.11.0.2+rocm10.0.0`.
This is image identity/integrity/provenance verification, not HIP numerical or training acceptance.

Prepared immutable EDF run benchmark `runs/stack-candidate-001/` against the exact artifact checksum, one node/four ranks/twenty minutes, `csstaff`/`mi300`, reservation `cscs`.
Verified its frozen-input fingerprints, checksum match, Bash syntax, ShellCheck and EDF hook/entrypoint settings.
No `submission.json` exists and no GPU job was submitted; reference/Turbo/optimizer/UCCL proposals remain unimplemented.

### 2026-11-03: candidate testing authorized; first EDF GPU check submitted

The user explicitly authorized testing the built/exported candidate.
Submitted frozen EDF run `stack-candidate-001` as job `716421`, one node/four GPU ranks/twenty minutes, `csstaff`/`mi300`, reservation `cscs`, explicit concurrent node budget eight and `sbatch --wait` completion recording.
A pre-submission queue check found only the existing one-node interactive allocation `715971`, so the first test brings requested/allocated agent resources to two nodes, below the agreed limit.
The image-local suite checks metadata/patch/BFD provenance and TE/router/KDA/convolution/Muon GPU numerics, with no router environment override and no run-local router repair.
This is candidate acceptance, not a full mid-4b reproduction or a claim that reference optimization paths are installed.

Prepared and verified the remaining checksum-pinned EDF runs without submitting them: `uccl-component-candidate-001` (two nodes/fifteen minutes), `train-hybrid-candidate-001` (two nodes/thirty minutes) and `train-mid4b-proxy-candidate-001` (two nodes/thirty minutes).
All three passed frozen-input, EDF, Bash syntax and ShellCheck validation; their image checksum matches the recovered verified artifact.
The UCCL run includes the exact CXI/WRITEDATA/CPU-timeout contract and paired four-peer configs.
Training runs retain the image’s native router behavior with no run-local fallback and use ordinary all-to-all.
Review the first job’s numerical and clean-exit evidence before proceeding, announce resources and recheck pending/running jobs before each subsequent submission, and record frozen cross-rank acceptance rather than trusting submission success.
No proposed Turbo/reference Megatron/optimizer/additional UCCL source changes were implemented for these tests.

### 2026-11-03: candidate image-local GPU stack passed; UCCL component regression submitted

Job `716421` completed in 3m22s on `nid002412` with Slurm exit `0:0` and no orchestration stderr.
Frozen cross-rank acceptance passed for all four ranks; versionable report is benchmark `results/stack-candidate-716421.json`.
Every rank ran TE, smoke/real-shape router, KDA, causal-convolution and Muon numerical checks and exited cleanly.
Actual component logs report `Router backend: aten (ROCm default)`; child provenance confirms native execution control and no `APERTUS_ROCM_ROUTER_GEMM_BACKEND` environment override.
This validates the built image default at component-test shapes without a run-local router repair, not full mid-4b capacity/throughput/convergence or hybrid restart.

After announcing two nodes/eight GPU ranks/fifteen minutes and checking that only the one-node interactive allocation remained in the queue, submitted `uccl-component-candidate-001` as job `716422`, `csstaff`/`mi300`, reservation `cscs`, `--wait --node-budget 8`.
Total requested/allocated agent resources are at most three nodes at this admission point.
This runs the installed image-local UCCL numerical/fused-gradient/lifetime regression, including the expanded-zero and masked-unit probability gradients and clean post-destroy RCCL checks; it is not a full-model UCCL retry.
Ordinary-all-to-all small/proxy restart runs remain frozen and unsubmitted pending result review.
No image/source changes or new reference optimizations were introduced.

### 2026-11-03: candidate UCCL component/gradient regression passed; small restart submitted

Frozen cross-rank acceptance for `uccl-component-candidate-001` / job `716422` passed on all eight ranks, with clean job completion and child/supervisor exits.
Actual hosts are `nid002412` and `nid002414`; every child records the required CXI/WRITEDATA/600-second CPU-timeout contract.
Each rank passed four exact fused-gradient combinations (sync/async crossed with expanded-zero/masked-unit probability gradients), including the deliberately incoming stride `[0,0]`/four-byte backing-storage edge case without a repairing hook.
Image-local UCCL dispatch/combine numerics, tensor lifetime release and fresh post-destroy RCCL work passed.
Versionable evidence is benchmark `results/uccl-component-candidate-716422.json`.
This validates the installed gradient hardening and component/lifetime behavior; it does not establish full-model UCCL readiness or resolve the prior first-forward-combine host launch crash.

After announcing two nodes/eight ranks/thirty minutes and checking the queue/resource budget, submitted ordinary-all-to-all `train-hybrid-candidate-001` as job `716423`, `csstaff`/`mi300`, reservation `cscs`, `--wait --node-budget 8`.
The preceding UCCL job was completing cleanup at admission; even conservatively counting both tests plus the interactive node remains below eight nodes.
This small synthetic KDA/MoE/MuonMD run must save at iteration 2, load in a fresh process and complete updates through iteration 4 with zero skip/NaN counters.
It uses the candidate router default without a run-local override; the subsequent real-shape proxy remains prepared and unsubmitted.
No source changes or broad probe matrix expansion were made.

### 2026-11-03: candidate small hybrid restart passed; real-shape proxy submitted

Frozen cross-rank acceptance passed for job `716423` / `train-hybrid-candidate-001` on all eight ranks.
Both actual training children per rank exited zero; checkpoints at iterations 2 and 4 and fresh-process resume from 2 passed the existing optimizer/model/scheduler/RNG-sidecar acceptance.
Reported losses for updates 1–4 were 7.555790, 7.534986, 7.392283 and 7.323160, with finite gradient norms and zero skip/NaN counters.
All sixteen child provenances record native execution controls and no router backend environment override.
Versionable evidence: benchmark `results/train-hybrid-candidate-716423.json`.

Four stage-log occurrences warn that `destroy_process_group()` was not explicitly called before exit.
Preserve this caveat despite zero child/job exits; do not equate process completion with demonstrated explicit communicator shutdown.
Companion notes are `results/train-hybrid-candidate-716423-notes.json`; no cleanup code was changed for this test.

After the announced two-node/eight-rank/thirty-minute resource request and queue check, submitted `train-mid4b-proxy-candidate-001` as job `716427`, `csstaff`/`mi300`, reservation `cscs`, `--wait --node-budget 8`.
This is the four-layer/512-token synthetic proxy retaining reference projection/expert dimensions, ordinary all-to-all, and the image router default without a run-local repair.
Require the same checkpoint/fresh-process resume acceptance through iteration 4; this remains distinct from the full 32-layer/8192-token mid-4b recipe or its optimized reference execution.
No image or proposed reference/Turbo/optimizer/UCCL implementation changes were made.

### 2026-11-03: candidate real-shape restart passed; four-check acceptance complete

Job `716427` / `train-mid4b-proxy-candidate-001` completed in 6m47s on `nid002412` and `nid002414`, Slurm exit `0:0`.
Frozen cross-rank acceptance passed on all eight ranks: updates 1–2, checkpoint at 2, fresh-process resume from 2, updates 3–4 and checkpoint at 4, with zero skipped/NaN iterations and finite gradients.
Reported losses were 7.551331, 7.040591, 5.528292 and 5.057707; gradient norms were 4.079, 4.100, 3.969 and 3.899.
All sixteen training children exited zero and recorded native execution controls with no `APERTUS_ROCM_ROUTER_GEMM_BACKEND` environment override.
This accepts the candidate image router default in real projection/expert dimensions without the prior run-local ATen repair.

Versionable report: benchmark `results/train-mid4b-proxy-candidate-716427.json`.
Four explicit process-group-shutdown warnings are preserved in `results/train-mid4b-proxy-candidate-716427-notes.json`; zero child/job exits do not prove explicit communicator destruction.
The model remains four layers, sequence 512 and vocabulary 1024 with the conservative functionality-test schedule, not the full mid-4b recipe or reference optimization paths.

All four planned candidate checks have now passed against the same verified SquashFS checksum: image-local GPU stack `716421`, UCCL numerical/fused-gradient/lifetime component `716422`, small all-to-all hybrid restart `716423`, and real-shape all-to-all proxy restart `716427`.
Aggregate evidence is retained in benchmark `runs/image-build-candidate-001/candidate-acceptance.json` and versionable `results/candidate-image-64f5ff51-acceptance.json`.
The aggregate retains Enroot cleanup recovery, training shutdown warnings, full-model UCCL failure boundaries and full mid-4b non-equivalence explicitly.

This concludes the authorized four-check candidate acceptance sequence; no further jobs are automatically submitted.
No reference Megatron/Turbo/optimizer/additional UCCL proposals were implemented, and no images or changes were pushed, promoted or committed.
Full mid-4b reproduction remains the objective, requiring explicit recipe/default reconciliation, selected reference implementation review and a separately agreed resource/implementation step; do not call this proxy sequence the reproduction itself.

### 2026-11-03: full mid-4b recipe defaults and harness blockers reconciled without implementation

After the four candidate acceptance checks completed, continued with analysis/planning for the full mid-4b target while retaining the user’s source-change pause.
`MID4B-REPRODUCTION-PLAN.md` and benchmark `reference/mid4b-reproduction-plan-2026-11-03/` now capture actual parsed reference/candidate arguments, source/log fingerprints and a scoped implementation/resource plan.
Confirmed semantic mismatches include reference/candidate router scaling 2.5/None, Adam beta2 0.95/0.999, initialization std 0.0197642/0.02 and minimum-LR mode absolute/relative.
Adam beta1 0.9 and rotary base 10000 already match; omitted flags must not be counted automatically as semantic differences.

The existing geometry validator already accepts 32 layers/8192 tokens/vocabulary 200063; the blockers are hardcoded PP1, smoke learning rates/four iterations and the fixed restart/summary policy.
The native checkpoint source uses DP-rank optimizer filenames inside PP/expert-specific directories, so the current global-rank/basename-only inspector cannot simply be reused at PP2.
No model/harness/source implementation was changed to bypass these limitations.

For PP2/EP8/TP1/CP1/ETP1, four/eight/sixteen nodes give DP8/16/32, expert-DP1/2/4 and 144/72/36 microbatches per GBS1152 iteration.
Four nodes is the minimum matching PP2/EP8 rank layout but has the EDP1 memory-observation caveat; eight is reduced scale, sixteen is the reference scale.
All exceed standing one/two-node job approval, and sixteen exceeds the eight-node budget; an eight-node run must also account for the existing interactive allocation without unauthorized cancellation.
A tiny two-node PP2/EP2 restart could separately validate future harness changes within the existing short-job resource agreement, but has not been prepared/submitted.

Recommended next implementation scope is explicit recipe/PP2/checkpoint support first, with Turbo/reference kernel/optimizer/UCCL image ports and larger jobs separately gated.
Matching the native training recipe alone would still be a native baseline rather than reproduction of selected Turbo/T6/AITER/fusion/workspace paths.
The implementation pause needs explicit resolution before proceeding beyond this analysis; no further GPU jobs were submitted.

### 2026-11-03: recipe/harness scope resumed; 16-node authorization and in-memory warmup requirements

This entry supersedes earlier implementation/resource-pause wording without changing frozen experiments or historical evidence manifests.
The user explicitly selected recipe/harness-only implementation, then authorized autonomous continuation including an eventual 16-node test.
The user also requested the node-local in-memory warmup/cache gather/merge/join/reconciliation pattern from `boeschf/alps-vllm-benchmarks`.
Turbo/AITER installation and Megatron/optimizer/UCCL image-source ports remain outside the approved scope.

Implemented explicit recipe/model/train configurations, PP2-aware DP/EP/EDP and batch validation, configurable iteration/restart windows, native MPU/checkpoint-coordinate capture and global sidecar ownership checks.
The full reference recipe explicitly records router scale 2.5, Adam beta2 0.95, initialization std 0.0197642, minimum-LR mode absolute, the reference schedule, timeout 20 minutes and disabled evaluation with interval 100000.
Duplicate smoke flags in the partial builder were caught and removed; all 90 GPU-free harness tests now pass.
Generic/TE cross-entropy recipe fusion choices are still GPU-unvalidated; matching hyperparameters on native TE/all-to-all remains a native baseline, not optimized Turbo/T6 execution.

Tiny PP2/EP2 checkpoint/fresh-process restart job `716590` uses two nodes/eight ranks, the verified candidate SquashFS, account `csstaff`, partition `mi300` and reservation `cscs`.
Only the existing one-node interactive allocation was present at admission, and it was not cancelled or modified.
It passed all eight ranks in 8m57s, including updates 1–4, checkpoint/fresh-process resume from 2 and full PP/EP-qualified optimizer ownership.
Benchmark `results/train-pp2-ep2-candidate-716590.json` and companion notes retain four process-group cleanup warnings despite zero child/supervisor/job exits.
No full 16-node training job has been submitted.

Implemented an opt-in compiled-Triton-only cache backend and training orchestration, using upstream cache workflow revision `6a0a68d511a8a5728ca8667e3b22ebf98e9e9181` as a design reference.
All live compilation/archive/merge work uses verified executable tmpfs; every rank captures only after child exit, gathers immutable bundles and hydrates the same reconciled generation into its own mutable cache.
Warmup children have separate scratch/caches and no checkpoint/save/load flags; measured model/optimizer/RNG state starts fresh.
Native Triton `4cff872c` group records contain absolute rank-local paths and are canonicalized/relocated rather than copied blindly.
Archives reject traversal, links, duplicate/nonregular/oversized entries and unsafe artifact types; rank-local autotune state, locks and arbitrary extension/build trees are not persisted.
Whole-object conflicts stop the warmed run; persistent three-way conflicts preserve CURRENT, and successful generations are immutable with atomic pointer updates.
Compatibility includes immutable image/source/DSO identity, package versions, Python/HIP/Torch ABI, GPU architecture, kernel/machine, frozen harness source and workload configuration.
The default two-GiB snapshot limit per rank, polling runtime-size guard and operation-specific free-space checks are not hard HBM quotas.
Cold/cache-hydrated/disposable-warmup measurements are explicitly distinguished, and other JIT/autotuning can still run.

Next gates are tiny PP2 restart acceptance, distributed warmup/cache validation, native fusion validation and full-scale capacity, before assessing cold/warmed 16-node timing.
Do not ask again for the already-granted 16-node resource authorization or cancel the existing interactive builder allocation.

## Sources

- [Alps Extended Images usage](https://docs.cscs.ch/software/alps-extended-images/#usage).
- [GLM benchmark harness](https://github.com/boeschf/alps-vllm-benchmarks), reviewed at `6a0a68d511a8a5728ca8667e3b22ebf98e9e9181`.
- [AMD MoE best practices](https://rocm.blogs.amd.com/software-tools-optimization/primus-moe-package/README.html).
- [Primus components](https://www.amd.com/en/products/software/rocm/primus.html#components).
- User-provided Beverin DTensor/hipBLASLt and NUMA-fragmentation tickets.



### 2026-11-03: independent warmup and explicit-recipe tiny GPU checks

Following tiny PP2 restart acceptance, job `716596` tests the new disposable-warmup/cache protocol on two nodes/eight ranks with the legacy conservative recipe.
The independent two-node reference-recipe/fusion check `716597` failed before training: `auto` attention inherited smoke `NVTE_FLASH_ATTN=0` and `NVTE_UNFUSED_ATTN=0`, contradicting native Megatron’s all-one auto-selection contract.
The corrected reference experiment explicitly enables all three attention-selection flags, with a new pre-submission consistency test; the full CPU suite now passes 90 tests.
The failed run remains frozen and its separate failure report records that no training/fusion/capacity/performance result was obtained.
A new two-node retry tests six updates with checkpoint/fresh-process resume from 3; both this retry and the warmup test are pending.
Together with the unchanged interactive node, concurrent requests remain five nodes, below the existing short-job allowance.
No image changes or full 16-node submission occurred.


### 2026-11-03: reference fusion boundary and real-cache capture hardening

The corrected auto-attention check `716598` reached model execution but failed on first-update Apex `fused_weight_gradient_mlp_cuda` JIT compilation: unqualified `max` is undeclared in both hipified dense-gradient source variants.
This is an image/source compatibility limitation, not a cache lock issue; no Apex/Megatron/image patch was made within the harness-only scope.
Added independent `gradient_accumulation_fusion` recipe control and the named `native-no-gradient-fusion.json` deviation; other reference fusions/hyperparameters remain selected for the separate six-update/restart-from-3 test `716600`.
Auto attention can select image-native AITER FMHA through TE; that is not the reference’s missing direct Megatron Turbo/T6 integration or a new AITER installation.

Warmup job `716596` completed all eight warmup children and quiescent bundle gathering, but strict merge refused a same-key `hip_utils.cpython-312-x86_64-linux-gnu.so` whose bytes embed different rank-local temporary C source paths.
The retained manifests then exposed a second capture bug: required `.source` compiler artifacts were omitted, making every GPU group incomplete and skipped.
Agent-owned retry `716601` was explicitly cancelled rather than risk accepting a meaningless empty warmed generation; no user allocation or independent recipe job was cancelled.
The corrected policy preserves native `.source` artifacts, excludes host `.so` helpers, and rejects empty/no-GPU or incomplete warmup snapshots while retaining strict whole-object GPU conflict checks.
Fresh two-node cache retry `716602` is pending; independent recipe test `716600` is also pending.
All 93 CPU tests pass, including native `.source` preservation, empty/incomplete warmup rejection, deadline-bounded locks and persistent three-way preservation of unrelated concurrent additions.

Keep the failure, cancellation and GPU-union-analysis records in benchmark `results/`; they are not successful warmed measurements.
Next compare successful warmed tiny-model losses/gradients with cold job `716590` and its own warmup window, then validate persistent hydration before a full-scale baseline.
The already-authorized 16-node run remains technically gated, not awaiting renewed resource permission.


### 2026-11-03: named native recipe passes six-update restart acceptance

Job `716600` completed successfully in 8m49s on two nodes/eight ranks with tiny geometry, PP2/EP2 and the named `mid4b-native-no-gradient-fusion` recipe.
The frozen all-rank summary passed updates 1–6, checkpoint targets 3/6, explicit fresh-process resume from 3 and native PP/EP optimizer ownership.
Reported losses were `7.594733`, `7.601528`, `7.560375`, `7.533004`, `7.525237`, `7.523116`, with finite gradients and zero skipped/NaN counters.
Reference hyperparameters, auto attention and the other selected fusion flags passed at this tiny geometry; Apex gradient fusion remains explicitly disabled, not fixed.
Benchmark `results/train-reference-tiny-candidate-716600.json` and companion notes preserve the recipe, actual-child provenance and four process-group cleanup warnings despite zero exits.
This is neither full mid-4b capacity/performance nor exact reference-kernel/convergence acceptance.

The separate corrected cache job `716602` remains pending.
After it completes, validate nonempty GPU-object reconciliation, cold-versus-warmed tiny numerics/state isolation and persistent cross-job hydration before advancing to the authorized 16-node baseline.
The reference and native pinned training logger intentionally do not reset the first-update interval, so the second printed timing is a mean over the first two updates; from the third update in each fresh process it is a normal interval at log interval one.
Preserve raw logged timings and do not count either of the first two logs after initialization/resume as independent steady-state samples.


### 2026-11-03: GPU-cache lifecycle accepted; exact tiny state check and full-shape capacity preparation

Warmup/cache/restart `716602` passed all eight ranks: each measured cache hydrated 280 GPU artifacts across 2240 files from generation `cdbe163fa27d909d273afa7fe84e0d569ee476899ba8223a4d8622d7be9c4556`, then completed updates 1–4 and checkpoint/fresh-process resume from 2.
The persistent and per-run generation matched.
Independent read-only job `716605` used the identical frozen source/workload/image and no new warmup, hydrated the same persistent generation on all ranks and passed training/restart acceptance without publication.
Benchmark result reports and companion notes preserve six/four process-group cleanup warnings respectively despite zero child/supervisor/job exits.

Four-update losses differed from cold job `716590` by at most about 0.00122 (warmup-conditioned) and 0.00150 (read-only hydrated).
These close BF16 values are not bitwise model/optimizer/RNG state proof.
Added opt-in bounded tiny-model initialization fingerprints at the actual native train-call boundary, comparing model/optimizer tensors and Python/NumPy/Torch/Megatron RNG state before warmup versus measured updates.
The observer uses temporary CPU copies/views, no checkpoint or RNG advancement, a 256-MiB combined tensor-byte guard and restoration of the native function; it is disabled for full-size/performance runs.
The measured scheduler is constructed fresh from the original recipe; disposable warmup duration fields are not required to match.
All 98 CPU tests pass.
Two-node job `716607` validates this exact-state check with the named no-gradient-fusion native recipe, warmup and six-update/restart-from-3 protocol; result pending.

Prepared, but did not submit, `runs/train-mid4b-capacity-native-001` with full mid-4b dimensions, 16 nodes/64 ranks, TP1/PP2/CP1/EP8/ETP1, DP32/EDP4, GBS1152/MBS1 and 36 microbatches per iteration.
It requests two disposable warmup updates, four measured continuous updates, a final checkpoint at 4 and 45 minutes with a 40-minute supervisor cutoff.
Fingerprinting is off and the gradient-fusion deviation remains explicit.
The short full-shape capacity/kernel gate comes before committing to the full 26-update native baseline and does not establish reference-optimized execution or convergence.
Shared-filesystem free space is ample at the filesystem level, but that observation is not a user-quota guarantee; full model/optimizer checkpoints can require hundreds of GiB and must remain under the run directory.
After exact-state acceptance, use the already-granted 16-node authorization with sequential admission, account for the preserved interactive node and do not request the same resource permission again.


### 2026-11-03: exact initial-state acceptance complete; authorized 16-node capacity check submitted

Job `716607` passed all eight ranks with exact model/optimizer/RNG initialization equality between disposable warmup and measured children, then six measured updates with checkpoint/fresh-process resume from 3.
The exact comparisons covered model/buffer tensors, optimizer state and Python/NumPy/Torch/Megatron RNG at the real native train-call boundary, with iteration and consumed samples both zero before measured update 1.
Measured and warmup schedulers shared the explicit LR schedule and zero steps; only disposable duration metadata differed for constant zero weight decay.
Benchmark result, companion notes and the separate initial-state proof preserve this acceptance and six process-group cleanup warnings despite zero exits.
Later BF16/autotuned operations are not claimed bitwise identical.

Submitted the first full-geometry capacity check as **job `716613`**, using the already-granted sixteen-node authorization without requesting it again.
It requests sixteen exclusive MI300A nodes/64 ranks on `csstaff`/`mi300`, reservation `cscs`, up to 45 minutes, with a 40-minute per-probe cutoff and sequential admission after all small harness jobs completed.
The user-held interactive allocation `715971` remains untouched; that node plus the explicitly authorized full job requests up to seventeen nodes total.
The full target is 32 layers, sequence 8192, vocabulary 200063, TP1/PP2/CP1/EP8/ETP1, DP32/EDP4, GBS1152/MBS1 and 36 microbatches per update.
Two disposable full-shape warmup updates precede quiescent all-rank capture/gather/reconciliation/private hydration, four measured continuous updates and one final checkpoint at 4.
The exact-state fingerprint observer is off for this full-size/performance path; gradient fusion remains explicitly disabled after the recorded Apex compilation failure.
No image/Apex/Megatron/Turbo/UCCL source was changed, and the existing verified candidate SquashFS is unchanged.

This is a short full-shape capacity/kernel/checkpoint gate before the longer 26-update native baseline, not optimized reference reproduction or convergence evidence.
The result is pending; benchmark `results/train-mid4b-capacity-native-716613-admission.json` records resources, prerequisites, preserved allocation and limitations.
Next inspect all-rank startup/memory/warmup evidence, then require a zero full job exit and the frozen global acceptance summary before making any full-shape claim.


### 2026-11-03: full-shape grouped-GEMM failure isolated for smaller backend controls

Job `716613` failed after 4m36s during first-warmup backward: TE `GroupedLinear.backward` called `general_grouped_gemm` / `te_general_grouped_gemm` and reported `HIPBLASLT Error: 6` at the hipBLASLt matmul call.
Nineteen rank logs contain this direct error, followed by coordinated termination of other ranks; exit 137 is not itself an OOM diagnosis.
All 64 ranks across sixteen nodes passed executable-tmpfs startup, but there was no completed warmup/measured update, no reconciled full-shape cache and no full-size checkpoint acceptance.
The immutable failure report is benchmark `results/train-mid4b-capacity-native-716613-failure.json`, with pinned read-only TE source evidence under `reference/full-shape-grouped-gemm-2026-11-03/`.

Pinned TE already supports `NVTE_USE_GROUPED_GEMM_TRITON=1` for HIP/BF16 when gradient accumulation fusion is disabled.
No historical rocBLAS environment switch is assumed to work, and no image/global BLAS policy was changed.
Added a sequence-8192, four-layer/vocabulary-1024 proxy preserving mid-4b expert/attention dimensions, 32 local experts at EP8 and mean 2048 routed tokens per expert.
Two separate two-node/eight-rank jobs use identical frozen source, image and workload, differing only in that grouped-backend flag: native `716678` and Triton `716676`.
Both request `csstaff`/`mi300`, reservation `cscs`, thirty minutes with a twenty-five-minute supervisor cutoff, PP1/EP8, GBS8/MBS1 and updates 1–4/checkpoint-resume from 2.
These are cold regression controls, not warmed performance measurements or PP2/full-depth/full-vocabulary capacity tests.
The user-held interactive allocation `715971` remains untouched; total requested nodes are five.

The harness now rejects grouped Triton with gradient fusion enabled (the native backend would otherwise not be selected) and durably copies child provenance on failed warmup as well as success.
The current CPU suite passes 100 tests, including grouped-backend configuration guards, long-proxy geometry and failure-provenance retention.
The colleague-facing appendable summary is saved at `COLLEAGUE-UPDATES.md`; the original summary is retained and subsequent findings are appended.
Both backend controls are pending; require their results before another large submission.


### 2026-11-03: two-node reproduction, isolated matrix control and vLLM min/max precedent

Native long-sequence control `716678` failed first backward in TE grouped wgrad on all eight ranks; grouped-Triton control `716676` failed in non-grouped TE linear wgrad on all eight ranks.
Both reported hipBLASLt status 6 without any completed update, so sixteen nodes/full PP2 are not necessary to reproduce the problem and grouped Triton is not an end-to-end fix.
Retained failure provenance showed at least 111.55/109.11 GiB device free for native/Triton controls, with maximum Torch allocator peaks about 12.79/14.96 GiB; these are not complete CPU/library/tmpfs peak accounting, but do not support a simple proxy device-OOM diagnosis.
Separate failure reports retain each rank's environment and memory counters.

One-node job `716827` passed all four direct TE NT wgrad numerical controls against ATen FP32: 1280 input / 2880 output features, reduction dimension 128 versus 2049, BF16 versus FP32 output, and requested heuristic selection zero versus one.
These standalone shapes do not reproduce the training failure; neither dtype nor algorithm selection is established as its cause or workaround.
A bounded two-node grouped-Triton replay with `NVTE_LOG_GEMM_CONFIG=1` is submitted to capture the failing non-grouped call's actual configuration.

Following the user's vLLM ROCm pointer, confirmed the existing UCCL device min/max overload patch and the base rccl-tests forced-include workaround for the same missing-global-overload assumption.
Read-only inspection of installed Apex 1.13.0+rocm10.0.0 found five host workspace-size `max` calls in two source variants, independent of UCCL headers.
Copied sources with algorithm included and `std::max<uint64_t>` qualified passed both HIP translation units' host/gfx942 syntax checks in executable tmpfs.
No installed source, binary or image was changed; the evidence and unapplied proposed patch are in benchmark `reference/apex-minmax-compat-2026-11-03/`.
This is syntax-only validation, not a linked/JIT-loaded/numerically validated gradient-fusion fix and not resolution of hipBLASLt status 6.
Gradient fusion remains explicitly disabled pending separately scoped integration/runtime validation.
All 102 CPU tests pass; the Confluence-ready colleague update has these results appended.


### 2026-11-03: logged NT configuration captured for exact matrix replay

Two-node logged replay `716828` failed first backward on all eight ranks, with each last printed GEMM configuration `m=2560 k=8192 n=1280 transa=N transb=T A_type=6 B_type=6 D_type=6 grad=1`, no bias/GELU/FP8, alpha one and beta zero.
This identifies a latent-projection-size non-grouped NT weight-gradient call, not the earlier standalone 1280/2880 geometry that passed.
The failure report is benchmark `results/train-long-proxy-te-triton-log-716828-failure.json`.
The one-node matrix probe now supports an explicitly bounded captured-geometry mode: input 2560/output 1280 features, reduction 512 versus 8192, BF16 versus FP32 output and requested heuristic zero versus one.
The current CPU suite passes 103 tests, including exact captured-geometry construction; the one-node replay is submitted with a fifteen-minute allocation and ten-minute supervisor cutoff.
No model/optimizer/RCCL or source/image patch is part of the replay, and no dtype/heuristic workaround is claimed before its result.


### 2026-11-03: exact matrix failure reproduced and alternate heuristic promoted to training control

One-node captured-geometry job `716844` completed all four diagnostic cases before reporting its expected failed case.
For input/output features 2560/1280, NT BF16 wgrad passed at reduction 512 but failed with hipBLASLt status 6 at reduction 8192 under requested/default heuristic zero.
The same long geometry passed with FP32 output (relative L2 about 1.08e-6) and with BF16 output/requested heuristic one (relative L2 about 0.00166).
This reproduces the problem without model/optimizer/RCCL and supports a geometry/heuristic-specific backend failure; it is not blanket BF16 incompatibility, an OOM result or a full-model workaround yet.
Detailed diagnostic outcomes are in benchmark `results/te-wgrad-captured-716844-findings.json`.

Added a narrowly allowlisted and provenance-captured `TE_HIPBLASLT_ALGO_SELECTION` runtime setting restricted to reviewed values zero/one.
The new two-node/eight-rank `train-long-proxy-te-algo1-001` control retains native grouped GEMM, sequence 8192 and the named no-gradient-fusion recipe, enabling only heuristic one and configuration logging.
It tests four updates with checkpoint/fresh-process resume from 2 under a thirty-minute allocation/twenty-five-minute cutoff, `csstaff`/`mi300`, reservation `cscs`.
No image/default BLAS policy or installed source was changed; the current CPU suite passes 105 tests.
The training result remains pending, and no further sixteen-node retry is justified solely by the isolated matrix pass.


### 2026-11-03: heuristic-one training/restart accepted; full-shape retry submitted

#### Two-node acceptance

- Native grouped-GEMM/heuristic-one job `716847` completed with full-job exit zero.
- The frozen global summary accepted all eight ranks, updates 1–4, checkpoints at 2 and 4, and a fresh-process resume from update 2.
- Both actual child processes on every rank recorded `TE_HIPBLASLT_ALGO_SELECTION=1` and `NVTE_USE_GROUPED_GEMM_TRITON=0`.
- Logged losses were 7.511195, 7.513458, 7.471888 and 7.425287; gradient norms were finite and reported skip/NaN counters were zero.
- Checkpoint model/optimizer ownership was validated globally using the native checkpoint coordinates.
- This is a four-layer, sequence-8192, vocabulary-1024 PP1/EP8 proxy, not full-depth PP2/vocabulary-200063 capacity or throughput evidence.
- Cold startup and the fresh-process restart are not a steady-state timing window; the second logged timer in each process includes its first-update timer.
- Process-group cleanup warnings remain a separate caveat; this result does not claim teardown is fixed.

Evidence: benchmark `results/train-long-proxy-te-algo1-716847.json` and `results/train-long-proxy-te-algo1-716847-notes.json`.

#### Full-shape capacity retry

- Submitted `train-mid4b-capacity-native-002` as job `716850` under the existing sixteen-node authorization.
- Resources: 16 exclusive nodes / 64 ranks, `csstaff` / `mi300`, reservation `cscs`, maximum 45 minutes.
- Geometry: 32 layers, sequence 8192, vocabulary 200063, PP2/EP8/DP32/EDP4, GBS1152/MBS1 and 36 microbatches per update.
- Protocol: two disposable warmup updates, tmpfs cache capture/gather/reconciliation, four measured continuous updates and checkpoint at 4.
- The retry retains native grouped GEMM, heuristic one and disabled gradient fusion; detailed GEMM tracing is off to bound log growth.
- Submission used sequential admission after verifying no unfinished harness submission and only interactive allocation `715971` remaining active.
- That interactive allocation and its memory-backed Podman builder store remain untouched.
- Full-shape success still requires a zero full-job exit and frozen all-rank global summary; no update, capacity or performance acceptance is claimed at submission.
- No image/source rebuild or global BLAS-policy change was made.


### 2026-11-03: reservation bottleneck and unreserved replacement

- Job `716850` could not start: Slurm reported `ReqNodeNotAvail, May be reserved for other job` with no predicted start time.
- Reservation `cscs` contains 18 nodes, but `nid002412`, `nid002416`, `nid002418` and `nid002424` are drained with reason `ATOM task mce_test exited 1`.
- Only 14 non-drained nodes remain inside the reservation, below the requested 16; idle nodes elsewhere cannot satisfy a reservation-constrained job.
- The user explicitly authorized running outside the reservation.
- Cancelled only our pending job `716850`; accounting confirms zero elapsed time and no nodes assigned, so this was a scheduling cancellation rather than a training failure.
- Prepared `train-mid4b-capacity-native-003` with reservation unset.
- Resolved configuration comparison verifies reservation is the only changed setting; image identity and every frozen source hash are identical to the prior run.
- The replacement retains 16 nodes/64 ranks, `csstaff`/`mi300`, a 45-minute cap and the same full-shape warmup/four-update/checkpoint protocol.
- Interactive job `715971`, its builder store, and all prior frozen run inputs/evidence are preserved.

Replacement job **716859 is RUNNING** on sixteen nodes outside any reservation; training/capacity acceptance remains pending.


### 2026-11-03: full-shape cold warmup activity, not yet acceptance

- Job `716859` has 64 live ranks across sixteen unreserved nodes, with executable-tmpfs startup verified and cache misses on every rank.
- At approximately ten minutes, the first warmup update had not yet completed; no retained HIPBLASLt/OOM/traceback error was present at that observation.
- Sampled final-stage logs showed successful attention backward JIT compilation.
- A read-only overlapping step on the existing first node found four live training children at approximately 282–285% lifetime CPU, nonzero but heterogeneous GPU utilization, approximately 73.6–75.4 GiB reported VRAM used per GPU, and approximately 130.1 GiB host MemAvailable.
- These are one-node point-in-time samples, not proof of iteration progress or all-node/peak memory fit.
- Host and GPU figures share MI300A HBM and must not be added as independent capacity.
- AccumulateGrad stream-mismatch and collective-deprecation warnings are retained; neither is being treated as a fatal error or silently declared fixed.
- The user requested continued autonomous monitoring; milestone/error monitoring is active without changing the running configuration or submitting another GPU job.

Evidence: benchmark `results/train-mid4b-capacity-native-003-startup-observation.json`, `results/train-mid4b-capacity-native-003-activity-observation.json` and run-local `activity-first-warmup.json`.


### 2026-11-03: slow first update and smaller diagnostic prepared

- At the approximately twenty-minute observation, job `716859` still had 64 live ranks and no completed warmup update or cache-reconciliation/checkpoint milestone.
- Retained logs had no fatal exception, but live processes and increasing CPU counters do not establish useful iteration progress.
- Bounded debugger attachment on ranks 0 and 63 failed with `ptrace: Inappropriate ioctl for device`; no Python/native stack was obtained and no diagnostic signal was sent to training.
- The current run remains unchanged under its existing 2400-second probe cutoff / 45-minute allocation ceiling.
- Prepared, but did not submit, two-node/eight-rank `train-long-full-vocab-stack-001` as a smaller conditional follow-up.
- That diagnostic retains sequence 8192 and real vocabulary 200063 at reduced four-layer depth, PP1/EP8/EDP1, one microbatch per update, native grouped GEMM and heuristic one.
- It uses the existing opt-in Python stack dump every 60 seconds, two continuous updates/checkpoint at 2, quiet GEMM logging, a 25-minute probe cutoff and 30-minute allocation cap, with no reservation.
- This separates the previously untested full-vocabulary path from full depth/PP2/36-microbatch scaling; it is not a substitute for full-shape capacity or throughput acceptance.
- No additional GPU job will be launched while the sixteen-node job is active; another full-scale repeat is not the fallback for an undiagnosed timeout.
- The GPU-free harness suite now passes 106 tests, including construction/stack-watchdog isolation for the prepared diagnostic.
- The candidate image and installed sources remain unchanged; all frozen runs and the interactive allocation are preserved.

Evidence: current run `activity-stack-discovery-r0000.json`, `activity-stack-discovery-r0063.json`, `activity-native-stack-r0000.json` and `activity-native-stack-r0063.json`; benchmark `results/train-long-full-vocab-stack-001-readiness.json`.


### 2026-11-03: full-shape observation near the diagnostic cutoff

At approximately 34 minutes, all 64 ranks were still live but no warmup update, reconciliation or checkpoint had completed.
Endpoint samples showed nonzero CPU and GPU activity, including 100% reported utilization on several GPUs, but this cannot distinguish productive compute from communication spin/wait.
Cross-process maps/environment inspection was permission denied as well as debugger attachment; no actual stack, exact slow phase or unobserved child environment is being claimed.
The run remains subject to its original 2400-second probe cutoff and 45-minute allocation cap, with no extension or concurrent GPU job.
A frozen failure/timeout report will be retained before proceeding to the prepared two-node/full-vocabulary/Python-stack diagnostic.
Evidence: benchmark `results/train-mid4b-capacity-native-003-slow-warmup-observation.json` and run-local `activity-before-cutoff-r0000.json` / `activity-before-cutoff-r0063.json`.


### 2026-11-03: full-shape failure is a collective stall; targeted EDP2 control submitted

#### Full-shape outcome

- Unreserved job `716859` failed after 36m12s with full-job exit 1, before the 40-minute supervisor cutoff.
- No warmup update, measured update, cache reconciliation or checkpoint completed.
- The terminal evidence is a 20-minute distributed-collective watchdog/store timeout, not ordinary JIT startup or an accepted memory/capacity result.
- Ranks 0, 8 and 16 timed out a coalesced expert-data-parallel gradient synchronization at sequence 11, with last enqueued sequence 66 and last completed sequence 10.
- The other 29 PP0 ranks, including rank 24, were in `_log_global_router_metrics`, waiting to bootstrap the tensor-and-data-parallel communicator through TCPStore key `/default_pg/0//265//cuda//0`.
- All 32 PP1 ranks timed out a one-element default-group all-reduce in the rerun-state machine.
- These complementary waits narrow the failure to post-backward synchronization/communicator ordering or progress, but do not yet establish the cause or prove a transport defect.
- High observed CPU/GPU utilization was therefore not evidence of useful iteration progress.
- No `HIPBLASLT Error: 6` was retained in this retry, but that is not blanket acceptance of heuristic one for the full model.
- Slurm terminated the remaining ranks after task failure; resulting SIGTERM/cleanup exceptions are secondary evidence, not separate root causes.

Evidence: benchmark `results/train-mid4b-capacity-native-716859-failure.json` and `results/train-mid4b-capacity-native-716859-collective-divergence.json` retain final statuses, accounting, watchdog stacks and per-rank classification.

#### Smaller next test selected from the failure

- The previously prepared full-vocabulary EP8 proxy has EDP=1 and cannot exercise an expert-data-parallel collective, so it remains unsubmitted.
- Submitted `train-long-ep4-edp2-stack-001` as job `716927`: two nodes/eight ranks, PP1/EP4/DP8/EDP2, sequence 8192, four layers and vocabulary 1024.
- Each rank has 64 local experts rather than the full run's 32; the reduced topology is a targeted diagnostic, not full-shape acceptance.
- It retains native grouped GEMM, heuristic one, disabled gradient fusion and native overlapping gradient reduction.
- Protocol: two continuous updates, checkpoint at 2, periodic Python stack dumps every 60 seconds, quiet per-GEMM logging, cold private tmpfs caches, a 25-minute probe cutoff and a 30-minute allocation cap.
- Uses `csstaff`/`mi300` without a reservation and sequential admission after the sixteen-node job completed.
- Only the preserved interactive allocation was active at admission; no overlapping GPU training job or image/source change was introduced.
- The GPU-free suite passes 107 tests, including explicit EDP2 topology and diagnostic-watchdog construction.
- Full mid-4b capacity, full checkpoint acceptance, the 26-update baseline and throughput comparison remain blocked.


### 2026-11-03: EDP2 diagnostic hit an early native signal; single-setting timer control submitted

- Two-node EDP2 job `716927` failed before update 1: training children on ranks 0 and 5 returned `-11` (SIGSEGV), and peers were terminated after task failure.
- Both affected logs end partway through the first sixty-second Python stack dump, in Triton/FLA compilation or an interpreter trampoline.
- Other ranks produced complete timer-triggered stacks inside first-forward Triton autotuning.
- This is not yet reproduction of the full-shape expert-DP collective stall; an instrumentation-related native crash is a hypothesis, not an established cause.
- Retained failure record: benchmark `results/train-long-ep4-edp2-stack-716927-failure.json`.
- Submitted the single-setting control `train-long-ep4-edp2-no-stack-001` as job `716928`, after confirming prior completion and sequential admission.
- Resolved-configuration comparison verifies only `stack_dump_seconds` changed from 60 to 0; all frozen source hashes and the immutable image identity are identical.
- The run retains two nodes/eight ranks, PP1/EP4/EDP2, four layers, sequence 8192, vocabulary 1024, two updates/checkpoint at 2, native grouped GEMM, heuristic one, native overlapping gradients, a 25-minute cutoff and 30-minute allocation cap.
- It runs without a reservation; only preserved interactive allocation `715971` was active at admission.
- No installed package/image/source modification, transport-policy change or full-scale repeat was introduced.


### 2026-11-03: EDP2 no-timer acceptance and combined PP2/EDP2 control

- Job `716928` passed with full-job exit zero and frozen global acceptance on all eight ranks.
- The PP1/EP4/DP8/EDP2 control completed two updates and a globally validated model/optimizer checkpoint at 2.
- Actual training provenance on every rank retained heuristic one and native grouped GEMM selection zero.
- Reported losses were 7.517748 and 7.517900, gradient norms 4.036 and 4.035, with zero skip/NaN counters.
- This establishes reduced-model nontrivial expert-DP synchronization, not the full PP2/EDP4/36-microbatch case; no restart was part of this control.
- Disabling the periodic dump was the sole resolved-configuration change from the SIGSEGV run, but the node allocation changed, so timer causality is still unproven.
- Cold/second-printed timings are not independent steady-state samples and are not compared to the reference.

Evidence: benchmark `results/train-long-ep4-edp2-no-stack-716928.json` and `results/train-long-ep4-edp2-no-stack-716928-notes.json`.

Submitted the next targeted two-node/eight-rank control `train-long-pp2-ep2-edp2-001` as job `716931`, after sequential admission and the prior full-job exit.
It combines PP2/EP2/DP4/EDP2 at the same real width and sequence 8192, reduced four-layer depth/vocabulary1024, GBS4/MBS1 and one microbatch per update.
The run keeps native grouped GEMM, heuristic one and overlapping gradients, with periodic dumps disabled, two updates/checkpoint at 2, a 25-minute cutoff and 30-minute allocation cap, without a reservation.
EP2 gives 128 local experts rather than full EP8's 32; this is a combined pipeline/nontrivial-EDP diagnostic, not full-size acceptance.
No sixteen-node retry, image/source change or overlap/transport-policy workaround was introduced.
The CPU suite passes 108 tests; interactive allocation `715971` and all frozen evidence remain preserved.


### 2026-11-03: reduced PP2/EDP2 accepted; EDP4 control and cooperative diagnostics

#### PP2 acceptance

- Job `716931` passed with full-job exit zero after 8m36s and frozen global acceptance on all eight ranks.
- The reduced real-width/sequence-8192 PP2/EP2/DP4/EDP2 model completed two updates and checkpoint ownership/contents validation at 2.
- Reported losses were 7.538306 and 7.536028, gradient norms 4.053 and 4.050, with zero skip/NaN counters.
- Actual child provenance retained heuristic one and native grouped GEMM on all ranks.
- This did not reproduce the full-shape collective stall; depth four, vocabulary1024, EDP2/local128experts and one microbatch differ from the full EDP4/local32experts/36-microbatch run.
- No restart, exact-state equality or reference-throughput claim is attached to this two-update diagnostic.

Evidence: benchmark `results/train-long-pp2-ep2-edp2-716931.json` and `results/train-long-pp2-ep2-edp2-716931-notes.json`.

#### Opt-in cooperative stack sampling

- Added a harness-only `stack_dump_mode=cooperative` option with a positive diagnostic interval, requiring the instrumented training wrapper.
- It samples at most 32 main-thread Python frames from an ordinary Python daemon thread under the GIL, instead of using `faulthandler.dump_traceback_later`.
- It emits structured `APERTUS2_STACK_SAMPLE` records, has an explicit stop callback and records the actual mode/interval in child provenance.
- The sampler cannot run while another thread holds the GIL, cannot inspect native/GPU stacks and does not by itself prove training progress.
- Historical/default behavior is unchanged unless the cooperative mode is explicitly selected; performance runs with interval zero create no sampler thread.
- CPU unit tests and a real-thread two-sample host check passed; GPU runtime/numerical acceptance is pending.
- This is diagnostic harness work only, not an image-source or compute integration change, and does not establish that the earlier timer caused SIGSEGV.

#### Next EDP4 control

Submitted `train-long-ep2-edp4-cooperative-001` as job `716983`, sequentially after `716931` completed.
It uses two nodes/eight ranks, PP1/EP2/DP8/EDP4, four layers, sequence 8192, vocabulary1024 and GBS8/MBS1, so one microbatch per update.
It matches the full run's expert-replica count four but not its PP2/depth/vocabulary/accumulation geometry.
Native grouped GEMM, heuristic one and overlapping gradient reduction remain unchanged; cooperative Python samples are requested every 60 seconds.
The run has a 25-minute probe cutoff / 30-minute allocation cap, `csstaff`/`mi300`, no reservation, and no concurrent GPU training job.
All 114 CPU tests pass; full mid-4b capacity/checkpoint acceptance and the 26-update baseline remain blocked.


### 2026-11-03: EDP4 acceptance; PP2 with cross-node EDP4 on four nodes

- Job `716983` passed with full-job exit zero and frozen global acceptance on all eight ranks: two updates, checkpoint at 2 and native checkpoint ownership/contents validation.
- Reported losses were 7.529521 and 7.524575, gradient norms 4.034 and 4.032, with zero skip/NaN counters.
- Every rank recorded actual heuristic one/native grouped GEMM and cooperative sampling at a sixty-second interval, and emitted bounded Python stacks during GPU training without the earlier early native signal.
- This validates that diagnostic mode in this reduced run, not a general native/GPU profiler or a proof that the previous timer caused SIGSEGV.
- EDP4 alone is therefore not sufficient to reproduce the full stall at four-layer/PP1/vocabulary1024/one-microbatch geometry.
- Accepted PP2/EDP2 and PP1/EDP4 controls do not yet establish the combined PP2/cross-node-EDP4 path.

Evidence: benchmark `results/train-long-ep2-edp4-cooperative-716983.json` and `results/train-long-ep2-edp4-cooperative-716983-notes.json` retain global acceptance, actual diagnostic mode, sampled stacks and checkpoint/memory evidence.

Submitted the next reduced control `train-long-pp2-ep2-edp4-cooperative-001` as job `716995`: four nodes/16 ranks, PP2/EP2/DP8/EDP4, four layers, sequence 8192, vocabulary1024, GBS8/MBS1 and one microbatch per update.
With block rank placement, each PP stage spans two nodes and its EDP replica groups cross nodes; this combines the two independently passing features without another sixteen-node run.
It retains native grouped GEMM, heuristic one and overlapping gradients, with cooperative Python samples every 60 seconds, two updates/checkpoint at 2 and no cache-reconciliation warmup or restart.
Resources are `csstaff`/`mi300`, unreserved, sequential admission, maximum 30 minutes with a 25-minute probe cutoff; the preserved interactive allocation was the only active user job at admission.
The four-node control is a bounded diagnostic below the already-authorized sixteen-node scope, not a request for the short-job concurrent budget or a full-model repeat.
No compute/image/transport-policy change was introduced; all 115 CPU tests pass.
Full model depth/vocabulary/36-microbatch capacity, full-shape checkpoint acceptance and the 26-update baseline remain unvalidated.


### 2026-11-03: PP2/cross-node EDP4 accepted; accumulation-36 control submitted

- Job `716995` passed with full-job exit zero after 8m33s and frozen global acceptance on all sixteen ranks.
- Both updates completed with finite reported losses/gradients, zero skip/NaN counters and checkpoint ownership/contents validated at 2.
- Actual PP/EP coordinates and hostnames confirm four expert replicas per group spanning two hosts in each PP stage.
- The earlier six-minute PP0-backward / PP1-rerun-reduction stack split was transient; update 1 then completed, so those snapshots alone did not establish a reproduced stall.
- Native grouped GEMM, heuristic one and cooperative sampling were retained in actual child provenance.
- This accepts the reduced PP2/cross-node-EDP4 path at one microbatch per update, not the full-model failure case.

Evidence: benchmark `results/train-long-pp2-ep2-edp4-cooperative-716995.json` and `results/train-long-pp2-ep2-edp4-cooperative-716995-notes.json` retain checkpoint/memory evidence, reported updates and actual replica placement.

Submitted `train-long-pp2-ep2-edp4-accum36-001` as job `717038`, sequentially after the prior job completed.
The only resolved training-setting delta is GBS8 → GBS288 at DP8/MBS1, giving 36 microbatches per update, matching the full reference/native accumulation count.
Resources and shape remain four nodes/16 ranks, PP2/EP2/DP8/EDP4, four layers, sequence 8192, vocabulary1024, two updates/checkpoint at 2, a 25-minute probe cutoff and 30-minute allocation cap, `csstaff`/`mi300`, unreserved.
The image, native grouped backend, heuristic, gradient-fusion setting and overlapping-communication policy are unchanged.
The cooperative sampler now additionally records Unix nanoseconds for cross-host correlation; its monotonic timestamps remain host-local and must not be subtracted across hosts.
This is a diagnostic-metadata source delta only; no installed compute source or image change occurred.
All 116 CPU tests pass; full depth/vocabulary/EP8 capacity, full-size checkpoint acceptance and the 26-update baseline remain blocked.


### 2026-11-03: accumulation control still in progress; sync source audit

At the approximately seven-minute observation, accumulation-36 job `717038` had all sixteen ranks live, no fatal exception and no completed update yet.
Cooperative samples had moved from compilation into backward execution and the native batched pipeline communication synchronization; this was not yet the earlier post-backward expert-DP watchdog signature.
The prior one-microbatch control also had substantial cold skew, so these Python samples alone are not being called a reproduced deadlock.

A read-only pinned-source comparison checked the reference/native pipeline and gradient synchronization paths while the run continued.
The reference contains an optional `BEVERIN_NO_P2P_SYNC` gate, but that flag is absent from the retained executed `708188` log/environment dump; no enabled reference-path difference or workaround is claimed from its mere presence in source.
The pinned `finalize_model_grads.py` files are identical, including router-metric reductions.
Reference-only gradient-buffer O1 joins belong to its side-stream wgrad integration, not a generic native fix to port into this harness-only phase.
The audit and fingerprints are retained in benchmark `results/p2p-gradient-sync-source-audit.json`; no synchronization, overlap, transport or image policy was changed.
Monitoring remains bounded by the existing 25-minute probe cutoff / 30-minute allocation cap.


### 2026-11-03: accumulation-36 acceptance; real-vocabulary control submitted

- Job `717038` passed with full-job exit zero and frozen global acceptance on all sixteen ranks.
- The reduced PP2/EP2/cross-node-EDP4 model completed two updates with 36 microbatches/update, finite loss/gradient reports and zero skip/NaN counters.
- Reported losses were 7.537741 and 7.535954, gradient norms 4.039 and 4.037.
- Model/optimizer checkpoint ownership and contents were validated at update 2, including the expected 576 consumed samples for GBS288.
- The expert replica groups still span two hosts per PP stage.
- This does not reproduce the full-shape stall at reduced four-layer/vocabulary1024/EP2 geometry; the earlier cold Python synchronization samples were not themselves evidence of a deadlock.
- No steady-state/reference-throughput comparison or full-model capacity claim is attached to the two-update control.

Evidence: benchmark `results/train-long-pp2-ep2-edp4-accum36-717038.json` and `results/train-long-pp2-ep2-edp4-accum36-717038-notes.json`.

Submitted `train-full-vocab-pp2-ep2-edp4-accum36-001` as job `717068`, sequentially after acceptance.
Only vocabulary changes from 1024 to the reference/native 200063, plus its descriptive model name; all frozen source hashes and the immutable image identity are identical to `717038`.
The run keeps four nodes/16 ranks, PP2/EP2/DP8/EDP4, four layers, sequence 8192, GBS288/MBS1 and 36 microbatches/update, two updates/checkpoint at 2, native grouped GEMM, heuristic one and native overlapping gradient reduction.
Cooperative Python sampling remains enabled for diagnosis; allocation/probe limits remain 30/25 minutes, `csstaff`/`mi300`, without a reservation.
All 117 CPU tests pass; depth32/EP8 and full-shape warmup/checkpoint acceptance remain unvalidated.

The existing interactive allocation `715971` was at approximately 11h05 of its original 12-hour limit during admission, leaving about 55 minutes.
It has not been cancelled, extended or otherwise changed; the user was notified because its builder store is memory-backed.
The verified training SquashFS is durable and does not depend on that allocation staying alive.


### 2026-11-03: real-vocabulary run pending; eight-layer footprint control prepared

At the approximately seven-minute observation, real-vocabulary job `717068` still had sixteen live ranks, no fatal error and no completed update.
Its samples were in backward/pipeline communication, not yet the old full-shape post-backward watchdog signature; it remains under the original limits.

Prepared `train-eight-layer-full-vocab-pp2-edp4-001`, not submitted and explicitly gated on `717068` acceptance.
It retains four nodes/16 ranks, PP2/EP2/EDP4, vocabulary200063 and 36 microbatches/update; depth alone increases four → eight, plus its descriptive model name.
The immutable image and complete frozen source hashes are identical to the current run.
This increases the number of expert matrix elements and gradient buffers per rank toward the full case without another sixteen-node allocation.

| Geometry | PP0 local expert instances | PP1 local expert instances |
| --- | --- | --- |
| Full depth32, EP8 | 480 | 512 |
| Four-layer proxy, EP2 | 128 | 256 |
| Eight-layer proxy, EP2 | 384 | 512 |

The comparison assumes the existing balanced contiguous PP2 layout and first global layer dense.
The eight-layer PP1 stage matches the full run's expert fc1/fc2 matrix element count, not its complete model/optimizer/activation/library/CPU/tmpfs peak memory or total capacity.
The readiness record includes exact expert-matrix element and FP32-gradient byte counts and these limitations.
It keeps the same two-update/checkpoint and 25/30-minute probe/allocation limits if eventually admitted.
All 118 CPU tests pass; no extra GPU job or compute/image/policy change has been introduced while `717068` remains active.
Evidence: benchmark `results/train-eight-layer-full-vocab-pp2-edp4-001-readiness.json`.


### 2026-11-03: real-vocabulary accumulation accepted; eight-layer footprint run submitted

- Job `717068` passed with full-job exit zero and frozen global acceptance on all sixteen ranks.
- Real vocabulary200063, sequence8192, PP2/cross-nodeEDP4 and 36 microbatches/update completed two updates plus checkpoint ownership/contents validation at 2.
- Reported losses were 12.67025 and 12.66897, gradient norms 2.354 and 2.353, with zero skip/NaN counters.
- Checkpoint saving and subsequent CPU inspection were treated separately from completed training updates; process-group cleanup warnings remain documented.
- Actual training provenance retained native grouped GEMM and heuristic one on all ranks.
- Whole-child Torch allocator peaks were at most approximately 23.08 GiB for PP0 and 28.62 GiB for PP1; minimum captured device-free values were approximately 96.62 and 92.95 GiB respectively.
- Those counters are allocator peaks/point-in-time device-free samples, not complete CPU/library/tmpfs/APU HBM peak accounting, and are not extrapolated as a full-depth capacity result.

Evidence: benchmark `results/train-full-vocab-pp2-ep2-edp4-accum36-717068.json` and `results/train-full-vocab-pp2-ep2-edp4-accum36-717068-notes.json`.

Submitted the prepared eight-layer control `train-eight-layer-full-vocab-pp2-edp4-001` as job `717080`, sequentially after the prior job completed.
Depth4 → 8 and the model descriptor are the only resolved-input deltas; all frozen source hashes and the image identity remain identical.
It keeps four nodes/16 ranks, PP2/EP2/DP8/EDP4, real vocabulary200063, sequence8192 and 36 microbatches/update, two updates/checkpoint at 2 and cooperative Python samples.
The run remains unreserved on `csstaff`/`mi300`, with the same 25-minute probe cutoff / 30-minute allocation cap and native backend/overlap/transport policy.
Its second PP stage matches the full run's expert-matrix element count per rank; total memory, full32layer/EP8 execution and the 26-update baseline are not yet validated.
All 118 CPU tests pass; the interactive allocation, builder store and frozen historical evidence are untouched.


### 2026-11-03: eight-layer footprint acceptance; eight-node EP8/EDP4 diagnostic

- Job `717080` passed with full-job exit zero and frozen global acceptance on all sixteen ranks.
- The eight-layer, real-vocabulary, PP2/EP2/cross-nodeEDP4 control completed both 36-microbatch updates and checkpoint validation at 2.
- Reported losses were 12.68957 and 12.68478, gradient norms 2.367 and 2.368, with zero skip/NaN counters.
- Actual child provenance retained native grouped GEMM and heuristic one.
- This accepts the expert-matrix footprint surrogate whose second PP stage matches the full run's per-rank expert fc1/fc2 matrix element count; it is not full total memory/depth/EP8 capacity or performance acceptance.
- Neither the larger local expert-gradient footprint nor the already-tested vocabulary/accumulation/EDP4 combination reproduced the full stall at EP2.

Evidence: benchmark `results/train-eight-layer-full-vocab-pp2-edp4-717080.json` and `results/train-eight-layer-full-vocab-pp2-edp4-717080-notes.json`.

Submitted `train-sixteen-layer-pp1-ep8-edp4-001` as job `717115`, sequentially after the prior full-job exit and acceptance.
This is **eight nodes / 32 ranks**, not sixteen nodes: PP1/EP8/DP32/EDP4, sixteen layers, vocabulary200063, sequence8192, GBS1152/MBS1 and 36 microbatches/update.
It matches the full run's first-stage local depth, fifteen MoE layers × 32 local experts, data/expert group sizes and expert-replica rank strides, including EP groups spanning two nodes and EDP groups spanning four hosts under block placement.
It does not include pipeline communication and has a 32-rank default group instead of 64; both embedding endpoints reside in this single-stage model, so total memory is not identical to a full PP2 stage.
The purpose is to exercise full EP8/EDP4 inter-node expert fan-out separately from pipeline parallelism, before any further full sixteen-node attempt.
It retains the same immutable image, frozen harness source hashes, native grouped backend, heuristic one, overlapping gradients and cooperative Python samples; no compute/image/transport policy change was made.
Protocol remains two updates/checkpoint at 2, without warmup cache reconciliation or restart, `csstaff`/`mi300`, unreserved, maximum 30 minutes with a 25-minute probe cutoff.
The supervisor may stop before the 20-minute collective watchdog if cold setup exceeds five minutes; any stopping reason will be classified from actual evidence rather than assumed.
All 119 CPU tests pass, and the user's interactive allocation and memory-backed builder store remain untouched.


### 2026-11-03: EP8/EDP4 single-stage acceptance; reduced-depth full-rank topology submitted

- Job `717115` passed with full-job exit zero and frozen global acceptance on all 32 ranks.
- Sixteen local layers at PP1/EP8/DP32/EDP4, real vocabulary200063, sequence8192 and 36 microbatches/update completed two updates and checkpoint validation at 2.
- Reported losses were 12.69726 and 12.69526, gradient norms 2.356 and 2.349, with zero skip/NaN counters.
- Actual coordinates and hostnames confirm expert groups of eight ranks spanning two hosts and expert-replica groups of four ranks spanning four hosts.
- Thus full first-stage expert communication fan-out and local expert depth did not reproduce the failed full PP2 case without pipeline communication.
- Both embedding endpoints are present in this PP1 model, and its default process group has 32 rather than 64 ranks; no full-model/PP2 capacity, restart or throughput claim is attached.
- Checkpoint saving and CPU inspection were completed before acceptance; raw first/second logged intervals are not independent steady-state samples.

Evidence: benchmark `results/train-sixteen-layer-pp1-ep8-edp4-717115.json` and `results/train-sixteen-layer-pp1-ep8-edp4-717115-notes.json` retain actual two-host/four-host group placement and per-rank checkpoint/memory evidence.

Submitted `train-four-layer-full-topology-001` as job `717130` under the existing sixteen-node authorization, with sequential admission after `717115` acceptance.
This uses 16 nodes/64 ranks and the full failed run's TP1/PP2/CP1/EP8/DP32/EDP4 logical layout, vocabulary200063, sequence8192, GBS1152/MBS1 and 36 microbatches/update.
The native recipe and requested environment are identical to failed full-shape `716859`, including heuristic one, native grouped GEMM, disabled gradient fusion and quiet GEMM tracing.
The model is only four layers, with two continuous updates/checkpoint at2 and cooperative Python sampling, no disposable warmup/reconciliation phase or restart.
This is a topology-isolation diagnostic, not a repeat of the 32-layer model or full-model capacity/performance acceptance.
Parallel-group dimensions match the failed case, but reduced depth changes collective payload sizes/counts/timing, so identical executed collective sequences are not claimed.
It keeps the same image and complete frozen source hashes as accepted `717115`; the diagnostic wrapper differs from the historical failed full run.
Resources are `csstaff`/`mi300`, unreserved, at most 45 minutes with a 40-minute probe cutoff so the normal 20-minute collective watchdog can report if the failure recurs after cold startup.
The user's interactive allocation was the only active user job at admission and has not been modified.
All 120 CPU tests pass; full 32-layer capacity, full-shape warmup/checkpoint acceptance and the 26-update baseline remain unresolved.


### 2026-11-03: actual full-topology startup parity verified

All 64 native startup coordinate tuples in reduced-depth job `717130` match the failed full run `716859` at the same global rank: TP/EP/DP/EDP/PP coordinates were checked from actual child logs, not only intended configuration.
Host placement confirms eight expert groups spanning two hosts each, sixteen expert-replica groups spanning four hosts each and 32 pipeline pairs spanning two hosts each.
All 64 runtime roots are backed by executable, writable tmpfs.
This verifies actual startup coordinates and inferred group placement, not a full Torch process-group registry dump or identical collective payload/order/timing; physical nodes and model depth/protocol differ.
Training/checkpoint/full-job acceptance is still pending, and no capacity/root-cause conclusion is drawn from startup parity.
Evidence: benchmark `results/train-four-layer-full-topology-001-startup-parity.json`.


Resource notice: interactive allocation `715971` was observed at 11h57m36s of its original 12-hour limit, leaving approximately two minutes.
The user was notified again of the RAM-backed builder-cache expiry risk; no cancellation, extension, backup or cleanup was performed without further instruction.
The verified SquashFS and experiment evidence are already durable on shared storage and do not depend on that allocation.


### 2026-11-05: full-topology acceptance finalized; depth-only full-model test submitted

#### Accepted control after the session restart

- Frozen global acceptance for job `717130` is now finalized: full-job exit zero, all 64 ranks passed both updates and checkpoint ownership/contents validation at update 2.
- This control used four layers with the full PP2/EP8/DP32/EDP4 topology, sequence 8192, vocabulary 200063 and 36 microbatches per update.
- Reported losses were 12.67185 and 12.67054, gradient norms 2.355 and 2.353, with zero skip/NaN counters.
- Actual startup coordinates match failed full-depth `716859` rank by rank; EP groups span two hosts, EDP groups four hosts and pipeline pairs two hosts.
- Reduced depth changes collective payload/count/timing and memory; this success does not resolve the earlier full-depth synchronization failure.
- The resumed GPU-free harness suite passes 120 tests, and current harness source hashes exactly match accepted `717130`.

Evidence: benchmark `results/train-four-layer-full-topology-717130.json`, `results/train-four-layer-full-topology-717130-notes.json` and `results/train-four-layer-full-topology-001-startup-parity.json`.

#### Controlled full-depth experiment

Submitted `train-mid4b-full-depth-controlled-001` as job `721544`, sequentially under renewed autonomous authorization.
Only model depth changes from 4 to 32 layers, plus its descriptive model name, relative to accepted `717130`; the complete frozen harness source and image identity are identical.
The SquashFS survived the clean-node restart, and fresh verification reproduced SHA-256 `b4f5300c8ecbb69a442e44c0044007a09e56742422b4a867b34eac716f51ea9d` for the 31,528,255,488-byte artifact.
No Podman operation, local-cache recovery or image rebuild was needed.

| Setting | Controlled full-depth run |
| --- | --- |
| Resources | 16 nodes / 64 ranks, `csstaff` / `mi300`, default queue, no reservation |
| Limits | 45-minute allocation, 2400-second probe cutoff |
| Model | 32 layers, sequence 8192, vocabulary 200063 |
| Parallelism | TP1 / PP2 / CP1 / EP8 / DP32 / EDP4 |
| Batch | GBS1152 / MBS1, 36 microbatches per update |
| Protocol | Two continuous cold updates, checkpoint at 2; no restart or disposable warmup/cache phase |
| Diagnostics | Cooperative main-thread Python samples every 60 seconds, Unix and host-local monotonic timestamps |
| Backend | Native TE grouped GEMM, heuristic one, gradient fusion disabled; existing per-run transport/overlap policy unchanged |

The user interactive allocation `721530` on `nid002508` remains untouched; no other training job was active at admission.
A full-depth pass requires zero full-job exit and frozen global acceptance, not merely live ranks or completed update prints.
Checkpoint saving and validation remain separate milestones; if this gate passes, full-size fresh-process restart and full-shape warmup/cache reconciliation are still required before the 26-update native baseline.
If it stalls, retain actual rank phases/stacks, memory/provenance and stopping reason before choosing a targeted comparison; do not blindly repeat the large allocation.
Native baseline scope remains distinct from Imanol's optimized Turbo/T6/direct-AITER/custom optimizer execution, production convergence and throughput parity.
Evidence: benchmark `results/train-mid4b-full-depth-controlled-001-admission.json` and run-local `submission.json` / `launcher.log`.


## Follow-up, 2026-11-05 — full-depth startup parity and first bounded observation

- At approximately 7½ minutes, job `721544` had all 64 ranks live, no retained fatal exception and no completed update.
- The latest cooperative samples put all PP0 ranks in batched pipeline communication and all PP1 ranks in backward execution; these are Python snapshots, not proof of useful GPU progress or a deadlock.
- The earlier failed run's post-backward expert-DP/router-metric/default-group divergence has not been established at this observation.
- All actual startup TP/EP/DP/EDP/PP tuples match both accepted four-layer `717130` and failed full-depth `716859` at the same global rank.
- Actual host placement confirms two-host EP groups, four-host EDP groups and two-host pipeline pairs; every rank runtime root is executable, writable tmpfs.
- No runtime flag, source, image, allocation limit or concurrency policy was changed; monitoring continues within the original bounds.

Evidence: benchmark `results/train-mid4b-full-depth-controlled-001-startup-parity.json` and append-only `results/train-mid4b-full-depth-controlled-001-observations.jsonl`.


### 2026-11-05: full-depth run cancelled on request; communicator diagnostic submitted

#### Full-depth outcome and retained evidence

- Cancelled only job `721544` at the user's explicit request after 25m13s; it completed zero training updates and no checkpoint.
- Slurm records `CANCELLED`, not a watchdog-confirmed failure in this run; the supervisor's nonzero exit and rank SIGTERM errors follow that cancellation.
- All 16 nodes were released before admitting another training job; interactive allocation `721530` remains untouched.
- Cooperative samples retained a persistent post-backward split: ranks 0/8/16 at `finalize_model_grads.py:335`, the other 29 PP0 ranks in `_log_global_router_metrics` all-reduce, and all 32 PP1 ranks in the rerun-state global reduction.
- A read-only extraction of the exact installed sources confirms line 335 is `int(sample_count_tensor.item())`, after `finish_grad_sync`.
- A CPU-visible `.item()` wait can reflect outstanding GPU work; it does not establish a defect in that scalar read.
- This resembles the previous full-depth wait pattern but is not an exact replay of its watchdog evidence, nor proof of a transport defect.

Evidence: benchmark `results/train-mid4b-full-depth-controlled-721544-before-cancel.json`, `results/train-mid4b-full-depth-controlled-721544-cancelled.json`, `results/train-mid4b-full-depth-controlled-001-live-source-audit.json` and extracted-source hashes under `reference/router-metric-communicator-2026-11-05/`.

#### Targeted hypothesis and opt-in diagnostic

First-use router-metric communicator initialization may conflict with outstanding GPU work from gradient synchronization.
This is a hypothesis to test, not an established root cause.
Added opt-in harness control `probe.prime_router_metric_communicator=true`, which requires the instrumented entrypoint; existing/default runs do not enable it.
It wraps distributed initialization and performs one constant-valued scalar all-reduce on the exact tensor/data/context-parallel group used by global router metrics, before model creation and training RNG seeding.
It explicitly completes that GPU work, checks the scalar sum against group size, then completes a global barrier before training proceeds.
The control does not disable metrics, change loss or optimizer logic, modify model tensors, consume random numbers through its constant allocation, or change the image's communication policy.
Actual group membership, numerical result and diagnostic execution status are logged and retained in child provenance.
Global acceptance additionally requires successful priming on every rank and consistent actual group membership; a requested flag alone is insufficient.
This changes first-use communicator timing and adds a startup collective/barrier, so it is a diagnostic comparison, not identical execution or a validated performance optimization.
All 127 GPU-free tests pass, including positive/negative priming, hook restoration, config isolation and all-rank evidence checks.

#### Small runtime validation first

Submitted `train-router-communicator-priming-small-001` as job `721576`, sequentially after the cancelled job fully released its allocation.
It uses two nodes/eight ranks, PP2/EP2/DP4/EDP2, a tiny four-layer model with sequence 128, two updates and checkpoint validation at 2.
The allocation is capped at 15 minutes with a 720-second probe cutoff, using `csstaff`/`mi300` without a reservation.
It retains the existing native grouped backend, heuristic one and disabled gradient fusion, with cooperative Python stacks.
The purpose is to validate the diagnostic hook under real ROCm/distributed execution and then verify training/checkpoint functionality, not to reproduce or resolve the full-depth stall at tiny scale.
No other GPU training job is active, and no image rebuild or Podman operation was introduced.
Any full-depth diagnostic comparison remains gated on this small run's zero full-job exit and frozen global acceptance.
Full-model correctness/capacity, full-size restart, full-shape cache reuse and the 26-update baseline remain blocked.


## Follow-up, 2026-11-05 — communicator priming executed; small training acceptance pending

All eight ranks of small job `721576` completed the opt-in startup all-reduce, obtained the expected sum of four and completed the global synchronization before training.
The two PP-stage metric groups have four ranks each on one host in this two-node block placement; this validates the hook's real ROCm execution, not full32-rank inter-node metric-group behavior.
At approximately seven minutes, no training update or fatal error was logged; samples had moved from pipeline communication into PP0 backward and PP1 rerun-state reduction.
The run remains bounded by its original 720-second probe cutoff and 15-minute allocation, without an extension or concurrent job.
Hook execution alone is not training/checkpoint acceptance and is not evidence that the full-depth blocker is fixed.
If it cannot complete, the next comparison will hold this tiny configuration and frozen source constant while disabling only priming; no larger primed run is admitted on startup evidence alone.
Evidence: benchmark `results/train-router-communicator-priming-small-721576-initialization.json` and the run's live logs.


### 2026-11-05: small diagnostic accepted; full-depth first-use comparison submitted

#### Small diagnostic acceptance

- Job `721576` passed with full-job exit zero and frozen global acceptance on all eight ranks.
- Both updates completed with finite losses/gradients and zero skip/NaN counters, and checkpoint ownership/contents validation passed at update 2.
- Reported losses were 7.593303 and 7.588262, gradient norms 1.959 and 1.985.
- Actual child provenance confirms successful communicator priming on every rank, two consistent groups of four, and the expected scalar reduction result.
- This establishes real-runtime hook/training/checkpoint functionality, not an improvement over an unprimed matched control or a fix for the full-depth stall.
- The cold first-update PP-stage skew resolved; samples of PP0 backward or gradient synchronization and PP1 rerun-state reduction were not alone proof of a stall.

Evidence: benchmark `results/train-router-communicator-priming-small-721576.json` and `results/train-router-communicator-priming-small-721576-notes.json`.

#### Full-depth targeted comparison

Submitted `train-mid4b-full-depth-primed-001` as job `721592`, after the small run completed and its allocation was released.
It uses the full 32-layer model on 16 nodes/64 ranks, PP2/EP8/DP32/EDP4, sequence 8192, vocabulary 200063 and GBS1152/MBS1 with 36 microbatches per update.
The built training-entrypoint arguments are exactly identical to cancelled unprimed `721544`; the only execution-control change is early priming of the actual global-router metric communicator, before model/RNG setup.
The immutable image and frozen harness source are identical to accepted small diagnostic `721576`.
Native TE grouped GEMM, heuristic one, disabled gradient fusion, metrics, loss, optimizer and gradient-communication policy remain unchanged.
This deliberately changes first-use communicator timing and adds startup collective/barrier work; it is not identical execution or an established optimization.

The allocation cap is reduced from 45 to 30 minutes, and the probe cutoff from 2400 to 1500 seconds, to avoid extending another inconclusive large run.
The protocol remains two cold continuous updates and checkpoint validation at 2, without restart or cache reconciliation.
It runs unreserved on `csstaff`/`mi300`, with sequential admission; interactive allocation `721530` was the only active user job before submission and remains untouched.
Acceptance requires successful actual priming on all 64 ranks, complete updates, checkpoint validation and zero full-job exit.
The 25-minute supervisor cutoff can precede a 20-minute collective watchdog after cold setup; any stopped run must be classified by its actual stopping reason.
No image rebuild, Podman recovery, concurrent training job or timeout extension was introduced.
Evidence: benchmark `results/train-mid4b-full-depth-primed-001-admission.json` and run-local `submission.json` / rank logs.


## Follow-up, 2026-11-05 — full-scale communicator initialization verified

All 64 ranks of `721592` completed the early router-metric communicator all-reduce and global synchronization before training.
Actual membership confirms two groups of 32 ranks, each spanning eight hosts, with the expected scalar sum of 32 on every rank.
The first bounded monitor tail omitted the older priming records on verbose ranks 0 and 32; a direct full-log check recovered them, and the external monitor now reads the startup markers separately from its bounded live stack tail.
At approximately seven minutes, there was no logged training update or fatal exception; full-depth training/checkpoint acceptance remains pending.
This establishes that the target communicator can initialize successfully before gradient work at full scale, not that first-use initialization caused the previous stall or that priming fixes training.
The original 25-minute probe cutoff and 30-minute allocation cap remain unchanged, with no concurrent GPU training job.
Evidence: benchmark `results/train-mid4b-full-depth-primed-721592-initialization.json`.


## Follow-up, 2026-11-05 — bounded live HBM observations during the primed full-depth run

At the approximately 14-minute observation, `721592` had no logged update or fatal exception; samples remained in backward/GEMM/pipeline synchronization rather than the earlier persistent router-metric split.
Read-only snapshots from one PP0 host and one PP1 host retained node `/proc/meminfo` and GPU memory/utilization counters without importing Torch or modifying training.
Point-in-time GPU memory accounting was approximately 75–76 GiB per GPU on the sampled PP0 host and 58–60 GiB on the sampled PP1 host; these are not full-run allocator peaks or proof of capacity through optimizer/checkpoint execution.
MI300A CPU/GPU/tmpfs share HBM, so these counters must not be added to host/RSS/tmpfs counters as separate memory pools.
The diagnostic EDF steps did not expose the training steps' private `/tmp` roots; empty diagnostic-step `/tmp` does not mean empty training caches, and training-cache bytes were not measured by this snapshot.
High GPU utilization is not by itself evidence of productive progress.
The original 25-minute probe cutoff and 30-minute allocation cap remain unchanged.
Evidence: benchmark `results/train-mid4b-full-depth-primed-721592-live-memory-001.json`.


## Follow-up, 2026-11-05 — early metric-communicator priming is not sufficient

The full-depth primed run `721592` has retained a persistent post-backward split over repeated samples, with zero completed updates despite successful early communicator initialization on all 64 ranks.
Ranks 0/8/24 block on the microbatch metric sample-count `.item()` at `finalize_model_grads.py:335`.
The other 29 PP0 ranks have returned from the global-router all-reduce API call and block on the subsequent GPU result read at line 477.
All 32 PP1 ranks wait in the rerun-state scalar reduction.

Unlike unprimed `721544`, the 29 PP0 ranks are no longer visibly stuck inside the all-reduce API's first-use call; this does not establish GPU collective completion.
The evidence shows that priming this router-metric communicator does not eliminate the observed full-depth wait and weakens a pure lazy-bootstrap explanation.
Outstanding collective/stream dependencies or progress remain hypotheses; no invalid collective ordering, network defect or specific library regression has been proven.
No update/checkpoint/capacity success or definitive watchdog outcome is claimed from this live observation.
The run will stop under its existing 25-minute probe cutoff / 30-minute allocation cap without extension.
No unchanged full-size repeat is planned; further work should target per-group collective sequence/payload and stream-completion evidence before another large allocation.
Evidence: benchmark `results/train-mid4b-full-depth-primed-721592-persistent-wait.json` and retained startup/source/memory records.


## Follow-up, 2026-11-05 — primed full-depth diagnostic stopped at its cutoff

- Job `721592` failed under the configured 1500-second rank-supervisor budget; Slurm records 25m21s, exit 1:0, and the waiting launcher records full-job return code 1.
- Four ranks explicitly reported the configured diagnostic timeout; the other 60 reported SIGTERM after srun terminated the failed step.
- No training update or checkpoint completed.
- All 64 startup communicator primes had succeeded, but the persistent three-rank PP0 sample-count read / 29-rank PP0 global-metric result read / 32-rank PP1 rerun-state split remained through the final samples.
- This is a negative result for router-metric communicator priming as a sufficient remedy, not proof of a specific Megatron, PyTorch, RCCL or transport defect.
- The stop was the supervisor cutoff, not an observed new 20-minute collective-watchdog timeout or an allocation extension.
- All 16 training nodes have been released; only preserved interactive allocation `721530` remains active.
- No new GPU job is submitted, no unchanged full-size repeat is planned, and no image rebuild or global runtime policy change was made.
- Further diagnosis should obtain per-group collective sequence/payload and stream-completion evidence using bounded, validated instrumentation rather than infer the unfinished GPU operation from Python stack locations alone.

Evidence: benchmark `results/train-mid4b-full-depth-primed-721592-failure.json`, run-local `submission-completion.json` / `slurm-721592.err`, and retained initialization, persistent-wait, source and memory records.
Full-depth correctness/capacity, full-size restart, full-shape cache reconciliation and the 26-update baseline remain unvalidated.


## Follow-up, 2026-11-05 — reference Megatron source comparison started

Frozen a separate export of reference commit `d1277d86e0c1925f8438fd7630d86d05ab398dcd`; the native checkout, candidate image and all previous runs remain unchanged.
The source inventory contains 2420 regular files and one reviewed internal symlink, with per-file hashes and a deterministic 9,599,568-byte compressed archive.
The `_research` Git submodule pointer is recorded but its external contents are not included; this is the committed Megatron source export, not a reconstruction of all reference runtime dependencies.
No compatibility patch has been applied to this clean reference artifact.

The harness can now freeze that archive inside each prepared run, stage and verify it in rank-private executable tmpfs, select its actual pretraining entrypoint, and verify loaded Megatron module paths/hashes in child provenance.
It rejects mixed native/reference module origins and requires actual-source evidence during acceptance.
The image's prebuilt CPU BFD DSO is reused only after confirming that its C++ source is byte-identical; other source-local JIT outputs remain in private tmpfs rather than modifying a shared checkout.
Existing runs without an explicit source manifest keep their original image-source behavior.
The reference-source comparison explicitly disables Turbo, T6, O1 side-stream work, custom attention/fusion gates and reference prewarm, while keeping native backend/library controls unchanged.
This is labeled reference checkout on native runtime, not execution-equivalent optimized reference reproduction.
The code audit confirms that the reference O1 joins have no side streams to join when those features are disabled.

Submitted import/API-only job `721708`, `megatron-reference-import-001`: one node, one rank, four CPUs, a five-minute allocation cap and a 240-second probe cutoff, using `csstaff`/`mi300` without a reservation.
It checks the clean reference entrypoint and relevant training, optimizer, expert and synchronization imports against the existing image, without constructing a model or running training.
All 138 GPU-free harness tests pass, including source inventory, bounded staging, immutable artifact checks, actual-module origin checks, BFD source matching, environment isolation and import-probe bounds.
No two-node training or full-model source comparison will be submitted before the import check is accepted.
No Podman operation or image rebuild has been introduced; GPU/diagnostic allocations remain sequential and interactive allocation `721530` is untouched.
Evidence: benchmark `reference/megatron-checkout-2026-11-05/`, `results/megatron-reference-import-001-admission.json` and run-local frozen source inputs.


## Follow-up, 2026-11-05 — reference-source imports accepted; small restart submitted

Import/API job `721708` completed with full-job exit zero and passed its frozen global summarizer.
Actual loaded module origins and hashes match clean reference revision `d1277d86e0c1925f8438fd7630d86d05ab398dcd`, with no mixed native Megatron checkout or source patches.
The effective Turbo, T6 and O1 flags were all false.
Imports initialized a CUDA/HIP context but constructed no model; this acceptance does not yet cover training, optimizer execution or restart.
The image CPU BFD DSO was seeded only after its C++ source matched; its SHA-256 is `884977d815113041be6a92f9cc8536b7f6c8ddfc8fb26e142a251df1fc1afe48`.
Source archive SHA-256 is `0ae108a350f69b4d4ce3e95ad75e6d7eb083edc773407144d76b9e5ddb11f9f5`; frozen artifact manifest SHA-256 is `e7551871da4da4a9a25c55c9417597e4d3c6184154cd5046af9f97e795b737d3`.
Evidence: benchmark `results/megatron-reference-import-721708.json` and run `megatron-reference-import-001`.

Submitted sequential two-node/eight-rank job `721747`, run `train-reference-checkout-small-restart-001`, after all 140 CPU tests passed and frozen source/image, AST, Bash, TOML and scheduler admission checks succeeded.
The small four-layer model uses hidden size 256, sequence length 128, vocabulary 1024, eight experts/top-2, PP2/EP2/DP4/EDP2 and GBS4/MBS1.
The protocol is updates 1–2, checkpoint 2, fresh-process load at 2, then updates 3–4 and checkpoint 4.
It uses `csstaff`/`mi300` without a reservation, a 15-minute allocation cap and a 720-second supervisor cutoff.
The reference-source artifact is unchanged and unpatched; the existing explicit ATen router control sets only the router's `te_general_gemm` fallback and matches the native candidate's effective router path.
Native TE grouped GEMM, heuristic-one selection, ordinary all-to-all, disabled gradient-accumulation fusion and cooperative stack sampling remain selected.
Turbo/T6/O1/reference prewarm remain off; communicator priming is also off.
No other diagnostic job was active at admission, and interactive allocation `721530` is preserved.
The small run must pass full-job exit, all-rank/stage source provenance, checkpoint ownership and fresh-process resume acceptance before any full-depth comparison.
Collective sequence/payload and completion tracing remains a required separate small-tested gate before that comparison; no claim of full-depth progress or optimized-reference equivalence is made.


## Follow-up, 2026-11-05 — private-checkout CPU dataset-helper staging correction

Two-node reference-source job `721747` failed before any update after 1 minute 23 seconds (`FAILED`, exit `1:0`).
Seven nonzero ranks reported `ModuleNotFoundError: No module named 'megatron.core.datasets.helpers_cpp'` while constructing `MockGPTDataset`.
The reference `initialize._compile_dependencies()` builds that CPU extension only on global rank 0 and then barriers, assuming that all ranks share one writable checkout.
Our intentionally rank-private tmpfs trees violate that shared-filesystem assumption; rank 0's successful compilation did not publish a binary into the other seven trees.
This is a source-staging defect and does not yet establish a training, optimizer or collective compatibility failure.
The failed run, logs, frozen inputs and full-job completion record remain intact.
Evidence: benchmark `results/train-reference-checkout-small-restart-721747-failure.json` and run `train-reference-checkout-small-restart-001`.

The explicit staging correction builds `helpers_cpp` from the frozen reference `helpers.cpp` and its unchanged Makefile in every private rank tree before distributed training starts.
Each CPU build has a 120-second subprocess bound and a durable log, and its source hashes, command and resulting DSO hash are added to actual-child source provenance.
No Megatron source patch, GPU warmup, optimized-reference kernel, library change or image rebuild is introduced.
All 141 CPU tests passed before preparing the retry.
Sequential job `721759`, run `train-reference-checkout-small-restart-002`, uses byte-identical resolved configuration and reference artifact to the failed small run: two nodes/eight ranks, PP2/EP2, updates 1–2/save2 followed by fresh-process load2/updates3–4/save4, 15-minute cap and 720-second supervisor cutoff.
The only active pre-existing allocation at admission was preserved interactive job `721530`.

Collective diagnostics are now implemented in the working harness but were not enabled in `721759` and have not yet passed a GPU compatibility gate.
The opt-in trace records bounded per-group host-call order, actual group membership/name, native sequence observations, tensor shape/dtype/size/strides, reduction/split arguments and caller frames without reading GPU tensor values.
API returns are explicitly labeled as not establishing GPU completion.
A bounded native PyTorch NCCL/RCCL JSON flight-recorder snapshot is taken periodically and at ordinary completion, with capability checks against the installed build; recorder timing adds diagnostic overhead but no explicit barriers, Work waits or stream synchronization.
The offline report retains native states and flags only candidate all-reduce signature mismatches, with explicit caveats about rotation, concurrency, coalescing and uninstrumented calls.
Before any full-depth reference-source comparison, the small training/restart gate must pass and tracing must pass its own small GPU run.


## Follow-up, 2026-11-05 — clean reference source passes small training/save/restart

Two-node/eight-rank job `721759`, run `train-reference-checkout-small-restart-002`, completed successfully in 9 minutes 4 seconds (`COMPLETED`, full-job exit `0:0`).
Its frozen global summarizer accepted all eight ranks, updates 1–4, checkpoint targets 2/4, actual PP2/EP2/DP4/EDP2 checkpoint ownership and fresh-process resume from update 2.
Actual Megatron module paths and hashes matched reference commit `d1277d86e0c1925f8438fd7630d86d05ab398dcd` on every rank and both fresh-process stages, with no mixed checkout or source patches.
Effective Turbo/T6/O1 flags were false everywhere.
The explicit ATen router fallback, native TE grouped GEMM/heuristic one, ordinary all-to-all and disabled gradient fusion remain named native-backend choices.
The newly staged CPU dataset helper was present in every private tree; all ranks reported the same DSO SHA-256 `911d464df87a3e56b9c15c51751674c050946790487f9724e4ee86cc018d146d`, built from recorded frozen C++/Makefile hashes.
This establishes clean-source small-model training, optimizer/save and restart compatibility in the existing image, not full-depth progress, full-shape cache acceptance, convergence or optimized-reference equivalence.
Evidence: benchmark `results/train-reference-checkout-small-restart-721759.json` and the run's frozen per-rank stage provenance/checkpoint records.

After all 152 CPU tests passed, submitted sequential job `721782`, run `train-reference-checkout-small-traced-001`, to validate host-call and native NCCL/RCCL flight-recorder diagnostics on that same four-update/save2/restart/save4 small protocol.
The resolved configuration differs from accepted `721759` only by enabling `probe.collective_trace`; image and clean source artifact are unchanged.
It retains two nodes/eight ranks, a 15-minute allocation cap, 720-second supervisor cutoff, `csstaff`/`mi300`, no reservation and no concurrent diagnostic allocation.
The trace has two bounded 8-MiB host-log files per stage, a 4096-entry native recorder ring and one bounded 16-MiB native JSON snapshot per stage, refreshed every 30 seconds.
The existing native timing recorder adds GPU event/diagnostic overhead; no explicit synchronization, collective, Work wait or device tensor-value read is added by the Python instrumentation.
Frozen global acceptance additionally requires actual host calls and native completed entries on every rank/stage, while the offline report inventories pending host calls/native states and candidate all-reduce metadata mismatches without treating API returns as completion.

Prepared but did not submit `train-mid4b-reference-checkout-traced-001`: full 32-layer geometry, sixteen nodes/64 ranks, PP2/EP8/DP32/EDP4 and 36 microbatches per update, with two target updates/save2, a 30-minute allocation cap and a 1500-second supervisor cutoff.
Its emitted training arguments are identical to the prior native full-depth unprimed control; the wrapper selects the clean reference source, explicit ATen router fallback and the same tracing implementation currently being small-tested.
No communicator priming, GPU prewarm, cache hydration or reference optimization is enabled.
Submission remains gated on successful `721782` trace compatibility/publication and sequential scheduler admission; preparation is not authorization to bypass those gates.
The candidate image, native checkout, failed runs and interactive allocation `721530` are unchanged.


### Native recorder capability observed during small trace job `721782`

The existing image exposes PyTorch's native NCCL/RCCL JSON flight-recorder binding and publishes usable snapshots from all eight ranks without requiring a new image.
Startup snapshots contain actual `completed` communication entries, group names/memberships, native collective sequence IDs, input/output shapes and dtypes, timing fields, process-group status and communicator state.
The recorder schema reports `version="2.10"`; this is a trace-format version and must not be confused with the installed Torch package version.
The native recorder reports `comm_lib_version="2.30.4"`, whereas retained reference execution logs reported RCCL `2.27.7` and OFI plugin `1.4.0`.
This is additional concrete stack-difference evidence, not proof that the version difference causes the full-depth wait; complete plugin/build/runtime alignment remains a separate investigation.
Evidence: benchmark `results/collective-trace-721782-native-recorder-capability.json` and run-local native snapshots.
These startup observations do not yet accept the traced training/restart protocol.
The offline analyzer now also aligns native submitted operations by recorded group membership/name and native collective sequence ID, checking operation identity and all-reduce shape/dtype metadata independently of host API returns.
Ambiguous coalesced/concurrent sequence observations and missing rotated/truncated rows are not treated as proof of divergence.
Native snapshot collection itself can contend for the GIL or block in driver/event queries even though the Python instrumentation adds no explicit synchronization; stale/missing snapshots do not establish GPU progress or the absence of progress.


## Follow-up, 2026-11-05 — traced small restart accepted; bounded full-depth source comparison submitted

Job `721782`, `train-reference-checkout-small-traced-001`, completed successfully in 9 minutes 8 seconds (`COMPLETED`, full-job exit `0:0`).
The run's frozen global summarizer accepted all eight ranks and both fresh-process stages: updates 1–4, checkpoints 2/4, load from checkpoint 2, actual reference-source module provenance, and host-call/native-flight evidence.
No source patch or new image was needed.
Both offline stage reports found zero host all-reduce signature mismatch candidates and zero native group/sequence submission-signature mismatch candidates.
All 160 critical router-metric/rerun-state all-reduce calls across eight ranks and two stages matched native `completed` entries using group identity and the observed before-sequence/native-entry offset.
Trace publication succeeded without rotation or native snapshot errors; retained trace artifacts total approximately 0.55–0.70 MB per rank across both stages.
The successful run still retained 12 scheduled entries per PP0 rank and 20 per PP1 rank, comprising P2P send/receive and coalesced `_broadcast_oop` subentries.
Those scheduled subentries are a calibrated recorder limitation: scheduled state alone cannot establish unfinished GPU work or a deadlock, even when `retired=false`.
Native group status, actual operation sequence and payload metadata, host/source stacks, completed parent operations and job outcomes must be interpreted together.
Evidence: benchmark `results/train-reference-checkout-small-traced-721782.json`, both `*-stage{0,1}-collectives.json` reports and `results/collective-trace-721782-critical-path-calibration.json`.

After checking those gates, immutable prepared inputs, training-argument equality and sequential admission, submitted job `721795`, run `train-mid4b-reference-checkout-traced-001`.
This is the single approved bounded full-depth clean-reference-checkout comparison on the unchanged native candidate image: sixteen nodes/64 ranks, full 32-layer/sequence8192/vocabulary200063 mid-4b geometry, PP2/EP8/DP32/EDP4, GBS1152/MBS1 and 36 microbatches per update.
The target is two cold updates and checkpoint 2, with no fresh-process restart in this diagnostic, a 30-minute allocation cap and 1500-second supervisor cutoff.
It uses `csstaff`/`mi300` without a reservation, with actual tracing/source-selection/runtime code byte-identical to the accepted small traced run.
The emitted training arguments match the prior native full-depth unprimed control; the clean reference source is selected only through the explicit frozen source wrapper.
The router uses the named ATen fallback; native TE grouped GEMM/heuristic one and ordinary all-to-all remain selected, while gradient fusion, communicator priming, Turbo, T6, O1, reference prewarm and cache hydration remain disabled.
Reference revision is `d1277d86e0c1925f8438fd7630d86d05ab398dcd`, with source archive SHA-256 `0ae108a350f69b4d4ce3e95ad75e6d7eb083edc773407144d76b9e5ddb11f9f5` and no source patches.
The candidate SquashFS identity remains SHA-256 `b4f5300c8ecbb69a442e44c0044007a09e56742422b4a867b34eac716f51ea9d`.
At admission no diagnostic job was active, and preserved interactive allocation `721530` was the only existing allocation.
All 153 CPU harness tests pass; no commit, image rebuild, native checkout replacement or removal of prior runs was performed.
The diagnostic has not yet established any full-depth update, checkpoint, full-shape cache correctness, 26-update baseline, throughput parity or convergence.
Evidence: benchmark `results/train-mid4b-reference-checkout-traced-001-admission.json` and the run's frozen inputs/submission record.


## Follow-up, 2026-11-05 — first full-depth update observed, then expert-routing concentration and OOM

Full-depth clean-reference-checkout/tracing job `721795`, run `train-mid4b-reference-checkout-traced-001`, failed after 16 minutes 51 seconds (`FAILED`, full-job exit `1:0`), before its 1500-second supervisor cutoff.
It logged the first completed full 32-layer/64-rank mid-4b update: loss `12.70746`, gradient norm `2.331`, zero skipped/NaN iteration counters and consumed samples `1152`.
The printed first-update interval was `888018.5 ms`; this cold interval includes compilation/startup and diagnostic overhead and is not a warmed throughput measurement or a comparison against the reference steps 10–26.
The subsequent update failed before checkpoint 2, which does not exist.
The frozen global summarizer correctly rejects the run; this is progress evidence, not two-update/global training/checkpoint acceptance.

Retained host/native traces were published for all 64 ranks.
Critical router-metric, rerun-state and quantile-update reductions map to native completed entries on all 64 ranks, so this variant passed the previously observed post-backward wait.
Offline analysis found zero host all-reduce signature mismatch candidates and zero native group/sequence signature mismatch candidates in the retained trace windows.
That is not proof of complete collective correctness, and the successful-small-run calibration still applies: scheduled coalesced/P2P subentries are not independently proof of unfinished GPU work.
The source checkout and tracing both changed relative to the earlier untraced native controls; the result does not uniquely attribute progress to a Megatron source difference or prove that the old wait is fixed without instrumentation.
Only failing ranks 0 and 16 published final actual-module/hash provenance before peer abort; both verified the clean reference revision with optimized reference features disabled.
The other 62 ranks have immutable staged-source inputs and actual source-root caller paths in their traces but lack final module/hash records; do not claim complete all-rank final-source acceptance.

Ranks 0 and 16 raised `torch.OutOfMemoryError` in the second update's weighted expert activation: `fused_bias_sssglu.sssglu`, at `(0.5 + u / (1 + torch.abs(u))) * y_2`, requesting approximately 1.41 GiB.
At rank0 the error reported 121.60 GiB allocated by Torch and 273.08 MiB device-free; rank16 reported 120.36 GiB allocated and 988.25 MiB device-free on 128 GiB devices.
`expandable_segments:True` was already enabled; a generic allocator-switch retry is not supported by this evidence.
Torch allocator accounting excludes CPU/compiler/tmpfs and other-library allocations sharing MI300A HBM.

The all-to-all metadata provides a stronger memory lead than the allocation site alone.
After the first-update log, selected dispatches in EP groups `[0..7]` and `[16..23]` delivered all 524288 routed rows to EP rank0 while peer ranks received zero; rank32 also showed this concentration in its own EP group.
Each sending rank supplied 65536 rows (`8192 tokens × top8`), making 524288 the entire eight-rank group's routed volume and eight times the balanced expert-rank share.
The latent dispatch buffer is `[524288,1280]` BF16, or 1.25 GiB; one half of an expert activation at FFN 1440 is 1.40625 GiB, consistent with the failed allocation.
Before the first-update log, retained rank0 and rank16 dispatch maxima were 70560 and 70670 rows rather than 524288.
This establishes severe post-update expert-rank concentration, not its numerical cause: logits, quantile beta/histogram contents and per-expert state were not captured, and no checkpoint 1 was requested.
The first-log timestamp in the routing report is correctly converted from Europe/Zurich CET to UTC; its before/after labels are approximate wall-time phases, not a cross-node causal ordering guarantee.

The next technical gate is a focused reduced numerical diagnostic of router logits/probabilities, quantile beta/histograms and expert-rank loads before and after the first update, with 256 experts/top-8 retained where feasible.
Separately isolate source changes from tracing before describing the original full-depth wait as fixed.
A fused activation may reduce temporary memory but does not establish that eightfold routing concentration is numerically correct; do not hide this evidence with capacity dropping, batch/recipe changes or an unchanged full-size rerun.
No new GPU diagnostic or additional full-size job was submitted after this failure.
Interactive allocation `721530`, the native checkout/image and all frozen successful/failed runs remain preserved.
Full-depth checkpoint/restart, full-shape cache reconciliation, continuous 26-update timing, throughput parity and convergence remain unvalidated.

Machine-readable evidence in benchmark `experiments/apertus2/results/`: `train-mid4b-reference-checkout-traced-721795-failure.json`, `*-collectives.json`, `*-rank-inventory.json` and `*-progress-and-routing.json`.
Raw OOM traces are in the run's `ranks/0000/stage-0.log` and `ranks/0016/stage-0.log`; the first finite update is in `ranks/0063/stage-0.log`.


## Follow-up, 2026-11-05 — user-requested forced-balancing diagnostic

The user requested a forced-load-balancing trial after the full-depth source/tracing variant completed one update and then failed with expert-rank concentration and OOM.
The pinned reference checkout already implements `TransformerConfig.moe_router_force_load_balancing` and its native `RandomSTE` path, but its CLI does not expose the option.
The explicit harness control `probe.force_load_balancing=true` now enables that existing implementation without changing the image or committed Megatron source.
The child wrapper registers `--moe-router-force-load-balancing` through Megatron's supported `extra_args_provider` hook, appends the flag once, and lets the ordinary parsed-args-to-TransformerConfig path propagate it.
The actual parsed-argument dump consequently reports `True`; this is not an unreported mutation of a model configuration after parsing.
The diagnostic separately verifies each constructed router's effective flag and counts actual calls to the original random-logit function by layer.
Frozen global acceptance requires parsed enablement plus real RandomSTE execution on every rank and fresh-process stage, in addition to the existing source, trace and checkpoint/restart gates.
Existing experiments without this opt-in retain their previous behavior, and direct/cache-lifecycle combinations are rejected for this diagnostic.

Only the forced-logit setting changes relative to accepted traced small run `721782` and failed full-depth comparison `721795`.
Quantile balancing, histogram settings and sequence auxiliary loss remain enabled to isolate one setting at a time.
The native RandomSTE branch replaces learned logits with random values and passes gradients through; it encourages statistical balance rather than enforcing exact expert quotas.
This alters training semantics and is not Imanol-equivalent, learned-routing correctness or convergence evidence.
Reference job `708188` explicitly recorded forced balancing `False` and forced biased logits `None`.

Submitted sequential small compatibility job `722139`, run `train-reference-checkout-small-forced-routing-001`: two nodes/eight ranks, PP2/EP2, the accepted tiny geometry, updates1–2/save2 followed by fresh-process load2/updates3–4/save4, a15-minute allocation cap and720-second supervisor cutoff.
It uses `csstaff`/`mi300` without a reservation and preserves interactive allocation `721530`.
All162 CPU tests passed before submission; three additional offline load-analysis tests bring the current suite to165 passing tests.
Initial small-run evidence already shows parsed enablement and real random-logit execution; full-job and restart acceptance remain pending.

Prepared but did not submit `train-mid4b-reference-checkout-forced-routing-001`: the same full32-layer/64-rank PP2/EP8/DP32/EDP4, two-update/save2 protocol as `721795`, with a30-minute allocation cap and1500-second probe cutoff.
Its resolved configuration differs only by `probe.force_load_balancing=true`; image, clean source artifact, quantile/training recipe, ordinary all-to-all and native backend settings are unchanged.
The forced-routing/runtime implementation is byte-identical to that currently being small-tested.
Submission remains gated on small actual-path/source/trace/save/restart acceptance, observed loads and sequential admission.

A separate offline `scripts/analyze-routing-loads.py` measures retained forward-dispatch receive shapes by native EP-group sequence and verifies complete-group send/receive volume conservation, without reading GPU tensor contents.
It ignores probability/combine/backward collectives, distinguishes partial trace windows from zero peer load and reports measured concentration independently of the requested flag.
On prior failed `721795`, it identifies5520 complete dispatch groups, five partial groups, no complete-group volume mismatch and a worst concentration factor8.0.
On accepted unforced small `721782` stage0, it finds12 complete groups, no partial/volume mismatches and a maximum ratio1.16796875.
These provide calibrated comparison evidence, not a promised balance threshold for the new experiment.
Evidence: benchmark `results/train-reference-checkout-small-forced-routing-001-admission.json`, `results/train-mid4b-reference-checkout-forced-routing-001-preparation.json` and the prior runs' `*-routing-loads.json` reports.


## Follow-up, 2026-11-05 — small forced-routing restart accepted; full-depth trial submitted

User-requested forced-routing small job `722139`, run `train-reference-checkout-small-forced-routing-001`, completed successfully in 9 minutes 12 seconds (`COMPLETED`, full-job exit `0:0`).
The run's frozen global summarizer accepted all eight ranks and both fresh-process stages: updates 1–4, checkpoints 2/4, resume from checkpoint 2, actual clean-reference module provenance, native collective tracing and actual forced-logit execution.
The actual parsed flag was true and per-router observations confirmed the original RandomSTE path ran with the effective model flag enabled.
Quantile balancing plus sequence auxiliary loss, histogram estimation and all prior native-backend controls remained selected.
This is synthetic-routing compatibility/restart acceptance, not learned-routing correctness, exact balancing quotas or Imanol-equivalent training/convergence.

Independent forward-dispatch shape analysis found 12 complete EP-group dispatches in stage0 and14 in stage1, with no partial groups or send/receive volume mismatches.
Worst received-load/balanced-share ratios were1.04296875 and1.0859375 respectively.
Both stage collective analyses found zero host all-reduce and zero native group/sequence signature mismatch candidates in retained evidence.
This validates actual observed load behavior at the small8-expert/top2 geometry; full256-expert/top8 behavior still requires observation.
Evidence: benchmark `results/train-reference-checkout-small-forced-routing-722139.json` and its stage0/1 `*-routing-loads.json` and `*-collectives.json` reports.

After those gates, immutable-input checks and sequential admission, submitted `722153`, run `train-mid4b-reference-checkout-forced-routing-001`.
It retains the previous full-depth comparison's16 nodes/64 ranks,32 layers, sequence8192,256 experts/top8, PP2/EP8/DP32/EDP4, GBS1152/MBS1 and36 microbatches per update.
The target remains two cold updates and checkpoint2, with a30-minute allocation cap and1500-second supervisor cutoff, using `csstaff`/`mi300` without a reservation.
The only resolved configuration difference from failed `721795` is `probe.force_load_balancing=true`; clean source, image, quantile/auxiliary balancing, learning recipe, native TE grouped GEMM/heuristic one, ATen router fallback and collective tracing are unchanged.
The forced-routing and runtime implementation is byte-identical to the accepted small job, with no source patches or image rebuild.
All165 CPU tests pass, and preserved interactive allocation `721530` was the only existing allocation at admission.
No result from this full-depth forced-routing trial is yet accepted; do not infer memory safety, checkpoint success or throughput from the small pass.
The eventual diagnostic result must remain distinct from Imanol's executed reference, which had forced balancing disabled.
Evidence: benchmark `results/train-mid4b-reference-checkout-forced-routing-001-admission.json` and the run's frozen source/configuration/submission records.


## Follow-up, 2026-11-05 — full forced-routing loads balanced, but CXI/RCCL metric collective failed

User-requested full-depth forced-routing job `722153`, run `train-mid4b-reference-checkout-forced-routing-001`, failed after 10 minutes 4 seconds (`FAILED`, full-job exit `1:0`), before the supervisor cutoff.
It reported no completed optimizer update and produced no checkpoint2; its frozen global summarizer rejects the run.
No `OutOfMemoryError` appears in the retained training logs.
The observed load result is nevertheless positive and separate from job acceptance: strict forward-only tracing finds4464 complete EP-group dispatches, no partial groups or volume mismatches, and a peak expert-rank load/balanced-share ratio of1.0125885009765625, compared with8.0 in the previous unforced run.
4464 equals31 MoE layers ×36 microbatches ×4 EP groups per relevant pipeline stage, covering the first update's full forward-dispatch count.
This establishes near-balanced routing during that first forward/backward window, not after an optimizer update.
Because no update completed, it does NOT yet establish that forced logits prevent the previous post-update routing collapse or second-update OOM.

The first explicitly reported task error was rank40 on `nid002764`, preceded by six `cxil_map: write error` lines.
It raised `torch.distributed.DistBackendError` / `ncclInternalError` while entering `finalize_model_grads._log_global_router_metrics`, line463, with reported NCCL/RCCL version2.30.4.
The failing operation was first-use sequence0 of group266, ranks32–63, SUM of a BF16 `[16,256]` tensor (8192 bytes).
All32 PP1 ranks entered that same group/sequence with matching shape, dtype and reduction operation; no obvious payload mismatch was found there.
The PP0 counterpart group265 had29 retained enter records, with ranks0,16,24 not yet at that global-metric call.
All64 ranks published host/native traces, with zero host or native signature mismatch candidates in the retained windows; those negative findings do not prove complete ordering, stream or network correctness.
Rank40's failure provenance reported approximately51.82GiB peak Torch allocation and68.47GiB device-free, so this particular error is not supported as a device-capacity OOM.
Other CPU/tmpfs/library HBM use and CXI mapping/registration resources remain separate from device allocator capacity.
The trailing `ncclAllToAllv is deprecated` text is retained error context, not an established cause of the internal error.
The exact CXI errno/resource/hardware/driver or communication-library cause remains unestablished.

Actual forced-path first-use markers cover all MoE layers on every rank:15 layers per PP0 rank and16 per PP1 rank, with quantile balancing and histogram mode still selected.
Only rank40 published final module/hash and execution-counter provenance before peer abort; its clean reference source was verified and it recorded576 original RandomSTE calls (16 layers ×36 microbatches).
Do not claim full all-rank final-source or two-update acceptance from the first-use markers.
A zero-byte `core_nid002742_138677` artifact was created at peer termination and is preserved; it is not usable evidence of the initiating failure.
Slurm released the diagnostic allocation; preserved interactive allocation `721530` is the only remaining allocation, and no replacement job or image rebuild was initiated.

The next investigation is the CXI/libfabric/RCCL mapping/communicator-initialization error and critical metric-group progress, retaining forced routing as an isolation control rather than repeating the unchanged full-size job.
The original learned-routing numerical issue remains a separate unresolved question.
Full-depth checkpoint/restart, full-shape cache handling and the26-update performance baseline are still unvalidated; synthetic forced routing also remains non-equivalent to Imanol's reference training.
Machine-readable benchmark evidence: `results/train-mid4b-reference-checkout-forced-routing-722153-failure.json`, `*-collectives.json`, `*-rank-inventory.json`, `*-final-frontier.json` and `*-forward-only-loads.json`.
The initiating exception is in the run's `ranks/0040/stage-0.log`; scheduler task-failure ordering is in `slurm-722153.err`.

### Offline dispatch-classification correction

Source review found that the generic `all2all_dispatch` helper is used by backward as well as forward.
The offline load analyzer now requires an actual `token_dispatch` caller frame instead of accepting that helper alone; a dedicated no-GPU regression test was added, and all166 CPU tests pass.
Strict reanalysis preserves every reported peak ratio: unforced full8.0, forced small stages1.04296875/1.0859375, forced full1.0125885009765625.
It removes backward-helper-only extra observations: forced full now has4464 complete/zero partial groups, unforced full5520 complete/zero partial groups, and forced small12 complete groups in each stage rather than12/14.
New `*-forward-only-loads.json` reports retain the corrected classification, while earlier reports and all frozen raw runs remain preserved.
This is an offline analysis correction only; it did not change either GPU experiment or its runtime code.
