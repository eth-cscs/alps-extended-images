# Apertus 2 on MI300A — technical update, 2026-11-03

## Latest status: 2026-11-05

- Clean reference Megatron source passes small training/save/restart and traced restart on the native image (`721759`, `721782`).
- Full-depth comparison `721795` logged one finite update, then failed in update2 with expert-activation OOM and severe EP-rank token concentration; the two-update/checkpoint gate failed.
- The prior critical reductions completed on all64 ranks in the source+trace variant, but source-only causality is unproven; optimized-reference equivalence, full-scale restart/cache handling and the26-update baseline remain unvalidated.
- Forced-routing small save/restart `722139` passed; full-depth forced trial `722153` observed near-balanced loads but failed before its first completed update with CXI mapping-write/RCCL internal error.
- Post-update routing/OOM prevention is not yet established; no new diagnostic allocation is active.
- The image/native checkout and interactive allocation `721530` are preserved; append-only experiment details and evidence follow below.

## Initial status: 2026-11-03

- The native Alps candidate image builds successfully and passes component tests, reduced-model training and checkpoint/restart tests.
- Pipeline-parallel training, memory-backed JIT warmup, cache reconciliation and cross-job cache hydration have passed two-node tests.
- Warmup and measured training start with **identical model, optimizer and RNG state**, verified on all eight ranks in a dedicated test.
- The first **16-node full mid-4b attempt failed during the first warmup backward pass**, in Transformer Engine grouped-GEMM weight gradients.
- No full-shape update completed.
  Full-model capacity, throughput parity and convergence remain unvalidated.
- Current work remains recipe/harness-only.
  No Turbo/T6/optimizer integration port or image-source patch was made during this phase.

## Reference target

The target is Imanol’s executed mid-4b run, job `708188`.

| Setting | Reference |
| --- | --- |
| Model | Approximately 46.4B total / 3.88B active parameters |
| Layers / hidden size | 32 / 2560 |
| Sequence length / vocabulary | 8192 / 200063 |
| Attention heads / query groups | 20 / 10 |
| Experts / top-k | 256 / 8 |
| Expert FFN / latent size | 1440 / 1280 |
| Nodes / GPUs | 16 / 64 |
| Parallelism | TP1, PP2, CP1, EP8, ETP1 |
| Data parallelism / expert data parallelism | DP32 / EDP4 |
| Global / micro batch | 1152 / 1 |
| Microbatches per update | 36 |
| Executed updates | 26 |
| Reference steps 10–26 | Mean 24.085 s; median 23.290 s |
| Throughput derived from that mean | Approximately 6,122 tokens/s/GPU |

This is a short synthetic training reference, not convergence evidence.

## Stack and execution-path findings

The native stack is not a package-for-package replacement for the reference.

| Component | Reference | Native candidate |
| --- | --- | --- |
| ROCm family | 7.2.1 | 10.0.0 |
| PyTorch | 2.10.0, custom Git build | 2.13.0+rocm10.0.0 |
| Triton | 3.6.0 | 3.8.0+git4cff872c |
| Transformer Engine | 2.12.0.dev0+40434cf6 | 2.20.0.dev0+b56021a |
| Megatron source | `d1277d86e0c1…` | `0789b8e42955…`, plus native compatibility patches |
| Emerging-Optimizers | Captured source reports 0.3.1 | 0.2.0 |

Native Torch reports HIP `7.15.26333`; the HIP version string and ROCm image-family label use different version namespaces.

Important execution differences:

- The reference selects **Turbo expert GEMMs, T6 expert exchange, direct AITER attention, custom fusions, optimizer workspaces and K7 batching**.
- Its printed `alltoall` argument does not mean it uses the unmodified native dispatcher.
  Modified Megatron code selects T6 through Beverin flags.
- Native training currently uses **TE expert GEMMs and ordinary all-to-all**.
- Native TE can select embedded AITER attention kernels.
  This is not the reference’s direct AITER/Megatron integration.
- The pinned Megatron upstream trees differ in 45 files, before accounting for native compatibility patches.
- Runtime substitutions also matter: the reference replaces autotuner behavior, Turbo tile selectors, Origami fallback handling and the recompute-extension loader.
- The ordinary Newton–Schulz functions and selected `polar_express` coefficients match between the audited optimizer sources.
  This does not reproduce the reference’s Megatron-side batching, workspace reuse or overlap.
- Installing newer libraries alone would not activate the missing Megatron integrations.

**Matching the recipe on the native stack establishes a native baseline, not reproduction of the optimized reference execution.**

## Candidate image

The canonical base and application image built successfully.

| Item | Result |
| --- | --- |
| Candidate tag suffix | `alps7-dev-64f5ff51e3f53ac123c1a0cec5cfd6a5` |
| Local application image ID | `1de9bd00e3bd…` |
| SquashFS size | 31,528,255,488 bytes |
| SquashFS SHA-256 | `b4f5300c8ecbb69a442e44c0044007a09e56742422b4a867b34eac716f51ea9d` |
| Publication | Not pushed or promoted |

Three candidate fixes addressed native compatibility:

| Change | Purpose |
| --- | --- |
| Make dispatch gradients contiguous before event capture | Handle expanded/non-contiguous gradients safely |
| Router-only ATen GEMM fallback on HIP | Avoid the failing router GEMM path without changing global BLAS policy |
| Prebuild CPU BFD support | Remove first-use BFD compilation from training startup |

These are not ports of the reference optimization series.

Additional validation:

- Image/repository suite: **264 passed, one skipped** at the candidate-build stage.
- Three built-image CPU regressions passed.
- Inspection of 27 ELF libraries found no CUDA dependencies.
- Real Podman cache tests verified software-layer reuse, metadata-label refresh and component-input invalidation.
- Enroot export returned nonzero during cleanup.
  The completed artifact independently passed a full filesystem read, checksum and identity checks.

## Recipe corrections

The earlier proxy used functionality-test defaults, not the reference training recipe.

| Setting | Earlier native proxy | Reference / explicit recipe |
| --- | --- | --- |
| Router probability scaling | Unset | 2.5 |
| Adam β₂ | 0.999 | 0.95 |
| Initialization standard deviation | 0.02 | 0.0197642 |
| Minimum-LR mode | Relative | Absolute |
| LR / matrix LR / gains LR | 0.0001 / 0.001 / 0.0001 | 0.00426 / 0.0213 / 0.00426 |
| LR schedule | Constant | Linear; warmup 718, decay 71829 updates |
| Distributed timeout | 5 minutes | 20 minutes |

Adam β₁ (`0.9`) and rotary base (`10000`) already matched.

Two compatibility issues appeared when enabling the explicit recipe:

- **Attention selection:** native Megatron requires all three TE attention-selection flags to be enabled for `attention_backend=auto`.
  The inherited smoke environment disabled flash and unfused attention.
  The configuration was corrected and contradictory settings are now rejected before submission.
- **Gradient-accumulation fusion:** Apex JIT compilation fails because `max` is undeclared in two hipified gradient source files.
  The image was not patched.
  Tests now use an explicitly named **no-gradient-fusion native recipe**, retaining the other selected reference settings.

## Experiment results

| Job | Configuration / purpose | Result |
| --- | --- | --- |
| `716421` | One node, four ranks; TE, router, KDA, convolution and Muon components | Passed, including the image-default router |
| `716422` | Two nodes, eight ranks; UCCL component, gradient and lifetime checks | Passed, including post-destroy RCCL |
| `716423` | Two-node tiny hybrid training | Passed updates 1–4 with checkpoint/resume from 2 |
| `716427` | Two-node real-width proxy: four layers, sequence 512, vocabulary 1024 | Passed updates 1–4 and restart without a run-local router override |
| `716590` | Two-node tiny PP2/EP2 training | Passed checkpoint ownership checks and restart from 2 |
| `716597` | Tiny explicit-reference recipe | Failed before training because of contradictory auto-attention environment settings |
| `716598` | Corrected auto-attention configuration | Failed on first-update Apex gradient-fusion compilation |
| `716600` | Tiny named no-gradient-fusion recipe, PP2/EP2 | Passed six updates with checkpoint/resume from 3 |
| `716596` | First distributed warmup/cache test | Warmup completed; reconciliation refused a divergent host helper; later audit found missing `.source` coverage |
| `716601` | Cache retry | Agent-cancelled after identifying that the filter could omit every GPU compilation group |
| `716602` | Corrected warmup, gather, reconciliation, publication and training | Passed on all eight ranks, including restart |
| `716605` | Independent read-only cache hydration, without another warmup | Passed on all eight ranks, including restart |
| `716607` | Warmup plus exact initialization fingerprints and native recipe | Passed exact model/optimizer/RNG comparison and six-update/restart acceptance |
| `716613` | Full mid-4b geometry, 16 nodes / 64 ranks | Failed during first warmup backward pass; no completed update |

UCCL component success does **not** resolve the earlier full-model UCCL first-forward-combine crash at host address `0x18`.
Ordinary all-to-all remains the established training path.
The successful reference used T6, not UCCL.

## Harness and checkpoint changes

The harness now supports:

- Separate model, recipe and experiment configurations.
- PP-aware topology and batch validation:
  - DP is world size divided by PP.
  - EP divides DP.
  - Batch arithmetic uses DP, not world size.
- Configurable continuous and checkpoint/restart windows.
- Actual native MPU coordinates and checkpoint paths captured inside each training child.
- Global checkpoint ownership validation across PP, DP and EP.
- Frozen source/configuration snapshots, EDF, commands, image identity and runtime provenance.

A key checkpoint finding:

- Optimizer sidecars are named by **data-parallel rank**, not global rank.
- The same basename can occur in different pipeline/expert directories.
- Validation now uses complete native paths and global ownership coverage.

The current GPU-free harness suite passes **100 tests**.
These are separate from GPU acceptance.

## Memory-backed warmup and cache results

The workflow follows the pattern reviewed in `boeschf/alps-vllm-benchmarks`, pinned at `6a0a68d511a8…`.

- Compilation, scratch, archive assembly and reconciliation use verified executable tmpfs.
- Every rank has a private mutable cache.
- Warmup uses fresh training children with separate scratch and no checkpoint/save/load flags.
- All warmup children exit before capture.
- Rank bundles are gathered and reconciled as whole compilation objects.
- Each measured rank hydrates its own private copy.
- Persistent generations are immutable, compatibility-keyed and published through an atomic pointer update.
- Conflicts are recorded rather than silently combining incompatible files.

Issues found and corrected:

| Finding | Treatment |
| --- | --- |
| Triton group manifests contain absolute rank-local paths | Canonicalize paths in snapshots and relocate during hydration |
| HIP host-helper `.so` files embed temporary source paths and differ across ranks | Exclude host helpers from persistence |
| Native compilation groups require `.source` artifacts | Preserve `.source` alongside compiled artifacts and metadata |
| An empty snapshot could otherwise produce a meaningless “warm” result | Reject empty/no-GPU snapshots and incomplete groups |

Validated results:

- Job `716602` hydrated **280 GPU artifacts / 2,240 files** on every measured rank.
- Job `716605` reused the same persistent generation in a separate allocation without warmup or publication.
- Job `716607` verified exact pre-update model, optimizer and Python/NumPy/Torch/Megatron RNG fingerprints on all eight ranks.
- Cold and conditioned losses were close but not bitwise identical.
  Maximum four-update loss differences were approximately `0.00122` with warmup and `0.00150` with read-only hydration.
  These values alone are not state-isolation proof; the fingerprint test provides that check.

Limits:

- Autotuner state, host helpers, arbitrary extension build trees and accelerator tuning databases are not persisted.
- Cache-conditioned runs can still perform host compilation, process-local autotuning or other JIT work.
- CPU allocations, GPU tensors, compiler scratch and tmpfs share MI300A HBM.
- Byte/file limits, polling and free-space guards are not hard memory quotas.

## First full-shape failure

Job `716613` requested:

- Full 32-layer, sequence-8192, vocabulary-200063 model.
- TP1/PP2/CP1/EP8/ETP1; DP32/EDP4.
- GBS1152/MBS1.
- Two disposable warmup updates, then four measured updates and a checkpoint at 4.
- Native no-gradient-fusion recipe.

Observed:

- All 64 ranks passed executable-tmpfs startup checks across 16 nodes.
- The run failed after approximately **4m36s** during the first warmup backward pass.
- The direct error appeared in 19 rank logs:

```text
TE GroupedLinear.backward
  → general_grouped_gemm
  → te_general_grouped_gemm
  → hipblasLtMatmul
  → HIPBLASLT Error: 6
```

- Other ranks were terminated after the task failure.
- No warmup update, measured update or full-size checkpoint completed.
- Exit 137 is not sufficient evidence of OOM.
  The direct backend failure is established; its underlying cause is not.
- Failure handling also exposed a provenance gap: warmup-child provenance could be lost before result construction.
  The supervisor now retains child provenance on failed warmup as well as success.

## Remaining caveats and next tests

- Passing training jobs still emit process-group-not-explicitly-destroyed warnings despite zero child, supervisor and job exits.
  Explicit teardown hygiene is not claimed fixed.
- The first two printed timing values in each fresh process are not independent steady-state samples.
  The second is a mean over the first two updates because of the pinned logger’s reset behavior.
- Full-size capacity, the 26-update native baseline and performance comparison remain pending.
- Chonk and Megachonk remain separate follow-up targets.

Next diagnostic:

1. Use a two-node, four-layer, sequence-8192 proxy with the same expert dimensions.
   PP1/EP8 gives 32 local experts and an average of approximately 2,048 routed tokens per expert.
2. Compare the existing native grouped-GEMM path with TE’s existing BF16 Triton path:
   `NVTE_USE_GROUPED_GEMM_TRITON=1`.
3. Keep gradient fusion disabled, as required by that backend.
4. Require training/checkpoint acceptance before another full-scale attempt.

The backend comparison is configured but **has not yet been executed**.
It is a scoped runtime diagnostic, not an image patch or global BLAS-policy change.

## Evidence locations

| Evidence | Location |
| --- | --- |
| Harness root | `/iopsstor/scratch/cscs/boeschf/amd-vs-gh200-moe-training/experiments/apertus2/` |
| Frozen runs and logs | `runs/` |
| Acceptance, failure and comparison reports | `results/` |
| Reference/source audits | `reference/` |
| Candidate SquashFS | `runs/image-build-candidate-001/apertus2-megatron-rocm-candidate.sqsh` |
| Full-shape failure report | `results/train-mid4b-capacity-native-716613-failure.json` |
| Pinned TE grouped-GEMM source audit | `reference/full-shape-grouped-gemm-2026-11-03/` |
| Living implementation plan | `Alps-Images/apps/apertus-2/ROCM-TRAINING-PLAN.md` |
| Recipe/reproduction plan | `Alps-Images/apps/apertus-2/MID4B-REPRODUCTION-PLAN.md` |

Historical runs remain preserved.
Changes are uncommitted; nothing has been pushed, promoted or merged.


## Follow-up, 2026-11-03 — grouped-backend controls submitted

| Job | Resources / workload | Grouped-GEMM selection | Status |
| --- | --- | --- | --- |
| `716678` | Two nodes/eight ranks; four layers, sequence 8192, vocabulary 1024; PP1/EP8; GBS8/MBS1 | Native (`NVTE_USE_GROUPED_GEMM_TRITON=0`) | Submitted; result pending |
| `716676` | Identical source/image/workload | Triton (`NVTE_USE_GROUPED_GEMM_TRITON=1`) | Submitted; result pending |

- The only resolved configuration difference is the grouped-backend flag.
- Both runs use the named no-gradient-fusion recipe and test four updates with checkpoint/fresh-process resume from 2.
- The proxy retains 32 local experts and mean 2048 routed tokens per expert, but does not reproduce full PP2 scheduling, depth or vocabulary.
- These are cold correctness/diagnostic controls, not warmed throughput measurements.
- The native TE backend condition and configuration guards have been checked against the pinned source.
- Failure-time child provenance is now retained even when warmup fails before result construction.
- The GPU-free suite passes 100 tests; GPU results for these controls are pending.


## Follow-up, 2026-11-03 — backend controls and existing min/max compatibility fix

### Long-sequence training controls

| Job | Selection | Result |
| --- | --- | --- |
| `716678` | Native TE grouped GEMM | Failed first backward in `GroupedLinear.backward` / `general_grouped_gemm`, with `HIPBLASLT Error: 6` on all eight ranks |
| `716676` | TE grouped Triton | Failed first backward in non-grouped `Linear.backward` / `general_gemm`, with the same error on all eight ranks |

- The failure is reproducible on two nodes with reduced depth/vocabulary; sixteen nodes and full PP2 scheduling are not required to trigger it.
- Grouped Triton is not an end-to-end workaround: the failing call moved to a non-grouped TE linear weight gradient.
- Retained child provenance showed at least 111.55 GiB device memory free for the native control and 109.11 GiB for the Triton control at capture.
- Maximum reported Torch allocator peaks were approximately 12.79 GiB and 14.96 GiB respectively.
- These point-in-time/free and allocator counters do not establish full-run CPU/library/tmpfs peak usage, but they do not support a simple device-OOM explanation for these proxy failures.
- Neither control completed an update or checkpoint.

### Isolated matrix diagnostic

One-node job `716827` removed the model, optimizer and RCCL and tested direct TE `general_gemm` weight-gradient calls against an ATen FP32 reference.
All four cases passed numerical checks.

| Input / output features | Reduction dimension | Output | Requested heuristic selection | Result |
| --- | --- | --- | --- | --- |
| 1280 / 2880 | 128 | BF16 | 0 | Passed |
| 1280 / 2880 | 2049 | BF16 | 0 | Passed |
| 1280 / 2880 | 2049 | FP32 | 0 | Passed |
| 1280 / 2880 | 2049 | BF16 | 1 | Passed |

- BF16 relative L2 errors were approximately 0.00166; the FP32-output error was approximately 4.54e-7.
- These isolated shapes did not reproduce the training failure and do not establish an algorithm-selection or dtype workaround.
- The next bounded replay enables native GEMM configuration logging around the failing training call to capture its actual dimensions/layout/dtypes.

### Apex JIT `max` investigation, following the vLLM ROCm build

- The vLLM ROCm build already addresses missing global integer `min/max` overloads in `Alps-Images/apps/vllm/patches/rocm/uccl/0004-rocm-restore-device-math-overloads.patch`.
- The ROCm base separately uses an `<algorithm>` forced include with `using std::min/max` for rccl-tests.
- These fixes are scoped to their compilation targets; neither automatically fixes Apex.
- The installed candidate contains Apex `1.13.0+rocm10.0.0`.
- Its five failing calls are host-side workspace-size comparisons in two source variants, not device kernels.
- Copies of the canonical and hipified sources were changed to include `<algorithm>` and use `std::max<uint64_t>(...)`.
- Both copied HIP translation units passed `hipcc -fsyntax-only` for host/gfx942 in verified executable tmpfs.
- The installed package and image were not changed.
- Linking, JIT loading and numerical behavior of the patched extension remain untested; gradient fusion is still disabled in the current training recipe.
- This syntax fix is separate from the TE `HIPBLASLT Error: 6` investigation.

Evidence: harness `reference/apex-minmax-compat-2026-11-03/` contains original/patched source snapshots, hashes, compiler commands/output, findings and an unapplied proposed patch.
The GPU-free harness suite now passes 102 tests.


## Follow-up, 2026-11-03 — failing training GEMM configuration captured

- Logged replay `716828` reproduced the first-backward error on all eight ranks.
- Each log ended with the same non-grouped NT weight-gradient configuration before `HIPBLASLT Error: 6`:
  - `m=2560`, `n=1280`, `k=8192`.
  - BF16 inputs and output; gradient mode; no bias, GELU or FP8.
  - `alpha=1`, `beta=0`.
- This differs from the standalone 1280-input / 2880-output matrices that passed in job `716827`.
- A fresh one-node diagnostic replays the captured geometry with a sequence-512 control, sequence-8192 BF16/FP32 output controls and an alternate requested heuristic selection.
- It is a matrix-only diagnostic with no model, optimizer or RCCL; the result is pending.
- The GPU-free suite passes 103 tests.


## Follow-up, 2026-11-03 — exact matrix replay reproduces default-heuristic failure

One-node job `716844` replayed the captured non-grouped NT geometry without model, optimizer or RCCL.

| Input / output features | Reduction dimension | Output | Requested heuristic selection | Result |
| --- | --- | --- | --- | --- |
| 2560 / 1280 | 512 | BF16 | 0 | Passed |
| 2560 / 1280 | 8192 | BF16 | 0 | Failed with `HIPBLASLT Error: 6` |
| 2560 / 1280 | 8192 | FP32 | 0 | Passed; relative L2 error about 1.08e-6 |
| 2560 / 1280 | 8192 | BF16 | 1 | Passed; relative L2 error about 0.00166 |

- This reproduces the default long-reduction BF16 failure independently of training and gives an alternate-heuristic workaround candidate for that geometry.
- It does not yet establish a workaround for every training GEMM or full-model capacity.
- A new two-node sequence-8192 training/restart control uses the native grouped backend and the per-run `TE_HIPBLASLT_ALGO_SELECTION=1` setting.
- The harness only permits reviewed selections zero/one and retains the actual value in child provenance.
- GEMM configuration logging is enabled; no image/default BLAS policy change was made.
- Gradient fusion remains disabled; the Apex std::max fix is still an unapplied, syntax-validated proposal.
- The GPU-free suite passes 105 tests.

The native-grouped/heuristic-one end-to-end proxy control is job `716847`; its result is pending.

A second full-shape capacity run is prepared, not submitted, and gated on job `716847`; detailed GEMM tracing is disabled in that full-scale configuration to avoid excessive log growth.


## Follow-up, 2026-11-03 — heuristic-one training/restart accepted; full-shape retry submitted

### Two-node acceptance

- Native grouped-GEMM/heuristic-one job `716847` completed with full-job exit zero.
- The frozen global summary accepted all eight ranks, updates 1–4, checkpoints at 2 and 4, and a fresh-process resume from update 2.
- Both actual child processes on every rank recorded `TE_HIPBLASLT_ALGO_SELECTION=1` and `NVTE_USE_GROUPED_GEMM_TRITON=0`.
- Logged losses were 7.511195, 7.513458, 7.471888 and 7.425287; gradient norms were finite and reported skip/NaN counters were zero.
- Checkpoint model/optimizer ownership was validated globally using the native checkpoint coordinates.
- This is a four-layer, sequence-8192, vocabulary-1024 PP1/EP8 proxy, not full-depth PP2/vocabulary-200063 capacity or throughput evidence.
- Cold startup and the fresh-process restart are not a steady-state timing window; the second logged timer in each process includes its first-update timer.
- Process-group cleanup warnings remain a separate caveat; this result does not claim teardown is fixed.

Evidence: benchmark `results/train-long-proxy-te-algo1-716847.json` and `results/train-long-proxy-te-algo1-716847-notes.json`.

### Full-shape capacity retry

- Submitted `train-mid4b-capacity-native-002` as job `716850` under the existing sixteen-node authorization.
- Resources: 16 exclusive nodes / 64 ranks, `csstaff` / `mi300`, reservation `cscs`, maximum 45 minutes.
- Geometry: 32 layers, sequence 8192, vocabulary 200063, PP2/EP8/DP32/EDP4, GBS1152/MBS1 and 36 microbatches per update.
- Protocol: two disposable warmup updates, tmpfs cache capture/gather/reconciliation, four measured continuous updates and checkpoint at 4.
- The retry retains native grouped GEMM, heuristic one and disabled gradient fusion; detailed GEMM tracing is off to bound log growth.
- Submission used sequential admission after verifying no unfinished harness submission and only interactive allocation `715971` remaining active.
- That interactive allocation and its memory-backed Podman builder store remain untouched.
- Full-shape success still requires a zero full-job exit and frozen all-rank global summary; no update, capacity or performance acceptance is claimed at submission.
- No image/source rebuild or global BLAS-policy change was made.


## Follow-up, 2026-11-03 — reservation bottleneck and unreserved replacement

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


## Follow-up, 2026-11-03 — full-shape cold warmup activity, not yet acceptance

- Job `716859` has 64 live ranks across sixteen unreserved nodes, with executable-tmpfs startup verified and cache misses on every rank.
- At approximately ten minutes, the first warmup update had not yet completed; no retained HIPBLASLt/OOM/traceback error was present at that observation.
- Sampled final-stage logs showed successful attention backward JIT compilation.
- A read-only overlapping step on the existing first node found four live training children at approximately 282–285% lifetime CPU, nonzero but heterogeneous GPU utilization, approximately 73.6–75.4 GiB reported VRAM used per GPU, and approximately 130.1 GiB host MemAvailable.
- These are one-node point-in-time samples, not proof of iteration progress or all-node/peak memory fit.
- Host and GPU figures share MI300A HBM and must not be added as independent capacity.
- AccumulateGrad stream-mismatch and collective-deprecation warnings are retained; neither is being treated as a fatal error or silently declared fixed.
- The user requested continued autonomous monitoring; milestone/error monitoring is active without changing the running configuration or submitting another GPU job.

Evidence: benchmark `results/train-mid4b-capacity-native-003-startup-observation.json`, `results/train-mid4b-capacity-native-003-activity-observation.json` and run-local `activity-first-warmup.json`.


## Follow-up, 2026-11-03 — slow first update and smaller diagnostic prepared

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


## Follow-up, 2026-11-03 — full-shape observation near the diagnostic cutoff

At approximately 34 minutes, all 64 ranks were still live but no warmup update, reconciliation or checkpoint had completed.
Endpoint samples showed nonzero CPU and GPU activity, including 100% reported utilization on several GPUs, but this cannot distinguish productive compute from communication spin/wait.
Cross-process maps/environment inspection was permission denied as well as debugger attachment; no actual stack, exact slow phase or unobserved child environment is being claimed.
The run remains subject to its original 2400-second probe cutoff and 45-minute allocation cap, with no extension or concurrent GPU job.
A frozen failure/timeout report will be retained before proceeding to the prepared two-node/full-vocabulary/Python-stack diagnostic.
Evidence: benchmark `results/train-mid4b-capacity-native-003-slow-warmup-observation.json` and run-local `activity-before-cutoff-r0000.json` / `activity-before-cutoff-r0063.json`.


## Follow-up, 2026-11-03 — full-shape failure is a collective stall; targeted EDP2 control submitted

### Full-shape outcome

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

### Smaller next test selected from the failure

- The previously prepared full-vocabulary EP8 proxy has EDP=1 and cannot exercise an expert-data-parallel collective, so it remains unsubmitted.
- Submitted `train-long-ep4-edp2-stack-001` as job `716927`: two nodes/eight ranks, PP1/EP4/DP8/EDP2, sequence 8192, four layers and vocabulary 1024.
- Each rank has 64 local experts rather than the full run's 32; the reduced topology is a targeted diagnostic, not full-shape acceptance.
- It retains native grouped GEMM, heuristic one, disabled gradient fusion and native overlapping gradient reduction.
- Protocol: two continuous updates, checkpoint at 2, periodic Python stack dumps every 60 seconds, quiet per-GEMM logging, cold private tmpfs caches, a 25-minute probe cutoff and a 30-minute allocation cap.
- Uses `csstaff`/`mi300` without a reservation and sequential admission after the sixteen-node job completed.
- Only the preserved interactive allocation was active at admission; no overlapping GPU training job or image/source change was introduced.
- The GPU-free suite passes 107 tests, including explicit EDP2 topology and diagnostic-watchdog construction.
- Full mid-4b capacity, full checkpoint acceptance, the 26-update baseline and throughput comparison remain blocked.


## Follow-up, 2026-11-03 — EDP2 diagnostic hit an early native signal; single-setting timer control submitted

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


## Follow-up, 2026-11-03 — EDP2 no-timer acceptance and combined PP2/EDP2 control

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


## Follow-up, 2026-11-03 — reduced PP2/EDP2 accepted; EDP4 control and cooperative diagnostics

### PP2 acceptance

- Job `716931` passed with full-job exit zero after 8m36s and frozen global acceptance on all eight ranks.
- The reduced real-width/sequence-8192 PP2/EP2/DP4/EDP2 model completed two updates and checkpoint ownership/contents validation at 2.
- Reported losses were 7.538306 and 7.536028, gradient norms 4.053 and 4.050, with zero skip/NaN counters.
- Actual child provenance retained heuristic one and native grouped GEMM on all ranks.
- This did not reproduce the full-shape collective stall; depth four, vocabulary1024, EDP2/local128experts and one microbatch differ from the full EDP4/local32experts/36-microbatch run.
- No restart, exact-state equality or reference-throughput claim is attached to this two-update diagnostic.

Evidence: benchmark `results/train-long-pp2-ep2-edp2-716931.json` and `results/train-long-pp2-ep2-edp2-716931-notes.json`.

### Opt-in cooperative stack sampling

- Added a harness-only `stack_dump_mode=cooperative` option with a positive diagnostic interval, requiring the instrumented training wrapper.
- It samples at most 32 main-thread Python frames from an ordinary Python daemon thread under the GIL, instead of using `faulthandler.dump_traceback_later`.
- It emits structured `APERTUS2_STACK_SAMPLE` records, has an explicit stop callback and records the actual mode/interval in child provenance.
- The sampler cannot run while another thread holds the GIL, cannot inspect native/GPU stacks and does not by itself prove training progress.
- Historical/default behavior is unchanged unless the cooperative mode is explicitly selected; performance runs with interval zero create no sampler thread.
- CPU unit tests and a real-thread two-sample host check passed; GPU runtime/numerical acceptance is pending.
- This is diagnostic harness work only, not an image-source or compute integration change, and does not establish that the earlier timer caused SIGSEGV.

### Next EDP4 control

Submitted `train-long-ep2-edp4-cooperative-001` as job `716983`, sequentially after `716931` completed.
It uses two nodes/eight ranks, PP1/EP2/DP8/EDP4, four layers, sequence 8192, vocabulary1024 and GBS8/MBS1, so one microbatch per update.
It matches the full run's expert-replica count four but not its PP2/depth/vocabulary/accumulation geometry.
Native grouped GEMM, heuristic one and overlapping gradient reduction remain unchanged; cooperative Python samples are requested every 60 seconds.
The run has a 25-minute probe cutoff / 30-minute allocation cap, `csstaff`/`mi300`, no reservation, and no concurrent GPU training job.
All 114 CPU tests pass; full mid-4b capacity/checkpoint acceptance and the 26-update baseline remain blocked.


## Follow-up, 2026-11-03 — EDP4 acceptance; PP2 with cross-node EDP4 on four nodes

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


## Follow-up, 2026-11-03 — PP2/cross-node EDP4 accepted; accumulation-36 control submitted

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


## Follow-up, 2026-11-03 — accumulation control still in progress; sync source audit

At the approximately seven-minute observation, accumulation-36 job `717038` had all sixteen ranks live, no fatal exception and no completed update yet.
Cooperative samples had moved from compilation into backward execution and the native batched pipeline communication synchronization; this was not yet the earlier post-backward expert-DP watchdog signature.
The prior one-microbatch control also had substantial cold skew, so these Python samples alone are not being called a reproduced deadlock.

A read-only pinned-source comparison checked the reference/native pipeline and gradient synchronization paths while the run continued.
The reference contains an optional `BEVERIN_NO_P2P_SYNC` gate, but that flag is absent from the retained executed `708188` log/environment dump; no enabled reference-path difference or workaround is claimed from its mere presence in source.
The pinned `finalize_model_grads.py` files are identical, including router-metric reductions.
Reference-only gradient-buffer O1 joins belong to its side-stream wgrad integration, not a generic native fix to port into this harness-only phase.
The audit and fingerprints are retained in benchmark `results/p2p-gradient-sync-source-audit.json`; no synchronization, overlap, transport or image policy was changed.
Monitoring remains bounded by the existing 25-minute probe cutoff / 30-minute allocation cap.


## Follow-up, 2026-11-03 — accumulation-36 acceptance; real-vocabulary control submitted

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


## Follow-up, 2026-11-03 — real-vocabulary run pending; eight-layer footprint control prepared

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


## Follow-up, 2026-11-03 — real-vocabulary accumulation accepted; eight-layer footprint run submitted

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


## Follow-up, 2026-11-03 — eight-layer footprint acceptance; eight-node EP8/EDP4 diagnostic

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


## Follow-up, 2026-11-03 — EP8/EDP4 single-stage acceptance; reduced-depth full-rank topology submitted

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


## Follow-up, 2026-11-03 — actual full-topology startup parity verified

All 64 native startup coordinate tuples in reduced-depth job `717130` match the failed full run `716859` at the same global rank: TP/EP/DP/EDP/PP coordinates were checked from actual child logs, not only intended configuration.
Host placement confirms eight expert groups spanning two hosts each, sixteen expert-replica groups spanning four hosts each and 32 pipeline pairs spanning two hosts each.
All 64 runtime roots are backed by executable, writable tmpfs.
This verifies actual startup coordinates and inferred group placement, not a full Torch process-group registry dump or identical collective payload/order/timing; physical nodes and model depth/protocol differ.
Training/checkpoint/full-job acceptance is still pending, and no capacity/root-cause conclusion is drawn from startup parity.
Evidence: benchmark `results/train-four-layer-full-topology-001-startup-parity.json`.


## Follow-up, 2026-11-05 — full-topology acceptance finalized; depth-only full-model test submitted

### Accepted control after the session restart

- Frozen global acceptance for job `717130` is now finalized: full-job exit zero, all 64 ranks passed both updates and checkpoint ownership/contents validation at update 2.
- This control used four layers with the full PP2/EP8/DP32/EDP4 topology, sequence 8192, vocabulary 200063 and 36 microbatches per update.
- Reported losses were 12.67185 and 12.67054, gradient norms 2.355 and 2.353, with zero skip/NaN counters.
- Actual startup coordinates match failed full-depth `716859` rank by rank; EP groups span two hosts, EDP groups four hosts and pipeline pairs two hosts.
- Reduced depth changes collective payload/count/timing and memory; this success does not resolve the earlier full-depth synchronization failure.
- The resumed GPU-free harness suite passes 120 tests, and current harness source hashes exactly match accepted `717130`.

Evidence: benchmark `results/train-four-layer-full-topology-717130.json`, `results/train-four-layer-full-topology-717130-notes.json` and `results/train-four-layer-full-topology-001-startup-parity.json`.

### Controlled full-depth experiment

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


## Colleague briefing, 2026-11-05 — work completed and what remains

Status checked at 2026-11-05 09:04 UTC.

### Plain-language overview

- The goal is to run the full Apertus 2 training setup on the AMD machines and then compare its speed with Imanol's run.
- We built a reusable software image, fixed several software compatibility problems, and checked the affected operations with focused tests.
- Small training tests can complete, save their training state, and restart; saving and restarting lets training continue after a job ends.
- Larger diagnostic tests also completed training steps and saved their state, including a smaller model using all 16 machines and 64 GPUs planned for the full run.
- An earlier full-size attempt became stuck while the GPUs were exchanging and combining training results; we have not yet established the underlying cause.
- We varied model size and communication layout in controlled tests to narrow the problem, rather than repeatedly launching the same full-size job.
- The current test restores the full 32-layer model while keeping the software and all other settings identical to the successful smaller-model control.
- At the latest check, this full-size test was still running after approximately 17 minutes, without a reported fatal error, but had not completed its first training step.
- We therefore cannot yet claim that full-size training works, that it can restart reliably, or that its speed matches Imanol's run.
- Only one training experiment is running at a time; the saved software image survived the session restart, so rebuilding was unnecessary.

### Evidence behind the progress

| Completed work | What it establishes | What it does not establish |
| --- | --- | --- |
| Candidate software image and focused numerical/compatibility checks | A reproducible, usable native AMD stack for the tested paths | Every optional optimization or full-size training correctness |
| Small training, saved-state and fresh-process restart tests | Basic training and restart functionality | Full-size restart or long-run convergence |
| Four-node controls with real vocabulary, 8192-token sequences and 36 microbatches per update | These settings work together at reduced model depth | Full-model memory capacity or full EP8 communication |
| Eight-node, 16-layer test `717115` | Two training updates and saved-state validation with EP8/EDP4 | The combined full-depth PP2 case |
| Sixteen-node, four-layer test `717130` | Two training updates and saved-state validation on all 64 ranks with the full intended communication layout | Full 32-layer capacity or the cause of the earlier failure |
| 120 passing CPU-only harness tests | Submission, diagnostics and acceptance checks pass their automated tests | GPU training success by themselves |

### Current experiment and next decision

Job `721544` is the only active training experiment: 16 nodes / 64 ranks, full 32 layers, sequence 8192, vocabulary 200063, PP2/EP8/DP32/EDP4 and 36 microbatches per update.
It is bounded to two training updates and a saved-state check, with a 40-minute probe cutoff inside a 45-minute allocation.
At the approximately 17-minute check, all 64 ranks were live, no fatal exception or completed update was logged, and Python samples were distributed across backward execution, matrix multiplication and synchronization calls.
Changing Python sample locations is not proof of a completed training step or useful GPU progress.
The previous failure's exact post-backward collective-wait pattern has not been established in this run at this observation.
If the run passes, the next gates are full-size fresh-process restart and full-shape compilation-cache preparation/reuse, followed by the continuous 26-update speed baseline.
If it fails, the retained per-rank diagnostics will determine the next targeted test; no blind large-scale repeat or speculative rebuild is planned.

### Comparison with Imanol's run

We are targeting the same model, batch and training recipe using the native AMD implementation, not yet an identical optimized software path.
Our run still disables gradient fusion and uses native Transformer Engine expert matrix operations and ordinary all-to-all communication rather than Imanol's Turbo/T6/direct-AITER integrations.
A successful native run will provide a baseline; it will not by itself establish matching speed, identical numerical trajectories or production convergence.

Detailed evidence is retained under benchmark `experiments/apertus2/results/` and frozen `experiments/apertus2/runs/`, especially `train-four-layer-full-topology-717130.json`, `train-mid4b-full-depth-controlled-001-admission.json`, `train-mid4b-full-depth-controlled-001-startup-parity.json` and the current run's rank logs.


## Follow-up, 2026-11-05 — full-depth run cancelled on request; communicator diagnostic submitted

### Full-depth outcome and retained evidence

- Cancelled only job `721544` at the user's explicit request after 25m13s; it completed zero training updates and no checkpoint.
- Slurm records `CANCELLED`, not a watchdog-confirmed failure in this run; the supervisor's nonzero exit and rank SIGTERM errors follow that cancellation.
- All 16 nodes were released before admitting another training job; interactive allocation `721530` remains untouched.
- Cooperative samples retained a persistent post-backward split: ranks 0/8/16 at `finalize_model_grads.py:335`, the other 29 PP0 ranks in `_log_global_router_metrics` all-reduce, and all 32 PP1 ranks in the rerun-state global reduction.
- A read-only extraction of the exact installed sources confirms line 335 is `int(sample_count_tensor.item())`, after `finish_grad_sync`.
- A CPU-visible `.item()` wait can reflect outstanding GPU work; it does not establish a defect in that scalar read.
- This resembles the previous full-depth wait pattern but is not an exact replay of its watchdog evidence, nor proof of a transport defect.

Evidence: benchmark `results/train-mid4b-full-depth-controlled-721544-before-cancel.json`, `results/train-mid4b-full-depth-controlled-721544-cancelled.json`, `results/train-mid4b-full-depth-controlled-001-live-source-audit.json` and extracted-source hashes under `reference/router-metric-communicator-2026-11-05/`.

### Targeted hypothesis and opt-in diagnostic

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

### Small runtime validation first

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


## Follow-up, 2026-11-05 — small diagnostic accepted; full-depth first-use comparison submitted

### Small diagnostic acceptance

- Job `721576` passed with full-job exit zero and frozen global acceptance on all eight ranks.
- Both updates completed with finite losses/gradients and zero skip/NaN counters, and checkpoint ownership/contents validation passed at update 2.
- Reported losses were 7.593303 and 7.588262, gradient norms 1.959 and 1.985.
- Actual child provenance confirms successful communicator priming on every rank, two consistent groups of four, and the expected scalar reduction result.
- This establishes real-runtime hook/training/checkpoint functionality, not an improvement over an unprimed matched control or a fix for the full-depth stall.
- The cold first-update PP-stage skew resolved; samples of PP0 backward or gradient synchronization and PP1 rerun-state reduction were not alone proof of a stall.

Evidence: benchmark `results/train-router-communicator-priming-small-721576.json` and `results/train-router-communicator-priming-small-721576-notes.json`.

### Full-depth targeted comparison

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


## Follow-up, 2026-11-05 — reference/native differences around the synchronization blocker

The visible wait is in a shared Megatron code path, but the execution and communication stacks feeding it are not equivalent.

| Area | Imanol's retained reference | Current native baseline | Interpretation |
| --- | --- | --- | --- |
| Gradient finalization and router metrics | `finalize_model_grads.py` SHA-256 `7ec2fc944b4bad74537cf07a752bc4ba8646ad3bfe291908fe2d4e462cd948c4` | Exact same hash in the installed candidate | The scalar reads/router-metric reductions themselves are not a source difference |
| Gradient-buffer synchronization | Reference adds `_o1.join()` in `start_grad_sync` and `finish_grad_sync` for its O1 side-stream wgrad integration | Those joins and the reference O1 integration are absent | Different gradient-readiness/stream dependency behavior; not proof that native is missing a generic barrier |
| Expert compute/communication | Turbo/fused wgrad and selected T6 dispatcher | Native TE grouped GEMM, ordinary all-to-all and disabled gradient fusion | Collective timing and stream dependencies differ despite matched model/batch settings |
| Startup | Selected Beverin synthetic forward/backward prewarm and selected reductions | Cold native execution; current diagnostic primes only the router-metric group | Current priming does not reproduce reference prewarm |
| Distributed/runtime libraries | Custom Torch `2.10.0+git94c6e04`, ROCm 7.2.1 family, scratch gfx942 RCCL rebuild; logs report RCCL 2.27.7 and OFI plugin 1.4.0 | Actual candidate child reports Torch `2.13.0+rocm10.0.0`, SDK-wheel RCCL and image Alps network libraries | Significant lower-stack differences; native RCCL's complete runtime version is not established by the earlier audit |

The reference pipeline source also contains an optional `BEVERIN_NO_P2P_SYNC` gate, but it is absent from the retained executed reference log/environment; no enabled skip-sync path is inferred from source presence alone.
Native explicitly sets `NCCL_P2P_DISABLE=1` per run; the flag was absent from the captured reference pre-import environment, which is not a complete final child-environment proof.
The evidence therefore supports investigating the Megatron/PyTorch/RCCL interaction and stream/collective ordering, not attributing the failure to the common router-metric source or blindly copying the reference O1 joins.
A simple metric-communicator prime tests one first-use hypothesis only; it does not align the full reference synchronization stack.

Evidence: benchmark `results/p2p-gradient-sync-source-audit.json`, `results/train-mid4b-full-depth-controlled-001-live-source-audit.json`, `reference/stack-comparison-2026-11-03/`, actual accepted `721576` child provenance, and image-repository `STACK-DIFFERENCES.md`.


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
