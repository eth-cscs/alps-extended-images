# Next step: full mid-4b recipe reconciliation

## Status and scope

The built native candidate passed all four planned component/proxy acceptance checks in jobs `716421`, `716422`, `716423` and `716427`.
Those passes do not reproduce reference job `708188`.
The initial analysis below is retained as historical context.
The user subsequently selected **recipe/harness-only implementation**, authorized autonomous continuation including an eventual **16-node test**, and requested verified in-memory warmup with gather/merge/reconciliation following `boeschf/alps-vllm-benchmarks`.
Recipe/PP2/checkpoint support and a bounded compiled-Triton cache protocol are now implemented and CPU-tested in the external training harness, and tiny PP2/native-recipe plus GPU-cache lifecycle acceptance has passed; exact initialization-state equality is accepted at tiny geometry, and full-shape validation remains pending.
Turbo/AITER installation and Megatron/optimizer/UCCL image-source ports remain outside this scope.

Primary evidence is the actual parsed-argument output in reference `708188` and candidate proxy `716427`, not assumptions about omitted flags.
Machine-readable findings and source/log fingerprints are in benchmark `experiments/apertus2/reference/mid4b-reproduction-plan-2026-11-03/`.

## Confirmed training-semantic differences

| Setting | Executed reference | Accepted candidate proxy | Required treatment |
| --- | --- | --- | --- |
| Router top-k probability scaling | `2.5` | `None` | Explicitly select `2.5`; native source applies no extra multiplication when unset |
| Adam beta1 | `0.9` | `0.9` | Already matches, but make the full recipe explicit |
| Adam beta2 | `0.95` | `0.999` | Explicitly select `0.95` |
| Initialization std | `0.0197642` | `0.02` | Explicitly select the reference value |
| Rotary base | `10000` | `10000` | No observed semantic difference; explicit selection improves auditability |
| Custom-group minimum-LR mode | `absolute` | `relative` | Explicitly select `absolute` |
| LR / matrix LR / gains LR | `0.00426` / `0.0213` / `0.00426` | `0.0001` / `0.001` / `0.0001` | Replace functionality-test settings with a separate reference recipe |
| Schedule | Linear; warmup 718; decay 71829 iterations | Constant | Record the actual reference schedule even for a short initial reproduction window |
| Attention backend | `auto` | `fused` | Review with the selected attention implementation; flags alone do not activate missing AITER code |
| Gradient accumulation fusion | `True` | `False` | Review native/reference implementation compatibility rather than treating as equivalent |

The router difference changes routed probabilities, not merely printed metadata.
The matching `polar_express` coefficients and ordinary Newton–Schulz functions do not make the entire optimizer recipe equivalent when hyperparameters and Megatron batching/workspaces differ.
Keep the successful smoke/proxy recipes unchanged as regression fixtures instead of silently repurposing them.

## Full reference geometry and layout

The target is 32 layers, hidden size 2560, 20 attention heads, ten query groups, dense FFN 12960, 256 experts/top-8, expert FFN 1440, latent size 1280 and shared expert FFN 1440.
Sequence length and maximum positions are 8192; NullTokenizer vocabulary is 200063 with divisibility 128, yielding 200064 padded vocabulary entries.
The attention pattern is 3:1 KDA/global attention over eight groups of four layers; the first layer is dense and the remaining 31 are MoE.
The reference uses TP1/PP2/CP1/EP8/ETP1, MBS1, GBS1152, selective `moe_act layernorm` recomputation and 26 logged updates.

The existing model-geometry validator already accepts the full numeric dimensions.
The blocker is not a four-layer/512-token schema limit: the command builder and acceptance path still hardcode the smoke execution policy.
A GPU-free construction with full dimensions confirms it still emits PP1, four training iterations and the functionality-test learning rates.

## Harness changes required before a faithful recipe can run

1. Expose PP and validate it against rank count, model depth, DP and EP.
   For the supported TP1/CP1/ETP1 case, DP is `world / PP`, EP must divide DP, and GBS must be divisible by DP × MBS rather than treating world size as DP.
2. Put initialization, router scaling, optimizer hyperparameters and scheduling in explicit versionable recipe settings.
   Preserve intentionally disabled fusions/backend deviations as named choices rather than undocumented defaults.
3. Separate a reference-length training/performance run from the fixed four-update restart probe.
   The current iteration parser expects `/ 4`, the stage loop expects updates 1–2 then 3–4, and the summary requires exactly those windows.
4. Make checkpoint inspection PP/EP-aware before running full PP2 restart acceptance.
   Native checkpointing names sidecars by **data-parallel rank**, while the directory identifies tensor/pipeline/expert ranks.
   The current inspector assumes sidecar suffixes are globally unique across all world ranks, which only works for its existing PP1 scope.
   Record actual child parallel coordinates and inspect the rank’s actual resume files; do not load every shard on every rank or infer ownership only from basename/global rank.
5. Add GPU-free tests for topology, recipe construction, iteration windows and checkpoint ownership while retaining all existing regression behavior.
   A separate tiny two-node PP2/EP2 restart can validate the new harness path within the standing short-job resource agreement; it is not the full reference configuration.

The original analysis made no implementation changes.
The later scoped implementation adds these mechanics while preserving frozen historical runs; it must pass the tiny PP2/EP2 restart and distributed warmup/cache tests before larger-run claims.

## Resources and superseding authorization

With four ranks per node, PP2/EP8 and MBS1:

| Nodes | Ranks | DP | Expert DP | Microbatches/iteration at GBS1152 | Meaning |
| --- | --- | --- | --- | --- | --- |
| 4 | 16 | 8 | 1 | 144 | Minimum rank count for PP2/EP8; reduced scale and EDP1 memory results can mislead |
| 8 | 32 | 16 | 2 | 72 | Reduced-scale native baseline, not the reference throughput layout |
| 16 | 64 | 32 | 4 | 36 | Exact reference scale |

The earlier planning-only resource blocker is superseded by the user’s explicit authorization for an eventual 16-node test.
Do not request the same authorization again solely because sixteen exceeds the earlier eight-node limit.
Technical readiness, accurate accounting for the existing interactive allocation and sequential admission for the larger baseline still apply; do not cancel the user’s allocation.
Do not use CPU expert offload as an MI300A memory escape hatch, and retain shared CPU/GPU/tmpfs HBM accounting.

## Implementation fidelity remains a separate gate

A matched model/training recipe on native TE and ordinary all-to-all establishes a **native baseline**, not reproduction of the reference’s selected Turbo/T6/AITER/fusion/workspace execution.
Reference-pin Turbo reproduction and modernization onto a newer Turbo release are distinct choices because the Megatron integration uses private APIs, custom pointer/wgrad kernels and runtime selector replacements.
Keep the existing source/runtime/UCCL audit as the dependency/feature checklist; a library installation or environment flag cannot create missing Megatron code.
The old hipBLASLt solution indexes and Triton 3.6 tuning files are not validated native defaults.

The approved implementation scope is recipe/PP2/checkpoint support and training warmup/cache orchestration.
Image-source changes remain paused; the 16-node baseline is resource-authorized but still gated on technical validation.
Reference recipe fusions and full-scale capacity/performance are not established by the CPU tests or candidate proxy acceptance.


## Resumed harness implementation: 2026-11-03

The external harness now has explicit versioned model/recipe/experiment configurations, PP-aware DP batch math, configurable continuous/restart windows and actual native checkpoint-coordinate capture.
The CPU suite currently passes 90 tests, including duplicate-command-flag regressions that caught incomplete early builder edits.
The full recipe also explicitly matches reference distributed timeout/evaluation settings: 20 minutes, evaluation iterations zero and interval 100000.
The tiny two-node/eight-rank PP2/EP2 checkpoint/restart was submitted as job `716590`, with the accepted candidate SquashFS, `csstaff`/`mi300`, reservation `cscs` and the existing one-node interactive allocation left intact.
It passed all eight ranks in 8m57s, with checkpoint at 2/fresh-process resume through 4 and full PP/EP-qualified ownership coverage.
Four process-group cleanup warnings remain despite zero child/supervisor/job exits; benchmark result and companion notes preserve that caveat.
No full mid-4b run has been submitted.

The opt-in cache protocol uses reviewed upstream revision `6a0a68d511a8a5728ca8667e3b22ebf98e9e9181` as a design reference, not serving-specific settings.
Live compilation, archive assembly and reconciliation use verified executable tmpfs; shared storage receives immutable snapshots and rendezvous/diagnostic records.
Warmup executes full-shape updates in fresh children with separate scratch/caches and no save/load/checkpoint flags, so measured children do not inherit model/optimizer/RNG state.
Every rank quiesces, captures compiled Triton artifacts, gathers its bundle, joins a whole-object reconciliation and receives a private hydrated copy before measurement.
Native Triton group manifests embed absolute rank paths, which must be canonicalized in snapshots and relocated during hydration.
Whole-object conflicts stop the warmed run instead of silently splicing incompatible group/binary versions.
Persistent publication uses three-way base/current/job conflict checks, immutable generations and an atomic CURRENT switch; conflicts retain the previous pointer.
Only compiled Triton artifacts are persisted initially; autotune state, locks, links, scratch/build trees and other accelerator databases are excluded.
Consequently cache-conditioned measurements may still include process-local autotuning or non-Triton compilation and must not be called zero-JIT steady state.

The compiled-GPU cache lifecycle passed tiny distributed warmup/restart and independent cross-job hydration in jobs `716602`/`716605`.
Exact initialization-state and full-shape validation remain separate gates before warmed performance claims.
CPU/GPU allocations, private caches, compiler scratch, archive/extraction buffers and tmpfs all compete for MI300A HBM; polling and per-operation guards are not a hard memory quota.
Keep warmup/startup, cold updates, warmed updates and the optimized reference steady window separately reported.


### Subsequent tiny-recipe and cache checks

Job `716596` is the two-node/eight-rank PP2/EP2 disposable-warmup/cache protocol test with the legacy conservative recipe.
The independent tiny reference-recipe job `716597` failed before training because `attention_backend=auto` inherited smoke-only `NVTE_FLASH_ATTN=0` and `NVTE_UNFUSED_ATTN=0`.
Native Megatron requires all three TE attention-selection flags to be one for `auto`; the reference experiment now explicitly matches that contract and validation rejects contradictions before submission.
Frozen failed inputs/logs remain intact, with a separate failure report under benchmark `results/train-reference-tiny-candidate-716597-failure.json`.
A new two-node retry uses the same tiny geometry and six-update/restart-from-3 protocol; its result is pending.
No image/source port was used to address this configuration failure.


### Latest native limitations and capture fixes

The corrected auto-attention tiny run `716598` reached first-update execution, then failed compiling Apex gradient fusion because `max` is undeclared in two hipified source files.
The harness now exposes independent `gradient_accumulation_fusion` control; the named no-gradient-fusion native deviation is under test in two-node job `716600`, with the remaining reference choices preserved.
No image/Apex/Megatron source patch was made.

Cache run `716596` gathered all eight quiescent warmup bundles but refused a divergent host HIP helper containing embedded rank-local source paths.
Its retained manifests also revealed missing `.source` coverage, causing every GPU compilation group to be skipped.
Agent-owned retry `716601` was stopped rather than risk an empty-cache warmed claim; user allocation `715971` and independent recipe job `716600` were untouched.
The corrected capture policy includes required `.source`, excludes host `.so` helpers and explicitly rejects empty/no-GPU or incomplete warmup snapshots.
Fresh two-node cache job `716602` is pending, and the CPU suite now passes 93 tests.
These are technical validation gates before the already-authorized 16-node native baseline, not a request for further resource permission.


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
