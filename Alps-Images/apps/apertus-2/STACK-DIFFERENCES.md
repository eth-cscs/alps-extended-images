# Imanol reference versus the native Apertus 2 ROCm stack

## Primus image progress: 2026-11-06

### Current blocking findings

The source-corrected Primus 26.3 base passed GPU/library checks, strict Enroot export, and fresh two-node/eight-rank validation 724239: MIOpen passed, RCCL completed 18/18 sizes with zero wrong values, and GPU-OSU completed 18/18 sizes without a preload shim.
Its exact-config Docker backup preserves image ID `5e4d3a6a1b0411c0d862fa4b463173f5845d7d7a24c38405378c6e78adbb5adf`.
The checker-corrected 26.3 app compiled as `483272f80fd87d6a2ae70319dd6edaad3c683f256601bf24e1b8e704718bd0b6`, preserving 11 vendor packages / 4293 files, but its strict AITER backward comparison failed on 3/32768 elements; no accepted app export followed.
Recovery 724549 found the old node-local store absent and is restoring the exact parent before rebuilding the same frozen app content in `26.3/app-attempt-005`.
The rebuilt app will be archived before an explicitly external attention diagnostic adds FP64-reference and BF16-math-control measurements while retaining the original strict rejection.
No 26.3 attention tolerance or release policy has been changed.

The same-content Primus 26.8 parent rebuilt as `c7dc20175cdbab02421542630103e88973fbfc9649e7d1e70cf041c1dfd647c8`, preserving 14 vendor packages / 5164 files and MPI5.
It passed GPU/library checks and strict Enroot export with SHA-256 `d29af3ac02584a65a1b62477146f16cd93b48ccb0e95b004f09e2a09f36fecf8`; its exact-config Docker backup is preserved.
Fresh validation 724548 passed runtime tests but its reviewer lacked INFO-level transport evidence; that result and the reviewer failure are retained in `26.8/validation-002`.
Repeat 724551 passed MIOpen, RCCL 18/18 with zero wrong values and AWS Libfabric logging, and GPU-OSU 18/18, establishing the rebuilt image's fresh base gate.
Recovery 724550 found the old node-local store absent, restored the exact parent ID, verified frozen inputs and the full base gate, and received the corrected child handoff.
Both recovery builders archive compiled children before GPU checks, so a numerical rejection will not prevent exact-config preservation.
The corrected 26.8 child has now compiled as `7ec22d34344ec3001cfd5d4b6ac3485eae6e86d6cfbadc5d698a3afa8b8230ec`, passed the packaged GPU stack check, and passed strict export with SHA-256 `faa88a6ed8848f86374302cdb78f0583f0e1d88929dafd09c1d617005ccf7726`.
Its exact-config Docker archive is preserved; independent cold-stack validation is queued for admission after the active numerical diagnostic.

The checker now reads native reload evidence from rank 0 independently of rank 7 metrics and retains all checkpoint ownership, optimizer/RNG ancestry, source identity and finite-update guards.
Its tests passed 67 focused cases and 333 total cases with one skipped.
Frozen `app-inputs-011` identifies canonical app suffixes `c736a041810bb7fe7dbcd420f0794861` (26.3) and `a00cdf813c7373a5f69792117ce4c3e4` (26.8); previous frozen snapshots and failed evidence remain intact.

26.8 full-path diagnostic 724245 reproduced the first post-reload NaN and preserved eight checksummed raw gradient snapshots.
Only rank 2 had nonfinite local inputs: two FP32 128×128 matrices contained 2304 and 256 NaNs, with remaining finite magnitudes around 10³⁴.
Their full layout matches expert FC2 weight gradients for decoder layers 2/3 in a prior named healthy capture; those names remain inferred rather than directly recorded in the failed call.
This establishes corrupted gradient inputs in that run, not their producer or root cause.
Read-only source analysis shows selective `moe_act` recomputation sets FC2 `_o1_nodefer`, so the deferred FC2 queue is not the expected path for this configuration; immediate side-stream work and recomputed storage lifetime need investigation.
Standalone GPU probe 724554 reproduced a storage-lifetime defect: all four custom-alias-only `record_stream` cases reused storage before the side-stream read, corrupting every output element.
All twelve native-storage, true-owner-recording and retained-owner controls passed.
The exact Torch commit's allocator skips stream recording for custom-deleter DataPtrs, matching the reference storage-sharing extension's pointer type.
Full-path diagnostic pair 724559 completed finite updates 1–4 and strict external save2/load2/save4 checks in both the baseline and retained-FC2-input trials.
Each trial produced 16 lifetime reports and exercised 144 immediate FC2 calls; peak held inputs were zero and three respectively.
The intended control executed, but the healthy baseline means this pair does not establish a causal training fix.
Actual-kernel probe 724576 then reproduced corruption in all four custom-alias-only cases using the image’s unchanged `vk_accum` FC2 wgrad kernel with controlled delayed execution and storage reuse.
All 262144 output elements per failing case matched the replacement-input arithmetic, while all twelve native-storage, true-owner-recording and retained-owner controls were exact.
This confirms the lifetime defect can corrupt the actual wgrad kernel; it does not establish that the original training failure used this schedule.
Bounded full-training pair 724579 applied the same explicit side-stream delay in both trials and retains FC2 inputs only in the second, using natural training allocator activity without injected replacement allocations.
Its perturbed schedule is diagnostic and cannot establish unmodified training reliability.
Its baseline rejected a NaN at update 1; eight raw snapshots are checksummed, with nonfinite FP32 128×128 gradient inputs on six ranks.
The retained-input trial completed finite updates 1–4 and strict external save2/load2/save4 checks on all eight ranks, with 16 reports, 144 immediate FC2 calls and peak retention of three inputs.
All eight baseline lifetime hooks were installed, but process termination prevented its exit-time call reports; summary zero call counts are unavailable evidence, not proof of zero calls.
This supports lifetime causality under the deliberately delayed schedule; original-run causality and a production fix remain unaccepted.
Reverse-order repeat 724599 reproduced the split: retained inputs passed finite updates 1–4 and strict eight-rank restart; the delayed baseline failed at update 1 with eight preserved raw snapshots.
The improved failure capture retained eight baseline lifetime reports covering 48 immediate FC2 calls with peak retention zero; the successful trial recorded 16 reports, 144 calls and peak retention three.
Both trial orders now support the lifetime mechanism in the perturbed training schedule.
No production source fix has been demonstrated.
Earlier uninstrumented failures 723856/723879, native local-bucket NaN 724075 and the separate cold-start 1000-second compilation-budget failure remain blocking.
Healthy observed or reload-only runs do not waive these failures.
Neither app has accepted training/restart or four-node validation, and full-depth, performance and publication gates remain closed.
All changes remain uncommitted and unpublished, with P2P-off and Alps networking defaults retained.

### Earlier build and diagnostic evidence (preserved)

Both original bases compiled, preserved their tracked vendor compute bytes, pass single-GPU numerical and actual RCCL/MPI loader audits, and have Enroot exports with exit zero plus independent full-data reads and checksums.
Primus26.8 retains MPI5.0.10 and passed two-node/eight-rank MIOpen, RCCL all-reduce and GPU-aware OSU all-to-all validation in job723432.
Primus26.3 uses the approved private MPI4.1.6 compatibility build; its unchanged Torch loads only Alps MPI4 core libraries, while26.8 and existing images retain the shared MPI5 implementation.
The26.3 private fixes include an OFI `offsetof` macro replacement, an explicit RCCL-to-retained-ROCm-SMI final relink without recompiling device objects, and forwarding stale system RCCL CMake metadata to the Alps build.
Its two-node RCCL all-reduce completed32B through4MiB with zero errors in job723802 after excluding the vendor hwloc OpenGL discovery component with per-launch `HWLOC_COMPONENTS=-gl`.
Stack traces established that merely clearing `DISPLAY`/`XAUTHORITY` did not prevent `hwloc_gl` from blocking in `XOpenDisplay` before aws-ofi network initialization.
The current MPI4 launcher selects per-launch `PMIX_MCA_gds=hash` to avoid unsupported host-Slurm `shmem2` component selection; these launcher/discovery controls do not alter image network defaults.
Separate Slurm steps resolved the second-MPI startup failure, but full26.3 validation723823 remains failed: GPU-aware OSU reaches MPI_Alltoall and reports CXI memory-mapping write errors on every rank, while RCCL again completes with zero wrong values.
The pinned MPI4 OFI component lacks MPI5-style FI_HMEM negotiation and accelerator registration support.
Capability queries confirmed CXI supports HMEM with mr_mode0, requiring no explicit application-side registration in this configuration.
A bounded preload diagnostic723839 enabled API1.18/HMEM negotiation and completed all18 GPU OSU sizes across eight ranks on two nodes in18seconds, with the unchanged base image and cm/ofi/cxi transport.
The corresponding private source patch now requests FI_HMEM/API1.18 without advertising FI_MR_HMEM callbacks that MPI4 does not implement; it still requires a rebuilt image and full base validation, and no transport defaults have changed.
The26.8 Apertus2 app compiled and passed package/source metadata, the preserved14-package/5164-file compute fingerprint and pointer-GEMM GPU checks, but its original fixed attention-gradient threshold failed.
An independent three-seed FP64 audit reproduced sparse failures with AITER relative L2 error below0.37%, versus approximately0.17% for BF16 PyTorch math; changing AITER conversion mode did not fix them.
A GPU diagnostic using mounted revised tests passed the exact amd-aiter0.1.22.post1 upstream BF16-control-relative gradient criterion plus added finite/nonnegative and one-BF16-epsilon relative-L2 guards; original mismatch counts remain visible.
That policy is pinned to upstream commit b4d9154d125e09efbe098d986e40fea3549c1244 and applies only to that AITER version;26.3 retains the original strict check.
The frozen app-inputs009 canonical26.8 app c2133f5e0d077b44045690eced172115 rebuilt as image8fd11f0b179dd84d7b1fee8bcc4a5c030e7f954169bef37e6b44d5261c1670ec and passed the revised in-image GPU stack check with unchanged compute fingerprint.
Its Enroot export exited0, passed a full filesystem read and SHA-2569e0301cc7e6265f89fe69703cbab35682547550a56653c58cf199ed54c5b8970.
Independent fresh-cache exported-image stack test723840 passed.
Two-node restart test723844 failed before model construction because vendor NVTE_FLASH_ATTN=0 conflicts with the reference auto backend; the CI launcher now clears the three NVTE attention selectors exactly as the reference launcher does, without changing image defaults.
Image references remain unchanged and validation hashes change; local checks pass315 with one skipped and49 focused tests.
Corrected cold restart test723851 reached the reference AITER path but hit the unchanged1000-second stage limit while compiling351 kernel objects per rank, before completing an update or checkpoint.
Warm-cache functional restart test723856 is separate: it imports three checksummed AITER modules from executable node-local tmpfs through an explicit PYTHONPATH allowlist and verifies the unchanged packaged tests and per-node compute fingerprint.
Warm functional723856 saved iteration2 but failed the strict finite-update guard: iteration1 reported grad norm inf and iteration2 reported2.106, despite finite reported losses.
The failed checkpoint and all logs are preserved; no restart/pipeline gate opens from this run.
Post-native-norm audit723866 captured finite inputs on all eight ranks and native norms1.602435/2.089203 matching aggregated FP64 references, so it did not reproduce the original failure.
That diagnostic perturbs timing and is not a fix or acceptance; an uninstrumented repeat is required to assess the unresolved behavior.
A CPU audit found no nonfinite tensor values in the16 saved model/optimizer files, which does not retroactively validate the prior nonfinite gradient norm.
No cold-start training, completed restart, full-depth or performance acceptance is claimed.
The26.3 app remains blocked on its full base gate.
Successor723842 is rebuilding the private HMEM-corrected canonical base9ebd8bb0806680810baec2ef43cfb120 from frozenbase-inputs009 after a strict exact-ID Docker restore; its prior node-local build cache did not survive allocation expiry.
Its bootstrap/SDK-cleanup compute fingerprint remains unchanged at11 packages and4293 files, which is not full base or app acceptance.
Its saved OCI archive is checksummed, but restore723813 correctly failed the exact-ID guard because OCI config conversion changed the image ID.
A Docker-format archive subsequently preserved the original config ID and loaded with the exact original image ID into a fresh separate Podman store on the existing builder node; identity guards were not waived.
The rebuilt26.8 app also has a checksummed, config-ID-preserving Docker archive; that separate archive has not yet been restored.
All prior failures, snapshots and the older native candidate are preserved; no commits, pushes, stable tags or registry promotions have been made.
The private HMEM source correction passed48 focused and314 full tests with one skipped; the subsequent CI attention-launcher correction passes49 focused and315 full tests with one skipped.
The user-requested delegated P2P omission probe723803 completed: both controls passed all18 sizes with zero wrong values, while omission passed1/3 on26.3 and0/3 on26.8, with four HIP invalid-device-pointer failures and one timeout.
INFO logs confirm actual local P2P/IPC with the flag absent versus SHM/direct/direct in controls; AWS Libfabric/CXI networking remained selected.
Failed launches have no correctness result; omission is unreliable on both bases, and the image P2P-off default remains unchanged.


## Primus rebuild checkpoint: 2026-11-05

The active objective is independent Primus26.3/reference-generation and Primus26.8/newer-stack bases and Apertus2 children, not another P2P study or full-depth run on the frozen native candidate.
Neither new base has completed compilation, export or validation; upstream-only pointer-GEMM forward/dgrad/fused-wgrad numerics passed on both stacks with `DEBUG_CLR_AQL_DEV_QUEUE=0`, which is not image/training acceptance.
Job722906 (26.3) exited after SDK cleanup removed Torch's MPI4 dependencies; job722902 (26.8) compiled RCCL and the network stack, then failed compiling rccl-tests because CMake de-duplicated split `-x hip` flags.
The reviewed fixes preserve the system SDK, use joined `-xhip` test options, and confine checksum-pinned OpenMPI4.1.6 to `primus-rocm:26.3` and its child.
This MPI exception is user-approved because the unchanged26.3 Torch requires `libmpi_cxx.so.40`;26.8 and existing images retain MPI5 and all Alps network defaults, including P2P off.
Both new Primus bases select Alps-rebuilt upstream RCCL2.30.4 (`rocm-systems@6b0e43f341195e203754e08f850e437ff2fc09f9`), not a CSCS RCCL fork.
All changes remain uncommitted and unpublished; frozen prior inputs/logs and the earlier native candidate are preserved.
Next gates are compilation, compute-byte preservation and actual library loading, full export integrity and GPU smoke, two-node base validation, then each app's stack and save2/load2/save4 acceptance before any full-depth/performance claim.
Orchestration and append-only evidence remain in benchmark `experiments/apertus2/runs/image-build-primus-bases-001/`.


## Scope and evidence

This comparison is pinned on 2026-11-03 and separates Imanol’s executed reference job `708188`, the older native image actually exercised in job `716336`, and the new native candidate being built.
The initial audit was taken while the candidate app was building.
The candidate subsequently built and its exported artifact passed independent full-read/checksum verification after an Enroot cleanup failure; GPU acceptance remains pending, and older-image measurements are not measurements of the candidate.
The comparison is an audit, not an apply-all optimization list or a claim of numerical/performance equivalence.

The durable machine-readable evidence is in benchmark `experiments/apertus2/reference/stack-comparison-2026-11-03/`, under `/iopsstor/scratch/cscs/boeschf/amd-vs-gh200-moe-training`.
Its `comparison.json` links package, source, environment and recipe evidence; `training-arguments-comparison.json` records exact reference/native argument arrays.
Historical captures and raw training runs are unchanged.

| Identity | Exact value |
| --- | --- |
| Reference executed Megatron | `d1277d86e0c1925f8438fd7630d86d05ab398dcd` |
| Reference Primus image | `/iopsstor/scratch/cscs/schlag/beverin/images/primus_v26.3.sqsh` |
| Current accessible Primus image SHA-256 | `7db644ae56cd4b076cb15d87fa3d47f0fc03221f49ccef35388c1a460d4e56ef` |
| Native measured app digest | `sha256:e0177f0ddbde95ae4abba2b0c951403e48828c35ecb4a9b5a9d7e09e3d63661a` |
| Native upstream Megatron | `0789b8e429553c2f17c350afa8619c69d01bec25` |
| Native candidate tag | `apertus-2-megatron-rocm:alps7-dev-64f5ff51e3f53ac123c1a0cec5cfd6a5` |
| Candidate canonical base tag | `pytorch-rocm:rocm10.0-ubuntu24.04-py3.12-torch2.13-alps7-dev-8780fe9102fd5a190ba8252cf3c40ac0` |
| Completed local canonical base image ID | `d94606f0f9b25eb9df664497286a0dd9bd81333aecd19326a286513db7ddec92` |

Image tags use prefix `jfrog.svc.cscs.ch/docker-group-csstaff/alps-images/`.
The reference image is 39,376,191,488 bytes and was inspected read-only without executing it.
Its newly recorded checksum identifies the currently accessible file; no checksum taken before job `708188` proves historical byte identity.

## Package and source identities

Reference image versions below are read directly from retained package metadata or Torch’s `version.py`, not inferred from the Primus filename.
Native versions in the third column are actual older-image training-child provenance, except source revisions explicitly identified as recipe pins.
The candidate preserves the native base constraints and declares these component pins, but must independently verify its final installation.

| Component | Reference | Native observed / declared candidate |
| --- | --- | --- |
| OS / Python family | Ubuntu 24.04.4 image; Python 3.12 environment | Ubuntu 24.04 base family; Python 3.12 |
| Torch | `2.10.0+git94c6e04`; source `94c6e047c2aa9f11c4cca60b4b90ea57b0eec54d` | `2.13.0+rocm10.0.0` |
| ROCm / HIP | ROCm `7.2.1`; Torch HIP `7.2.53211`; reference RCCL log `7.2.53211-e1a6bc5663` | ROCm `10.0.0` family; earlier archived-image preflight `716120` observed Torch HIP `7.15.26333` |
| Triton | `3.6.0` | `3.8.0+git4cff872c.rocm10.0.0` |
| NumPy | `2.4.4` | `2.5.2` |
| Transformer Engine | `2.12.0.dev0+40434cf6` | `2.20.0.dev0+b56021a`; pin `b56021a0e4a2e4a7822482ffdf5353368a777e0e` |
| AITER | `0.1.1.dev1611+gf299f579a.d20260512`; retained HEAD `f299f579a63b8de4f008e19b6f8867dbfa9eaf39` | No standalone `amd-aiter` package observed or installed by the candidate recipe |
| Primus-Turbo | `0.2.0+3cd482d`; retained HEAD `3cd482def8bb4814b1799fe481555559fff040b8` | No `primus-turbo` package observed or installed by the candidate recipe |
| FLA | Scratch setup installs `flash-linear-attention==0.5.2` / `fla-core==0.5.2`, then replaces KDA `gate.py` from Swiss AI | Swiss AI pin `1820dba7e15fb927294779f23ac5097eb3927b25`, installed `flash-linear-attention==0.5.2` |
| causal-conv1d | Image `1.5.0.post8` | `1.7.0`; pin `cd81f0413cad2fc1e6f17e785ac39f59aae690cd` |
| Emerging-Optimizers | Captured scratch Python source reports `0.3.1`; revision `1f29ffd2acbf047f3dd3efdd8e9469d9305a8f77` | `0.2.0`; pin `cc1385eec1966aea2aebfb6dd6fc66a898312a55` |
| UCCL / DeepEP wrapper | Not the selected successful-reference training dispatcher | Native `uccl==0.1.1` / `deep-ep==0.1.0`; UCCL pin `18b588a048a3a3fe3d4bd60aa5678c66e284df31` |

ROCm family labels and HIP compiler/runtime version strings are different namespaces; ROCm 10 does not imply a Torch HIP string beginning with `10`.
Native TE can use embedded AITER-related kernels without providing the standalone AITER Python API selected by the reference.
Retained Git HEADs identify source revisions, not proof that every image working-tree file or compiled binary matches that revision.

The audit retains 320 reference package-metadata entries and compares 102 native core/lock entries in `package-comparison.json`.
Reference scratch `PYTHONPATH` overlays can supersede image packages, and native lock entries are declarations rather than a fresh candidate runtime inventory.
Missing metadata is not proof that a package was absent from the historical process.

## Exact source differences and confirmed matches

`megatron-source-comparison.json` and `native-upstream-to-reference.patch` compare the exact pinned committed trees.
There are **45 differing files: 19 added, 23 modified and 3 deleted in the reference relative to native upstream**.
All 43 compared files present in the captured reference Python archive match the corresponding committed contents, including the otherwise unchanged fused-dispatch file.
The three deletions remove native tranche-end code/documentation/tests; they are not missing performance patches to copy back.
The candidate’s two local Megatron patches make its source differ from the reference in 46 files, because its fused-gradient hardening changes one additional file.

Confirmed matches are as important as differences:

- `RouterGatingLinearFunction`, `FusedDispatch` and `_load_bfd_c_library` are AST-identical between the unpatched native pin and executed reference.
- `megatron/core/optimizer/layer_wise_optimizer.py`, `megatron/training/checkpointing.py` and `megatron/core/datasets/bfd_pack.cpp` are byte-identical.
- The reference’s captured FLA KDA `gate.py` is byte-identical to native FLA pin `1820dba7e15fb927294779f23ac5097eb3927b25`.
  Both have SHA-256 `b350410cc605d1b3fac082d1e8384d7e9897e3aaa36c00f3d01d8d7fd48320ae`.
  This does not establish equivalence of the complete FLA trees.

`optimizer-source-comparison.json` and `native-optimizer-to-reference.patch` compare captured reference Python files against the native optimizer pin.
There are 51 byte-different Python files across the source trees, including 26 under `emerging_optimizers/`; these are source differences, not 51 demonstrated numerical regressions.
The reference adds `batched_newton_schulz_step` and a 3-D `torch.baddbmm` path absent from native, and adds a `deepseekv4` coefficient set.
Native instead retains the `moonlight` scale option removed from the reference snapshot; the captured reference training recipe selects `shape_up`, not `moonlight`.
Other differences include optimizer APIs, closure handling, logging, tests and additional optimizer families.

## Selected execution paths, not merely installed packages

| Area | Reference selected path | Native established baseline / candidate |
| --- | --- | --- |
| Expert GEMMs | Primus-Turbo Triton, pointer path, K4 pointer tables, model-specific tiles, fused FP32 wgrad | TE expert GEMMs; Turbo integration and tile tuning not ported |
| Global attention | Direct AITER plus K6 chunk scheduling | Native TE CK attention; standalone AITER integration pending |
| KDA / normalization / gates | Six KDA levers plus Beverin QK norm and fused output-gate paths | Six KDA levers enabled; Beverin attention/fusion series not ported |
| Expert communication | `BEVERIN_T6_EP=1`, `BEVERIN_T6_FP8=1`, `BEVERIN_T6_FUSED=1` | Ordinary all-to-all validated for training; UCCL experimental |
| Router / CE / activation fusions | K3 top-k, empty dgrad, CE and grad-sum; Triton SSSGLU; TE LM-head wgrad | Reference optimization series not applied; conservative functionality flags |
| Optimizer / overlap | NS workspaces, K7 batching, O1 shared-expert and wgrad side streams | Basic MuonMD path; those reference optimizations not ported |
| Startup | Beverin prewarm, copied AITER enum seed, external share-storage DSO, saved FLA configs and Triton failure guard | Native JIT with private verified tmpfs; candidate explicitly prebuilds CPU BFD |

The printed reference CLI says `--moe-token-dispatcher-type alltoall`, but its modified MoE layer selects the T6 dispatcher when the Beverin flag is set.
Therefore the successful reference is not evidence that the unmodified native all-to-all path, or UCCL, reproduces its expert-exchange behavior.
A3/T7 dispatchers, K2 convolution experiments, CPU offload and disabled `BEVERIN_EXPERT_RECOMPUTE=0` / `BEVERIN_DISPATCH_STASH=0` are not requirements merely because their files or flags appear in the archive.

## Megatron runtime patches: what the source-file diff alone misses

The user’s priority is the effective Megatron implementation, including modifications installed at import/runtime, rather than package-version alignment alone.
Three layers must be reconciled separately: the patched execution checkout, environment-selected branches within it, and imported-module/function replacements.
The focused audit `runtime-and-uccl-followup/review.json` records these distinctions and links checksummed source/hook snapshots.

| Mechanism | Reference code and behavior | Native baseline status |
| --- | --- | --- |
| Triton autotuner monkeypatch | `bench2/b2/pyhooks/sitecustomize.py` replaces `Autotuner._bench`; `RuntimeError` returns infinite timing instead of aborting the run; `B2_AUTOTUNE_SKIP` defaults to `1` | No equivalent reference hook selected |
| Turbo tile-selector replacements | `beverin_turbo.py::_install_tiles` replaces Turbo’s forward/dgrad and variable-K configuration selectors, and corresponding local Beverin bindings, using `BEVERIN_GG_TILES`; unmatched shapes retain original fallback | Neither Turbo nor this integration installed |
| Origami ranking fallback | `beverin_turbo.py::_ops` replaces Turbo `_safe_rank_configs` with a wrapper catching `Exception` and returning no candidates on ranking failure | Not installed; separate from the Triton autotuner guard |
| Recompute-extension substitution | `beverin_prewarm.py::_preload_share_storage` loads `BEVERIN_SHARE_STORAGE_SO` and assigns `megatron.core.tensor_parallel.random._share_storage_ext` instead of first-use `load_inline` | No reference scratch DSO injected; current-ABI source rebuild would be required |
| Synthetic startup prewarm | `beverin_prewarm.py::maybe_prewarm` performs synthetic forward/backward and selected reductions with state-restoration logic; share-storage substitution can operate independently | Reference prewarm implementation not ported |
| Diagnostic hook chaining | The B2 hook chains the shared Chonk `sitecustomize.py`, which registers SIGUSR1 faulthandler output and conditionally loads T5 tracing | Not a general GEMM/router override; T5 tracing is not enabled merely by the file’s presence |

The captured reference selected `BEVERIN_PREWARM=1`, `BEVERIN_PREWARM_LOG=1`, `BEVERIN_PREWARM_OPT=0` and `BEVERIN_PREWARM_EMPTY_CACHE=0`; do not replace its actual selections with module defaults.
The shared Chonk hook inspected in this follow-up matches the SHA-256 in the original capture, rather than being assumed unchanged.

The reference also patches Megatron’s `te_general_gemm` wrapper to pass `workspace` only if `inspect.signature(general_gemm)` accepts it, accommodating its TE 2.12 API.
Our native wrapper and newer TE require a compatibility review, not automatic application of an older-API workaround.
The router class being AST-identical does not imply that every function it calls, or its runtime library, is identical.

The source-integration groups still requiring reconciliation are expert Turbo/pointer/wgrad code, T6 dispatch, direct AITER/K6 attention, normalization/gate/fusion paths, optimizer workspaces/batching, and side-stream/deferred-wgrad overlap.
Installing the dependencies alone does not activate those Megatron paths, and enabling environment variables cannot create missing source implementations.

## Primus-Turbo assessment: atop the native image, not instead of it

Consider a native source build of Primus-Turbo on the existing Alps base as a separate proposed integration, not a switch to the complete Primus image.
No Turbo installation or implementation is authorized by this audit.
Reference-pin reproduction at `3cd482def8bb4814b1799fe481555559fff040b8` and adoption of a modern Turbo release are separate investigations because the reference Megatron code calls private Turbo APIs and supplies custom pointer/wgrad kernels.

- The reference requirements pin `triton==3.6.0` and `setuptools==69.5.1`; installing them wholesale conflicts with the native constrained stack.
- The reference setup conditionally adds AITER `f299f579a63b8de4f008e19b6f8867dbfa9eaf39` and Origami from rocm-libraries `01abf3db57f692c7ff70200b0697ed10335fb1e3` when those modules are absent.
  Dependency installation must therefore be reviewed explicitly rather than allowed to replace the base environment implicitly.
- `grouped_gemm_compute_offs` needs the compiled Turbo extension; a Python-only copy is insufficient.
- Validate the existing Torch/HIP/Triton ABI/compiler combination and MI300A’s 228-CU topology rather than assuming generic gfx942/MI300X tuning is transferable.
- Integration tests would need variable/empty expert groups, pointer ownership and offsets around 2 GiB, FP32 gradient accumulation, repeated microbatches and checkpoint/recompute/overlap behavior.

The pinned `requirements.txt` and `setup.py` are retained as evidence under `runtime-and-uccl-followup/reference-turbo-build/`.
These are assessment requirements, not completed installation or performance validation.

## Optimizer relevance: library upgrade is not the K7 port

The directly selected optimizer optimizations live in the modified Megatron `md_decoupling.py`, in addition to the different Emerging-Optimizers revision.
`BEVERIN_NS_WS=1` reuses persistent BF16 Newton–Schulz work buffers, while `BEVERIN_K7_BATCH=1` batches same-shaped expert parameters for Newton–Schulz GEMMs and hypersphere operations.
K7 implements its own `_newton_schulz_batched` routine in Megatron and imports coefficient sets/iteration helpers; it does not obtain its complete behavior by calling the newer library’s `batched_newton_schulz_step`.
Upgrading Emerging-Optimizers alone therefore does not reproduce K7.

The follow-up AST audit confirms that the ordinary `newton_schulz_step`, `newton_schulz_step_tsyrk`, and `simple`, `polar_express` and `cans` coefficient sets match between the captured reference optimizer and our pin.
The reference’s additional library batched helper is absent from native, but a version difference is not proof of different mathematics for the selected ordinary path.
Workspace reuse, batching, state preservation and reduction order still need separate numerical/restart and end-to-end performance assessment before any port.

## UCCL reconciliation with Imanol’s separate A2 tests

The successful mid-4b job `708188` used T6, not UCCL.
Imanol’s separate UCCL A2 component/exchange tests are a different reference, with a different source, build environment, tuning and teardown acceptance level.

| Identity | Reference A2 | Native |
| --- | --- | --- |
| Upstream source | `0578e0b819831f35c4ad17008e65e2159746ef80` | `18b588a048a3a3fe3d4bd60aa5678c66e284df31` |
| Local modifications | Saved `uccl-a2h/src.diff`; later committed as `6f2a0084f564dd796e5a90bcb0d2d862abaac80c` | `0001-alps-cxi-rocm-ep.patch` and `0002-preserve-torch-resource-ownership.patch` |
| Saved build provenance | `src.rev` names `0578e0b8`; its dirty diff is byte-identical to the committed `0578e0b8..6f2a0084` delta | Pinned upstream plus separately retained native patches |
| Build/runtime integration | Torch 2.10 / ROCm 7.2.1, host Libfabric headers and hook runtime | Torch 2.13 / native SDK, canonical image-built Alps Libfabric |

| Feature | Reference A2 | Native pinned source plus patches |
| --- | --- | --- |
| HIP memory registration | `FI_HMEM_ROCR` | Same intent via native CXI patch |
| CXI atomic signal-buffer allocation | HIP+CXI uses `hipExtMallocWithFlags(..., hipDeviceMallocUncached)` | `USE_LIBFABRIC_CXI` takes the ordinary `cudaMalloc`/HIP allocation branch; neither native patch changes this branch |
| Endpoint creation | Serialized, retry/backoff, shared per-thread domain/endpoint layout and optional per-peer endpoints | Serialized discovery/initialization and physical-device selection; not the A2 sharing/retry series |
| Completion/posting | O(1) bookkeeping, post `EAGAIN` retries | Full A2 completion/posting changes not carried |
| Optional modes/diagnostics | `UCCL_CXI_SIGNAL`, shared/per-peer endpoint options, alternate barrier, statistics and NVL-full sleep controls | Full A2 extension series not carried; not all reference modes were correct or faster |
| D2H handle tables | Managed allocations | Device-resident handle tables from our upstream pin; preserve this fix rather than downgrade |
| Buffer/stream ownership | Private native stream; historical component tests reported teardown faults and hard-exit workarounds | Torch retains RDMA-allocation ownership and supplies a borrowed pool stream; clean teardown is part of our component acceptance |
| Reviewed tuning example | A2 EP16 dispatch `64,40,512,32,512`, combine `64,7,512,32,512` | Our EP8 controls use paired `64,8,512,16,512`; neither is directly transferable tuning for the other test |

**The uncached HIP+CXI atomic allocation is a concrete reference difference missing from our current UCCL patch set.**
This deserves feature-level review, but it is not an established explanation or fix for our host-side first-forward-combine launch crash.
The `hipDeviceMallocUncached` branch already present under a later HIP `elif` in native upstream does not cover CXI, because the earlier `USE_LIBFABRIC_CXI` branch wins.
Native patch `0002` changes `UcclProxy::stop()` ownership, not that allocation branch.

His A2 notes report correct component/exchange cases at EP8/16/32, but no demonstrated clean full-model training acceptance equivalent to ours.
They found no performance win over T6 and retained teardown caveats; the A1 follow-up also corrects an earlier latency-tail observation as a communication-stream allocator-preallocation artifact.
Those historical measurements neither prove the current native UCCL path correct nor justify discarding native lifetime/handle fixes.
The proposed review is feature-by-feature reconciliation, not replacing the pin or applying the A2 patch wholesale.
The retained `src.rev`, `src.diff`, source snapshots, A2 notes and A1 correction are in `runtime-and-uccl-followup/`.

## Network, runtime and tuning differences

Reference EDF enables the CXI and OFI injection hooks and explicitly selects a scratch aws-ofi-rccl 1.4.0 v5-to-v6 plugin shim.
The reference log confirms RCCL `2.27.7`, aws-ofi-rccl `1.4.0` and provider `cxi`; scratch RCCL/plugin binary hashes remain in the original capture.
Reference hook-libfabric version claims in EDF comments are documentary, not a captured loaded-library measurement.

Native EDF disables CXI injection, omits the OFI injection hook, and sources the image’s Alps environment explicitly before Python.
Actual native child mappings include SDK-wheel RCCL, `/usr/lib/librccl-net.so`, `/usr/lib/libfabric.so.1.32.0` and `/usr/lib/libcxi.so.1.5.0`.
The candidate recipe pins the Alps Libfabric source to `5de2d6a2df83595bcc38531cb91bf59495559495` and OFI plugin source to `a2a6d08ea98296f99596b5c18b30b8a20d74f609`.
Its profile keeps bundled RCCL (`ROCM_REBUILD_RCCL=0`), unlike the reference’s scratch gfx942 RCCL rebuild.
A complete native RCCL runtime version string was not captured by this comparison; do not infer it from the image-family name or DSO soname.

| Setting | Reference evidence | Native job `716336` evidence |
| --- | --- | --- |
| `NCCL_CROSS_NIC` | `1` | `0` |
| `NCCL_SOCKET_IFNAME` | `hsn0,hsn1,hsn2,hsn3` | `hsn` |
| `FI_CXI_DEFAULT_TX_SIZE` | Launcher finally selects `2048` | `16384` |
| `NCCL_NET_GDR_LEVEL` | Launcher finally selects `LOC` | Not captured in actual child environment |
| `NCCL_P2P_DISABLE` | Not present in captured pre-import environment | `1` |
| `HSA_ENABLE_INTERRUPT` | `0` | Not captured in actual child environment |
| Allocator | `expandable_segments:True` | `expandable_segments:True` |
| CPU / NUMA | Saved explicit per-rank CPU masks and A5 interleave selection `1-3` | Harness `cpu_bind=none`; actual affinity/NUMA diagnostics retained |
| hipBLASLt | Saved mid-4b solution-index override | No matching native tuning table selected |
| FLA tuning | `FLA_CACHE_MODE=fuzzy`, fourteen saved Triton-3.6 config files | No historical training table selected |

The reference launcher prints its environment **before** unsetting `NVTE_FLASH_ATTN`, `NVTE_FUSED_ATTN` and `NVTE_UNFUSED_ATTN`, and before changing CXI TX size/GDR level.
`environment-comparison.json` preserves that raw snapshot separately from explicit subsequent launcher changes, rather than calling the printed values final training settings.
The printed reference `NVTE_USE_HIPBLASLT=1` is not evidence of a reference rocBLAS workaround, and the pinned native TE has no effective historical `NVTE_USE_ROCBLAS` backend switch.
No image-level global BLAS or P2P policy change is part of the candidate.

## Training recipes and validation are not equivalent

| Dimension | Reference `708188` | Native passing proxy `716336` |
| --- | --- | --- |
| Nodes / ranks / PP / EP | 16 / 64 / 2 / 8 | 2 / 8 / 1 / 8 |
| Layers / sequence / vocabulary | 32 / 8192 / 200063 | 4 / 512 / 1024 |
| Hidden / heads / query groups | 2560 / 20 / 10 | 2560 / 20 / 10 |
| Experts / top-k / expert FFN / latent | 256 / 8 / 1440 / 1280 | 256 / 8 / 1440 / 1280 |
| Global / micro batch | 1152 / 1 | 8 / 1 |
| Updates / restart | 26 logged updates; no restart acceptance established here | Updates 1–2, checkpoint, fresh-process load at 2, updates 3–4 |
| LR / matrix LR / gains LR | `0.00426` / `0.0213` / `0.00426` | `0.0001` / `0.001` / `0.0001` |
| Schedule | Linear, 718 warmup iterations, 71829 decay iterations | Constant functionality-test schedule |
| Router top-k scale | Explicit `2.5` | Observed parsed value `None`; no additional probability multiplication |
| Init std / rotary base / Adam betas | Explicit `0.0197642` / `10000` / `0.9,0.95` | Later candidate proxy parsed values: `0.02` / `10000` / `0.9,0.999` |
| Attention / CE / generic fusions | Auto attention, TE CE fusion, reference custom paths | Explicit fused attention; TE CE not selected; several generic fusions disabled |

The exact option diff includes equivalent spellings such as `1.0` versus `1`, so an argument difference is not automatically a semantic difference.
Reference mid-4b steady-window mean is `24.0845176471 s` for steps 10–26, or `6122.439409 tokens/s/GPU` from the mean.
The proxy’s finite losses and successful restart establish neither full-model capacity nor throughput parity nor production convergence.
Full mid-4b reproduction still needs the full geometry, schedule, router settings, pipeline behavior and selected optimizations reconciled explicitly.
The later [`MID4B-REPRODUCTION-PLAN.md`](MID4B-REPRODUCTION-PLAN.md) resolves these omitted defaults from actual parsed logs and identifies PP2/checkpoint/resource blockers without implementing changes.

## Existing native portability patch set

The native recipe also retains the following portability fixes, separate from both the reference optimization series and the three new candidate changes.
Their full patch contents and SHA-256 hashes are snapshotted under the audit’s `native-recipe/` and `native-recipe-inputs.json`.

| Patch path relative to `patches/rocm/` | Native behavior |
| --- | --- |
| `uccl/0001-alps-cxi-rocm-ep.patch` | Physical GPU selection, serialized node-local barrier cleanup/CXI initialization, ROCR HMEM and ROCm launch/device fixes reconciled onto pinned UCCL |
| `uccl/0002-preserve-torch-resource-ownership.patch` | Torch retains RDMA-allocation ownership and owns the borrowed communication stream, including recorded tensors surviving buffer destruction |
| `transformer-engine/0001-declare-hip-translation-units.patch` | Compile GPU `_hip.cpp` and CK attention sources as HIP rather than ordinary C++ |
| `transformer-engine/0002-resolve-runtime-only-roctx-library.patch` | Link the versioned `libroctx64.so.4` runtime when the wheel SDK lacks an unversioned development link |
| `ck-jit/0001-isolate-real-compiler-environment.patch` | Preserve interceptor discovery but remove `ROCM_PATH` / `HIP_PATH` from actual compiler/linker subprocess environments |

The two UCCL patches do not establish full-model UCCL training readiness, and the CK-JIT fix does not install the standalone training AITER package.

## Native-only candidate changes

These changes were motivated by reproduced native failures, not copied missing reference patches:

1. Materialize fused-dispatch gradients before recording the event dependency, covering the confirmed expanded-gradient layout hazard.
2. Default HIP router GEMM to the existing ATen branch, retain that choice for backward, and expose `APERTUS_ROCM_ROUTER_GEMM_BACKEND=te|aten` for diagnostics.
3. Prebuild BFD’s CPU DSO during image construction rather than relying on concurrent rank startup compilation.

The corresponding upstream reference/native functions were identical before these changes.
Router-only ATen passed the older-image real-shape proxy with a run-local override, but the candidate must pass again without that override.
These fixes are not claimed to repair the separate UCCL first-forward-combine HIP launch crash.

## Remaining exactness limits

- Candidate build/export identity and selected package/source/test provenance are now recorded in benchmark `runs/image-build-candidate-001/`; the original exporter cleanup failure is retained separately from artifact verification.
  Candidate GPU numerics, actual training-child library mappings and training acceptance remain pending.
- Reference full FLA source identity, TE full source revision/working-tree state, historical loaded-library mappings and job-time image checksum are not established by the current captures.
- Reference image package metadata and retained Git HEADs do not establish compiled-binary equivalence to source or exclude external runtime overlays.
- No source diff or short numerical test measures the performance contribution of an optimization; controlled native A/B tests remain necessary.

## Candidate build/export follow-up

The candidate local image ID is `1de9bd00e3bd6f8c3cee5d59b25024fc6ea3d04643517fe93461c213911425d3`.
Its exported SquashFS SHA-256 is `b4f5300c8ecbb69a442e44c0044007a09e56742422b4a867b34eac716f51ea9d`, with 31,528,255,488 file bytes.
The image passed three CPU regressions and build-time metadata/patch/BFD/ELF checks; the exported filesystem independently passed full metadata/file-data reading and selected provenance checks.
Enroot returned 1 after writing the image because of its cleanup path; that failure is preserved, not described as clean export-command success.
`candidate-build-export-record.json` in the audit links the actual build/artifact evidence without turning those checks into GPU or full mid-4b acceptance.
The first checksum-pinned EDF stack check is prepared but not submitted.


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


## Follow-up, 2026-11-05 — flag ablation completed: SENDRECV was already selected; P2P remains unreliable

The delegated native flag study completed in job `722504` after17minutes1second, releasing its two nodes from the single30-minute allocation.
It ran42 sequential fresh eight-rank Slurm/PMIx launches of the exact baked-in CI all-reduce command on `nid002478` / `nid002514`, both with kernel `6.4.0-150600.23.125-default`.
The main agent verified the frozen launcher hashes,42 distinct MPI namespaces, all42 matrix entries and all10 complete zero-error/eight-rank numerical passes.
The parent allocation's `COMPLETED` state means the study finished, not that every case passed.

| Native configuration | Numerical passes / fresh launches | P2P/IPC selected |
| --- | --- | --- |
| Complete colleague bundle, P2P disabled | 3 / 3 | 0 / 3 |
| Complete colleague bundle, P2P enabled | 2 / 5 | 5 / 5 |
| Only `NCCL_P2P_DISABLE=0` relative to image defaults | 1 / 10 | 10 / 10 |

Every tested unset-protocol launch selected SENDRECV, including successful minimum and leave-one-out cases.
Explicit `OFI_NCCL_PROTOCOL=SENDRECV` was therefore not a transport switch in these controlled tests on this node pair.
This strengthens the earlier caution against attributing the initial successful smoke tests to SENDRECV alone.
Mechanically, `NCCL_P2P_DISABLE=0` alone is sufficient to select P2P/IPC, and one such minimum launch completed the full numerical sweep with every other investigated optional override absent or restored to image defaults.
It is NOT sufficient evidence of reliable communication: the minimum failed9/10 launches, and the complete bundle failed3/5.
Across all32 failed launches,29 logged invalid-device-pointer errors and five reached the native deadline, with two launches in both categories.
Representative full-bundle failure: `cases/003-full-r2/ranks/0001/native.log:432–441` reports `[FATAL ERROR]: HIP failure: 'invalid device pointer'`, followed by RCCL `transport/p2p_tmp.cc:450` / `:625` during communicator setup.
The error occurred on both tested hosts; the experiment does not establish the exact kernel/RCCL/IPC defect or universal behavior on every node.

Leave-one-out tests removing explicit SENDRECV, threshold192 or forced flush each retained a1/2 numerical pass.
Removing CROSS_NIC, PXN_DISABLE, GDR_LEVEL or P2P_LEVEL each gave0/2, but these samples cannot establish necessity amid intermittent failures, especially when the all-default minimum itself passed once.
Five tested small-addition combinations were also unreliable; no robust minimal P2P-on combination was found.
Single-pair sampling, only two trials for most variants, and potentially non-IID initialization failures prevent reliable causal ranking or strong statistical sufficiency claims.
The study distinguishes IPC activation from correct operation and does not prove that an untested combination cannot work.
The image's existing P2P-disabled default should not be removed on the strength of the earlier individual successes.
The complete new bundle with P2P disabled is the better-supported fallback among these small controls, not a guarantee of full-scale training reliability.

Main full-depth training `722506` was still `PENDING(Resources)` at the latest scheduler check; no training allocation was consuming those sixteen nodes yet.
Its user-requested P2P-enabled configuration remains unchanged; no job was cancelled, silently switched to P2P-off, or replaced after the ablation findings.
Interactive721530 and unrelated CI722408 remain preserved, and no new ablation allocation was submitted.
The successful small training/restart and transport gates remain valid individual observations, but do not establish startup reliability across repeated launches.

All agent artifacts are retained under benchmark `experiments/apertus2/ablations/p2p-flags-2026-11-05/`: `REPORT.md`, `matrix.json` / `matrix.csv`, `findings.json`, `verification.json`, pinned source audit and every case's logs/provenance.
Main integration evidence is `results/p2p-flag-ablation-722504-integration.json`, with artifact hashes and independently checked matrix totals.
The main178-test CPU suite and image/defaults remain unchanged.


## Follow-up, 2026-11-05 — legacy IPC follow-up finished without a reliable P2P fix

The delegated legacy-IPC study has finished and released reservation nodes `nid002456` / `nid002458`; no replacement allocation was requested.
Job `722681` ended `FAILED` after16minutes21seconds because the controller encountered empty per-rank exit metadata after the final failed native step.
The agent preserved the original21-case controller matrix and transparently reconstructed the final failed case from raw native/exit/accounting evidence in `matrix-recovered.json`, yielding22 launches rather than excluding the interrupted failure.
The main agent verified frozen source hashes, the22-case total and all four complete eight-rank/18-size zero-error numerical passes.

| Configuration | Legacy `1` passes | Legacy explicitly unset passes |
| --- | --- | --- |
| Full P2P-on bundle | 1 / 5 | 0 / 5 |
| Minimal P2P-on | 0 / 5 | 1 / 5 |
| P2P-off safety control | 1 / 1 | 1 / 1 |

All nine failing legacy-on launches reached the native timeout without logged invalid-device-pointer errors; absence of that error is not success when the operation hangs.
The added flag therefore did not establish repeated numerical sufficiency or a reliable P2P-on combination.
The loaded ROCr library contains the flag name, and a supporting public-source audit plus live process-owned IPC export-socket differences support actual runtime path selection rather than merely exporting an ignored variable.
The public source revision was not established as the exact bundled source; socket observations are not a function-level trace or proof of the exact IPC defect.
Complete child environment captures matched the designated flag value or absence; some immediate captures were empty during startup and are explicitly marked unavailable, not treated as unset.
On the subset with all eight child environments directly confirmed, repeated failing legacy-on launches remain, so incomplete startup captures do not explain away the result.
The P2P-off controls passed; samples are small and failures may be non-IID, so neither a robust production failure rate nor a universal root cause is claimed.

No main training job was altered by this flag study; cancelled `722605` had already been stopped separately at the user's request.
No training or native ablation GPU job is now active; the latest scheduler view contains only the preserved interactive allocation.
Keep the image's P2P-disabled default, and do not repeat the unchanged full model on the basis of the single legacy-on success.
The next recommended technical gate remains a lightweight32-rank FP32/61440-element plain-versus-coalesced reduction reproducer with a P2P-disabled control, before returning to model checkpoint/restart and learned-routing issues.
No such new allocation has been launched automatically by this report integration.
Evidence: benchmark `experiments/apertus2/ablations/p2p-flags-2026-11-05/legacy-ipc-001/REPORT.md`, `matrix-recovered.json`, `findings.json`, raw case logs, and main `results/p2p-legacy-ipc-722681-integration.json`.


## Outcome, 2026-11-05 — no-overlap full-depth trial timed out after post-step scalar-reduction stall

Full-depth P2P-off/no-gradient-overlap **722718** ended `FAILED`, exit1:0, after25m23s at the configured1500-second supervisor cutoff.
No complete logged update or checkpoint was produced, and strict global acceptance rejects the run.
All64 supervisor states are failed;13 explicitly retain the1500-second cutoff, while other peer/shutdown errors are termination effects rather than proven initiating causes.
No explicit OOM, invalid-device-pointer or CXI exception was observed before shutdown.
The GPU allocation is released; only preserved interactive721530 remained in the latest scheduler view, and no replacement allocation was submitted.

The supported small no-overlap path is accepted by722711, but disabling overlap is not a sufficient full-scale fix.
Native dense gradient reductions, including FP32 `[61440]`, and global BF16 router-metric `[15,256]` / `[16,256]` reductions completed on all64 ranks in722718.
All4464 first-forward dispatch groups were complete, with no volume mismatch and1.0125885009765625 peak-to-balanced ratio.
The run instead waits on the later optimizer-success/gradnorm scalar-reduction frontier in model-parallel group81/ranks8,40: both enter Int32 `[1]` MIN with matching metadata, rank8 completes and enters the FP32 `[1]` MAX, while rank40's first reduction remains started.
At24m29s, sampled native completions were778–779seconds old; the remaining ranks were blocked behind default-world LR logging or second-stage loss reduction/device scalar access.
Different current sequence numbers do not establish a mismatched first collective, and zero retained signature-mismatch candidates do not prove stream/progress/order correctness.
Final training-child source/module/runtime provenance was not published on any rank after termination, so frozen inputs/actual DDP markers are not treated as final all-rank source acceptance.

Keep P2P disabled and stop full-model flag tuning.
The recommended next gate is a bounded communication-only reproduction of pairwise Int32 MIN →FP32 MAX with immediate scalar reads, first isolated and then interleaved with the observed DP-loss/default-world logging groups and preceding GPU/collective work.
A serialized-progress control is a diagnostic contrast, not authorization to insert arbitrary barriers into training or an established fix.
No new allocation/image rebuild/source patch/commit/push/promotion was performed during this result integration.
Full-model checkpoint/restart/cache handling and26-update baseline remain blocked.
Evidence: benchmark `ablations/gradient-overlap-off-2026-11-05/REPORT.md`, `outcome.json`, timestamped observations, and `results/train-mid4b-no-grad-overlap-p2poff-722718-{timeout,collectives,forward-loads,precutoff-frontier,model-pair81-live}.json`.

## Primus reference-derived image work: 2026-11-05

The user authorized independent Primus26.3 and26.8 base/app builds, retaining our RCCL, Alps network stack and unchanged network defaults including P2P off.
This supersedes the earlier native-only direction for this new work, not the frozen native experiments or existing PyTorch-based variants.
The registered selectors are `primus-rocm:26.3` / `primus-rocm:26.8` and Apertus2 `megatron-primus263-rocm` / `megatron-primus268-rocm`.
The digest-pinned upstream audits passed and preserve each release's compute generation;26.3 matches the captured reference Torch/Triton/TE versions, while26.8 is a separate newer-stack comparison.
The app recipe vendors the exact reference Megatron source archive, pins the reference optimizer and FLA gate, rebuilds ABI-specific CPU extensions, and contains portable compute-only defaults.
Private reference Turbo APIs on26.8 remain a compatibility gate, not an assumed success.
The first build array722833 failed before executing its first RUN body because chroot isolation could not set supplemental groups.
OCI retries722859 and722860 are running on two reserved nodes with owned `/dev/shm/podman-22404-primus-*` storage; interactive721530 is untouched.
All image implementation remains uncommitted in this repository; orchestration, frozen contexts, manifests and logs are under benchmark `experiments/apertus2/runs/image-build-primus-bases-001/`.
No base/app build, multi-node validation, full-depth restart or performance acceptance is claimed, and no registry promotion has occurred.

The OCI26.8 retry then reached package installation but failed because the wrapper's single-UID namespace cannot assign `man-db` ownership;26.3's matching wrapper build was agent-stopped before repeating this blocker.
Native rootless Podman preflight confirmed full subordinate UID/GID mappings on the Slurm host.
Replacement build jobs722886 and722887 use native host Podman with OCI isolation, explicit layer caching, exactly one canonical output tag, and owned node-local storage; the frozen image recipes/hashes are unchanged.
An upstream26.8 GPU import probe confirmed a missing private Turbo offset symbol; the app now has a version-gated three-file import adapter without vendor-library changes or disabling optimized paths.
Adapter/source provenance is tested on CPU, while GPU numerical checks, image compilation and training gates are still pending.

Native build-RUN preflight passed with explicit cgroupfs/host cgroup namespace; current build jobs are722906 for26.3 and722902 for26.8, still using unchanged frozen base recipes.
The latest repository suite passed288 tests with1 skip, and reviewed app inputs/scripts are frozen as `app-inputs-001` without releasing app-ready markers.
The unmodified26.3 reference pointer-GEMM forward/dgrad/fused-wgrad standalone GPU probe passed; this is upstream compute evidence, not new-base/network/training acceptance.
The26.8 source adapter passes reference API imports, but a traced numerical probe fails before any reference kernel: vendor Torch2.14 segfaults on its first GPU `fill_()`.
Both allocator modes and seccomp-unconfined fail; the preserved Torch2.13 candidate passes fill/normal/divide on the same node under Enroot.
Native GDB reproduces the crash after bootstrap inside `libamdhip64`'s `hipLaunchKernel`, with rocprofiler-SDK interposition visible; a targeted control of the vendor-enabled device-memory AQL ring is pending, not yet an established cause or fix.
Our RCCL/network settings and P2P-off policy remain unchanged.
