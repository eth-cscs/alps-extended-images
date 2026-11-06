# Imanol reference versus the native Apertus 2 ROCm stack

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
