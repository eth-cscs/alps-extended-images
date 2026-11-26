# Native training UCCL patch disposition

The training pin is `52d4184c89deb5631be6dac6a9b1c40896c9e26e`, verified as upstream HEAD during preparation.
This supersedes `18b588a048a3a3fe3d4bd60aa5678c66e284df31`; serving images and their independent pins/patches are not changed.
Both patches are checked and applied in name order, and the image metadata check reverse-checks their actual application.
The deeper history review below supersedes the initial source-only judgment that every retained correction was necessary.
The working recipe now removes the redundant cleanup relocation/barrier, makes CXI NUMA affinity follow the selected physical NIC rank, and passes borrowed communication streams by an optional trailing native keyword without shifting existing positional arguments.
Primus26.8 selects the minimal `patches/rocm/uccl-primus268/0001-alps-cxi-rocm-ep.patch` plus the shared ownership0002patch; its min/max, volatile-snapshot and builtin-math rewrites are removed after successful joint host/device code generation on the exact stack.
The older non-Primus ROCm10.0 target retains its separate compiler-compatibility0001patch because this audit did not establish removal safety on that unrelated toolchain.
The base reference is unchanged by these app-only corrections; the app receives a separate source snapshot through `build.py`.

| Correction | Disposition at52d4184 | Reason |
| --- | --- | --- |
| Local CXI discovery/domain mutex | Discarded from0001 | Upstream `g_cxi_setup_mu` already serializes provider setup and supplies bounded fresh-info retries; a second mutex is redundant |
| Physical GPU/NIC rank selection | Retained/rebased in0001, with matching NUMA lookup | Upstream selects by logical `ctx.local_rank`; pass `cfg_.nic_local_rank`, including initial discovery, fresh-info retry and new affinity lookup |
| Node-local barrier cleanup | Removed from the working recipe | Existing topology all-gather already orders every rank's original cleanup before proxy construction; the extra group barrier is redundant |
| HIP ROCR HMEM registration | Retained/rebased in0001 | Upstream CXI registration still selects `FI_HMEM_CUDA`; HIP needs `FI_HMEM_ROCR` and `attr.device.rocr` |
| HIP cooperative extended launch | Retained in0001 | HIP builds need `hipLaunchConfig_t`/`hipLaunchKernelEx` rather than CUDA extended-launch types |
| ROCm10 integer min/max, volatile-head snapshots and device fabs/fmax | Removed from the selected26.8 patch | Joint optimized host/device codegen passes without all three groups on the exact10.1 stack; unrelated10.0 compiler support is not silently removed |
| RDMA allocation ownership | Retained in0002 | Proxies still attempt to free the wrapper's borrowed DLPack/Torch allocation |
| Communication stream lifetime | Retained with positional-API correction in0002 | Use a borrowed Torch pool stream for DeepEP, preserve seven native positional arguments and destroy only legacy native-owned streams |
| Runtime top-k capacity | Not replayed | Already upstream before the old training pin |
| Device-memory D2H handle table | Not replayed | Still upstream via `cudaMalloc(&d_handles, ...)`; retain it rather than returning to managed memory |

The patch application audit is a source-level check, not evidence that GPU launch, CXI registration or destruction is correct on a scheduled run.
The image includes a separate two-node component/lifetime smoke and preserves P2P-disabled/network defaults.

## Commit-history audit:18b588a to52d4184

This review covers all45commits in `18b588a048a3a3fe3d4bd60aa5678c66e284df31..52d4184c89deb5631be6dac6a9b1c40896c9e26e`, their messages, the complete EP/shared-runtime interval diff, and earlier function history where an important assumption predates the old pin.
It is not based only on a clean `git apply` or matching source text.
The selected path is native ROCm10.1/gfx942 high-throughput DeepEP, four visible GPUs and four block-ordered Slurm ranks per node, `NUM_MAX_NVL_PEERS=4`, with exclusive same-UID UCCL use of those nodes.
Low-latency/dedup, visibility remapping, partial nodes, raw native-constructor users, CUDA images and serving/P2P transports are not selected-path acceptance claims.
Compiler-only requirements can nevertheless affect the selected image because `ep/setup.py` compiles all EP translation units, including unused LL kernels.

The retained raw evidence is under `apertus-amd/artifacts/image-builds/primus268-uccl52d4184-aiter-20261112/uccl-history-audit/`: full commit messages, interval diff, earlier overlap/lifetime history, CPU ordering probe/results and a proposed corrected0001patch.
The original frozen base build context is not changed by this review.
After proving that the app context did not yet exist and the canonical base reference was unchanged, the cleanup/NUMA and positional-API proposals were applied to the working recipe for its later app snapshot.
The companion metadata check now checks ownership-guarded destruction rather than incorrectly banning every native stream destruction, including the preserved legacy-owned path.
The retained proposals and `proposal-check-v2.json` record the seven-positional-plus-keyword AST/signature mapping test and exact proposal hashes; `working-recipe-reapply-stage1.json` records fresh application of the corrected working patches.
`working-recipe-reapply-minimal.json` then records the selected minimal26.8 patch set and byte-identical upstream cleanup/config/internode/ordinary-LL source inputs, with the canonical base still unchanged and the app context not yet created.
Any native correction must be snapshotted through `build.py` into a new retained attempt and published only through `publish.py`.

### Upstream changes that alter the interpretation

| Upstream change | Relevance to retained intent |
| --- | --- |
| [34bdf4f / PR985][r985], before the old pin | Already has a blocking topology `all_gather_object()` after cleanup; introduces distinct barrier-local ranks |
| [c8b236f / PR989][r989], before the old pin | Separates runtime device ordinal, dense node rank and BDF-ordered physical/NIC rank; the later CXI branch did not fully adopt that split |
| [7696801 / PR957][r957], before the old pin | Introduces allocator `record_stream`/communication-stream allocation fixes; those lifetimes remain relevant after Buffer destruction |
| [c1c3c3f / PR963][r963], before the old pin | Makes the non-SM90 path cooperative too; the HIP launch patch preserves that policy rather than newly enabling cooperation |
| [4358969 / PR1079][r1079] | Supports four-peer HT layouts and adds LL dedup; dedup is explicitly disabled on HIP, not a replacement for ordinary HT or its compiler requirements |
| [54e216e / PR1052][r1052] | Final merged change restores the original device barrier and adds a meaningful test; its message records a rejected protocol rewrite, so it must not be misread as a host cleanup fix |
| [53fe8ad / PR1109][r1109], [cd0ea9e / PR1101][r1101] | Supersede the local CXI setup mutex with upstream serialization plus fresh-info domain retries |
| [1bad71a / PR1123][r1123] | Fixes the Python `EventHandle()` default to record on the current stream; does not make a native stream immortal or release allocator-recorded tensors |
| [f04b49b / PR1147][r1147] | Stops/drops registered proxies before interpreter finalization; does not transfer scratch ownership or remove `UcclProxy::stop()`'s free |
| [161e563 / PR1135][r1135] | Separately fixes `FifoProxy::stop()` to call `destroy(false)` because Python owns its scratch; corroborates the ownership contract, but does not change the selected `UcclProxy` stop path |
| [900b8c4 / PR1139][r1139] | Newly pins CXI threads using `/sys/class/cxi/cxi<cfg.local_rank>`; a physical-NIC selection patch must now keep this lookup consistent too |
| [343c1e4 / PR1137][r1137] | Enables hybrid MR descriptors; correct ROCR registration remains necessary for the main and atomic GPU MRs whose descriptors posts now use directly |
| [96d228a / PR1115][r1115], [e454752 / PR1143][r1143] | Add CXI EAGAIN progress/retry and sibling-ring QUIET ordering; neither changes scratch/stream ownership nor substitutes for compiler fixes |
| [bc9eff0 / PR1086][r1086], [78adb5d / PR1072][r1072] | Refactor shared GPU adapters and preserve runtime ordinals in topology discovery; do not add HIP extended-launch mappings, global min/max or ROCR MR selection |
| [52d4184 / PR1136][r1136] | Adds an AMD LL-kernel `__syncthreads()` after shared barrier initialization; this is a different barrier from POSIX-shm cleanup and does not supersede or justify a host group barrier |

PR1089 (`33f1dbb82eab0529428bcd3703dc8c847abd19e9`) and PR1103 (`a799e9f49d6fef74d8f09fa72d29c57b8c26c418`) change serving/P2P IPC import contexts, not the EP Buffer's direct IPC path.
Their peer-access fixes are not evidence that ROCm EP IPC or RCCL P2P is fixed, and no network default is changed on their account.
PR1127 (`9e3f831acf0985a6bba8d4737d187ac51c429b70`) turns off the serving/P2P ROCm profile's default CXI flag, not the selected training extension's explicit `USE_LIBFABRIC_CXI=1` build.
The remaining interval commits concern collective/RDMA/P2P transports, benchmarks or CI rather than these retained EP ownership/launch/registration functions; their complete messages remain in the evidence inventory.
There are no interval changes to `ep/include/ep_launch.cuh`, `ep/include/ep_configs.cuh` or `ep/include/ep_utils.cuh`.
The current published Primus26.8 Torch2.14 hipify mapping file was also inspected: it maps ordinary `cudaLaunchKernel` and `cudaLaunchCooperativeKernel`, but has no entries for `cudaLaunchKernelEx`, `cudaLaunchConfig_t` or launch-attribute types.
That exact-stack evidence supports the HIP launch adaptation independently of UCCL source similarity; retained compiler ablations still provide the final compile verdict.

### Per-hunk classification

Hunk identifiers below follow the initially retained v1 patch order and original `@@ -line` anchors, preserved in the audit's `original-0001.patch`/`original-0002.patch`, rather than renumbering away removed hunks.
The top disposition table describes the corrected working recipe; this detailed matrix retains the reasoning about every initially retained hunk, including the removed and corrected ones.
“Essential” is a source/contract conclusion for the stated path, not a claim of completed GPU validation.
“Unproven” requires ablation on the exact pinned compiler, not automatic retention merely because the old hunk still applies.
Tests T1–T8 are defined below.

| Hunk | Classification and selected-path scope | History/assumption review | Required checks |
| --- | --- | --- | --- |
| 0001/H01, `bench/utils.py -595`, delete original cleanup | **Redundant**; recommend remove | [r985], [r989] already place topology all-gather between cleanup and every proxy construction; original cleanup cannot run late past that collective in this group | T1 |
| 0001/H02, `bench/utils.py -615`, leader cleanup plus new barrier | **Redundant**, with extra synchronization cost; recommend remove | Moves cleanup after the existing all-gather and then adds another group barrier to restore ordering that already existed; does not isolate another same-mode group/job | T1; concurrent same-UID jobs remain unsupported |
| 0001/H03, `cxi_transport.hpp -20`, overload declaration | **Unproven necessity on identity-mapped full nodes**; compatibility scaffold | Supports physical NIC selection from [r989]; no functional difference when device ordinal and NIC rank match, but full visibility alone is not proof of identity | T2, non-CXI compile |
| 0001/H04, `ep_configs.cuh -65`, global integer min/max templates | **Redundant compiler workaround on selected10.1; removed** | No explicit replacement in [r1086]/[r1072], but joint optimized codegen succeeds without it; dropping broad templates also avoids unnecessary overload/signedness scope | T5 passed for selected stack; GPU checks remain separate |
| 0001/H05, `ep_launch.cuh -2`, HIP cooperative config | **Essential HIP API adaptation**; selected HT launches use it | [r963] already requires cooperation in DISABLE_SM90; preserve grid/block/stream/cooperative=1 and omit only CUDA cluster attributes; do not replace with an ordinary noncooperative launch | T4 |
| 0001/H06, `ep_launch.cuh -24`, guard close and HIP launch API | **Essential HIP API adaptation**; selected HT launches use it | [r1086] supplies gpu-runtime aliases but no cudaLaunchKernelEx compatibility alias; [r1079]/[r1136] do not replace this macro; CUDA path stays in the else branch | T4, CUDA preprocessor regression |
| 0001/H07, `cxi_transport.cpp -24`, no-CXI overload/stub | **Unproven necessity on selected CXI path**; build compatibility only | Completes the virtual/overloaded API for builds without USE_LIBFABRIC_CXI; not executed by this image's chosen CXI transport | T2 non-CXI link/import |
| 0001/H08, `cxi_transport.cpp -83`, ROCR HMEM/device union | **Essential** for selected HIP GPU MRs | [r1086] switches runtime helpers to HIP but leaves FI_HMEM_CUDA; [r1137] directly uses registered descriptors, not a substitute registration; header must see __HIP_PLATFORM_AMD__ for host C++ too | T3 and preprocessor macro check |
| 0001/H09, `cxi_transport.cpp -204`, overload and initial selection | **Unproven necessity on identity-mapped full nodes**; compatibility required when mapping differs | [r989] distinguishes physical NIC and runtime ordinal; [r1072] preserves runtime ordinals rather than equating them with BDF order; new [r1139] NUMA lookup makes the original retained patch incomplete for remapping | T2; proposed correction aligns NUMA lookup |
| 0001/H10, `cxi_transport.cpp -226`, retry selection | **Essential conditional companion** to H09 | [r1101] now fetches fresh fi_info on retry; using ctx.local_rank only on retry could switch NICs and violate the initial physical selection | T2 with injected first fi_domain failure |
| 0001/H11, `internode.cu -1439`, dispatch forward-head snapshot | **Redundant compiler workaround on selected10.1; removed** | [r1079]/[r990] retain the upstream volatile shared-head read; joint optimized codegen passes with the original loop, so no local memory-ordering rewrite is justified | T5 passed; HT stress still required |
| 0001/H12, `internode.cu -2577`, combine forwarder-head snapshot | **Redundant compiler workaround on selected10.1; removed** | Joint codegen passes without this and H04; [r1143] changes CPU-proxy QUIET order, not this device load; restore the upstream loop | T5 passed; cached/uncached HT combine required |
| 0001/H13, `internode.cu -2866`, first receiver-head snapshot | **Redundant compiler workaround on selected10.1; removed** | No interval replacement, but exact-stack object compilation handles the original volatile argument; unchanged upstream threshold logic is retained | T5 passed; multi-RDMA-rank combine required |
| 0001/H14, `internode.cu -2994`, second receiver-head snapshot | **Redundant compiler workaround on selected10.1; removed** | Distinct loop/threshold from H13, compiled jointly without all compatibility groups rather than inferred from clean application | T5 passed; backpressure/retirement stress required |
| 0001/H15, `internode.cu -3019`, final forwarder-head snapshot | **Redundant compiler workaround on selected10.1; removed** | Optimized codegen succeeds with the original loop; no local fences/atomics change is introduced | T5 passed; all forwarding channels required |
| 0001/H16, `internode_ll.cu -187`, builtin fabs/fmax | **Redundant compiler workaround on selected10.1; removed**; LL not selected | [r1136] fixes a different AMD barrier; ordinary LL builds without the math rewrite even with all compatibility groups omitted; this is not FP8/NaN/signed-zero numerical acceptance | T5 passed; LL numerical tests only if supported/selected |
| 0001/H17, `proxy.cpp -272`, pass cfg.nic_local_rank | **Unproven necessity on identity-mapped full nodes**; conditional compatibility | Carries [r989]'s split into CXI, not superseded by [r1072]; [r1139] leaves affinity using cfg.local_rank, potentially harmful for placement/performance on remapped devices unless corrected | T2; do not claim arbitrary remapping support |
| 0002/H01, `bench/buffer.py -179`, Torch pool stream and borrowed pointer | **Essential** for selected recorded-tensor lifetime contract | [r957] records tensors/allocations on the communication stream; [r1123] changes event recording, not stream ownership; pool lifetime must outlive Buffer | T6, T7 |
| 0002/H02, `bench/buffer.py -662`, return pool stream | **Essential** integration with H01 | Allocation and record_stream must refer to the same pool stream passed into native code, not a newly wrapped/different handle | T6, T7; stream identity assertions |
| 0002/H03, `uccl_ep.cc -347`, constructor pointer argument | **Essential selected-wrapper integration; potentially harmful to raw positional callers** | Correct wrapper passes the extra handle, but inserting a required argument before byte counts changes the native Python API and can shift old positional arguments; no interval commit fixes that ABI risk | T6, T8; unsupported raw API must not silently be accepted as qualified |
| 0002/H04, `uccl_ep.cc -376`, borrow rather than create stream | **Essential** for selected lifetime contract | [r957] leaves pending allocator stream records beyond native destruction; [r1079]'s new LL calls do not transfer ownership; reject null handle rather than silently using default stream | T6, T7 |
| 0002/H05, `uccl_ep.cc -674`, do not destroy borrowed stream | **Essential**; avoids invalid allocator-event handle | Native destroy still synchronizes and frees only its workspace/IPC/D2H resources; [r1147] proxy atexit cleanup is independent and must not destroy the pool stream | T6, T7, error/atexit subprocesses |
| 0002/H06, `uccl_ep.cc -1968`, nanobind signature/arg names | **Essential selected-wrapper integration; potentially harmful to raw positional callers** | Must match H03/H01 exactly; old direct ep.Buffer calls are not backward-compatible just because deep_ep.Buffer works | T8; a trailing keyword/compatible overload is a possible separate hardening |
| 0002/H07, `uccl_proxy.cpp -155`, destroy(false) | **Essential** for selected DLPack/Torch scratch owner | get_rdma_buffer's DLPack deleter still cudaFree/hipFree's the pointer; Proxy::destroy(true) also frees it; [r1147] adds another stop route, not ownership transfer; [r1135] fixes analogous FifoProxy ownership only | T6 including explicit destroy, GC and atexit paths |

### Ordering and ownership conclusions

The original sequence on both pins is `cleanup -> detect_group_topology/all_gather_object -> create proxies -> peer-metadata exchange -> existing startup barrier -> start proxy threads`.
Every process enters the topology collective after its own cleanup, and cannot finish gathering all ranks' topology objects before every rank entered it.
Consequently a late rank cannot perform its pre-collective cleanup after another rank has started constructing the same group's proxies.
The CPU eight-rank/two-node ordering probe delays the final cleanup on each node and observes no early proxy construction on either unmodified pin.
It records one topology all-gather and one explicit startup barrier per rank upstream, versus the extra explicit barrier introduced by the local leader-cleanup hunk.
This is a host-ordering proof/test, not GPU barrier or CXI acceptance.
Mode-filtered cleanup and the original startup synchronization remain after recommended removal; no global synchronization is being removed from the selected protocol.

The scratch path is `allocate_rdma_buffer_dlpack -> from_dlpack -> Buffer.scratch`, with the DLPack deleter retaining its own ptr/device/free operation.
UcclProxy receives only a raw borrowed address, and Proxy::destroy(true) does not clear the DLPack owner's pointer.
Thus the allocation-ownership patch is not superseded by interpreter-finalization cleanup and does not remove the separate proxy-owned atomic-buffer/ring releases.
At-exit stop now makes failure cleanup more likely to encounter that same borrowed-address ownership bug if left unfixed.
Constructor failure, asynchronous outstanding work and Python finalization still require dedicated tests; normal explicit destruction alone is not proof of all failure paths.

The communication-stream patch keeps `_record_stream_safe`, existing event dependencies and native synchronization intact.
A Torch pool stream avoids destroying the handle later used by the caching allocator when recorded tensor storage is released after Buffer destruction.
It does not replace [r1123]'s side-stream event fix, does not justify removing record_stream calls, and must preserve [r1052]'s one-stream-per-Buffer serialized HT collective ordering.
The native positional API change remains a separately identified compatibility risk outside the selected patched wrapper.

### Required tests and evidence boundaries

The first exact-stack diagnostic attempts used incomplete standalone hipify/header setup and fail their patched controls; they are retained but do not support any patch-necessity conclusion.
`compile-attempt3` instead uses the same `torch.utils.cpp_extension.CUDAExtension` hipify entry point as `ep/setup.py`, with Torch2.14.0+rocm10.1.0, SDK10.1.0 and AMD clang24.0.0git `d6f6cb691863e9258034ab48c1dfce21689d5606`.
Its syntax controls pass for intranode, internode and ordinary LL; separate omission of global min/max, volatile snapshots or builtin math also passes, whereas omission of HIP extended launch fails on unresolved CUDA launch types/functions.
Those individual syntax results do not establish that every compatibility hunk can be removed together or that device code generation succeeds.
`compile-attempt5` completes optimized host/device object compilation using the actual Torch COMMON_HIP_FLAGS/COMMON_HIPCC_FLAGS, `-O3`, C++20, gfx942, four peers and the selected EP defines.
Both the all-retained controls and joint omission of all three groups pass for intranode, internode and ordinary LL, with the device frontend explicitly present in `compiler-driver-jobs.log`.
Omitting HIP extended launch fails with unresolved CUDA launch config/attribute types and `cudaLaunchKernelEx`, confirming that adaptation remains necessary.
This justifies dropping compiler-only kernel/header rewrites from the selected26.8 profile, not changing the vendor stack or asserting GPU numerical correctness.

- **T1 — cleanup ordering:** retained CPU eight-rank probe on both upstream pins and the originally patched version; then repeated fresh two-node HT construction/destruction with a delayed rank, unchanged original mode filter and no concurrent same-UID same-mode job.
- **T2 — topology:** log visible ordinal, PCI BDF, physical/NIC rank, selected CXI domain, retry domain and proxy CPU/NUMA placement; test the selected full-node layout first, with a remapped-device case as separate compatibility qualification, plus a forced initial fi_domain failure and non-CXI compile/link.
- **T3 — registration:** compile host C++ with the exact HIP guard, confirm main and atomic registration use FI_HMEM_ROCR and the runtime device index, then run exact two-node dispatch/combine with canonical libfabric and hybrid-descriptor logs.
- **T4 — launch:** compile the exact10.1 HIP launch APIs and compare grid/block/stream/cooperative attributes with upstream's non-SM90 intent; scheduled GPU checks must include actual HT shapes, occupancy/residency limits and repeated launches, not merely import or a small successful kernel.
- **T5 — compiler ablations:** compile the same pinned source/toolchain/flags with and without min/max, each volatile-snapshot group and builtin device math; no GPU is required for compilation, but numerical/backpressure tests remain necessary before calling device semantics equivalent.
- **T6 — lifetime:** sync/async dispatch/combine, allocation-on-comm-stream both ways, tensor release after Buffer destruction, pool-stream reuse, repeated create/destroy, scratch aliases, subsequent RCCL work, constructor-error and interpreter-exit subprocesses; include the full-node selected HIP path.
- **T7 — stream/event semantics:** retain upstream PR1123 producer/consumer side-stream regression, record_stream identity checks, graph capture/replay where selected, and the single-comm-stream collective ordering tested by PR1052.
- **T8 — native API compatibility:** verify the nanobind signature against the patched wrapper and explicitly test/reject legacy positional native construction, rather than silently interpreting a byte count as a stream pointer; public deep_ep.Buffer qualification does not establish raw ep.Buffer compatibility.

Image compilation with every local workaround present cannot establish that the workarounds are essential; the completed joint ablation is why the selected26.8 patch is smaller than the legacy10.0 patch.
Exact10.1 compiler-ablation results are retained as described above; no scheduled GPU check or numerical acceptance is claimed by this document.
No redundant/harmful finding is repaired by mutating a running context or an immutable published image.

[r985]: https://github.com/uccl-project/uccl/commit/34bdf4f49c8e33d618b983afc34fc5bec8686cfa
[r989]: https://github.com/uccl-project/uccl/commit/c8b236f4baa2a8de0b3e565d847808cfa1e0f022
[r957]: https://github.com/uccl-project/uccl/commit/7696801eda290519bd73e75061f3a3eb43d29612
[r963]: https://github.com/uccl-project/uccl/commit/c1c3c3f32ab017fae9c6d198b8879846aaa04d50
[r1079]: https://github.com/uccl-project/uccl/commit/4358969234a124055a1c1a143374050895f658dc
[r990]: https://github.com/uccl-project/uccl/commit/b4ab198ecc014c6e780354213614c0b10738bb7e
[r1052]: https://github.com/uccl-project/uccl/commit/54e216ee24d59748377578199bf28792fe3ef59e
[r1109]: https://github.com/uccl-project/uccl/commit/53fe8adfa0f43dda7748849e127a654fa252c9ab
[r1101]: https://github.com/uccl-project/uccl/commit/cd0ea9e99aebafd6a92c47d459a9dfcc46e885bc
[r1123]: https://github.com/uccl-project/uccl/commit/1bad71ab85ed261d156b867a448c898ebae53405
[r1147]: https://github.com/uccl-project/uccl/commit/f04b49b2671cafc5cf367c003442be73cfd73845
[r1135]: https://github.com/uccl-project/uccl/commit/161e56321e3df958efc3af33b32412e21ef24a21
[r1139]: https://github.com/uccl-project/uccl/commit/900b8c4cb61015d36da2e3b815bbc94d3df9aa3c
[r1137]: https://github.com/uccl-project/uccl/commit/343c1e4102216951748b50d6fe93045f6d44416a
[r1115]: https://github.com/uccl-project/uccl/commit/96d228accdd4b3c74b020e8be79c3db43d7fdc7c
[r1143]: https://github.com/uccl-project/uccl/commit/e454752cbe809d65a77b071d5f8b5c7bac132ee5
[r1086]: https://github.com/uccl-project/uccl/commit/bc9eff01e073fc24bc50c8ddf0995d92720d82d5
[r1072]: https://github.com/uccl-project/uccl/commit/78adb5dce345c1ba5c52c77c6de8e09bcd4fe4cb
[r1136]: https://github.com/uccl-project/uccl/commit/52d4184c89deb5631be6dac6a9b1c40896c9e26e
