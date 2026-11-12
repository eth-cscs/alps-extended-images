# vLLM App Image

This image builds vLLM `v0.31.0` from source on top of `pytorch-cuda:26.07-py3` (CUDA) and `pytorch-rocm:rocm10.0-ubuntu24.04-py3.12-torch2.13` (ROCm).

vLLM `v0.31.0` targets Torch `2.13`, which exposes the stable-libtorch APIs (`Tensor::layout()`, the `from_blob` deleter overload, `torch._opaque_base`) that the previous NVIDIA Torch `2.11.0a0` snapshots were missing.
Three narrow compatibility patches remain: both images disable CUDA graphs for the DeepEP high-throughput backend before the compilation-mode early return and prefer vLLM's complete bundled `triton_kernels` tree over the incomplete package in the NGC base; the CUDA image requires both CuteDSL and QuACK before enabling the optional `ll_bf16` kernel.
The ROCm image installs the matching upstream `amd-aiter` `0.1.23` release from its pinned commit and sets both `PYTORCH_ROCM_ARCH` and `GPU_ARCHS` to `gfx942`, the ISA used by MI300A and MI300X.
AITER's full build-time precompilation is disabled because its FlyDSL AOT path ignores `GPU_ARCHS` and compiles targets from multi-architecture CSV data.
The image instead prebuilds AITER's core module without requiring a visible GPU and stores it in `/opt/aiter-jit`; required `gfx942` kernels remain available through runtime JIT.
The image also prebuilds TVM FFI's ROCm Torch DLPack addon into `/opt/alps/tvm-ffi`, avoiding concurrent Ray-worker compilation into a shared home-directory cache.

The default AITER target is MI300A: `CU_NUM=228` is set at build time and in the runtime environment because `GPU_ARCHS=gfx942` alone selects AITER's 304-CU MI300X default.
An MI300X build must use `--build-arg CU_NUM=304`; an explicit runtime `CU_NUM` override is also honored and must match the GPU.
The AITER CI test checks the selected CU count against the visible device.

Both accelerator variants build UCCL's expert-parallel extension from upstream commit `e22a3e8fcd636f7efbbd90900458087a41ff73ce`, install its `uccl.ep` module and DeepEP-compatible `deep_ep` wrapper, and default the transport to Slingshot CXI.
CXI domains are selected from each worker's node-wide physical GPU rank rather than its process-local visible-device index, so Ray workers that each see one logical GPU use the corresponding `cxi0` through `cxi3` device.
The CUDA image builds the native extension for SM90, explicitly selects its Grace Hopper memory paths even when container builds cannot see a GPU, and uses libfabric's `FI_HMEM_CUDA` interface.
The ROCm image applies five app-local compatibility fixes for the ROCm 10/CXI stack: use `FI_HMEM_ROCR` for AMD GPU allocations, select HIP's extended kernel-launch types and API, restore device math overloads removed in ROCm 10, snapshot volatile shared-memory heads before minimum comparisons, and check the runtime top-k rather than the compile-time maximum against MI300's wave64 dispatch capacity.
Both variants serialize stale per-node UCCL shared-memory barrier cleanup before proxy construction, preventing ranks from unlinking barriers created by peers.
They also serialize CXI provider initialization within each worker process because UCCL creates transports from several proxy threads and concurrent `fi_getinfo` calls can corrupt libfabric allocator state.
Mooncake keeps the base-matched RDMA development packages through compilation, then deliberately purges `libibverbs-dev`, `libnl-3-dev`, `libnl-route-3-dev`, and any inherited `librdmacm-dev`.
RDMA CM headers are not a new installation requirement.
Final cleanup rejects collateral package removals and does not run autoremove; required runtime libraries remain.

The CUDA image also installs Run:ai Model Streamer `0.16.1`, verifying the architecture-specific release wheel against a pinned SHA256.
UCCL-EP remains its expert-parallel implementation; NIXL is not installed.
vLLM's `nvidia-cutlass-dsl[cu13]` and `quack-kernels` requirements are retained, enabling the low-latency BF16 router without crossing incompatible CuteDSL APIs.
The build checks their installed versions against the selected vLLM source's requirements instead of fixed assertions tied to one release.

Both builds constrain Python package installation to the Torch and Triton versions supplied by their canonical base images; the CUDA build also preserves base-provided TorchVision and TorchAudio versions.
A final image check rejects dependency resolution that replaces any of those packages.
Runtime Python ignores user-site packages (`PYTHONNOUSERSITE=1`).
Build-time package installation uses `package-helpers.sh`, which routes pip through the CSCS PyPI proxy without forcing that index on image users.
Image labels record the pinned UCCL revision and, where applicable, the AITER revision and Model Streamer version.

The ROCm app preserves the base SDK's runtime library search order when registering linker paths, while builds use the base-selected development prefix.

`COMMON_PATCH_DIR=patches/common`, `CUDA_PATCH_DIR=patches/cuda`, and `ROCM_PATCH_DIR=patches/rocm` in `profile.env` are recursive hash roots, not patch-application commands.
Component subdirectories (`aiter`, `mooncake`, `uccl`, and `vllm`) separate patches for different source trees; the Containerfiles copy the relevant component sets, applying common patches before accelerator-specific patches.
AITER patches live under `patches/rocm/aiter/` and are applied before building the ROCm image's AITER wheel and core module.
Patch-content changes under one accelerator root invalidate only that accelerator's image and validation hashes; shared patch changes affect both.

The Containerfiles copy each component's patches immediately before its build.
Common Python tooling, AITER (ROCm), UCCL, vLLM, Model Streamer (CUDA), and Mooncake have separate build steps, with their version arguments declared where first needed.
A vLLM-only patch therefore preserves cached dependency builds; changing an earlier component still rebuilds later layers.
Accelerator-specific source lists in `profile.env` keep changes to `sources/install-aiter.sh` and the ROCm build helper out of CUDA hashes.

Each compiled-component step installs and removes newly introduced build-only APT packages in the same layer.
This repeats package setup on cold builds but avoids retaining deleted compiler packages in lower image layers.
Cleanup normalizes both snapshot and requested package names to preserve inherited development packages regardless of architecture qualifiers; the CUDA base's matching RDMA headers remain available until Mooncake's explicit final purge.
Only `libibverbs1`, not `libibverbs-dev`, is held during component cleanup.
Base-provided Torch/Triton constraints and ROCm SDK shell state are restored for each component.

The CUDA vLLM source build uses 64 build jobs and four NVCC threads; its UCCL extension build uses 16 jobs.
Alps CI builders reserve 400 GiB.
Smaller builders should reduce `MAX_JOBS` and `NVCC_THREADS` with build-argument overrides; `UCCL_MAX_JOBS` is independently configurable.

The two-node CUDA and ROCm CI tests import both UCCL modules, construct a low-latency UCCL-EP buffer across eight GPUs, and verify an identity dispatch/combine round trip.
This exercises CXI endpoint setup, GPU-memory registration, and the UCCL data path in addition to the Ray NCCL/RCCL all-reduce.

The ROCm variant keeps a GPU-less-safe build-time smoke check: `import vllm` runs the platform probe and the module-level GCN-arch query, which initializes CUDA and fails on build containers without a visible GPU.
The build only imports torch and locates the vLLM package via `importlib.util.find_spec`; the real ROCm `import vllm` coverage lives in the `ray-rccl-uccl-ep` test job.

Older vLLM releases may need fewer changes.
Releases `0.22` and older are suspected to work without these compatibility patches, but that still needs to be verified against the Alps CUDA/HPC stack.

## AITER (ROCm variant only)

The ROCm variant installs [AITER](https://github.com/ROCm/aiter), AMD's AI Tensor Engine for ROCm, as a non-editable wheel built by `sources/install-aiter.sh` from the exact revision recorded in the image label.
The wheel carries the sources required by AITER's runtime JIT; the build checkout is removed.
AITER provides MLA prefill/decode, fused MoE, RMS norm, and intra-node all-reduce kernels, including the ROCm sparse-MLA implementations required by models such as GLM-5.x.

vLLM defaults `VLLM_ROCM_USE_AITER` to false.
Set `VLLM_ROCM_USE_AITER=1` in the EDF or job environment to enable its AITER paths.
Kernels not already present in the preseeded image cache are JIT-compiled with `hipcc` on first use.

`AITER_JIT_DIR` defaults to the writable `/opt/aiter-jit` cache, which contains AITER's prebuilt core module and accepts additional runtime-compiled kernels.
The ROCm Ray/UCCL CI job imports AITER from this cache.
The ROCm-only `aiter` CI job verifies that the image cache is writable, copies it to a private test cache, resolves vLLM's AITER MLA backend, JIT-compiles an RMS-norm kernel, and compares its output with Torch without polluting the shared cache.

The patches in `patches/rocm/aiter/` are applied to the pinned AITER sources at build time, in name order, until they are merged upstream:

- `0001-mi300a-xcd-count.patch`: derive the XCD count from the GPU (6 on MI300A) instead of assuming 8.
- `0002`/`0003`: tuned fused MoE and blockscale GEMM configs for GLM-5.3 FP8 on MI300A (gfx942, 228 CUs).
  AITER merges them from `aiter/configs/model_configs/` automatically.
  The pinned resolver honors `CU_NUM=228` alongside `GPU_ARCHS=gfx942`; the image sets both, so disabling `GPU_ARCHS` is not required.

## Mooncake (CXI KV transfer)

Both variants additionally install the [Mooncake](https://github.com/kvcache-ai/Mooncake) transfer engine with the HPE Slingshot (CXI) backend (`USE_CXI`, [kvcache-ai/Mooncake#2535](https://github.com/kvcache-ai/Mooncake/pull/2535)).
It is built from a pinned tag by `sources/install-mooncake.sh` (which is part of the app content hash) and installed as the `mooncake-transfer-engine` wheel, providing the `mooncake.engine` and `mooncake.store` modules used by vLLM's `MooncakeConnector` and `MooncakeStoreConnector` KV connectors.

Mooncake is compiled against the libfabric installed by the Alps base image (`/usr`), so it always matches the stack's libfabric version and shares a single in-process libfabric instance with the aws-ofi-nccl NCCL plugin.
This is deliberate: a second, bundled libfabric would race for the CXI devices and leave one consumer with an empty provider list.

`patches/common/mooncake/` fixes CXI device discovery in the Mooncake store client, which otherwise comes up without a transport.

On Slingshot, select the CXI transport in the extra config instead of the default `rdma` protocol, e.g.:

```bash
vllm serve <model> \
    --kv-transfer-config \
    '{"kv_connector": "MooncakeConnector", "kv_role": "kv_producer",
      "kv_connector_extra_config": {"mooncake_protocol": "cxi"}}'
```

`device_name` may be left empty; Mooncake auto-discovers CXI devices.
The image also ships the `mooncake_master` store service and the `transfer_engine_bench` utility for transport debugging.
A single-node CXI smoke test (`mooncake_smoke.sh`) runs in CI for both the transfer engine and the store.
