# vLLM App Image

This image builds vLLM `v0.29.0` from source on top of `pytorch-cuda:26.07-py3` (CUDA) and `pytorch-rocm:rocm10.0-ubuntu24.04-py3.12-torch2.13` (ROCm).

vLLM `v0.29.0` targets Torch `2.13`, which exposes the stable-libtorch APIs (`Tensor::layout()`, the `from_blob` deleter overload, `torch._opaque_base`) that the previous NVIDIA Torch `2.11.0a0` snapshots were missing. The compatibility patches that earlier versions needed are therefore gone.
The ROCm image installs `amd-aiter` `0.1.19` from its pinned upstream commit and sets both `PYTORCH_ROCM_ARCH` and `GPU_ARCHS` to `gfx942`, the ISA used by MI300A and MI300X. AITER's full build-time precompilation is disabled because its FlyDSL AOT path ignores `GPU_ARCHS` and compiles targets from multi-architecture CSV data. AITER remains enabled in vLLM and JIT-compiles required kernels for `gfx942` at runtime.

The ROCm variant keeps a GPU-less-safe build-time smoke check: `import vllm` runs the platform probe and the module-level GCN-arch query, which initializes CUDA and fails on build containers without a visible GPU. The build only imports torch and locates the vLLM package via `importlib.util.find_spec`; the real `import vllm` coverage lives in the ray-rccl test job.

Older vLLM releases may need fewer changes. Releases `0.22` and older are suspected to work without these compatibility patches, but that still needs to be verified against the Alps CUDA/HPC stack.
