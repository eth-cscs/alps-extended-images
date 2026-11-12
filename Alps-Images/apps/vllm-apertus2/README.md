# Apertus2 vLLM on Alps

Apertus2 vLLM serving images built from `swiss-ai/vllm` release tags.
The CUDA/GH200 images use `pytorch-cuda:26.07-py3` and build the Apertus2 vLLM fork, UCCL EP/P2P, the `deep_ep` wrapper, and NIXL 1.3.2 with its UCCL plugin from pinned sources.
The ROCm/MI300A image uses `pytorch-rocm:rocm10.0-ubuntu24.04-py3.12-torch2.13` and follows [`apps/vllm/Containerfile.rocm`](../vllm/Containerfile.rocm): AITER, UCCL-EP over CXI, and Mooncake with the CXI transport for KV transfer.
All images preserve the base Torch stack.

| Target | Image | Containerfile | vLLM tag | Wheel version |
|--------|-------|---------------|----------|---------------|
| `cuda` | `vllm-apertus2-cuda` | `Containerfile` | [`v0.28.0+Apertus2.2`](https://github.com/swiss-ai/vllm/releases/tag/v0.28.0%2BApertus2.2) | `0.28.0+apertus2.2` |
| `v031-cuda` | `vllm-apertus2-v031-cuda` | `Containerfile.v031` | [`v0.31.0+Apertus2.1`](https://github.com/swiss-ai/vllm/releases/tag/v0.31.0%2BApertus2.1) | `0.31.0+apertus2.1` |
| `v031-rocm` | `vllm-apertus2-v031-rocm` | `Containerfile.v031.rocm` | [`v0.31.0+Apertus2.1`](https://github.com/swiss-ai/vllm/releases/tag/v0.31.0%2BApertus2.1) | `0.31.0+apertus2.1` |

The two CUDA Containerfiles differ only in `VLLM_REF` and `VLLM_VERSION_OVERRIDE`.
Each image sets `ALPS_VLLM_APERTUS2_VERSION` to the wheel version, and the build check and smoke test assert it.
vLLM 0.31 uses Model Runner V2 by default; set `VLLM_USE_V2_MODEL_RUNNER=0` to select V1.

All targets pin UCCL to `uccl-project/uccl@52d4184c89deb5631be6dac6a9b1c40896c9e26e` (`main`, 2026-11-11).
It includes the runtime EP configuration overrides (#1056) and the CXI hybrid MR descriptor, proxy pinning, and warp quiet changes (#1137, #1139, #1143).

## ROCm image

The ROCm recipe reuses the `apps/vllm` ROCm build helpers, copied to `sources/` so that this app's content hash covers them.
It applies these `git format-patch` series from `patches/rocm/` with `git apply --check` before `git apply`:

| Directory | Patches | Source |
|-----------|---------|--------|
| `aiter/` | MI300A XCD count and GLM-5.3 tuning tables | Same as `apps/vllm/patches/rocm/aiter` (AITER `v0.1.23`) |
| `vllm/` | DeepEP HT CUDA-graph guard, bundled `triton_kernels`, bench cached tokens, two ROCm 10 build fixes | `apps/vllm/patches/{common,rocm}/vllm`, renumbered; all apply to `v0.31.0+Apertus2.1` |
| `uccl/` | Physical-rank CXI domain selection, shared-barrier cleanup, four ROCm build fixes | `apps/vllm/patches/{common,rocm}/uccl`, renumbered |
| `mooncake/` | CXI topology discovery for the Store | Same as `apps/vllm/patches/common/mooncake` (Mooncake `v0.3.13.post1`) |

UCCL `52d4184c` differs from the `e22a3e8f` pin of `apps/vllm`:

- `0001-cxi-use-physical-gpu-rank.patch` is rebased: upstream still selects `cxi<local_rank>` from the logical device. The rebase also makes the NUMA lookup of #1139 use the physical rank.
- `serialize-cxi-initialization` is dropped: upstream serializes CXI setup across proxy threads (#1101, #1109).
- `rocm-check-runtime-topk-warp-capacity` is dropped: upstream checks the runtime top-k against the warp capacity (#1057).

ROCm tests in `tests-rocm/` check Apertus2 imports, the installed version, UCCL configuration overrides, and an SSSGLU Triton kernel on one GPU.
The Ray/RCCL UCCL-EP, Mooncake CXI, and AITER tests are copies of the `apps/vllm` tests.

## Serving examples

[Chonk and Megachonk examples](example/README.md) provide standalone `sbatch`
Bash launchers using the pinned, tested image and `allgather_reducescatter`.
They retain BF16 defaults and expose optional online FP8 scopes and Chonk EPLB.
The example EDF selects the validated content-hashed image. Override `CONTAINER_EDF`
to select another image. The moving `alps7-dev` tag requires CI promotion before use.

## Build and publish

Open a PR, then comment:

```text
cscs-ci run build-images
```

The existing pipeline discovers this app through `profile.env`, builds against
the canonical PyTorch 26.07 base, runs vetnode and the GPU import smoke test,
and promotes validated images to JFrog/GHCR under the repository's publish policy.
The stable JFrog refs are:

```text
jfrog.svc.cscs.ch/docker-group-csstaff/alps-images/vllm-apertus2-cuda:alps7-dev       # vLLM 0.28
jfrog.svc.cscs.ch/docker-group-csstaff/alps-images/vllm-apertus2-v031-cuda:alps7-dev  # vLLM 0.31
jfrog.svc.cscs.ch/docker-group-csstaff/alps-images/vllm-apertus2-v031-rocm:alps7-dev  # vLLM 0.31, ROCm
```

UCCL defaults to `UCCL_MAX_JOBS=8`. PR #64's follow-up build with 16 jobs failed
with `runtime/cgo: pthread_create failed: Resource temporarily unavailable`.
That trace does not establish an OOM kill. Its builder was an ARM64 Kubernetes
pod with a 400 GiB memory request/limit, not a Slurm CLI allocation. Eight jobs
reduce compiler concurrency; a new CI build must verify the mitigation.

## Runtime and validation

UCCL EP and P2P default to CXI with safe threading. NIXL uses its UCCL plugin
for disaggregated KV transfer. NCCL retains the base Alps AWS Libfabric plugin.
The runtime removes inherited NIXL wheels and installs source-built `nixl-cu13`
plus the `nixl` dispatcher. The build checks CUDA 13 binding selection, package
dependencies, and shared-library linkage. The shipped
GPU test checks Apertus2/GLM imports, UCCL configuration overrides, and NIXL
plugin discovery. It is not a multi-node serving validation.
