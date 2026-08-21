[![build-images](https://gitlab.com/cscs-ci/ci-testing/webhook-ci/mirrors/4655938191952498/3557919080247023/badges/main/pipeline.svg?ignore_skipped=true&key_text=build-images&key_width=90)](https://gitlab.com/cscs-ci/ci-testing/webhook-ci/mirrors/4655938191952498/3557919080247023/-/pipelines?ref=main)

# Alps Extended Images

Container images that extend NVIDIA NGC and AMD ROCm base images with a fully-optimized HPC networking stack tailored for the [Alps supercomputer](https://www.cscs.ch/computers/alps) at [CSCS](https://www.cscs.ch). The images replace generic bundled communication components with libraries compiled specifically for the Slingshot CXI interconnect, enabling efficient GPU-accelerated collective communication across the Alps fabric.

Image pipeline managed via: https://cicd-ext-mw.cscs.ch

## Overview

Vendor GPU framework images ship with generic HPC libraries that are not optimized for the Slingshot network fabric used on Alps. This project rebuilds the relevant HPC networking stack — libfabric, NCCL/RCCL plugins, UCX, UCC, OpenMPI, and accelerator-specific components — against the CXI provider and installs the result on top of each supported base image.

The resulting images are validated on multi-node Slurm allocations before being promoted to stable registries. CUDA images run on Clariden GH200; ROCm images run on Beverin MI300.

## Image Variants

### CUDA Base Images

Each variant corresponds to an NGC CUDA container extended with the Alps HPC stack:

| Image | NGC Base | Use Case |
|---------|----------|----------|
| `pytorch-cuda:26.06-py3-alps7-dev` | `nvcr.io/nvidia/pytorch:26.06-py3`             | GPU-accelerated PyTorch workloads |
| `pytorch-cuda:26.02-py3-alps7-dev` | `nvcr.io/nvidia/pytorch:26.02-py3`             | GPU-accelerated PyTorch workloads |
| `pytorch-cuda:26.01-py3-alps7-dev` | `nvcr.io/nvidia/pytorch:26.01-py3`             | GPU-accelerated PyTorch workloads |
| `pytorch-cuda:25.12-py3-alps7-dev` | `nvcr.io/nvidia/pytorch:25.12-py3`             | GPU-accelerated PyTorch workloads |
| `nemo-cuda:25.11.01-alps7-dev`     | `nvcr.io/nvidia/nemo:25.11.01`                 | Speech & language model training  |
| `nemo-cuda:26.02-alps7-dev`        | `nvcr.io/nvidia/nemo:26.02`                    | Speech & language model training  |
| `physicsnemo-cuda:25.11-alps7-dev` | `nvcr.io/nvidia/physicsnemo/physicsnemo:25.11` | Physics-informed neural networks  |

### ROCm Base Images

Each variant corresponds to an AMD ROCm container extended with the Alps HPC stack:

| Image | ROCm Base | Use Case |
|-------|-----------|----------|
| `pytorch-rocm:rocm7.14-ubuntu24.04-py3.12-torch2.11-alps7-dev` | `docker.io/rocm/pytorch:rocm7.14_ubuntu24.04_py3.12_pytorch_release_2.11.0` | ROCm PyTorch workloads on MI300-class systems |

### Application Images

Application images are built on top of accelerator-specific base images and include additional software for specific workloads. CUDA app variants are available for the listed workloads; ROCm app variants are added when the app explicitly opts in with its own Containerfile and tests.

| Image | Base | Description |
|-------|------|-------------|
| `apertus-1p5-cuda:alps7-dev` | `pytorch-cuda:26.02-py3` | Megatron-LM distributed LLM pretraining |
| `apertus-2-cuda:alps7-dev`   | `pytorch-cuda:26.02-py3` | Multi-model ML benchmark suite (pplx-garden, DeepEP, quack-kernels) |
| `sfttrainer-cuda:alps7-dev`  | `pytorch-cuda:26.02-py3` | Supervised fine-tuning trainer image |
| `verl-cuda:alps7-dev`        | `pytorch-cuda:26.02-py3` | VeRL reinforcement learning workloads |
| `vllm-cuda:alps7-dev`        | `pytorch-cuda:26.02-py3` | vLLM serving workloads built from source with Alps/NVIDIA PyTorch compatibility patches |
| `vllm-rocm:alps7-dev`        | `pytorch-rocm:rocm7.14-ubuntu24.04-py3.12-torch2.11` | vLLM serving workloads built from source for ROCm/MI300 |

## HPC Stack Components

The base installers use shared defaults from `common/alps-stack-versions.env`, shared build primitives from `common/install-alps-hpc-stack.sh`, plus family-specific wrappers from `NGC/` or `ROCm/`. They purge preinstalled generic network-stack packages/files where appropriate, then build and install the following libraries. Shared primitives export `LIBFABRIC_PREFIX`, `UCX_PREFIX`, `UCC_PREFIX`, `OMPI_PREFIX`, and `HWLOC_PREFIX`; shared aws-ofi plugin logic is configured by the family wrappers with CUDA or ROCm support.

| Component | Version | Purpose |
|-----------|---------|---------|
| libfabric (CXI provider) | 2.6.0 | High-speed network fabric abstraction for Slingshot |
| NCCL | 2.30.7-1 | NVIDIA collective communications (allreduce, alltoall, …) |
| aws-ofi-nccl | 1.20.0 | Routes NCCL traffic over libfabric/OFI |
| NVSHMEM | 3.6.5-0 | GPU symmetric heap memory for peer-to-peer transfers |
| UCX | 1.20.1 | Unified Communication X transport layer |
| UCC | 1.8.0 | Unified Collective Communications abstraction |
| OpenMPI | 5.0.10 | MPI implementation linked against OFI and UCX |
| GDRCopy | 2.5.1 | GPU Direct RDMA copy utilities |
| XPMEM | — | Cross-process memory regions for intra-node GPU sharing |
| NCCL Tests | 2.18.2 | Collective benchmark suite |
| OSU Micro-benchmarks | 7.5.2 | Point-to-point latency and bandwidth measurements |

CUDA components are compiled with CUDA support and architecture-specific flags for NVIDIA Hopper (SM90/SM90a). ROCm components keep the bundled RCCL by default, build aws-ofi-rccl against the Alps libfabric stack, and build rccl-tests for MI250/MI300 targets (`gfx90a`, `gfx942`) using the ROCm SDK from the image profile. RCCL rebuilds are opt-in with `ROCM_REBUILD_RCCL=1`.

Patches for upstream issues in libfabric, NCCL, and aws-ofi-nccl are maintained under `Alps-Images/patches/` and are included in base image hashes. Application-specific patches are kept under `Alps-Images/apps/<app>/patches/` and are included in app image hashes.

## Runtime Environment

Runtime environment fragments configure Slingshot-based collective communication. NGC images assemble `common/alps-runtime.common.env`, `common/alps-runtime.nccl-like.env`, and `NGC/alps-runtime.cuda.env`; ROCm images assemble `common/alps-runtime.common.env`, `common/alps-runtime.nccl-like.env`, and `ROCm/alps-runtime.rocm.env` into `/opt/alps/env/alps-runtime.env`:

- **NCCL/RCCL-compatible defaults**: uses the AWS libfabric transport (`NCCL_NET=AWS Libfabric`) and Slingshot-oriented NIC/socket/protocol/channel tuning
- **NCCL/CUDA-specific defaults**: NVIDIA topology/GDR tuning and CUDA/NVSHMEM settings
- **NCCL plugin cleanup**: clears inherited `NCCL_NET_PLUGIN` settings so NCCL can use the Alps libfabric plugin selected by `NCCL_NET`
- **CXI / libfabric**: provider selection, memory registration caching, rendezvous and RX match-mode settings
- **NVSHMEM**: libfabric remote transport over the Cassini provider, CUDA VMM disabled
- **OpenMPI / PMIX**: security modules, byte transfer layer restricted to supported backends
- **CUDA**: JIT cache disabled for shared-filesystem compatibility

## CI/CD Pipeline

The GitLab entry point is `ci-pipelines/build-alps-extended-images.yaml`. It generates a child pipeline at runtime because the required work depends on registry state: missing canonical images are built, images without a current tested marker are validated, and validated images with stale or missing stable refs are published. If all images are current, the child pipeline contains only a lightweight `no-work-required` job.

The generated child pipeline can contain these stages:

1. **build-base** — builds missing CUDA and ROCm HPC-extended canonical base images
2. **test-base** — validates base images on Slurm allocations with environment checks, collective benchmarks, CUDA `vetnode`, and ROCm/PyTorch smoke tests
3. **mark-base-tested** — creates tested-marker tags for base images that passed the current validation
4. **build-apps** — builds missing application images on top of validated canonical base image refs
5. **test-apps** — runs app workload tests and app `vetnode` coverage declared in app-local `ci.yaml` files
6. **mark-app-tested** — creates tested-marker tags for app images that passed the current validation
7. **publish** — promotes validated images to stable JFrog and GHCR refs while preserving stable-tag immutability rules

**Image tagging strategy:** canonical image tags include deterministic content hashes, allowing the pipeline to detect unchanged inputs and skip unnecessary rebuilds. App hashes include the app accelerator variant, canonical base image ref, selected app Containerfile, `profile.env`, optional declared tests, optional declared app-local patches, and copied shared helper inputs when present. Tested-marker tags have the form `<canonical-tag>-tested-<validation-hash>` and are valid only when they point to the same digest as the canonical image.

## Manual Builds

Images can be built manually on Alps with `podman` using generated build scripts that reuse the CI image-reference logic. See `manual-build/README.md`.

## Acknowledgements

Alps extended base images have been developed in collaboration with the Swiss AI engineers. Special thanks to [@EduardDurech](https://github.com/EduardDurech) for the many contributions ranging from discovering bottlenecks and major bugs to patching underlying libraries.
