# Alps Extended Images

Container images that extend NVIDIA NGC base images with a fully-optimized HPC networking stack tailored for the [Alps supercomputer](https://www.cscs.ch/computers/alps) at [CSCS](https://www.cscs.ch). The images replace the bundled HPC-X components in NGC containers with libraries compiled specifically for the Slingshot CXI interconnect, enabling efficient GPU-accelerated collective communication across the Alps fabric.

Image pipeline managed via: https://cicd-ext-mw.cscs.ch

---

## Overview

NVIDIA NGC images ship with generic HPC libraries that are not optimized for the Slingshot network fabric used on Alps. This project rebuilds the full HPC networking stack — libfabric, NCCL, NVSHMEM, UCX, UCC, OpenMPI, and their transitive dependencies — against the CXI provider and installs the result on top of each supported NGC base image.

The resulting images are validated on multi-node Slurm allocations (clariden-gh200) before being promoted to stable registries.

---

## Image Variants

### NGC Base Images

Each variant corresponds to an NGC container extended with the Alps HPC stack:

| Variant | NGC Base | Use Case |
|---------|----------|----------|
| `pytorch-25.12-py3` | `nvcr.io/nvidia/pytorch:25.12-py3` | GPU-accelerated PyTorch workloads |
| `pytorch-26.01-py3` | `nvcr.io/nvidia/pytorch:26.01-py3` | GPU-accelerated PyTorch workloads |
| `nemo-25.11.01` | `nvcr.io/nvidia/nemo:25.11.01` | Speech & language model training |
| `physicsnemo-25.11` | `nvcr.io/nvidia/physicsnemo/physicsnemo:25.11` | Physics-informed neural networks |
| `physicsnemo-26.03` | `nvcr.io/nvidia/physicsnemo/physicsnemo:26.03` | Physics-informed neural networks |

### Application Images

Application images are built on top of the NGC base images and include additional software for specific workloads:

| Image | Base | Description |
|-------|------|-------------|
| `apertus-1p5` | `pytorch-26.01-py3` | Megatron-LM distributed LLM pretraining |
| `apertus-2` | `pytorch-26.01-py3` | Multi-model ML benchmark suite (pplx-garden, DeepEP, quack-kernels) |

---

## HPC Stack Components

The `common/install-alps-hpc-stack.sh` script builds and installs the following libraries:

| Component | Version | Purpose |
|-----------|---------|---------|
| libfabric (CXI provider) | commit `102872c` | High-speed network fabric abstraction for Slingshot |
| NCCL | 2.29.3-1 | NVIDIA collective communications (allreduce, alltoall, …) |
| aws-ofi-nccl | custom | Routes NCCL traffic over libfabric/OFI |
| NVSHMEM | 3.4.5-0 | GPU symmetric heap memory for peer-to-peer transfers |
| UCX | 1.19.1 | Unified Communication X transport layer |
| UCC | 1.6.0 | Unified Collective Communications abstraction |
| OpenMPI | 5.0.9 | MPI implementation linked against OFI and UCX |
| GDRCopy | 2.5.1 | GPU Direct RDMA copy utilities |
| XPMEM | — | Cross-process memory regions for intra-node GPU sharing |
| NCCL Tests | — | Collective benchmark suite |
| OSU Micro-benchmarks | 7.5.2 | Point-to-point latency and bandwidth measurements |

All components are compiled with CUDA support (auto-detected) and architecture-specific flags for NVIDIA Hopper (SM90/SM90a).

Patches for upstream issues in libfabric, NCCL, and aws-ofi-nccl are maintained under `patches/`.

---

## Runtime Environment

`common/alps-runtime.env` configures the runtime environment for Slingshot-based collective communication:

- **NCCL**: uses the AWS libfabric transport (`NCCL_NET=AWS Libfabric`), protocol tuning, 4 channels per peer
- **CXI / libfabric**: provider selection, memory registration caching, rendezvous and RX match-mode settings
- **NVSHMEM**: libfabric remote transport over the Cassini provider, CUDA VMM disabled
- **OpenMPI / PMIX**: security modules, byte transfer layer restricted to supported backends
- **CUDA**: JIT cache disabled for shared-filesystem compatibility

---


## CI/CD Pipeline

The GitLab CI pipeline (`ci-pipelines/build-alps-extended-images.yaml`) runs five stages:

1. **build-base** — builds NGC+HPC extended base images; uses content hashing to skip unchanged variants
2. **test-base** — validates base images on 2–4 node Slurm allocations:
   - environment variable checks (FI_PROVIDER, NCCL settings)
   - collective benchmarks (NCCL alltoall, NVSHMEM latency, OSU bandwidth)
   - hardware verification via the `vetnode` framework
3. **build-apps** — builds application images on top of promoted base images
4. **test-apps** — runs end-to-end workload tests:
   - `apertus-1p5`: Megatron pretraining (2 nodes, 8 GPUs)
   - `apertus-2/pplx-garden`: perplexity garden benchmarks (2 nodes, 2 GPUs)
   - `apertus-2/DeepEP`: DeepEP benchmarks (1 node, 1 GPU)
5. **publish** — promotes all tested images to stable registries; overwrites are blocked on existing stable tags

**Image tagging strategy:** each image name encodes a SHA256 hash of its source files, allowing the pipeline to detect unchanged inputs and skip unnecessary rebuilds.

---

## Acknowledgements

Alps extended base images have been developed in collaboration with the Swiss AI engineers. Special thanks to [@EduardDurech](https://github.com/EduardDurech) for the many contributions ranging from discovering bottlenecks and major bugs to patching underlying libraries.
