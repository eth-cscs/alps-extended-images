# Apertus2 vLLM on Alps

CUDA/GH200 serving image based on `pytorch-cuda:26.07-py3`. The recipe builds
the pinned Apertus2 vLLM fork, UCCL EP/P2P, the `deep_ep` wrapper, and NIXL 1.3.2 with its
UCCL plugin from pinned sources. It preserves the base Torch/CUDA stack.

The source pin is `swiss-ai/vllm@200556e646ff4e6c13dc1b443d973830485f3c0f`
from `apertus2/main`. It includes the MoE/EPLB interface and FP8 SSSGLU DeepGEMM changes.
The wheel version is `0.28.0+apertus2.200556e64`; the commit identifies the fork snapshot.
UCCL is pinned to
`andresnowak/uccl@0c166355310a04b2344ca0c5c0df818b04d26af0`.

## Serving examples

[Chonk and Megachonk examples](example/README.md) provide standalone `sbatch`
Bash launchers using the pinned, tested image and `allgather_reducescatter`.
They retain BF16 defaults and expose optional online FP8 scopes and Chonk EPLB.
The example EDF keeps the previous tested digest. Override `CONTAINER_EDF` to select
a newer compatible image. The moving `alps7-dev` tag requires CI promotion before use.

## Build and publish

Open a PR, then comment:

```text
cscs-ci run build-images
```

The existing pipeline discovers this app through `profile.env`, builds against
the canonical PyTorch 26.07 base, runs vetnode and the GPU import smoke test,
and promotes validated images to JFrog/GHCR under the repository's publish policy.
The stable JFrog ref is:

```text
jfrog.svc.cscs.ch/docker-group-csstaff/alps-images/vllm-apertus2-cuda:alps7-dev
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
