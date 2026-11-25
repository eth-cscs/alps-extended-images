# `nemo-rl-apertus` container image

This image trains Apertus 1.0 (`ApertusForCausalLM`) and Apertus 1.5 (`Apertus1p5ForConditionalGeneration`) with NeMo-RL, using Megatron training and vLLM generation.
It builds on upstream NeMo-RL, unlike the `nemo-rl` image, which builds on the swiss-ai forks for Apertus 2.
The sources are installed in the image with their patches already applied, so launches need no source checkouts or runtime patching.

## Components

| Component | Revision | Apertus support |
|-----------|----------|-----------------|
| NeMo-RL | `NVIDIA-NeMo/RL` main `83151e70` | None needed. |
| Megatron-Bridge | `ae0a1b02`, NeMo-RL's submodule pin | `patches/megatron-bridge`: NVIDIA-NeMo/Megatron-Bridge#7041 rebased onto that pin. |
| Megatron-Core | `NVIDIA/Megatron-LM` `e0a4bfed`, Megatron-Bridge's submodule pin | None needed: the bridges carry their own xIELU MLP. NVIDIA/Megatron-LM#9633 would make it unnecessary. |
| vLLM | `v0.33.0`, NeMo-RL's pin | `patches/vllm`: vllm-project/vllm#50496 (Apertus 1.5 model and reasoning parser). Apertus 1.0 is upstream. |
| transformers | `v5.17.0` | `patches/transformers`: Apertus 1.5 and WavTokenizer from huggingface/transformers#47662, ported to 5.17.0. Apertus 1.0 is upstream. |

The installed vLLM reports version `0.33.0+apertus1p5`.

The transformers version is held at 5.17.0, one minor release above NeMo-RL's 5.16.1 pin.
vLLM 0.33 and NeMo-RL's Megatron-Bridge pin both require `<5.21`, and huggingface/transformers#47662 is only on transformers main (5.24 development), so no release satisfies them all.
Replace the backport with a release once one contains Apertus 1.5 and the vLLM and Megatron-Bridge ranges admit it.

## Runtime and additions

The image keeps the base image's Torch, Triton and NumPy versions.
Other runtime packages are pinned in [`sources/runtime-requirements.txt`](sources/runtime-requirements.txt), at the versions NeMo-RL main's `uv.lock` resolves.
A few are held at Python 3.12-compatible releases instead.
Ray workers use the container interpreter (`NEMO_RL_PY_EXECUTABLES_SYSTEM=1`), not per-worker venvs.

The image leaves out the Apertus 2 and MoE kernel stack of the `nemo-rl` image: UCCL/DeepEP, DeepGEMM, grouped_gemm, FlashKDA and Emerging-Optimizers.
It also leaves out NeMo Gym, NIXL and Mooncake, which back optional environments and data-plane backends.

NeMo-RL main and NemoLens declare Python `>=3.13` in their package metadata.
This Python 3.12 image overrides that metadata for those source installs only.

The image preloads TorchMemorySaver's CUDA 13 hook.
Set `LD_PRELOAD=` to disable it.

## Build

Run the [manual-build workflow](../../../manual-build/README.md) from the repository root.
This workflow selects the canonical base image.
