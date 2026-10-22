# `nemo-rl` container image

This image supports Megatron training and generation with Apertus2 MoE kernels.
It also includes the Apertus2 vLLM fork for generation. Keep NeMo-RL,
Megatron-Bridge and Megatron-LM in external checkouts. Bind-mount these checkouts
and set `PYTHONPATH`.

## Examples

See [`example/`](example/README.md) for Chonk GSM8K training with AdamW or Muon/MD,
using Megatron training and either vLLM or Megatron generation.

[`example/vllm/nemo-gym/`](example/vllm/nemo-gym/) uses a remote NeMo Gym math
verifier.

## Build

Run the [manual-build workflow](../../../manual-build/README.md) from the
repository root. This workflow selects the canonical base image.

## Runtime and additions

The image uses the base image's Torch, Triton and NumPy pins for runtime packages.
It creates NeMo Gym server venvs at runtime. See
[`sources/package-additions.in`](sources/package-additions.in) for package pins.

NemoLens 0.1.0 requires Python `>=3.13` in its package metadata. This Python 3.12
image overrides that metadata for the NemoLens wheel only. The SDK and exporter
use the base OpenTelemetry 1.45.1 and protobuf 6.33.6 packages.

The image preloads TorchMemorySaver's CUDA 13 hook. Applications still use its
region/pause/resume API. Set `LD_PRELOAD=` to disable the hook.
See [`sources/nemo-run-constraints.txt`](sources/nemo-run-constraints.txt) for
NemoRun compatibility constraints. The image does not include FlashMLA,
Diffusers or qwen-vl-utils.

## Smoke-test scope

`/opt/tests/nemo-rl/import-smoke.sh` checks GPU imports and selected packages.
The FlashInfer test runs the cached SM90a FP16 `silu_and_mul` kernel.
The NemoLens test exports a trace in memory. It does not use the network.
The MathDx test covers cuFFTDx only. It does not cover cuBLASDx or cuSolverDx.
