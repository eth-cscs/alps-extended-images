# `nemo-rl` container image

NeMo-RL image for Megatron policy/generation and Apertus2 MoE kernels, with the
Apertus2 vLLM fork available for vLLM generation. NeMo-RL, Megatron-Bridge and
Megatron-LM remain external checkouts: bind-mount them and configure `PYTHONPATH`.

## Example

[`example/`](example/) contains a standalone Chonk GSM8K Slurm/Ray launcher.
It uses Megatron training and vLLM generation with an HF model input. One shell
launcher uses separate YAML and prompt files. See the example
README for pinned sources, batch sizes and the unvalidated memory/refit limits.

## Build

Use the repository's [manual-build workflow](../../../manual-build/README.md)
from the repository root to resolve the canonical base image.

## Runtime and additions

The image installs runtime dependencies against the base image's Torch, Triton
and NumPy pins. NeMo Gym server venvs are still built at runtime. Package pins are
in [`sources/package-additions.in`](sources/package-additions.in), with Hadamard
pinned separately in the Containerfile. Hadamard is built from upstream source;
the source-built FlashInfer JIT cache targets SM90a.

NemoLens 0.1.0 declares Python `>=3.13`; its wheel is installed on this Python
3.12 image with a metadata override for that wheel only. Its SDK/exporter
requirements use the base OpenTelemetry 1.45.1 / protobuf 6.33.6 stack.

TorchMemorySaver's CUDA 13 hook is preloaded by default. Applications still use
its region/pause/resume API; set `LD_PRELOAD=` to disable the hook. NemoRun's
compatibility constraints are in [`sources/nemo-run-constraints.txt`](sources/nemo-run-constraints.txt).
FlashMLA, Diffusers and qwen-vl-utils are excluded.

## Smoke-test scope

`/opt/tests/nemo-rl/import-smoke.sh` runs GPU imports and focused package tests.
The FlashInfer check exercises the cached SM90a FP16 `silu_and_mul` kernel;
NemoLens exports a trace to an in-memory exporter without network access. MathDx
coverage is cuFFTDx only; cuBLASDx and cuSolverDx are not covered. The existing
Megatron-only pins were training-tested on 8 GH200 GPUs; the updated image has
not been built or training-tested.
