# `nemo-rl` container image

This image supports Megatron training and generation with Apertus2 MoE kernels.
It also includes the Apertus2 vLLM fork for generation. The pinned NeMo-RL,
Megatron-Core, Megatron-Bridge and Gym sources are installed in the image,
with release patches already applied. Normal launches need no source checkouts
or runtime patching.

## Examples

See [`example/`](example/README.md) for Chonk GSM8K training with AdamW or Muon/MD,
using Megatron training and either vLLM or Megatron generation.

[`example/vllm/nemo-gym/`](example/vllm/nemo-gym/) uses a remote NeMo Gym math
verifier.

## Build

Run the [manual-build workflow](../../../manual-build/README.md) from the
repository root. This workflow selects the canonical base image.

## Runtime and additions

The image keeps the base image's Torch, Triton and NumPy versions.
It includes the remote Gym client. Gym can still create isolated server venvs
for local environments. See
[`sources/package-additions.in`](sources/package-additions.in) for package pins.

NemoLens 0.1.0 requires Python `>=3.13` in its package metadata. This Python 3.12
image overrides that metadata for the NemoLens wheel only. The SDK and exporter
use the base OpenTelemetry 1.45.1 and protobuf 6.33.6 packages.

The image preloads TorchMemorySaver's CUDA 13 hook. Applications still use its
region/pause/resume API. Set `LD_PRELOAD=` to disable the hook.
NemoRun's TorchX dependency uses the `urllib3` pin in `package-additions.in`.
The image does not include FlashMLA, Diffusers or qwen-vl-utils.

## Source stack and development

[`sources/install-training-stack.sh`](sources/install-training-stack.sh) pins the
source revisions, merges the NeMo fixes and applies [`patches/`](patches/) at build time. The image records source revisions,
patched tree hashes and patch checksums under `/opt/alps/`.

For development, use the launcher's
[development overrides](example/README.md#development-overrides) or external
experiment tools. Select a checkout or ref, PRs and patches for each component.
Put validated changes into the next image release.
Normal launches do not fetch or patch code.

Linters, type checkers and upstream test plugins are not added to the runtime
image. Install them only when needed, for example into a scratch-local venv:

```bash
uv venv --system-site-packages "$SCRATCH/tmp/nemo-rl-tests/venv"
uv pip install --python "$SCRATCH/tmp/nemo-rl-tests/venv/bin/python" \
  -r /opt/tests/nemo-rl/requirements.txt
```

## Smoke-test scope

`/opt/tests/nemo-rl/import-smoke.sh` checks GPU imports and selected packages.
The FlashInfer test runs the cached SM90a FP16 `silu_and_mul` kernel.
The NemoLens test exports a trace in memory.
