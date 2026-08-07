# vLLM App Image

This image currently builds vLLM `v0.25.1` from source on top of `ngc-pytorch:26.02-py3`.

The app-local patches in `patches/` are required for NVIDIA PyTorch `26.02` and `26.03`, whose Torch `2.11.0a0` snapshots do not expose all APIs assumed by vLLM `v0.25.1` after the stable-libtorch migration. In particular, `torch::stable::Tensor::layout()`, the stable `from_blob` deleter overload, and `torch._opaque_base` are missing or incompatible.

Revisit these patches when vLLM moves to Torch `2.12` as its baseline. At that point we may be able to move to a newer NVIDIA PyTorch container and drop some or all compatibility patches.

Older vLLM releases may need fewer changes. Releases `0.22` and older are suspected to work without these compatibility patches, but that still needs to be verified against the Alps CUDA/HPC stack.
