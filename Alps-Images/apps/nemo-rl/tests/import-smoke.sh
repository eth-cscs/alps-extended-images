#!/bin/bash
# Every dependency NeMo-RL resolves at import time must be present in the image —
# the whole point of baking them in is that jobs need no PYTHONPATH overlays.
# Unlike the build-time check, this runs on a GPU node, so the CUDA extensions
# (TransformerEngine, flashinfer, deep_gemm, grouped_gemm, uccl.ep) are asserted
# rather than merely reported.
set -euo pipefail

python3 - <<'PY'
import importlib
import importlib.metadata as md

import torch

assert torch.cuda.is_available(), "no CUDA device visible"
print("torch     ", torch.__version__, "| devices:", torch.cuda.device_count())

EXPECTED = {
    "transformer_engine": "2.17",
    "ray": "2.56.",
    "openai": "2.7.2",
    "megatron-energon": "7.4.0",
    "transformers": "5.8.1",
    "hydra-core": "1.3.2",
    "flashinfer-python": "0.6.18",
}
for pkg, prefix in EXPECTED.items():
    got = md.version(pkg)
    assert got.startswith(prefix), f"{pkg}: expected {prefix}*, got {got}"
    print(f"{pkg:<20} {got}")

REQUIRED = [
    # NeMo-RL core
    "ray", "hydra", "omegaconf", "transformers", "megatron.energon",
    "math_verify", "mlflow", "tensordict", "swanlab", "zstandard", "openai",
    "wandb", "datasets", "accelerate", "torchdata", "tiktoken", "sentencepiece",
    # Megatron generation backend: NeMo-RL hardcodes sampling_backend="flashinfer",
    # so InferenceConfig.__post_init__ raises ImportError without these two.
    "tvm_ffi", "flashinfer",
    # policy / kernel stack
    "transformer_engine.pytorch", "deep_gemm", "grouped_gemm",
    "emerging_optimizers", "fla",
    # MoE token dispatch
    "uccl.ep", "deep_ep",
    # async checkpoint save
    "nvidia_resiliency_ext",
    # the Megatron generation backend serves its OpenAI-compatible endpoint (the one
    # NeMo Gym drives) with Quart under hypercorn
    "quart", "hypercorn",
    # KDA / mamba kernels
    "causal_conv1d", "mamba_ssm", "cutlass", "flash_kda",
    # refit / data-plane transports
    # nccl4py imports as `nccl`; TransferQueue as `transfer_queue`
    "awscrt", "nccl", "nixl", "transfer_queue",
]
for name in REQUIRED:
    importlib.import_module(name)
    print("ok        ", name)

from flashinfer.sampling import top_k_top_p_sampling_from_probs  # noqa: F401
from deep_ep import Buffer  # noqa: F401
print("all imports ok")
PY
