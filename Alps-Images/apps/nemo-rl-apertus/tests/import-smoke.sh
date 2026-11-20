#!/bin/bash
# Every dependency NeMo-RL resolves at import time must be present in the image, and the
# Apertus 1.0/1.5 support added by the image's patches must be registered in transformers,
# vLLM and Megatron-Bridge. Runs on a GPU node, so CUDA extensions are imported, not only
# located. Performs no installs, downloads or model fetches.
set -euo pipefail

python3 - <<'PY'
import importlib
import importlib.metadata as md

import torch

assert torch.cuda.is_available(), "no CUDA device visible"
print("torch     ", torch.__version__, "| devices:", torch.cuda.device_count())

EXPECTED = {
    "transformers": "5.17.0",
    "vllm": "0.33.0+apertus1p5",
    "transformer_engine": "2.18.",
    "ray": "2.58.0",
    "openai": "2.45.0",
    "flashinfer-python": "0.7.2.post1",
    "flashinfer-jit-cache-sm90a": "0.7.2.post1",
    "nvidia-cutlass-dsl": "4.8.0",
    "megatron-energon": "7.4.1",
    "nvidia-resiliency-ext": "0.7.0",
}
for pkg, prefix in EXPECTED.items():
    got = md.version(pkg)
    assert got.startswith(prefix), f"{pkg}: expected {prefix}*, got {got}"
    print(f"{pkg:<28} {got}")

REQUIRED = [
    # Installed sources
    "nemo_rl", "megatron.core", "megatron.bridge", "vllm", "vllm._rust_tool_parser",
    # NeMo-RL runtime
    "ray", "hydra", "omegaconf", "transformers", "megatron.energon", "math_verify",
    "mlflow", "swanlab", "wandb", "tensordict", "zstandard", "datasets", "accelerate",
    "torchdata", "tiktoken", "sentencepiece", "multistorageclient", "nemo_lens",
    # Megatron generation backend: sampling_backend="flashinfer" is hardcoded.
    "tvm_ffi", "flashinfer", "tilelang",
    "quart", "hypercorn",
    # Megatron training stack
    "transformer_engine.pytorch", "modelopt.torch.opt", "nvidia_resiliency_ext",
    "causal_conv1d", "mamba_ssm", "fast_hadamard_transform", "fla", "diffusers", "cutlass",
    # Refit / data-plane transports (nccl4py imports as `nccl`)
    "awscrt", "nccl", "transfer_queue",
]
for name in REQUIRED:
    importlib.import_module(name)
    print("ok        ", name)

from flashinfer.sampling import top_k_top_p_sampling_from_probs  # noqa: F401

# transformers: Apertus 1.0 is in the release, Apertus 1.5 comes from the backport.
from transformers import (  # noqa: F401
    Apertus1p5Config,
    Apertus1p5ForConditionalGeneration,
    Apertus1p5Processor,
    ApertusConfig,
    ApertusForCausalLM,
)
from transformers.models.wavtokenizer import WavTokenizerFeatureExtractor  # noqa: F401

# vLLM: both architectures resolve to a model class.
from vllm.model_executor.models.registry import ModelRegistry

for arch in ("ApertusForCausalLM", "Apertus1p5ForConditionalGeneration"):
    assert arch in ModelRegistry.get_supported_archs(), arch
    print("vLLM arch ", arch, "->", ModelRegistry._try_load_model_cls(arch))
from vllm.reasoning import ReasoningParserManager

assert ReasoningParserManager.get_reasoning_parser("apertus"), "apertus reasoning parser"

# Megatron-Bridge: both bridges register, and the Apertus decoder spec uses the
# xIELU MLP subclass.
from megatron.bridge.models import ApertusBridge  # noqa: F401
from megatron.bridge.models.apertus import APERTUS1P5_AVAILABLE, Apertus1p5Bridge  # noqa: F401
from megatron.bridge.models.apertus.apertus_bridge import ApertusMLP, MCoreXIELU  # noqa: F401

assert APERTUS1P5_AVAILABLE, "Apertus1p5Bridge not registered"
print("all imports ok")
PY
