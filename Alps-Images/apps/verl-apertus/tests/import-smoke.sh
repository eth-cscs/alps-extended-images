#!/usr/bin/env bash
# Checks, on a GPU node, that the Apertus 1.0 and 1.5 additions of the verl-apertus image are
# installed and wired up: versions, the xIELU CUDA kernel against its Python reference, the
# Megatron bridges, the SGLang Apertus 1.5 model, and the verl patches.
set -euo pipefail

python3 - <<'PY'
import importlib.metadata as m
import pathlib

import torch

expected = {
    "verl": "0.9.1",
    "sglang": "0.5.16",
    "transformers": "5.15.0.dev0",
    "safetensors": "0.8.0",
    "megatron-core": "0.19.0",
    "megatron-bridge": "0.6.1",
    "xielu": "0.1",
}
for dist, version in expected.items():
    installed = m.version(dist)
    assert installed == version, f"{dist} {installed}, expected {version}"
    print(f"{dist} {installed}")

# transformers: both Apertus generations.
from transformers import (  # noqa: E402
    Apertus1p5Config,
    Apertus1p5ForConditionalGeneration,  # noqa: F401
    ApertusForCausalLM,  # noqa: F401
    AutoConfig,
)

assert isinstance(AutoConfig.for_model("apertus1p5"), Apertus1p5Config)

# xIELU: the CUDA kernel is loaded and matches the Python implementation.
from transformers.activations import XIELUActivation  # noqa: E402

act = XIELUActivation(dtype=torch.bfloat16).cuda()
assert act._xielu_cuda_obj is not None, "xIELU CUDA kernel not loaded"
x = torch.randn(4, 256, 4096, device="cuda", dtype=torch.bfloat16)
torch.testing.assert_close(act(x), act._xielu_python(x), rtol=2e-2, atol=2e-2)
print("xIELU CUDA kernel matches the Python reference")

# Megatron-Core and Megatron-Bridge.
import megatron.core  # noqa: E402,F401
from megatron.bridge.models import Apertus1p5Bridge, ApertusBridge  # noqa: E402,F401

# SGLang: the Apertus 1.5 model (PR 32979) imports, and qwen3_asr registers next to the fork's.
import sglang.srt.configs.qwen3_asr  # noqa: E402,F401
from sglang.srt.models.apertus_mm import EntryClass  # noqa: E402

names = {c.__name__ for c in (EntryClass if isinstance(EntryClass, list) else [EntryClass])}
assert "Apertus1p5ForConditionalGeneration" in names, names

from sglang.srt.models.apertus import EntryClass as ApertusEntryClass  # noqa: E402

assert [c.__name__ for c in ApertusEntryClass] == ["ApertusForCausalLM"], ApertusEntryClass

# verl: the text-only forward for Apertus 1.5 keeps other models' vision path.
from types import SimpleNamespace  # noqa: E402

from verl.workers.engine.megatron.transformer_impl import _has_vision_model  # noqa: E402

assert _has_vision_model(Apertus1p5Config()) is False
assert _has_vision_model(SimpleNamespace(model_type="qwen2_5_vl", vision_config={})) is True
assert _has_vision_model(SimpleNamespace(model_type="apertus")) is False

import verl  # noqa: E402

root = pathlib.Path(verl.__file__).parent
assert "repetition_penalty=config.repetition_penalty" in (root / "trainer/ppo/v1/agent_loop_tq.py").read_text()
assert "_finalize_model_grads_with_empty_cache" in (root / "utils/megatron_utils.py").read_text()
print("verl-apertus import smoke test passed")
PY
