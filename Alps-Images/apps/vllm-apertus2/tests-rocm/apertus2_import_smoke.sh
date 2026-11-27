#!/usr/bin/env bash
# Single-GPU smoke test for the ROCm Apertus2 vLLM image: the installed fork
# version, Apertus2/KDA imports, UCCL-EP and deep_ep configuration overrides,
# and one SSSGLU Triton kernel launch on the GPU.
set -euo pipefail

[[ "${UCCL_EP_TRANSPORT:-}" == cxi ]]

python3 - <<'PY'
import importlib.metadata as metadata
import inspect
import os

import torch

assert torch.cuda.is_available(), "no GPU visible"
assert torch.version.hip, "torch is not a ROCm build"
print("torch", torch.__version__, "hip", torch.version.hip)

vllm_version = metadata.version("vllm")
expected_vllm_version = os.environ["ALPS_VLLM_APERTUS2_VERSION"]
assert vllm_version == expected_vllm_version, (vllm_version, expected_vllm_version)
print("vllm", vllm_version)

from vllm.config.quantization import QuantizationConfigArgs

expert_only = QuantizationConfigArgs(moe={"weight": "fp8_per_block_static"})
assert expert_only.linear is None
assert expert_only.moe is not None

from uccl import ep
import deep_ep

assert hasattr(ep, "Buffer")
for name in (
    "get_low_latency_rdma_size_hint",
    "low_latency_dispatch",
    "low_latency_combine",
    "get_dispatch_layout",
    "dispatch",
    "combine",
):
    assert hasattr(deep_ep.Buffer, name), name
print("deep_ep.Buffer", inspect.signature(deep_ep.Buffer.__init__))

os.environ["UCCL_EP_DISPATCH_CONFIG"] = "24,12,512,32,512"
os.environ["UCCL_EP_COMBINE_CONFIG"] = "24,2,512,24,512"
dispatch = deep_ep.Buffer.get_dispatch_config(8)
combine = deep_ep.Buffer.get_combine_config(8)
assert dispatch.num_sms == 24
assert dispatch.num_max_rdma_chunked_send_tokens == 32
assert combine.num_sms == 24
assert combine.num_max_rdma_chunked_send_tokens == 24

from vllm.model_executor.models.apertus2 import Apertus2KDAForCausalLM
from vllm.models.kimi_k3.amd.ops.third_party.kda import chunk, fused_recurrent
from vllm.third_party.flash_linear_attention.ops.kda import FusedRMSNormGated

assert Apertus2KDAForCausalLM is not None
assert chunk is not None and fused_recurrent is not None
assert FusedRMSNormGated is not None

from vllm.model_executor.layers.activation import sssglu_and_mul_triton

x = torch.randn(7, 2 * 576, dtype=torch.bfloat16, device="cuda")
out = torch.empty(7, 576, dtype=torch.bfloat16, device="cuda")
sssglu_and_mul_triton(out, x)
gate, up = x.float().chunk(2, dim=-1)
ref = ((torch.nn.functional.softsign(gate - 1.0) + 0.5) * up).to(torch.bfloat16)
torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
print("OK: SSSGLU Triton kernel matches the reference on", torch.cuda.get_device_name())
PY

echo "OK: Apertus2 vLLM ROCm import smoke test passed"
