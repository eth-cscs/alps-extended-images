#!/usr/bin/env bash
set -euo pipefail

python3 - <<'PY'
import importlib.metadata as metadata
import inspect
import os

import torch

assert torch.cuda.is_available(), "no CUDA device visible"
print("torch", torch.__version__, "cuda", torch.version.cuda)

vllm_version = metadata.version("vllm")
expected_vllm_version = os.environ["ALPS_VLLM_APERTUS2_VERSION"]
assert vllm_version == expected_vllm_version, (vllm_version, expected_vllm_version)
assert metadata.version("nixl") == "1.3.2"
assert metadata.version("runai-model-streamer") == "0.15.7"
assert metadata.version("transformers") == "5.17.0"
print("vllm", vllm_version)

from vllm.config.quantization import QuantizationConfigArgs

expert_only = QuantizationConfigArgs(moe={"weight": "fp8_per_block_static"})
assert expert_only.linear is None
assert expert_only.moe is not None
mlp = QuantizationConfigArgs(
    linear={"weight": "fp8_per_block_static"},
    moe={"weight": "fp8_per_block_static"},
)
assert mlp.linear is not None and mlp.moe is not None

from uccl import ep, p2p
import deep_ep

assert hasattr(ep, "Buffer")
assert hasattr(p2p, "Endpoint")
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
from vllm.model_executor.models.deepseek_v2 import GlmMoeDsaForCausalLM
from vllm.third_party.flash_linear_attention.ops.kda import FusedRMSNormGated

assert Apertus2KDAForCausalLM is not None
assert GlmMoeDsaForCausalLM is not None
assert FusedRMSNormGated is not None

PY

[[ "${UCCL_EP_TRANSPORT:-}" == cxi ]]
[[ "${UCCL_P2P_TRANSPORT:-}" == cxi ]]
[[ "${UCCL_CXI_THREADING:-}" == safe ]]
[[ -f "${NIXL_PLUGIN_DIR:?}/libplugin_UCCL.so" ]]

# UCCL's passive accept threads are process-lifetime. End this isolated probe
# directly after its assertions so Python teardown cannot block the smoke test.
python3 - <<'PY'
import os

import nixl

config = nixl.nixl_agent_config(backends=["UCCL"])
agent = nixl.nixl_agent("apertus2-smoke", config)
plugins = agent.get_plugin_list()
print("NIXL plugins", plugins, flush=True)
assert "UCCL" in plugins
os._exit(0)
PY
