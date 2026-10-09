#!/bin/bash
# Every dependency NeMo-RL resolves at import time must be present in the image —
# the whole point of baking them in is that jobs need no PYTHONPATH overlays.
# Unlike the build-time check, this runs on a GPU node, so the CUDA extensions
# (TransformerEngine, flashinfer, deep_gemm, grouped_gemm, uccl.ep) are asserted
# rather than merely reported.
set -euo pipefail

python3 - <<'PY'
import ctypes
import importlib
import importlib.metadata as md
from pathlib import Path

import torch

assert torch.cuda.is_available(), "no CUDA device visible"
print("torch     ", torch.__version__, "| devices:", torch.cuda.device_count())

EXPECTED = {
    "transformer_engine": "2.17",
    "ray": "2.56.",
    "openai": "2.7.2",
    "megatron-energon": "7.4.0",
    "transformers": "5.17.0",
    "hydra-core": "1.3.2",
    "flashinfer-python": "0.6.16.post3",
    "apache-tvm-ffi": "0.1.11",
    "tilelang": "0.1.12",
    "nvidia-cutlass-dsl": "4.6.2",
    "vllm": "0.28.0+apertus2",
    "runai-model-streamer": "0.15.7",
    "uccl": "0.1.1",
    "deep-ep": "0.1.0",
    "nixl": "1.3.2",
    "nixl-cu13": "1.3.2",
    "nccl-extensions": "0.1.0",
    "nvidia-nccl-cu13": "2.30.7",
}
for pkg, prefix in EXPECTED.items():
    got = md.version(pkg)
    assert got.startswith(prefix), f"{pkg}: expected {prefix}*, got {got}"
    print(f"{pkg:<20} {got}")

REQUIRED = [
    # NeMo-RL core
    "ray", "hydra", "omegaconf", "transformers", "megatron.energon", "vllm",
    "math_verify", "mlflow", "tensordict", "swanlab", "zstandard", "openai",
    "vllm._rust_tool_parser",
    "wandb", "datasets", "accelerate", "torchdata", "tiktoken", "sentencepiece",
    # Megatron generation backend: NeMo-RL hardcodes sampling_backend="flashinfer",
    # so InferenceConfig.__post_init__ raises ImportError without these two.
    "tvm_ffi", "flashinfer", "tilelang",
    # policy / kernel stack
    "transformer_engine.pytorch", "deep_gemm", "grouped_gemm",
    "emerging_optimizers", "fla",
    # MoE token dispatch
    "uccl.ep", "uccl.p2p", "deep_ep",
    # async checkpoint save
    "nvidia_resiliency_ext",
    # the Megatron generation backend serves its OpenAI-compatible endpoint (the one
    # NeMo Gym drives) with Quart under hypercorn
    "quart", "hypercorn",
    # KDA / mamba kernels
    "causal_conv1d", "mamba_ssm", "cutlass", "flash_kda",
    # refit / data-plane transports
    # nccl4py imports as `nccl`; TransferQueue as `transfer_queue`
    "awscrt", "nccl", "nccl.ep", "nccl.m2n", "nixl", "transfer_queue",
]
for name in REQUIRED:
    importlib.import_module(name)
    print("ok        ", name)

import nccl.ep as nccl_ep
import nccl.m2n as nccl_m2n

# Query the runtime: PyTorch's version helper can report its build-time headers.
nccl_lib = ctypes.CDLL("libnccl.so.2")
nccl_lib.ncclGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
nccl_lib.ncclGetVersion.restype = ctypes.c_int
nccl_version = ctypes.c_int()
assert nccl_lib.ncclGetVersion(ctypes.byref(nccl_version)) == 0
assert nccl_version.value == 23007, nccl_version.value
# The 0.1.0 Python wheel bundles the independently versioned EP library 0.2.0.
assert str(nccl_ep.get_lib_version()) == "0.2.0", nccl_ep.get_lib_version()
ep_lib = nccl_ep.get_lib_path()
assert ep_lib and ep_lib.is_file() and "cu13" in ep_lib.parts, ep_lib
m2n_lib = Path(nccl_m2n.__file__).parent / "lib/cu13/libnccl_m2n.so"
assert m2n_lib.is_file(), m2n_lib
ctypes.CDLL(str(m2n_lib))
print("NCCL extensions CUDA 13:", ep_lib, m2n_lib)

from flashinfer.sampling import top_k_top_p_sampling_from_probs  # noqa: F401
from uccl import ep, p2p
from deep_ep import Buffer

assert hasattr(ep, "Buffer")
assert hasattr(p2p, "Endpoint")
for name in (
    "get_low_latency_rdma_size_hint", "low_latency_dispatch", "low_latency_combine",
    "get_dispatch_layout", "dispatch", "combine",
):
    assert hasattr(Buffer, name), name

import nixl

assert nixl._bindings.__name__ == "nixl_cu13._bindings", nixl._bindings.__name__
assert not any(
    d.metadata["Name"].lower().replace("_", "-") == "nixl-cu12"
    for d in md.distributions()
)

from vllm.model_executor.models.apertus2 import Apertus2KDAForCausalLM
from vllm.third_party.flash_linear_attention.ops.kda import FusedRMSNormGated

print("Apertus2 vLLM:", Apertus2KDAForCausalLM, FusedRMSNormGated)
print("all imports ok")
PY

[[ "${UCCL_EP_TRANSPORT:-}" == cxi ]]
[[ "${UCCL_P2P_TRANSPORT:-}" == cxi ]]
[[ "${UCCL_CXI_THREADING:-}" == safe ]]
[[ -f "${NIXL_PLUGIN_DIR:?}/libplugin_UCCL.so" ]]

# Package additions include native GPU checks (FWHT, CuPy, cuFFTDx) and local
# fixtures only; the script performs no installs, downloads, or model fetches.
python3 /opt/tests/nemo-rl/package-additions-smoke.py

# As in the vLLM test, isolate the process-lifetime UCCL accept threads so Python
# teardown cannot hang this probe after its assertions have passed.
python3 - <<'PY'
import os
import nixl

config = nixl.nixl_agent_config(backends=["UCCL"])
agent = nixl.nixl_agent("nemo-rl-smoke", config)
plugins = agent.get_plugin_list()
print("NIXL plugins", plugins, flush=True)
assert "UCCL" in plugins
os._exit(0)
PY
