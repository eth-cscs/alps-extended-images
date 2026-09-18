#!/usr/bin/env bash
set -euo pipefail

# Single-node Mooncake smoke test for the CXI (HPE Slingshot) transport.
#
# 1. imports the modules used by vLLM's mooncake KV connectors
# 2. initializes a TransferEngine with the "cxi" protocol
# 3. runs a host-memory self-transfer through the CXI transport
# 4. runs a device-memory self-transfer when a GPU is visible
#
# A single Slurm task is enough: the engine uses P2PHANDSHAKE metadata and
# targets itself, so no second node or metadata server is required.

python - <<'PY'
import ctypes
import socket
import sys

try:
    import torch
except ImportError:
    torch = None


def get_ip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return socket.gethostbyname(socket.gethostname())


# Import torch before mooncake, matching vLLM. On rocm the reverse order maps
# the devel ROCm SDK tree next to torch's core tree and aborts in LLVM.
from mooncake.engine import TransferEngine
from mooncake.store import MooncakeDistributedStore  # noqa: F401

local_ip = get_ip()
engine = TransferEngine()
ret = engine.initialize(local_ip, "P2PHANDSHAKE", "cxi", "")
if ret != 0:
    sys.exit(f"mooncake CXI transport initialization failed with code {ret}")

rpc_port = engine.get_rpc_port()
target_name = f"[{local_ip}]:{rpc_port}" if ":" in local_ip else f"{local_ip}:{rpc_port}"
print(f"mooncake CXI engine initialized, targeting {target_name}")

size = 1 << 20  # 1 MiB
src = (ctypes.c_char * size).from_buffer_copy(b"A" * size)
dst = (ctypes.c_char * size)()

assert engine.register_memory(ctypes.addressof(src), size) == 0
assert engine.register_memory(ctypes.addressof(dst), size) == 0
ret = engine.transfer_sync_write(
    target_name, ctypes.addressof(src), ctypes.addressof(dst), size
)
if ret != 0:
    sys.exit(f"mooncake CXI host-memory transfer failed with code {ret}")
if bytes(dst[:256]) != b"A" * 256:
    sys.exit("mooncake CXI host-memory transfer corrupted data")
print("mooncake CXI host-memory self-transfer ok (1 MiB)")
engine.unregister_memory(ctypes.addressof(src))
engine.unregister_memory(ctypes.addressof(dst))

if torch is not None and torch.cuda.is_available():
    numel = 32 * 1024 * 1024  # 32 MiB of uint8
    src_tensor = torch.full((numel,), 42, dtype=torch.uint8, device="cuda")
    dst_tensor = torch.zeros(numel, dtype=torch.uint8, device="cuda")
    assert engine.register_memory(src_tensor.data_ptr(), src_tensor.nbytes) == 0
    assert engine.register_memory(dst_tensor.data_ptr(), dst_tensor.nbytes) == 0
    ret = engine.transfer_sync_write(
        target_name, src_tensor.data_ptr(), dst_tensor.data_ptr(), src_tensor.nbytes
    )
    if ret != 0:
        sys.exit(f"mooncake CXI device-memory transfer failed with code {ret}")
    torch.cuda.synchronize()
    if not bool((dst_tensor == 42).all().item()):
        sys.exit("mooncake CXI device-memory transfer corrupted data")
    print(f"mooncake CXI device-memory self-transfer ok ({src_tensor.nbytes // (1 << 20)} MiB)")
    engine.unregister_memory(src_tensor.data_ptr())
    engine.unregister_memory(dst_tensor.data_ptr())
else:
    print("no CUDA/HIP device visible; skipped device-memory transfer")

print("OK: mooncake CXI smoke test passed")
PY
