#!/usr/bin/env bash
set -euo pipefail

# Single-node Mooncake smoke test for the CXI (HPE Slingshot) transport.
#
# 1. imports the modules used by vLLM's mooncake KV connectors
# 2. initializes a TransferEngine with the "cxi" protocol
# 3. runs a host-memory self-transfer through the CXI transport
# 4. runs a device-memory self-transfer when a GPU is visible
# 5. runs a Store put/get against a local mooncake_master
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


# The IP only carries Mooncake's TCP handshake/metadata RPC for this
# self-transfer; the data path is CXI, addressed by libfabric. Prefer the
# node's own hostname address, and fall back to the source address of the
# default route (a UDP connect() only does a route lookup and sends nothing,
# so it needs no outbound connectivity).
def get_ip():
    try:
        ip = socket.gethostbyname(socket.gethostname())
        if not ip.startswith("127."):
            return ip
    except OSError:
        pass
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))  # TEST-NET-1, never contacted
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


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

print("OK: mooncake CXI TransferEngine checks passed")
PY

# Store setup() returns 0 even if the CXI transport failed; also scan the log.
tmp="$(mktemp -d)"
master_pid=""
cleanup() {
    [ -n "${master_pid}" ] && kill "${master_pid}" 2>/dev/null || true
    rm -rf "${tmp}"
}
trap cleanup EXIT

export MC_SMOKE_RPC_PORT="${MC_SMOKE_RPC_PORT:-50151}"
MC_SMOKE_METRICS_PORT="${MC_SMOKE_METRICS_PORT:-9103}"
master_bin="$(python -c 'import mooncake, os; print(os.path.join(os.path.dirname(mooncake.__file__), "mooncake_master"))')"

"${master_bin}" --rpc_port="${MC_SMOKE_RPC_PORT}" --metrics_port="${MC_SMOKE_METRICS_PORT}" \
    --logtostderr > "${tmp}/master.log" 2>&1 &
master_pid=$!

if ! python - <<'PY'
import os, socket, sys, time
port = int(os.environ["MC_SMOKE_RPC_PORT"])
deadline = time.time() + 30
while time.time() < deadline:
    try:
        socket.create_connection(("127.0.0.1", port), timeout=1).close()
        sys.exit(0)
    except OSError:
        time.sleep(0.5)
sys.exit(f"mooncake_master did not listen on 127.0.0.1:{port} within 30s")
PY
then
    cat "${tmp}/master.log" >&2
    exit 1
fi
echo "mooncake_master up on 127.0.0.1:${MC_SMOKE_RPC_PORT}"

python - <<'PY' 2>&1 | tee "${tmp}/store.log"
import os
import socket
import sys

try:
    import torch  # noqa: F401  (vLLM's import order)
except ImportError:
    pass

from mooncake.store import MooncakeDistributedStore


def get_ip():
    try:
        ip = socket.gethostbyname(socket.gethostname())
        if not ip.startswith("127."):
            return ip
    except OSError:
        pass
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))  # TEST-NET-1, never contacted
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


store = MooncakeDistributedStore()
ret = store.setup(
    get_ip(),
    "P2PHANDSHAKE",
    64 * 1024 * 1024,   # global_segment_size
    16 * 1024 * 1024,   # local_buffer_size
    "cxi",
    "",
    f"127.0.0.1:{os.environ['MC_SMOKE_RPC_PORT']}",
)
if ret != 0:
    sys.exit(f"mooncake Store setup(protocol=cxi) failed with code {ret}")

value = bytes(range(256)) * 4096  # 1 MiB
ret = store.put("mooncake-smoke-key", value)
if ret != 0:
    sys.exit(f"mooncake Store put failed with code {ret}")
got = store.get("mooncake-smoke-key")
if got != value:
    sys.exit(f"mooncake Store get returned {len(got)} bytes, expected the 1 MiB value")
print("mooncake Store put/get over CXI ok (1 MiB)")
store.close()
PY

if grep -E 'cxi_transport_install_failed|No available CXI devices|cannot initialize CXI resources|Failed to install CXI transport' \
        "${tmp}/store.log"; then
    echo "FAIL: CXI transport did not come up in the Store client (lines above)" >&2
    exit 1
fi

echo "OK: mooncake CXI smoke test passed"
