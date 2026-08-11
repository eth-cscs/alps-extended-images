#!/usr/bin/env bash
set -euo pipefail

NODE_ID="${SLURM_NODEID:-0}"
NUM_NODES="${SLURM_JOB_NUM_NODES:-1}"
GPUS_PER_NODE="${SLURM_GPUS_PER_NODE:-4}"
WORLD_SIZE="${VLLM_RAY_NCCL_WORLD_SIZE:-$((NUM_NODES * GPUS_PER_NODE))}"

HEAD_NODE="$(scontrol show hostnames "${SLURM_JOB_NODELIST:-}" | head -n 1)"
[[ -n "${HEAD_NODE}" ]] || HEAD_NODE="$(hostname)"
NODE_NAME="$(scontrol show hostnames "${SLURM_JOB_NODELIST:-}" | sed -n "$((NODE_ID + 1))p")"
[[ -n "${NODE_NAME}" ]] || NODE_NAME="$(hostname)"

RAY_PORT="${RAY_PORT:-$(( 25000 + (${SLURM_JOB_ID:-1} % 20000) ))}"
MASTER_PORT="${MASTER_PORT:-$(( 45000 + (${SLURM_JOB_ID:-1} % 10000) ))}"

RUN_DIR="${VLLM_RAY_NCCL_RUN_DIR:-${CI_PROJECT_DIR:-${PWD}}/.vllm-ray-nccl-${SLURM_JOB_ID:-ci}}"
LOG_DIR="${RUN_DIR}/logs"
DONE_FILE="${RUN_DIR}/done"
FAIL_FILE="${RUN_DIR}/fail"
mkdir -p "${LOG_DIR}"

HEAD_IP="$(getent hosts "${HEAD_NODE}" | awk 'NR == 1 { print $1 }' || true)"
HEAD_IP="${HEAD_IP:-${HEAD_NODE}}"
NODE_IP="$(getent hosts "${NODE_NAME}" | awk 'NR == 1 { print $1 }' || true)"
NODE_IP="${NODE_IP:-${NODE_NAME}}"

export HEAD_NODE HEAD_IP RAY_PORT MASTER_PORT NUM_NODES WORLD_SIZE FAIL_FILE
export RAY_TMPDIR="/tmp/ray-${SLURM_JOB_ID:-ci}-${NODE_ID}"
export RAY_DEDUP_LOGS="0"

cleanup() {
    local rc=$?
    if [[ "${NODE_ID}" == "0" ]]; then
        [[ "${rc}" -eq 0 ]] || touch "${FAIL_FILE}"
        touch "${DONE_FILE}"
    fi
    ray stop --force >/dev/null 2>&1 || true
    exit "${rc}"
}
trap cleanup EXIT

wait_for_head() {
    local deadline=$((SECONDS + 180))
    while (( SECONDS < deadline )); do
        if python - "${HEAD_IP}" "${RAY_PORT}" <<'PY' >/dev/null 2>&1
import socket
import sys

host, port = sys.argv[1], int(sys.argv[2])
with socket.create_connection((host, port), timeout=2):
    pass
PY
        then
            return 0
        fi
        sleep 2
    done
    echo "ERROR: Ray head ${HEAD_IP}:${RAY_PORT} did not become reachable" >&2
    return 1
}

rm -rf "${RAY_TMPDIR}"
if [[ "${NODE_ID}" == "0" ]]; then
    rm -f "${DONE_FILE}" "${FAIL_FILE}"
    ray start \
        --head \
        --node-ip-address="${NODE_IP}" \
        --port="${RAY_PORT}" \
        --num-gpus="${GPUS_PER_NODE}" \
        --include-dashboard=false \
        --disable-usage-stats \
        --block >"${LOG_DIR}/ray-head-${NODE_ID}.log" 2>&1 &
    RAY_PID=$!
else
    wait_for_head
    ray start \
        --address="${HEAD_IP}:${RAY_PORT}" \
        --node-ip-address="${NODE_IP}" \
        --num-gpus="${GPUS_PER_NODE}" \
        --disable-usage-stats \
        --block >"${LOG_DIR}/ray-worker-${NODE_ID}.log" 2>&1 &
    RAY_PID=$!
fi

if [[ "${NODE_ID}" != "0" ]]; then
    while [[ ! -e "${DONE_FILE}" && ! -e "${FAIL_FILE}" ]]; do
        if ! kill -0 "${RAY_PID}" 2>/dev/null; then
            echo "ERROR: Ray worker exited before the test completed" >&2
            touch "${FAIL_FILE}"
            exit 1
        fi
        sleep 2
    done
    [[ ! -e "${FAIL_FILE}" ]]
    exit 0
fi

python - <<'PY'
import datetime
import os
import socket
import time

import ray

head_ip = os.environ["HEAD_IP"]
ray_port = os.environ["RAY_PORT"]
master_port = os.environ["MASTER_PORT"]
num_nodes = int(os.environ["NUM_NODES"])
world_size = int(os.environ["WORLD_SIZE"])
fail_file = os.environ["FAIL_FILE"]

ray.init(address=f"{head_ip}:{ray_port}")

deadline = time.time() + 300
while True:
    resources = ray.cluster_resources()
    alive_nodes = [node for node in ray.nodes() if node.get("Alive")]
    gpus = int(resources.get("GPU", 0))
    print(f"Ray resources: nodes={len(alive_nodes)} gpus={gpus} resources={resources}")
    if len(alive_nodes) >= num_nodes and gpus >= world_size:
        break
    if os.path.exists(fail_file):
        raise RuntimeError("Ray worker startup failed")
    if time.time() > deadline:
        raise TimeoutError(f"Ray cluster did not reach {num_nodes} nodes and {world_size} GPUs")
    time.sleep(2)


@ray.remote(num_gpus=1)
class Worker:
    def info(self):
        import socket
        from ray.util import get_node_ip_address

        return {
            "host": socket.gethostname(),
            "ip": get_node_ip_address(),
        }

    def run(self, rank, world_size, master_addr, master_port):
        import datetime
        import os
        import socket

        import torch
        import torch.distributed as dist
        import vllm

        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ.setdefault("NCCL_DEBUG", "WARN")

        torch.cuda.set_device(0)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(seconds=180),
        )
        value = torch.ones(1, device="cuda")
        dist.all_reduce(value)
        got = int(value.item())
        dist.destroy_process_group()

        if got != world_size:
            raise AssertionError(f"rank {rank}: expected all_reduce={world_size}, got {got}")

        return {
            "rank": rank,
            "host": socket.gethostname(),
            "torch": torch.__version__,
            "vllm": getattr(vllm, "__version__", "unknown"),
        }


workers = [Worker.remote() for _ in range(world_size)]
placement = ray.get([worker.info.remote() for worker in workers])
hosts = {item["host"] for item in placement}
print(f"vLLM Ray actor placement: {placement}")
if len(hosts) < 2:
    raise AssertionError(f"expected Ray actors on at least 2 hosts, got {hosts}")

master_addr = placement[0]["ip"]
results = ray.get([
    worker.run.remote(rank, world_size, master_addr, master_port)
    for rank, worker in enumerate(workers)
])

print(f"vLLM Ray/NCCL smoke results: {results}")
PY
