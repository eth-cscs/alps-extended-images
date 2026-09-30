#!/bin/bash
# NeMo-RL drives every worker through Ray, so a job dies at startup if the head
# cannot come up or if a Ray actor fails to see the container interpreter. This
# exercises the same lifecycle the launcher uses: scrub the Slurm topology (Ray
# owns worker placement), start a head, run one GPU actor, stop.
set -euo pipefail

# Per job, so two runners on one node do not share a temp dir.
RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray-nemo-rl-smoke-${SLURM_JOB_ID:-$$}}"
export RAY_TMPDIR
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

rm -rf "${RAY_TMPDIR}"
ray stop --force >/dev/null 2>&1 || true
trap 'ray stop --force >/dev/null 2>&1 || true; rm -rf "${RAY_TMPDIR}"' EXIT

while IFS='=' read -r name _; do
  [[ -n "${name}" ]] || continue
  case "${name}" in PMI*|PMIX*|MPI*|OMPI*|SLURM_*) unset "${name}" ;; esac
done < <(env)

# -I, not -i: -i resolves the hostname through /etc/hosts, which inside a container
# maps to 127.0.0.1, and a head bound to loopback is unreachable from any other node.
node_ip="$(hostname -I | awk '{print $1}')"
[[ -n "${node_ip}" ]] || { echo "no routable address from hostname -I" >&2; exit 1; }

# Without this, num_gpus=0 leaves the actor unschedulable and the test hangs
# instead of reporting the one thing it exists to check.
num_gpus="$(python3 -c 'import torch; print(torch.cuda.device_count())')"
[[ "${num_gpus}" -gt 0 ]] || { echo "no CUDA devices visible to the container" >&2; exit 1; }

ray start --head \
  --disable-usage-stats \
  --node-ip-address="${node_ip}" \
  --port=1200 \
  --num-gpus="${num_gpus}" \
  --include-dashboard=false \
  --min-worker-port=2000 --max-worker-port=2999 \
  --temp-dir="${RAY_TMPDIR}"

python3 - <<'PY'
import ray
import torch

ray.init(address="auto")


@ray.remote(num_gpus=1)
def probe():
    # NEMO_RL_PY_EXECUTABLES_SYSTEM=1 means this actor runs the container
    # interpreter, so the baked-in deps must be importable here too.
    import flashinfer  # noqa: F401
    import transformer_engine.pytorch  # noqa: F401

    x = torch.ones(1024, device="cuda")
    return torch.cuda.get_device_name(0), float(x.sum().item())


name, total = ray.get(probe.remote())
assert total == 1024.0, total
print("ray actor ok on", name)
PY
