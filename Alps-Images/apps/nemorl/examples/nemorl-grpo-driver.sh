#!/bin/bash

# Shared NeMo-RL GRPO driver for Alps.
#
# This script is sourced by the train-*.sh entry points after they have
# exported all required variables and generated the training configuration
# (env.toml + YAML config).
#
# Required environment (set by the caller):
#   TRAINING_HOME, TRAINING_CONFIG, CHECKPOINT_HOME, LOCAL_MODEL_DIR
#   MODEL_NAME, MODEL_REPO, HF_TOKEN_PATH
#   NEMORL_DIR, NEMORL_IMAGE
#   MASTER_NODE_IP, PORT, RAY_ADDRESS
#   YAML_NAME

set -euo pipefail

mkdir -p "${TRAINING_HOME}"
mkdir -p "${TRAINING_CONFIG}"
cd "${TRAINING_HOME}"

# -----------------------------------------------------------------------------
# Download the model once to shared storage.
#
# We check not only that the directory exists but also that a real (non-LFS-
# pointer) config.json is present inside it. A previous run may have created
# the directory but failed mid-download (network error, LFS not fetched,
# etc.), leaving an incomplete model that `AutoTokenizer.from_pretrained`
# later chokes on with a misleading "You need to have sentencepiece or
# tiktoken installed" error.
# -----------------------------------------------------------------------------
_model_complete=false
if [ -d "${LOCAL_MODEL_DIR}" ] && [ -f "${LOCAL_MODEL_DIR}/config.json" ]; then
    # An LFS pointer file starts with "version https://git-lfs.github.com/spec/v1".
    # A real config.json starts with "{" (JSON).
    if head -c 1 "${LOCAL_MODEL_DIR}/config.json" | grep -q '{'; then
        _model_complete=true
    fi
fi

if [ "${_model_complete}" = false ]; then
    if [ -d "${LOCAL_MODEL_DIR}" ]; then
        echo "WARNING: ${LOCAL_MODEL_DIR} exists but appears incomplete (missing or LFS-pointer config.json). Re-downloading." >&2
        rm -rf "${LOCAL_MODEL_DIR}"
    else
        echo "Downloading ${MODEL_REPO}/${MODEL_NAME} to ${LOCAL_MODEL_DIR}..."
    fi
    HF_DOWNLOAD_DIR="${HOME}/tmp/hf_download_${SLURM_JOB_ID}"
    mkdir -p "${HF_DOWNLOAD_DIR}"
    pushd "${HF_DOWNLOAD_DIR}" >/dev/null || exit 1

    uvx hf download "${MODEL_REPO}/${MODEL_NAME}" --local-dir "${LOCAL_MODEL_DIR}"

    popd >/dev/null || true
    rm -rf "${HF_DOWNLOAD_DIR}"
else
    echo "Model already present at ${LOCAL_MODEL_DIR}, skipping download."
fi

# -----------------------------------------------------------------------------
# Start the Ray cluster manually (verl-style).
#
# NeMo-RL's run_grpo.py only starts a single-node local cluster automatically;
# for multi-node it expects an existing Ray cluster. This explicit head + worker
# setup is therefore required here, just as it is for verl.
#
# Note: NeMo-RL's canonical launcher is ray.sub, which also injects topology-
# aware Ray resources (nvlink_domain_*, topo_rank, worker_units). This manual
# start works, but without those extras NeMo-RL falls back to generic instead of
# topology-aware scheduling. For a first comparison run this is acceptable.
# -----------------------------------------------------------------------------
srun --mpi=pmix --network=disable_rdzv_get -N "${SLURM_JOB_NUM_NODES}" --ntasks-per-node=1 -u \
    --kill-on-bad-exit=1 \
    --environment="${TRAINING_CONFIG}/env.toml" \
    --container-writable bash -c '

unset PYTHONOPTIMIZE

# -----------------------------------------------------------------------------
# Salvage Ray error logs before the node-local tmpfs vanishes with the job.
#
# Workers that die silently mid-import (slurm-3046402 ReplayBuffer SIGSEGV,
# slurm-3079454 MegatronPolicyWorker) leave their only trace in
# /tmp/ray/session_*/logs/worker-*.err on the node — gone at teardown.  Copy
# non-empty error files to Lustre on exit (also on SIGTERM from
# --kill-on-bad-exit; slurm allows a grace period before SIGKILL).
# -----------------------------------------------------------------------------
salvage_ray_logs() {
    _dst="${TRAINING_HOME}/ray_err_logs/${SLURM_JOB_ID}"
    mkdir -p "${_dst}" 2>/dev/null || return 0
    find /tmp/ray/session_latest/logs /tmp/ray/session_*/logs -maxdepth 1 \
        \( -name "worker-*.err" -o -name "raylet.err" -o -name "gcs_server.err" \) \
        -size +0c 2>/dev/null | sort -u | head -50 | while read -r _f; do
        cp "${_f}" "${_dst}/$(hostname)_$(basename "${_f}")" 2>/dev/null || true
    done
}
trap salvage_ray_logs EXIT TERM

# -----------------------------------------------------------------------------
# Bind host memory to the CPU (LPDDR) NUMA nodes.
#
# On GH200 each GPU exposes its HBM as a memory-only NUMA node.  Under LPDDR
# pressure the kernel places host pages there — including cudaHostAlloc
# PINNED pages, which are unevictable — invisibly consuming GPU memory that
# NVML attributes to no process (the "phantom", HANDOFF.md §8,
# slurm-3044145).  Binding to the CPU-bearing nodes prevents this; CUDA
# device allocations go through the GPU driver and are unaffected by the
# host mempolicy.  The node list is detected, not hardcoded: LPDDR nodes
# are the ones with a non-empty cpulist.
# -----------------------------------------------------------------------------
NUMACTL=""
MEMBIND_NODES=""
for _node_dir in /sys/devices/system/node/node[0-9]*; do
    if [ -f "${_node_dir}/cpulist" ] && [ -n "$(tr -d "[:space:]" < "${_node_dir}/cpulist")" ]; then
        _node_id="${_node_dir##*/node}"
        MEMBIND_NODES="${MEMBIND_NODES:+${MEMBIND_NODES},}${_node_id}"
    fi
done
# Fallback if sysfs is not available in the container: parse `numactl -H`
# ("node <N> cpus: <list>" — CPU-bearing nodes have a non-empty list).
if [ -z "${MEMBIND_NODES}" ] && command -v numactl >/dev/null 2>&1; then
    MEMBIND_NODES="$(numactl -H 2>/dev/null | awk "/^node [0-9]+ cpus:/ { if (NF > 3) printf \"%s%s\", (n++ ? \",\" : \"\"), \$2 }")"
fi
if command -v numactl >/dev/null 2>&1 && [ -n "${MEMBIND_NODES}" ]; then
    NUMACTL="numactl --membind=${MEMBIND_NODES}"
    echo "Rank ${SLURM_PROCID}: binding host memory to NUMA nodes ${MEMBIND_NODES}"
    if [ "${SLURM_PROCID}" -eq 0 ]; then
        numactl -H || true
    fi
else
    echo "WARNING: Rank ${SLURM_PROCID}: numactl unavailable or no CPU NUMA nodes detected (MEMBIND_NODES=\"${MEMBIND_NODES}\"); running WITHOUT membind — the phantom-memory mitigation is inactive." >&2
fi

# If a token file was passed, read it into the environment as a fallback for
# libraries that do not natively honour HF_TOKEN_PATH.
if [[ -n "${HF_TOKEN_PATH}" && -f "${HF_TOKEN_PATH}" && -z "${HF_TOKEN}" ]]; then
    export HF_TOKEN="$(cat "${HF_TOKEN_PATH}")"
    export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi

cd "${NEMORL_DIR}"

git remote remove fork 2>/dev/null || true
git remote add fork "${NEMORL_FORK_URL}"
git fetch fork "${NEMORL_BRANCH}"
git switch -C "${NEMORL_BRANCH}" "fork/${NEMORL_BRANCH}"

# Prepare the uv environment with retries, and GATE on success.  A single
# flaky fetch (e.g. the flash-attn wheel from GitHub) otherwise kills
# ray start on this node silently and the job hangs forever at N-1/N nodes
# joined (slurm-3045644).  All later uv invocations use --no-sync so the
# network is never touched again after this point.
_uv_ready=false
for _attempt in 1 2 3 4 5; do
    if uv run ray --version >/dev/null 2>&1; then
        _uv_ready=true
        break
    fi
    echo "Rank ${SLURM_PROCID}: uv env preparation attempt ${_attempt}/5 failed; retrying in 30s..." >&2
    sleep 30
done
if [ "${_uv_ready}" != true ]; then
    echo "FATAL: Rank ${SLURM_PROCID}: uv environment could not be prepared after 5 attempts; aborting job." >&2
    exit 1
fi

# Signal that this rank has finished its uv setup.
RANK_DONE_FILE="${TRAINING_CONFIG}/rank_${SLURM_JOB_ID}_${SLURM_PROCID}_done"
touch "${RANK_DONE_FILE}"

if [ "${SLURM_PROCID}" -eq 0 ]; then
    # Wait until every rank has finished its uv setup.
    for other_rank in $(seq 1 $((SLURM_JOB_NUM_NODES - 1))); do
        OTHER_DONE_FILE="${TRAINING_CONFIG}/rank_${SLURM_JOB_ID}_${other_rank}_done"
        echo "Rank 0: waiting for rank ${other_rank} uv setup..."
        while [ ! -f "${OTHER_DONE_FILE}" ]; do
            sleep 1
        done
    done

    # Rank 0 starts the Ray head node (under membind: raylet children — all
    # Ray actors — inherit the memory policy).
    ${NUMACTL} uv run --no-sync ray start --head \
        --node-ip-address="${MASTER_NODE_IP}" \
        --port="${PORT}" \
        --num-cpus="${SLURM_CPUS_PER_TASK}" \
        --num-gpus=4 \
        --disable-usage-stats || true

    # Signal that the Ray head is ready.
    touch "${TRAINING_CONFIG}/ray_open_${SLURM_JOB_ID}"

    # Wait until all worker nodes have joined — with a timeout, so a single
    # node that failed to start Ray aborts the job instead of hanging at
    # N-1/N for the whole allocation (slurm-3045644).
    _join_waited=0
    while true; do
        alive_nodes=$(uv run --no-sync ray status | awk "/Active:/{flag=1;next}/Pending:/{flag=0}flag" | grep -c "node_" || true)
        if ! [[ "$alive_nodes" =~ ^[0-9]+$ ]]; then
            alive_nodes=0
        fi
        if [ "$alive_nodes" -ge "$SLURM_JOB_NUM_NODES" ]; then
            break
        fi
        if [ "${_join_waited}" -ge 1800 ]; then
            echo "FATAL: Rank 0: only ${alive_nodes}/${SLURM_JOB_NUM_NODES} nodes joined the Ray cluster after 30 min; aborting job." >&2
            exit 1
        fi
        echo "Rank 0: waiting for all nodes to join [$alive_nodes/$SLURM_JOB_NUM_NODES]"
        sleep 5
        _join_waited=$((_join_waited + 5))
    done

    # Run the NeMo-RL training driver (membind for consistency; its own
    # allocations are small, the heavy lifting is in the Ray actors).
    ${NUMACTL} uv run --no-sync python examples/run_grpo.py --config "${TRAINING_CONFIG}/${YAML_NAME}"

    # Gracefully stop the Ray cluster. This lets the worker nodes exit their
    # ray start --block cleanly instead of being killed by srun teardown,
    # which otherwise prints "Ray subprocesses exited unexpectedly" messages.
    echo "Rank 0: stopping Ray cluster..."
    uv run --no-sync ray stop --grace-period 30 || true

    sleep 5s
else
    # Worker ranks wait for the Ray head to be ready, then join.
    RAY_OPEN_FILE="${TRAINING_CONFIG}/ray_open_${SLURM_JOB_ID}"
    echo "Rank ${SLURM_PROCID}: waiting for Ray head..."
    while [ ! -f "${RAY_OPEN_FILE}" ]; do
        sleep 1
    done

    ${NUMACTL} uv run --no-sync ray start \
        --address="${RAY_ADDRESS}" \
        --node-ip-address="$(hostname -i)" \
        --num-cpus="${SLURM_CPUS_PER_TASK}" \
        --num-gpus=4 \
        --block || true
fi
'
