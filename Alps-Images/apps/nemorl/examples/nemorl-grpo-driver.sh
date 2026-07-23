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
# -----------------------------------------------------------------------------
if [ ! -d "${LOCAL_MODEL_DIR}" ]; then
    echo "Downloading ${MODEL_REPO}/${MODEL_NAME} to ${LOCAL_MODEL_DIR}..."
    HF_DOWNLOAD_DIR="${HOME}/tmp/hf_download_${SLURM_JOB_ID}"
    mkdir -p "${HF_DOWNLOAD_DIR}"
    pushd "${HF_DOWNLOAD_DIR}" >/dev/null || exit 1

    if [[ -n "${HF_TOKEN_PATH}" && -f "${HF_TOKEN_PATH}" ]]; then
        export HF_TOKEN="$(cat "${HF_TOKEN_PATH}")"
    fi

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
    --environment="${TRAINING_CONFIG}/env.toml" \
    --container-writable bash -c '

unset PYTHONOPTIMIZE

# If a token file was passed, read it into the environment as a fallback for
# libraries that do not natively honour HF_TOKEN_PATH.
if [[ -n "${HF_TOKEN_PATH}" && -f "${HF_TOKEN_PATH}" && -z "${HF_TOKEN}" ]]; then
    export HF_TOKEN="$(cat "${HF_TOKEN_PATH}")"
    export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi

cd "${NEMORL_DIR}"

# Initializing ray to ensure that it is in cash (which the container should have ensured) and
#  will load fast when we start the container.
uv run ray --version >/dev/null 2>&1 || true

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

    # Rank 0 starts the Ray head node.
    uv run ray start --head \
        --node-ip-address="${MASTER_NODE_IP}" \
        --port="${PORT}" \
        --num-cpus="${SLURM_CPUS_PER_TASK}" \
        --num-gpus=4 \
        --disable-usage-stats || true

    # Signal that the Ray head is ready.
    touch "${TRAINING_CONFIG}/ray_open_${SLURM_JOB_ID}"

    # Wait until all worker nodes have joined.
    while true; do
        alive_nodes=$(uv run ray status | awk "/Active:/{flag=1;next}/Pending:/{flag=0}flag" | grep -c "node_" || true)
        if ! [[ "$alive_nodes" =~ ^[0-9]+$ ]]; then
            alive_nodes=0
        fi
        if [ "$alive_nodes" -ge "$SLURM_JOB_NUM_NODES" ]; then
            break
        fi
        echo "Rank 0: waiting for all nodes to join [$alive_nodes/$SLURM_JOB_NUM_NODES]"
        sleep 5
    done

    # Run the NeMo-RL training driver.
    uv run python examples/run_grpo.py --config "${TRAINING_CONFIG}/${YAML_NAME}"

    # Gracefully stop the Ray cluster. This lets the worker nodes exit their
    # ray start --block cleanly instead of being killed by srun teardown,
    # which otherwise prints "Ray subprocesses exited unexpectedly" messages.
    echo "Rank 0: stopping Ray cluster..."
    uv run ray stop --grace-period 30 || true

    sleep 5s
else
    # Worker ranks wait for the Ray head to be ready, then join.
    RAY_OPEN_FILE="${TRAINING_CONFIG}/ray_open_${SLURM_JOB_ID}"
    echo "Rank ${SLURM_PROCID}: waiting for Ray head..."
    while [ ! -f "${RAY_OPEN_FILE}" ]; do
        sleep 1
    done

    uv run ray start \
        --address="${RAY_ADDRESS}" \
        --node-ip-address="$(hostname -i)" \
        --num-cpus="${SLURM_CPUS_PER_TASK}" \
        --num-gpus=4 \
        --block || true
fi
'
