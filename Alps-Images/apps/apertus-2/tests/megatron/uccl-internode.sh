#!/usr/bin/env bash
set -euo pipefail

: "${SLURM_PROCID:?requires Slurm}"
: "${SLURM_LOCALID:?requires Slurm}"
: "${SLURM_NTASKS:?requires Slurm}"
: "${SLURM_JOB_NUM_NODES:?requires Slurm}"

if [[ "$SLURM_JOB_NUM_NODES" -ne 2 || "$SLURM_NTASKS" -ne 8 ]]; then
    echo 'Expected two GH200 nodes with four tasks/GPUs per node' >&2
    exit 1
fi

export RANK="$SLURM_PROCID"
export LOCAL_RANK="$SLURM_LOCALID"
export WORLD_SIZE="$SLURM_NTASKS"
export LOCAL_WORLD_SIZE=4

if [[ -z "${MASTER_ADDR:-}" ]]; then
    hosts="$(scontrol show hostnames "$SLURM_JOB_NODELIST")"
    export MASTER_ADDR="${hosts%%$'\n'*}"
fi

export MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 40000))}"
export UCCL_EP_TRANSPORT=cxi
export UCCL_P2P_TRANSPORT=cxi
export UCCL_CXI_THREADING=safe

# Run the dispatch/combine test from the pinned UCCL source.
cd /opt/uccl/ep/bench

exec python3 test_internode.py \
    --num-processes 4 \
    --num-tokens 128 \
    --hidden 512 \
    --num-experts 256 \
    --num-topk 8 \
    --pressure-test-mode 0
