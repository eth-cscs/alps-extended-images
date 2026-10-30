#!/usr/bin/env bash
set -euo pipefail

# CI launches one Slurm task per GPU, not one torchrun launcher per node.
: "${SLURM_PROCID:?requires Slurm}"
: "${SLURM_LOCALID:?requires Slurm}"
: "${SLURM_NTASKS:?requires Slurm}"
: "${SLURM_JOB_NUM_NODES:?requires Slurm}"
[[ "$SLURM_JOB_NUM_NODES" -eq 2 && "$SLURM_NTASKS" -eq 8 ]] || {
    echo 'Expected two GH200 nodes with four tasks/GPUs per node' >&2
    exit 1
}
export RANK="$SLURM_PROCID" LOCAL_RANK="$SLURM_LOCALID"
export WORLD_SIZE="$SLURM_NTASKS" LOCAL_WORLD_SIZE=4
if [[ -z "${MASTER_ADDR:-}" ]]; then
    hosts="$(scontrol show hostnames "$SLURM_JOB_NODELIST")"
    export MASTER_ADDR="${hosts%%$'\n'*}"
fi
export MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 40000))}"
export UCCL_EP_TRANSPORT=cxi

# Includes dispatch/combine correctness and tuning from the exact UCCL source
# built into this image. No source checkout or torchrun process tree is needed.
cd /opt/uccl/ep/bench
exec python3 test_internode.py \
    --num-processes 4 \
    --num-tokens 128 \
    --hidden 512 \
    --num-experts 256 \
    --num-topk 8 \
    --pressure-test-mode 0
