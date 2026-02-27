#!/usr/bin/env bash
set -euo pipefail

export RANK="${SLURM_NODEID}"
export WORLD_SIZE="${SLURM_NTASKS}"
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export MASTER_PORT=$(( 20000 + (SLURM_JOB_ID % 40000) ))

cd /opt/DeepEP

export PYTHONPATH=$(pwd)

python tests/test_intranode.py --num-processes=4 --num-experts=256 --num-tokens=8192

NVSHMEM_SYMMETRIC_SIZE=4G \
python tests/test_intranode.py --num-processes=4 --num-experts=256 --num-tokens=8192
