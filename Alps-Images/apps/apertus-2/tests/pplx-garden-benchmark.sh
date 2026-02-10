#!/usr/bin/env bash
set -euo pipefail

export NUM_NODES="$SLURM_JOB_NUM_NODES"
export NODE_RANK="$SLURM_NODEID"
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export MASTER_PORT=$(( 20000 + (SLURM_JOB_ID % 40000) ))
echo "NUM_NODES=$NUM_NODES NODE_RANK=$NODE_RANK MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"

export FI_CXI_ENABLE_WRITEDATA=1

cd /opt/pplx-garden

python3 -m benchmarks.bench_all_to_all \
    --world-size $((NUM_NODES * 4)) \
    --nets-per-gpu 1 \
    --init-method=env:// \
    --node-rank="$NODE_RANK" \
    --nvlink=4 \
    --output=./log_output \
    --dp-size 1 \
    --max-num-tokens 128 \
    --max-private-tokens 128 \
    --num-experts 256  \
    --hidden-dim 7168 \
    --hidden-dim-scale 56 \
    --num-experts-per-token 8 \
    --in-dtype=float16 \
    --out-dtype=float16 \
    --scale-dtype=float32 \
    --num-warmup 20 \
    --num-repeats 30
