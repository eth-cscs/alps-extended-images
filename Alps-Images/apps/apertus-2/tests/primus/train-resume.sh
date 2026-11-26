#!/usr/bin/env bash
# Two-node EP8/PP1 or four-node EP4/PP2 reference-path checkpoint/restart test.
set -euo pipefail
umask 077
export RANK="${SLURM_PROCID:?Slurm ranks required}"
export WORLD_SIZE="${SLURM_NTASKS:?Slurm world required}"
export LOCAL_RANK="${SLURM_LOCALID:?Slurm local rank required}"
pp="${APERTUS_TEST_PP:-1}"
ep="${APERTUS_TEST_EP:-8}"
[[ "${SLURM_NTASKS_PER_NODE:-4}" == 4 ]] || { echo 'T6 needs four consecutive EP ranks per node' >&2; exit 1; }
[[ "$WORLD_SIZE" == "$((8*pp))" && "$((ep%4))" == 0 ]] || exit 1
shared="${APERTUS_TEST_SHARED_ROOT:-$PWD}/apertus2-primus-${SLURM_JOB_ID:?}-${pp}-${ep}"
mkdir -p "$shared/ranks" "$shared/checkpoints"
private=$(mktemp -d /tmp/apertus2-primus-rank.XXXXXXXX)
[[ "$(findmnt -n -o FSTYPE -T "$private")" == tmpfs ]] || { echo 'JIT tests require node-local tmpfs' >&2; exit 1; }
printf '#!/bin/sh\nexit 0\n' > "$private/exec-check"
chmod +x "$private/exec-check"
"$private/exec-check"
export TMPDIR="$private/tmp" XDG_CACHE_HOME="$private/cache"
export TRITON_CACHE_DIR="$private/triton" TORCH_EXTENSIONS_DIR="$private/torch-extensions"
export TORCHINDUCTOR_CACHE_DIR="$private/inductor" AITER_JIT_DIR="$private/aiter"
export MIOPEN_USER_DB_PATH="$private/miopen-db" MIOPEN_CUSTOM_CACHE_DIR="$private/miopen-cache"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" \
    "$TORCHINDUCTOR_CACHE_DIR" "$AITER_JIT_DIR" "$MIOPEN_USER_DB_PATH" "$MIOPEN_CUSTOM_CACHE_DIR"
export MAX_JOBS=2 OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if [[ -z "${MASTER_ADDR:-}" ]]; then
    MASTER_ADDR=$(python3 - <<'PYTHON'
import os
nodes=os.environ['SLURM_JOB_NODELIST']
if '[' in nodes:
    prefix,tail=nodes.split('[',1)
    print(prefix+tail.split(']',1)[0].split(',',1)[0].split('-',1)[0])
else: print(nodes.split(',',1)[0])
PYTHON
)
fi
export MASTER_ADDR
port=$((20000+SLURM_JOB_ID%20000))
args=(
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size "$pp"
    --context-parallel-size 1 --expert-tensor-parallel-size 1 --expert-model-parallel-size "$ep"
    --num-layers 4 --hidden-size 256 --ffn-hidden-size 512 --num-attention-heads 2
    --group-query-attention --num-query-groups 1 --normalization RMSNorm --qk-layernorm --sandwich-norm
    --residual-output-scaling --scale-embeddings-by-sqrt-hidden --untie-embeddings-and-output-weights
    --disable-bias-linear --sssglu --position-embedding-type rope --no-rope-freq 1
    --attention-dropout 0 --hidden-dropout 0 --make-vocab-size-divisible-by 128
    --experimental-attention-variant kda --enable-experimental --linear-attention-freq '[1,1,1,0]'
    --linear-attention-safe-output-gate --attention-output-gate --linear-conv-kernel-dim 4
    --linear-key-head-dim 128 --linear-value-head-dim 128 --linear-num-key-heads 2 --linear-num-value-heads 2
    --attention-backend auto --transformer-impl transformer_engine
    --num-experts 8 --moe-router-topk 2 --moe-ffn-hidden-size 128 --moe-latent-size 128
    --moe-shared-expert-intermediate-size 128 --moe-layer-freq '[0,1,1,1]'
    --moe-grouped-gemm --moe-permute-fusion --moe-token-dispatcher-type alltoall
    --moe-router-score-function sigmoid --moe-router-dtype fp32
    --moe-router-load-balancing-type quantile_balancing seq_aux_loss --moe-aux-loss-coeff 0.0 1e-4
    --bf16 --main-grads-dtype fp32 --optimizer md_decoupling --use-layer-wise-distributed-optimizer
    --overlap-grad-reduce --weight-decay 0 --use-orthogonal-updates --muon-coefficient-type polar_express
    --muon-scale-mode shape_up --muon-router-scale-mode none --muon-momentum 0.95 --muon-use-nesterov
    --hypersphere-mode row --hypersphere-embedding-mode row --hypersphere-router-mode row
    --hypersphere-radius-mode fan_in --md-router-use-orthogonal-updates True
    --hypersphere-gains-mode rowcol --hypersphere-gains-mode-router rowcol --hypersphere-gains-mode-embedding none
    --gain-parametrization softplus --lr 0.0001 --min-lr 0.0001 --matrix-lr 0.001 --gains-lr 0.0001
    --lr-decay-style constant --clip-grad 1 --seed 1234
    --recompute-granularity selective --recompute-modules moe_act layernorm
    --micro-batch-size 1 --global-batch-size 8 --seq-length 128 --max-position-embeddings 128
    --train-iters 4 --exit-interval 2 --cross-entropy-loss-fusion --cross-entropy-fusion-impl te
    --mock-data --tokenizer-type NullTokenizer --vocab-size 1024 --split '100,0,0'
    --num-workers 0 --no-create-attention-mask-in-dataloader
    --eval-iters 0 --eval-interval 1000 --log-interval 1 --distributed-timeout-minutes 10
    --ckpt-format torch --save "$shared/checkpoints" --save-interval 2
)
for stage in 0 1; do
    export MASTER_PORT=$((port+stage))
    extra=()
    iteration=2
    if ((stage)); then extra=(--load "$shared/checkpoints" --exit-on-missing-checkpoint); iteration=4; fi
    timeout --signal=TERM --kill-after=30s 1000s python3 -B -u /opt/tests/apertus-2/primus/train-entry.py \
        --record "$shared/ranks/${RANK}-stage${stage}.json" --iteration "$iteration" "${args[@]}" "${extra[@]}" \
        > "$shared/ranks/${RANK}-stage${stage}.log" 2>&1
    python3 /opt/tests/apertus-2/primus/check-checkpoint.py "$shared" "$RANK" "$stage"
done
printf 'passed\n' > "$shared/ranks/${RANK}.passed"
# CPU/file-only evidence rendezvous after both training processes have returned.
if (( RANK == 0 )); then
    deadline=$((SECONDS+180))
    until [[ "$(find "$shared/ranks" -maxdepth 1 -name '*.passed' | wc -l)" == "$WORLD_SIZE" ]]; do
        ((SECONDS<deadline)) || { echo 'Missing rank acceptance' >&2; exit 1; }
        sleep 1
    done
    python3 /opt/tests/apertus-2/primus/check-checkpoint.py "$shared" all all
fi
printf 'PASS: rank %s reference compute paths, save2/load2/save4; evidence %s\n' "$RANK" "$shared"
