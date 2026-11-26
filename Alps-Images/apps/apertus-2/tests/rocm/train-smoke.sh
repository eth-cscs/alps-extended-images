#!/usr/bin/env bash
# One Slurm task, one node, one MI300A GPU. No production launcher or input data.
set -euo pipefail
umask 077

run_dir=$(mktemp -d "${TMPDIR:-/tmp}/apertus2-train.XXXXXXXX")
cleanup() {
    status=$?
    if (( status == 0 )); then
        rm -rf -- "$run_dir"
    else
        printf 'Training smoke failed; private checkpoints/logs retained at %s\n' "$run_dir" >&2
    fi
}
trap cleanup EXIT
export TMPDIR="$run_dir/tmp"
export XDG_CACHE_HOME="$run_dir/cache"
export TRITON_CACHE_DIR="$run_dir/triton"
export TORCH_EXTENSIONS_DIR="$run_dir/torch-extensions"
export TORCHINDUCTOR_CACHE_DIR="$run_dir/torchinductor"
export MIOPEN_USER_DB_PATH="$run_dir/miopen-db"
export MIOPEN_CUSTOM_CACHE_DIR="$run_dir/miopen-cache"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" \
    "$TORCH_EXTENSIONS_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
    "$MIOPEN_USER_DB_PATH" "$MIOPEN_CUSTOM_CACHE_DIR"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NVTE_FUSED_ATTN=1 NVTE_FUSED_ATTN_CK=1 NVTE_FUSED_ATTN_AOTRITON=0
export NVTE_FLASH_ATTN=0 NVTE_UNFUSED_ATTN=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_BLOCKING_WAIT=1

python - <<'PY'
import torch

assert torch.version.hip, f"Expected ROCm PyTorch, got {torch.__version__}"
assert torch.cuda.is_available(), "No HIP GPU available"
arch = torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]
assert arch == "gfx942", f"Expected gfx942, got {arch}"
print(f"Training smoke: torch={torch.__version__}, HIP={torch.version.hip}, GPU={arch}")
PY

# Keep total iterations and LR schedule identical across the two real processes.
# The first exits at iteration 2; the second must load 2 and execute only 3 and 4.
# Dense TE explicitly exercises CK fused bf16 attention; the standalone KDA
# numerical smoke covers KDA independently of this checkpoint/restart workload.
args=(
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1
    --num-layers 2 --hidden-size 256 --ffn-hidden-size 512
    --num-attention-heads 4 --seq-length 128 --max-position-embeddings 128
    --micro-batch-size 1 --global-batch-size 2
    --train-iters 4 --exit-interval 2
    --transformer-impl transformer_engine --attention-backend fused --bf16
    --normalization RMSNorm --position-embedding-type rope
    --attention-dropout 0 --hidden-dropout 0
    --no-gradient-accumulation-fusion --no-persist-layer-norm
    --no-masked-softmax-fusion --no-bias-gelu-fusion
    --no-bias-dropout-fusion --no-rope-fusion
    --optimizer adam --lr 0.0001 --min-lr 0.0001 --lr-decay-style constant
    --weight-decay 0 --clip-grad 1.0 --seed 1234
    --mock-data --tokenizer-type NullTokenizer --vocab-size 1024
    --split "100,0,0" --num-workers 0 --data-cache-path "$run_dir/data-cache"
    --eval-iters 0 --eval-interval 1000 --log-interval 1
    --distributed-timeout-minutes 5
    --ckpt-format torch --save "$run_dir/checkpoints" --save-interval 2
)

check_checkpoint() {
    python - "$run_dir" "$1" <<'PY'
import math
import re
import sys
from pathlib import Path

import torch

root, iteration = Path(sys.argv[1]), int(sys.argv[2])
checkpoint_dir = root / "checkpoints"
tracker = checkpoint_dir / "latest_checkpointed_iteration.txt"
assert tracker.read_text().strip() == str(iteration), tracker.read_text()
checkpoint = checkpoint_dir / f"iter_{iteration:07d}" / "mp_rank_00" / "model_optim_rng.pt"
# This is a trusted checkpoint produced above, not downloaded/user-supplied pickle.
state = torch.load(checkpoint, map_location="cpu", weights_only=False)
assert state["iteration"] == iteration, state["iteration"]
for key in ("model", "optimizer", "opt_param_scheduler", "rng_state"):
    assert key in state and state[key], f"Missing checkpoint state: {key}"
assert state["args"].consumed_train_samples == iteration * 2

log = (root / ("initial.log" if iteration == 2 else "resume.log")).read_text()
expected = [1, 2] if iteration == 2 else [3, 4]
seen = []
for line in log.splitlines():
    match = re.search(r"\biteration\s+(\d+)\s*/\s*4\s*\|", line)
    if not match:
        continue
    seen.append(int(match.group(1)))
    loss = re.search(r"\blm loss:\s*([^\s|]+)", line)
    assert loss and math.isfinite(float(loss.group(1))), f"Nonfinite/missing loss: {line}"
    for label in ("number of skipped iterations", "number of nan iterations"):
        counter = re.search(re.escape(label) + r":\s*(\d+)", line)
        assert counter and int(counter.group(1)) == 0, line
assert seen == expected, f"Expected training iterations {expected}, got {seen}"
if iteration == 4:
    assert re.search(r"successfully loaded checkpoint from .*?at iteration\s+2\b", log), \
        "Resume did not report loading iteration 2"
print(f"Verified finite loss at iterations {seen}; checkpoint iteration={iteration}, optimizer/scheduler/RNG saved")
PY
}

# Fresh torchrun processes also exercise destruction/reinitialization of RCCL/TE.
cd /opt/megatron-lm
timeout --signal=TERM --kill-after=30s 900s \
    torchrun --standalone --nnodes=1 --nproc-per-node=1 pretrain_gpt.py "${args[@]}" \
    2>&1 | tee "$run_dir/initial.log"
check_checkpoint 2
timeout --signal=TERM --kill-after=30s 900s \
    torchrun --standalone --nnodes=1 --nproc-per-node=1 pretrain_gpt.py "${args[@]}" \
    --load "$run_dir/checkpoints" --exit-on-missing-checkpoint \
    2>&1 | tee "$run_dir/resume.log"
check_checkpoint 4
printf 'PASS: native ROCm TE bf16 GPT training saved iteration 2 and resumed through iteration 4\n'
