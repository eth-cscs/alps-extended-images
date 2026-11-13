#!/usr/bin/env bash
#SBATCH --job-name=vllm-chonk
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=800G
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/tmp/vllm-chonk-%j.out


# NOTE: The caches for kernels are created only on the container RAM, meaning this makes cold startups slower



set -euo pipefail

# Submit from this directory. FP8=1 and EPLB=1 opt into newer vLLM features.
MODEL=${MODEL:-/capstor/store/cscs/swissai/infra01/apertus_checkpoints/v2/hf/chonk/pretraining/long_context/long-context-iter-0003576}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-65536}
CONTAINER_EDF=${CONTAINER_EDF:-$SLURM_SUBMIT_DIR/../environment.toml}
RUN_DIR=${RUN_DIR:-/iopsstor/scratch/cscs/$USER/tmp/vllm-chonk-$SLURM_JOB_ID}
FP8=${FP8:-0}
FP8_SCOPE=${FP8_SCOPE:-experts}
MOE_BACKEND=${MOE_BACKEND:-triton}
EPLB=${EPLB:-0}
[[ $FP8 == [01] && $EPLB == [01] ]] || { echo 'FP8 and EPLB must be 0 or 1' >&2; exit 2; }
[[ $FP8_SCOPE == experts || $FP8_SCOPE == mlp ]] || { echo 'FP8_SCOPE must be experts or mlp' >&2; exit 2; }

mkdir -p "$RUN_DIR"
exec > >(tee "$RUN_DIR/server.log") 2>&1

[[ -f "$MODEL/config.json" && -f "$CONTAINER_EDF" ]]

mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
head_ip=$(getent ahostsv4 "${nodes[0]}" | awk 'NR == 1 {print $1}')
: "${head_ip:?Could not resolve head node}"
printf 'http://%s:8000\n' "$head_ip" | tee "$RUN_DIR/server-url.txt"

# One Slurm task per node; vLLM owns the internal TP/DP workers, not torchrun or Ray.
srun \
  --nodes="$SLURM_JOB_NUM_NODES" \
  --ntasks="$SLURM_JOB_NUM_NODES" \
  --ntasks-per-node=1 \
  --gpus-per-node=4 \
  --cpus-per-task=288 \
  --mpi=pmix \
  --network=disable_rdzv_get \
  --kill-on-bad-exit=1 \
  --environment="$CONTAINER_EDF" \
  --container-mounts="${SCRATCH}:${SCRATCH},${HOME}:${HOME},/capstor:/capstor,/iopsstor:/iopsstor,/ritom:/ritom" \
  -u bash -s -- "$head_ip" "$MODEL" "$MAX_MODEL_LEN" "$SLURM_JOB_NUM_NODES" "$FP8" "$EPLB" "$FP8_SCOPE" "$MOE_BACKEND" <<'VLLM_ROLE'
set -euo pipefail

head_ip=$1
model=$2
max_model_len=$3
node_count=$4
fp8=$5
eplb=$6
fp8_scope=$7
moe_backend=$8
loader_distributed=true
if [[ $fp8 == 1 ]]; then loader_distributed=false; fi
printf 'Online FP8=%s; scope=%s; EPLB=%s; MoE backend=%s\n' "$fp8" "$fp8_scope" "$eplb" "$moe_backend"

tp=1
local_dp=$((4 / tp))
start_rank=$((SLURM_NODEID * local_dp))

export LOCAL_WORLD_SIZE=4
export CUDA_VISIBLE_DEVICES=0,1,2,3

export XDG_CACHE_HOME="/tmp/vllm-${SLURM_JOB_ID}-${SLURM_NODEID}"
export VLLM_CACHE_ROOT="$XDG_CACHE_HOME/vllm"
export FLASHINFER_WORKSPACE_BASE="$XDG_CACHE_HOME/flashinfer"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/inductor"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/extensions"
export PYTHONPYCACHEPREFIX="$XDG_CACHE_HOME/pycache"
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1

mkdir -p "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

args=(
  "$model"
  --host 0.0.0.0
  --served-model-name chonk

  --tensor-parallel-size "$tp"
  --data-parallel-size "$((node_count * local_dp))"
  --data-parallel-size-local "$local_dp"
  --enable-expert-parallel
  --all2all-backend allgather_reducescatter
  --moe-backend "$moe_backend"

  --dtype bfloat16
  --trust-remote-code
  --numa-bind

  --load-format runai_streamer
  --model-loader-extra-config "{\"distributed\":$loader_distributed,\"concurrency\":4,\"memory_limit\":17179869184}"

  --max-model-len "$max_model_len"
  --gpu-memory-utilization 0.95
  --max-num-seqs 8
  --max-num-batched-tokens 512

  --data-parallel-address "$head_ip"
  --data-parallel-rpc-port 13345
)

if [[ $fp8 == 1 ]]; then
  quant_config='{"moe":{"weight":"fp8_per_block_static"}}'
  if [[ $fp8_scope == mlp ]]; then
    quant_config='{"moe":{"weight":"fp8_per_block_static"},"linear":{"weight":"fp8_per_block_static"},"ignore":["re:^(?!.*\\.mlp\\.(?:experts(?:\\.|$)|(?:shared_experts\\.)?(?:gate_up_proj|gate_proj|up_proj|down_proj)(?:\\.|$))).*"]}'
  fi
  args+=(--quantization online --quantization-config "$quant_config")
fi
if [[ $eplb == 1 ]]; then
  args+=(--enable-eplb --eplb-config '{"num_redundant_experts":0,"use_async":false,"log_balancedness":true}')
fi

if (( start_rank == 0 )); then
  args+=(--api-server-count 1 --port 8000)
else
  args+=(--data-parallel-start-rank "$start_rank" --headless)
fi

exec vllm serve "${args[@]}"
VLLM_ROLE
