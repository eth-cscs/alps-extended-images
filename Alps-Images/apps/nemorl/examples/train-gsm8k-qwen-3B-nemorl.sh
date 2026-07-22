#!/bin/bash

#SBATCH --nodes=8
#SBATCH --account=csstaff
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=8:00:00

# Specific settings
export NEMORL_IMAGE="/capstor/scratch/cscs/phimuell/.uenv-images/__ML__/nemo_rl_2026_07_21_I.sqsh"
export HF_TOKEN_PATH="${HOME}/.hf/clariden_token"

#================================================


# W&B is enabled in the NeMo-RL config below. Make sure the API key is exported
# in the environment before submitting; it is forwarded into the container.
export WANDB_API_KEY="${WANDB_API_KEY:-}"
if [[ -z "$WANDB_API_KEY" ]]; then
    echo "[WARNING] WANDB_API_KEY is not set; W&B logging will fail or run in offline mode." >&2
fi

# HuggingFace token: strongly recommended for multi-node runs. Without it, all
# policy workers download the model unauthenticated from the HF Hub and easily
# hit rate limits, causing stragglers that fail the torch distributed rendezvous.
# Prefer HF_TOKEN_PATH so the secret is not embedded in the job script/env.toml.
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-${HUGGINGFACE_TOKEN_PATH:-}}"
if [[ -z "$HF_TOKEN_PATH" ]]; then
    echo "[WARNING] HF_TOKEN_PATH is not set; HuggingFace downloads may be rate-limited on many workers." >&2
fi

export MODEL_NAME="Qwen2.5-3B-Instruct"
export MODEL_REPO="Qwen"
export WANDB_PROJECT_NAME="async-grpo-gsm8k"
export WANDB_RUN_NAME="${MODEL_NAME}-nemorl-vllm-dtensor-fsdp2-sync-${SLURM_JOB_NUM_NODES}n-${SLURM_JOB_ID}"
# Maybe `EXPERIMENT_NAME` should not contain slurm ID.
export EXPERIMENT_NAME="${WANDB_RUN_NAME}"
export TRAINING_HOME="/capstor/scratch/cscs/${USER}/RL/${MODEL_NAME}"
export TRAINING_CONFIG="${TRAINING_HOME}/config"
export CHECKPOINT_HOME="${TRAINING_HOME}/checkpoints/${EXPERIMENT_NAME}"
export LOCAL_MODEL_DIR="${TRAINING_HOME}/models/${MODEL_NAME}"

# Path where the container file installs the repo.
export NEMORL_DIR="/workdir/nemo_rl"

mkdir -p ${TRAINING_HOME}
mkdir -p ${TRAINING_CONFIG}
cd ${TRAINING_HOME}

# Ray cluster settings (must be defined before the env.toml heredoc).
export MASTER_NODE=$(hostname)
export MASTER_NODE_IP=$(hostname -i)
export PORT=6382
export RAY_ADDRESS="${MASTER_NODE_IP}:${PORT}"

# -----------------------------------------------------------------------------
# Container environment (matches the verl env.toml style).
# -----------------------------------------------------------------------------

cat > "${TRAINING_CONFIG}/env.toml" <<- EOF
image = "${NEMORL_IMAGE}"
mounts = ["/capstor", "/iopsstor", "/users", "/tmp"]
workdir = "${NEMORL_DIR}"
# Needed because of venvs that is written inside the container.
writable = true
entrypoint = true
[env]
PMIX_MCA_psec = "native"
WANDB_API_KEY = "${WANDB_API_KEY}"
HF_TOKEN_PATH = "${HF_TOKEN_PATH}"
RAY_ADDRESS = "${RAY_ADDRESS}"
MASTER_NODE_IP = "${MASTER_NODE_IP}"
PORT = "${PORT}"
# Keep Triton kernel cache on node-local storage; NFS-backed $HOME can give
# "Stale file handle" during JIT compilation.
TRITON_CACHE_DIR = "/tmp/triton_cache_${SLURM_JOB_ID}"
TRITON_HOME = "/tmp/triton_home_${SLURM_JOB_ID}"
[annotations]
com.hooks.cxi.enabled = "false"
EOF

# -----------------------------------------------------------------------------
# NeMo-RL GRPO config.  This mirrors train-gsm8k-qwen-3B.sh as closely as
# possible but uses the vLLM backend first, since SGLang has not been validated
# with NeMo-RL yet.  The SGLang block is kept below, commented out, so you can
# switch to it after validation.
#
# Note on rollout.n: in the verl script this is explicitly set to 16 (not a
# default). It maps to NeMo-RL's grpo.num_generations_per_prompt.
# -----------------------------------------------------------------------------
cat > "${TRAINING_CONFIG}/grpo_gsm8k_nemorl.yaml" <<- EOF
defaults: ${NEMORL_DIR}/examples/configs/grpo_math_1B.yaml

grpo:
  # 256 prompts per step x 16 generations = 4096 sequences per global batch.
  num_prompts_per_step: 256
  num_generations_per_prompt: 16
  # NOTE: capped at 1 epoch / 27 steps to match the async comparison run.
  # Keep this in sync with train-gsm8k-qwen-3B-full-async-nemorl.sh.
  max_num_epochs: 1
  max_num_steps: 27
  val_period: 50
  val_at_start: false
  val_at_end: true
  # Use the full GSM8K test split (1,319 test examples).
  max_val_samples: 1319
  val_batch_size: 256
  normalize_rewards: true
  use_leave_one_out_baseline: true

loss_fn:
  # Matches verl algorithm.kl_ctrl.kl_coef.
  reference_policy_kl_penalty: 0.001

checkpointing:
  checkpoint_dir: ${CHECKPOINT_HOME}
  save_period: 50
  keep_top_k: 10

policy:
  # Use the locally cached copy downloaded once below; avoids all workers
  # hitting the HuggingFace Hub simultaneously.
  model_name: "${LOCAL_MODEL_DIR}"
  train_global_batch_size: 4096
  train_micro_batch_size: 4
  logprob_batch_size: 4
  # 512 prompt + 1024 response (the verl values were commented; made explicit).
  max_total_sequence_length: 1536

  dtensor_cfg:
    automodel_kwargs:
      # Matches verl actor_rollout_ref.model.override_config.attn_implementation.
      attn_implementation: flash_attention_2

  optimizer:
    name: "torch.optim.AdamW"
    kwargs:
      lr: 1.0e-6
      weight_decay: 0.01
      betas: [0.9, 0.999]
      eps: 1e-8

  generation:
    # Active backend: vLLM.  Keep the settings close to the SGLang block below.
    backend: "vllm"
    use_async_rollouts: false
    max_new_tokens: 1024
    temperature: 1.0
    top_p: 1.0
    top_k: null
    vllm_cfg:
      async_engine: false
      precision: bfloat16
      kv_cache_dtype: "auto"
      tensor_parallel_size: 1
      pipeline_parallel_size: 1
      expert_parallel_size: 1
      # Matches verl rollout.gpu_memory_utilization.
      gpu_memory_utilization: 0.5
      max_model_len: 1536
      enforce_eager: false
      use_tqdm: true
      use_deep_gemm: false
      num_last_layers_in_bf16: 0
      num_first_layers_in_bf16: 0
      enable_vllm_metrics_logger: false
      vllm_metrics_logger_interval: 0.5
    vllm_kwargs: {}
    colocated:
      enabled: true

    # # SGLang backend (commented out).  Enable this block and comment the vLLM
    # # block above once SGLang has been validated with NeMo-RL.
    # backend: "sglang"
    # use_async_rollouts: false
    # max_new_tokens: 1024
    # temperature: 1.0
    # top_p: 1.0
    # top_k: null
    # sglang_cfg:
    #   model_path: \${policy.model_name}
    #   dtype: \${policy.precision}
    #   context_length: \${policy.max_total_sequence_length}
    #   allow_auto_truncate: true
    #   tp_size: 1
    #   dp_size: 1
    #   pp_size: 1
    #   ep_size: 1
    #   random_seed: 42
    #   max_running_requests: null
    #   # Matches verl rollout.gpu_memory_utilization.
    #   mem_fraction_static: 0.5
    #   skip_server_warmup: true
    #   disable_piecewise_cuda_graph: true
    #   disable_cuda_graph: true
    #   sglang_server_config:
    #     needs_offload: true
    #     cpu_weight_backup: true
    #     sglang_server_concurrency: 1024
    #     pause_generation_mode: retract
    #     num_gpus: \${cluster.gpus_per_node}
    #     num_gpus_per_engine: \${policy.generation.sglang_cfg.tp_size}
    #   sglang_router_config:
    #     use_external_router: false
    # colocated:
    #   enabled: true

# Replace the parent's OpenMathInstruct-2 data setup with GSM8K.
data:
  _override_: true
  max_input_seq_length: \${policy.max_total_sequence_length}
  shuffle: true
  num_workers: 1
  use_multiple_dataloader: false
  train:
    dataset_name: "gsm8k"
    split: train
  validation:
    dataset_name: "gsm8k"
    split: test
  default:
    prompt_file: null
    system_prompt_file: ${NEMORL_DIR}/examples/prompts/gsm8k.txt
    processor: "math_hf_data_processor"
    env_name: "math"

env:
  math:
    num_workers: 8
    math_verify_impl: "hf_math_verify"

logger:
  log_dir: ${TRAINING_HOME}/logs
  wandb_enabled: true
  tensorboard_enabled: false
  mlflow_enabled: false
  swanlab_enabled: false
  monitor_gpus: true
  wandb:
    project: ${WANDB_PROJECT_NAME}
    name: ${WANDB_RUN_NAME}

cluster:
  num_nodes: ${SLURM_JOB_NUM_NODES}
  gpus_per_node: 4
  master_port_range_low: 25000
  master_port_range_high: 28000
EOF

# -----------------------------------------------------------------------------
# Download the model once to shared storage (mirrors the verl script).
# Using a local copy avoids all policy workers hitting the HF Hub at the same
# time, which was the cause of the previous 28/32 rendezvous timeout.
#
# We use a dedicated, disposable uv venv with the standalone `hf` CLI
# (https://pypi.org/project/hf/) instead of the NeMo-RL container.
# -----------------------------------------------------------------------------
if [ ! -d "${LOCAL_MODEL_DIR}" ]; then
    echo "Downloading ${MODEL_REPO}/${MODEL_NAME} to ${LOCAL_MODEL_DIR}..."
    HF_DOWNLOAD_DIR="${HOME}/tmp/hf_download_${SLURM_JOB_ID}"
    mkdir -p "${HF_DOWNLOAD_DIR}"
    pushd "${HF_DOWNLOAD_DIR}" >/dev/null || exit 1

    if ! command -v uv >/dev/null 2>&1; then
        echo "[ERROR] uv is not available on this node; cannot create download venv." >&2
        exit 1
    fi

    uv venv
    # shellcheck source=/dev/null
    source .venv/bin/activate
    uv pip install hf

    if [[ -n "${HF_TOKEN_PATH}" && -f "${HF_TOKEN_PATH}" ]]; then
        export HF_TOKEN="$(cat "${HF_TOKEN_PATH}")"
    fi

    hf download "${MODEL_REPO}/${MODEL_NAME}" --local-dir "${LOCAL_MODEL_DIR}"

    # Cleanup
    deactivate
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
srun --mpi=pmix --network=disable_rdzv_get -N ${SLURM_JOB_NUM_NODES} --ntasks-per-node=1 -u \
    --environment="${TRAINING_CONFIG}/env.toml" \
    --container-writable bash -c '

unset PYTHONOPTIMIZE

# If a token file was passed, read it into the environment as a fallback for
# libraries that do not natively honour HF_TOKEN_PATH.
if [[ -n "${HF_TOKEN_PATH}" && -f "${HF_TOKEN_PATH}" && -z "${HF_TOKEN}" ]]; then
    export HF_TOKEN="$(cat "${HF_TOKEN_PATH}")"
    export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi

cd ${NEMORL_DIR}

# Initializing ray to ensure that it is in cash (which the container should have ensured) and
#  will load fast when we start the container.
uv run ray --version >/dev/null 2>&1 || true

# Signal that this rank has finished its uv setup.
RANK_DONE_FILE="${TRAINING_CONFIG}/rank_${SLURM_JOB_ID}_${SLURM_PROCID}_done"
touch "${RANK_DONE_FILE}"

# Chain barrier: rank k waits for rank k-1. This gives rank 0 priority to
# finish its setup and open the Ray cluster before later ranks proceed.
#if [ "${SLURM_PROCID}" -gt 0 ]; then
#    PREV_RANK=$((SLURM_PROCID - 1))
#    PREV_DONE_FILE="${TRAINING_CONFIG}/rank_${SLURM_JOB_ID}_${PREV_RANK}_done"
#    echo "Rank ${SLURM_PROCID}: waiting for rank ${PREV_RANK} uv setup..."
#    while [ ! -f "${PREV_DONE_FILE}" ]; do
#        sleep 1
#    done
#fi

if [ ${SLURM_PROCID} -eq 0 ]; then
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
        --node-ip-address=${MASTER_NODE_IP} \
        --port=${PORT} \
        --num-cpus=${SLURM_CPUS_PER_TASK} \
        --num-gpus=4 \
        --disable-usage-stats || true

    # Signal that the Ray head is ready.
    touch "${TRAINING_CONFIG}/ray_open_${SLURM_JOB_ID}"

    # Wait until all worker nodes have joined.
    while true; do
        alive_nodes=$(uv run ray status | awk "/Active:/{flag=1;next}/Pending:/{flag=0}flag" | grep "node_" | wc -l)
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
    uv run python examples/run_grpo.py --config ${TRAINING_CONFIG}/grpo_gsm8k_nemorl.yaml

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
        --node-ip-address=$(hostname -i) \
        --num-cpus=${SLURM_CPUS_PER_TASK} \
        --num-gpus=4 \
        --block || true
fi
'
