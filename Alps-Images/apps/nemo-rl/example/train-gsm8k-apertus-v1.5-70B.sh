#!/bin/bash

#SBATCH --nodes=16
#SBATCH --account=csstaff
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=4:00:00

# ─────────────────────────────────────────────────────────────────────────────
# Apertus-v1.5-70B GRPO on GSM8K with NeMo-RL: Megatron trainer + standalone vLLM
# rollout, async single-controller trainer, per-step NCCL weight refit. NeMo-RL
# counterpart of the verl recipe
# apps/verl/apertus-benchmarks/rl-bench-apertus-v1.5-70B-sglang-megatron-v1-separate-async.sh
# (branch Add-megatron-rl-recipes). Run log and hazards: ../CLAUDE.md.
#
#   * image   : nemo-rl-cuda (this app). It ships the
#               whole runtime stack (ray 2.56, vLLM 0.28.0+apertus2, TE 2.17,
#               transformers 5.17, flashinfer) but NO NeMo-RL / Megatron-Bridge /
#               Megatron-LM source: those are external checkouts put on PYTHONPATH.
#   * source  : Alvorecer721/Nemo-RL (the fork carrying the Apertus recipes) at a
#               pinned commit, cloned once to Lustre together with its
#               Megatron-Bridge submodule and that fork's nested Megatron-LM.
#   * recipe  : the fork's own, cluster-validated
#               examples/configs/recipes/llm/grpo-apertus1p5-70b-16n4g-tp2pp4-gsm8k-2k-windowed-optimized.yaml
#               (reference run: job 3352055, docs/apertus-reference-runs.md) is
#               inherited through `defaults:`; this script only overrides the node
#               split, step count and logging. 16 nodes = 12 trainer (48 GPUs,
#               TP2/PP4 -> DP6) + 4 rollout (4 vLLM TP=4 replicas), 48 prompts x 16
#               generations per step, 92 steps, thinking off, 2048-token context.
#
# No Apertus-1.5 source patches are needed, unlike the verl recipes: the SFT
# checkpoint declares architectures=[ApertusForCausalLM] / model_type=apertus
# (checked in its config.json), which stock transformers 5.17, the image vLLM
# (registry: ApertusForCausalLM) and the fork ApertusBridge all support natively.
# NeMo-RL patches the installed vLLM itself at worker start (xIELU beta/eps as
# engine-owned constants); both anchors were checked against the image vLLM
# source (swiss-ai/vllm@d4d41485) and match.
#
# Differences from the fork's own launcher (infra/slurm/cscs/autoresearch/
# run_apertus_bench.sh + ray.sub), all forced by the image:
#   * workers run the container interpreter (NEMO_RL_PY_EXECUTABLES_SYSTEM=1,
#     the image default) instead of the fork image's baked /opt/ray_venvs, so
#     the fork's uv.lock pins (vLLM 0.29, ray 2.58, TE 2.18, transformers 5.12)
#     are NOT what runs -- the preflight below prints what does.
#   * Ray is started verl-style (one srun, head on rank 0) without ray.sub's
#     nvlink_domain_* / topo_rank resources; NeMo-RL falls back to generic
#     (non topology-aware) placement.
#   * no CUDA xIELU kernel (not in the image): the Megatron side uses the
#     bridge's jit-fused Python xIELU -- same math, slower activation.
#
# NOT YET VALIDATED: runs 3621638 / 3621707 / 3621888 failed in the preflight
# (see ../CLAUDE.md).
# ─────────────────────────────────────────────────────────────────────────────

export NEMORL_IMAGE="jfrog.svc.cscs.ch/docker-group-csstaff/alps-images/nemo-rl-cuda:alps7-dev-7eab7fc435699b52c9668a663afde4b3"

export MODEL_NAME="Apertus-v1.5-70B"

# HF-format snapshot dir (config.json, tokenizer files, chat_template.jinja,
# safetensors). Same SFT checkpoint as the verl recipe. No trailing slash.
export MODEL_CHECKPOINT_PATH="${MODEL_CHECKPOINT_PATH:-/capstor/store/cscs/swissai/infra01/RL_Infra/models/ap1p5-70b-sft-262k-2700_corr}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_CHECKPOINT_PATH}}"

# NeMo-RL source. Pin a commit: the recipe schema and the vLLM patches move with it.
export NEMORL_REPO="${NEMORL_REPO:-https://github.com/Alvorecer721/Nemo-RL.git}"
export NEMORL_REF="${NEMORL_REF:-ce1c6818815a1ac87dae4e983d2a5d188ee0fae2}"
# nemo-lens (nemo.lens, pure Python telemetry): nemo_rl imports it unconditionally.
# The image ships 0.1.0 (older tags: none, run 3621638); the fork needs >= 0.2.0,
# so the rev it pins in its pyproject is checked out and put first on PYTHONPATH.
export LENS_REPO="https://github.com/NVIDIA-NeMo/Lens.git"
export LENS_REF="b0f977d414b2f89938604a0b7eaa78ee08bc8700"
export NEMORL_RECIPE="examples/configs/recipes/llm/grpo-apertus1p5-70b-16n4g-tp2pp4-gsm8k-2k-windowed-optimized.yaml"

export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-92}"
export SEED="${SEED:-42}"
# Only the boxed prompt (examples/prompts/apertus_bench_gsm8k_system_boxed.txt)
# exists in the fork; the verl recipe trains GSM8K on [[[N]]] instead.
export ANSWER_MARKER="boxed"

export PROJECT_NAME="apertus-benchmarks"
export WANDB_ENTITY="${WANDB_ENTITY:-ml-team-cscs}"   # the recipe default
export EXPERIMENT_NAME="${MODEL_NAME}-gsm8k-nemorl-vllm-megatron-async-${SLURM_JOB_NUM_NODES}n"
export RUN_NAME="${EXPERIMENT_NAME}-run-${SLURM_JOB_ID}"
export TRAINING_HOME=/capstor/scratch/cscs/${USER}/RL/${MODEL_NAME}
export TRAINING_CONFIG=/tmp
export RUN_DIR=${TRAINING_HOME}/nemo-rl-runs/${RUN_NAME}
export NEMORL_DIR=${TRAINING_HOME}/nemo-rl-src/${NEMORL_REF:0:12}
export LENS_DIR=${TRAINING_HOME}/nemo-rl-src/lens-${LENS_REF:0:12}
# Optional: Python packages nemo_rl imports that the image lacks, installed once
# with --no-deps into a shared --target dir (bump the suffix when the list
# changes). Empty for the current image; the tag before it (alps7-dev-cfcfede4...)
# needed "soundfile==0.14.0" (run 3621888). The preflight prints what is missing.
export EXTRA_PIP_PACKAGES="${EXTRA_PIP_PACKAGES:-}"
export EXTRA_SITE=${TRAINING_HOME}/nemo-rl-src/extra-site-v1
if [ -z "${EXTRA_PIP_PACKAGES}" ] || [ -f "${EXTRA_SITE}/.alps-ready" ]; then
    export EXTRA_SITE_READY=1
else
    export EXTRA_SITE_READY=0
fi
export BRIDGE_SUBDIR="3rdparty/Megatron-Bridge-workspace/Megatron-Bridge"
# HF -> Megatron converted checkpoint, written once and reused across runs.
# Must be shared storage: every trainer rank loads it.
export NRL_MEGATRON_CHECKPOINT_DIR=${TRAINING_HOME}/nemo-rl-mcore-ckpts

# Node split: the trainer world must be a multiple of TP x PP = 2 x 4 = 8 GPUs.
export ROLLOUT_NNODES="${ROLLOUT_NNODES:-4}"
export TRAINING_NNODES=$(( SLURM_JOB_NUM_NODES - ROLLOUT_NNODES ))
if [ "${TRAINING_NNODES}" -le 0 ] || [ $(( TRAINING_NNODES * 4 % 8 )) -ne 0 ]; then
    echo "FATAL: ${TRAINING_NNODES} trainer nodes x 4 GPUs is not a positive multiple of TP2 x PP4 = 8"
    exit 1
fi

mkdir -p "${TRAINING_HOME}" "${RUN_DIR}/tb" "${NRL_MEGATRON_CHECKPOINT_DIR}"
cd "${TRAINING_HOME}"

if [ -f "/users/${USER}/.wandb_api_key" ]; then
    export WANDB_API_KEY=$(cat /users/${USER}/.wandb_api_key)
    export WANDB_ENABLED=true
else
    echo "WARNING: /users/${USER}/.wandb_api_key not found, W&B logging disabled"
    export WANDB_ENABLED=false
fi
export WANDB_SILENT=true

cat > "${TRAINING_CONFIG}/env.toml" <<- EOF
image = "${NEMORL_IMAGE}"
mounts = ["/capstor", "/iopsstor", "/users","/tmp"]
workdir = "${TRAINING_HOME}"
writable = true
entrypoint = true
[env]
PMIX_MCA_psec = "native"
[annotations]
com.hooks.cxi.enabled = "false"
EOF

# Child of the fork recipe: plain values only. Read by the driver on rank 0,
# which is this batch host (/tmp is mounted into the container). This heredoc is unquoted, so no
# OmegaConf interpolations and no backticks in here -- bash would expand them.
cat > "${TRAINING_CONFIG}/grpo_gsm8k_nemorl.yaml" <<- EOF
defaults: ${NEMORL_DIR}/${NEMORL_RECIPE}
grpo:
  max_num_steps: ${TOTAL_TRAINING_STEPS}
policy:
  generation:
    colocated:
      enabled: false
      resources:
        num_nodes: ${ROLLOUT_NNODES}
        gpus_per_node: 4
    vllm_cfg:
      env_vars:
        PYTHONPATH: ${NEMORL_DIR}:${LENS_DIR}/src:${EXTRA_SITE}
logger:
  wandb_enabled: ${WANDB_ENABLED}
  wandb:
    project: ${PROJECT_NAME}
    entity: ${WANDB_ENTITY}
    name: ${RUN_NAME}
cluster:
  gpus_per_node: 4
  num_nodes: ${SLURM_JOB_NUM_NODES}
EOF

[ -r "${MODEL_CHECKPOINT_PATH}/model.safetensors.index.json" ] \
    || { echo "FATAL: no checkpoint at ${MODEL_CHECKPOINT_PATH}"; exit 1; }
[ -r "${TOKENIZER_PATH}/chat_template.jinja" ] \
    || { echo "FATAL: no chat_template.jinja in ${TOKENIZER_PATH}"; exit 1; }

# ── NeMo-RL + Megatron-Bridge + Megatron-LM checkout (once, shared) ──────────
# Cloned on a single node into a temp dir and renamed into place, so every node
# of every job sees one complete tree at the pinned commit. Only the two
# submodules the Megatron/vLLM path imports are initialised (Gym, Automodel and
# the xIELU kernels are not).
if [ ! -f "${NEMORL_DIR}/.alps-checkout-ready" ] || [ ! -f "${LENS_DIR}/.alps-checkout-ready" ] \
        || [ "${EXTRA_SITE_READY}" -ne 1 ]; then
    srun --mpi=pmix --network=disable_rdzv_get -N 1 --ntasks=1 -u \
        --environment="${TRAINING_CONFIG}/env.toml" \
        --container-writable bash -c '
    set -eo pipefail
    if [ "${EXTRA_SITE_READY}" -ne 1 ]; then
        etmp="${EXTRA_SITE}.tmp.${SLURM_JOB_ID}"
        rm -rf "${etmp}"
        python3 -m pip install --no-cache-dir --no-deps --break-system-packages \
            --target "${etmp}" ${EXTRA_PIP_PACKAGES}
        echo "${EXTRA_PIP_PACKAGES}" > "${etmp}/.alps-ready"
        mv -T "${etmp}" "${EXTRA_SITE}" 2>/dev/null || rm -rf "${etmp}"
        echo "Extra packages (${EXTRA_PIP_PACKAGES}) ready at ${EXTRA_SITE}"
    fi
    if [ ! -f "${LENS_DIR}/.alps-checkout-ready" ]; then
        ltmp="${LENS_DIR}.tmp.${SLURM_JOB_ID}"
        rm -rf "${ltmp}"
        mkdir -p "${ltmp}"
        git -C "${ltmp}" init -q
        git -C "${ltmp}" remote add origin "${LENS_REPO}"
        git -C "${ltmp}" fetch -q --depth 1 origin "${LENS_REF}"
        git -C "${ltmp}" checkout -q FETCH_HEAD
        [ -d "${ltmp}/src/nemo/lens" ] || { echo "FATAL: no src/nemo/lens in ${LENS_REPO}@${LENS_REF}"; exit 1; }
        git -C "${ltmp}" rev-parse HEAD > "${ltmp}/.alps-checkout-ready"
        mv -T "${ltmp}" "${LENS_DIR}" 2>/dev/null || rm -rf "${ltmp}"
        echo "nemo-lens checkout ready at ${LENS_DIR}"
    fi
    if [ -f "${NEMORL_DIR}/.alps-checkout-ready" ]; then
        exit 0
    fi
    tmp="${NEMORL_DIR}.tmp.${SLURM_JOB_ID}"
    rm -rf "${tmp}"
    mkdir -p "${tmp}"
    git -C "${tmp}" init -q
    git -C "${tmp}" remote add origin "${NEMORL_REPO}"
    git -C "${tmp}" fetch -q --depth 1 origin "${NEMORL_REF}"
    git -C "${tmp}" checkout -q FETCH_HEAD
    git -C "${tmp}" submodule update --init --depth 1 "${BRIDGE_SUBDIR}" \
        || git -C "${tmp}" submodule update --init "${BRIDGE_SUBDIR}"
    git -C "${tmp}/${BRIDGE_SUBDIR}" submodule update --init --depth 1 3rdparty/Megatron-LM \
        || git -C "${tmp}/${BRIDGE_SUBDIR}" submodule update --init 3rdparty/Megatron-LM
    [ -f "${tmp}/${NEMORL_RECIPE}" ] || { echo "FATAL: recipe ${NEMORL_RECIPE} not in ${NEMORL_REF}"; exit 1; }
    [ -d "${tmp}/${BRIDGE_SUBDIR}/src/megatron/bridge" ] || { echo "FATAL: Megatron-Bridge submodule is empty"; exit 1; }
    [ -d "${tmp}/${BRIDGE_SUBDIR}/3rdparty/Megatron-LM/megatron/core" ] || { echo "FATAL: Megatron-LM submodule is empty"; exit 1; }

    # fcntl.flock is unsupported in-container on Lustre (ENOLCK / stale file
    # handle): drop the lock around the
    # read-only config load in megatron-bridge. Best effort.
    python3 - "${tmp}/${BRIDGE_SUBDIR}/src/megatron/bridge/models/hf_pretrained/safe_config_loader.py" <<"PYEOF"
import sys
path = sys.argv[1]
lines = open(path).read().split("\n")
n = 0
for i, line in enumerate(lines):
    if line.lstrip().startswith("with filelock."):
        lines[i] = line[: len(line) - len(line.lstrip())] + "with contextlib.nullcontext():"
        n += 1
if n:
    lines.insert(0, "import contextlib")
    open(path, "w").write("\n".join(lines))
    print("Patched", n, "filelock site(s) in", path)
else:
    print("WARNING: no filelock site found in", path)
PYEOF

    git -C "${tmp}" rev-parse HEAD > "${tmp}/.alps-checkout-ready"
    if mv -T "${tmp}" "${NEMORL_DIR}" 2>/dev/null; then
        echo "NeMo-RL checkout ready at ${NEMORL_DIR}"
    else
        echo "Another job created ${NEMORL_DIR} first, discarding this copy"
        rm -rf "${tmp}"
    fi
    '
    [ -f "${NEMORL_DIR}/.alps-checkout-ready" ] && [ -f "${LENS_DIR}/.alps-checkout-ready" ] \
        && { [ -z "${EXTRA_PIP_PACKAGES}" ] || [ -f "${EXTRA_SITE}/.alps-ready" ]; } \
        || { echo "FATAL: NeMo-RL / nemo-lens / extra-package setup under ${TRAINING_HOME}/nemo-rl-src failed"; exit 1; }
else
    echo "NeMo-RL checkout $(cat "${NEMORL_DIR}/.alps-checkout-ready") already present at ${NEMORL_DIR}"
fi

export MASTER_NODE=$(hostname)
export MASTER_NODE_IP=$(hostname -i)
export PORT=6382
export RAY_ADDRESS="${MASTER_NODE_IP}:${PORT}"

srun --mpi=pmix --network=disable_rdzv_get -N ${SLURM_JOB_NUM_NODES} --ntasks-per-node=1 -u \
    --kill-on-bad-exit=1 \
    --environment="${TRAINING_CONFIG}/env.toml" \
    --container-writable bash -c '

RANK=${SLURM_PROCID}
JOB_ID=${SLURM_JOB_ID}
NUM_NODES=${SLURM_JOB_NUM_NODES}
NUM_CPUS=${SLURM_CPUS_PER_TASK}
NODE_IP=$(hostname -i)

cd "${NEMORL_DIR}"

# Checkout first: sibling projects carry their own top-level tools/tests/examples.
export PYTHONPATH="${NEMORL_DIR}:${LENS_DIR}/src:${NEMORL_DIR}/${BRIDGE_SUBDIR}/src:${NEMORL_DIR}/${BRIDGE_SUBDIR}/3rdparty/Megatron-LM:${EXTRA_SITE}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

# The image preloads the TorchMemorySaver CUDA hook for SGLang-style
# pause/resume. Neither vLLM nor Megatron uses it here, and the recipe runs the
# Megatron workers with expandable_segments, which TorchMemorySaver rejects.
export LD_PRELOAD=

# Recipe inputs (resolved through oc.env in the driver).
export AP_EXPERIMENT_DIR="${NEMORL_DIR}"
export AP_CKPT="${MODEL_CHECKPOINT_PATH}"
export AP_TOKENIZER="${TOKENIZER_PATH}"
export AP_RUN_DIR="${RUN_DIR}"
export AP_SEED="${SEED}"
export AP_ANSWER_MARKER="${ANSWER_MARKER}"

# Same runtime knobs as the fork launcher (run_apertus_bench.sh).
export NCCL_DEBUG=WARN
export NRL_REFIT_NUM_STREAMS=2
export RAY_DEDUP_LOGS=0
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export VLLM_DISABLE_PYNCCL=1

# Compile / JIT / dataset caches off Lustre (no flock there) onto node-local /tmp.
export HF_HOME=/tmp/hf_${JOB_ID}
export TRITON_CACHE_DIR=/tmp/triton_${JOB_ID}
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_${JOB_ID}
export VLLM_CACHE_ROOT=/tmp/vllm_cache_${JOB_ID}
export FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer_${JOB_ID}

if [ "${RANK}" -eq 0 ]; then
    # Source fix-ups for running the checkout on this image, applied to the shared
    # checkout before anything imports it (idempotent; workers import it only
    # after the head is up).
    #   nemo_rl/models/policy/__init__.py asserts transformers < 5.13 as a
    #   reminder to re-check a DeepSeek-V3/Moonlight tokenizer workaround; the
    #   image has 5.17.0 and the workaround is a no-op for Apertus (run 3621707).
    python3 - "${NEMORL_DIR}/nemo_rl/models/policy/__init__.py" <<"PYEOF" || { echo "FATAL: could not relax the transformers version assert"; exit 1; }
import sys
path = sys.argv[1]
src = open(path).read()
old = "    assert PkgVersion(transformers.__version__) < PkgVersion(\"5.13.0\"), ("
new = "    if not PkgVersion(transformers.__version__) < PkgVersion(\"5.13.0\"): print("
if new in src:
    print("transformers version assert already relaxed")
elif src.count(old) == 1:
    open(path, "w").write(src.replace(old, new))
    print("Relaxed the transformers<5.13 assert in", path)
else:
    raise SystemExit("anchor not found in " + path)
PYEOF

    # Preflight: what the image actually provides, that the external checkouts
    # import against it, and that the recipe validates -- before Ray starts.
    python3 - <<"PYEOF" || { echo "FATAL: NeMo-RL preflight failed on $(hostname)"; exit 1; }
import importlib
import importlib.metadata as md
import os
from pathlib import Path

for pkg in ("torch", "vllm", "transformers", "ray", "transformer_engine", "flashinfer-python", "openai"):
    try:
        print(f"{pkg}: {md.version(pkg)}")
    except Exception as exc:
        print(f"{pkg}: NOT INSTALLED ({exc})")

import importlib.util
import traceback

# Every third-party top-level module in the module-level import closure of the
# entrypoint and the worker modules (static scan of the pinned checkout).
closure = (
    "PIL absl aiohttp cffi datasets fastapi httpx huggingface_hub hydra math_verify matplotlib "
    "megatron mlflow nemo numpy nvidia_resiliency_ext omegaconf packaging pandas "
    "prometheus_client psutil pyarrow pydantic pynvml ray regex requests rich scipy "
    "soundfile swanlab tensordict torch torchdata tqdm transformers typing_extensions "
    "uvicorn wandb yaml zmq"
).split()
missing = [name for name in closure if importlib.util.find_spec(name) is None]
print("third-party modules not importable:", missing if missing else "none")

failed = []
for mod in (
    "nemo.lens",
    "megatron.core",
    "megatron.bridge",
    "nemo_rl",
    "megatron.bridge.models.apertus.apertus_bridge",
    "nemo_rl.models.policy.workers.megatron_policy_worker",
    "nemo_rl.models.generation.vllm.vllm_worker",
    "nemo_rl.models.generation.vllm.vllm_worker_async",
    "nemo_rl.experience.sync_rollout_actor",
    "nemo_rl.algorithms.async_utils",
    "nemo_rl.environments.bracket_math_environment",
    "examples.run_grpo_single_controller",
):
    try:
        m = importlib.import_module(mod)
        print("import", mod, "OK", getattr(m, "__file__", None))
    except Exception:
        failed.append(mod)
        print("import", mod, "FAILED")
        traceback.print_exc()
if failed:
    raise SystemExit(f"preflight: {len(failed)} import(s) failed: {failed}")

from omegaconf import OmegaConf
from nemo_rl.algorithms.single_controller_utils.config import MasterConfig, validate_single_controller_config
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

register_omegaconf_resolvers()
resolved = OmegaConf.to_container(load_config(Path("/tmp/grpo_gsm8k_nemorl.yaml")), resolve=True)
config = MasterConfig.model_validate(resolved)
validate_single_controller_config(config)
policy = config.policy
mcfg = policy["megatron_cfg"]
print(
    "config OK:",
    "model", policy["model_name"],
    "tp", mcfg["tensor_model_parallel_size"],
    "pp", mcfg["pipeline_model_parallel_size"],
    "nodes", config.cluster["num_nodes"],
    "gen_nodes", policy["generation"]["colocated"]["resources"]["num_nodes"],
    "steps", config.grpo.max_num_steps,
    "prompts", config.grpo.num_prompts_per_step, "x", config.grpo.num_generations_per_prompt,
    "gbs", policy["train_global_batch_size"],
    "seq", policy["max_total_sequence_length"],
)
PYEOF
fi

# Ray owns worker placement: drop the Slurm/PMI launcher variables before the
# raylet (and through it every actor) inherits them.
while IFS="=" read -r name _; do
    case "${name}" in PMI*|PMIX*|MPI*|OMPI*|SLURM_*) unset "${name}" ;; esac
done < <(env)

if [ "${RANK}" -eq 0 ]; then
    ray start --head \
        --node-ip-address=${MASTER_NODE_IP} \
        --port=${PORT} \
        --num-cpus=${NUM_CPUS} \
        --num-gpus=4 \
        --object-store-memory=68719476736 \
        --include-dashboard=false \
        --disable-usage-stats || true
    touch "${RUN_DIR}/ray_head_ready"

    waited=0
    while true; do
        alive_nodes=$(ray status | awk "/Active:/{flag=1;next}/Pending:/{flag=0}flag" | grep "node_" | wc -l)
        if ! [[ "$alive_nodes" =~ ^[0-9]+$ ]]; then
            alive_nodes=0
        fi
        if [ "$alive_nodes" -ge "${NUM_NODES}" ]; then
            break
        fi
        if [ "${waited}" -ge 1800 ]; then
            echo "FATAL: only ${alive_nodes}/${NUM_NODES} nodes joined the Ray cluster after 30 min"
            exit 1
        fi
        echo "Waiting for all nodes to join [$alive_nodes/${NUM_NODES}]"
        sleep 5
        waited=$((waited + 5))
    done

    python3 -m examples.run_grpo_single_controller \
        --config ${TRAINING_CONFIG}/grpo_gsm8k_nemorl.yaml 2>&1 | tee "${RUN_DIR}/run.log"
    rc=${PIPESTATUS[0]}

    ray stop --grace-period 30 || true
    exit "${rc}"
else
    # Worker nodes join once the head is up (it runs the preflight first; run
    # 3621638: a fixed sleep had the workers give up before the head existed).
    while [ ! -f "${RUN_DIR}/ray_head_ready" ]; do
        sleep 2
    done
    sleep 5
    ray start \
        --address="${RAY_ADDRESS}" \
        --node-ip-address=${NODE_IP} \
        --num-cpus=${NUM_CPUS} \
        --num-gpus=4 \
        --object-store-memory=68719476736 \
        --disable-usage-stats \
        --block || true
fi


'
