#!/bin/bash

#SBATCH --nodes=1
#SBATCH --account=csstaff
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --time=2:00:00

# ─────────────────────────────────────────────────────────────────────────────
# One-time offline conversion: Apertus-v1.5-70B HF checkpoint -> Megatron native
# distributed checkpoint, so future training runs skip the ~30-57 min redundant
# HF-safetensors load every run pays today (see CLAUDE.md "Model load phase"
# analysis and convert_apertus1p5_hf_to_mcore.py's own docstring for the full
# root-cause explanation and the exact verl/megatron-bridge API calls used).
#
# 1 node / 4 GPUs, TP=4 PP=1 EP=1 -- NOT the real training recipe's TP=8, to
# keep this to a single node with a plain `torchrun --standalone` launch
# (avoids a multi-node torchrun rendezvous, which nothing else in this repo
# does yet). Megatron's dist_checkpointing format is reshardable -- the saved
# checkpoint can still be loaded at TP=8/PP=1/EP=1 (or any other layout) by
# the real training recipe; the TP/PP/EP used here only has to be one Megatron
# can build the model at, not the layout training will actually run under.
#
# Usage:
#   sbatch convert-apertus1p5-hf-to-mcore.sh
#   HF_MODEL_PATH=/path/to/other/checkpoint OUTPUT_PATH=/path/to/output \
#     sbatch convert-apertus1p5-hf-to-mcore.sh
#   RUN_ROUNDTRIP_TEST=1 sbatch convert-apertus1p5-hf-to-mcore.sh   # verify before trusting
#
# NEVER RUN ON A CLUSTER. Every individual API call is verified against real
# verl v0.9.0 / megatron-bridge v0.6.0 source (fetched and read directly), and
# this reuses the training recipe's own already-validated Group 1a (swiss-ai
# transformers wheel) + Group 2 (Megatron/Bridge Apertus1p5Bridge compare-diffs
# + xielu wheel) setup verbatim -- but the conversion script itself
# (convert_apertus1p5_hf_to_mcore.py) has never been executed. Run with
# RUN_ROUNDTRIP_TEST=1 first and inspect the log before pointing a real
# training run at the output.
# ─────────────────────────────────────────────────────────────────────────────

set -o pipefail

export VERL_IMAGE="jfrog.svc.cscs.ch/docker-group-csstaff/alps-images/verl-cuda:alps7-dev-af30d3905eedb02f"
export MODEL_NAME="Apertus-v1.5-70B"
export TRAINING_HOME=/capstor/scratch/cscs/${USER}/RL-debug/${MODEL_NAME}
export TRAINING_CONFIG=/tmp
mkdir -p ${TRAINING_HOME}
cd ${TRAINING_HOME}

export HF_MODEL_PATH="${HF_MODEL_PATH:-/capstor/store/cscs/swissai/infra01/RL_Infra/models/ap1p5-70b-sft-262k-2700_corr}"
export OUTPUT_PATH="${OUTPUT_PATH:-${TRAINING_HOME}/mcore-dist-ckpt/ap1p5-70b-sft-262k-2700_corr-tp4pp1ep1}"
export CONVERT_TP="${CONVERT_TP:-4}"
export CONVERT_PP="${CONVERT_PP:-1}"
export CONVERT_EP="${CONVERT_EP:-1}"
export RUN_ROUNDTRIP_TEST="${RUN_ROUNDTRIP_TEST:-0}"

if [ ! -d "${HF_MODEL_PATH}" ]; then
    echo "FATAL: HF_MODEL_PATH=${HF_MODEL_PATH} does not exist"
    exit 1
fi
if [ -d "${OUTPUT_PATH}" ] && [ -n "$(ls -A "${OUTPUT_PATH}" 2>/dev/null)" ]; then
    echo "FATAL: OUTPUT_PATH=${OUTPUT_PATH} already exists and is not empty -- remove it or point"
    echo "       OUTPUT_PATH at a fresh directory before rerunning the conversion."
    exit 1
fi

echo "Converting ${HF_MODEL_PATH} -> ${OUTPUT_PATH} (TP=${CONVERT_TP} PP=${CONVERT_PP} EP=${CONVERT_EP})"

cat > "${TRAINING_CONFIG}/env.toml" <<- EOF
image = "${VERL_IMAGE}"
mounts = ["/capstor", "/iopsstor", "/users", "/tmp"]
workdir = "/workspace/verl"
writable = true
entrypoint = true
[env]
PMIX_MCA_psec = "native"
HF_TOKEN = "$(cat ~/HF_TOKEN)"
[annotations]
com.hooks.cxi.enabled = "false"
EOF

# ══════════════════════════════════════════════════════════════════════════
# Group 1a: swiss-ai/transformers wheel (apertus1p5 model_type registration --
# needed for AutoConfig.from_pretrained / AutoBridge.from_hf_pretrained to
# recognize the checkpoint at all). Identical to the training recipe's own
# block -- verbatim, so the built wheel cache under ${TRAINING_HOME}/wheels is
# shared/reusable with any training run against the same TRAINING_HOME.
# Group 1b/1c/1d (SGLang apertus1p5 support) are NOT needed here -- this
# script never touches SGLang or the rollout side.
# ══════════════════════════════════════════════════════════════════════════
export SWISS_AI_TRANSFORMERS_SHA=986d6dfa97cc6675a65f6d052e41ab316b0649eb
export SWISS_AI_WHEEL_DIR=${TRAINING_HOME}/wheels
mkdir -p ${SWISS_AI_WHEEL_DIR}
if ! ls ${SWISS_AI_WHEEL_DIR}/transformers-*.whl >/dev/null 2>&1 \
    || ! ls ${SWISS_AI_WHEEL_DIR}/safetensors-*.whl >/dev/null 2>&1 \
    || [ "$(cat ${SWISS_AI_WHEEL_DIR}/transformers.sha 2>/dev/null)" != "${SWISS_AI_TRANSFORMERS_SHA}" ]; then
    echo "Building swiss-ai/transformers@${SWISS_AI_TRANSFORMERS_SHA} + safetensors wheels..."
    rm -f ${SWISS_AI_WHEEL_DIR}/transformers-*.whl
    srun --mpi=pmix --network=disable_rdzv_get -N 1 --ntasks=1 -u \
        --environment="${TRAINING_CONFIG}/env.toml" \
        --container-writable bash -c '
        set -e
        curl -sfL "https://github.com/swiss-ai/transformers/archive/${SWISS_AI_TRANSFORMERS_SHA}.tar.gz" \
            -o /tmp/swiss-ai-transformers.tar.gz
        [ -s /tmp/swiss-ai-transformers.tar.gz ]
        mkdir -p /tmp/swiss-ai-transformers
        tar xzf /tmp/swiss-ai-transformers.tar.gz -C /tmp/swiss-ai-transformers --strip-components=1
        pip wheel --no-deps -w /tmp/swiss-ai-wheel /tmp/swiss-ai-transformers
        pip download --no-deps -d /tmp/swiss-ai-wheel "safetensors>=0.8.0"
        cp /tmp/swiss-ai-wheel/transformers-*.whl /tmp/swiss-ai-wheel/safetensors-*.whl ${SWISS_AI_WHEEL_DIR}/
    '
    ls ${SWISS_AI_WHEEL_DIR}/transformers-*.whl >/dev/null 2>&1 \
        && ls ${SWISS_AI_WHEEL_DIR}/safetensors-*.whl >/dev/null 2>&1 \
        || { echo "FATAL: swiss-ai/transformers or safetensors wheel build failed"; exit 1; }
    echo "${SWISS_AI_TRANSFORMERS_SHA}" > ${SWISS_AI_WHEEL_DIR}/transformers.sha
else
    echo "swiss-ai/transformers and safetensors wheels already present for ${SWISS_AI_TRANSFORMERS_SHA}, skipping build."
fi

# ══════════════════════════════════════════════════════════════════════════
# Group 2: Megatron support for Apertus 1.5 -- identical to the training
# recipe's own block, verbatim (same compare-diff URLs/SHAs, same xielu wheel).
# ══════════════════════════════════════════════════════════════════════════
export MEGATRON_LM_FORK_REF="9898eb2be164641600f197368f765a904ad73812"      # theely/Megatron-LM apertus1p5-support head
export MEGATRON_BRIDGE_FORK_REF="53bea08c8417627d5e83af6c9c16163a47e3a8b1"  # theely/Megatron-Bridge apertus1p5-support head
curl -sfL "https://github.com/theely/Megatron-LM/compare/NVIDIA:core_v0.19.0...theely:${MEGATRON_LM_FORK_REF}.diff" \
    -o ${TRAINING_CONFIG}/megatron-lm-apertus1p5.diff \
    || { echo "FATAL: could not download theely/Megatron-LM ${MEGATRON_LM_FORK_REF} compare diff"; exit 1; }
[ -s ${TRAINING_CONFIG}/megatron-lm-apertus1p5.diff ] \
    || { echo "FATAL: theely/Megatron-LM compare diff is empty"; exit 1; }
curl -sfL "https://github.com/theely/Megatron-Bridge/compare/NVIDIA-NeMo:v0.6.0...theely:${MEGATRON_BRIDGE_FORK_REF}.diff" \
    -o ${TRAINING_CONFIG}/megatron-bridge-apertus1p5.diff \
    || { echo "FATAL: could not download theely/Megatron-Bridge ${MEGATRON_BRIDGE_FORK_REF} compare diff"; exit 1; }
[ -s ${TRAINING_CONFIG}/megatron-bridge-apertus1p5.diff ] \
    || { echo "FATAL: theely/Megatron-Bridge compare diff is empty"; exit 1; }
sbcast -f ${TRAINING_CONFIG}/megatron-lm-apertus1p5.diff     ${TRAINING_CONFIG}/megatron-lm-apertus1p5.diff
sbcast -f ${TRAINING_CONFIG}/megatron-bridge-apertus1p5.diff ${TRAINING_CONFIG}/megatron-bridge-apertus1p5.diff
echo "Fetched theely/Megatron-{LM,Bridge} apertus1p5 compare diffs (${MEGATRON_LM_FORK_REF} / ${MEGATRON_BRIDGE_FORK_REF})."

export XIELU_SHA=2a55f6b9efa64954bf173e63297a2b2f99741b69
export XIELU_WHEEL_DIR=${TRAINING_HOME}/wheels
mkdir -p ${XIELU_WHEEL_DIR}
if ! ls ${XIELU_WHEEL_DIR}/xielu-*.whl >/dev/null 2>&1 \
    || [ "$(cat ${XIELU_WHEEL_DIR}/xielu.sha 2>/dev/null)" != "${XIELU_SHA}" ]; then
    echo "Building rubber-duck-debug/xielu@${XIELU_SHA} wheel..."
    srun --mpi=pmix --network=disable_rdzv_get -N 1 --ntasks=1 -u \
        --environment="${TRAINING_CONFIG}/env.toml" \
        --container-writable bash -c '
        set -e
        export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
        rm -rf /tmp/xielu-src
        curl -sfL "https://github.com/rubber-duck-debug/xielu/archive/${XIELU_SHA}.tar.gz" \
            -o /tmp/xielu-src.tar.gz
        [ -s /tmp/xielu-src.tar.gz ]
        mkdir -p /tmp/xielu-src
        tar xzf /tmp/xielu-src.tar.gz -C /tmp/xielu-src --strip-components=1
        rm -f ${XIELU_WHEEL_DIR}/xielu-*.whl
        pip wheel --no-build-isolation --no-deps -w /tmp/xielu-wheel /tmp/xielu-src
        cp /tmp/xielu-wheel/xielu-*.whl ${XIELU_WHEEL_DIR}/
    '
    ls ${XIELU_WHEEL_DIR}/xielu-*.whl >/dev/null 2>&1 \
        || { echo "FATAL: rubber-duck-debug/xielu wheel build failed"; exit 1; }
    echo "${XIELU_SHA}" > ${XIELU_WHEEL_DIR}/xielu.sha
else
    echo "rubber-duck-debug/xielu wheel already present and up to date (${XIELU_SHA}), skipping build."
fi

# Fetch the conversion script from the branch (same convention as reward.py /
# dataset_prepare.py in the training recipe -- no script assumes a local git
# checkout exists on the cluster filesystem; BASH_SOURCE would resolve to the
# sbatch spool copy anyway, not a checkout path -- see CLAUDE.md run 3129805).
# Must be committed + pushed to the branch before this job starts, or the
# curl below 404s (loud FATAL, not a silent stale copy).
export CONVERT_PY_URL="https://raw.githubusercontent.com/eth-cscs/alps-extended-images/refs/heads/Add-megatron-rl-recipes/Alps-Images/apps/verl/apertus-benchmarks/convert_apertus1p5_hf_to_mcore.py"
curl -sfL "${CONVERT_PY_URL}" -o "${TRAINING_CONFIG}/convert_apertus1p5_hf_to_mcore.py" \
    || { echo "FATAL: could not download convert_apertus1p5_hf_to_mcore.py from ${CONVERT_PY_URL}"; exit 1; }
grep -q "def main" "${TRAINING_CONFIG}/convert_apertus1p5_hf_to_mcore.py" \
    || { echo "FATAL: downloaded convert_apertus1p5_hf_to_mcore.py has no main()"; exit 1; }
sbcast -f "${TRAINING_CONFIG}/convert_apertus1p5_hf_to_mcore.py" "${TRAINING_CONFIG}/convert_apertus1p5_hf_to_mcore.py"

# ══════════════════════════════════════════════════════════════════════════
# Run the conversion: apply Group 1a + Group 2, then torchrun the script.
# Single node, plain --standalone launch (no multi-node rendezvous needed at
# TP=4/PP=1/EP=1 = 4 processes = this node's 4 GH200 GPUs).
# ══════════════════════════════════════════════════════════════════════════
srun --mpi=pmix --network=disable_rdzv_get -N 1 --ntasks=1 -u \
    --environment="${TRAINING_CONFIG}/env.toml" \
    --container-writable bash -c '
    set -e

    pip install --no-deps ${SWISS_AI_WHEEL_DIR}/transformers-*.whl ${SWISS_AI_WHEEL_DIR}/safetensors-*.whl
    python3 -c "import transformers, safetensors; print(\"transformers:\", transformers.__version__); print(\"safetensors:\", safetensors.__version__)"

    pip install --no-deps ${XIELU_WHEEL_DIR}/xielu-*.whl \
        || { echo "FATAL: rubber-duck-debug/xielu wheel install failed"; exit 1; }
    python3 -c "import xielu.ops; print(\"xielu:\", xielu.__file__)"

    SITE_PKG=$(python3 -c "import os, megatron.core as m; print(os.path.dirname(os.path.dirname(os.path.dirname(m.__file__))))")
    patch --batch -p1 -d "${SITE_PKG}" < ${TRAINING_CONFIG}/megatron-lm-apertus1p5.diff \
        || { echo "FATAL: theely/Megatron-LM apertus1p5 compare diff failed to apply"; exit 1; }
    patch --batch -p2 -d "${SITE_PKG}" --fuzz=3 < ${TRAINING_CONFIG}/megatron-bridge-apertus1p5.diff \
        || { echo "FATAL: theely/Megatron-Bridge apertus1p5 compare diff failed to apply"; exit 1; }
    python3 -c "import megatron.core, megatron.bridge; from megatron.bridge.models import Apertus1p5Bridge; print(\"megatron.core:\", megatron.core.__file__, \"| Apertus1p5Bridge OK\")" \
        || { echo "FATAL: Apertus1p5Bridge not importable after applying the Megatron compare diffs"; exit 1; }

    echo "Starting conversion: HF_MODEL_PATH=${HF_MODEL_PATH} OUTPUT_PATH=${OUTPUT_PATH} TP=${CONVERT_TP} PP=${CONVERT_PP} EP=${CONVERT_EP} RUN_ROUNDTRIP_TEST=${RUN_ROUNDTRIP_TEST}"
    torchrun --standalone --nproc_per_node=4 ${TRAINING_CONFIG}/convert_apertus1p5_hf_to_mcore.py
'

echo "Conversion job finished. Output (if successful): ${OUTPUT_PATH}"
echo "To use it in a training recipe, set in actor.megatron (and ref.megatron, if a reference"
echo "policy is ever built): use_dist_checkpointing: True, dist_checkpointing_path: ${OUTPUT_PATH}"
