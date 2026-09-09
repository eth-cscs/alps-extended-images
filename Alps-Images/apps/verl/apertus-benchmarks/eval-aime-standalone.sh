#!/bin/bash

#SBATCH --nodes=4
#SBATCH --account=csstaff
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=1:30:00

# ─────────────────────────────────────────────────────────────────────────────
# Standalone, generation-only AIME-2024 eval of an already-saved Apertus-v1.5-70B
# checkpoint (verl.trainer.main_generation_server -- no Megatron, no V1 trainer,
# no TransferQueue, no weight sync: SGLang loads the checkpoint's real HF export
# directly with load_format=auto and just serves it). Written 2026-09-09 to
# diagnose whether rl-bench-apertus-v1.5-70B-sglang-megatron-v1-separate-async.sh
# runs 3330187/3331427's AIME avg@32 regression after DAPO-Math RL training
# (9-10% -> 3% pass@1) is a real reasoning loss or a train/eval PROMPT-FORMAT
# mismatch: training only ever rewards the [[[N]]] bracket marker (dapo_math
# rows), while the AIME eval prompt asks for \boxed{}; GRPO could plausibly have
# taught the policy to always emit brackets regardless of what the eval prompt
# asks for, in which case reward.py's boxed-only AIME scorer would zero out
# correct-but-bracket-formatted answers -- a scoring artifact, not a capability
# loss. Neither training run saved validation generations (`val_generations: 0`,
# `rollout_data_dir`/`validation_data_dir`: null), so this dumps raw per-sample
# text to disk instead of re-running training.
#
# Usage: CKPT_RUN_ID=<slurm job id whose checkpoint to eval> sbatch eval-aime-standalone.sh
#   (or set CKPT_PATH directly to point at any other HF-format checkpoint dir)
#
# Reuses the parent recipe's Group 1 (Apertus 1.5 support) patches verbatim --
# same image, same swiss-ai transformers wheel, same SGLang PR #32979 + local
# fixes -- since SGLang still needs those to recognize the apertus1p5 model
# class and to tolerate the LM-only export (checkpoint.strict: False means no
# vision/audio tower weights were saved). Everything Megatron/Bridge/xielu/
# V1-trainer-specific in the parent recipe is DROPPED: this script never builds
# a Megatron model, never syncs weights, never touches TransferQueue -- SGLang
# just loads the checkpoint's model/huggingface/ export straight off Lustre.
# xielu is Megatron-Bridge-specific (MCoreXIELU) -- transformers' own
# XIELUActivation (what SGLang's apertus1p5 model reuses) has a working
# pure-pytorch fallback (confirmed by the FSDP2 recipe running without it).

set -o pipefail

export VERL_IMAGE="jfrog.svc.cscs.ch/docker-group-csstaff/alps-images/verl-cuda:alps7-dev-af30d3905eedb02f"
export MODEL_NAME="Apertus-v1.5-70B"
export TRAINING_HOME=/capstor/scratch/cscs/${USER}/RL/${MODEL_NAME}
export TRAINING_CONFIG=/tmp

: "${CKPT_RUN_ID:?set CKPT_RUN_ID=<slurm job id whose checkpoint to eval> (or set CKPT_PATH directly)}"
export CKPT_EXPERIMENT_DIR="${TRAINING_HOME}/checkpoints/Apertus-v1.5-70B-dapo-math-verl-sglang-megatron-v1-separate-async-40n-run-${CKPT_RUN_ID}"
export CKPT_STEP="${CKPT_STEP:-global_step_92}"
export MODEL_PATH="${CKPT_PATH:-${CKPT_EXPERIMENT_DIR}/${CKPT_STEP}/actor/model/huggingface}"
if [ ! -d "${MODEL_PATH}" ]; then
    echo "FATAL: MODEL_PATH=${MODEL_PATH} does not exist"
    exit 1
fi

# The AIME-2024 test split already exists on Lustre from the training runs
# that produced this checkpoint (BENCHMARK=dapo-math, boxed AIME prompt,
# DATASET_PROMPT_VERSION v3-*-boxed-aime) -- reuse it rather than rebuilding.
export DATA_PATH="${DATA_PATH:-${TRAINING_HOME}/data/dapo-math/test.parquet}"
if [ ! -f "${DATA_PATH}" ]; then
    echo "FATAL: DATA_PATH=${DATA_PATH} does not exist -- run the training recipe once first, or set DATA_PATH"
    exit 1
fi

mkdir -p "${TRAINING_HOME}/eval-gen"
export OUTPUT_PATH="${TRAINING_HOME}/eval-gen/${CKPT_RUN_ID}-${CKPT_STEP}-aime-gen.parquet"

# Same AIME-2024 val sampling as the training recipe (user spec, 2026-09-08):
# n=32, T=0.6, top_p=0.95, top_k disabled.
export N_SAMPLES="${N_SAMPLES:-32}"
export TEMPERATURE="${TEMPERATURE:-0.6}"
export TOP_P="${TOP_P:-0.95}"
export PROMPT_LENGTH="${PROMPT_LENGTH:-2048}"
export RESPONSE_LENGTH="${RESPONSE_LENGTH:-12288}"
export ROLLOUT_TP="${ROLLOUT_TP:-4}"

echo "Evaluating checkpoint run-${CKPT_RUN_ID}/${CKPT_STEP}"
echo "  MODEL_PATH=${MODEL_PATH}"
echo "  DATA_PATH=${DATA_PATH}"
echo "  OUTPUT_PATH=${OUTPUT_PATH}"
echo "  n=${N_SAMPLES} T=${TEMPERATURE} top_p=${TOP_P} tp=${ROLLOUT_TP} nodes=${SLURM_JOB_NUM_NODES}"

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
# Group 1: add Apertus 1.5 support (verbatim from the parent training recipe).
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

export SGLANG_APERTUS_PATCH_URL="https://github.com/sgl-project/sglang/pull/32979.diff"
curl -sfL "${SGLANG_APERTUS_PATCH_URL}" -o ${TRAINING_CONFIG}/sglang-apertus1p5-full.diff \
    || { echo "FATAL: could not download sglang PR #32979 diff"; exit 1; }
[ -s ${TRAINING_CONFIG}/sglang-apertus1p5-full.diff ] \
    || { echo "FATAL: sglang PR #32979 diff is empty"; exit 1; }
# Allowlist (not blocklist): see the parent recipe's own comment (run 3152782) --
# only the 4 files confirmed load-bearing are fatal-if-they-fail; everything
# else the PR touches under python/sglang/ is best-effort, self-adapting to
# the PR's continued drift (it is open/unmerged and re-fetched fresh every run).
awk '
    /^diff --git a\// { keep = 0 }
    /^diff --git a\/python\/sglang\/srt\/configs\/qwen3_asr\.py/ { keep = 1 }
    /^diff --git a\/python\/sglang\/srt\/models\/apertus\.py/ { keep = 1 }
    /^diff --git a\/python\/sglang\/srt\/models\/apertus_mm\.py/ { keep = 1 }
    /^diff --git a\/python\/sglang\/srt\/multimodal\/processors\/apertus_mm\.py/ { keep = 1 }
    keep { print }
' ${TRAINING_CONFIG}/sglang-apertus1p5-full.diff > ${TRAINING_CONFIG}/sglang-apertus1p5.diff
[ -s ${TRAINING_CONFIG}/sglang-apertus1p5.diff ] \
    || { echo "FATAL: filtered sglang PR #32979 diff is empty"; exit 1; }
sbcast -f ${TRAINING_CONFIG}/sglang-apertus1p5.diff ${TRAINING_CONFIG}/sglang-apertus1p5.diff

awk '
    /^diff --git a\// { keep = 1 }
    /^diff --git a\/python\/sglang\/srt\/configs\/qwen3_asr\.py/ { keep = 0 }
    /^diff --git a\/python\/sglang\/srt\/models\/apertus\.py/ { keep = 0 }
    /^diff --git a\/python\/sglang\/srt\/models\/apertus_mm\.py/ { keep = 0 }
    /^diff --git a\/python\/sglang\/srt\/multimodal\/processors\/apertus_mm\.py/ { keep = 0 }
    /^diff --git a\/test\// { keep = 0 }
    /^diff --git a\/docs\// { keep = 0 }
    /^diff --git a\/docs_new\// { keep = 0 }
    keep { print }
' ${TRAINING_CONFIG}/sglang-apertus1p5-full.diff > ${TRAINING_CONFIG}/sglang-apertus1p5-best-effort.diff
sbcast -f ${TRAINING_CONFIG}/sglang-apertus1p5-best-effort.diff ${TRAINING_CONFIG}/sglang-apertus1p5-best-effort.diff

cat > "${TRAINING_CONFIG}/sglang-apertus1p5-local-fixes.diff" <<- 'EOF'
# Local fixes on top of sgl-project/sglang#32979 (still open/unmerged).
# Apply *after* that PR's patch — this diff is against the new file it adds
# (python/sglang/srt/models/apertus_mm.py), not upstream SGLang.
#
# 1. _init_component_model calls component_config.to_dict() unconditionally,
#    assuming a PreTrainedConfig-like object; against the transformers
#    commit this script installs it's already a plain dict. Fixed to accept
#    either shape.
# 2. load_component_weight raises when a checkpoint key claims to be a
#    vision/audio tensor but the tower model has no matching parameter —
#    the checkpoint's vision-tokenizer weight names don't line up with the
#    structure this PR builds. Harmless to skip: this benchmark is
#    text-only GSM8K and never calls get_image_feature/get_audio_feature,
#    so vision/audio tower weights never need to be numerically correct,
#    only present so the model instantiates. The language-model weight path
#    (everything load_component_weight returns False for) is untouched.
#
# Re-diff against the PR's current head if it moves and this stops applying.
--- a/python/sglang/srt/models/apertus_mm.py
+++ b/python/sglang/srt/models/apertus_mm.py
@@ -50,7 +50,11 @@
     component_config: Any,
     model_cls: type[nn.Module] | None = None,
 ) -> nn.Module:
-    config_dict = component_config.to_dict()
+    config_dict = (
+        component_config.to_dict()
+        if hasattr(component_config, "to_dict")
+        else dict(component_config)
+    )
     config = AutoConfig.for_model(config_dict.pop("model_type"), **config_dict)
     return AutoModel.from_config(config) if model_cls is None else model_cls(config)

@@ -301,9 +305,16 @@

             component_tensor = component_tensors.get(name)
             if component_tensor is None:
-                raise ValueError(
-                    f"No vision/audio tensor matches checkpoint key: {name}"
+                # Vision/audio tower weights never need to be numerically
+                # correct for a text-only benchmark that never calls
+                # get_image_feature/get_audio_feature -- skip instead of
+                # aborting the whole load.
+                print(
+                    "[apertus1p5-local-fixes] WARNING: skipping unmatched "
+                    f"vision/audio checkpoint key {name}",
+                    flush=True,
                 )
+                return True
             weight_loader = getattr(
                 component_tensor, "weight_loader", default_weight_loader
             )
EOF
sbcast -f ${TRAINING_CONFIG}/sglang-apertus1p5-local-fixes.diff ${TRAINING_CONFIG}/sglang-apertus1p5-local-fixes.diff

export MASTER_NODE_IP=$(hostname -i)
export PORT=6382
export RAY_ADDRESS="${MASTER_NODE_IP}:${PORT}"
export WANDB_MODE=disabled
export RAY_memory_usage_threshold=0.99

srun --mpi=pmix --network=disable_rdzv_get -N ${SLURM_JOB_NUM_NODES} --ntasks-per-node=1 -u \
    --environment="${TRAINING_CONFIG}/env.toml" \
    --container-writable bash -c '

git -C /workspace/verl --no-pager log --oneline -1 || true

pip install --no-deps ${SWISS_AI_WHEEL_DIR}/transformers-*.whl ${SWISS_AI_WHEEL_DIR}/safetensors-*.whl
python3 -c "import transformers, safetensors; print(\"transformers:\", transformers.__version__)"

patch --batch -p2 -d /usr/local/lib/python3.12/dist-packages < ${TRAINING_CONFIG}/sglang-apertus1p5.diff \
    || { echo "FATAL: sglang PR #32979 patch failed to apply"; exit 1; }
patch --batch -p2 -d /usr/local/lib/python3.12/dist-packages < ${TRAINING_CONFIG}/sglang-apertus1p5-best-effort.diff \
    || echo "WARNING: one or more best-effort sglang PR #32979 hunks did not apply -- continuing"
patch --batch -p2 -d /usr/local/lib/python3.12/dist-packages < ${TRAINING_CONFIG}/sglang-apertus1p5-local-fixes.diff \
    || { echo "FATAL: local apertus_mm.py fixes failed to apply"; exit 1; }

export FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer_${SLURM_JOB_ID}
mkdir -p $FLASHINFER_WORKSPACE_BASE
export TRITON_CACHE_DIR=/tmp/triton_${SLURM_JOB_ID}
mkdir -p $TRITON_CACHE_DIR
export SGLANG_DISABLE_CUDA_GRAPH=1
export VERL_LOGGING_LEVEL=INFO

if [ $SLURM_PROCID -eq 0 ]; then
    ray start --head \
        --node-ip-address=$MASTER_NODE_IP \
        --port=$PORT \
        --num-cpus=${SLURM_CPUS_PER_TASK} \
        --num-gpus=4 \
        --disable-usage-stats || true

    while true; do
            alive_nodes=$(ray status | awk "/Active:/{flag=1;next}/Pending:/{flag=0}flag" | grep "node_" | wc -l)
            if ! [[ "$alive_nodes" =~ ^[0-9]+$ ]]; then
                alive_nodes=0
            fi
            if [ "$alive_nodes" -ge "$SLURM_JOB_NUM_NODES" ]; then
                break
            fi
            echo "Waiting for all nodes to join [$alive_nodes/$SLURM_JOB_NUM_NODES]"
            sleep 5
    done

    HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_generation_server \
        trainer.nnodes=${SLURM_JOB_NUM_NODES} \
        trainer.n_gpus_per_node=4 \
        data.train_files=${DATA_PATH} \
        data.prompt_key=prompt \
        +data.output_path=${OUTPUT_PATH} \
        actor_rollout_ref.model.path=${MODEL_PATH} \
        actor_rollout_ref.model.trust_remote_code=False \
        actor_rollout_ref.rollout.name=sglang \
        actor_rollout_ref.rollout.load_format=auto \
        actor_rollout_ref.rollout.skip_tokenizer_init=False \
        actor_rollout_ref.rollout.temperature=${TEMPERATURE} \
        actor_rollout_ref.rollout.top_p=${TOP_P} \
        actor_rollout_ref.rollout.prompt_length=${PROMPT_LENGTH} \
        actor_rollout_ref.rollout.response_length=${RESPONSE_LENGTH} \
        actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
        actor_rollout_ref.rollout.n=${N_SAMPLES}

    RC=$?

    if [ $RC -eq 0 ]; then
        python3 <<- "PYEOF"
	import os, re
	import pandas as pd
	output_path = os.environ["OUTPUT_PATH"]
	df = pd.read_parquet(output_path)
	n_rows_ = len(df)
	n_resp_ = len(df.iloc[0]["responses"])
	print(f"rows={n_rows_} responses/row={n_resp_}")
	bracket_re = re.compile(r"\[\[\[(.*?)\]\]\]", re.DOTALL)
	boxed_re = re.compile(r"\\boxed\{")
	n_bracket = n_boxed = n_neither = n_both = 0
	sample_dump = []
	for _, row in df.iterrows():
	    for resp in row["responses"]:
	        has_b = bool(bracket_re.search(resp))
	        has_x = bool(boxed_re.search(resp))
	        if has_b and has_x:
	            n_both += 1
	        elif has_b:
	            n_bracket += 1
	        elif has_x:
	            n_boxed += 1
	        else:
	            n_neither += 1
	        if len(sample_dump) < 6:
	            sample_dump.append(resp[-400:])
	total = n_bracket + n_boxed + n_neither + n_both
	print(f"marker usage across {total} generations: bracket-only={n_bracket} boxed-only={n_boxed} both={n_both} neither={n_neither}")
	print("--- last 400 chars of first 6 responses ---")
	for s in sample_dump:
	    print(repr(s))
	    print("---")
	PYEOF
    fi

    exit $RC
else
    sleep 15
    ray start \
        --address="${RAY_ADDRESS}" \
        --node-ip-address=$(hostname -i) \
        --num-cpus=${SLURM_CPUS_PER_TASK} \
        --num-gpus=4 \
        --block || true
fi

'
