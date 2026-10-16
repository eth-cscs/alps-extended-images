#!/bin/bash

#SBATCH --nodes=32
#SBATCH --account=csstaff
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=8:00:00

# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK copy of train-gsm8k-chonk-megatron-gen.sh, for comparison with a
# Verl run. Differences from that script (everything else is identical):
#   * 92 GRPO steps, 48 prompts x 16 samples per step, seed 42.
#   * learning rate 1e-6 (LR), weight decay 0.01, constant schedule without
#     warm-up (Verl default; NeMo-RL steps the scheduler once per GRPO step).
#   * responses up to 2048 tokens (sequence limit 3072).
#   * training sampling temperature 1.0, top-p 1.0; validation temperature 0.6,
#     top-p 0.95 on the GSM8K test split (1319 prompts), at steps 0, 46 and 92.
#   * patches/nemo-rl-benchmark-metrics.patch: validation sampling on the
#     Megatron generation path, exact token entropy, clip fraction, k1 KL, and
#     Verl-named copies of the step and validation metrics in wandb.
#   * own source checkout (SRC_VERSION), normal partition, 8 h limit.
#
# Apertus 2 "Chonk" GRPO on GSM8K with NeMo-RL, Megatron for training AND
# generation (no vLLM), on as-vanilla-as-possible libraries.
#
# Companion to train-gsm8k-chonk.sh (the image author's launcher: vLLM rollout,
# swiss-ai forks of NeMo-RL and Megatron-Bridge plus open PRs merged at launch).
# The recipe below is that launcher's grpo-chonk-vllm.yaml with the generation
# block swapped; the launch mechanics are those of
# train-gsm8k-apertus-v1.5-70B.sh. Self-contained: sbatch it or submit its text.
#
# Sources (cloned once to Lustre, all pinned):
#   NeMo-RL          NVIDIA-NeMo/RL@f196ef0a            vanilla + 2-file patch
#   Megatron-Bridge  NVIDIA-NeMo/Megatron-Bridge@07722453  vanilla + 4-file patch
#                    + 1-file md_decoupling optimizer dispatch patch
#                    + models/apertus2/ copied verbatim from
#                      swiss-ai/Megatron-Bridge@f3bc8e7e (6 new files)
#   Megatron-LM      swiss-ai/Megatron-LM-MoE PR #88 @b0844260  the fork (KDA layers,
#                    latent MoE, quantile-balancing router are not in stock
#                    Megatron-LM) + 3-file patch: upstream's native refit API,
#                    its NCCL refit deadlock fix and round-robin task ids (only
#                    used by the refit_transport: mcore fallback)
#   nemo-lens        NVIDIA-NeMo/Lens@b85578fc          the rev NeMo-RL pins
# The five patches (four + the benchmark one) are embedded below and checked in under patches/ (keep them
# byte-identical); each hunk and why it is needed is listed in the patch header.
#
# Deliberately NOT taken from the forks (see ../CLAUDE.md for the reasoning):
#   * bridge: native / legacy Megatron-checkpoint loading (auto_bridge,
#     model_load_save, mlm_compat, checkpointing) -- this starts from the HF
#     checkpoint; offloading-expert weight views (swiss-ai bridge PR #4) and the
#     CPU->CUDA step in the Megatron->HF export -- only needed with expert
#     offloading, which is off here.
#   * NeMo-RL: upstream PR #2517 (vLLM data-parallel), swiss-ai PR #5 (load
#     checkpoints under no_grad), the rest of swiss-ai PR #1.
# Differences from the author's recipe that follow from that: no expert
# offloading (model_overrides block dropped -> more trainer GPU memory), and
# generation by Megatron-Core's dynamic engine (eager), refit by NeMo-RL's
# shard-to-shard NCCL reshard (refit_transport: nccl_reshard, see the
# generation block).
# The optimizer is the author's md_decoupling block (Chonk's pretraining
# optimizer), dispatched by patches/bridge-md-decoupling-optimizer.patch.
#
# KL penalty against the starting policy: 0 (author: 0.01, REFERENCE_KL_PENALTY).
# Kept off by user decision (2026-10-15): it may not be needed.
#
# UNTESTED. No run yet. First things that can go wrong, in order: imports of the
# patched vanilla bridge against the Megatron-LM fork (preflight), trainer OOM
# without expert offloading, KDA under the inference engine, the refit EP32 -> EP8.
# ─────────────────────────────────────────────────────────────────────────────

export NEMORL_IMAGE="jfrog.svc.cscs.ch/docker-group-csstaff/alps-images/nemo-rl-cuda:alps7-dev-7eab7fc435699b52c9668a663afde4b3"

export MODEL_NAME="Apertus2-Chonk"
# HF-format snapshot (config.json with architectures=[Apertus2KDAForCausalLM],
# configuration_apertus2.py, tokenizer, safetensors). No trailing slash.
export HF_MODEL="${HF_MODEL:-/capstor/store/cscs/swissai/infra01/apertus_checkpoints/v2/hf/chonk/pretraining/long_context/long-context-iter-0003576}"

export NEMO_REPO="https://github.com/NVIDIA-NeMo/RL.git"
export NEMO_REF="f196ef0a1ccb54d6ce71cd63277fb6a0e01168db"
export BRIDGE_REPO="https://github.com/NVIDIA-NeMo/Megatron-Bridge.git"
export BRIDGE_REF="077224531ad4d6d0caef767a84e619b256aa5cb6"
export BRIDGE_APERTUS2_REPO="https://github.com/swiss-ai/Megatron-Bridge.git"
export BRIDGE_APERTUS2_REF="f3bc8e7e6a722adde1176b2ba8980109e09493ef"
export MLM_REPO="https://github.com/swiss-ai/Megatron-LM-MoE.git"
# b0844260 = a9e0c497 (the commit the vLLM example pins) + one commit on the same
# PR (#88), "Complete nonempty EP dummy forwards": idle expert-parallel ranks get
# a non-empty dummy batch. Without it generation crashes in the KDA layer during
# prefill (runs 3623722, 3624750).
export MLM_REF="b0844260b2a0888c711dca230b88ec13b20f3229"
export LENS_REPO="https://github.com/NVIDIA-NeMo/Lens.git"
export LENS_REF="b85578fc2b736a1804705e537001b5f45e9c715d"
# Bump when a ref or an embedded patch changes: it names the shared checkout.
export SRC_VERSION="v7bench"

# Batch sizes of the author's recipe; topology differs: 16 training nodes
# (64 GPUs, EP32 -> 8 experts per GPU, expert data-parallel 2) instead of 8
# nodes at EP16, + 16 generation nodes (64 GPUs, EP8 -> 8 replicas). Without
# the author's expert offloading, md_decoupling's layer-wise optimizer can
# only shard expert state over the expert data-parallel group; at 8 nodes
# (EP32, expert DP 1) every rank kept full fp32 masters, momentum and gradient
# copies of its experts and ran out of memory in the first step (run 3670987).
export TRAIN_NODES=16
export INFERENCE_NODES=$(( SLURM_JOB_NUM_NODES - TRAIN_NODES ))
export TOTAL_NODES=${SLURM_JOB_NUM_NODES}
# Micro-batch 1 with the global batch kept at 128 (2 accumulated micro-batches
# per trainer rank at 64 ranks): at micro-batch 4 the trainer runs out of GPU memory in the
# loss (vocabulary-wide logits for 4 x 1280 tokens on top of ~49 GB of model and
# optimizer state without expert offloading, run 3628358).
export TRAIN_MICRO_BATCH="${TRAIN_MICRO_BATCH:-1}"
export INFERENCE_REQUESTS="${INFERENCE_REQUESTS:-32}"
export MAX_NUM_STEPS="${MAX_NUM_STEPS:-92}"
export NUM_PROMPTS="${NUM_PROMPTS:-48}"
export NUM_GENERATIONS="${NUM_GENERATIONS:-16}"
# Benchmark knobs (see the header). md_decoupling has three learning rates: LR
# is the base one (Adam-updated parameters); MATRIX_LR (Muon-updated matrices)
# and GAINS_LR (hypersphere gains) default to the author's values.
export LR="${LR:-1.0e-6}"
export MATRIX_LR="${MATRIX_LR:-5.0e-5}"
export GAINS_LR="${GAINS_LR:-2.0e-6}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
export LR_DECAY_STYLE="${LR_DECAY_STYLE:-constant}"
export LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-0}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
# Prompt (data.max_input_seq_length 900) + response; also the generation
# engine length (max_model_len follows it).
export MAX_TOTAL_SEQUENCE_LENGTH="${MAX_TOTAL_SEQUENCE_LENGTH:-3072}"
export VAL_PERIOD="${VAL_PERIOD:-$(( MAX_NUM_STEPS / 2 ))}"
# KL penalty against the starting policy. 0.0 (author: 0.01), user decision: > 0
# makes NeMo-RL build a reference model on the trainer GPUs at setup and keep
# its ~32 GB GPU copy (see the loss_fn block of the recipe); with the native
# refit that no longer fits (run 3659594). 0.0 skips the reference model.
export REFERENCE_KL_PENALTY="${REFERENCE_KL_PENALTY:-0.0}"
export TRAIN_GLOBAL_BATCH="${TRAIN_GLOBAL_BATCH:-128}"
if [ "${INFERENCE_NODES}" -le 0 ] || [ $(( INFERENCE_NODES * 4 % 8 )) -ne 0 ]; then
    echo "FATAL: ${INFERENCE_NODES} generation nodes x 4 GPUs is not a positive multiple of EP8"
    exit 1
fi
if [ $(( NUM_PROMPTS * NUM_GENERATIONS % TRAIN_GLOBAL_BATCH )) -ne 0 ] \
        || [ $(( TRAIN_GLOBAL_BATCH % (TRAIN_NODES * 4 * TRAIN_MICRO_BATCH) )) -ne 0 ]; then
    echo "FATAL: rollout batch ${NUM_PROMPTS} x ${NUM_GENERATIONS} / global batch ${TRAIN_GLOBAL_BATCH} / micro-batch ${TRAIN_MICRO_BATCH} x $(( TRAIN_NODES * 4 )) ranks do not divide"
    exit 1
fi

export PROJECT_NAME="${PROJECT_NAME:-apertus-benchmarks}"   # same W&B project as the Apertus 1.5 recipes
export WANDB_ENTITY="${WANDB_ENTITY:-ml-team-cscs}"
export EXPERIMENT_NAME="${MODEL_NAME}-gsm8k-nemorl-benchmark-${SLURM_JOB_NUM_NODES}n"
export RUN_NAME="${EXPERIMENT_NAME}-run-${SLURM_JOB_ID}"
export TRAINING_HOME=/capstor/scratch/cscs/${USER}/RL/${MODEL_NAME}
export TRAINING_CONFIG=/tmp
export RUN_DIR=${TRAINING_HOME}/nemo-rl-runs/${RUN_NAME}
export SRC_DIR=${TRAINING_HOME}/src-${SRC_VERSION}-${NEMO_REF:0:8}-${BRIDGE_REF:0:8}-${MLM_REF:0:8}
export NEMO_RL_PATH=${SRC_DIR}/nemo-rl
# HF -> Megatron converted checkpoint: written once, shared by every rank.
export NRL_MEGATRON_CHECKPOINT_DIR=${TRAINING_HOME}/nemo-rl-mcore-ckpts

mkdir -p "${TRAINING_HOME}" "${RUN_DIR}/patches" "${NRL_MEGATRON_CHECKPOINT_DIR}"
cd "${TRAINING_HOME}"

[ -r "${HF_MODEL}/config.json" ] || { echo "FATAL: no HF checkpoint at ${HF_MODEL}"; exit 1; }

# W&B logging, as in the Apertus 1.5 recipes: on when the key file exists.
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
NCCL_PXN_DISABLE = "1"
NCCL_CROSS_NIC = "0"
[annotations]
com.hooks.cxi.enabled = "false"
EOF

# ── Embedded patches (copies of patches/*.patch) ─────────────────────────────
cat > "${RUN_DIR}/patches/bridge-apertus2-shared-minimal.patch" << 'CHONK_PATCH_EOF'
# Minimal edits to shared Megatron-Bridge files for Apertus 2 (Chonk) on the swiss-ai Megatron-LM-MoE fork.
# Base: NVIDIA-NeMo/Megatron-Bridge@077224531ad4d6d0caef767a84e619b256aa5cb6. Source of every hunk: swiss-ai/Megatron-Bridge@f3bc8e7e6a722adde1176b2ba8980109e09493ef
# (Apache-2.0). The models/apertus2/ package itself is NOT in this patch: the launcher checks it out
# verbatim from the fork. Hunks:
#   models/__init__.py        register the Apertus2 bridge; make DeepSeek / Nemotron-Omni imports optional
#   models/deepseek/__init__.py, models/nemotron_omni/__init__.py
#                             import guards: the Megatron-LM fork lacks mla_qk_norm_config and the
#                             dynamic-resolution context-parallel helpers these packages import
#   models/model_provider.py  keep the quantile-balancing router buffer (qb_beta) in fp32
diff --git a/src/megatron/bridge/models/__init__.py b/src/megatron/bridge/models/__init__.py
index 5ccb3939..65a3dd53 100644
--- a/src/megatron/bridge/models/__init__.py
+++ b/src/megatron/bridge/models/__init__.py
@@ -13,6 +13,12 @@
 # limitations under the License.
 
 # Import model providers for easy access
+from megatron.bridge.models.apertus2 import (
+    Apertus2Bridge,
+    Apertus2ModelBuilder,
+    Apertus2ModelConfig,
+    Apertus2ModelProvider,
+)
 from megatron.bridge.models.bailing import (
     BailingMoeV2Bridge,
 )
@@ -28,10 +34,11 @@ from megatron.bridge.models.conversion.param_mapping import (
     ReplicatedMapping,
     RowParallelMapping,
 )
-from megatron.bridge.models.deepseek import (
-    DeepSeekV2Bridge,
-    DeepSeekV3Bridge,
-)
+from megatron.bridge.models.deepseek import DEEPSEEK_AVAILABLE
+
+
+if DEEPSEEK_AVAILABLE:
+    from megatron.bridge.models.deepseek import DeepSeekV2Bridge, DeepSeekV3Bridge
 from megatron.bridge.models.ernie import (
     Ernie45Bridge,
 )
@@ -140,10 +147,11 @@ from megatron.bridge.models.muse_glimmer import (
 from megatron.bridge.models.nemotron import (
     NemotronBridge,
 )
-from megatron.bridge.models.nemotron_omni import (
-    NemotronOmniBridge,
-    NemotronOmniModel,
-)
+from megatron.bridge.models.nemotron_omni import NEMOTRON_OMNI_AVAILABLE
+
+
+if NEMOTRON_OMNI_AVAILABLE:
+    from megatron.bridge.models.nemotron_omni import NemotronOmniBridge, NemotronOmniModel
 from megatron.bridge.models.nemotron_vl import (
     NemotronVLBridge,
     NemotronVLModel,
@@ -218,9 +226,12 @@ __all__ = [
     "RowParallelMapping",
     "AutoMapping",
     "BailingMoeV2Bridge",
-    # DeepSeek Models
-    "DeepSeekV2Bridge",
-    "DeepSeekV3Bridge",
+    # Apertus Models
+    "Apertus2Bridge",
+    "Apertus2ModelBuilder",
+    "Apertus2ModelConfig",
+    "Apertus2ModelProvider",
+    # DeepSeek Models are appended when the MCore MLA resolver is available.
     # ERNIE Text-Only Models
     "Ernie45Bridge",
     # ERNIE VL Models
@@ -307,8 +318,7 @@ __all__ = [
     "NemotronVLModel",
     "NemotronVLBridge",
     "NemotronVLModelProvider",
-    "NemotronOmniBridge",
-    "NemotronOmniModel",
+    # Nemotron Omni models are appended when MCore dynamic-resolution CP is available.
     # ASR Models
     "Qwen3ASRBridge",
     "Qwen3ASRModel",
@@ -334,3 +344,8 @@ __all__ = [
     "ExaoneMoeBridge",
     "ExaoneMoeModelProvider",
 ]
+
+if DEEPSEEK_AVAILABLE:
+    __all__.extend(("DeepSeekV2Bridge", "DeepSeekV3Bridge"))
+if NEMOTRON_OMNI_AVAILABLE:
+    __all__.extend(("NemotronOmniBridge", "NemotronOmniModel"))
diff --git a/src/megatron/bridge/models/deepseek/__init__.py b/src/megatron/bridge/models/deepseek/__init__.py
index 968868c2..3c8151c2 100644
--- a/src/megatron/bridge/models/deepseek/__init__.py
+++ b/src/megatron/bridge/models/deepseek/__init__.py
@@ -12,13 +12,18 @@
 # See the License for the specific language governing permissions and
 # limitations under the License.
 
-from megatron.bridge.models.deepseek.deepseek_v2_bridge import DeepSeekV2Bridge  # noqa: F401
-from megatron.bridge.models.deepseek.deepseek_v3_bridge import DeepSeekV3Bridge  # noqa: F401
-from megatron.bridge.models.deepseek.deepseek_v4_bridge import DeepSeekV4Bridge  # noqa: F401
+from importlib.util import find_spec
 
 
-__all__ = [
-    "DeepSeekV2Bridge",
-    "DeepSeekV3Bridge",
-    "DeepSeekV4Bridge",
-]
+# TODO: Remove this guard when the Apertus2 MCore fork provides the MLA QK-norm resolver.
+# Other model families import deepseek.common without requiring DeepSeek model classes.
+DEEPSEEK_AVAILABLE = find_spec("megatron.core.transformer.mla_qk_norm_config") is not None
+
+if DEEPSEEK_AVAILABLE:
+    from megatron.bridge.models.deepseek.deepseek_v2_bridge import DeepSeekV2Bridge
+    from megatron.bridge.models.deepseek.deepseek_v3_bridge import DeepSeekV3Bridge
+    from megatron.bridge.models.deepseek.deepseek_v4_bridge import DeepSeekV4Bridge
+
+    __all__ = ["DeepSeekV2Bridge", "DeepSeekV3Bridge", "DeepSeekV4Bridge"]
+else:
+    __all__ = []
diff --git a/src/megatron/bridge/models/model_provider.py b/src/megatron/bridge/models/model_provider.py
index 4717974e..ea7fdeff 100644
--- a/src/megatron/bridge/models/model_provider.py
+++ b/src/megatron/bridge/models/model_provider.py
@@ -71,9 +71,10 @@ def _apply_mixed_precision_wrapper(
         for submodule in model_module.modules():
             # Preserve the existing MCore expert-bias contract.
             if hasattr(submodule, "_maintain_float32_expert_bias"):
-                expert_bias = getattr(submodule, "expert_bias", None)
-                if expert_bias is not None:
-                    keep_in_fp32.append((submodule, "expert_bias", expert_bias.data.clone()))
+                for buffer_name in ("expert_bias", "qb_beta"):
+                    buffer = getattr(submodule, buffer_name, None)
+                    if buffer is not None:
+                        keep_in_fp32.append((submodule, buffer_name, buffer.data.float().clone()))
 
             # Model-specific modules can mark direct parameters that must not be
             # truncated by Float16Module's recursive half()/bfloat16() cast.
diff --git a/src/megatron/bridge/models/nemotron_omni/__init__.py b/src/megatron/bridge/models/nemotron_omni/__init__.py
index c2cf239a..86624859 100644
--- a/src/megatron/bridge/models/nemotron_omni/__init__.py
+++ b/src/megatron/bridge/models/nemotron_omni/__init__.py
@@ -14,33 +14,44 @@
 
 """Nemotron Omni model family (Vision-Language + Audio) for Megatron Bridge."""
 
-from megatron.bridge.models.nemotron_omni.modeling_nemotron_omni import NemotronOmniModel
-from megatron.bridge.models.nemotron_omni.modeling_nemotron_omni_llava import NemotronOmniLlavaModel
-from megatron.bridge.models.nemotron_omni.nemotron_omni_bridge import (
-    NemotronOmniBridge,
-    NemotronOmniLlavaBridge,
-)
-from megatron.bridge.models.nemotron_omni.nemotron_omni_provider import (
-    NemotronOmniLlavaModelProvider,
-    NemotronOmniModelProvider,
-    NemotronVLModelProvider,
+from megatron.core.models.multimodal import context_parallel as _context_parallel
+
+
+# TODO: Remove this guard when the Apertus2 MCore fork provides dynamic-resolution CP.
+NEMOTRON_OMNI_AVAILABLE = all(
+    hasattr(_context_parallel, name)
+    for name in ("gather_from_context_parallel_ranks_dynamic_res", "split_to_context_parallel_ranks_dynamic_res")
 )
 
+if NEMOTRON_OMNI_AVAILABLE:
+    from megatron.bridge.models.nemotron_omni.modeling_nemotron_omni import NemotronOmniModel
+    from megatron.bridge.models.nemotron_omni.modeling_nemotron_omni_llava import NemotronOmniLlavaModel
+    from megatron.bridge.models.nemotron_omni.nemotron_omni_bridge import (
+        NemotronOmniBridge,
+        NemotronOmniLlavaBridge,
+    )
+    from megatron.bridge.models.nemotron_omni.nemotron_omni_provider import (
+        NemotronOmniLlavaModelProvider,
+        NemotronOmniModelProvider,
+        NemotronVLModelProvider,
+    )
 
-__all__ = [
-    "NemotronOmniModel",
-    "NemotronOmniBridge",
-    "NemotronOmniModelProvider",
-    "NemotronOmniLlavaModel",
-    "NemotronOmniLlavaBridge",
-    "NemotronOmniLlavaModelProvider",
-    "NemotronVLModelProvider",
-    "BridgeSoundEncoder",
-]
+    __all__ = [
+        "NemotronOmniModel",
+        "NemotronOmniBridge",
+        "NemotronOmniModelProvider",
+        "NemotronOmniLlavaModel",
+        "NemotronOmniLlavaBridge",
+        "NemotronOmniLlavaModelProvider",
+        "NemotronVLModelProvider",
+        "BridgeSoundEncoder",
+    ]
+else:
+    __all__ = []
 
 
 def __getattr__(name: str):
-    if name == "BridgeSoundEncoder":
+    if name == "BridgeSoundEncoder" and NEMOTRON_OMNI_AVAILABLE:
         from megatron.bridge.models.nemotron_omni.nemotron_omni_sound import BridgeSoundEncoder
 
         return BridgeSoundEncoder
CHONK_PATCH_EOF
cat > "${RUN_DIR}/patches/bridge-md-decoupling-optimizer.patch" << 'CHONK_PATCH_EOF'
# Applied by train-gsm8k-chonk-megatron-gen.sh (embedded byte-identically): dispatches
# optimizer=md_decoupling to the Megatron-LM fork's factory.
# Base: NVIDIA-NeMo/Megatron-Bridge@077224531ad4d6d0caef767a84e619b256aa5cb6. Source:
# swiss-ai/Megatron-Bridge@f3bc8e7e6a722adde1176b2ba8980109e09493ef (Apache-2.0), first hunk of
# training/optim.py only. Apply after bridge-apertus2-shared-minimal.patch.
diff --git a/src/megatron/bridge/training/optim.py b/src/megatron/bridge/training/optim.py
index 32a77160..32ef9423 100644
--- a/src/megatron/bridge/training/optim.py
+++ b/src/megatron/bridge/training/optim.py
@@ -85,7 +85,20 @@ def setup_optimizer(
                 f"applied {len(mup_overrides)} optimizer param-group override(s)."
             )
 
-    if hasattr(optimizer_config, "provide"):
+    if optimizer_config.optimizer == "md_decoupling":
+        # MDDecoupling has its own parameter partitioning and cannot use the
+        # generic optimizer factory (which would flatten matrix parameters).
+        from megatron.core.optimizer.md_decoupling import get_megatron_mddecoupling_optimizer
+
+        optimizer = get_megatron_mddecoupling_optimizer(
+            config=optimizer_config,
+            model_chunks=model_chunks,
+            config_overrides=config_overrides,
+            use_gloo_process_groups=use_gloo_process_groups,
+            layer_wise_distributed_optimizer=optimizer_config.use_layer_wise_distributed_optimizer,
+            pg_collection=pg_collection,
+        )
+    elif hasattr(optimizer_config, "provide"):
         optimizer = optimizer_config.provide(
             model_chunks=model,
             config_overrides=config_overrides,
CHONK_PATCH_EOF
cat > "${RUN_DIR}/patches/mlm-refit-execution-batch-bytes.patch" << 'CHONK_PATCH_EOF'
# Native refit on the swiss-ai Megatron-LM-MoE fork: upstream-compatible API + NCCL deadlock fix + parallel windows.
# Base: swiss-ai/Megatron-LM-MoE@b0844260b2a0888c711dca230b88ec13b20f3229 (Apache-2.0).
#   megatron/core/resharding/refit.py
#       local: accept execution_batch_bytes in prepare_swap_model_weights / swap_model_weights /
#       get_or_create_service with upstream Megatron-Core's signature (NVIDIA/Megatron-LM #6878,
#       5eb0744c7). Vanilla NeMo-RL f196ef0a always passes it (resolved from
#       refit_execution_batch_bytes, default: its packed-tensor size). With a string nccl_m2n backend
#       the value sets M2N's grouped-submission limit, as upstream does; other backends keep the
#       fork's model-wide submission, upstream's bounded batching is NOT ported. A faithful port
#       needs #5872, #5187, #6826, #6133 and #6878 with conflicts in execution.py / refit.py.
#   megatron/core/resharding/copy_services/nccl_copy_service.py
#       from upstream NVIDIA/Megatron-LM #7189 (325c9936b), its run()-side layer only: issue the
#       plan's P2P ops in windows of 256 global task ids instead of one batch_isend_irecv. One
#       ncclGroup with the whole model is split by NCCL into rank-dependent kernel plans and
#       deadlocks (pytorch/pytorch#174288); seen here as a 600 s NCCL timeout on all 96 ranks in
#       the first refit (run 3659197). The planner-side batching of #7189 is not ported.
#   megatron/core/resharding/planner.py
#       local: number the global task ids round-robin over destination ranks instead of
#       destination rank by destination rank. With rank-major ids each 256-id window above holds
#       one generator rank's transfers, so the 64 generator ranks were filled one at a time
#       (run 3667575: ~92 s per refit, ~25 GB/s for ~2.3 TB). Upstream #7189 instead splits the
#       plan into parameter batches spanning all ranks; the fork has no plan batching.
diff --git a/megatron/core/resharding/copy_services/nccl_copy_service.py b/megatron/core/resharding/copy_services/nccl_copy_service.py
index 95f5ad5da..37c78289d 100644
--- a/megatron/core/resharding/copy_services/nccl_copy_service.py
+++ b/megatron/core/resharding/copy_services/nccl_copy_service.py
@@ -2,6 +2,7 @@
 from __future__ import annotations
 
 import logging
+import os
 from typing import List, Optional
 
 import torch
@@ -18,6 +19,19 @@ class NCCLCopyService(CopyService):
     a batch of point-to-point sends and recvs.
     """
 
+    # From upstream NVIDIA/Megatron-LM #7189 (325c9936b), windowing layer only: a
+    # single ncclGroup holding a whole model's P2P ops is split by NCCL into
+    # several kernel plans at rank-dependent points, and matching sends/recvs
+    # that end up in different plans deadlock
+    # (https://github.com/pytorch/pytorch/issues/174288). run() therefore issues
+    # the remote ops as consecutive batch_isend_irecv calls over windows of
+    # TASK_WINDOW global task ids. Sender and receiver of a transfer share its
+    # task id, so both place it in the same window. Upstream windows only plans
+    # with more than TASK_WINDOW tasks; this fork's plans carry no global task
+    # count, so it windows whenever every op has a task id, which is the same
+    # decision on every rank. Set to 0 to disable.
+    TASK_WINDOW = int(os.environ.get("MEGATRON_NCCL_COPY_TASK_WINDOW", "256"))
+
     def __init__(self, group=None):
         super().__init__(group=group)
         self.send_ops: List[SendOp] = []
@@ -43,6 +57,37 @@ class NCCLCopyService(CopyService):
         """Fork execution adapter retained for task-ID-aware reshard calls."""
         self.submit_recv(dest_tensor, src_rank, task_id=task_id)
 
+    def _should_window(self, remote_sends: List[SendOp], remote_recvs: List[RecvOp]) -> bool:
+        if self.TASK_WINDOW <= 0:
+            return False
+        return all(op.task_id is not None for op in remote_sends) and all(
+            op.task_id is not None for op in remote_recvs
+        )
+
+    def _run_in_task_windows(self, remote_sends: List[SendOp], remote_recvs: List[RecvOp]) -> None:
+        """Issue the pending remote ops as one batch_isend_irecv per global task-id window."""
+        width = self.TASK_WINDOW
+        windows: dict[int, list] = {}
+        for op in remote_sends:
+            windows.setdefault(op.task_id // width, []).append(
+                dist.P2POp(dist.isend, op.tensor, op.dest_rank, group=self.group)
+            )
+        for op in remote_recvs:
+            windows.setdefault(op.task_id // width, []).append(
+                dist.P2POp(dist.irecv, op.tensor, op.src_rank, group=self.group)
+            )
+        if self.rank == 0:
+            logger.info(
+                "Executing batched communication in %d task-id windows of %d (max %d ops/window)",
+                len(windows),
+                width,
+                max((len(w) for w in windows.values()), default=0),
+            )
+        for key in sorted(windows):
+            reqs = dist.batch_isend_irecv(windows[key])
+            for req in reqs:
+                req.wait()
+
     def run(self):
         total_ops = len(self.send_ops) + len(self.recv_ops)
         if self.rank == 0:
@@ -66,16 +111,19 @@ class NCCLCopyService(CopyService):
                 for send_op, recv_op in pairs:
                     recv_op.tensor.copy_(send_op.tensor)
 
-        p2p_ops = []
-        for op in remote_sends:
-            p2p_ops.append(dist.P2POp(dist.isend, op.tensor, op.dest_rank, group=self.group))
-        for op in remote_recvs:
-            p2p_ops.append(dist.P2POp(dist.irecv, op.tensor, op.src_rank, group=self.group))
-
-        if p2p_ops:
-            reqs = dist.batch_isend_irecv(p2p_ops)
-            for req in reqs:
-                req.wait()
+        if self._should_window(remote_sends, remote_recvs):
+            self._run_in_task_windows(remote_sends, remote_recvs)
+        else:
+            p2p_ops = []
+            for op in remote_sends:
+                p2p_ops.append(dist.P2POp(dist.isend, op.tensor, op.dest_rank, group=self.group))
+            for op in remote_recvs:
+                p2p_ops.append(dist.P2POp(dist.irecv, op.tensor, op.src_rank, group=self.group))
+
+            if p2p_ops:
+                reqs = dist.batch_isend_irecv(p2p_ops)
+                for req in reqs:
+                    req.wait()
 
         # Make sure the copy stream is finished
         torch.cuda.current_stream().wait_stream(self._copy_stream)
diff --git a/megatron/core/resharding/planner.py b/megatron/core/resharding/planner.py
index f3423a01e..8c99b6db8 100644
--- a/megatron/core/resharding/planner.py
+++ b/megatron/core/resharding/planner.py
@@ -348,9 +348,19 @@ def _iter_global_transfer_ops(
     """
     # Shared between a send and its recv; NVSHMEM builds a schedule from it and
     # local copies match on it.
-    next_task_id = 0
+    #
+    # Task ids are numbered round-robin over the destination ranks (the k-th
+    # transfer of every destination rank, then the (k+1)-th, ...), not
+    # destination rank by destination rank. NCCLCopyService issues its P2P ops
+    # in consecutive windows of task ids; with rank-major ids a window holds one
+    # destination rank's transfers, so the destinations are served one at a time
+    # (run 3667575: ~92 s per refit, about one GPU's network bandwidth for the
+    # whole model). Upstream (NVIDIA/Megatron-LM #7189) avoids this with
+    # planner-side parameter batches, which this fork does not have.
+    per_dst: list[list[tuple]] = []
     for dst_rank in sorted(dst_param_metadata_by_rank):
         dst_rank_params = dst_param_metadata_by_rank[dst_rank]
+        transfers = []
         for resolved_name, dst_metadata in dst_rank_params.items():
             src_meta_list = _find_source_metadata(src_param_metadata, resolved_name)
             if not src_meta_list:
@@ -364,9 +374,22 @@ def _iter_global_transfer_ops(
                 resolved_name, src_metadata, dst_metadata, dst_rank
             )
             for src_rank, src_slice, dst_slice in sources:
-                task_id = next_task_id
+                transfers.append(
+                    (dst_rank, src_rank, src_slice, dst_slice, src_metadata, dst_metadata)
+                )
+        per_dst.append(transfers)
+
+    task_ids: list[list[int]] = [[0] * len(transfers) for transfers in per_dst]
+    next_task_id = 0
+    for k in range(max((len(transfers) for transfers in per_dst), default=0)):
+        for i, transfers in enumerate(per_dst):
+            if k < len(transfers):
+                task_ids[i][k] = next_task_id
                 next_task_id += 1
-                yield task_id, dst_rank, src_rank, src_slice, dst_slice, src_metadata, dst_metadata
+
+    for i, transfers in enumerate(per_dst):
+        for k, transfer in enumerate(transfers):
+            yield (task_ids[i][k], *transfer)
 
 
 def _tensor_mesh(metadata: ParameterMetadata) -> tuple[int, ...]:
diff --git a/megatron/core/resharding/refit.py b/megatron/core/resharding/refit.py
index 77e25de9c..092667554 100644
--- a/megatron/core/resharding/refit.py
+++ b/megatron/core/resharding/refit.py
@@ -118,7 +118,9 @@ _service_cache: dict[tuple[str, int | None], CopyService] = {}
 _plan_cache: dict[_PlanCacheKey, Any] = {}
 
 
-def get_or_create_service(backend: RefitBackendName, group=None) -> CopyService:
+def get_or_create_service(
+    backend: RefitBackendName, group=None, execution_batch_bytes: int | None = None
+) -> CopyService:
     """Get or create a cached CopyService instance for the given backend.
 
     This avoids expensive repeated allocations (especially for NVSHMEM buffers)
@@ -127,15 +129,19 @@ def get_or_create_service(backend: RefitBackendName, group=None) -> CopyService:
     Args:
         backend: Backend name ("nccl", "nccl_m2n", "gloo", or "nvshmem").
         group: Optional process group for the backend.
+        execution_batch_bytes: Optional soft byte limit for execution staging.
+            For NCCL M2N, this overrides its grouped-submission limit. None
+            preserves M2N's existing environment setting or 256 MiB default.
     """
-    cache_key = (backend, id(group) if group is not None else None)
+    service_batch_bytes = execution_batch_bytes if backend == "nccl_m2n" else None
+    cache_key = (backend, id(group) if group is not None else None, service_batch_bytes)
     if cache_key in _service_cache:
         return _service_cache[cache_key]
 
     if backend == "nccl":
         service = NCCLCopyService(group=group)
     elif backend == "nccl_m2n":
-        service = NCCLM2NCopyService(group=group)
+        service = NCCLM2NCopyService(group=group, max_group_bytes=execution_batch_bytes)
     elif backend == "gloo":
         service = GlooCopyService(group=group)
     elif backend == "nvshmem":
@@ -327,6 +333,7 @@ def prepare_swap_model_weights(
     group=None,
     src_rank_offset: int = 0,
     dst_rank_offset: int = 0,
+    execution_batch_bytes: int | None = None,
 ):
     """Pre-build and cache the reshard plan and any format-conversion transforms.
 
@@ -356,7 +363,11 @@ def prepare_swap_model_weights(
         group: Optional process group for collective communication.
         src_rank_offset: Rank offset for source (training) workers.
         dst_rank_offset: Rank offset for destination (inference) workers.
+        execution_batch_bytes: Accepted for API compatibility with upstream
+            Megatron-Core (NVIDIA/Megatron-LM #6878). This fork builds one
+            model-wide plan, so the value does not change the plan.
     """
+    del execution_batch_bytes  # API compatibility only, see the docstring.
     src_core, tgt_core, num_experts = _unwrap_model_cores(src_model, target_model)
     plan = _build_or_get_plan(
         src_core, tgt_core, num_experts, group, src_rank_offset, dst_rank_offset
@@ -378,6 +389,7 @@ def swap_model_weights(
     transform: Optional[ReshardTransform] = None,
     num_dst_pools: int = 1,
     dst_pool_index: int = 0,
+    execution_batch_bytes: int | None = None,
 ):
     """
     Orchestrate weight swap/refit.
@@ -397,9 +409,16 @@ def swap_model_weights(
         transform: Optional ReshardTransform for custom format conversion.
             If None, the cached transform (from prepare_swap_model_weights)
             is used automatically when the receiver needs MXFP8 conversion.
+        execution_batch_bytes: Accepted for API compatibility with upstream
+            Megatron-Core (NVIDIA/Megatron-LM #6878). With a string
+            ``nccl_m2n`` backend it sets M2N's grouped-submission limit, as
+            upstream does. Other backends keep this fork's model-wide
+            submission; upstream's bounded batching is not ported.
     """
     if isinstance(refit_method, str):
-        service = get_or_create_service(refit_method, group=group)
+        service = get_or_create_service(
+            refit_method, group=group, execution_batch_bytes=execution_batch_bytes
+        )
     elif isinstance(refit_method, CopyService):
         service = refit_method
     else:
CHONK_PATCH_EOF
cat > "${RUN_DIR}/patches/nemo-rl-benchmark-metrics.patch" << 'CHONK_PATCH_EOF'
# Benchmark-only NeMo-RL patch (train-gsm8k-chonk-megatron-gen-benchmark.sh); apply after
# nemo-rl-chonk-megatron-gen-minimal.patch. Base: NVIDIA-NeMo/RL@f196ef0a. Marked CHONK_BENCHMARK.
#   * Validation sampling on the plain (non-NeMo-Gym) rollout path: validate() switches the
#     Megatron generation workers to generation.val_temperature / val_top_p / val_top_k and back
#     (vanilla only honours them on the NeMo-Gym path). New set_generation_sampling() on
#     MegatronGeneration and its worker mixin.
#   * Verl-comparable actor metrics in ClippedPGLossFn: entropy (exact, from the full logits,
#     opt-in with NRL_LOG_TOKEN_ENTROPY=1 in prepare_loss_input), pg_clipfrac, ppo_kl (k1).
#   * Verl-named copies of the step metrics logged next to NeMo-RL's own (critic/*,
#     response_length/*, prompt_length/*, actor/*, training/rollout_probs_diff_*, perf/*,
#     timing_s/*, training/global_step|epoch) and of validation
#     (val-core/<NRL_VERL_VAL_DATA_SOURCE, default openai/gsm8k>/reward/mean@k).
diff --git a/nemo_rl/algorithms/grpo.py b/nemo_rl/algorithms/grpo.py
index e155183..bcc44b7 100644
--- a/nemo_rl/algorithms/grpo.py
+++ b/nemo_rl/algorithms/grpo.py
@@ -2559,6 +2559,126 @@ def _create_advantage_estimator(master_config: MasterConfig):
     return adv_estimator
 
 
+# CHONK_BENCHMARK: Verl-named copies of the GRPO step metrics, so a NeMo-RL run
+# and a Verl run can be overlaid in wandb. Verl's names: critic/score,
+# critic/rewards, critic/advantages, critic/returns, response_length,
+# prompt_length, actor/*, training/rollout_probs_diff_*, perf/*, timing_s/*.
+_VERL_TIMING_KEYS = {
+    "generation": "timing_s/gen",
+    "reward_calculation": "timing_s/reward",
+    "policy_and_reference_logprobs": "timing_s/old_log_prob",
+    "advantage_calculation": "timing_s/adv",
+    "policy_training": "timing_s/update_actor",
+    "prepare_for_generation/transfer_and_update_weights": "timing_s/update_weights",
+    "total_step_time": "timing_s/step",
+}
+
+
+def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
+    return cfg.get(key, default) if isinstance(cfg, dict) else getattr(cfg, key, default)
+
+
+def _verl_stats(out: dict[str, float], name: str, x: torch.Tensor, clip: Any = None) -> None:
+    x = x.detach().float().flatten()
+    if x.numel() == 0:
+        return
+    out[f"{name}/mean"] = x.mean().item()
+    out[f"{name}/max"] = x.max().item()
+    out[f"{name}/min"] = x.min().item()
+    if clip is not None:
+        out[f"{name}/clip_ratio"] = (x >= clip).float().mean().item()
+
+
+def _verl_compat_train_metrics(
+    *,
+    metrics: dict[str, Any],
+    timing_metrics: dict[str, Any],
+    train_data: Any,
+    repeated_batch: Any,
+    input_lengths: torch.Tensor,
+    master_config: "MasterConfig",
+    total_num_gpus: int,
+    epoch: int,
+    step: int,
+) -> dict[str, float]:
+    """Verl-named step metrics, computed from the step's training batch.
+
+    NeMo-RL sums its token-normalized loss metrics over the optimizer batches
+    of a step (each batch normalized by its own token count); Verl reports the
+    mean over mini-batches, hence the division by the number of batches.
+    """
+    out: dict[str, float] = {}
+    token_mask = train_data["token_mask"].bool() & train_data["sample_mask"].bool().unsqueeze(-1)
+    scores = torch.as_tensor(repeated_batch["total_reward"])
+    _verl_stats(out, "critic/score", scores)
+    _verl_stats(out, "critic/rewards", scores)
+    advantages = train_data["advantages"][token_mask]
+    _verl_stats(out, "critic/advantages", advantages)
+    _verl_stats(out, "critic/returns", advantages)
+    _verl_stats(
+        out,
+        "response_length",
+        train_data["token_mask"].sum(dim=-1),
+        clip=master_config.policy["generation"]["max_new_tokens"],
+    )
+    _verl_stats(
+        out,
+        "prompt_length",
+        torch.as_tensor(repeated_batch["length"]),
+        clip=_cfg_get(master_config.data, "max_input_seq_length"),
+    )
+    probs_diff = (
+        train_data["generation_logprobs"][token_mask].exp()
+        - train_data["prev_logprobs"][token_mask].exp()
+    ).abs()
+    if probs_diff.numel() > 0:
+        out["training/rollout_probs_diff_max"] = probs_diff.max().item()
+        out["training/rollout_probs_diff_mean"] = probs_diff.mean().item()
+        out["training/rollout_probs_diff_std"] = probs_diff.std().item()
+
+    num_batches = max(
+        1, len(train_data["advantages"]) // master_config.policy["train_global_batch_size"]
+    )
+    for verl_name, name in (
+        ("actor/entropy", "entropy"),
+        ("actor/pg_clipfrac", "pg_clipfrac"),
+        ("actor/ppo_kl", "ppo_kl"),
+        ("actor/pg_loss", "loss"),
+        ("actor/kl_loss", "kl_penalty"),
+    ):
+        if name in metrics:
+            out[verl_name] = float(np.mean(metrics[name])) / num_batches
+    if "approx_entropy" in metrics:
+        out["actor/entropy_approx"] = float(np.mean(metrics["approx_entropy"])) / num_batches
+    out["actor/grad_norm"] = float(np.mean(metrics["grad_norm"]))
+    if "lr" in metrics:
+        out["actor/lr"] = float(np.mean(metrics["lr"]))
+    out["actor/kl_coef"] = float(_cfg_get(master_config.loss_fn, "reference_policy_kl_penalty", 0.0))
+
+    total_tokens = float(input_lengths.sum().item())
+    step_time = float(timing_metrics.get("total_step_time", 0.0))
+    out["perf/total_num_tokens"] = total_tokens
+    out["perf/time_per_step"] = step_time
+    if step_time > 0:
+        out["perf/throughput"] = total_tokens / step_time / total_num_gpus
+    for name, verl_name in _VERL_TIMING_KEYS.items():
+        if name in timing_metrics:
+            out[verl_name] = float(timing_metrics[name])
+    out["training/global_step"] = float(step)
+    out["training/epoch"] = float(epoch)
+    return out
+
+
+def _verl_compat_val_metrics(val_metrics: dict[str, Any], k: int) -> dict[str, float]:
+    """Verl's val-core/<data_source>/reward/mean@k naming for the validation metrics."""
+    source = os.environ.get("NRL_VERL_VAL_DATA_SOURCE", "openai/gsm8k")
+    out = {f"val-core/{source}/reward/mean@{k}": float(val_metrics["accuracy"])}
+    if "pass_k" in val_metrics:
+        out[f"val-core/{source}/reward/best@{k}/mean"] = float(val_metrics["pass_k"])
+    out[f"val-aux/{source}/response_length/mean"] = float(val_metrics["avg_length"])
+    return out
+
+
 def _clip_grpo_advantages(
     advantages: torch.Tensor,
     grpo_config: GRPOConfig,
@@ -4101,6 +4221,22 @@ def _grpo_train_impl(
 
             if refit_metrics:
                 logger.log_metrics(refit_metrics, total_steps + 1, prefix="refit")
+            # CHONK_BENCHMARK: Verl-named copies of this step's metrics.
+            logger.log_metrics(
+                _verl_compat_train_metrics(
+                    metrics=metrics,
+                    timing_metrics=timing_metrics,
+                    train_data=train_data,
+                    repeated_batch=repeated_batch,
+                    input_lengths=input_lengths,
+                    master_config=master_config,
+                    total_num_gpus=total_num_gpus,
+                    epoch=current_epoch,
+                    step=total_steps + 1,
+                ),
+                total_steps + 1,
+                prefix="",
+            )
             logger.log_metrics(metrics, total_steps + 1, prefix="train")
             logger.log_metrics(
                 performance_metrics, total_steps + 1, prefix="performance"
@@ -4238,6 +4374,19 @@ def validate(
         bucket_scope(Bucket.OVERHEAD),
     ):
         print(f"▶ Starting validation at step {step}...", flush=True)
+        # CHONK_BENCHMARK: the plain rollout path samples with the generation
+        # workers' config; switch it to the validation profile (NeMo-Gym
+        # passes val_* per request instead) and back after the loop.
+        generation_config = master_config.policy["generation"]
+        switch_val_sampling = not should_use_nemo_gym(master_config) and hasattr(
+            policy_generation, "set_generation_sampling"
+        )
+        if switch_val_sampling:
+            policy_generation.set_generation_sampling(
+                temperature=generation_config["val_temperature"],
+                top_p=generation_config["val_top_p"],
+                top_k=generation_config["val_top_k"],
+            )
         # >= 1 is validated in setup().
         val_num_generations_per_prompt = (
             master_config.grpo.val_num_generations_per_prompt
@@ -4346,6 +4495,13 @@ def validate(
 
             all_message_logs.extend(to_env)
 
+        if switch_val_sampling:
+            policy_generation.set_generation_sampling(
+                temperature=generation_config["temperature"],
+                top_p=generation_config["top_p"],
+                top_k=generation_config["top_k"],
+            )
+
         # Calculate validation metrics. accuracy is the mean reward over all
         # rollouts; grouped validation (val_num_generations_per_prompt > 1)
         # additionally reports pass@k over each prompt's k rollouts as pass_k.
@@ -4380,6 +4536,12 @@ def validate(
         }
         if pass_k is not None:
             val_metrics["pass_k"] = pass_k
+        if logger is not None:  # CHONK_BENCHMARK
+            logger.log_metrics(
+                _verl_compat_val_metrics(val_metrics, val_num_generations_per_prompt),
+                step,
+                prefix="",
+            )
 
         # Print sample conversations only once at the end of validation
         try:
diff --git a/nemo_rl/algorithms/loss/loss_functions.py b/nemo_rl/algorithms/loss/loss_functions.py
index 1eecd5c..2f6bcfe 100755
--- a/nemo_rl/algorithms/loss/loss_functions.py
+++ b/nemo_rl/algorithms/loss/loss_functions.py
@@ -377,6 +377,10 @@ class ClippedPGLossFn(LossFunction):
             "policy_kl_error": MetricNormalizer.TOKENS,
             "js_divergence_error": MetricNormalizer.TOKENS,
             "approx_entropy": MetricNormalizer.TOKENS,
+            # CHONK_BENCHMARK: Verl-comparable actor metrics (see __call__).
+            "entropy": MetricNormalizer.TOKENS,
+            "pg_clipfrac": MetricNormalizer.TOKENS,
+            "ppo_kl": MetricNormalizer.TOKENS,
             # Keyed on sequence_level_importance_ratios, NOT loss_type.
             "sampling_importance_ratio": (
                 MetricNormalizer.SEQUENCES
@@ -808,6 +812,35 @@ class ClippedPGLossFn(LossFunction):
                 global_normalization_factor=global_valid_toks,
             )
 
+        # CHONK_BENCHMARK: Verl-comparable actor metrics. entropy: exact token
+        # entropy of the current policy from the full logits (prepare_loss_input
+        # stores it as data["token_entropy"]); pg_clipfrac: fraction of tokens
+        # whose clipped surrogate term is the active one; ppo_kl: k1 KL
+        # E[log pi_old - log pi_curr] over the response tokens.
+        with torch.no_grad():
+            token_entropy = data.get("token_entropy") if hasattr(data, "get") else None
+            entropy_metric = (
+                masked_mean(
+                    token_entropy, mask, global_normalization_factor=global_valid_toks
+                ).item()
+                if token_entropy is not None
+                else 0.0
+            )
+            pg_clipfrac = (
+                masked_mean(
+                    (loss2 > loss1).float(),
+                    mask,
+                    global_normalization_factor=global_valid_toks,
+                ).item()
+                if not self.use_cispo
+                else 0.0
+            )
+            ppo_kl = masked_mean(
+                prev_logprobs - curr_logprobs.detach(),
+                mask,
+                global_normalization_factor=global_valid_toks,
+            ).item()
+
         # -----------------------------------------------------------------
         # VAPO: positive-example NLL loss on correct samples (reward > 0)
         # L = L_PPO + μ · L_NLL(correct)
@@ -874,6 +907,9 @@ class ClippedPGLossFn(LossFunction):
                 "sampling_importance_ratio": sample_importance_ratio.item(),
                 "num_valid_samples": sample_mask.sum().item(),
                 "approx_entropy": seq_entropy_approx.item(),
+                "entropy": entropy_metric,
+                "pg_clipfrac": pg_clipfrac,
+                "ppo_kl": ppo_kl,
                 **_is_filter_metrics,
                 "positive_nll_loss": nll_loss.item(),
             },
diff --git a/nemo_rl/algorithms/loss/loss_input.py b/nemo_rl/algorithms/loss/loss_input.py
index a62da1e..5295102 100644
--- a/nemo_rl/algorithms/loss/loss_input.py
+++ b/nemo_rl/algorithms/loss/loss_input.py
@@ -14,6 +14,7 @@
 
 """Builds the loss-function input dict from model logits."""
 
+import os
 from typing import TYPE_CHECKING, Any, Optional
 
 import torch
@@ -197,6 +198,28 @@ def _prepare_opd_full_loss_input(
     return loss_input
 
 
+@torch.no_grad()
+def _token_entropy(
+    logits: torch.Tensor,
+    vocab_parallel_group: Optional[torch.distributed.ProcessGroup],
+    chunk_size: int = 256,
+) -> torch.Tensor:
+    """CHONK_BENCHMARK: [B, S] entropy -sum_v p log p of next-token logits."""
+    if (
+        vocab_parallel_group is not None
+        and torch.distributed.get_world_size(vocab_parallel_group) > 1
+    ):
+        # ChunkedDistributedEntropy returns sum_v p log p (the negative entropy).
+        return -ChunkedDistributedEntropy.apply(  # type: ignore[misc]
+            logits, chunk_size, vocab_parallel_group, True
+        )
+    out = []
+    for s0 in range(0, logits.shape[1], chunk_size):
+        log_p = torch.log_softmax(logits[:, s0 : s0 + chunk_size].float(), dim=-1)
+        out.append(-(log_p.exp() * log_p).sum(dim=-1))
+    return torch.cat(out, dim=1)
+
+
 def prepare_loss_input(
     logits: torch.Tensor,
     data: BatchedDataDict[Any],
@@ -292,6 +315,13 @@ def prepare_loss_input(
                     cp_sharder=cp_sharder,
                 )
 
+        # CHONK_BENCHMARK: exact per-token entropy of the current policy for the
+        # Verl-comparable actor/entropy metric (opt-in, metric only, no grad).
+        if os.environ.get("NRL_LOG_TOKEN_ENTROPY") == "1" and logits.dim() == 3:
+            data["token_entropy"] = _token_entropy(
+                logits[:, : data["input_ids"].shape[1] - 1], vocab_parallel_group
+            )
+
         loss_input = {"next_token_logprobs": logprobs}
 
     elif loss_fn.input_type == LossInputType.OPD_FULL:
diff --git a/nemo_rl/models/generation/megatron/megatron_generation.py b/nemo_rl/models/generation/megatron/megatron_generation.py
index 5b4e6fd..5795338 100644
--- a/nemo_rl/models/generation/megatron/megatron_generation.py
+++ b/nemo_rl/models/generation/megatron/megatron_generation.py
@@ -600,6 +600,22 @@ class MegatronGeneration(GenerationInterface):
             result_batch["gen_leader_worker_idx"] = [0]
             yield index, result_batch
 
+    def set_generation_sampling(
+        self, temperature: float, top_p: float, top_k: Optional[int]
+    ) -> None:
+        """CHONK_BENCHMARK: switch every generation worker's sampling profile.
+
+        Requests are built on worker 0 from its config, read per request, so
+        validation can sample with its own temperature / top-p / top-k.
+        """
+        futures = self._policy.worker_group.run_all_workers_single_data(
+            "set_generation_sampling",
+            temperature=temperature,
+            top_p=top_p,
+            top_k=top_k,
+        )
+        ray.get(futures)
+
     def prepare_for_generation(self, *args: Any, **kwargs: Any) -> bool:
         """Initialize / re-enter inference mode on every worker.
 
diff --git a/nemo_rl/models/generation/megatron/megatron_worker.py b/nemo_rl/models/generation/megatron/megatron_worker.py
index b55954a..f36db40 100644
--- a/nemo_rl/models/generation/megatron/megatron_worker.py
+++ b/nemo_rl/models/generation/megatron/megatron_worker.py
@@ -972,6 +972,15 @@ class MegatronGenerationMixin:
         """Return this worker's OpenAI server base URL (None if not the leader)."""
         return self.base_url
 
+    def set_generation_sampling(
+        self, temperature: float, top_p: float, top_k: Optional[int]
+    ) -> None:
+        """CHONK_BENCHMARK: sampling profile for subsequent requests (validation)."""
+        gen_cfg = self.cfg["generation"]
+        gen_cfg["temperature"] = temperature
+        gen_cfg["top_p"] = top_p
+        gen_cfg["top_k"] = top_k
+
     def _build_sampling_params(
         self,
         greedy: bool,
CHONK_PATCH_EOF
cat > "${RUN_DIR}/patches/nemo-rl-chonk-megatron-gen-minimal.patch" << 'CHONK_PATCH_EOF'
# Minimal edits to NeMo-RL for Apertus 2 (Chonk) with Megatron generation on the nemo-rl-cuda image.
# Base: NVIDIA-NeMo/RL@f196ef0a1ccb54d6ce71cd63277fb6a0e01168db (Apache-2.0).
#   nemo_rl/models/generation/megatron/megatron_worker.py
#       from swiss-ai/Nemo-RL PR #1: hand the KDA (linear-attention) per-request state config to
#       Megatron-Core's inference engine, and advertise the node IP (not the container hostname) to
#       the data-parallel inference coordinator.
#   nemo_rl/models/policy/__init__.py
#       local: the image ships transformers 5.17; turn the '< 5.13' assert guarding a DeepSeek-V3
#       tokenizer workaround into a printed warning.
diff --git a/nemo_rl/models/generation/megatron/megatron_worker.py b/nemo_rl/models/generation/megatron/megatron_worker.py
index 6c88b7d..b55954a 100644
--- a/nemo_rl/models/generation/megatron/megatron_worker.py
+++ b/nemo_rl/models/generation/megatron/megatron_worker.py
@@ -528,6 +528,15 @@ class MegatronGenerationMixin:
         mamba_inference_state_config = MambaInferenceStateConfig.from_model(
             inference_model
         )
+        kda_inference_state_config = None
+        if getattr(model_config, "experimental_attention_variant", None) == "kda":
+            from megatron.core.inference.config import KDAInferenceStateConfig
+
+            kda_inference_state_config = KDAInferenceStateConfig.from_model(
+                inference_model
+            )
+            if kda_inference_state_config is None:
+                raise ValueError("KDA model has no KDA inference state")
         is_hybrid_model = mamba_inference_state_config is not None
         if is_hybrid_model:
             if (
@@ -590,6 +599,10 @@ class MegatronGenerationMixin:
             "image_preprocessing_config": image_preprocessing_config,
             "video_preprocessing_config": video_preprocessing_config,
         }
+        if kda_inference_state_config is not None:
+            inference_config_kwargs["kda_inference_state_config"] = (
+                kda_inference_state_config
+            )
         _apply_optional_inference_config_kwargs(
             inference_config_kwargs, mcore_generation_config
         )
@@ -627,9 +640,12 @@ class MegatronGenerationMixin:
 
     async def _start_inference_coordinator(self):
         """Start the inference coordinator and engine loop."""
+        from nemo_rl.distributed.virtual_cluster import _get_node_ip_local
+
         self.coordinator_addr = await self.dynamic_inference_engine.start_listening_to_data_parallel_coordinator(
             inference_coordinator_port=None,
             launch_inference_coordinator=True,
+            hostname=_get_node_ip_local(),
         )
         if torch.distributed.get_rank() == 0:
             from megatron.core.inference.inference_client import InferenceClient
diff --git a/nemo_rl/models/policy/__init__.py b/nemo_rl/models/policy/__init__.py
index 1c0c1f4..34eb287 100644
--- a/nemo_rl/models/policy/__init__.py
+++ b/nemo_rl/models/policy/__init__.py
@@ -50,7 +50,7 @@ def _patch_transformers_tokenizer_class_set():
     # TODO: remove this patch (and the assert below) once the deepseek_v3
     # entries actually disappear upstream.
     # https://github.com/NVIDIA-NeMo/RL/issues/2764
-    assert PkgVersion(transformers.__version__) < PkgVersion("5.13.0"), (
+    if not PkgVersion(transformers.__version__) < PkgVersion("5.13.0"): print(
         f"transformers {transformers.__version__} detected. "
         "The deepseek_v3 tokenizer-blocklist patch was verified against <5.13. "
         "Check if the upstream fix now applies and remove this patch if so."
CHONK_PATCH_EOF

# ── Recipe: the author's grpo-chonk-vllm.yaml, generation block replaced ─────
# Quoted heredoc: the OmegaConf interpolations must reach the file verbatim.
cat > "${RUN_DIR}/grpo-chonk-megatron-gen.yaml" << 'CHONK_YAML_EOF'
defaults:
  - ${oc.env:NEMO_RL_PATH}/examples/configs/grpo_math_1B_megatron.yaml

cluster:
  num_nodes: ${oc.decode:${oc.env:TOTAL_NODES}}
  gpus_per_node: 4

grpo:
  async_grpo:
    enabled: false
  max_num_steps: ${oc.decode:${oc.env:MAX_NUM_STEPS}}
  max_num_epochs: 6
  num_prompts_per_step: ${oc.decode:${oc.env:NUM_PROMPTS}}
  num_generations_per_prompt: ${oc.decode:${oc.env:NUM_GENERATIONS}}
  normalize_rewards: true
  use_leave_one_out_baseline: true
  advantage_clip_low: -5.0
  advantage_clip_high: 5.0
  # Validation on the GSM8K test split at the start, the middle and the end
  # (steps 0, VAL_PERIOD and MAX_NUM_STEPS), the whole split in one batch.
  val_period: ${oc.decode:${oc.env:VAL_PERIOD}}
  val_at_start: true
  val_at_end: true
  val_batch_size: 1319
  max_val_samples: 1319
  val_num_generations_per_prompt: 1
  seed: 42

checkpointing:
  enabled: false
  checkpoint_dir: ${oc.env:RUN_DIR}/checkpoints
  save_period: 50
  metric_name: null

policy:
  model_name: ${oc.env:HF_MODEL}
  tokenizer:
    name: ${oc.env:HF_MODEL}
    chat_template: "{{ bos_token }}{% for message in messages %}{{ message['content'] }}{% endfor %}"
  precision: bfloat16
  train_global_batch_size: ${oc.decode:${oc.env:TRAIN_GLOBAL_BATCH}}
  train_micro_batch_size: ${oc.decode:${oc.env:TRAIN_MICRO_BATCH}}
  generation_batch_size: ${oc.decode:${oc.env:INFERENCE_REQUESTS}}
  logprob_batch_size: 1
  offload_optimizer_for_logprob: true
  max_total_sequence_length: ${oc.decode:${oc.env:MAX_TOTAL_SEQUENCE_LENGTH}}
  make_sequence_length_divisible_by: 2
  sequence_packing:
    enabled: false
  dynamic_batching:
    enabled: false

  megatron_cfg:
    enabled: true
    tensor_model_parallel_size: 1
    pipeline_model_parallel_size: 1
    # EP32 (8 of the 256 experts per trainer GPU) plus activation checkpointing:
    # without expert offloading EP16 does not fit a 95 GB GPU. Run 3640152 ran out
    # of memory in the forward of a later micro-batch with ~49 GB of weights and
    # gradients, ~31 GB of activations and ~12 GB outside PyTorch, before any Adam
    # state had been allocated.
    expert_model_parallel_size: 32
    expert_tensor_parallel_size: 1
    context_parallel_size: 1
    sequence_parallel: true
    activation_checkpointing: true
    gradient_accumulation_fusion: true
    cuda_graph_impl: none
    freeze_moe_router: true
    moe_router_load_balancing_type: quantile_balancing
    moe_router_dtype: fp32
    moe_router_bias_update_rate: 0.0
    moe_token_dispatcher_type: alltoall
    moe_permute_fusion: true
    apply_rope_fusion: false
    env_vars:
      NRL_REFIT_BUFFER_MEMORY_RATIO: "0.005"
      NRL_REFIT_NUM_BUFFERS: "1"
    # Keep the checkpoint's quantile-balancing thresholds (qb_beta) fixed. Vanilla
    # NeMo-RL silently drops moe_router_quantile_balancing_freeze as a top-level
    # megatron_cfg key (the author's fork forwards it), so the trainer re-estimated
    # qb_beta from every GSM8K batch and the refit shipped it to the generator:
    # reward 0.57 -> 0.25 after one step even at learning rate 0 (runs 3642870,
    # 3648142, 3652972). model_overrides is the vanilla route to model-config fields.
    # Not carried over from the author's model_overrides: coarse-grained expert
    # offloading with in-place FP8 parameters (needs swiss-ai Megatron-Bridge PR #4).
    model_overrides:
      moe_router_quantile_balancing_freeze: true
    checkpoint:
      async_save: false
      save: ${oc.env:RUN_DIR}/megatron-save
    # The author's block verbatim: the fork's md_decoupling optimizer Chonk was
    # pretrained with (Muon-style orthogonal updates on hypersphere-constrained
    # weights, learned gains kept in optimizer state). Needs
    # patches/bridge-md-decoupling-optimizer.patch (vanilla Bridge would route it
    # to the generic factory) and emerging-optimizers (in the image). It cannot
    # use the standard distributed optimizer (flattens matrices) and shards its
    # state layer-wise instead; the precision-aware optimizer is off, so the
    # moments are fp32 (the fork asserts that).
    optimizer:
      optimizer: md_decoupling
      lr: ${oc.decode:${oc.env:LR}}
      min_lr: 2.0e-7
      min_lr_mode: absolute
      matrix_lr: ${oc.decode:${oc.env:MATRIX_LR}}
      gains_lr: ${oc.decode:${oc.env:GAINS_LR}}
      weight_decay: ${oc.decode:${oc.env:WEIGHT_DECAY}}
      bf16: true
      fp16: false
      params_dtype: float32
      adam_beta1: 0.9
      adam_beta2: 0.95
      adam_eps: 1.0e-8
      use_distributed_optimizer: false
      use_layer_wise_distributed_optimizer: true
      use_precision_aware_optimizer: false
      hypersphere_preserve_init: true
      hypersphere_mode: row
      hypersphere_embedding_mode: row
      hypersphere_router_mode: row
      hypersphere_radius_mode: fan_in
      hypersphere_gains_mode: rowcol
      hypersphere_gains_mode_router: rowcol
      hypersphere_gains_mode_embedding: none
      gain_parametrization: softplus
      use_orthogonal_updates: true
      muon_coefficient_type: polar_express
      muon_scale_mode: shape_up
      muon_router_scale_mode: none
      muon_momentum: 0.95
      muon_use_nesterov: true
      clip_grad: ${policy.max_grad_norm}
      optimizer_cpu_offload: false
      optimizer_offload_fraction: 0.0
    scheduler:
      lr_decay_style: ${oc.env:LR_DECAY_STYLE}
      lr_decay_iters: 1200
      lr_warmup_iters: ${oc.decode:${oc.env:LR_WARMUP_ITERS}}
      lr_warmup_init: 1.0e-7
      start_weight_decay: ${oc.decode:${oc.env:WEIGHT_DECAY}}
      end_weight_decay: ${oc.decode:${oc.env:WEIGHT_DECAY}}
      weight_decay_incr_style: constant
    distributed_data_parallel_config:
      grad_reduce_in_fp32: false
      overlap_grad_reduce: true
      overlap_param_gather: false
      use_custom_fsdp: false
      data_parallel_sharding_strategy: optim_grads_params

  generation:
    backend: megatron
    temperature: 1.0
    top_p: 1.0
    top_k: -1
    # Validation sampling (patches/nemo-rl-benchmark-metrics.patch applies it on
    # this Megatron generation path).
    val_temperature: 0.6
    val_top_p: 0.95
    val_top_k: null
    max_new_tokens: ${oc.decode:${oc.env:MAX_NEW_TOKENS}}
    stop_strings: ["Question:", "</s>", "<|im_end|>"]
    colocated:
      enabled: false
      resources:
        num_nodes: ${oc.decode:${oc.env:INFERENCE_NODES}}
        gpus_per_node: 4
    # NeMo-RL's shard-to-shard reshard: the FFN/expert projections (95.9% of
    # Chonk's 201.7 GiB) move once per region trainer -> generator and are
    # broadcast across generator replicas; the rest (KDA, latent projections,
    # router, embeddings, shared experts) is a packed broadcast. Refit 2.6 s vs
    # 16 s for the native mcore refit on the same 32 nodes (runs 3674045,
    # 3673498). Runs on NeMo-RL's pure-Python transport (NRL_XFERDTENSOR_PYTHON
    # below): nccl4py's native m2n reshard needs NCCL symmetric memory, which
    # the cross-node communicator over the Slingshot plugin lacks (run 3673870).
    # Fallbacks: refit_transport: mcore (Megatron-Core's native refit, needs the
    # Megatron-LM patch above and mcore_generation_config.refit_backend: nccl,
    # which must be absent otherwise), or null (NeMo-RL's packed collective
    # through the bridge mapping, runs 3653929-3655036).
    refit_transport: nccl_reshard
    mcore_generation_config:
      tensor_model_parallel_size: 1
      pipeline_model_parallel_size: 1
      expert_model_parallel_size: 8
      # Generation CUDA graphs stay at NeMo-RL's defaults (cuda_graph_impl: local,
      # inference_cuda_graph_scope: block); capture works with the KDA + MoE
      # layers (run 3624750). Padding experts is required for EP > 1 with CUDA
      # graphs when the layers are not Megatron-Core's inference_optimized ones.
      moe_pad_experts_for_cuda_graph_inference: true
      # kv_cache_management_mode stays at the default (persist): NeMo-RL requires
      # it to agree with grpo.async_grpo.recompute_kv_cache_after_weight_updates
      # (false), and synchronous GRPO has no request in flight across a refit.
      # The trainer recomputes log-probabilities over the top-p nucleus
      # (generation.top_p is forwarded to it as sampling params), so the
      # generator has to report them after its sampling processors as well.
      # raw_logprobs only gives parity when top_p is 1.0 and top_k is off
      # (run 3653929: raw vs renormalised, a constant offset up to -ln(0.9)).
      logprobs_mode: processed_logprobs

data:
  max_input_seq_length: 900
  shuffle: true
  num_workers: 1
  train:
    dataset_name: gsm8k
    split_validation_size: 0.0
  validation:
    dataset_name: gsm8k
    split: test
  default:
    prompt_file: ${oc.env:RUN_DIR}/gsm8k-five-shot.txt
    system_prompt_file: null
    processor: math_hf_data_processor
    env_name: math

loss_fn:
  # TEMPORARILY 0 (author: 0.01; REFERENCE_KL_PENALTY). Any value > 0 makes
  # NeMo-RL build a second, reference copy of the model on the trainer GPUs at
  # setup and keep its weights on the host; each step swaps them in to compute
  # the reference log-probabilities. At EP16 that copy did not fit (run
  # 3622931); EP32 halves the per-GPU model (32 GB allocated at init). At EP32
  # it fits with the default refit (3655036) but not with the native one (3659594).
  reference_policy_kl_penalty: ${oc.decode:${oc.env:REFERENCE_KL_PENALTY}}
  reference_policy_kl_type: k3
  kl_input_clamp_value: 20.0
  kl_output_clamp_value: 10.0
  ratio_clip_min: 0.2
  ratio_clip_max: 0.2
  ratio_clip_c: null
  use_on_policy_kl_approximation: false

logger:
  log_dir: ${oc.env:RUN_DIR}/nemo-rl-logs
  wandb_enabled: ${oc.decode:${oc.env:WANDB_ENABLED}}
  wandb:
    project: ${oc.env:PROJECT_NAME}
    entity: ${oc.env:WANDB_ENTITY}
    name: ${oc.env:RUN_NAME}
  tensorboard_enabled: false
  num_val_samples_to_print: 8
CHONK_YAML_EOF

# The author's five-shot GSM8K prompt (copy of example/gsm8k-five-shot.txt).
cat > "${RUN_DIR}/gsm8k-five-shot.txt" << 'CHONK_PROMPT_EOF'
Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Answer: Natalia sold 48/2 = 24 clips in May. Natalia sold 48+24 = 72 clips altogether in April and May. #### 72

Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Answer: Weng earns 12/60 = $0.2 per minute. Working 50 minutes, she earned 0.2 x 50 = $10. #### 10

Question: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?
Answer: Betty has only 100/2 = $50. Betty's grandparents gave her 15*2 = $30. This means she needs 100-50-30-15 = $5 more. #### 5

Question: Julie is reading a 120-page book. Yesterday, she read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?
Answer: Julie read 12 x 2 = 24 pages today. So she has read a total of 12 + 24 = 36 pages. There are 120 - 36 = 84 pages left. She should read 84/2 = 42 pages. #### 42

Question: James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?
Answer: He writes 3*2 = 6 pages every time. So he writes 6*2 = 12 pages per week. That means he writes 12*52 = 624 pages a year. #### 624

Question: {}
Answer:
CHONK_PROMPT_EOF

# ── Source checkouts (once, shared) ──────────────────────────────────────────
# Built on one node in a temp dir and renamed into place, so every node of every
# job sees one complete tree. Patches must apply or the job stops.
if [ ! -f "${SRC_DIR}/.alps-ready" ]; then
    srun --mpi=pmix --network=disable_rdzv_get -N 1 --ntasks=1 -u \
        --environment="${TRAINING_CONFIG}/env.toml" \
        --container-writable bash -c '
    set -eo pipefail
    tmp="${SRC_DIR}.tmp.${SLURM_JOB_ID}"
    rm -rf "${tmp}"
    fetch() {  # dir url sha
        mkdir -p "$1"
        git -C "$1" init -q
        git -C "$1" remote add origin "$2"
        git -C "$1" fetch -q --depth 1 origin "$3"
        git -C "$1" checkout -q FETCH_HEAD
    }
    fetch "${tmp}/nemo-rl" "${NEMO_REPO}" "${NEMO_REF}"
    fetch "${tmp}/bridge" "${BRIDGE_REPO}" "${BRIDGE_REF}"
    fetch "${tmp}/megatron-lm" "${MLM_REPO}" "${MLM_REF}"
    fetch "${tmp}/lens" "${LENS_REPO}" "${LENS_REF}"

    # The Apertus 2 bridge package, verbatim from the fork: new files only.
    git -C "${tmp}/bridge" fetch -q --depth 1 "${BRIDGE_APERTUS2_REPO}" "${BRIDGE_APERTUS2_REF}"
    git -C "${tmp}/bridge" checkout -q FETCH_HEAD -- src/megatron/bridge/models/apertus2
    git -C "${tmp}/bridge" reset -q

    git -C "${tmp}/bridge" apply "${RUN_DIR}/patches/bridge-apertus2-shared-minimal.patch" \
        || { echo "FATAL: bridge patch does not apply to ${BRIDGE_REF}"; exit 1; }
    git -C "${tmp}/bridge" apply "${RUN_DIR}/patches/bridge-md-decoupling-optimizer.patch" \
        || { echo "FATAL: bridge optimizer patch does not apply to ${BRIDGE_REF}"; exit 1; }
    git -C "${tmp}/nemo-rl" apply "${RUN_DIR}/patches/nemo-rl-chonk-megatron-gen-minimal.patch" \
        || { echo "FATAL: NeMo-RL patch does not apply to ${NEMO_REF}"; exit 1; }
    git -C "${tmp}/nemo-rl" apply "${RUN_DIR}/patches/nemo-rl-benchmark-metrics.patch" \
        || { echo "FATAL: NeMo-RL benchmark patch does not apply"; exit 1; }
    git -C "${tmp}/megatron-lm" apply "${RUN_DIR}/patches/mlm-refit-execution-batch-bytes.patch" \
        || { echo "FATAL: Megatron-LM patch does not apply to ${MLM_REF}"; exit 1; }

    [ -d "${tmp}/megatron-lm/megatron/core" ] || { echo "FATAL: Megatron-LM checkout is empty"; exit 1; }
    [ -d "${tmp}/lens/src/nemo/lens" ] || { echo "FATAL: nemo-lens checkout is empty"; exit 1; }
    {
        echo "nemo-rl ${NEMO_REF} + nemo-rl-chonk-megatron-gen-minimal.patch + nemo-rl-benchmark-metrics.patch"
        echo "bridge ${BRIDGE_REF} + models/apertus2 from ${BRIDGE_APERTUS2_REF} + bridge-apertus2-shared-minimal.patch + bridge-md-decoupling-optimizer.patch"
        echo "megatron-lm ${MLM_REF} + mlm-refit-execution-batch-bytes.patch"
        echo "lens ${LENS_REF}"
        git -C "${tmp}/nemo-rl" status --short
        git -C "${tmp}/bridge" status --short
        git -C "${tmp}/megatron-lm" status --short
    } > "${tmp}/.alps-ready"
    if mv -T "${tmp}" "${SRC_DIR}" 2>/dev/null; then
        echo "Source checkouts ready at ${SRC_DIR}"
        cat "${SRC_DIR}/.alps-ready"
    else
        echo "Another job created ${SRC_DIR} first, discarding this copy"
        rm -rf "${tmp}"
    fi
    '
    [ -f "${SRC_DIR}/.alps-ready" ] || { echo "FATAL: source checkout at ${SRC_DIR} failed"; exit 1; }
else
    echo "Source checkouts already present at ${SRC_DIR}"
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

cd "${NEMO_RL_PATH}"

# NeMo-RL first: the other checkouts carry their own top-level tools/tests/examples.
export PYTHONPATH="${NEMO_RL_PATH}:${SRC_DIR}/lens/src:${SRC_DIR}/bridge/src:${SRC_DIR}/megatron-lm"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
# Refit staging limits from the vLLM recipe of the image author.
export NRL_REFIT_BUFFER_MEMORY_RATIO=0.005
export NRL_REFIT_NUM_BUFFERS=1
# refit_transport: nccl_reshard on the pure-Python xferdtensor of NeMo-RL (batched
# P2P + replica broadcasts); see the generation block of the recipe.
export NRL_XFERDTENSOR_PYTHON=1
# Benchmark: exact token entropy for the Verl-comparable actor/entropy metric.
export NRL_LOG_TOKEN_ENTROPY=1
export NCCL_DEBUG=WARN
export RAY_DEDUP_LOGS=0

# Compile / JIT / dataset caches and the bridge config lock off Lustre (no flock
# there) onto node-local /tmp.
export HF_HOME=/tmp/hf_${JOB_ID}
export MEGATRON_CONFIG_LOCK_DIR=/tmp/mbridge_locks_${JOB_ID}
export TRITON_CACHE_DIR=/tmp/triton_${JOB_ID}
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_${JOB_ID}
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions_${JOB_ID}
export CUDA_CACHE_PATH=/tmp/cuda_cache_${JOB_ID}
export FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer_${JOB_ID}
# The KDA backward kernels are JIT-compiled by TileLang on the first training
# step; its binary cache defaults to ~/.tilelang/cache (Lustre) and 32 trainer
# ranks racing on it end in a stale file handle (run 3639449).
export TILELANG_CACHE_DIR=/tmp/tilelang_${JOB_ID}

if [ "${RANK}" -eq 0 ]; then
    # Preflight: versions, that the patched vanilla checkouts import against the
    # Megatron-LM fork, that the Apertus 2 bridge is registered, and that the
    # recipe loads -- before Ray starts.
    python3 - <<"PYEOF" || { echo "FATAL: preflight failed on $(hostname)"; exit 1; }
import importlib
import importlib.metadata as md
import os
import traceback
from pathlib import Path

for pkg in ("torch", "transformers", "ray", "transformer_engine", "flashinfer-python", "flash-linear-attention"):
    try:
        print(f"{pkg}: {md.version(pkg)}")
    except Exception as exc:
        print(f"{pkg}: NOT INSTALLED ({exc})")

failed = []
for mod in (
    "nemo.lens",
    "megatron.core",
    "megatron.core.ssm.kimi_delta_attention",
    "megatron.core.inference.engines.dynamic_engine",
    "megatron.core.resharding.refit",
    "emerging_optimizers",
    "megatron.core.optimizer.md_decoupling",
    "megatron.bridge",
    "megatron.bridge.models",
    "megatron.bridge.models.apertus2",
    "nemo_rl",
    "nemo_rl.models.policy.workers.megatron_policy_worker",
    "nemo_rl.models.generation.megatron.megatron_generation",
    "nemo_rl.models.generation.megatron.megatron_worker",
    "nemo_rl.algorithms.grpo",
    "nemo_rl.environments.math_environment",
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

# The bridge must resolve this checkpoint to the Apertus 2 implementation.
from transformers import AutoConfig
from megatron.bridge import AutoBridge

hf_cfg = AutoConfig.from_pretrained(os.environ["HF_MODEL"], trust_remote_code=True)
bridge = AutoBridge.from_hf_config(hf_cfg)
provider = bridge.to_megatron_provider(load_weights=False)
print("architectures:", hf_cfg.architectures, "-> provider", type(provider).__name__)
print(
    "provider:",
    "layers", getattr(provider, "num_layers", None),
    "experts", getattr(provider, "num_moe_experts", None),
    "attention_variant", getattr(provider, "experimental_attention_variant", None),
    "transformer_impl", getattr(provider, "transformer_impl", None),
)

from omegaconf import OmegaConf
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

register_omegaconf_resolvers()
cfg = OmegaConf.to_container(load_config(Path(os.environ["RUN_DIR"]) / "grpo-chonk-megatron-gen.yaml"), resolve=True)
gen = cfg["policy"]["generation"]
print(
    "config OK:",
    "nodes", cfg["cluster"]["num_nodes"],
    "gen_nodes", gen["colocated"]["resources"]["num_nodes"],
    "steps", cfg["grpo"]["max_num_steps"],
    "prompts", cfg["grpo"]["num_prompts_per_step"], "x", cfg["grpo"]["num_generations_per_prompt"],
    "gbs", cfg["policy"]["train_global_batch_size"],
    "gen_backend", gen["backend"],
    "refit", gen["refit_transport"],
    "optimizer", cfg["policy"]["megatron_cfg"]["optimizer"]["optimizer"],
    "kl_penalty", cfg["loss_fn"]["reference_policy_kl_penalty"],
    "wandb", cfg["logger"]["wandb_enabled"],
)
opt = cfg["policy"]["megatron_cfg"]["optimizer"]
sched = cfg["policy"]["megatron_cfg"]["scheduler"]
print(
    "benchmark config:",
    "seed", cfg["grpo"]["seed"],
    "lr", opt["lr"], "matrix_lr", opt["matrix_lr"], "gains_lr", opt["gains_lr"],
    "wd", opt["weight_decay"], sched["start_weight_decay"], sched["end_weight_decay"],
    "schedule", sched["lr_decay_style"], "warmup", sched["lr_warmup_iters"],
    "train T/top_p", gen["temperature"], gen["top_p"],
    "val T/top_p/top_k", gen["val_temperature"], gen["val_top_p"], gen["val_top_k"],
    "max_new_tokens", gen["max_new_tokens"], "max_seq", cfg["policy"]["max_total_sequence_length"],
    "engine_max_len", gen["mcore_generation_config"]["max_model_len"],
    "val", cfg["grpo"]["val_at_start"], cfg["grpo"]["val_period"], cfg["grpo"]["val_at_end"],
    cfg["data"]["validation"],
)
PYEOF
fi

# Ray owns worker placement: drop the Slurm/PMI launcher variables before the
# raylet (and through it every actor) inherits them.
while IFS="=" read -r name _; do
    case "${name}" in PMI*|PMIX*|MPI*|OMPI*|SLURM_*) unset "${name}" ;; esac
done < <(env)

# Host memory on the CPU NUMA nodes only. On GH200 each GPU HBM is a NUMA node
# too, and without a binding the kernel placed ordinary host pages of the
# trainer processes there: ~28 GB of them on one GPU, outside any CUDA
# allocation, and that GPU ran out of memory in training (runs 3659594,
# 3669084, 3669588). The raylet, and through it every worker, inherits the
# policy. NUMA_BIND_HOST_MEMORY=0 turns it off.
NUMA_BIND=""
if [ "${NUMA_BIND_HOST_MEMORY:-1}" = "1" ]; then
    command -v numactl > /dev/null || { echo "FATAL: numactl not found"; exit 1; }
    CPU_MEM_NODES=$(for d in /sys/devices/system/node/node[0-9]*; do
        [ -n "$(cat "${d}/cpulist")" ] && echo "${d##*/node}"; done | paste -sd, -)
    [ -n "${CPU_MEM_NODES}" ] || { echo "FATAL: no NUMA node with CPUs found"; exit 1; }
    NUMA_BIND="numactl --membind=${CPU_MEM_NODES}"
    echo "host memory bound to NUMA nodes ${CPU_MEM_NODES} on $(hostname)"
fi

if [ "${RANK}" -eq 0 ]; then
    ${NUMA_BIND} ray start --head \
        --node-ip-address=${MASTER_NODE_IP} \
        --port=${PORT} \
        --num-cpus=${NUM_CPUS} \
        --num-gpus=4 \
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

    ${NUMA_BIND} python3 examples/run_grpo.py \
        --config "${RUN_DIR}/grpo-chonk-megatron-gen.yaml" 2>&1 | tee "${RUN_DIR}/run.log"
    rc=${PIPESTATUS[0]}

    ray stop --grace-period 30 || true
    exit "${rc}"
else
    # Worker nodes join once the head is up (it runs the preflight first).
    while [ ! -f "${RUN_DIR}/ray_head_ready" ]; do
        sleep 2
    done
    sleep 5
    ${NUMA_BIND} ray start \
        --address="${RAY_ADDRESS}" \
        --node-ip-address=${NODE_IP} \
        --num-cpus=${NUM_CPUS} \
        --num-gpus=4 \
        --disable-usage-stats \
        --block || true
fi


'
