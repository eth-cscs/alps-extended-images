#!/bin/bash

#SBATCH --nodes=24
#SBATCH --account=csstaff
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=2:00:00

# ─────────────────────────────────────────────────────────────────────────────
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
#                    + models/apertus2/ copied verbatim from
#                      swiss-ai/Megatron-Bridge@f3bc8e7e (6 new files)
#   Megatron-LM      swiss-ai/Megatron-LM-MoE PR #88 @b0844260  the fork, untouched (KDA
#                    layers, latent MoE, quantile-balancing router are not in
#                    stock Megatron-LM)
#   nemo-lens        NVIDIA-NeMo/Lens@b85578fc          the rev NeMo-RL pins
# The two patches are embedded below and checked in under patches/ (keep them
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
# offloading (model_overrides block dropped -> more trainer GPU memory),
# TEMPORARILY Adam instead of the md_decoupling optimizer Chonk was pretrained
# with (TODO: change back, see the optimizer block of the recipe), and
# generation by Megatron-Core's dynamic engine (CUDA graphs on, NeMo-RL
# defaults), refit by NeMo-RL's default collective through the bridge mapping.
#
# BEFORE THE FINAL RUN, restore: the md_decoupling optimizer. It is disabled only
# to bring the loop up. The KL penalty against the starting policy is back at
# the author's 0.01 (REFERENCE_KL_PENALTY). ALSO WANTED BACK: the native
# Megatron-to-Megatron refit (refit_transport: mcore), see the generation block
# of the recipe.
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
export SRC_VERSION="v3"

# Topology and batch sizes of the author's recipe, except trainer EP32 instead
# of EP16 (memory, see the recipe below): 8 training nodes (32 GPUs,
# EP32, 8 experts per GPU) + 16 generation nodes (64 GPUs, EP8 -> 8 replicas).
export TRAIN_NODES=8
export INFERENCE_NODES=$(( SLURM_JOB_NUM_NODES - TRAIN_NODES ))
export TOTAL_NODES=${SLURM_JOB_NUM_NODES}
# Micro-batch 1 with the global batch kept at 128 (4 accumulated micro-batches
# per trainer rank): at micro-batch 4 the trainer runs out of GPU memory in the
# loss (vocabulary-wide logits for 4 x 1280 tokens on top of ~49 GB of model and
# optimizer state without expert offloading, run 3628358).
export TRAIN_MICRO_BATCH="${TRAIN_MICRO_BATCH:-1}"
export INFERENCE_REQUESTS="${INFERENCE_REQUESTS:-32}"
export MAX_NUM_STEPS="${MAX_NUM_STEPS:-2}"
export NUM_PROMPTS="${NUM_PROMPTS:-32}"
# KL penalty against the starting policy (author's value). > 0 makes NeMo-RL
# build a reference model on the trainer GPUs at setup (see the loss_fn block
# of the recipe); 0.0 skips it.
export REFERENCE_KL_PENALTY="${REFERENCE_KL_PENALTY:-0.01}"
export TRAIN_GLOBAL_BATCH="${TRAIN_GLOBAL_BATCH:-$(( TRAIN_NODES * 4 * 4 ))}"
if [ "${INFERENCE_NODES}" -le 0 ] || [ $(( INFERENCE_NODES * 4 % 8 )) -ne 0 ]; then
    echo "FATAL: ${INFERENCE_NODES} generation nodes x 4 GPUs is not a positive multiple of EP8"
    exit 1
fi
if [ $(( NUM_PROMPTS * 8 % TRAIN_GLOBAL_BATCH )) -ne 0 ] \
        || [ $(( TRAIN_GLOBAL_BATCH % (TRAIN_NODES * 4 * TRAIN_MICRO_BATCH) )) -ne 0 ]; then
    echo "FATAL: rollout batch ${NUM_PROMPTS} x 8 / global batch ${TRAIN_GLOBAL_BATCH} / micro-batch ${TRAIN_MICRO_BATCH} x $(( TRAIN_NODES * 4 )) ranks do not divide"
    exit 1
fi

export EXPERIMENT_NAME="${MODEL_NAME}-gsm8k-nemorl-megatron-gen-${SLURM_JOB_NUM_NODES}n"
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
  num_generations_per_prompt: 8
  normalize_rewards: true
  use_leave_one_out_baseline: true
  advantage_clip_low: -5.0
  advantage_clip_high: 5.0
  val_period: 0
  val_at_start: false
  val_at_end: false

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
  max_total_sequence_length: 1280
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
    # TEMPORARY: Adam. Chonk was pretrained with the fork's md_decoupling optimizer
    # (Muon-style orthogonal updates, hypersphere-constrained weights with learned
    # gains); Adam does not maintain those constraints. TODO: switch back to
    # md_decoupling -- restore the optimizer block of the author's
    # grpo-chonk-vllm.yaml and apply patches/bridge-md-decoupling-optimizer.patch.
    # Learning rates and betas below are the author's; everything not set here is
    # NeMo-RL's stock Adam config (distributed, precision-aware optimizer).
    optimizer:
      optimizer: adam
      lr: 2.0e-6
      min_lr: 2.0e-7
      weight_decay: 0.0
      adam_beta1: 0.9
      adam_beta2: 0.95
      adam_eps: 1.0e-8
      # Adam moments in bf16 instead of the fp32 default, to halve the optimizer
      # state that is allocated at the first optimizer step (same memory budget
      # as above). Goes away with the Adam block when md_decoupling is restored.
      exp_avg_dtype: bf16
      exp_avg_sq_dtype: bf16
      clip_grad: ${policy.max_grad_norm}
    scheduler:
      lr_decay_style: cosine
      lr_decay_iters: 1200
      lr_warmup_iters: 60
      lr_warmup_init: 1.0e-7
      start_weight_decay: 0.0
      end_weight_decay: 0.0
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
    top_p: 0.9
    top_k: -1
    max_new_tokens: 256
    stop_strings: ["Question:", "</s>", "<|im_end|>"]
    colocated:
      enabled: false
      resources:
        num_nodes: ${oc.decode:${oc.env:INFERENCE_NODES}}
        gpus_per_node: 4
    # NeMo-RL's own refit (the default): the trainer streams weights through the
    # bridge mapping over an NCCL collective and the generator maps them back.
    # Same trainer-side path as the author's vLLM recipe. The native Megatron
    # refit (refit_transport: mcore) is not usable: vanilla NeMo-RL passes
    # execution_batch_bytes to megatron.core.resharding.refit, which the
    # Megatron-LM fork's version does not accept (run 3623177).
    # TODO: RE-ENABLE THE NATIVE PATH (refit_transport: mcore + refit_backend: nccl
    # under mcore_generation_config) once NeMo-RL and the Megatron-LM fork agree on
    # that API; it avoids the conversion through the HF layout on every refit.
    refit_transport: null
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
      # (generation.top_p: 0.9 is forwarded to it as sampling params), so the
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
  validation: null
  default:
    prompt_file: ${oc.env:RUN_DIR}/gsm8k-five-shot.txt
    system_prompt_file: null
    processor: math_hf_data_processor
    env_name: math

loss_fn:
  # 0.01 as in the author's recipe (REFERENCE_KL_PENALTY). Any value > 0 makes
  # NeMo-RL build a second, reference copy of the model on the trainer GPUs at
  # setup and keep its weights on the host; each step swaps them in to compute
  # the reference log-probabilities. At EP16 that copy did not fit (run
  # 3622931); EP32 halves the per-GPU model (32 GB allocated at init).
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
  wandb_enabled: false
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
    git -C "${tmp}/nemo-rl" apply "${RUN_DIR}/patches/nemo-rl-chonk-megatron-gen-minimal.patch" \
        || { echo "FATAL: NeMo-RL patch does not apply to ${NEMO_REF}"; exit 1; }

    [ -d "${tmp}/megatron-lm/megatron/core" ] || { echo "FATAL: Megatron-LM checkout is empty"; exit 1; }
    [ -d "${tmp}/lens/src/nemo/lens" ] || { echo "FATAL: nemo-lens checkout is empty"; exit 1; }
    {
        echo "nemo-rl ${NEMO_REF} + nemo-rl-chonk-megatron-gen-minimal.patch"
        echo "bridge ${BRIDGE_REF} + models/apertus2 from ${BRIDGE_APERTUS2_REF} + bridge-apertus2-shared-minimal.patch"
        echo "megatron-lm ${MLM_REF}"
        echo "lens ${LENS_REF}"
        git -C "${tmp}/nemo-rl" status --short
        git -C "${tmp}/bridge" status --short
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

    python3 examples/run_grpo.py \
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
    ray start \
        --address="${RAY_ADDRESS}" \
        --node-ip-address=${NODE_IP} \
        --num-cpus=${NUM_CPUS} \
        --num-gpus=4 \
        --disable-usage-stats \
        --block || true
fi


'
