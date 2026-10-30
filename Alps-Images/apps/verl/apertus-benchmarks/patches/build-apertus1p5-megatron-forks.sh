#!/bin/bash
#
# build-apertus1p5-megatron-forks.sh
# ─────────────────────────────────────────────────────────────────────────────
# One-shot: push an `apertus1p5-support` branch to each theely/* fork carrying
# the minimal Apertus-1.5 support that the recipe
# rl-bench-apertus-v1.5-70B-sglang-megatron-v1-separate-async.sh then applies at
# runtime as a GitHub `compare` diff onto the image's STOCK megatron packages.
#
# NO fork wheels, no wqwqazwsxedc, no runtime clone/wheel-build. Each branch is
# a SINGLE commit off the exact UPSTREAM commit the image's package resolves to,
# so `<tag>...<branch>.diff` (merge-base semantics) is exactly the delta:
#
#   theely/Megatron-LM      base: NVIDIA/Megatron-LM       core_v0.19.0 (5be9626709af)
#                           delta: patches/mcore-apertus1p5.patch  (2 hunks)
#                             - transformer/mlp.py:  instantiate + call a module
#                               activation_func (MCoreXIELU) even when
#                               use_te_activation_func is False
#                             - distributed/finalize_model_grads.py:  sum xIELU
#                               alpha_p/alpha_n grads across the TP domain
#
#   theely/Megatron-Bridge  base: NVIDIA-NeMo/Megatron-Bridge  v0.6.0 (51885cf132b2)
#                           (no v0.6.1 tag exists; 0.6.0->0.6.1 is a patch release
#                            with no models/ API change -- the recipe applies the
#                            diff with `patch --fuzz=3` to absorb it)
#                           delta:
#                             - models/apertus/{__init__,apertus_bridge}.py
#                               (plain ApertusForCausalLM bridge + MCoreXIELU +
#                                get_apertus_decoder_block_spec -- the helpers
#                                apertus1p5 reuses; upstream 0.6.x has no apertus
#                                support at all)
#                             - models/apertus1p5/{__init__,apertus1p5_bridge}.py
#                               (Apertus1p5Bridge, from patches/apertus1p5_bridge.py)
#                             - models/__init__.py: import + __all__ (the
#                               @register_bridge decorator only fires on import)
#                             - models/hf_pretrained/safe_config_loader.py:
#                               filelock -> contextlib.nullcontext (fcntl.flock
#                               is unsupported in-container on CSCS Lustre)
#                           (NO qwen3_asr fix -- v0.6.0 already guards it with
#                            `if Qwen3ASRConfig.model_type in CONFIG_MAPPING`.)
#
# The actual apertus code is identical to what runs 3240861/3243271/3243467
# already built and ran end-to-end against stock megatron-core 0.19.0 /
# megatron-bridge 0.6.1 (they trained degenerately only because of the reward
# function, since fixed -- see CLAUDE.md "The zero-reward chain").
#
# Requires: git + push access to github.com/theely/Megatron-{LM,Bridge}.
#           gh (optional) -- if present, also opens a review PR against each
#           upstream main and prints its URL.
# Run from anywhere; needs this repo checked out (reads the patches/ dir).
#
# Usage:  ./build-apertus1p5-megatron-forks.sh [workdir]
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

WORKDIR="${1:-$(mktemp -d)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MCORE_PATCH="${SCRIPT_DIR}/mcore-apertus1p5.patch"
APERTUS_BRIDGE_SRC="${SCRIPT_DIR}/apertus_bridge.py"
APERTUS1P5_BRIDGE_SRC="${SCRIPT_DIR}/apertus1p5_bridge.py"

MLM_UPSTREAM="https://github.com/NVIDIA/Megatron-LM.git"
MLM_BASE_TAG="core_v0.19.0"
BRIDGE_UPSTREAM="https://github.com/NVIDIA-NeMo/Megatron-Bridge.git"
BRIDGE_BASE_TAG="v0.6.0"

THEELY_MLM="https://github.com/theely/Megatron-LM.git"
THEELY_BRIDGE="https://github.com/theely/Megatron-Bridge.git"

BRANCH="apertus1p5-support"

for f in "${MCORE_PATCH}" "${APERTUS_BRIDGE_SRC}" "${APERTUS1P5_BRIDGE_SRC}"; do
    [ -f "$f" ] || { echo "FATAL: $f not found"; exit 1; }
done
python3 -m py_compile "${APERTUS_BRIDGE_SRC}" "${APERTUS1P5_BRIDGE_SRC}"

HAVE_GH=0
command -v gh >/dev/null 2>&1 && HAVE_GH=1

mkdir -p "${WORKDIR}"; cd "${WORKDIR}"
echo ">> workdir: ${WORKDIR}"

_gitcfg() {
    git config user.name  "$(git config user.name  || echo apertus1p5-bot)"
    git config user.email "$(git config user.email || echo apertus1p5@localhost)"
}

# ── theely/Megatron-LM ──────────────────────────────────────────────────────
echo; echo "════════ theely/Megatron-LM ════════"
rm -rf Megatron-LM
git clone --filter=blob:none --no-tags "${MLM_UPSTREAM}" Megatron-LM
cd Megatron-LM
git fetch --no-tags origin "refs/tags/${MLM_BASE_TAG}:refs/mlmbase"
git remote set-url origin "${THEELY_MLM}"
_gitcfg

git checkout -B "${BRANCH}" refs/mlmbase
patch -p1 < "${MCORE_PATCH}"
python3 -m py_compile megatron/core/transformer/mlp.py megatron/core/distributed/finalize_model_grads.py
git add -A
git commit -q -m "apertus: support a module activation_func + TP-summed xIELU grads

Apertus 1.5 uses xIELU, whose learnable alpha_p/alpha_n/beta/eps live in a
module activation_func (MCoreXIELU) rather than a fused TE activation.

- transformer/mlp.py: instantiate and call submodules.activation_func when it
  is a module, regardless of config.use_te_activation_func (which the Apertus
  provider sets False).
- distributed/finalize_model_grads.py: sum-reduce grads of params flagged
  .sum_gradients_across_tp_domain (the replicated xIELU alpha scalars) across
  the TP domain, alongside the existing sequence-parallel / qk_layernorm case."
git push -f origin "${BRANCH}"
MLM_HEAD_SHA="$(git rev-parse HEAD)"
MLM_PR_URL=""
if [ "${HAVE_GH}" = "1" ]; then
    MLM_PR_URL="$(gh pr create --repo NVIDIA/Megatron-LM \
        --base main --head "theely:${BRANCH}" \
        --title "Apertus: module activation_func + TP-summed xIELU grads" \
        --body "Minimal megatron-core delta for Apertus (xIELU) support, cut from \`${MLM_BASE_TAG}\`. Also consumed out-of-tree as \`compare/NVIDIA:${MLM_BASE_TAG}...theely:${BRANCH}.diff\`." 2>/dev/null || true)"
fi
cd ..

# ── theely/Megatron-Bridge ─────────────────────────────────────────────────
echo; echo "════════ theely/Megatron-Bridge ════════"
rm -rf Megatron-Bridge
git clone --filter=blob:none --no-tags "${BRIDGE_UPSTREAM}" Megatron-Bridge
cd Megatron-Bridge
git fetch --no-tags origin "refs/tags/${BRIDGE_BASE_TAG}:refs/bridgebase"
git remote set-url origin "${THEELY_BRIDGE}"
_gitcfg

git checkout -B "${BRANCH}" refs/bridgebase

M="src/megatron/bridge/models"
mkdir -p "${M}/apertus" "${M}/apertus1p5"

cat > "${M}/apertus/__init__.py" <<'EOF'
from megatron.bridge.models.apertus.apertus_bridge import ApertusBridge

__all__ = ["ApertusBridge"]
EOF
cp "${APERTUS_BRIDGE_SRC}" "${M}/apertus/apertus_bridge.py"

cat > "${M}/apertus1p5/__init__.py" <<'EOF'
from megatron.bridge.models.apertus1p5.apertus1p5_bridge import Apertus1p5Bridge

__all__ = ["Apertus1p5Bridge"]
EOF
cp "${APERTUS1P5_BRIDGE_SRC}" "${M}/apertus1p5/apertus1p5_bridge.py"

python3 - "${M}/__init__.py" <<'EOF'
import sys
p = sys.argv[1]
s = open(p).read()

anchor_imp = "from megatron.bridge.models.bailing import ("
add_imp = (
    "from megatron.bridge.models.apertus import ApertusBridge\n"
    "from megatron.bridge.models.apertus1p5 import Apertus1p5Bridge\n"
)
assert anchor_imp in s, "bailing import anchor not found in models/__init__.py"
if "from megatron.bridge.models.apertus import" not in s:
    s = s.replace(anchor_imp, add_imp + anchor_imp, 1)

anchor_all = '    "BailingMoeV2Bridge",'
add_all = '    "ApertusBridge",\n    "Apertus1p5Bridge",\n'
assert anchor_all in s, "BailingMoeV2Bridge __all__ anchor not found"
if '"Apertus1p5Bridge"' not in s:
    s = s.replace(anchor_all, add_all + anchor_all, 1)

open(p, "w").write(s)
print("  patched models/__init__.py")
EOF

python3 - "${M}/hf_pretrained/safe_config_loader.py" <<'EOF'
import re, sys
p = sys.argv[1]
s = open(p).read()
if "import contextlib" not in s:
    s = s.replace("import time\n", "import time\nimport contextlib\n", 1)
s2 = re.sub(
    r"with filelock\.FileLock\([^\n]*\):",
    "with contextlib.nullcontext():  # filelock disabled: config is written once "
    "by rank 0, read-only after; fcntl.flock is unsupported in-container on CSCS Lustre",
    s,
)
assert s2 != s, "filelock line not found in safe_config_loader.py"
open(p, "w").write(s2)
print("  patched safe_config_loader.py")
EOF

python3 -m py_compile \
    "${M}/apertus/apertus_bridge.py" "${M}/apertus1p5/apertus1p5_bridge.py" \
    "${M}/__init__.py" "${M}/hf_pretrained/safe_config_loader.py"

git add -A
git commit -q -m "models: add Apertus / Apertus 1.5 bridges

Upstream 0.6.x has no Apertus support. This adds:

- models/apertus/: ApertusBridge for ApertusForCausalLM, plus MCoreXIELU
  (a Megatron-side wrapper of transformers' XIELUActivation that requires the
  CUDA xielu kernel and TP-sums its alpha grads) and get_apertus_decoder_block_spec
  (RMSNorm q/k-layernorm + MCoreXIELU MLP activation).
- models/apertus1p5/: Apertus1p5Bridge for the multimodal
  Apertus1p5ForConditionalGeneration -- hyperparameters read from
  config.text_config, HF weight keys under model.language_model.*, pruned-LM-head
  handling (output_vocab_size < vocab_size), vision/audio tokenizer towers left
  unmapped (target is a text-only GPTModel).
- models/__init__.py: import both (the @register_bridge decorator only fires on
  import) and add to __all__.
- hf_pretrained/safe_config_loader.py: replace the filelock around
  AutoConfig.from_pretrained with contextlib.nullcontext -- fcntl.flock raises
  ENOLCK/ESTALE in-container on CSCS Lustre and the config is written once by
  rank 0 and read-only afterward."
git push -f origin "${BRANCH}"
BRIDGE_HEAD_SHA="$(git rev-parse HEAD)"
BRIDGE_PR_URL=""
if [ "${HAVE_GH}" = "1" ]; then
    BRIDGE_PR_URL="$(gh pr create --repo NVIDIA-NeMo/Megatron-Bridge \
        --base main --head "theely:${BRANCH}" \
        --title "models: add Apertus / Apertus 1.5 bridges" \
        --body "Adds ApertusBridge + Apertus1p5Bridge + safe_config_loader flock fix, cut from \`${BRIDGE_BASE_TAG}\`. Also consumed out-of-tree as \`compare/NVIDIA-NeMo:${BRIDGE_BASE_TAG}...theely:${BRANCH}.diff\`." 2>/dev/null || true)"
fi
cd ..

# ── done ───────────────────────────────────────────────────────────────────
echo
echo "═══════════════════════════════════════════════════════════════════════"
echo "Pushed theely/Megatron-{LM,Bridge}:${BRANCH}"
echo
echo "The recipe (Group 2) is already wired to:"
echo "  https://github.com/theely/Megatron-LM/compare/NVIDIA:${MLM_BASE_TAG}...theely:${BRANCH}.diff"
echo "  https://github.com/theely/Megatron-Bridge/compare/NVIDIA-NeMo:${BRIDGE_BASE_TAG}...theely:${BRANCH}.diff"
echo
echo "For a reproducible pin (survives a force-push), set in Group 2 instead:"
echo "  export MEGATRON_LM_FORK_REF=${MLM_HEAD_SHA}"
echo "  export MEGATRON_BRIDGE_FORK_REF=${BRIDGE_HEAD_SHA}"
[ -n "${MLM_PR_URL}" ]    && echo && echo "Megatron-LM review PR:     ${MLM_PR_URL}"
[ -n "${BRIDGE_PR_URL}" ] && echo "Megatron-Bridge review PR: ${BRIDGE_PR_URL}"
echo "═══════════════════════════════════════════════════════════════════════"
