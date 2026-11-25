#!/usr/bin/env bash
# Assemble the NeMo-RL, Megatron-Core and Megatron-Bridge sources at image build time.
# Upstream NeMo-RL main with the Megatron-Bridge and Megatron-Core revisions it pins,
# plus the Apertus 1.0/1.5 bridges (NVIDIA-NeMo/Megatron-Bridge#7041) as a patch.
set -euo pipefail

# NVIDIA-NeMo/RL main (2026-11-20). Its pyproject pins vllm==0.33.0, and its
# 3rdparty/Megatron-Bridge-workspace submodule points at BRIDGE_REF below.
NEMO_RL_SOURCE=https://github.com/NVIDIA-NeMo/RL.git
NEMO_REF=83151e70d2814dad321b53128a685dd199300d16

# The Megatron-LM revision of Megatron-Bridge's 3rdparty/Megatron-LM submodule at BRIDGE_REF.
# The Apertus bridges carry their own xIELU MLP, so no Megatron-Core patch is needed
# (NVIDIA/Megatron-LM#9633 would make that MLP subclass unnecessary).
MEGATRON_SOURCE=https://github.com/NVIDIA/Megatron-LM.git
MCORE_REF=e0a4bfed7f1e233b857187ceff0104645858d7d1

# NeMo-RL's Megatron-Bridge submodule revision. patches/megatron-bridge carries #7041 rebased onto it.
BRIDGE_SOURCE=https://github.com/NVIDIA-NeMo/Megatron-Bridge.git
BRIDGE_REF=ae0a1b026f52517c55cf19416510f7f895a24c0c

# --------
root="${TRAINING_STACK_ROOT:-/opt}"
metadata="${TRAINING_STACK_METADATA_DIR:-/opt/alps}"
patches="${TRAINING_STACK_PATCH_DIR:-/opt/alps/training-stack-patches}"
mkdir -p "$root" "$metadata"

# usage: checkout_pinned SOURCE REF DEST
checkout_pinned() {
    git init -q "$3"
    git -C "$3" remote add origin "$1"
    git -C "$3" fetch -q --depth 1 origin "$2"
    git -C "$3" checkout -q --detach FETCH_HEAD
    test "$(git -C "$3" rev-parse HEAD)" = "$2"
}

nemo="$root/nemo-rl"
core="$root/megatron-lm"
bridge="$root/megatron-bridge"
checkout_pinned "$NEMO_RL_SOURCE" "$NEMO_REF" "$nemo"
checkout_pinned "$MEGATRON_SOURCE" "$MCORE_REF" "$core"
checkout_pinned "$BRIDGE_SOURCE" "$BRIDGE_REF" "$bridge"

: > "$metadata/applied-patches.txt"
for patch in "$patches/megatron-bridge/"*.patch; do
    git -C "$bridge" apply --index "$patch"
    sha256sum "$patch" >> "$metadata/applied-patches.txt"
done
{
    # Include the pins and patched index trees, not timestamp-dependent commit IDs.
    grep -E '^(NEMO|MEGATRON|MCORE|BRIDGE)_[A-Z0-9_]+=' "${BASH_SOURCE[0]}"
    for target in "$nemo" "$core" "$bridge"; do
        printf '%s tree ' "${target##*/}"
        git -C "$target" write-tree
    done
    cat "$metadata/applied-patches.txt"
} > "$metadata/source-provenance.txt"
