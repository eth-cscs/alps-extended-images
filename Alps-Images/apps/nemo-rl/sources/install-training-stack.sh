#!/usr/bin/env bash
# Assemble the release sources at image build time, before installing them.
set -euo pipefail
# Source revisions used by the validated GSM8K examples.
NEMO_RL_SOURCE=https://github.com/swiss-ai/Nemo-RL.git
NEMO_REF=f196ef0a1ccb54d6ce71cd63277fb6a0e01168db

NEMO_PR5_REF=fe7e3fe71c80c1630487c09a7d49a27e27a3b71e
NEMO_PR1_REF=40a766b21dcfc65c9548fd0cc1ff02869c585869
NEMO_PR6_REF=2bd5dc71c955d9d7bdb12a9d3849529a4ad0d292
NEMO_PR7_REF=e7784d7c052b9d13960dafa15694355b4edf660c
NEMO_PR9_REF=1abfe7fff686168b0cd97161e8ebe064f67a4c05
NEMO_VLLM_DP_REF=05db6ab36b7a0c528c178cbeb44903dbe82fda5d # Keep the tested Nemo RL VLLM DP #2517 revision.
NEMO_FP8_REF=19843e5a523254ebf3bed49a47a1e852dca29a70

MEGATRON_SOURCE=https://github.com/swiss-ai/Megatron-LM-MoE.git
MCORE_REF=0536034901962160aac9742d134a84c55e28d224

BRIDGE_SOURCE=https://github.com/swiss-ai/Megatron-Bridge.git
BRIDGE_REF=077224531ad4d6d0caef767a84e619b256aa5cb6
BRIDGE_PR5_REF=6ccdb3c369491374f1fef4320886e8bb1394bb3e

NEMO_GYM_SOURCE=https://github.com/swiss-ai/nemo-gym.git
NEMO_GYM_REF=93f232f02074deda2fec13a6848f22e5dca741f5

# --------
root="${TRAINING_STACK_ROOT:-/opt}"
metadata="${TRAINING_STACK_METADATA_DIR:-/opt/alps}"
patches="${TRAINING_STACK_PATCH_DIR:-/opt/alps/training-stack-patches}"
mkdir -p "$root" "$metadata"
export GIT_AUTHOR_NAME='Alps image' GIT_COMMITTER_NAME='Alps image'
export GIT_AUTHOR_EMAIL=alps@example.invalid GIT_COMMITTER_EMAIL=alps@example.invalid
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=commit.gpgsign GIT_CONFIG_VALUE_0=false

nemo="$root/nemo-rl"
git clone --no-checkout "$NEMO_RL_SOURCE" "$nemo"
git -C "$nemo" checkout --detach "$NEMO_REF"
for ref in "$NEMO_PR5_REF" "$NEMO_PR1_REF" "$NEMO_PR9_REF"; do
    git -C "$nemo" fetch "$NEMO_RL_SOURCE" "$ref"
    git -C "$nemo" merge --no-edit "$ref"
done
git -C "$nemo" fetch https://github.com/NVIDIA-NeMo/RL.git "$NEMO_VLLM_DP_REF"
git -C "$nemo" merge --no-commit --no-ff -Xtheirs FETCH_HEAD
# Retain the fork's nightly-test budget, as in the tested runtime composition.
git -C "$nemo" restore --source=HEAD -- tests/unit/test_recipes_and_test_suites.py
git -C "$nemo" add tests/unit/test_recipes_and_test_suites.py
git -C "$nemo" commit -m 'Merge pinned upstream vLLM DP PR 2517'
git -C "$nemo" fetch "$NEMO_RL_SOURCE" "$NEMO_FP8_REF"
git -C "$nemo" merge --no-edit "$NEMO_FP8_REF"
for ref in "$NEMO_PR6_REF" "$NEMO_PR7_REF"; do
    git -C "$nemo" fetch "$NEMO_RL_SOURCE" "$ref"
    git -C "$nemo" merge --no-edit "$ref"
done

core="$root/megatron-lm"
git clone --no-checkout "$MEGATRON_SOURCE" "$core"
git -C "$core" fetch "$MEGATRON_SOURCE" "$MCORE_REF"
git -C "$core" checkout --detach "$MCORE_REF"

bridge="$root/megatron-bridge"
git clone --no-checkout "$BRIDGE_SOURCE" "$bridge"
git -C "$bridge" checkout --detach "$BRIDGE_REF"
git -C "$bridge" fetch "$BRIDGE_SOURCE" "$BRIDGE_PR5_REF"
git -C "$bridge" merge --ff-only "$BRIDGE_PR5_REF"

gym="$root/nemo-gym"
git clone --no-checkout "$NEMO_GYM_SOURCE" "$gym"
git -C "$gym" checkout --detach "$NEMO_GYM_REF"

: > "$metadata/applied-patches.txt"
for family in nemo-rl megatron; do
    target="$nemo"
    [[ "$family" != megatron ]] || target="$core"
    for patch in "$patches/$family/"*.patch; do
        git -C "$target" apply --index "$patch"
        sha256sum "$patch" >> "$metadata/applied-patches.txt"
    done
done
{
    # Include the pins and patched index trees, not timestamp-dependent merge IDs.
    grep -E '^(NEMO|MEGATRON|MCORE|BRIDGE)_[A-Z0-9_]+=' "${BASH_SOURCE[0]}"
    for target in "$nemo" "$core" "$bridge"; do
        printf '%s tree ' "${target##*/}"
        git -C "$target" write-tree
    done
    cat "$metadata/applied-patches.txt"
} > "$metadata/source-provenance.txt"
