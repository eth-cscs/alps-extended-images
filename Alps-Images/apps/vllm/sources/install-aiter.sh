#!/usr/bin/env bash
# Install AITER (AMD's kernel library for LLM inference on ROCm) into the ROCm vLLM image.
# vLLM on ROCm needs it (or flash-attn) for MLA prefill.
#
# AITER is installed editable from /opt/aiter: its JIT compiles kernels from those sources
# with hipcc on first use and caches the resulting modules in AITER_JIT_DIR. This script
# imports AITER and resolves vLLM's AITER MLA prefill backend once, so the modules those
# steps need are compiled into AITER_JIT_DIR at image build time.
#
# Environment overrides:
#   AITER_REPO      git remote (default: upstream GitHub)
#   AITER_REF       tag or branch to clone (default: v0.1.24).
#   AITER_JIT_DIR   JIT module cache baked into the image (default: /opt/aiter-jit).
#   GPU_ARCHS       gfx targets for the JIT (default: PYTORCH_ROCM_ARCH, else gfx942).

set -euo pipefail

AITER_REPO="${AITER_REPO:-https://github.com/ROCm/aiter.git}"
AITER_REF="${AITER_REF:-v0.1.24}"
export AITER_JIT_DIR="${AITER_JIT_DIR:-/opt/aiter-jit}"
export GPU_ARCHS="${GPU_ARCHS:-${PYTORCH_ROCM_ARCH:-gfx942}}"

# shellcheck source=/dev/null
source /opt/alps/package-helpers.sh

src_dir=/opt/aiter
rm -rf "${src_dir}"
git clone -q --depth 1 --branch "${AITER_REF}" "${AITER_REPO}" "${src_dir}"
git -C "${src_dir}" submodule update --init --recursive --depth 1

# --no-build-isolation: build against the image's ROCm torch.
cd "${src_dir}"
pip_install python --no-cache-dir --no-build-isolation -e .

# Import compiles AITER's core module into the image (GPU_ARCHS pinned above).
mkdir -p "${AITER_JIT_DIR}"
python -c "import aiter"
python -c "
from vllm.v1.attention.backends.mla.prefill.registry import MLAPrefillBackendEnum as B
B.ROCM_AITER_FA.get_class()
"

shopt -s nullglob
baked_modules=("${AITER_JIT_DIR}"/*.so)
shopt -u nullglob
[[ ${#baked_modules[@]} -gt 0 ]] \
    || { echo "ERROR: no AITER modules in ${AITER_JIT_DIR}" >&2; exit 1; }
echo "OK: AITER ${AITER_REF} installed; baked modules: ${baked_modules[*]##*/}"
