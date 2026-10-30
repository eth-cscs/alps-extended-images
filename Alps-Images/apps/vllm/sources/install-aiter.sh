#!/usr/bin/env bash
# Build and install AITER for the ROCm vLLM image.
#
# The caller must provide the image's pinned ref/commit and invoke this script
# while the ROCm compiler environment, build dependencies, and PIP_CONSTRAINT
# are active. The installed wheel carries AITER's JIT sources; /opt/aiter is
# removed after installation. Full multi-architecture AOT remains disabled.
#
# Inputs:
#   AITER_REPO          git remote (default: upstream GitHub)
#   AITER_REF           tag or branch to clone (required)
#   AITER_COMMIT        exact commit expected for AITER_REF (required)
#   GPU_ARCHS           semicolon-separated gfx targets (required)
#   CU_NUM              target compute units (default: 228 for MI300A)
#   MAX_JOBS            maximum parallel build jobs (default: 32)
#   AITER_JIT_DIR       writable preseeded runtime cache (default: /opt/aiter-jit)
#   AITER_PATCH_DIR     patches applied in name order (default: /opt/alps/aiter-patches)
#   ALPS_PACKAGE_HELPERS package helper path (default: /opt/alps/package-helpers.sh)

set -euo pipefail

AITER_REPO="${AITER_REPO:-https://github.com/ROCm/aiter.git}"
AITER_REF="${AITER_REF:?AITER_REF is required}"
AITER_COMMIT="${AITER_COMMIT:?AITER_COMMIT is required}"
GPU_ARCHS="${GPU_ARCHS:?GPU_ARCHS is required}"
CU_NUM="${CU_NUM-228}"
MAX_JOBS="${MAX_JOBS:-32}"
AITER_JIT_DIR="${AITER_JIT_DIR:-/opt/aiter-jit}"
AITER_PATCH_DIR="${AITER_PATCH_DIR:-/opt/alps/aiter-patches}"
ALPS_PACKAGE_HELPERS="${ALPS_PACKAGE_HELPERS:-/opt/alps/package-helpers.sh}"
PIP_CONSTRAINT="${PIP_CONSTRAINT:?PIP_CONSTRAINT is required}"

[[ -f "${ALPS_PACKAGE_HELPERS}" ]] \
    || { echo "ERROR: package helpers not found: ${ALPS_PACKAGE_HELPERS}" >&2; exit 1; }
[[ -f "${PIP_CONSTRAINT}" ]] \
    || { echo "ERROR: pip constraint not found: ${PIP_CONSTRAINT}" >&2; exit 1; }
command -v hipcc >/dev/null \
    || { echo "ERROR: hipcc is not available in PATH" >&2; exit 1; }

# shellcheck source=/dev/null
source "${ALPS_PACKAGE_HELPERS}"

src_dir=/opt/aiter
dist_dir=/tmp/aiter-dist
rm -rf "${src_dir}" "${dist_dir}" "${AITER_JIT_DIR}"

git clone --depth=1 --branch "${AITER_REF}" --recurse-submodules --shallow-submodules \
    "${AITER_REPO}" "${src_dir}"
[[ "$(git -C "${src_dir}" rev-parse HEAD)" == "${AITER_COMMIT}" ]] \
    || { echo "ERROR: ${AITER_REF} did not resolve to ${AITER_COMMIT}" >&2; exit 1; }

[[ -d "${AITER_PATCH_DIR}" ]] \
    || { echo "ERROR: missing ${AITER_PATCH_DIR}; the app Containerfile must COPY it" >&2; exit 1; }
shopt -s nullglob
aiter_patch_files=("${AITER_PATCH_DIR}"/*.patch)
shopt -u nullglob
for patch_file in "${aiter_patch_files[@]}"; do
    echo "Applying AITER patch: ${patch_file}"
    git -C "${src_dir}" apply --check "${patch_file}"
    git -C "${src_dir}" apply --verbose "${patch_file}"
done

cd "${src_dir}"
pip_install python --no-cache-dir -r requirements.txt
mkdir -p "${dist_dir}"
echo "Building AITER wheel without multi-architecture AOT; runtime JIT target: ${GPU_ARCHS}, CU_NUM=${CU_NUM}"
env -u ROCM_PATH -u HIP_PATH \
    PREBUILD_KERNELS=0 \
    AITER_USE_SYSTEM_TRITON=1 \
    GPU_ARCHS="${GPU_ARCHS}" \
    CU_NUM="${CU_NUM}" \
    MAX_JOBS="${MAX_JOBS}" \
    python setup.py bdist_wheel --dist-dir "${dist_dir}"
pip_install python --no-cache-dir --no-deps "${dist_dir}"/*.whl

# Build the core module through jit/core.py loaded standalone. Importing the
# installed aiter package can probe for a GPU, but image builders have none.
mkdir -p "${AITER_JIT_DIR}"
env -u ROCM_PATH -u HIP_PATH \
    AITER_USE_SYSTEM_TRITON=1 \
    AITER_JIT_DIR="${AITER_JIT_DIR}" \
    GPU_ARCHS="${GPU_ARCHS}" \
    CU_NUM="${CU_NUM}" \
    MAX_JOBS="${MAX_JOBS}" \
    python - "${src_dir}" <<'PY'
import sys

src_dir = sys.argv[1]
sys.path.insert(0, f"{src_dir}/aiter")
from jit import core

args = core.get_args_of_build("module_aiter_core")
core.build_module(
    md_name="module_aiter_core",
    srcs=args["srcs"],
    flags_extra_cc=args["flags_extra_cc"],
    flags_extra_hip=args["flags_extra_hip"],
    blob_gen_cmd=args["blob_gen_cmd"],
    extra_include=args["extra_include"],
    extra_ldflags=args["extra_ldflags"],
    verbose=False,
    is_python_module=True,
    is_standalone=False,
    torch_exclude=args["torch_exclude"],
    third_party=args["third_party"],
)
PY

rm -rf "${AITER_JIT_DIR}/build"
mkdir -p "${AITER_JIT_DIR}/build"
shopt -s nullglob
baked_modules=("${AITER_JIT_DIR}"/module_aiter_core*.so)
shopt -u nullglob
[[ ${#baked_modules[@]} -gt 0 ]] \
    || { echo "ERROR: no AITER core module in ${AITER_JIT_DIR}" >&2; exit 1; }
chmod -R a+rwX "${AITER_JIT_DIR}"

installed_version="$(python -c 'import importlib.metadata as m; print(m.version("amd-aiter"))')"
echo "OK: AITER ${installed_version} (${AITER_COMMIT}) installed with ${#aiter_patch_files[@]} patch(es); preseeded cache modules: ${baked_modules[*]##*/}"
rm -rf "${src_dir}" "${dist_dir}"
