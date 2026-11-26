#!/usr/bin/env bash
# Add native training UCCL and a JIT seed without changing Primus compute packages.
set -euo pipefail
root=/opt/alps/apertus2-primus
# shellcheck source=Alps-Images/common/package-helpers.sh
source /opt/alps/package-helpers.sh
# shellcheck source=/dev/null
source /opt/alps/env/alps-rocm-build.env
export PIP_CONSTRAINT="$root/base-pins.txt"
export MAX_JOBS="${APERTUS_BUILD_JOBS:-4}" CMAKE_BUILD_PARALLEL_LEVEL="${APERTUS_BUILD_JOBS:-4}"
export ROCM_HOME="$ROCM_BUILD_PREFIX"
export PATH="$ROCM_SDK_BIN:$ROCM_BUILD_PREFIX/llvm/bin:$PATH"
export PYTORCH_ROCM_ARCH=gfx942 HIP_ARCHITECTURES=gfx942
# Let the wheel SDK compiler discover its own resource paths.
unset ROCM_PATH HIP_PATH
case "${1:?component required}" in
    uccl)
        # Runtime libraries survive removal of newly added development headers.
        apt_get update
        apt_get install -y --no-install-recommends libibverbs1 libnl-3-200 libnl-route-3-200 libnuma1
        build_packages=(pkg-config libibverbs-dev libnl-3-dev libnl-route-3-dev libnuma-dev)
        added_packages=()
        for package in "${build_packages[@]}"; do
            if ! dpkg-query -W -f='${db:Status-Abbrev}' "$package" 2>/dev/null | grep -q '^ii'; then
                added_packages+=("$package")
            fi
        done
        apt_get install -y --no-install-recommends "${build_packages[@]}"
        pip_install python3 --no-cache-dir --no-deps -r "$root/uccl-requirements.lock"
        bash /opt/alps/apertus2-rocm/install-uccl.sh
        purge_apt_build_deps_exact "${added_packages[@]}"
        rm -rf /var/lib/apt/lists/*
        ;;
    aiter)
        export GPU_ARCHS=gfx942 CU_NUM=228 AITER_USE_SYSTEM_TRITON=1
        export AITER_JIT_DIR=/opt/aiter-jit
        python3 "$root/seed-aiter.py"
        ;;
    *) echo "Unknown component: $1" >&2; exit 2 ;;
esac
