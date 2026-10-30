#!/usr/bin/env bash
# Re-establish shell-local SDK/constraint state for each independently cached RUN.
# Build-only APT packages are installed and removed within that same layer.
# shellcheck source=Alps-Images/common/package-helpers.sh
source /opt/alps/package-helpers.sh

vllm_build_begin() {
    export PIP_CONSTRAINT=/opt/alps/vllm-base-pins.txt
    # Keep the build-dependency list available to vllm_build_end.
    declare -ag vllm_apt_build_deps=(
        build-essential cargo cmake curl git libibverbs-dev libnl-3-dev
        libnl-route-3-dev libnuma-dev ninja-build pkg-config rustc python3-dev
    )
    local -a vllm_apt_install_packages=(ca-certificates libibverbs1 "${vllm_apt_build_deps[@]}")
    snapshot_apt_packages /tmp/vllm-apt-before.txt
    # Match unqualified dependency names even when dpkg adds an architecture.
    local -a apt_packages_before
    mapfile -t apt_packages_before < /tmp/vllm-apt-before.txt
    printf '%s\n' "${apt_packages_before[@]%%:*}" > /tmp/vllm-apt-before.txt
    apt_get update
    apt_get install -y --no-install-recommends "${vllm_apt_install_packages[@]}"
    rm -rf /var/lib/apt/lists/*

    # shellcheck source=/dev/null
    source /opt/alps/env/alps-rocm-build.env
    # shellcheck source=Alps-Images/common/install-alps-hpc-stack.sh
    source /opt/alps/install-alps-hpc-stack.sh
    # shellcheck source=Alps-Images/ROCm/install-alps-rocm-components.sh
    source /opt/alps/install-alps-rocm-components.sh
    ROCM_BUILD_PREFIX="${ROCM_BUILD_PREFIX:?ROCM_BUILD_PREFIX missing from alps-rocm-build.env}"
    test -d "${ROCM_BUILD_PREFIX}"
    local rocm_ldconfig_file=/etc/ld.so.conf.d/99-alps-vllm-rocm-sdk.conf
    local rocm_lib_dir rocm_root rocm_libdir
    : > "${rocm_ldconfig_file}"
    while IFS= read -r rocm_lib_dir; do
        printf '%s\n' "${rocm_lib_dir}" >> "${rocm_ldconfig_file}"
    done < <(rocm_sdk_ldconfig_dirs)
    for rocm_root in $(rocm_sdk_prefix_candidates); do
        test -d "${rocm_root}" || continue
        while IFS= read -r rocm_lib_dir; do
            printf '%s\n' "${rocm_lib_dir}" >> "${rocm_ldconfig_file}"
        done < <(find "${rocm_root}" -type f -name 'librocm-openblas.so*' -printf '%h\n')
    done
    # Keep wheel runtime libraries ahead of development SDK copies, as in the base.
    awk '!seen[$0]++' "${rocm_ldconfig_file}" > "${rocm_ldconfig_file}.tmp"
    mv "${rocm_ldconfig_file}.tmp" "${rocm_ldconfig_file}"
    export ROCM_HOME="${ROCM_BUILD_PREFIX}" ROCM_PATH="${ROCM_BUILD_PREFIX}" HIP_PATH="${ROCM_BUILD_PREFIX}"
    export PATH="${ROCM_BUILD_PREFIX}/bin:${ROCM_BUILD_PREFIX}/llvm/bin:${PATH}"
    rocm_libdir="${ROCM_BUILD_PREFIX}/lib64"
    test -d "${rocm_libdir}" || rocm_libdir="${ROCM_BUILD_PREFIX}/lib"
    export LD_LIBRARY_PATH="${rocm_libdir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
}

vllm_build_end() {
    APT_CLEANUP_HOLD_PACKAGES=libibverbs1 \
        cleanup_new_apt_build_deps /tmp/vllm-apt-before.txt "${vllm_apt_build_deps[@]}"
    rm -f /tmp/vllm-apt-before.txt
    rm -rf /root/.cache/pip
}
