#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/package-helpers.sh"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

STACK_VERSIONS_FILE="${SCRIPT_DIR}/alps-stack-versions.env"
[[ -f "${STACK_VERSIONS_FILE}" ]] || die "Missing shared stack defaults: ${STACK_VERSIONS_FILE}"
# shellcheck disable=SC1090
source "${STACK_VERSIONS_FILE}"

apt_install_build_deps() {

    # Use JFrog Artifactory as an APT proxy/cache for Ubuntu packages, to speed
    # up installs and reduce load on Ubuntu mirrors.
    sed -i \
        -e 's|http://archive.ubuntu.com/ubuntu|https://jfrog.svc.cscs.ch/artifactory/ubuntu|' \
        -e 's|http://security.ubuntu.com/ubuntu|https://jfrog.svc.cscs.ch/artifactory/ubuntu|' \
        -e 's|http://ports.ubuntu.com/ubuntu-ports|https://jfrog.svc.cscs.ch/artifactory/ubuntu-ports|' \
        /etc/apt/sources.list.d/ubuntu.sources
    printf '%s\n%s' "Acquire::http::AllowRedirect "true";" "Acquire::http::Pipeline-Depth "0";" \
        > /etc/apt/apt.conf.d/99-jfrog-proxy
    apt_cmd -o "Acquire::https::Verify-Peer=false" update
    apt_cmd -o "Acquire::https::Verify-Peer=false" install -y ca-certificates

    apt_get update
    apt_get install -y --no-install-recommends \
        build-essential ca-certificates pkg-config automake autoconf libtool cmake \
        bc gdb strace wget curl git bzip2 python3 gfortran \
        rdma-core numactl \
        libconfig-dev libuv1-dev libfuse-dev libfuse3-dev libyaml-dev libnl-3-dev \
        libnuma-dev libsensors-dev libcurl4-openssl-dev libjson-c-dev \
        libsox-fmt-all \
        devscripts debhelper fakeroot dh-make
    rm -rf /var/lib/apt/lists/*
}

remove_efa() {
    rm -rf /opt/amazon/efa || true
    grep -R "/opt/amazon/efa" -n /etc/ld.so.conf.d || true
    for f in /etc/ld.so.conf.d/*; do
        [[ -f "$f" ]] || continue
        if grep -q "/opt/amazon/efa" "$f"; then rm -f "$f"; fi
    done
    ldconfig
}

purge_preinstalled_network_stack() {
    local package_patterns=(
        '*aws-ofi-nccl*'
        '*efa*'
        '*hcoll*'
        '*nccl*'
        '*nvshmem*'
        '*sharp*'
        '*spectrum-x*'
        '*ucx*'
        '*ucc*'
        'libfabric*'
    )
    local packages=() pkg pattern

    while IFS= read -r pkg; do
        for pattern in "${package_patterns[@]}"; do
            if [[ "$pkg" == $pattern ]]; then
                packages+=("$pkg")
                break
            fi
        done
    done < <(dpkg-query -W -f='${binary:Package}\n' 2>/dev/null || true)

    if [[ "${#packages[@]}" -gt 0 ]]; then
        printf 'Purging preinstalled network-stack packages: %s\n' "${packages[*]}"
        apt_get purge -y "${packages[@]}" || true
        apt_get autoremove -y || true
    fi

    local dirs=(
        /opt/amazon/aws-ofi-nccl
        /opt/amazon/efa
        /opt/amazon/ofi-nccl
        /opt/aws-ofi-nccl
        /opt/hpcx/hcoll
        /opt/hpcx/nccl_mrc_plugin
        /opt/hpcx/nccl_rdma_sharp_plugin
        /opt/hpcx/nccl_spectrum-x_plugin
        /opt/hpcx/ncclnet_plugin
        /opt/hpcx/ompi
        /opt/hpcx/ompi4
        /opt/hpcx/ompi5
        /opt/hpcx/sharp
        /opt/hpcx/ucc
        /opt/hpcx/ucx
        /usr/local/mpi
        /usr/local/ucx
        /usr/local/ucc
    )
    local d
    for d in "${dirs[@]}"; do
        if [[ -e "$d" ]]; then
            echo "Removing preinstalled network-stack path: $d"
            rm -rf "$d" || true
        fi
    done

    local lib_roots=(
        /usr/local/cuda/lib64
        /usr/local/cuda/targets/aarch64-linux/lib
        /usr/local/cuda/targets/x86_64-linux/lib
        /usr/local/lib
        /usr/local/lib64
        /usr/lib
        /usr/lib64
        /usr/lib/aarch64-linux-gnu
        /usr/lib/x86_64-linux-gnu
    )
    local root
    for root in "${lib_roots[@]}"; do
        [[ -d "$root" ]] || continue
        find "$root" -maxdepth 1 \( -type f -o -type l \) \( \
            -name 'libaws-ofi-nccl*' -o \
            -name 'libfabric*' -o \
            -name 'libhcoll*' -o \
            -name 'libnccl*' -o \
            -name 'libnvshmem*' -o \
            -name 'libsharp*' -o \
            -name 'libucc*' -o \
            -name 'libucm*' -o \
            -name 'libucp*' -o \
            -name 'libucs*' -o \
            -name 'libuct*' \
        \) -print -delete || true
    done

    remove_efa
    ldconfig
}

remove_hpcx_plugins() {
    # REMOVE_HPCX_DIRS can be space-separated or newline-separated
    if [[ -n "${REMOVE_HPCX_DIRS:-}" ]]; then
        while IFS= read -r d; do
            [[ -z "$d" ]] && continue
            echo "Removing HPCX plugin dir: $d"
            rm -rf "$d" || true
        done < <(printf "%s\n" ${REMOVE_HPCX_DIRS})
        ldconfig
    fi
}

apply_patch_if_set() {
    local patch_rel="${1:-}"
    [[ -z "$patch_rel" ]] && return 0

    local patch="/opt/alps/patches/${patch_rel}"
    [[ -f "$patch" ]] || die "Patch not found: ${patch}"

    git apply --check --whitespace=nowarn "$patch"
    git apply --whitespace=nowarn "$patch"
}

make_jobs() {
    printf '%s\n' "${ALPS_BUILD_JOBS:-$(nproc)}"
}

cmake_build_jobs() {
    printf '%s\n' "${ALPS_CMAKE_BUILD_JOBS:-${ALPS_BUILD_JOBS:-$(nproc)}}"
}

ensure_known_stack_step() {
    local family="${1:?family required}"
    local step="${2:?step required}"
    shift 2

    local known_step
    for known_step in "$@"; do
        [[ "${step}" == "${known_step}" ]] && return 0
    done

    die "Unknown ${family} stack step: ${step}"
}

run_stack_step_sequence() {
    local runner="${1:?runner required}"
    shift

    local step
    for step in "$@"; do
        "${runner}" "${step}"
    done
}

build_boost() {
    local jobs="${BOOST_BUILD_JOBS:-$(nproc)}"

    wget https://archives.boost.io/release/${BOOST_VER}/source/boost_${BOOST_VER//./_}.tar.bz2 -O /tmp/boost.tar.bz2
    tar -xjf /tmp/boost.tar.bz2 -C /tmp
    pushd "/tmp/boost_${BOOST_VER//./_}"
    ./bootstrap.sh
    ./b2 \
        --with-headers \
        --with-program_options \
        --layout=system \
        toolset=gcc \
        variant=release \
        link=shared \
        threading=multi \
        runtime-link=shared \
        install -j"${jobs}"
    popd
    rm -rf "/tmp/boost_${BOOST_VER//./_}" /tmp/boost.tar.bz2
    ldconfig
}

build_xpmem() {
    local ref="${XPMEM_REF}"
    git clone https://github.com/hpc/xpmem.git /tmp/xpmem
    pushd /tmp/xpmem
    git checkout "${ref}"
    ./autogen.sh
    ./configure --prefix=/usr --with-default-prefix=/usr --disable-kernel-module
    make -j"$(make_jobs)"
    make install
    popd
    rm -rf /tmp/xpmem
    ldconfig
}

init_hpc_stack_prefixes() {
    : "${HPCX_PREFIX:=/opt/hpcx}"
    : "${LIBFABRIC_PREFIX:=/usr}"
    : "${UCX_PREFIX:=${HPCX_PREFIX}/ucx}"
    : "${UCC_PREFIX:=${HPCX_PREFIX}/ucc}"
    : "${OMPI_PREFIX:=${HPCX_PREFIX}/ompi}"
    : "${HWLOC_PREFIX:=${OMPI_PREFIX}}"

    export HPCX_PREFIX LIBFABRIC_PREFIX UCX_PREFIX UCC_PREFIX OMPI_PREFIX HWLOC_PREFIX
}

record_alps_version_var() {
    local name="${1:?name required}"
    local value="${2:-}"

    mkdir -p /opt/alps/env
    printf 'export %s=%q\n' "${name}" "${value}" >> /opt/alps/env/alps-versions.env
}

build_cxi_bits_common() {
    init_hpc_stack_prefixes

    git clone --depth 1 --branch "${CASSINI_HEADERS_VERSION}" https://github.com/HewlettPackard/shs-cassini-headers.git /tmp/shs-cassini-headers
    cp -r /tmp/shs-cassini-headers/include/* /usr/include/
    cp -r /tmp/shs-cassini-headers/share/* /usr/share/
    rm -rf /tmp/shs-cassini-headers

    git clone --depth 1 --branch "${CXI_DRIVER_VERSION}" https://github.com/HewlettPackard/shs-cxi-driver.git /tmp/shs-cxi-driver
    cp -r /tmp/shs-cxi-driver/include/* /usr/include/
    rm -rf /tmp/shs-cxi-driver

    git clone --depth 1 --branch "${LIBCXI_VERSION}" https://github.com/HewlettPackard/shs-libcxi.git /tmp/shs-libcxi
    pushd /tmp/shs-libcxi
    ./autogen.sh
    ./configure --prefix="${LIBFABRIC_PREFIX}" "$@"
    make -j"$(make_jobs)"
    make install
    popd
    rm -rf /tmp/shs-libcxi
    ldconfig
}

build_libfabric_common() {
    init_hpc_stack_prefixes

    git clone https://github.com/ofiwg/libfabric.git /tmp/libfabric
    pushd /tmp/libfabric
    git reset --hard "${LIBFABRIC_COMMIT}"
    apply_patch_if_set "${LIBFABRIC_PATCH}"
    ./autogen.sh
    ./configure --prefix="${LIBFABRIC_PREFIX}" \
        "$@" \
        --enable-xpmem=/usr \
        --enable-cxi
    make -j"$(make_jobs)"
    make install

    local fi_info_bin="${LIBFABRIC_PREFIX}/bin/fi_info"
    if [[ ! -x "${fi_info_bin}" ]]; then
        fi_info_bin="$(command -v fi_info)"
    fi
    record_alps_version_var LIBFABRIC_VERSION "$(${fi_info_bin} --version | head -n 1 | awk '{ print $2; }')"
    record_alps_version_var LIBFABRIC_COMMIT "${LIBFABRIC_COMMIT}"
    popd
    rm -rf /tmp/libfabric
    ldconfig
}

build_ucx_common() {
    init_hpc_stack_prefixes

    rm -rf "${UCX_PREFIX}"
    curl -fsSL "https://github.com/openucx/ucx/releases/download/v${UCX_VERSION}/ucx-${UCX_VERSION}.tar.gz" -o /tmp/ucx.tar.gz
    tar -C /tmp -xzf /tmp/ucx.tar.gz
    pushd "/tmp/ucx-${UCX_VERSION}"
    mkdir -p build && cd build
    ../configure \
        --prefix="${UCX_PREFIX}" \
        "$@" \
        --enable-mt \
        --enable-devel-headers
    make -j"$(make_jobs)"
    make install
    record_alps_version_var UCX_VERSION "${UCX_VERSION}"
    popd
    rm -rf "/tmp/ucx-${UCX_VERSION}" /tmp/ucx.tar.gz
}

build_ucc_common() {
    init_hpc_stack_prefixes

    rm -rf "${UCC_PREFIX}"
    git clone --depth 1 --branch "v${UCC_VERSION}" https://github.com/openucx/ucc.git /tmp/ucc
    pushd /tmp/ucc
    ./autogen.sh
    ./configure \
        --prefix="${UCC_PREFIX}" \
        --with-ucx="${UCX_PREFIX}" \
        "$@"
    make -j"$(make_jobs)"
    make install
    record_alps_version_var UCC_VERSION "${UCC_VERSION}"
    popd
    rm -rf /tmp/ucc
}

build_ompi5_common() {
    init_hpc_stack_prefixes

    rm -rf "${OMPI_PREFIX}"
    curl -fsSL "https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-${OMPI_VER}.tar.gz" -o /tmp/ompi.tar.gz
    tar -C /tmp -xzf /tmp/ompi.tar.gz
    pushd "/tmp/openmpi-${OMPI_VER}"
    ./configure \
        --prefix="${OMPI_PREFIX}" \
        --with-ofi="${LIBFABRIC_PREFIX}" \
        --with-ucx="${UCX_PREFIX}" \
        --with-ucc="${UCC_PREFIX}" \
        --enable-oshmem \
        "$@"
    make -j"$(make_jobs)"
    make install
    rm -rf /usr/local/mpi
    ln -s "${OMPI_PREFIX}" /usr/local/mpi
    record_alps_version_var OMPI_VERSION "${OMPI_VER}"
    popd
    rm -rf "/tmp/openmpi-${OMPI_VER}" /tmp/ompi.tar.gz
    ldconfig
}

build_aws_ofi_nccl_common() {
    init_hpc_stack_prefixes

    git clone "${AWS_OFI_NCCL_REPO}" /tmp/aws-ofi-nccl
    pushd /tmp/aws-ofi-nccl
    git reset --hard "${AWS_OFI_NCCL_COMMIT}"

    if git config -f .gitmodules --get-regexp '^submodule\..*\.path$' 2>/dev/null | grep -q ' 3rd-party/efa-dp-direct$'; then
        git submodule update --init --recursive 3rd-party/efa-dp-direct
    fi
    apply_patch_if_set "${AWS_OFI_NCCL_PATCH}"

    unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH
    export CPPFLAGS="${CPPFLAGS:-}"
    export CFLAGS="${CFLAGS:-}"
    export CXXFLAGS="${CXXFLAGS:-}"
    CPPFLAGS="$(echo "$CPPFLAGS" | sed 's| -isystem /usr/include||g')"
    CFLAGS="$(echo "$CFLAGS" | sed 's| -isystem /usr/include||g')"
    CXXFLAGS="$(echo "$CXXFLAGS" | sed 's| -isystem /usr/include||g')"
    export CPPFLAGS CFLAGS CXXFLAGS

    ./autogen.sh
    ./configure \
        --prefix=/usr \
        --disable-tests \
        --with-libfabric="${LIBFABRIC_PREFIX}" \
        --with-mpi="${OMPI_PREFIX}" \
        --with-hwloc="${HWLOC_PREFIX}" \
        "$@"

    # Some base images inject /usr/include as -isystem, which breaks GCC system
    # header handling during aws-ofi-nccl builds.
    find . \( \
        -name 'Makefile' -o -name 'Makefile.in' -o -name 'Makefile.am' -o -name '*.mk' -o -name 'config.status' -o -name 'libtool' \
    \) -type f -print0 \
    | xargs -0 -r sed -i 's| -isystem /usr/include||g'

    make -j"$(make_jobs)"
    make install

    record_alps_version_var AWS_OFI_NCCL_VERSION "$(./m4/get_version.sh)"
    record_alps_version_var AWS_OFI_NCCL_COMMIT "${AWS_OFI_NCCL_COMMIT}"

    popd
    rm -rf /tmp/aws-ofi-nccl
    ldconfig
}

ensure_debian_packaging_tools() {
    if command -v debuild >/dev/null 2>&1; then
        return 0
    fi

    apt_get update
    apt_get install -y --no-install-recommends devscripts debhelper fakeroot dh-make
    rm -rf /var/lib/apt/lists/*

    command -v debuild >/dev/null 2>&1 || die "debuild not found after installing devscripts"
}

clean_up() {
    cleanup_apt_build_deps \
        pkg-config automake autoconf libtool cmake \
        libconfig-dev libuv1-dev libfuse-dev libfuse3-dev libyaml-dev libsensors-dev libcurl4-openssl-dev \
        fakeroot dh-make
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    die "This file provides stack build functions; run a family installer instead."
fi
