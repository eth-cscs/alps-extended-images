#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

die() {
    echo "ERROR: $*" >&2
    exit 1
}

apt_install_build_deps() {
    apt-get update
    apt-get install -y --no-install-recommends \
        build-essential ca-certificates pkg-config automake autoconf libtool cmake \
        bc gdb strace wget curl git bzip2 python3 gfortran \
        rdma-core numactl \
        libconfig-dev libuv1-dev libfuse-dev libfuse3-dev libyaml-dev libnl-3-dev \
        libnuma-dev libsensors-dev libcurl4-openssl-dev libjson-c-dev libibverbs-dev \
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

remove_hpcx_plugins() {
    # REMOVE_HPCX_DIRS can be space-separated or newline-separated
    if [[ -n "${REMOVE_HPCX_DIRS:-}" ]]; then
        while IFS= read -r d; do
            [[ -z "$d" ]] && continue
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

detect_cuda_dir() {
    if [[ -n "${CUDA_HOME:-}" && -d "${CUDA_HOME}" ]]; then
        echo "${CUDA_HOME}"
        return 0
    fi
    if [[ -n "${CUDA_PATH:-}" && -d "${CUDA_PATH}" ]]; then
        echo "${CUDA_PATH}"
        return 0
    fi
    if command -v nvcc >/dev/null 2>&1; then
        # nvcc is typically in <CUDA_DIR>/bin/nvcc
        local nvcc_path
        nvcc_path="$(command -v nvcc)"
        echo "$(cd "$(dirname "$nvcc_path")/.." && pwd)"
        return 0
    fi
    if [[ -d /usr/local/cuda ]]; then
        echo /usr/local/cuda
        return 0
    fi
    return 1
}

build_xpmem() {
    local ref="${XPMEM_REF}"
    git clone https://github.com/hpc/xpmem.git /tmp/xpmem
    pushd /tmp/xpmem
    git checkout "${ref}"
    ./autogen.sh
    ./configure --prefix=/usr --with-default-prefix=/usr --disable-kernel-module
    make -j"$(nproc)"
    make install
    popd
    rm -rf /tmp/xpmem
    ldconfig
}

build_gdrcopy() {
    local ver="${GDRCOPY_VER}"
    git clone --depth 1 --branch "v${ver}" https://github.com/NVIDIA/gdrcopy.git /tmp/gdrcopy
    pushd /tmp/gdrcopy
    make CC=gcc CUDA="${CUDA_DIR}" lib -j"$(nproc)"
    make lib_install
    popd
    rm -rf /tmp/gdrcopy
    ldconfig
}

build_cxi_bits() {
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
    ./configure --prefix=/usr --with-cuda="${CUDA_DIR}"
    make -j"$(nproc)"
    make install
    popd
    rm -rf /tmp/shs-libcxi
    ldconfig
}

build_libfabric() {
    git clone https://github.com/ofiwg/libfabric.git /tmp/libfabric
    pushd /tmp/libfabric
    git reset --hard "${LIBFABRIC_COMMIT}"
    apply_patch_if_set "${LIBFABRIC_PATCH}"
    ./autogen.sh
    ./configure --prefix=/usr \
        --with-cuda="${CUDA_DIR}" \
        --enable-cuda-dlopen \
        --enable-gdrcopy-dlopen \
        --enable-xpmem=/usr \
        --enable-cxi
    make -j"$(nproc)"
    make install
    popd
    rm -rf /tmp/libfabric
    ldconfig
}

build_nccl_deb() {
    curl -fsSL "https://github.com/NVIDIA/nccl/archive/refs/tags/v${NCCL_VER}.tar.gz" -o /tmp/nccl.tar.gz
    tar -C /tmp -xzf /tmp/nccl.tar.gz
    pushd "/tmp/nccl-${NCCL_VER}"
    apply_patch_if_set "${NCCL_PATCH}"
    make -j"$(nproc)" pkg.debian.build CUDA_HOME="${CUDA_DIR}"
    dpkg -i build/pkg/deb/*.deb
    popd
    rm -rf "/tmp/nccl-${NCCL_VER}" /tmp/nccl.tar.gz
    ldconfig
}

build_ucx() {
    local hpcx=/opt/hpcx
    rm -rf "${hpcx}/ucx"
    curl -fsSL "https://github.com/openucx/ucx/releases/download/v${UCX_VERSION}/ucx-${UCX_VERSION}.tar.gz" -o /tmp/ucx.tar.gz
    tar -C /tmp -xzf /tmp/ucx.tar.gz
    pushd "/tmp/ucx-${UCX_VERSION}"
    mkdir -p build && cd build
    ../configure \
        --prefix="${hpcx}/ucx" \
        --with-cuda="${CUDA_DIR}" \
        --with-gdrcopy=/usr/local \
        --enable-mt \
        --enable-devel-headers
    make -j"$(nproc)"
    make install
    popd
    rm -rf "/tmp/ucx-${UCX_VERSION}" /tmp/ucx.tar.gz
}

build_ucc() {
    local hpcx=/opt/hpcx
    rm -rf "${hpcx}/ucc"
    git clone --depth 1 --branch "v${UCC_VERSION}" https://github.com/openucx/ucc.git /tmp/ucc
    pushd /tmp/ucc
    ./autogen.sh

    local gencode_sm90='-gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90'
    local gencode_sm90a='-gencode arch=compute_90a,code=sm_90a -gencode arch=compute_90a,code=compute_90a'
    local gencode="${gencode_sm90} ${gencode_sm90a}"

    ./configure \
        --prefix="${hpcx}/ucc" \
        --with-ucx="${hpcx}/ucx" \
        --with-cuda="${CUDA_DIR}" \
        --with-nvcc-gencode="${gencode}" \
        --with-nccl
    make -j"$(nproc)"
    make install
    popd
    rm -rf /tmp/ucc
}

build_ompi5() {
    local hpcx=/opt/hpcx
    rm -rf "${hpcx}/ompi"
    curl -fsSL "https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-${OMPI_VER}.tar.gz" -o /tmp/ompi.tar.gz
    tar -C /tmp -xzf /tmp/ompi.tar.gz
    pushd "/tmp/openmpi-${OMPI_VER}"
    ./configure \
        --prefix="${hpcx}/ompi" \
        --with-ofi=/usr \
        --with-ucx="${hpcx}/ucx" \
        --with-ucc="${hpcx}/ucc" \
        --enable-oshmem \
        --with-cuda="${CUDA_DIR}" \
        --with-cuda-libdir="${CUDA_DIR}/lib64/stubs"
    make -j"$(nproc)"
    make install
    popd
    rm -rf "/tmp/openmpi-${OMPI_VER}" /tmp/ompi.tar.gz
    ldconfig
}

build_aws_ofi_nccl() {
    git clone https://github.com/aws/aws-ofi-nccl.git /tmp/aws-ofi-nccl
    pushd /tmp/aws-ofi-nccl
    git reset --hard "${AWS_OFI_NCCL_COMMIT}"
    apply_patch_if_set "${AWS_OFI_NCCL_PATCH}"
    ./autogen.sh
    CPPFLAGS="" ./configure \
        --prefix=/usr \
        --with-libfabric=/usr \
        --with-cuda="${CUDA_DIR}" \
        --with-mpi=/opt/hpcx/ompi \
        --with-hwloc=/opt/hpcx/ompi

    # critical fix: remove /usr/include being injected as -isystem
    grep -R --line-number --fixed-string " -isystem /usr/include" . || true
    find . -name 'Makefile' -o -name 'Makefile.in' | xargs sed -i 's| -isystem /usr/include||g'

    make -j"$(nproc)"
    make install
    popd
    rm -rf /tmp/aws-ofi-nccl
    ldconfig
}

build_nccl_tests() {
    git clone --depth 1 --branch "v${NCCL_TESTS_VER}" https://github.com/NVIDIA/nccl-tests.git /tmp/nccl-tests
    pushd /tmp/nccl-tests
    MPI=1 MPI_HOME=/opt/hpcx/ompi CUDA_HOME="${CUDA_DIR}" make -j"$(nproc)"
    install -d /usr/local/bin
    find build -maxdepth 1 -type f -executable -name '*_perf' -print -exec install -m 0755 {} /usr/local/bin/ \;
    popd
    rm -rf /tmp/nccl-tests
}

build_osu() {
    curl -fsSL "http://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-${OSU_VERSION}.tar.gz" -o /tmp/osu.tar.gz
    tar --no-same-owner --no-same-permissions -C /tmp -xzf /tmp/osu.tar.gz
    pushd "/tmp/osu-micro-benchmarks-${OSU_VERSION}"
    CC=/opt/hpcx/ompi/bin/mpicc \
    CFLAGS="-O3 -lcuda -lnvidia-ml" \
    ./configure \
        --prefix=/usr/local \
        --enable-cuda \
        --with-cuda-include="${CUDA_DIR}/include" \
        --with-cuda-libpath="${CUDA_DIR}/lib64"
    make -j"$(nproc)"
    make install
    popd
    rm -rf "/tmp/osu-micro-benchmarks-${OSU_VERSION}" /tmp/osu.tar.gz
    ldconfig
}

install_python_pkgs() {
    python -m pip install ${PIP_PACKAGES}
}

main() {
    CUDA_DIR="$(detect_cuda_dir)" || die "Could not determine CUDA directory..."
    export CUDA_DIR
    export CUDA_HOME="${CUDA_HOME:-$CUDA_DIR}"
    export CUDA_PATH="${CUDA_PATH:-$CUDA_DIR}"

    apt_install_build_deps

    remove_efa
    remove_hpcx_plugins

    build_xpmem
    build_gdrcopy
    build_cxi_bits
    build_libfabric
    build_nccl_deb
    build_ucx
    build_ucc
    build_ompi5
    build_aws_ofi_nccl
    build_nccl_tests
    build_osu

    install_python_pkgs
}

main "$@"
