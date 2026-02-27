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
    # Produces: ext-profiler/inspector/libnccl-profiler-inspector.so
    pushd ext-profiler/inspector
    make -j"$(nproc)" CUDA_HOME="${CUDA_DIR}"
    install -D -m 0644 libnccl-profiler-inspector.so /usr/local/lib/libnccl-profiler-inspector.so
    popd
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
        --with-libfabric=/usr \
        --with-cuda="${CUDA_DIR}" \
        --with-mpi=/opt/hpcx/ompi \
        --with-hwloc=/opt/hpcx/ompi

    # critical fix: remove /usr/include being injected as -isystem
    find . \( \
        -name 'Makefile' -o -name 'Makefile.in' -o -name 'Makefile.am' -o -name '*.mk' -o -name 'config.status' -o -name 'libtool' \
    \) -type f -print0 \
    | xargs -0 -r sed -i 's| -isystem /usr/include||g'

    make -j"$(nproc)"
    make install
    popd
    rm -rf /tmp/aws-ofi-nccl
    ldconfig
}

build_nvshmem() {
    : "${NVSHMEM_PREFIX:=/opt/nvshmem}"
    : "${NVSHMEM_BUILDDIR:=/tmp/nvshmem-build}"
    : "${NVSHMEM_SRC_DIR:=/tmp/nvshmem-src}"
    : "${NVSHMEM_CUDA_ARCH:=90}"
    : "${NVSHMEM_ENABLE_PYTHON:=1}"
    : "${NVSHMEM_ENABLE_TESTS:=1}"

    # Remove preinstalled NVSHMEM
    apt-get update
    apt-get purge -y 'libnvshmem*-cuda-*' 'nvshmem*' || true
    apt-get autoremove -y || true

    # Remove CUDA symlinks/copies that can shadow our install
    rm -f "${CUDA_DIR}/lib64/libnvshmem"*".so"* || true
    rm -f "${CUDA_DIR}/targets/"*/lib/libnvshmem*".so"* || true
    rm -rf /usr/lib/*/nvshmem || true

    rm -rf "${NVSHMEM_SRC_DIR}" "${NVSHMEM_BUILDDIR}"
    mkdir -p "${NVSHMEM_BUILDDIR}"

    # Clone repo
    git clone --depth 1 --branch "v${NVSHMEM_VER}" https://github.com/NVIDIA/nvshmem.git ${NVSHMEM_SRC_DIR}

    NVSHMEM_BUILD_EXAMPLES=0 \
    NVSHMEM_BUILD_TESTS=1 \
    NVSHMEM_DEBUG=0 \
    NVSHMEM_DEVEL=0 \
    NVSHMEM_DEFAULT_PMI2=0 \
    NVSHMEM_DEFAULT_PMIX=1 \
    NVSHMEM_DISABLE_COLL_POLL=1 \
    NVSHMEM_ENABLE_ALL_DEVICE_INLINING=0 \
    NVSHMEM_GPU_COLL_USE_LDST=0 \
    NVSHMEM_LIBFABRIC_SUPPORT=1 \
    NVSHMEM_MPI_SUPPORT=1 \
    NVSHMEM_MPI_IS_OMPI=1 \
    NVSHMEM_NVTX=1 \
    NVSHMEM_PMIX_SUPPORT=1 \
    NVSHMEM_SHMEM_SUPPORT=1 \
    NVSHMEM_TEST_STATIC_LIB=0 \
    NVSHMEM_TIMEOUT_DEVICE_POLLING=0 \
    NVSHMEM_TRACE=0 \
    NVSHMEM_USE_DLMALLOC=0 \
    NVSHMEM_USE_NCCL=1 \
    NVSHMEM_USE_GDRCOPY=1 \
    NVSHMEM_VERBOSE=0 \
    NVSHMEM_DEFAULT_UCX=0 \
    NVSHMEM_UCX_SUPPORT=1 \
    NVSHMEM_IBGDA_SUPPORT=0 \
    NVSHMEM_IBGDA_SUPPORT_GPUMEM_ONLY=0 \
    NVSHMEM_IBDEVX_SUPPORT=0 \
    NVSHMEM_IBRC_SUPPORT=0 \
    LIBFABRIC_HOME=/usr \
    NCCL_HOME=/usr \
    GDRCOPY_HOME=/usr/local \
    MPI_HOME=/opt/hpcx/ompi \
    PMIX_HOME=/opt/hpcx/ompi \
    cmake -S "${NVSHMEM_SRC_DIR}" -B "${NVSHMEM_BUILDDIR}" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${NVSHMEM_PREFIX}" \
        -DCMAKE_CUDA_ARCHITECTURES="${NVSHMEM_CUDA_ARCH}"

    cmake --build "${NVSHMEM_BUILDDIR}" -j"$(nproc)"
    cmake --install "${NVSHMEM_BUILDDIR}"

    # Ensure loader finds our NVSHMEM without LD_LIBRARY_PATH
    cat > /etc/ld.so.conf.d/99-nvshmem.conf <<EOF
${NVSHMEM_PREFIX}/lib
${NVSHMEM_PREFIX}/lib64
EOF
    ldconfig

    # pip install wheel (build installs/copies wheels into prefix but does not install into python)
    if [[ "${NVSHMEM_ENABLE_PYTHON}" == "1" ]]; then
        if python -c 'import nvshmem.core as _' >/dev/null 2>&1; then
            echo "[nvshmem4py] already importable; skipping wheel install"
        else
            local cp_tag mach cuda_major best
            cp_tag="$(python -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')"
            mach="$(python -c 'import platform; print(platform.machine())')"
            cuda_major="$("${CUDA_DIR}/bin/nvcc" --version | awk '/release [0-9]+/ {for(i=1;i<=NF;i++) if($i=="release"){gsub(",","",$(i+1)); split($(i+1),a,"."); print a[1]; exit}}')"

            # Prefer the most specific wheel: linux_<arch> > manylinux
            # Search both build/dist and install tree dist locations
            best="$(
              find "${NVSHMEM_BUILDDIR}/dist" "${NVSHMEM_PREFIX}/lib" "${NVSHMEM_PREFIX}/lib64" \
                   -type f -name "nvshmem4py_cu${cuda_major}-*.whl" 2>/dev/null \
              | grep -E "${cp_tag}-${cp_tag}-linux_${mach}\.whl$" \
              | sort -V | tail -n1 || true
            )"
            if [[ -z "${best}" ]]; then
              best="$(
                find "${NVSHMEM_BUILDDIR}/dist" "${NVSHMEM_PREFIX}/lib" "${NVSHMEM_PREFIX}/lib64" \
                     -type f -name "nvshmem4py_cu${cuda_major}-*.whl" 2>/dev/null \
                | grep -E "${cp_tag}-${cp_tag}-.*manylinux.*_${mach}\.whl$" \
                | sort -V | tail -n1 || true
              )"
            fi
            [[ -n "${best}" ]] || die "[nvshmem4py] no suitable wheel found (cu=${cuda_major}, cp=${cp_tag}, arch=${mach})"

            python -m pip install --no-cache-dir --no-deps --force-reinstall "${best}"
            req="${NVSHMEM_SRC_DIR}/nvshmem4py/requirements_cuda${cuda_major}.txt"
            [[ -f "${req}" ]] || die "nvshmem4py requirements not found: ${req}"
            
            # Install nvshmem4py deps *except* the pip-provided NVSHMEM runtime (nvidia-nvshmem-cuXX).
            # Avoid upgrades unless needed.
            python -m pip install --no-cache-dir --upgrade-strategy only-if-needed -r <(
                grep -Ev '^\s*(nvidia[-_])?nvshmem(-cu[0-9]+)?\s*([=<>!~].*)?\s*$' "${req}"
            )

            python -c 'import nvshmem.core as _; print("nvshmem4py ok")'
        fi
    fi

    rm -rf "${NVSHMEM_SRC_DIR}" "${NVSHMEM_BUILDDIR}"
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
    build_nvshmem

    build_nccl_tests
    build_osu

    install_python_pkgs
}

main "$@"
