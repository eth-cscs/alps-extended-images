#!/usr/bin/env bash

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

build_gdrcopy() {
    local ver="${GDRCOPY_VER}"
    git clone --depth 1 --branch "v${ver}" https://github.com/NVIDIA/gdrcopy.git /tmp/gdrcopy
    pushd /tmp/gdrcopy
    make CC=gcc CUDA="${CUDA_DIR}" lib -j"$(make_jobs)"
    make lib_install
    popd
    rm -rf /tmp/gdrcopy
    ldconfig
}

build_cxi_bits() {
    build_cxi_bits_common --with-cuda="${CUDA_DIR}"
}

build_libfabric() {
    build_libfabric_common \
        --with-cuda="${CUDA_DIR}" \
        --enable-cuda-dlopen \
        --enable-gdrcopy-dlopen
}

build_nccl_deb() {
    ensure_debian_packaging_tools

    curl -fsSL "https://github.com/NVIDIA/nccl/archive/refs/tags/v${NCCL_VER}.tar.gz" -o /tmp/nccl.tar.gz
    tar -C /tmp -xzf /tmp/nccl.tar.gz
    pushd "/tmp/nccl-${NCCL_VER}"
    apply_patch_if_set "${NCCL_PATCH}"
    make -j"$(make_jobs)" pkg.debian.build CUDA_HOME="${CUDA_DIR}"
    dpkg -i build/pkg/deb/*.deb
    mkdir -p /opt/alps/env
    printf 'export NCCL_VERSION=%q\n' "${NCCL_VER}" >> /opt/alps/env/alps-versions.env
    # Produces: ext-profiler/inspector/libnccl-profiler-inspector.so
    pushd plugins/profiler/inspector
    make -j"$(make_jobs)" CUDA_HOME="${CUDA_DIR}"
    install -D -m 0644 libnccl-profiler-inspector.so /usr/local/lib/libnccl-profiler-inspector.so
    popd
    popd
    rm -rf "/tmp/nccl-${NCCL_VER}" /tmp/nccl.tar.gz
    ldconfig
}

build_ucx() {
    build_ucx_common \
        --with-cuda="${CUDA_DIR}" \
        --with-gdrcopy=/usr/local
}

build_ucc() {
    local gencode_sm90='-gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90'
    local gencode_sm90a='-gencode arch=compute_90a,code=sm_90a -gencode arch=compute_90a,code=compute_90a'
    local gencode="${gencode_sm90} ${gencode_sm90a}"

    build_ucc_common \
        --with-cuda="${CUDA_DIR}" \
        --with-nvcc-gencode="${gencode}" \
        --with-nccl
}

build_ompi5() {
    build_ompi5_common \
        --with-cuda="${CUDA_DIR}" \
        --with-cuda-libdir="${CUDA_DIR}/lib64/stubs"
}

build_aws_ofi_nccl() {
    build_aws_ofi_nccl_common \
        --with-cuda="${CUDA_DIR}" \
        "$@"
}

build_nvshmem() {
    : "${NVSHMEM_PREFIX:=/opt/nvshmem}"
    : "${NVSHMEM_BUILDDIR:=/tmp/nvshmem-build}"
    : "${NVSHMEM_SRC_DIR:=/tmp/nvshmem-src}"
    : "${NVSHMEM_CUDA_ARCH:=90}"
    : "${NVSHMEM_ENABLE_PYTHON:=1}"
    : "${NVSHMEM_ENABLE_TESTS:=1}"

    # Remove preinstalled NVSHMEM
    apt_get update
    local nvshmem_packages=() pkg
    while IFS= read -r pkg; do
        case "${pkg}" in
            libnvshmem*-cuda-*|nvshmem*) nvshmem_packages+=("${pkg}");;
        esac
    done < <(dpkg-query -W -f='${binary:Package}\n' 2>/dev/null || true)
    if [[ "${#nvshmem_packages[@]}" -gt 0 ]]; then
        apt_get purge -y "${nvshmem_packages[@]}" || true
        apt_get autoremove -y || true
    fi

    # Remove CUDA symlinks/copies that can shadow our install
    rm -f "${CUDA_DIR}/lib64/libnvshmem"*.so* || true
    rm -f "${CUDA_DIR}/targets/"*/lib/libnvshmem*.so* || true
    rm -rf /usr/lib/*/nvshmem || true

    rm -rf "${NVSHMEM_SRC_DIR}" "${NVSHMEM_BUILDDIR}"
    mkdir -p "${NVSHMEM_BUILDDIR}"

    # Clone repo
    git clone --depth 1 --branch "v${NVSHMEM_VER}" https://github.com/NVIDIA/nvshmem.git "${NVSHMEM_SRC_DIR}"

    # Apply local nvshmem4py patch set
    pushd "${NVSHMEM_SRC_DIR}" >/dev/null
    apply_patch_if_set "${NVSHMEM_PATCH}"
    popd >/dev/null

    init_hpc_stack_prefixes

    local mpi_home="${MPI_HOME:-${OMPI_PREFIX}}"
    if [[ -e "${mpi_home}" ]]; then
        mpi_home="$(realpath -e "${mpi_home}")"
    fi

    # Nemo 25.11 injects /opt/venv site-packages through PYTHONPATH. NVSHMEM's
    # CMake build creates nested venvs, and that inherited PYTHONPATH can shadow
    # their freshly installed build dependencies.
    env -u PYTHONPATH \
    NVSHMEM_BUILD_EXAMPLES=0 \
    NVSHMEM_BUILD_TESTS="$([[ "${NVSHMEM_ENABLE_TESTS}" == "1" ]] && echo 1 || echo 0)" \
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
    LIBFABRIC_HOME="${LIBFABRIC_PREFIX}" \
    NCCL_HOME=/usr \
    GDRCOPY_HOME=/usr/local \
    MPI_HOME="${mpi_home}" \
    PMIX_HOME="${mpi_home}" \
    SHMEM_HOME="${mpi_home}" \
    UCX_HOME="${UCX_PREFIX}" \
    CUDAToolkit_ROOT="${CUDA_DIR}" \
    cmake -S "${NVSHMEM_SRC_DIR}" -B "${NVSHMEM_BUILDDIR}" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${NVSHMEM_PREFIX}" \
        -DCUDAToolkit_ROOT="${CUDA_DIR}" \
        -DMPI_C_COMPILER="${mpi_home}/bin/mpicc" \
        -DMPI_CXX_COMPILER="${mpi_home}/bin/mpicxx" \
        -DCMAKE_CUDA_ARCHITECTURES="${NVSHMEM_CUDA_ARCH}"

    env -u PYTHONPATH cmake --build "${NVSHMEM_BUILDDIR}" -j"$(cmake_build_jobs)"
    cmake --install "${NVSHMEM_BUILDDIR}"

    # Ensure loader finds our NVSHMEM without LD_LIBRARY_PATH
    cat > /etc/ld.so.conf.d/99-nvshmem.conf <<EOF
${NVSHMEM_PREFIX}/lib
${NVSHMEM_PREFIX}/lib64
EOF

    mkdir -p /opt/alps/env
    printf 'export NVSHMEM_VERSION=%q\n' "${NVSHMEM_VER}" >> /opt/alps/env/alps-versions.env

    ldconfig

    # Build installs/copies wheels into the tree, but does not install into python.
    if [[ "${NVSHMEM_ENABLE_PYTHON}" == "1" ]]; then
        if python -c 'import nvshmem.core as _' >/dev/null 2>&1; then
            echo "[nvshmem4py] already importable; skipping wheel install"
        else
            local cp_tag mach cuda_major best req
            local constraint_file

            cp_tag="$(python -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')"
            mach="$(python -c 'import platform; print(platform.machine())')"
            cuda_major="$("${CUDA_DIR}/bin/nvcc" --version | awk '
                /release [0-9]+/ {
                    for (i = 1; i <= NF; i++) {
                        if ($i == "release") {
                            gsub(",", "", $(i+1))
                            split($(i+1), a, ".")
                            print a[1]
                            exit
                        }
                    }
                }'
            )"

            # Prefer the most specific wheel: linux_<arch> > manylinux
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

            pip_install python --no-cache-dir --no-deps --force-reinstall "${best}"

            req="${NVSHMEM_SRC_DIR}/nvshmem4py/requirements_cuda${cuda_major}.txt"
            [[ -f "${req}" ]] || die "nvshmem4py requirements not found: ${req}"

            constraint_file="$(mktemp)"

            # If cuda-pathfinder is already installed and satisfies the upstream
            # requirement, pin it to the installed version for this install.
            REQ_FILE="${req}" CONSTRAINT_FILE="${constraint_file}" python - <<'PY'
import os
import re
from importlib.metadata import version, PackageNotFoundError

try:
    from packaging.requirements import Requirement
except Exception:
    raise SystemExit(0)

req_file = os.environ["REQ_FILE"]
constraint_file = os.environ["CONSTRAINT_FILE"]

def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()

pathfinder_req = None
with open(req_file, "r", encoding="utf-8") as f:
    for raw in f:
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        try:
            req = Requirement(line)
        except Exception:
            continue
        if norm(req.name) == "cuda-pathfinder":
            pathfinder_req = req
            break

if pathfinder_req is None:
    raise SystemExit(0)

installed = None
for candidate in ("cuda-pathfinder", "cuda.pathfinder", "cuda_pathfinder"):
    try:
        installed = version(candidate)
        break
    except PackageNotFoundError:
        pass

if installed is None:
    raise SystemExit(0)

if (not pathfinder_req.specifier) or pathfinder_req.specifier.contains(installed, prereleases=True):
    with open(constraint_file, "w", encoding="utf-8") as out:
        out.write(f"cuda-pathfinder=={installed}\n")
    print(f"[nvshmem4py] pinning cuda-pathfinder to installed version: {installed}")
PY

            if [[ -s "${constraint_file}" ]]; then
                pip_install python --no-cache-dir -c "${constraint_file}" -r "${req}"
            else
                pip_install python --no-cache-dir -r "${req}"
            fi

            rm -f "${constraint_file}"

            python -c 'import nvshmem.core as _; print("nvshmem4py ok")'
        fi
    fi

    rm -rf "${NVSHMEM_SRC_DIR}" "${NVSHMEM_BUILDDIR}"
}

build_nccl_tests() {
    init_hpc_stack_prefixes

    git clone --depth 1 --branch "v${NCCL_TESTS_VER}" https://github.com/NVIDIA/nccl-tests.git /tmp/nccl-tests
    pushd /tmp/nccl-tests
    MPI=1 MPI_HOME="${OMPI_PREFIX}" CUDA_HOME="${CUDA_DIR}" make -j"$(make_jobs)"
    install -d /usr/local/bin
    find build -maxdepth 1 -type f -executable -name '*_perf' -print -exec install -m 0755 {} /usr/local/bin/ \;
    popd
    rm -rf /tmp/nccl-tests
}

build_osu() {
    init_hpc_stack_prefixes

    curl -fsSL "http://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-${OSU_VERSION}.tar.gz" -o /tmp/osu.tar.gz
    tar --no-same-owner --no-same-permissions -C /tmp -xzf /tmp/osu.tar.gz
    pushd "/tmp/osu-micro-benchmarks-${OSU_VERSION}"
    CC="${OMPI_PREFIX}/bin/mpicc" \
    CFLAGS="-O3 -lcuda -lnvidia-ml" \
    ./configure \
        --prefix=/usr/local \
        --enable-cuda \
        --with-cuda-include="${CUDA_DIR}/include" \
        --with-cuda-libpath="${CUDA_DIR}/lib64"
    make -j"$(make_jobs)"
    make install
    popd
    rm -rf "/tmp/osu-micro-benchmarks-${OSU_VERSION}" /tmp/osu.tar.gz
    ldconfig
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    die "This file provides CUDA stack build functions; run install-alps-cuda-stack.sh instead."
fi
