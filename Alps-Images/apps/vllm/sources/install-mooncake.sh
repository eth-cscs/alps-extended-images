#!/usr/bin/env bash
# Install the Mooncake transfer engine with the HPE Slingshot (CXI) transport
# into a vLLM app image.
#
# Mooncake's CXI backend (build flag USE_CXI, kvcache-ai/Mooncake#2535) is a
# libfabric transport. It is built against the libfabric installed by the
# Alps base image so that Mooncake and the in-process aws-ofi-nccl NCCL
# plugin share a single libfabric instance. Bundling a second libfabric copy
# would let whichever instance initializes first claim the CXI devices and
# leave the other with an empty provider list (see the EFA packaging notes
# in Mooncake's scripts/build_wheel.sh).
#
# The Python side is installed from Mooncake's own mooncake-wheel packaging
# (pip name: mooncake-transfer-engine), which provides the "mooncake.engine"
# and "mooncake.store" modules imported by vLLM's mooncake KV connectors.
#
# Usage: install-mooncake.sh {cuda|rocm}
#
# Environment overrides:
#   MOONCAKE_REPO        git remote (default: upstream GitHub)
#   MOONCAKE_REF         pinned tag or commit (default: v0.3.13.post1; the
#                        CXI backend ships since v0.3.12)
#   LIBFABRIC_PREFIX     Alps libfabric prefix (default: /usr)
#   MOONCAKE_BUILD_JOBS  parallel build jobs (default: 32)
set -euo pipefail

MOONCAKE_REPO="${MOONCAKE_REPO:-https://github.com/kvcache-ai/Mooncake.git}"
MOONCAKE_REF="${MOONCAKE_REF:-v0.3.13.post1}"
LIBFABRIC_PREFIX="${LIBFABRIC_PREFIX:-/usr}"
MOONCAKE_BUILD_JOBS="${MOONCAKE_BUILD_JOBS:-32}"
ALPS_PACKAGE_HELPERS="${ALPS_PACKAGE_HELPERS:-/opt/alps/package-helpers.sh}"

accel="${1:-}"
case "${accel}" in
    cuda|rocm) ;;
    *) echo "ERROR: usage: install-mooncake.sh {cuda|rocm}" >&2; exit 1 ;;
esac

# shellcheck source=/dev/null
source "${ALPS_PACKAGE_HELPERS}"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

python_bin="$(command -v python || command -v python3)"
[[ -n "${python_bin}" ]] || die "no python interpreter found on PATH"

src_dir="/tmp/mooncake-src"
build_dir="${src_dir}/build"
wheel_dir="${src_dir}/mooncake-wheel"
wheel_pkg_dir="${wheel_dir}/mooncake"

# Locate the libfabric installed by the Alps base image and pin the build to
# it explicitly instead of relying on CMake's default search.
libfabric_include="${LIBFABRIC_PREFIX}/include"
[[ -f "${libfabric_include}/rdma/fabric.h" ]] \
    || die "libfabric headers not found under ${libfabric_include}"
libfabric_lib=""
for candidate in \
    "${LIBFABRIC_PREFIX}/lib/libfabric.so" \
    "${LIBFABRIC_PREFIX}/lib64/libfabric.so" \
    "${LIBFABRIC_PREFIX}/lib/$(uname -m)-linux-gnu/libfabric.so"; do
    if [[ -e "${candidate}" ]]; then
        libfabric_lib="${candidate}"
        break
    fi
done
[[ -n "${libfabric_lib}" ]] \
    || die "libfabric library not found under ${LIBFABRIC_PREFIX}"

echo "INFO: building Mooncake ${MOONCAKE_REF} (${accel}) against ${libfabric_lib}"

cmake_args=(
    -G Ninja
    -DCMAKE_BUILD_TYPE=Release
    -DUSE_CXI=ON
    -DLIBFABRIC_INCLUDE_DIR="${libfabric_include}"
    -DLIBFABRIC_LIBRARY="${libfabric_lib}"
    -DWITH_EP=OFF
    -DWITH_STORE_RUST=OFF
    -DBUILD_UNIT_TESTS=OFF
    -DPython3_EXECUTABLE="${python_bin}"
)

case "${accel}" in
    cuda)
        # Mooncake's USE_CUDA path hardcodes /usr/local/cuda for headers and
        # libraries, matching the NGC base images.
        cuda_dir="/usr/local/cuda"
        [[ -d "${cuda_dir}" ]] || die "CUDA toolkit not found at ${cuda_dir}"
        cuda_stubs="${cuda_dir}/lib64/stubs"
        [[ -d "${cuda_stubs}" ]] || die "CUDA stubs not found at ${cuda_stubs}"
        export PATH="${cuda_dir}/bin:${PATH}"
        export LD_LIBRARY_PATH="${cuda_stubs}:${LD_LIBRARY_PATH:-}"
        export LIBRARY_PATH="${cuda_stubs}:${LIBRARY_PATH:-}"
        cmake_args+=(
            -DUSE_CUDA=ON
            -DCMAKE_EXE_LINKER_FLAGS="-L${cuda_stubs}"
        )
        ;;
    rocm)
        # Reuse the ROCm SDK environment recorded by the Alps ROCm base image
        # (hipify-perl on PATH, HIP CMake packages via CMAKE_PREFIX_PATH).
        rocm_env_file="/opt/alps/env/alps-rocm-build.env"
        [[ -f "${rocm_env_file}" ]] || die "missing ${rocm_env_file}; not an Alps ROCm base image?"
        # shellcheck source=/dev/null
        source "${rocm_env_file}"
        # shellcheck source=/dev/null
        source /opt/alps/install-alps-hpc-stack.sh
        # shellcheck source=/dev/null
        source /opt/alps/install-alps-rocm-components.sh
        load_rocm_sdk_env
        export PATH="${ROCM_BUILD_PREFIX}/bin:${ROCM_BUILD_PREFIX}/llvm/bin:${PATH}"
        rocm_libdir="${ROCM_BUILD_PREFIX}/lib64"
        [[ -d "${rocm_libdir}" ]] || rocm_libdir="${ROCM_BUILD_PREFIX}/lib"
        export LD_LIBRARY_PATH="${rocm_libdir}:${LD_LIBRARY_PATH:-}"
        cmake_args+=(
            -DUSE_HIP=ON
            -DUSE_CUDA=OFF
        )
        ;;
esac

# Build dependencies are removed again after the wheel is installed. Runtime
# libraries that the installed Mooncake modules link against are marked
# manually installed so they survive cleanup_new_apt_build_deps' autoremove.
# libibverbs-dev stays installed: the base images keep verbs headers around
# (see APT_CLEANUP_HOLD_PACKAGES in package-helpers.sh) and Mooncake's RDMA
# transport links against libibverbs.
mooncake_apt_build_deps=(
    cmake
    git
    libboost-dev
    libcurl4-openssl-dev
    libgflags-dev
    libgoogle-glog-dev
    libjsoncpp-dev
    libmsgpack-dev
    libnuma-dev
    libunwind-dev
    liburing-dev
    libxxhash-dev
    libyaml-cpp-dev
    libzstd-dev
    ninja-build
    patchelf
    pkg-config
    python3-dev
)
snapshot_apt_packages /tmp/mooncake-apt-before.txt
apt_get update
apt_get install -y --no-install-recommends \
    ca-certificates libibverbs-dev "${mooncake_apt_build_deps[@]}"
rm -rf /var/lib/apt/lists/*

rm -rf "${src_dir}"
git clone --recursive "${MOONCAKE_REPO}" "${src_dir}"
git -C "${src_dir}" checkout -q "${MOONCAKE_REF}"
git -C "${src_dir}" submodule update --init --recursive

cmake -S "${src_dir}" -B "${build_dir}" "${cmake_args[@]}"
cmake --build "${build_dir}" -j"${MOONCAKE_BUILD_JOBS}"

# USE_CXI is cached as UNINITIALIZED (no option() declaration); the cxi
# object files prove the flag actually took effect.
mooncake_use_cxi="$(grep -E '^USE_CXI:' "${build_dir}/CMakeCache.txt" \
    | head -n 1 | cut -d= -f2- || true)"
case "${mooncake_use_cxi}" in
    ON|on|TRUE|true|YES|yes|1) ;;
    *) die "Mooncake was not configured with USE_CXI=ON (CMakeCache: ${mooncake_use_cxi:-unset})" ;;
esac
cxi_object_count="$(find "${build_dir}/mooncake-transfer-engine/src/transport/cxi_transport" \
    -name '*.o' 2>/dev/null | wc -l)"
[[ "${cxi_object_count}" -gt 0 ]] \
    || die "cxi_transport produced no object files; USE_CXI did not take effect"
echo "INFO: cxi_transport built (${cxi_object_count} objects)"

# Stage the artifacts that upstream's scripts/build_wheel.sh ships in its
# wheels. Mooncake links statically by default, so engine.so and store.so are
# self-contained apart from system libraries and libasio.so.
cp "${build_dir}"/mooncake-integration/engine.*.so "${wheel_pkg_dir}/engine.so"
cp "${build_dir}"/mooncake-integration/store.*.so "${wheel_pkg_dir}/store.so"
cp "${build_dir}/mooncake-common/libasio.so" "${wheel_pkg_dir}/libasio.so"
cp "${src_dir}/mooncake-integration/fabric_allocator_utils.py" \
    "${wheel_pkg_dir}/fabric_allocator_utils.py"
cp "${src_dir}/mooncake-integration/shared_segment.py" "${wheel_pkg_dir}/shared_segment.py"
cp "${src_dir}/mooncake-integration/store/async_store.py" "${wheel_pkg_dir}/async_store.py"
cp "${build_dir}/mooncake-store/src/mooncake_master" "${wheel_pkg_dir}/mooncake_master"
cp "${build_dir}/mooncake-store/src/mooncake_client" "${wheel_pkg_dir}/mooncake_client"
cp "${build_dir}/mooncake-transfer-engine/example/transfer_engine_bench" \
    "${wheel_pkg_dir}/transfer_engine_bench"
if [[ -f "${build_dir}/mooncake-transfer-engine/nvlink-allocator/nvlink_allocator.so" ]]; then
    cp "${build_dir}/mooncake-transfer-engine/nvlink-allocator/nvlink_allocator.so" \
        "${wheel_pkg_dir}/nvlink_allocator.so"
    cp "${src_dir}/mooncake-integration/allocator.py" "${wheel_pkg_dir}/allocator.py"
fi

# The build-tree artifacts resolve wheel-internal libraries (libasio.so) via
# build paths; pin them to $ORIGIN so they resolve next to the installed
# modules. libfabric is deliberately not bundled (see header comment).
# shellcheck disable=SC2016  # $ORIGIN must stay literal
patchelf --force-rpath --set-rpath '$ORIGIN' \
    "${wheel_pkg_dir}/engine.so" \
    "${wheel_pkg_dir}/store.so" \
    "${wheel_pkg_dir}/libasio.so" \
    "${wheel_pkg_dir}/mooncake_master" \
    "${wheel_pkg_dir}/mooncake_client" \
    "${wheel_pkg_dir}/transfer_engine_bench"
if [[ -f "${wheel_pkg_dir}/nvlink_allocator.so" ]]; then
    # shellcheck disable=SC2016  # $ORIGIN must stay literal
    patchelf --force-rpath --set-rpath '$ORIGIN' "${wheel_pkg_dir}/nvlink_allocator.so"
fi

# setuptools and numpy are already installed by the vLLM build; wheel is the
# only build requirement that may be missing.
pip_install "${python_bin}" --no-cache-dir wheel

# mooncake-wheel's pyproject.toml references a README.md that is materialized
# next to it only for the duration of the build (upstream does the same).
cp "${src_dir}/README.md" "${wheel_dir}/README.md"
pip_install "${python_bin}" --no-cache-dir --no-build-isolation "${wheel_dir}"
rm -f "${wheel_dir}/README.md"

mooncake_pkg_dir="$("${python_bin}" -c 'import mooncake; print(mooncake.__path__[0])')"
[[ -n "${mooncake_pkg_dir}" ]] || die "could not locate installed mooncake package"

# The engine must link the base image's libfabric: that is the whole point
# of this installer (single shared libfabric instance with the NCCL plugin).
libfabric_needed="$({ ldd "${mooncake_pkg_dir}/engine.so" 2>/dev/null || true; } \
    | awk '/libfabric/ { print $1; exit }')"
[[ -n "${libfabric_needed}" ]] \
    || die "mooncake engine.so does not link libfabric; USE_CXI build is broken"
echo "INFO: mooncake engine links ${libfabric_needed}"

# Mark dpkg-owned runtime libraries of the installed modules as manual so the
# autoremove below keeps them. Resolve owners by soname, not by ldd's
# resolved path (/lib vs /usr/lib trips dpkg -S on merged-/usr systems).
mapfile -t mooncake_elf_files < <(
    find "${mooncake_pkg_dir}" -type f \
        \( -name '*.so' -o -name 'mooncake_master' -o -name 'mooncake_client' \
           -o -name 'transfer_engine_bench' \) | sort
)
runtime_packages="$(mktemp)"
for elf_file in "${mooncake_elf_files[@]}"; do
    { ldd "${elf_file}" 2>/dev/null || true; } \
        | awk '/=> \// { print $1 } /^\/.*\.so/ { n = split($1, p, "/"); print p[n] }' \
        | sort -u \
        | while IFS= read -r soname; do
            dpkg -S "${soname}" 2>/dev/null | head -n 1 | cut -d: -f1 || true
        done >> "${runtime_packages}"
done
sort -u "${runtime_packages}" -o "${runtime_packages}"
[[ -s "${runtime_packages}" ]] \
    || die "could not resolve runtime library packages for mooncake"
mapfile -t runtime_pkg_list < "${runtime_packages}"
echo "INFO: keeping mooncake runtime library packages: ${runtime_pkg_list[*]}"
apt_mark manual "${runtime_pkg_list[@]}"
rm -f "${runtime_packages}"

cleanup_new_apt_build_deps /tmp/mooncake-apt-before.txt "${mooncake_apt_build_deps[@]}"
ldconfig

# Final verification against the post-cleanup image content.
"${python_bin}" -c \
    'from mooncake.engine import TransferEngine; from mooncake.store import MooncakeDistributedStore; print("mooncake imports ok")'

all_missing=""
for elf_file in "${mooncake_elf_files[@]}"; do
    all_missing+="$({ ldd "${elf_file}" 2>/dev/null || true; } | awk '/not found/ { print }')"$'\n'
done
ignored_missing="$(printf '%s\n' "${all_missing}" | awk '$1 == "libcuda.so.1" { print }')"
missing="$(printf '%s\n' "${all_missing}" | awk 'NF && $1 != "libcuda.so.1" { print }')"
if [[ -n "${ignored_missing}" ]]; then
    echo "INFO: ignoring build-time unresolved host CUDA driver dependencies:"
    echo "${ignored_missing}"
fi
if [[ -n "${missing}" ]]; then
    echo "FATAL: mooncake native modules have unresolved dependencies:" >&2
    echo "${missing}" >&2
    exit 1
fi

rm -rf "${src_dir}" /tmp/mooncake-apt-before.txt
echo "OK: Mooncake ${MOONCAKE_REF} (${accel}, CXI) installed"
