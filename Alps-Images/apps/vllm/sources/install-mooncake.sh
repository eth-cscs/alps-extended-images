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
#                        CXI backend ships since v0.3.12). Fetched shallowly;
#                        a full fetch is the fallback for remotes that refuse
#                        fetching a bare commit.
#   LIBFABRIC_PREFIX     Alps libfabric prefix (default: /usr)
#   MOONCAKE_BUILD_JOBS  parallel build jobs (default: MAX_JOBS, else 32)
#   MOONCAKE_PATCH_DIR   patches applied on top of MOONCAKE_REF
#                        (default: /opt/alps/mooncake-patches)
set -euo pipefail

MOONCAKE_REPO="${MOONCAKE_REPO:-https://github.com/kvcache-ai/Mooncake.git}"
MOONCAKE_REF="${MOONCAKE_REF:-v0.3.13.post1}"
LIBFABRIC_PREFIX="${LIBFABRIC_PREFIX:-/usr}"
MOONCAKE_BUILD_JOBS="${MOONCAKE_BUILD_JOBS:-${MAX_JOBS:-32}}"
MOONCAKE_PATCH_DIR="${MOONCAKE_PATCH_DIR:-/opt/alps/mooncake-patches}"
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

# Temporary build-only filesystem changes, undone on every exit path so a
# failed step cannot leave them behind.
created_ibverbs_link=""
cuda_stub_soname_dir=""
cleanup_build_temporaries() {
    if [[ -n "${created_ibverbs_link}" ]]; then
        rm -f "${created_ibverbs_link}"
    fi
    if [[ -n "${cuda_stub_soname_dir}" ]]; then
        rm -rf "${cuda_stub_soname_dir}"
    fi
}
trap cleanup_build_temporaries EXIT

# Copy the one file a build-tree glob must match (e.g. engine.<EXT_SUFFIX>).
copy_single_match() {
    local dest="$1"
    shift
    if [[ $# -ne 1 || ! -f "$1" ]]; then
        die "expected exactly one build artifact for ${dest##*/}, got: $*"
    fi
    cp "$1" "${dest}"
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

[[ -f /usr/local/include/boost/version.hpp ]] \
    || die "Boost headers not found under /usr/local/include; expected from the Alps base image"

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

ibverbs_apt_build_deps=(
    libibverbs-dev
    libnl-3-dev
    libnl-route-3-dev
)

mooncake_apt_build_deps=(
    cmake
    git
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
# The CUDA base supplies an RDMA development package set at version 63
# alongside its DOCA/rdma-core runtime, while the configured Ubuntu
# repositories offer version 50. Keep the matching base packages across the
# vLLM build rather than trying to reinstall incompatible versions here.
if [[ "${accel}" == "cuda" ]]; then
    for pkg in "${ibverbs_apt_build_deps[@]}"; do
        dpkg-query -W -f='${db:Status-Abbrev}' "${pkg}" 2>/dev/null \
            | grep -q '^ii ' \
            || die "matching CUDA base package ${pkg} is not installed"
    done
fi

snapshot_apt_packages /tmp/mooncake-apt-before.txt
apt_get update
apt_get install -y --no-install-recommends \
    ca-certificates "${ibverbs_apt_build_deps[@]}" "${mooncake_apt_build_deps[@]}"
rm -rf /var/lib/apt/lists/*

# The NGC base images carry an NVIDIA/DOCA libibverbs-dev whose unversioned
# libibverbs.so linker symlink is missing, so "-libverbs" fails to link (the
# ROCm images use Ubuntu's package and are unaffected). Provide the symlink
# for the build only; the installed modules reference the libibverbs.so.1
# soname at runtime.
ibverbs_libdir="/usr/lib/$(gcc -print-multiarch)"
if [[ ! -e "${ibverbs_libdir}/libibverbs.so" ]]; then
    [[ -e "${ibverbs_libdir}/libibverbs.so.1" ]] \
        || die "libibverbs.so.1 not found under ${ibverbs_libdir}"
    ln -sf libibverbs.so.1 "${ibverbs_libdir}/libibverbs.so"
    created_ibverbs_link="${ibverbs_libdir}/libibverbs.so"
    echo "INFO: created missing linker symlink ${created_ibverbs_link}"
fi

rm -rf "${src_dir}"
git init -q "${src_dir}"
git -C "${src_dir}" remote add origin "${MOONCAKE_REPO}"
if git -C "${src_dir}" fetch -q --depth 1 origin "${MOONCAKE_REF}"; then
    git -C "${src_dir}" checkout -q FETCH_HEAD
else
    echo "INFO: shallow fetch of ${MOONCAKE_REF} failed; fetching full history"
    git -C "${src_dir}" fetch -q --tags origin
    git -C "${src_dir}" checkout -q "${MOONCAKE_REF}"
fi
git -C "${src_dir}" submodule update --init --recursive

[[ -d "${MOONCAKE_PATCH_DIR}" ]] \
    || die "missing ${MOONCAKE_PATCH_DIR}; the app Containerfile must COPY it"
shopt -s nullglob
mooncake_patch_files=("${MOONCAKE_PATCH_DIR}"/*.patch)
shopt -u nullglob
for patch_file in "${mooncake_patch_files[@]}"; do
    echo "Applying Mooncake patch: ${patch_file}"
    git -C "${src_dir}" apply --check "${patch_file}"
    git -C "${src_dir}" apply --verbose "${patch_file}"
done

cmake -S "${src_dir}" -B "${build_dir}" "${cmake_args[@]}"
cmake --build "${build_dir}" -j"${MOONCAKE_BUILD_JOBS}"
if [[ -n "${created_ibverbs_link}" ]]; then
    rm -f "${created_ibverbs_link}"
    created_ibverbs_link=""
fi

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
copy_single_match "${wheel_pkg_dir}/engine.so" "${build_dir}"/mooncake-integration/engine.*.so
copy_single_match "${wheel_pkg_dir}/store.so" "${build_dir}"/mooncake-integration/store.*.so
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

# Assert patch 0001 was compiled into store.so.
grep -aq "Topology discovery complete for CXI" "${wheel_pkg_dir}/store.so" \
    || die "store.so lacks the CXI topology-discovery patch (${MOONCAKE_PATCH_DIR}/0001-*)"

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
APT_CLEANUP_HOLD_PACKAGES="libibverbs1 ibverbs-providers libnl-3-200 libnl-route-3-200" \
    cleanup_apt_build_deps "${ibverbs_apt_build_deps[@]}"
for pkg in "${ibverbs_apt_build_deps[@]}"; do
    if dpkg-query -W -f='${db:Status-Abbrev}' "${pkg}" 2>/dev/null \
        | grep -q '^ii '; then
        die "${pkg} remains installed after Mooncake build"
    fi
done
ldconfig

# Final verification against the post-cleanup image content.
# On CUDA, engine.so links libcuda.so.1, which only the NVIDIA container
# runtime injects; the toolkit stubs ship it as libcuda.so only. Expose the
# stub under its soname for this import alone, so it never lands in the image.
import_ld_path="${LD_LIBRARY_PATH:-}"
if [[ "${accel}" == "cuda" ]]; then
    cuda_stub_soname_dir="$(mktemp -d)"
    ln -s "${cuda_stubs}/libcuda.so" "${cuda_stub_soname_dir}/libcuda.so.1"
    import_ld_path="${cuda_stub_soname_dir}${import_ld_path:+:${import_ld_path}}"
fi
LD_LIBRARY_PATH="${import_ld_path}" "${python_bin}" -c \
    'from mooncake.engine import TransferEngine; from mooncake.store import MooncakeDistributedStore; print("mooncake imports ok")'
if [[ -n "${cuda_stub_soname_dir}" ]]; then
    rm -rf "${cuda_stub_soname_dir}"
    cuda_stub_soname_dir=""
fi

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
