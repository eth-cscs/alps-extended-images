#!/usr/bin/env bash

bootstrap_rocm_sdk() {
    : "${ROCM_VERSION:?ROCM_VERSION must be set}"
    : "${ROCM_PYPI_INDEX_URL:?ROCM_PYPI_INDEX_URL must be set to the ROCm wheel index}"
    : "${ROCM_PYTHON:=/opt/venv/bin/python}"
    : "${ROCM_SDK:=/opt/venv/bin/rocm-sdk}"

    [[ -x "${ROCM_PYTHON}" ]] || die "ROCm Python not found: ${ROCM_PYTHON}"
    local sdk_packages=(
        "rocm==${ROCM_VERSION}"
        "rocm-sdk-core==${ROCM_VERSION}"
        "rocm-sdk-libraries==${ROCM_VERSION}"
        "rocm-sdk-devel==${ROCM_VERSION}"
    )
    local target
    for target in ${RCCL_GPU_TARGETS//;/ }; do
        [[ -n "${target}" ]] || continue
        sdk_packages+=("rocm-sdk-device-${target}==${ROCM_VERSION}")
    done

    "${ROCM_PYTHON}" -m pip install --no-cache-dir --index-url "${ROCM_PYPI_INDEX_URL}" --no-deps "${sdk_packages[@]}"
    [[ -x "${ROCM_SDK}" ]] || die "rocm-sdk not found after installing rocm-sdk-devel: ${ROCM_SDK}"

    "${ROCM_SDK}" init

    ROCM_SDK_ROOT="$("${ROCM_SDK}" path --root)"
    ROCM_SDK_BIN="$("${ROCM_SDK}" path --bin)"
    ROCM_SDK_CMAKE="$("${ROCM_SDK}" path --cmake)"
    ROCM_CORE_DIR="$(ROCM_DIST_NAME=rocm-sdk-core "${ROCM_PYTHON}" - <<'PY'
import os
from importlib.metadata import distribution
print(distribution(os.environ["ROCM_DIST_NAME"]).locate_file(""))
PY
)"
    ROCM_CORE_PREFIX="$(ROCM_DIST_NAME=rocm-sdk-core ROCM_PACKAGE_DIR=_rocm_sdk_core "${ROCM_PYTHON}" - <<'PY'
import os
from importlib.metadata import distribution
print(distribution(os.environ["ROCM_DIST_NAME"]).locate_file(os.environ["ROCM_PACKAGE_DIR"]))
PY
)"
    ROCM_LIBRARIES_DIR="$(ROCM_DIST_NAME=rocm-sdk-libraries "${ROCM_PYTHON}" - <<'PY'
import os
from importlib.metadata import distribution
print(distribution(os.environ["ROCM_DIST_NAME"]).locate_file(""))
PY
)"
    ROCM_LIBRARIES_PREFIX="$(ROCM_DIST_NAME=rocm-sdk-libraries ROCM_PACKAGE_DIR=_rocm_sdk_libraries "${ROCM_PYTHON}" - <<'PY'
import os
from importlib.metadata import distribution
print(distribution(os.environ["ROCM_DIST_NAME"]).locate_file(os.environ["ROCM_PACKAGE_DIR"]))
PY
)"
    ROCM_DEVEL_DIR="$(ROCM_DIST_NAME=rocm-sdk-devel "${ROCM_PYTHON}" - <<'PY'
import os
from importlib.metadata import distribution
print(distribution(os.environ["ROCM_DIST_NAME"]).locate_file(""))
PY
)"
    ROCM_DEVEL_PREFIX="$(ROCM_DIST_NAME=rocm-sdk-devel ROCM_PACKAGE_DIR=_rocm_sdk_devel "${ROCM_PYTHON}" - <<'PY'
import os
from importlib.metadata import distribution
print(distribution(os.environ["ROCM_DIST_NAME"]).locate_file(os.environ["ROCM_PACKAGE_DIR"]))
PY
)"

    [[ -d "${ROCM_SDK_ROOT}" ]] || die "ROCM_SDK_ROOT is not a directory: ${ROCM_SDK_ROOT}"
    [[ -d "${ROCM_SDK_BIN}" ]] || die "ROCM_SDK_BIN is not a directory: ${ROCM_SDK_BIN}"
    [[ -d "${ROCM_SDK_CMAKE}" ]] || die "ROCM_SDK_CMAKE is not a directory: ${ROCM_SDK_CMAKE}"
    [[ -d "${ROCM_CORE_DIR}" ]] || die "ROCM_CORE_DIR is not a directory: ${ROCM_CORE_DIR}"
    [[ -d "${ROCM_CORE_PREFIX}" ]] || die "ROCM_CORE_PREFIX is not a directory: ${ROCM_CORE_PREFIX}"
    [[ -d "${ROCM_LIBRARIES_DIR}" ]] || die "ROCM_LIBRARIES_DIR is not a directory: ${ROCM_LIBRARIES_DIR}"
    [[ -d "${ROCM_LIBRARIES_PREFIX}" ]] || die "ROCM_LIBRARIES_PREFIX is not a directory: ${ROCM_LIBRARIES_PREFIX}"
    [[ -d "${ROCM_DEVEL_DIR}" ]] || die "ROCM_DEVEL_DIR is not a directory: ${ROCM_DEVEL_DIR}"
    [[ -d "${ROCM_DEVEL_PREFIX}" ]] || die "ROCM_DEVEL_PREFIX is not a directory: ${ROCM_DEVEL_PREFIX}"

    ROCM_BUILD_PREFIX="$(discover_rocm_build_prefix)"
    [[ -n "${ROCM_BUILD_PREFIX}" ]] || die "Could not find a ROCm prefix with HIP/HSA headers and runtime libraries"
    echo "Using ROCm build prefix: ${ROCM_BUILD_PREFIX}"

    local targets
    targets="$("${ROCM_SDK}" targets)"
    [[ "${targets}" == *gfx90a* ]] || die "rocm-sdk targets does not include gfx90a: ${targets}"
    [[ "${targets}" == *gfx942* ]] || die "rocm-sdk targets does not include gfx942: ${targets}"
    "${ROCM_SDK}" version

    export ROCM_PYTHON ROCM_SDK ROCM_SDK_ROOT ROCM_SDK_BIN ROCM_SDK_CMAKE
    export ROCM_CORE_DIR ROCM_CORE_PREFIX ROCM_LIBRARIES_DIR ROCM_LIBRARIES_PREFIX ROCM_DEVEL_DIR ROCM_DEVEL_PREFIX ROCM_BUILD_PREFIX
    export ROCM_HOME="${ROCM_BUILD_PREFIX}"
    export ROCM_PATH="${ROCM_BUILD_PREFIX}"
    export HIP_PATH="${ROCM_BUILD_PREFIX}"
    export PATH="${ROCM_SDK_BIN}:${PATH}"
    export CMAKE_PREFIX_PATH="${ROCM_SDK_CMAKE}:${ROCM_SDK_ROOT}:${ROCM_BUILD_PREFIX}:${ROCM_CORE_PREFIX}:${ROCM_LIBRARIES_PREFIX}:${ROCM_DEVEL_PREFIX}:${ROCM_CORE_DIR}:${ROCM_LIBRARIES_DIR}:${ROCM_DEVEL_DIR}:${CMAKE_PREFIX_PATH:-}"

    register_rocm_sdk_ldconfig
    install_amdsmi_python
    link_amdsmi_package_library
    patch_torch_rocm_amdsmi_device_count
    smoke_check_amdsmi_python
    persist_rocm_sdk_env
    record_alps_version_var ROCM_VERSION "${ROCM_VERSION}"
}

install_amdsmi_python() {
    local candidate amdsmi_src=""

    for candidate in \
        "${ROCM_CORE_PREFIX:-}/share/amd_smi" \
        "${ROCM_CORE_DIR:-}/_rocm_sdk_core/share/amd_smi" \
        "${ROCM_SDK_ROOT:-}/share/amd_smi"; do
        [[ -d "${candidate}" ]] || continue
        amdsmi_src="${candidate}"
        break
    done

    [[ -n "${amdsmi_src}" ]] || die "Could not find AMD SMI Python sources in ROCm Core SDK"
    "${ROCM_PYTHON}" -m pip install --no-cache-dir --no-deps "${amdsmi_src}"
    check_amdsmi_python_version
}

check_amdsmi_python_version() {
    ROCM_EXPECTED_VERSION="${ROCM_VERSION}" \
        ROCM_EXPECTED_COMMIT="${ROCM_SYSTEMS_COMMIT:-}" \
        "${ROCM_PYTHON}" - <<'PY'
import os
from importlib.metadata import PackageNotFoundError, version

for dist_name in ("amdsmi", "amd-smi"):
    try:
        installed = version(dist_name)
        break
    except PackageNotFoundError:
        continue
else:
    raise SystemExit("amdsmi distribution was not installed")

expected_commit = os.environ.get("ROCM_EXPECTED_COMMIT", "")
if expected_commit:
    expected_local = expected_commit[:8]
    local_version = installed.split("+", 1)[1] if "+" in installed else ""
    if local_version != expected_local:
        raise SystemExit(
            f"amdsmi version {installed} does not match ROCm Systems commit "
            f"{expected_commit}"
        )

print(
    f"amdsmi Python package version: {installed} "
    f"(ROCm SDK {os.environ['ROCM_EXPECTED_VERSION']})"
)
PY
}

smoke_check_amdsmi_python() {
    "${ROCM_PYTHON}" - <<'PY'
import amdsmi

print("amdsmi import ok")
try:
    amdsmi.amdsmi_init()
except Exception as exc:
    print(f"amdsmi init skipped without visible ROCm devices: {exc}")
else:
    try:
        handles = amdsmi.amdsmi_get_processor_handles()
        print(f"amdsmi device count: {len(handles)}")
    finally:
        shutdown = getattr(amdsmi, "amdsmi_shut_down", None)
        if shutdown is not None:
            shutdown()
PY
}

register_amdsmi_ldconfig() {
    local conf="/etc/ld.so.conf.d/99-alps-rocm-amdsmi.conf"
    local amdsmi_lib=""
    local candidate libdir

    while IFS= read -r candidate; do
        [[ -n "${candidate}" ]] || continue
        for libdir in "${candidate}" "${candidate}/lib" "${candidate}/lib64"; do
            [[ -d "${libdir}" ]] || continue
            if compgen -G "${libdir}/libamd_smi.so*" >/dev/null; then
                amdsmi_lib="${libdir}"
                break 2
            fi
        done
    done < <(rocm_sdk_prefix_candidates)

    # Fallback: search the Python environment for the library.
    if [[ -z "${amdsmi_lib}" ]]; then
        local site_pkgs
        site_pkgs="$("${ROCM_PYTHON}" -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null || true)"
        if [[ -n "${site_pkgs}" && -d "${site_pkgs}" ]]; then
            amdsmi_lib="$(find "${site_pkgs}" -maxdepth 3 \
                \( -type f -o -type l \) -name 'libamd_smi.so*' -printf '%h\n' 2>/dev/null | head -n1 || true)"
        fi
    fi

    [[ -n "${amdsmi_lib}" ]] || die "libamd_smi.so not found after installing amdsmi"

    if [[ ! -f "${conf}" ]] || ! grep -Fxq "${amdsmi_lib}" "${conf}"; then
        {
            [[ -s "${conf}" ]] && printf '\n'
            printf '%s\n' "${amdsmi_lib}"
        } >> "${conf}"
    fi

    ldconfig
    echo "Registered amdsmi library directory: ${amdsmi_lib}"
}

link_amdsmi_package_library() {
    local amdsmi_pkg_dir amdsmi_lib="" wrapper preload_file
    local candidate lib sysdeps_dir sysdeps_dirs=()

    amdsmi_pkg_dir="$(${ROCM_PYTHON} - <<'PY'
import importlib.util

spec = importlib.util.find_spec("amdsmi")
if spec is None or not spec.submodule_search_locations:
    raise SystemExit("amdsmi package was not found")
print(next(iter(spec.submodule_search_locations)))
PY
)"
    [[ -d "${amdsmi_pkg_dir}" ]] || die "amdsmi package directory not found: ${amdsmi_pkg_dir}"

    while IFS= read -r candidate; do
        [[ -n "${candidate}" ]] || continue
        lib="$(find "${candidate}" -maxdepth 2 \
            \( -type f -o -type l \) -name 'libamd_smi.so*' -print -quit 2>/dev/null || true)"
        if [[ -n "${lib}" && -z "${amdsmi_lib}" ]]; then
            amdsmi_lib="${lib}"
        fi
        for sysdeps_dir in "${candidate}/lib/rocm_sysdeps/lib" "${candidate}/lib64/rocm_sysdeps/lib"; do
            [[ -d "${sysdeps_dir}" ]] || continue
            compgen -G "${sysdeps_dir}/librocm_sysdeps_*.so*" >/dev/null || continue
            sysdeps_dirs+=("${sysdeps_dir}")
        done
    done < <(rocm_sdk_prefix_candidates)
    [[ -n "${amdsmi_lib}" ]] || die "libamd_smi.so* not found after installing amdsmi"
    [[ "${#sysdeps_dirs[@]}" -gt 0 ]] || die "ROCm sysdeps libraries not found for amdsmi"

    ln -sf "${amdsmi_lib}" "${amdsmi_pkg_dir}/libamd_smi.so"

    preload_file="${amdsmi_pkg_dir}/_alps_amdsmi_preload.py"
    {
        printf 'import ctypes\n'
        printf 'from pathlib import Path\n\n'
        printf '_SYSDEPS_DIRS = (\n'
        for sysdeps_dir in "${sysdeps_dirs[@]}"; do
            printf '    "%s",\n' "${sysdeps_dir}"
        done
        printf ')\n'
        printf '_REQUIRED = (\n'
        printf '    "librocm_sysdeps_nl_3.so.200",\n'
        printf '    "librocm_sysdeps_mnl.so.0",\n'
        printf '    "librocm_sysdeps_nl_genl_3.so.200",\n'
        printf ')\n\n'
        printf 'def preload_amdsmi_dependencies():\n'
        printf '    for name in _REQUIRED:\n'
        printf '        for directory in _SYSDEPS_DIRS:\n'
        printf '            path = Path(directory) / name\n'
        printf '            if path.exists():\n'
        printf '                ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)\n'
        printf '                break\n'
    } > "${preload_file}"

    wrapper="${amdsmi_pkg_dir}/amdsmi_wrapper.py"
    [[ -f "${wrapper}" ]] || die "amdsmi wrapper not found: ${wrapper}"
    "${ROCM_PYTHON}" - "${wrapper}" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text()
if "_alps_amdsmi_preload" not in text:
    marker = "import ctypes\n"
    replacement = (
        marker
        + "from ._alps_amdsmi_preload import preload_amdsmi_dependencies\n"
        + "preload_amdsmi_dependencies()\n"
    )
    if marker not in text:
        raise SystemExit(f"could not patch {path}: import marker not found")
    path.write_text(text.replace(marker, replacement, 1))
PY

    echo "Linked amdsmi package library: ${amdsmi_pkg_dir}/libamd_smi.so -> ${amdsmi_lib}"
    echo "Patched amdsmi dependency preload: ${preload_file}"
}


patch_torch_rocm_amdsmi_device_count() {
    "${ROCM_PYTHON}" - <<'PY'
import site
from pathlib import Path

roots = [Path(p) for p in site.getsitepackages()]
user_site = site.getusersitepackages()
if user_site:
    roots.append(Path(user_site))

for root in roots:
    path = root / "torch" / "cuda" / "__init__.py"
    if path.exists():
        break
else:
    print("PyTorch CUDA module not found; skipping ROCm amdsmi device_count patch")
    raise SystemExit(0)

text = path.read_text()
patched = """            if raw_cnt < 0:
                return raw_cnt
            # On some containerized ROCm systems, amdsmi can initialize but see
            # zero GPUs while the HIP runtime sees the allocated devices.
            if raw_cnt == 0:
                return -1
"""
if patched in text:
    print(f"PyTorch ROCm amdsmi device_count fallback already patched: {path}")
    raise SystemExit(0)

old = """            if raw_cnt <= 0:
                return raw_cnt
"""
if old not in text:
    raise SystemExit(f"could not patch {path}: amdsmi raw-count marker not found")

path.write_text(text.replace(old, patched, 1))
print(f"Patched PyTorch ROCm amdsmi device_count fallback: {path}")
PY
}

persist_rocm_sdk_env() {
    install -d /opt/alps/env
    {
        printf 'export ROCM_PYTHON=%q\n' "${ROCM_PYTHON}"
        printf 'export ROCM_SDK=%q\n' "${ROCM_SDK}"
        printf 'export ROCM_SDK_ROOT=%q\n' "${ROCM_SDK_ROOT}"
        printf 'export ROCM_SDK_BIN=%q\n' "${ROCM_SDK_BIN}"
        printf 'export ROCM_SDK_CMAKE=%q\n' "${ROCM_SDK_CMAKE}"
        printf 'export ROCM_CORE_DIR=%q\n' "${ROCM_CORE_DIR}"
        printf 'export ROCM_CORE_PREFIX=%q\n' "${ROCM_CORE_PREFIX}"
        printf 'export ROCM_LIBRARIES_DIR=%q\n' "${ROCM_LIBRARIES_DIR}"
        printf 'export ROCM_LIBRARIES_PREFIX=%q\n' "${ROCM_LIBRARIES_PREFIX}"
        printf 'export ROCM_DEVEL_DIR=%q\n' "${ROCM_DEVEL_DIR}"
        printf 'export ROCM_DEVEL_PREFIX=%q\n' "${ROCM_DEVEL_PREFIX}"
        printf 'export ROCM_BUILD_PREFIX=%q\n' "${ROCM_BUILD_PREFIX}"
        if [[ -n "${RCCL_PREFIX:-}" ]]; then
            printf 'export RCCL_PREFIX=%q\n' "${RCCL_PREFIX}"
        fi
        if [[ -n "${RCCL_INCLUDE_DIR:-}" ]]; then
            printf 'export RCCL_INCLUDE_DIR=%q\n' "${RCCL_INCLUDE_DIR}"
        fi
        if [[ -n "${RCCL_LIB_DIR:-}" ]]; then
            printf 'export RCCL_LIB_DIR=%q\n' "${RCCL_LIB_DIR}"
        fi
    } > /opt/alps/env/alps-rocm-build.env
}

rocm_sdk_prefix_candidates() {
    local candidate seen=""

    for candidate in \
        "${ROCM_BUILD_PREFIX:-}" \
        "${ROCM_DEVEL_PREFIX:-}" \
        "${ROCM_CORE_PREFIX:-}" \
        "${ROCM_LIBRARIES_PREFIX:-}" \
        "${ROCM_SDK_ROOT:-}" \
        "${ROCM_DEVEL_DIR:-}/_rocm_sdk_devel" \
        "${ROCM_CORE_DIR:-}/_rocm_sdk_core" \
        "${ROCM_LIBRARIES_DIR:-}/_rocm_sdk_libraries"; do
        [[ -n "${candidate}" ]] || continue
        case " ${seen} " in
            *" ${candidate} "*) ;;
            *)
                seen+=" ${candidate}"
                printf '%s\n' "${candidate}"
                ;;
        esac
    done
}

rocm_sdk_ldconfig_dirs() {
    local candidate libdir seen=""

    while IFS= read -r candidate; do
        [[ -n "${candidate}" ]] || continue
        for libdir in "${candidate}" "${candidate}/lib" "${candidate}/lib64"; do
            [[ -d "${libdir}" ]] || continue
            compgen -G "${libdir}/*.so*" >/dev/null || continue
            case " ${seen} " in
                *" ${libdir} "*) ;;
                *)
                    seen+=" ${libdir}"
                    printf '%s\n' "${libdir}"
                    ;;
            esac
        done
    done < <(rocm_sdk_prefix_candidates)
}

register_rocm_sdk_ldconfig() {
    local conf="/etc/ld.so.conf.d/99-alps-rocm-sdk.conf"
    local dirs=() dir

    while IFS= read -r dir; do
        [[ -n "${dir}" ]] || continue
        dirs+=("${dir}")
    done < <(rocm_sdk_ldconfig_dirs)

    [[ "${#dirs[@]}" -gt 0 ]] || die "No ROCm SDK runtime library directories found"
    printf '%s\n' "${dirs[@]}" > "${conf}"
    ldconfig
}

load_rocm_sdk_env() {
    local env_file="/opt/alps/env/alps-rocm-build.env"
    [[ -f "${env_file}" ]] || die "Missing ROCm build environment; run install-alps-rocm-stack.sh bootstrap first"
    # shellcheck disable=SC1090
    source "${env_file}"

    [[ -d "${ROCM_BUILD_PREFIX}" ]] || die "ROCm build prefix is not a directory: ${ROCM_BUILD_PREFIX}"
    export ROCM_HOME="${ROCM_BUILD_PREFIX}"
    export ROCM_PATH="${ROCM_BUILD_PREFIX}"
    export HIP_PATH="${ROCM_BUILD_PREFIX}"
    export PATH="${ROCM_SDK_BIN}:${PATH}"
    export CMAKE_PREFIX_PATH="${ROCM_SDK_CMAKE}:${ROCM_SDK_ROOT}:${ROCM_BUILD_PREFIX}:${ROCM_CORE_PREFIX}:${ROCM_LIBRARIES_PREFIX}:${ROCM_DEVEL_PREFIX}:${ROCM_CORE_DIR}:${ROCM_LIBRARIES_DIR}:${ROCM_DEVEL_DIR}:${CMAKE_PREFIX_PATH:-}"
    if [[ -n "${RCCL_PREFIX:-}" ]]; then
        export CMAKE_PREFIX_PATH="${RCCL_PREFIX}:${RCCL_LIB_DIR:-}:${RCCL_INCLUDE_DIR:-}:${CMAKE_PREFIX_PATH}"
    fi
}

discover_rocm_build_prefix() {
    local candidate libdir
    while IFS= read -r candidate; do
        [[ -n "${candidate}" ]] || continue
        [[ -f "${candidate}/include/hip/hip_runtime_api.h" ]] || continue
        [[ -f "${candidate}/include/hip/hip_runtime.h" ]] || continue
        [[ -f "${candidate}/include/hip/hip_version.h" ]] || continue
        [[ -f "${candidate}/include/hsa/hsa.h" ]] || continue
        [[ -f "${candidate}/include/hsa/hsa_ext_amd.h" ]] || continue
        if [[ -d "${candidate}/lib64" ]]; then
            libdir="${candidate}/lib64"
        else
            libdir="${candidate}/lib"
        fi
        [[ -d "${libdir}" ]] || continue
        if [[ -e "${libdir}/libamdhip64.so" && -e "${libdir}/libhsa-runtime64.so" ]] \
            && { [[ -e "${libdir}/libhsakmt.so" ]] || [[ -e "${libdir}/libhsakmt.a" ]]; }; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done < <(rocm_sdk_prefix_candidates)
    return 1
}

clone_rocm_systems() {
    : "${ROCM_SYSTEMS_REPO:?ROCM_SYSTEMS_REPO must be set}"
    : "${ROCM_SYSTEMS_COMMIT:?ROCM_SYSTEMS_COMMIT must be set}"
    : "${ROCM_SYSTEMS_SRC_DIR:=/tmp/rocm-systems}"

    if [[ -d "${ROCM_SYSTEMS_SRC_DIR}/.git" ]]; then
        return 0
    fi

    git clone "${ROCM_SYSTEMS_REPO}" "${ROCM_SYSTEMS_SRC_DIR}"
    pushd "${ROCM_SYSTEMS_SRC_DIR}"
    git reset --hard "${ROCM_SYSTEMS_COMMIT}"
    git submodule update --init --recursive --depth=1 projects/rccl projects/rccl-tests
    popd
}

configure_rccl() {
    : "${ROCM_REBUILD_RCCL:=0}"

    case "${ROCM_REBUILD_RCCL}" in
        0)
            use_bundled_rccl
            ;;
        1)
            build_rccl
            replace_wheel_rccl
            ;;
        *)
            die "ROCM_REBUILD_RCCL must be 0 or 1, got: ${ROCM_REBUILD_RCCL}"
            ;;
    esac

    : "${RCCL_PREFIX:?RCCL_PREFIX was not configured}"
    persist_rocm_sdk_env
}

use_bundled_rccl() {
    local header="" include_root="" lib="" source_lib_dir="" link_src candidate

    for candidate in "${ROCM_BUILD_PREFIX:-}" "${ROCM_DEVEL_PREFIX:-}" "${ROCM_SDK_ROOT:-}" "${ROCM_LIBRARIES_PREFIX:-}"; do
        [[ -n "${candidate}" && -d "${candidate}/include" ]] || continue
        [[ -e "${candidate}/lib/librccl.so" ]] || continue
        if [[ -f "${candidate}/include/rccl/rccl.h" || -f "${candidate}/include/rccl.h" || -f "${candidate}/include/nccl.h" ]]; then
            RCCL_PREFIX="${candidate}"
            RCCL_INCLUDE_DIR="${candidate}/include"
            RCCL_LIB_DIR="${candidate}/lib"
            echo "Using bundled RCCL prefix: ${RCCL_PREFIX}"
            export RCCL_PREFIX RCCL_INCLUDE_DIR RCCL_LIB_DIR
            export CMAKE_PREFIX_PATH="${RCCL_PREFIX}:${RCCL_LIB_DIR}:${RCCL_INCLUDE_DIR}:${CMAKE_PREFIX_PATH}"
            cat > /etc/ld.so.conf.d/99-alps-rocm-rccl.conf <<EOF
${RCCL_LIB_DIR}
EOF
            record_alps_version_var RCCL_VERSION "${ROCM_VERSION}"
            record_alps_version_var RCCL_SOURCE "bundled"
            ldconfig
            return 0
        fi
    done

    header="$(find "${ROCM_SDK_ROOT}" "${ROCM_DEVEL_PREFIX}" "${ROCM_DEVEL_DIR}" "${ROCM_CORE_PREFIX}" "${ROCM_LIBRARIES_DIR}" \
        \( -type f -o -type l \) \( -name 'nccl.h' -o -name 'rccl.h' \) \
        -print -quit || true)"
    lib="$(find "${ROCM_SDK_ROOT}" "${ROCM_DEVEL_PREFIX}" "${ROCM_DEVEL_DIR}" "${ROCM_LIBRARIES_PREFIX}" "${ROCM_LIBRARIES_DIR}" \
        \( -type f -o -type l \) -name 'librccl.so*' \
        -print -quit || true)"

    [[ -n "${header}" ]] || die "No bundled RCCL header found under ROCm SDK roots"
    [[ -n "${lib}" ]] || die "No bundled RCCL library found under ROCm SDK roots"

    RCCL_PREFIX="/opt/alps/rocm/rccl-bundled"
    RCCL_INCLUDE_DIR="${RCCL_PREFIX}/include"
    RCCL_LIB_DIR="${RCCL_PREFIX}/lib"
    include_root="$(dirname "${header}")"
    if [[ "$(basename "${include_root}")" == "rccl" || "$(basename "${include_root}")" == "nccl_device" ]]; then
        include_root="$(dirname "${include_root}")"
    fi
    source_lib_dir="$(dirname "${lib}")"

    rm -rf "${RCCL_PREFIX}"
    install -d "${RCCL_LIB_DIR}" "$(dirname "${RCCL_INCLUDE_DIR}")"
    ln -s "${include_root}" "${RCCL_INCLUDE_DIR}"
    for link_src in "${source_lib_dir}"/librccl.so* "${ROCM_BUILD_PREFIX}"/lib/libamdhip64.so* "${ROCM_BUILD_PREFIX}"/lib/libhsa-runtime64.so* "${ROCM_BUILD_PREFIX}"/lib/libhsakmt.so* "${ROCM_BUILD_PREFIX}"/lib64/libamdhip64.so* "${ROCM_BUILD_PREFIX}"/lib64/libhsa-runtime64.so* "${ROCM_BUILD_PREFIX}"/lib64/libhsakmt.so*; do
        [[ -e "${link_src}" ]] || continue
        ln -sf "${link_src}" "${RCCL_LIB_DIR}/$(basename "${link_src}")"
    done
    compgen -G "${RCCL_LIB_DIR}/librccl.so*" >/dev/null || die "No RCCL libraries linked under ${RCCL_LIB_DIR}"
    echo "Using bundled RCCL compatibility prefix: ${RCCL_PREFIX}"

    export RCCL_PREFIX RCCL_INCLUDE_DIR RCCL_LIB_DIR
    export CMAKE_PREFIX_PATH="${RCCL_PREFIX}:${RCCL_LIB_DIR}:${RCCL_INCLUDE_DIR}:${CMAKE_PREFIX_PATH}"

    cat > /etc/ld.so.conf.d/99-alps-rocm-rccl.conf <<EOF
${RCCL_LIB_DIR}
EOF
    record_alps_version_var RCCL_VERSION "${ROCM_VERSION}"
    record_alps_version_var RCCL_SOURCE "bundled"
    ldconfig
}

build_cxi_bits() {
    build_cxi_bits_common --with-rocm="${ROCM_BUILD_PREFIX}"
}

build_libfabric() {
    build_libfabric_common \
        --with-rocr="${ROCM_BUILD_PREFIX}"
}

build_rccl() {
    : "${RCCL_PREFIX:=/opt/alps/rocm/rccl}"
    : "${RCCL_GPU_TARGETS:?RCCL_GPU_TARGETS must be set}"
    : "${RCCL_BUILDDIR:=/tmp/rccl-build}"

    clone_rocm_systems
    rm -rf "${RCCL_PREFIX}" "${RCCL_BUILDDIR}"

    cmake -S "${ROCM_SYSTEMS_SRC_DIR}/projects/rccl" -B "${RCCL_BUILDDIR}" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${RCCL_PREFIX}" \
        -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}" \
        -DAMDGPU_TARGETS="${RCCL_GPU_TARGETS}" \
        -DCMAKE_HIP_COMPILER="${ROCM_SDK_BIN}/amdclang++"
    cmake --build "${RCCL_BUILDDIR}" -j"$(cmake_build_jobs)"
    cmake --install "${RCCL_BUILDDIR}"

    local rccl_lib
    rccl_lib="$(find "${RCCL_PREFIX}" -type f -name 'librccl.so*' | sort -V | tail -n1 || true)"
    [[ -n "${rccl_lib}" ]] || die "RCCL build did not install librccl under ${RCCL_PREFIX}"
    RCCL_LIB_DIR="$(dirname "${rccl_lib}")"
    RCCL_INCLUDE_DIR="${RCCL_PREFIX}/include"
    [[ -d "${RCCL_INCLUDE_DIR}" ]] || die "RCCL build did not install headers under ${RCCL_INCLUDE_DIR}"
    export RCCL_PREFIX RCCL_INCLUDE_DIR RCCL_LIB_DIR
    export CMAKE_PREFIX_PATH="${RCCL_PREFIX}:${RCCL_LIB_DIR}:${RCCL_INCLUDE_DIR}:${CMAKE_PREFIX_PATH}"

    record_alps_version_var RCCL_VERSION "${ROCM_VERSION}"
    record_alps_version_var RCCL_COMMIT "${ROCM_SYSTEMS_COMMIT}"
    record_alps_version_var RCCL_SOURCE "rebuilt"
    ldconfig
}

replace_wheel_rccl() {
    : "${RCCL_PREFIX:=/opt/alps/rocm/rccl}"

    local src_dir="" dst_dir="" d f
    for d in "${RCCL_PREFIX}/lib" "${RCCL_PREFIX}/lib64"; do
        if compgen -G "${d}/librccl.so*" >/dev/null; then
            src_dir="${d}"
            break
        fi
    done
    [[ -n "${src_dir}" ]] || die "No built RCCL libraries found under ${RCCL_PREFIX}"

    while IFS= read -r d; do
        dst_dir="${d}"
        break
    done < <(find "${ROCM_LIBRARIES_DIR}" "${ROCM_SDK_ROOT}" \
        \( -type f -o -type l \) -name 'librccl.so*' -printf '%h\n')
    [[ -n "${dst_dir}" ]] || die "No wheel-bundled RCCL library directory found"

    find "${dst_dir}" -maxdepth 1 \( -type f -o -type l \) -name 'librccl.so*' -print -delete
    for f in "${src_dir}"/librccl.so*; do
        [[ -e "${f}" ]] || continue
        ln -s "${f}" "${dst_dir}/$(basename "${f}")"
    done

    cat > /etc/ld.so.conf.d/99-alps-rocm-rccl.conf <<EOF
${src_dir}
EOF
    ldconfig
}

build_ucx() {
    build_ucx_common --with-rocm="${ROCM_BUILD_PREFIX}"
}

rocm_offload_arch_flags() {
    local targets="${1:?ROCm GPU targets required}"
    local target flags=()

    for target in ${targets//[;,]/ }; do
        [[ -n "${target}" ]] || continue
        flags+=("--offload-arch=${target}")
    done

    [[ "${#flags[@]}" -gt 0 ]] || die "No ROCm offload architectures derived from: ${targets}"
    printf '%s\n' "${flags[*]}"
}

build_ucc() {
    : "${RCCL_GPU_TARGETS:?RCCL_GPU_TARGETS must be set}"

    local ucc_rocm_arch_flags
    ucc_rocm_arch_flags="$(rocm_offload_arch_flags "${UCC_GPU_TARGETS:-${RCCL_GPU_TARGETS}}")"

    build_ucc_common \
        --with-rocm="${ROCM_BUILD_PREFIX}" \
        --with-rocm-arch="${ucc_rocm_arch_flags}" \
        --with-rccl="${RCCL_PREFIX}"
}

build_ompi5() {
    build_ompi5_common --with-rocm="${ROCM_BUILD_PREFIX}"
}

build_aws_ofi_rccl() {
    local cppflags="${CPPFLAGS:-}"
    local ldflags="${LDFLAGS:-}"
    local prefix libdir rocm_configure_prefix="" test_obj

    for prefix in \
        "${ROCM_BUILD_PREFIX}" \
        "${ROCM_CORE_PREFIX:-}" \
        "${ROCM_LIBRARIES_PREFIX:-}" \
        "${ROCM_DEVEL_PREFIX:-}" \
        "${ROCM_CORE_DIR:-}" \
        "${ROCM_LIBRARIES_DIR:-}" \
        "${ROCM_DEVEL_DIR:-}" \
        "${RCCL_PREFIX:-}"; do
        [[ -n "${prefix}" ]] || continue
        if [[ -d "${prefix}/include" ]]; then
            cppflags="-I${prefix}/include ${cppflags}"
        fi
        for libdir in "${prefix}/lib" "${prefix}/lib64"; do
            if [[ -d "${libdir}" ]]; then
                ldflags="-L${libdir} ${ldflags}"
            fi
        done
    done

    cppflags="-D__HIP_PLATFORM_AMD__=1 ${cppflags}"
    for prefix in \
        "${ROCM_BUILD_PREFIX}" \
        "${ROCM_DEVEL_PREFIX:-}" \
        "${ROCM_CORE_PREFIX:-}" \
        "${ROCM_LIBRARIES_PREFIX:-}" \
        "${ROCM_SDK_ROOT:-}"; do
        [[ -n "${prefix}" ]] || continue
        [[ -f "${prefix}/include/hip/hip_runtime_api.h" ]] || continue
        test_obj="$(mktemp /tmp/hip-runtime-api-check.XXXXXX.o)"
        if printf '%s\n' '#include <hip/hip_runtime_api.h>' \
            | g++ ${cppflags} -std=c++17 -x c++ -c -o "${test_obj}" -; then
            rocm_configure_prefix="${prefix}"
            rm -f "${test_obj}"
            break
        fi
        rm -f "${test_obj}"
    done
    [[ -n "${rocm_configure_prefix}" ]] || die "Could not compile-test hip/hip_runtime_api.h from ROCm SDK prefixes"
    echo "Using ROCm prefix for aws-ofi: ${rocm_configure_prefix}"

    # aws-ofi's configure probe does not reliably find HIP headers in ROCm
    # wheel SDK prefixes, so compile-test the header above and seed the cache.
    CPPFLAGS="${cppflags}" \
    LDFLAGS="${ldflags}" \
    ac_cv_header_hip_hip_runtime_api_h=yes \
    build_aws_ofi_nccl_common \
        --with-rocm="${rocm_configure_prefix}" \
        "$@"
}

build_rccl_tests() {
    : "${RCCL_TESTS_GPU_TARGETS:?RCCL_TESTS_GPU_TARGETS must be set}"
    : "${RCCL_TESTS_BUILDDIR:=/tmp/rccl-tests-build}"

    clone_rocm_systems
    rm -rf "${RCCL_TESTS_BUILDDIR}"
    cmake -S "${ROCM_SYSTEMS_SRC_DIR}/projects/rccl-tests" -B "${RCCL_TESTS_BUILDDIR}" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH="${OMPI_PREFIX};${RCCL_PREFIX};${RCCL_LIB_DIR:-};${RCCL_INCLUDE_DIR:-};${CMAKE_PREFIX_PATH}" \
        -DUSE_MPI=ON \
        -DGPU_TARGETS="${RCCL_TESTS_GPU_TARGETS}"
    cmake --build "${RCCL_TESTS_BUILDDIR}" -j"$(cmake_build_jobs)"

    install -d /usr/local/bin
    find "${RCCL_TESTS_BUILDDIR}" -maxdepth 1 -type f -executable -name '*_perf' -print -exec install -m 0755 {} /usr/local/bin/ \;
    rm -rf "${RCCL_TESTS_BUILDDIR}"
}

build_osu() {
    init_hpc_stack_prefixes

    curl -fsSL "http://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-${OSU_VERSION}.tar.gz" -o /tmp/osu.tar.gz
    tar --no-same-owner --no-same-permissions -C /tmp -xzf /tmp/osu.tar.gz
    pushd "/tmp/osu-micro-benchmarks-${OSU_VERSION}"
    CC="${OMPI_PREFIX}/bin/mpicc" \
    CXX="${OMPI_PREFIX}/bin/mpicxx" \
    CFLAGS="-O3" \
    ./configure \
        --prefix=/usr/local \
        --enable-rocm \
        --with-rocm="${ROCM_BUILD_PREFIX}"
    make -j"$(make_jobs)"
    make install
    popd
    rm -rf "/tmp/osu-micro-benchmarks-${OSU_VERSION}" /tmp/osu.tar.gz "${ROCM_SYSTEMS_SRC_DIR:-/tmp/rocm-systems}" "${RCCL_BUILDDIR:-/tmp/rccl-build}"
    ldconfig
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    die "This file provides ROCm stack build functions; run install-alps-rocm-stack.sh instead."
fi
