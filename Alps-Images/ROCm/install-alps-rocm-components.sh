#!/usr/bin/env bash

# Without the rocm-sdk-devel wheel (not published for all ROCm releases) there
# is no expanded devel tree and no conventional /opt/rocm layout. Build one
# from the installed core/libraries wheels so that:
# - clang finds the device bitcode via the default /opt/rocm search path,
# - the linker finds libamdhip64.so and other runtime libs,
# - aws-ofi-rccl and other consumers find headers and cmake packages.
# This mirrors what `rocm-sdk init` provides, without the devel payload.
build_rocm_compat_layout() {
    [[ -n "${ROCM_CORE_PREFIX:-}" ]] || die "build_rocm_compat_layout requires ROCM_CORE_PREFIX"
    [[ -n "${ROCM_LIBRARIES_PREFIX:-}" ]] || die "build_rocm_compat_layout requires ROCM_LIBRARIES_PREFIX"

    local root="/opt/rocm"
    rm -rf "${root}"
    install -d "${root}/lib" "${root}/bin" "${root}/include"

    ln -sf "${ROCM_CORE_PREFIX}"/lib/* "${root}/lib/" 2>/dev/null || true
    ln -sf "${ROCM_LIBRARIES_PREFIX}"/lib/* "${root}/lib/" 2>/dev/null || true
    ln -sf "${ROCM_CORE_PREFIX}"/include/* "${root}/include/" 2>/dev/null || true
    ln -sfn "${ROCM_CORE_PREFIX}/lib/llvm" "${root}/llvm"
    [[ -d "${ROCM_CORE_PREFIX}/lib/llvm/amdgcn" ]] && ln -sfn "${ROCM_CORE_PREFIX}/lib/llvm/amdgcn" "${root}/amdgcn"
    ln -sf "${ROCM_CORE_PREFIX}"/bin/* "${root}/bin/" 2>/dev/null || true

    # The runtime loader finds rocm_sysdeps via ldconfig, but the linker
    # resolves DT_NEEDED chains of linked .so files with its own search path.
    # Expose the sysdeps libraries in the compat lib dir so -L/opt/rocm/lib
    # also satisfies them (libhsa-runtime64 needs them).
    local sysdeps_dir="${ROCM_CORE_PREFIX}/lib/rocm_sysdeps/lib"
    if [[ -d "${sysdeps_dir}" ]]; then
        ln -sf "${sysdeps_dir}"/* "${root}/lib/" 2>/dev/null || true
    fi

    # Development symlinks that the wheel layout omits: clang's HIP linker
    # step expects libamdhip64.so next to the runtime soname, and consumers
    # link against librccl.so / libhsa-runtime64.so by dev name.
    for soname in libamdhip64 libhsa-runtime64 librccl; do
        local real=""
        real="$(compgen -G "${root}/lib/${soname}.so.*" | sort -V | tail -n1 || true)"
        [[ -n "${real}" ]] || continue
        ln -sf "$(basename "${real}")" "${root}/lib/${soname}.so"
        # Also at the wheel location: clang resolves symlinks and passes the
        # real directory to the linker.
        ln -sf "$(basename "${real}")" "$(dirname "${real}")/${soname}.so"
    done

    # Minimal hip package config so find_package(hip) and the hip::device /
    # hip::host / hip::hip targets work for CMake consumers (rccl-tests,
    # aws-ofi-rccl). The devel wheel would provide this; the wheel layout does
    # not ship any CMake config files.
    local hip_cmake_dir="${root}/lib/cmake/hip"
    install -d "${hip_cmake_dir}"
    sed -e "s|@ROCM_ROOT@|${root}|g" -e "s|@ROCM_VERSION@|${ROCM_VERSION}|g" \
        > "${hip_cmake_dir}/hip-config.cmake" <<'HIPCFG'
set(HIP_COMPILER "clang")
set(HIP_RUNTIME "amd")
set(hip_INCLUDE_DIRS "@ROCM_ROOT@/include")
set(hip_INCLUDE_DIR "${hip_INCLUDE_DIRS}")
set(HIP_INCLUDE_DIR "${hip_INCLUDE_DIRS}")
set(HIP_INCLUDE_DIRS "${hip_INCLUDE_DIRS}")
set(hip_LIBRARIES "hip::host;hip::device")
set(HIP_LIBRARIES "${hip_LIBRARIES}")
set(hip_VERSION "@ROCM_VERSION@")
foreach(_t host device hip runtime)
    if(NOT TARGET hip::${_t})
        add_library(hip::${_t} INTERFACE IMPORTED)
        set_target_properties(hip::${_t} PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${hip_INCLUDE_DIRS}"
            INTERFACE_LINK_LIBRARIES "-L@ROCM_ROOT@/lib;-lamdhip64"
            INTERFACE_COMPILE_DEFINITIONS "__HIP_PLATFORM_AMD__")
    endif()
endforeach()
HIPCFG
    # Minimal hip-lang package config: CMake's enable_language(HIP) loads this via
    # CMakeDetermineHIPCompiler/CMakeHIPInformation (vLLM's CMake build depends on it) and
    # expects the hip-lang::device runtime target to be defined. Modeled on the
    # hip-runtime-amd installed-tree package (hip-lang-config/hip-lang-targets).
    local hip_lang_cmake_dir="${root}/lib/cmake/hip-lang"
    install -d "${hip_lang_cmake_dir}"
    sed -e "s|@ROCM_ROOT@|${root}|g" \
        > "${hip_lang_cmake_dir}/hip-lang-config.cmake" <<'HIPLANGCFG'
set(HIP_COMPILER "clang")
set(HIP_RUNTIME "amd")
include("${CMAKE_CURRENT_LIST_DIR}/hip-lang-targets.cmake")
# Approved by CMake: lets CMake pick up the device runtime target without
# hardcoding it in CMakeHIPInformation.cmake consumers.
set(_CMAKE_HIP_DEVICE_RUNTIME_TARGET "hip-lang::device")
HIPLANGCFG
    sed -e "s|@ROCM_ROOT@|${root}|g" \
        > "${hip_lang_cmake_dir}/hip-lang-targets.cmake" <<'HIPLANGTARGETS'
if(NOT TARGET hip-lang::amdhip64)
    add_library(hip-lang::amdhip64 UNKNOWN IMPORTED)
    set_target_properties(hip-lang::amdhip64 PROPERTIES
        IMPORTED_LOCATION "@ROCM_ROOT@/lib/libamdhip64.so"
        INTERFACE_INCLUDE_DIRECTORIES "@ROCM_ROOT@/include"
        INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "@ROCM_ROOT@/include")
endif()
if(NOT TARGET hip-lang::host)
    add_library(hip-lang::host INTERFACE IMPORTED)
    set_target_properties(hip-lang::host PROPERTIES
        INTERFACE_LINK_LIBRARIES "hip-lang::amdhip64")
endif()
if(NOT TARGET hip-lang::device)
    add_library(hip-lang::device INTERFACE IMPORTED)
    set_target_properties(hip-lang::device PROPERTIES
        INTERFACE_LINK_LIBRARIES "hip-lang::host"
        INTERFACE_COMPILE_DEFINITIONS "$<$<COMPILE_LANGUAGE:HIP>:__HIP_PLATFORM_AMD__>")
endif()
HIPLANGTARGETS
    # Minimal RCCL package config mirroring the installed-tree layout so
    # find_package(RCCL CONFIG) and the roc::rccl target resolve. Points at
    # the bundled rccl library selected by use_bundled_rccl when present.
    local rccl_cmake_dir="${root}/lib/cmake/rccl"
    install -d "${rccl_cmake_dir}"
    sed -e "s|@ROCM_ROOT@|${root}|g" \
        > "${rccl_cmake_dir}/rccl-config.cmake" <<'RCCLCFG'
if(NOT TARGET roc::rccl)
    add_library(roc::rccl UNKNOWN IMPORTED)
    set_target_properties(roc::rccl PROPERTIES
        IMPORTED_LOCATION "@ROCM_ROOT@/lib/librccl.so"
        INTERFACE_INCLUDE_DIRECTORIES "@ROCM_ROOT@/include")
endif()
set(RCCL_LIBRARIES "roc::rccl")
set(RCCL_INCLUDE_DIRS "@ROCM_ROOT@/include")
RCCLCFG
    # ROCm version marker used by several build systems.
    install -d "${root}/.info"
    printf '%s\n' "${ROCM_VERSION}" > "${root}/.info/version"

    # Optional: devel-compat layer for wheel-only SDK layouts (no devel wheel).
    # Profiles without ROCM_LIBRARIES_REPO keep the minimal hip/hip-lang/rccl configs.
    if [[ -n "${ROCM_LIBRARIES_REPO:-}" && -n "${ROCM_LIBRARIES_COMMIT:-}" ]]; then
        generate_rocm_devel_compat
    fi
}

# The wheel-only SDK layout has no devel payload: no CMake package configs, so
# find_package() calls for a fixed set of packages (amd_comgr, rocrand, hiprand,
# rocblas, hipblas, miopen, hipfft, hipsparse, rocprim, hipcub, rocthrust,
# hipsolver, rocsolver, hiprtc) fail and extension builds like vLLM's cannot
# configure. Generate the missing package configs and install the public
# headers from the pinned rocm-libraries source tree (same therock tag family
# as ROCM_SYSTEMS_COMMIT).
generate_rocm_devel_compat() {
    [[ -n "${ROCM_LIBRARIES_REPO:-}" ]] || die "generate_rocm_devel_compat requires ROCM_LIBRARIES_REPO"
    [[ -n "${ROCM_LIBRARIES_COMMIT:-}" ]] || die "generate_rocm_devel_compat requires ROCM_LIBRARIES_COMMIT"

    local root="/opt/rocm"
    local src_dir="${ROCM_LIBRARIES_SRC_DIR:-/tmp/rocm-libraries}"

    if [[ ! -d "${src_dir}/.git" ]]; then
        git clone "${ROCM_LIBRARIES_REPO}" "${src_dir}"
    fi
    git -C "${src_dir}" fetch --all --quiet
    git -C "${src_dir}" checkout --quiet --detach
    git -C "${src_dir}" reset --hard "${ROCM_LIBRARIES_COMMIT}"

    # Header installs as "src|dst" pairs relative to the rocm-libraries checkout and
    # /opt/rocm/include. Layouts mirror the installed tree of each project.
    local header_installs=(
        "projects/rocprim/rocprim/include/rocprim|rocprim"
        "projects/rocthrust/thrust|thrust"
        "projects/hipcub/hipcub/include/hipcub|hipcub"
        "projects/rocrand/library/include/rocrand|rocrand"
        "projects/hiprand/library/include/hiprand|hiprand"
        "projects/hipblas/library/include|hipblas"
        "projects/hipblas-common/library/include/hipblas-common|hipblas-common"
        "projects/hipblaslt/library/include/hipblaslt|hipblaslt"
        "projects/hipsparselt/library/include|hipsparselt"
        "projects/hipsparse/library/include|hipsparse"
        "projects/hipsolver/library/include|hipsolver"
    )
    # Headers beyond the compile-time requirements above (miopen, rocblas, ...)
    # stay uninstalled: torch's ATen/hip headers pull in hipblas, hipblaslt, hipsolver,
    # and hipsparse; the remaining packages are consumed via find_package configs only.
    local -A lib_pkgs=(
        [amd_comgr]="libamd_comgr"
        [rocrand]="librocrand"
        [hiprand]="libhiprand"
        [rocblas]="librocblas"
        [hipblas]="libhipblas"
        [miopen]="libMIOpen"
        [hipfft]="libhipfft"
        [hipsparse]="libhipsparse"
        [hipsolver]="libhipsolver"
        [rocsolver]="librocsolver"
        [hiprtc]="libhiprtc"
        [hipblaslt]="libhipblaslt"
        [hipsparselt]="libhipsparselt"
        [hsa-runtime64]="libhsa-runtime64"
    )

    local pair src dst
    for pair in "${header_installs[@]}"; do
        src="${pair%%|*}"
        dst="${pair##*|}"
        [[ -d "${src_dir}/${src}" ]] || die "rocm-libraries header source missing: ${src_dir}/${src}"
        rm -rf "${root}/include/${dst}"
        install -d "${root}/include/${dst}"
        cp -a "${src_dir}/${src}/." "${root}/include/${dst}/"
    done
    install -d "${root}/include/hipblas"
    cat > "${root}/include/hipblas/hipblas-export.h" <<'HIPBLASEXPORT'
#ifndef HIPBLAS_EXPORT_H
#define HIPBLAS_EXPORT_H
#ifndef HIPBLAS_EXPORT
#define HIPBLAS_EXPORT __attribute__((visibility("default")))
#endif
#ifndef HIPBLAS_NO_EXPORT
#define HIPBLAS_NO_EXPORT __attribute__((visibility("hidden")))
#endif
#ifndef HIPBLAS_DEPRECATED
#define HIPBLAS_DEPRECATED __attribute__((__deprecated__))
#endif
#endif
HIPBLASEXPORT
    cat > "${root}/include/hipblas/hipblas-version.h" <<'HIPBLASVERSION'
#ifndef HIPBLAS_VERSION_H
#define HIPBLAS_VERSION_H
#define hipblasVersionMajor 3
#define hipblasVersionMinor 5
#define hipblasVersionPatch 0
#define hipblasVersionTweak 0
#define hipblasVersionK 100
#endif
HIPBLASVERSION
    install -d "${root}/include/hipblaslt"
    cat > "${root}/include/hipblaslt/hipblaslt-export.h" <<'HIPBLASLTEXPORT'
#ifndef HIPBLASLT_EXPORT_H
#define HIPBLASLT_EXPORT_H
#ifndef HIPBLASLT_EXPORT
#define HIPBLASLT_EXPORT __attribute__((visibility("default")))
#endif
#ifndef HIPBLASLT_NO_EXPORT
#define HIPBLASLT_NO_EXPORT __attribute__((visibility("hidden")))
#endif
#endif
HIPBLASLTEXPORT
    cat > "${root}/include/hipblaslt/hipblaslt-version.h" <<'HIPBLASLTVERSION'
#ifndef _HIPBLASLT_VERSION_H_
#define _HIPBLASLT_VERSION_H_
#define HIPBLASLT_VERSION_MAJOR 1
#define HIPBLASLT_VERSION_MINOR 0
#define HIPBLASLT_VERSION_PATCH 0
#define HIPBLASLT_VERSION_TWEAK 0
#endif
HIPBLASLTVERSION
    cat > "${root}/include/hipsparselt/hipsparselt-export.h" <<'HIPSPARSELTEXPORT'
#ifndef HIPSPARSELTEXPORT_H
#define HIPSPARSELTEXPORT_H
#ifndef HIPSPARSELTEXPORT
#define HIPSPARSELTEXPORT __attribute__((visibility("default")))
#endif
#ifndef HIPSPARSELTNOEXPORT
#define HIPSPARSELTNOEXPORT __attribute__((visibility("hidden")))
#endif
#endif
HIPSPARSELTEXPORT
    cat > "${root}/include/hipsparselt/hipsparselt-version.h" <<'HIPSPARSELTVERSION'
#ifndef HIPSPARSELT_VERSION_H
#define HIPSPARSELT_VERSION_H
#define HIPSPARSELT_VERSION_MAJOR 0
#define HIPSPARSELT_VERSION_MINOR 0
#define HIPSPARSELT_VERSION_PATCH 0
#endif
HIPSPARSELTVERSION
    cat > "${root}/include/thrust/rocthrust_version.hpp" <<'ROCTHRUSTVERSION'
#ifndef ROCTHRUST_VERSION_HPP_
#define ROCTHRUST_VERSION_HPP_
#define ROCTHRUST_VERSION 406000
#define ROCTHRUST_VERSION_MAJOR 4
#define ROCTHRUST_VERSION_MINOR 6
#define ROCTHRUST_VERSION_PATCH 0
#endif
ROCTHRUSTVERSION
    install -d "${root}/include/hipcub"
    cat > "${root}/include/hipcub/hipcub_version.hpp" <<'HIPCUBVERSION'
#ifndef HIPCUB_VERSION_HPP_
#define HIPCUB_VERSION_HPP_
#define HIPCUB_VERSION 406000
#define HIPCUB_VERSION_MAJOR 4
#define HIPCUB_VERSION_MINOR 6
#define HIPCUB_VERSION_PATCH 0
#define HIPCUB_CCCL_VERSION 20802
#endif
HIPCUBVERSION
    install -d "${root}/include/rocprim"
    cat > "${root}/include/rocprim/rocprim_version.hpp" <<'ROCPRIMVERSION'
#ifndef ROCPRIM_VERSION_HPP_
#define ROCPRIM_VERSION_HPP_
#define ROCPRIM_VERSION 406000
#define ROCPRIM_VERSION_MAJOR 4
#define ROCPRIM_VERSION_MINOR 6
#define ROCPRIM_VERSION_PATCH 0
#endif
ROCPRIMVERSION
    install -d "${root}/include/rocrand" "${root}/include/hiprand"
    cat > "${root}/include/rocrand/rocrand_version.h" <<'ROCRANDVERSION'
#ifndef ROCRAND_VERSION_H_
#define ROCRAND_VERSION_H_
#define ROCRAND_VERSION 500000
#endif
ROCRANDVERSION
    cat > "${root}/include/hiprand/hiprand_version.h" <<'HIPRANDVERSION'
#ifndef HIPRAND_VERSION_H_
#define HIPRAND_VERSION_H_
#define HIPRAND_VERSION 304000
#endif
HIPRANDVERSION

    # Generated headers the hipsparse/hipsolver packages normally ship from templates.
    cat > "${root}/include/hipsparse/hipsparse-export.h" <<'HIPSPARSEEXPORT'
#ifndef HIPSPARSE_EXPORT_H
#define HIPSPARSE_EXPORT_H
#ifndef HIPSPARSE_EXPORT
#define HIPSPARSE_EXPORT __attribute__((visibility("default")))
#endif
#ifndef HIPSPARSE_NO_EXPORT
#define HIPSPARSE_NO_EXPORT __attribute__((visibility("hidden")))
#endif
#endif
HIPSPARSEEXPORT
    cat > "${root}/include/hipsparse/hipsparse-version.h" <<'HIPSPARSEVERSION'
#ifndef HIPSPARSE_VERSION_H
#define HIPSPARSE_VERSION_H
#define hipsparseVersionMajor 4
#define hipsparseVersionMinor 6
#define hipsparseVersionPatch 0
#endif
HIPSPARSEVERSION
    cat > "${root}/include/hipsolver/internal/hipsolver-version.h" <<'HIPSOLVERVERSION'
#ifndef HIPSOLVER_VERSION_H
#define HIPSOLVER_VERSION_H
#define hipsolverVersionMajor 3
#define hipsolverVersionMinor 5
#define hipsolverVersionPatch 0
#define hipsolverVersionTweak 0
#endif
HIPSOLVERVERSION
    cat > "${root}/include/hipsolver/internal/hipsolver-export.h" <<'HIPSOLVEREXPORT'
#ifndef HIPSOLVER_EXPORT_H
#define HIPSOLVER_EXPORT_H
#ifndef HIPSOLVER_EXPORT
#define HIPSOLVER_EXPORT __attribute__((visibility("default")))
#endif
#ifndef HIPSOLVER_NO_EXPORT
#define HIPSOLVER_NO_EXPORT __attribute__((visibility("hidden")))
#endif
#endif
HIPSOLVEREXPORT
    # CMake package configs. Header-only packages get INTERFACE targets that
    # carry the include path; library-backed packages get IMPORTED locations.
    local pkg_cmake_dir cmake_config
    for pkg in rocprim hipcub rocthrust; do
        pkg_cmake_dir="${root}/lib/cmake/${pkg}"
        install -d "${pkg_cmake_dir}"
        cmake_config="${pkg_cmake_dir}/${pkg}-config.cmake"
        sed -e "s|@ROCM_ROOT@|${root}|g" -e "s|@PKG@|${pkg}|g" -e "s|@ROCM_VERSION@|${ROCM_VERSION}|g" \
            > "${cmake_config}" <<'HDRPKGCFG'
set(@PKG@_INCLUDE_DIR "@ROCM_ROOT@/include")
set(@PKG@_INCLUDE_DIRS "${@PKG@_INCLUDE_DIR}")
set(@PKG@_VERSION "@ROCM_VERSION@")
string(TOUPPER @PKG@ UPPERPKG)
set(${UPPERPKG}_INCLUDE_DIR "${@PKG@_INCLUDE_DIR}")
set(${UPPERPKG}_INCLUDE_DIRS "${@PKG@_INCLUDE_DIR}")
if(NOT TARGET roc::@PKG@)
    add_library(roc::@PKG@ INTERFACE IMPORTED)
    set_target_properties(roc::@PKG@ PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${@PKG@_INCLUDE_DIR}")
endif()
if(NOT TARGET hip::@PKG@)
    add_library(hip::@PKG@ INTERFACE IMPORTED)
    set_target_properties(hip::@PKG@ PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${@PKG@_INCLUDE_DIR}")
endif()
HDRPKGCFG
    done

    local soname real
    for pkg in "${!lib_pkgs[@]}"; do
        soname="${lib_pkgs[${pkg}]}"
        real="$(compgen -G "${root}/lib/${soname}.so.*" | sort -V | tail -n1 || true)"
        [[ -n "${real}" ]] || die "rocm devel compat: ${soname} not found in ${root}/lib"
        pkg_cmake_dir="${root}/lib/cmake/${pkg}"
        install -d "${pkg_cmake_dir}"
        cmake_config="${pkg_cmake_dir}/${pkg}-config.cmake"
        sed -e "s|@ROCM_ROOT@|${root}|g" -e "s|@PKG@|${pkg}|g" -e "s|@ROCM_VERSION@|${ROCM_VERSION}|g" -e "s|@LIBREAL@|${real}|g" \
            > "${cmake_config}" <<'LIBPKGCFG'
set(@PKG@_INCLUDE_DIR "@ROCM_ROOT@/include")
set(@PKG@_INCLUDE_DIRS "${@PKG@_INCLUDE_DIR}")
set(@PKG@_VERSION "@ROCM_VERSION@")
string(TOUPPER @PKG@ UPPERPKG)
set(${UPPERPKG}_INCLUDE_DIR "${@PKG@_INCLUDE_DIR}")
set(${UPPERPKG}_INCLUDE_DIRS "${@PKG@_INCLUDE_DIR}")
if(NOT TARGET roc::@PKG@)
    add_library(roc::@PKG@ UNKNOWN IMPORTED)
    set_target_properties(roc::@PKG@ PROPERTIES
        IMPORTED_LOCATION "@LIBREAL@"
        INTERFACE_INCLUDE_DIRECTORIES "${@PKG@_INCLUDE_DIR}")
endif()
# torch's Caffe2Targets links against hip::<pkg> for some packages and roc::<pkg> for
# others (hip::hiprand, roc::hipblas, hiprtc::hiprtc, ...); provide both spellings.
if(NOT TARGET hip::@PKG@)
    add_library(hip::@PKG@ UNKNOWN IMPORTED)
    set_target_properties(hip::@PKG@ PROPERTIES
        IMPORTED_LOCATION "@LIBREAL@"
        INTERFACE_INCLUDE_DIRECTORIES "${@PKG@_INCLUDE_DIR}")
endif()
if(NOT TARGET @PKG@::@PKG@)
    add_library(@PKG@::@PKG@ UNKNOWN IMPORTED)
    set_target_properties(@PKG@::@PKG@ PROPERTIES
        IMPORTED_LOCATION "@LIBREAL@"
        INTERFACE_INCLUDE_DIRECTORIES "${@PKG@_INCLUDE_DIR}")
endif()
LIBPKGCFG
        # The linker resolves -l<name> against the dev symlink, which the wheel
        # layout omits; create it next to the runtime soname.
        ln -sf "$(basename "${real}")" "${root}/lib/${soname}.so"
    done

    # clang 23 and CMake's enable_language(HIP) derive their ROCm root from the compiler
    # / hipconfig location, which is the wheel install (ROCM_CORE_PREFIX), not /opt/rocm.
    # Without env overrides they look for cmake packages and headers there, so mirror the
    # compat layer into the wheel prefix: package configs and the extra public headers.
    if [[ -z "${ROCM_CORE_PREFIX:-}" ]]; then
        die "rocm devel compat: ROCM_CORE_PREFIX is required to mirror the compat layer into the wheel install"
    fi
    [[ -d "${ROCM_CORE_PREFIX}" ]] || die "rocm devel compat: ROCM_CORE_PREFIX is not a directory: ${ROCM_CORE_PREFIX}"
    [[ "${ROCM_CORE_PREFIX}" != "${root}" ]] || die "rocm devel compat: ROCM_CORE_PREFIX must differ from ${root}"
    ln -sfn "${root}/lib/cmake" "${ROCM_CORE_PREFIX}/lib/cmake"
    local hdr
    for hdr in thrust hipcub rocprim rocrand hiprand hipblas hipblas-common hipblaslt hipsparselt hipsparse hipsolver; do
        ln -sfn "${root}/include/${hdr}" "${ROCM_CORE_PREFIX}/include/${hdr}"
    done
    ln -sfn "${root}/.info" "${ROCM_CORE_PREFIX}/.info"

    # The source clone is several GB and only the headers above are consumed; drop it
    # so it does not ship in the image layer.
    rm -rf "${src_dir}"

    echo "Generated ROCm devel compatibility layer (package configs + headers) at ${root}"
}

bootstrap_rocm_sdk() {
    : "${ROCM_VERSION:?ROCM_VERSION must be set}"
    : "${ROCM_PYPI_INDEX_URL:?ROCM_PYPI_INDEX_URL must be set to the ROCm wheel index}"
    : "${ROCM_PYTHON:=/opt/venv/bin/python}"
    : "${ROCM_SDK:=/opt/venv/bin/rocm-sdk}"

    [[ -x "${ROCM_PYTHON}" ]] || die "ROCm Python not found: ${ROCM_PYTHON}"

    # Install the SDK wheels unless the base image already ships them. Some
    # base images (e.g. the ROCm 10.0 PyTorch image) are built from an index
    # that is not publicly available, so reinstalling pinned versions would
    # fail; the preinstalled wheels are authoritative in that case.
    local -a sdk_packages=()
    local pkg have
    for pkg in "rocm==${ROCM_VERSION}" "rocm-sdk-core==${ROCM_VERSION}" "rocm-sdk-libraries==${ROCM_VERSION}"; do
        # Check the installed distribution, not an importable module: hyphens
        # are invalid in module names (find_spec would always miss them) and
        # "rocm" is a meta package with no importable module at all.
        if "${ROCM_PYTHON}" -c "import importlib.metadata, sys; sys.exit(0 if importlib.metadata.version(\"${pkg%%==*}\") else 1)" 2>/dev/null; then
            continue
        fi
        sdk_packages+=("${pkg}")
    done
    local target
    for target in ${RCCL_GPU_TARGETS//;/ }; do
        [[ -n "${target}" ]] || continue
        have="$("${ROCM_PYTHON}" -c "import importlib.util, sys; spec = importlib.util.find_spec(\"rocm_sdk_device_${target//-/_}\"); sys.exit(0 if spec else 1)" && echo yes || echo no)"
        [[ "${have}" == yes ]] || sdk_packages+=("rocm-sdk-device-${target}==${ROCM_VERSION}")
    done
    if [[ "${#sdk_packages[@]}" -gt 0 ]]; then
        "${ROCM_PYTHON}" -m pip install --no-cache-dir --index-url "${ROCM_PYPI_INDEX_URL}" --no-deps "${sdk_packages[@]}"
    else
        echo "ROCm SDK wheels already present in base image; skipping wheel install"
    fi

    # rocm-sdk-devel provides the expanded devel tree and the rocm-sdk CLI
    # helpers that need it. It is optional: when missing (ROCm 10.0 is not
    # published on the public wheel index), derive the SDK layout ourselves.
    local have_devel="no"
    "${ROCM_PYTHON}" -c "import rocm_sdk_devel" 2>/dev/null && have_devel="yes"
    if [[ "${have_devel}" == yes ]]; then
        if ! "${ROCM_PYTHON}" -m pip show rocm-sdk-devel >/dev/null 2>&1 \
            || [[ "$("${ROCM_PYTHON}" -m pip show rocm-sdk-devel 2>/dev/null | sed -n 's/^Version: //p')" != "${ROCM_VERSION}" ]]; then
            "${ROCM_PYTHON}" -m pip install --no-cache-dir --index-url "${ROCM_PYPI_INDEX_URL}" --no-deps "rocm-sdk-devel==${ROCM_VERSION}"
        fi
        "${ROCM_SDK}" init
    else
        echo "rocm-sdk-devel not available; deriving SDK layout without rocm-sdk init"
    fi

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
    if [[ "${have_devel}" == yes ]]; then
        ROCM_SDK_ROOT="$("${ROCM_SDK}" path --root)"
        ROCM_SDK_BIN="$("${ROCM_SDK}" path --bin)"
        ROCM_SDK_CMAKE="$("${ROCM_SDK}" path --cmake)"
    else
        ROCM_SDK_ROOT="${ROCM_CORE_PREFIX}"
        ROCM_SDK_BIN="${ROCM_CORE_PREFIX}/bin"
        ROCM_SDK_CMAKE="${ROCM_CORE_PREFIX}/lib/cmake"
    fi
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
    if [[ "${have_devel}" == yes ]]; then
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
    else
        ROCM_DEVEL_DIR=""
        ROCM_DEVEL_PREFIX=""
    fi

    [[ -d "${ROCM_SDK_ROOT}" ]] || die "ROCM_SDK_ROOT is not a directory: ${ROCM_SDK_ROOT}"
    [[ -d "${ROCM_SDK_BIN}" ]] || die "ROCM_SDK_BIN is not a directory: ${ROCM_SDK_BIN}"
    [[ -d "${ROCM_CORE_DIR}" ]] || die "ROCM_CORE_DIR is not a directory: ${ROCM_CORE_DIR}"
    [[ -d "${ROCM_CORE_PREFIX}" ]] || die "ROCM_CORE_PREFIX is not a directory: ${ROCM_CORE_PREFIX}"
    [[ -d "${ROCM_LIBRARIES_DIR}" ]] || die "ROCM_LIBRARIES_DIR is not a directory: ${ROCM_LIBRARIES_DIR}"
    [[ -d "${ROCM_LIBRARIES_PREFIX}" ]] || die "ROCM_LIBRARIES_PREFIX is not a directory: ${ROCM_LIBRARIES_PREFIX}"
    if [[ "${have_devel}" == yes ]]; then
        [[ -d "${ROCM_DEVEL_DIR}" ]] || die "ROCM_DEVEL_DIR is not a directory: ${ROCM_DEVEL_DIR}"
        [[ -d "${ROCM_DEVEL_PREFIX}" ]] || die "ROCM_DEVEL_PREFIX is not a directory: ${ROCM_DEVEL_PREFIX}"
    fi

    if [[ "${have_devel}" != yes ]]; then
        build_rocm_compat_layout
    fi

    ROCM_BUILD_PREFIX="$(discover_rocm_build_prefix)"
    [[ -n "${ROCM_BUILD_PREFIX}" ]] || die "Could not find a ROCm prefix with HIP/HSA headers and runtime libraries"
    echo "Using ROCm build prefix: ${ROCM_BUILD_PREFIX}"

    local targets
    targets="$("${ROCM_SDK}" targets)"
    local target
    for target in ${RCCL_GPU_TARGETS//;/ }; do
        [[ -n "${target}" ]] || continue
        [[ "${targets}" == *"${target}"* ]] || die "rocm-sdk targets does not include ${target}: ${targets}"
    done
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
    patch_torch_rocm_disable_implicit_amdsmi
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
            try:
                shutdown()
            except Exception:
                pass
PY
}

link_amdsmi_package_library() {
    local amdsmi_pkg_dir amdsmi_lib="" amdsmi_sysdeps_root="" wrapper preload_file needed
    local candidate lib prefix sysdeps_dir name
    local -a sysdeps_fallbacks=() sysdeps_dirs=() candidates=() ordered=() preload_names=()

    amdsmi_pkg_dir="$(${ROCM_PYTHON} - <<'PY'
import importlib.util

spec = importlib.util.find_spec("amdsmi")
if spec is None or not spec.submodule_search_locations:
    raise SystemExit("amdsmi package was not found")
print(next(iter(spec.submodule_search_locations)))
PY
)"
    [[ -d "${amdsmi_pkg_dir}" ]] || die "amdsmi package directory not found: ${amdsmi_pkg_dir}"

    # The ROCm runtime stack (librccl, libamdhip64 consumers) resolves
    # libamd_smi.so.26 through RPATHs that point into the ROCm core SDK
    # (e.g. $ORIGIN/../../_rocm_sdk_core/lib), so that is the copy mapped
    # into any process that imports torch. The amdsmi Python package must
    # load the same file: loading the byte-identical devel copy from a
    # different inode maps the same SONAME twice (ODR violation), which
    # corrupts libnl global state and segfaults at process exit once both
    # torch and amdsmi are imported (vLLM imports both). Prefer the core
    # prefix, then libraries, then the remaining candidates.

    while IFS= read -r prefix; do
        [[ -n "${prefix}" ]] || continue
        candidates+=("${prefix}")
    done < <(rocm_sdk_prefix_candidates)
    if [[ -n "${ROCM_CORE_PREFIX:-}" ]]; then
        ordered+=("${ROCM_CORE_PREFIX}")
        [[ -n "${ROCM_LIBRARIES_PREFIX:-}" ]] && ordered+=("${ROCM_LIBRARIES_PREFIX}")
        for prefix in "${candidates[@]}"; do
            case " ${ordered[*]} " in
                *" ${prefix} "*) ;;
                *) ordered+=("${prefix}") ;;
            esac
        done
        candidates=("${ordered[@]}")
    fi

    while IFS= read -r candidate; do
        [[ -n "${candidate}" ]] || continue
        lib="$(find "${candidate}" -maxdepth 2 \
            \( -type f -o -type l \) -name 'libamd_smi.so*' -print -quit 2>/dev/null || true)"
        if [[ -n "${lib}" && -z "${amdsmi_lib}" ]]; then
            amdsmi_lib="${lib}"
            amdsmi_sysdeps_root="${candidate}"
        fi
        for sysdeps_dir in "${candidate}/lib/rocm_sysdeps/lib" "${candidate}/lib64/rocm_sysdeps/lib"; do
            [[ -d "${sysdeps_dir}" ]] || continue
            compgen -G "${sysdeps_dir}/librocm_sysdeps_*.so*" >/dev/null || continue
            case " ${sysdeps_fallbacks[*]} " in
                *" ${sysdeps_dir} "*) ;;
                *) sysdeps_fallbacks+=("${sysdeps_dir}") ;;
            esac
            # Collect sysdeps directories that match the selected library's
            # root first, so the preload maps the same files the RPATH of
            # libamd_smi.so resolves to. Other prefixes stay as fallbacks.
            if [[ "${candidate}" == "${amdsmi_sysdeps_root}" ]]; then
                sysdeps_dirs+=("${sysdeps_dir}")
            fi
        done
    done < <(printf '%s\n' "${candidates[@]}")
    if [[ "${#sysdeps_dirs[@]}" -eq 0 ]]; then
        sysdeps_dirs=("${sysdeps_fallbacks[@]}")
    fi
    [[ -n "${amdsmi_lib}" ]] || die "libamd_smi.so* not found after installing amdsmi"

    ln -sf "${amdsmi_lib}" "${amdsmi_pkg_dir}/libamd_smi.so"

    # Derive the preload list from the ELF NEEDED entries of the selected
    # libamd_smi.so plus the DRM library it dlopens without a NEEDED entry
    # (used for amdgpu_device_initialize paths such as VRAM-usage queries), so
    # SONAME bumps in future ROCm releases are followed automatically.
    needed="$(readelf -d "${amdsmi_lib}" 2>/dev/null | sed -n 's/.*Shared library: \[\(.*\)\]/\1/p' | grep '^librocm_sysdeps_' || true)"
    [[ -n "${needed}" ]] || die "no rocm_sysdeps NEEDED entries found in ${amdsmi_lib}"
    while IFS= read -r name; do
        [[ -n "${name}" ]] || continue
        case " ${preload_names[*]:-} " in
            *" ${name} "*) ;;
            *) preload_names+=("${name}") ;;
        esac
    done <<<"${needed}"
    # dlopen-only dependency: not in NEEDED; required for DRM-backed queries
    # and to keep libamd_smi's opportunistic loader from warning per process.
    [[ " ${preload_names[*]} " == *" librocm_sysdeps_drm_amdgpu.so.1 "* ]] \
        || preload_names+=("librocm_sysdeps_drm_amdgpu.so.1")
    if ! find "${sysdeps_dirs[@]}" -maxdepth 1 -name 'librocm_sysdeps_drm_amdgpu.so*' -print -quit 2>/dev/null | grep -q .; then
        echo "WARNING: no librocm_sysdeps_drm_amdgpu.so* found in sysdeps dirs; DRM-backed amdsmi queries will degrade" >&2
    fi

    preload_file="${amdsmi_pkg_dir}/_alps_amdsmi_preload.py"
    {
        printf 'import ctypes\n'
        printf 'import sys\n'
        printf 'from pathlib import Path\n\n'
        printf '_SYSDEPS_DIRS = (\n'
        for sysdeps_dir in "${sysdeps_dirs[@]}"; do
            printf '    "%s",\n' "${sysdeps_dir}"
        done
        printf ')\n'
        printf '_REQUIRED = (\n'
        for name in "${preload_names[@]}"; do
            printf '    "%s",\n' "${name}"
        done
        printf ')\n\n'
        printf 'def preload_amdsmi_dependencies():\n'
        printf '    for name in _REQUIRED:\n'
        printf '        for directory in _SYSDEPS_DIRS:\n'
        printf '            path = Path(directory) / name\n'
        printf '            if path.exists():\n'
        printf '                ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)\n'
        printf '                break\n'
        printf '        else:\n'
        printf '            print(f"WARNING: amdsmi preload: {name} not found in any of {_SYSDEPS_DIRS}", file=sys.stderr)\n'
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


patch_torch_rocm_disable_implicit_amdsmi() {
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
    print("PyTorch CUDA module not found; skipping ROCm amdsmi import patch")
    raise SystemExit(0)

text = path.read_text()
patched = """        else:
            # Alps: block the implicit amdsmi import. Loading amdsmi here maps
            # a second copy of libamd_smi's sysdeps (libnl) next to the copy
            # the ROCm runtime stack loads via RPATH, corrupting libnl state
            # and segfaulting at exit. The amdsmi package remains importable
            # directly; vLLM's platform probe loads it explicitly and safely
            # because its preload maps the core-SDK copies first.
            raise ModuleNotFoundError(
                "amdsmi auto-import disabled in Alps ROCm images"
            )
"""
if patched in text:
    print(f"PyTorch ROCm amdsmi auto-import already disabled: {path}")
    raise SystemExit(0)

old = """        else:
            import ctypes
            from pathlib import Path
"""
if old not in text:
    raise SystemExit(f"could not patch {path}: amdsmi auto-import marker not found")

path.write_text(text.replace(old, patched, 1))
print(f"Disabled PyTorch ROCm amdsmi auto-import: {path}")
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
    local candidate
    local -A emitted=()

    for candidate in \
        "${ROCM_BUILD_PREFIX:-}" \
        "${ROCM_DEVEL_PREFIX:-}" \
        "${ROCM_CORE_PREFIX:-}" \
        "${ROCM_LIBRARIES_PREFIX:-}" \
        "/opt/rocm" \
        "${ROCM_SDK_ROOT:-}" \
        "${ROCM_DEVEL_DIR:-}/_rocm_sdk_devel" \
        "${ROCM_CORE_DIR:-}/_rocm_sdk_core" \
        "${ROCM_LIBRARIES_DIR:-}/_rocm_sdk_libraries"; do
        [[ -n "${candidate}" ]] || continue
        [[ -n "${emitted[${candidate}]:-}" ]] && continue
        emitted["${candidate}"]=1
        printf '%s\n' "${candidate}"
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
    pushd "${ROCM_SYSTEMS_SRC_DIR}" > /dev/null || return 1
    git reset --hard "${ROCM_SYSTEMS_COMMIT}"
    git submodule update --init --recursive --depth=1 projects/rccl projects/rccl-tests
    popd > /dev/null || return 1
}

# Generate the public rccl.h from the pinned rocm-systems source when the SDK
# only ships the runtime library. Mirrors what the rccl build would install.
generate_rccl_header_from_source() {
    local template="${ROCM_SYSTEMS_SRC_DIR}/projects/rccl/src/nccl.h.in"
    [[ -f "${template}" ]] || die "RCCL header template not found: ${template}"
    local version_mk="${ROCM_SYSTEMS_SRC_DIR}/projects/rccl/makefiles/version.mk"
    [[ -f "${version_mk}" ]] || die "RCCL version file not found: ${version_mk}"

    local major minor patch out_dir="/opt/alps/rocm/rccl-bundled-src"
    major="$(sed -n 's/^NCCL_MAJOR[[:space:]]*:= *\([0-9]*\).*/\1/p' "${version_mk}")"
    minor="$(sed -n 's/^NCCL_MINOR[[:space:]]*:= *\([0-9]*\).*/\1/p' "${version_mk}")"
    patch="$(sed -n 's/^NCCL_PATCH[[:space:]]*:= *\([0-9]*\).*/\1/p' "${version_mk}")"
    [[ -n "${major}" && -n "${minor}" && -n "${patch}" ]] || die "Could not parse RCCL version from ${version_mk}"

    local suffix version
    suffix="$(sed -n 's/^NCCL_SUFFIX[[:space:]]*:= *\([^#]*\).*/\1/p' "${version_mk}" | tr -d '[:space:]')"
    version="$(printf '%d%02d%02d' "${major}" "${minor}" "${patch}")"

    install -d "${out_dir}/include"
    sed -e "s/\${NCCL_MAJOR}/${major}/g" \
        -e "s/\${NCCL_MINOR}/${minor}/g" \
        -e "s/\${NCCL_PATCH}/${patch}/g" \
        -e "s/\${NCCL_SUFFIX}/${suffix}/g" \
        -e "s/\${NCCL_VERSION}/${version}/g" \
        "${template}" > "${out_dir}/include/rccl.h"
    # The installed tree ships nccl.h next to rccl/rccl.h because the device
    # headers include <nccl.h>.
    cp "${out_dir}/include/rccl.h" "${out_dir}/include/nccl.h"
    # Consumers of the installed tree include more than rccl.h (for example
    # rccl-tests includes <nccl_device.h>). The rccl build installs the
    # HIPIFIED device headers: hipify-perl translates them and renames
    # core.h to core_tmp.h (impl headers include ../core_tmp.h). Replicate
    # that for the bundled-source include tree.
    local src_include="${ROCM_SYSTEMS_SRC_DIR}/projects/rccl/src/include"
    local device_header
    for device_header in nccl_common.h nccl_device.h rccl_common.h rccl_float8.h rccl_vars.h; do
        [[ -f "${src_include}/${device_header}" ]] || continue
        hipify-perl -quiet-warnings "${src_include}/${device_header}" \
            -o "${out_dir}/include/${device_header}" 2>/dev/null \
            || install -m 0644 "${src_include}/${device_header}" "${out_dir}/include/${device_header}"
    done
    if [[ -d "${src_include}/nccl_device" ]]; then
        rm -rf "${out_dir}/include/nccl_device"
        cp -a "${src_include}/nccl_device" "${out_dir}/include/"
        while IFS= read -r device_header; do
            [[ -n "${device_header}" ]] || continue
            # hip_compat.h is copied as-is by the rccl build (it contains both
            # CUDA and HIP code paths); hipifying it breaks amdgcn builtins.
            if [[ "${device_header}" == */hip_compat.h ]]; then
                continue
            fi
            if hipify-perl -quiet-warnings "${device_header}" -o "${device_header}.hip" 2>/dev/null; then
                mv "${device_header}.hip" "${device_header}"
            else
                rm -f "${device_header}.hip"
            fi
        done < <(find "${out_dir}/include/nccl_device" -type f -name '*.h')
        # hipify renames basename-colliding includes with _tmp suffixes
        # (core.h -> core_tmp.h, gin.h -> gin_tmp.h, ...). Create the aliased
        # copies for every *_tmp.h include found in the hipified headers.
        local tmp_include device_dir="${out_dir}/include/nccl_device"
        while IFS= read -r tmp_include; do
            [[ -n "${tmp_include}" ]] || continue
            local tmp_base="${tmp_include##*/}"
            local orig="${tmp_base%_tmp.h}.h"
            # The renamed includes always refer to headers at the device dir
            # root (they appear as name_tmp.h or ../name_tmp.h from subdirs).
            if [[ -f "${device_dir}/${orig}" && ! -e "${device_dir}/${tmp_base}" ]]; then
                cp "${device_dir}/${orig}" "${device_dir}/${tmp_base}"
            fi
        done < <(grep -rhoE '#include "[a-zA-Z0-9_./]+_tmp\.h"' "${device_dir}" 2>/dev/null | sed 's/#include "//; s/"$//' | sort -u)
    fi
    echo "Generated rccl.h (${major}.${minor}.${patch}) from ${ROCM_SYSTEMS_COMMIT}" >&2
    printf '%s\n' "${out_dir}/include/rccl.h"
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

    # Search only SDK roots, never the bare site-packages parent directory
    # (LIBRARIES_DIR/CORE_DIR), which also contains torch and would match
    # torch/csrc/cuda/nccl.h. Skip empty devel paths.
    local -a header_roots=() lib_roots=() root
    for root in "${ROCM_SDK_ROOT}" "${ROCM_DEVEL_PREFIX}" "${ROCM_CORE_PREFIX}" "${ROCM_LIBRARIES_PREFIX}"; do
        [[ -n "${root}" ]] || continue
        header_roots+=("${root}")
    done
    for root in "${ROCM_SDK_ROOT}" "${ROCM_DEVEL_PREFIX}" "${ROCM_LIBRARIES_PREFIX}"; do
        [[ -n "${root}" ]] || continue
        lib_roots+=("${root}")
    done
    [[ "${#header_roots[@]}" -gt 0 ]] || die "No ROCm SDK roots to search for RCCL"
    header="$(find "${header_roots[@]}" \
        \( -type f -o -type l \) \( -name 'nccl.h' -o -name 'rccl.h' \) \
        -print -quit 2>/dev/null || true)"
    lib="$(find "${lib_roots[@]}" \
        \( -type f -o -type l \) -name 'librccl.so*' \
        -print -quit 2>/dev/null || true)"

    if [[ -z "${header}" ]]; then
        # Some SDK layouts ship librccl without the public header (e.g. the
        # ROCm 10.0 wheels). The bundled library matches the rocm-systems pin,
        # so generate rccl.h from the pinned source template.
        clone_rocm_systems
        header="$(generate_rccl_header_from_source)"
        [[ -n "${header}" ]] || die "Failed to generate RCCL header from ${ROCM_SYSTEMS_SRC_DIR}"
    fi
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
    # rccl-tests and other consumers include <rccl/rccl.h>, the installed-tree
    # layout that a real rccl build produces. Provide both include styles, and
    # expose them through the compat include root (/opt/rocm/include, not
    # ROCM_SDK_ROOT which points at the wheel prefix in devel-less layouts)
    # so CMake consumers that only use the SDK include path also find them.
    ln -s "${include_root}" "${RCCL_INCLUDE_DIR}/rccl" 2>/dev/null || true
    if [[ -d /opt/rocm/include ]]; then
        # Expose the generated public headers (rccl.h plus the installed-tree
        # include set such as nccl_device.h) through the compat include root.
        local rccl_header
        for rccl_header in "${include_root}"/*.h; do
            [[ -e "${rccl_header}" ]] || continue
            ln -sf "${rccl_header}" "/opt/rocm/include/$(basename "${rccl_header}")" 2>/dev/null || true
        done
        ln -sfn "${include_root}" /opt/rocm/include/rccl 2>/dev/null || true
        ln -sfn "${include_root}/nccl_device" /opt/rocm/include/nccl_device 2>/dev/null || true
    fi
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

    # Compiler layouts differ between SDK versions: the devel tree provides
    # bin/amdclang++, the wheel-only layout keeps clang++ under lib/llvm/bin.
    local rccl_cxx=""
    for candidate in "${ROCM_SDK_BIN}/amdclang++" "${ROCM_CORE_PREFIX}/lib/llvm/bin/clang++"; do
        [[ -x "${candidate}" ]] && { rccl_cxx="${candidate}"; break; }
    done
    [[ -n "${rccl_cxx}" ]] || die "No HIP C++ compiler found for RCCL build"

    cmake -S "${ROCM_SYSTEMS_SRC_DIR}/projects/rccl" -B "${RCCL_BUILDDIR}" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${RCCL_PREFIX}" \
        -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}" \
        -DGPU_TARGETS="${RCCL_GPU_TARGETS}" \
        -DCMAKE_HIP_COMPILER="${rccl_cxx}"
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
    # The linker does not consult the runtime loader cache for -l resolution,
    # so libfabric's static pkg-config dependencies (-lhsa-runtime64 from the
    # ROCR linkage) need an explicit -L for the wheel-based SDK layout.
    LDFLAGS="${LDFLAGS:-} -L${ROCM_BUILD_PREFIX}/lib" \
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
    # The hipified test sources use CUDA device intrinsics that only resolve
    # when the compiler runs in HIP mode (plain amdclang++ C++ compiles do not
    # define __HIP__, which gates those declarations in the HIP headers).
    # Compile only the hipified sources as HIP; global CXX flags would break
    # CMake's own compiler/Threads probes.
    # The hipified rccl device headers call CUDA-style unqualified min/max
    # in device code; provide them via a forced include on the test target.
    cat > /tmp/rccl-tests-force-include.h <<'FORCEINC'
#include <algorithm>
using std::min;
using std::max;
FORCEINC
    RCCL_TESTS_GPU_TARGETS="${RCCL_TESTS_GPU_TARGETS}" python3 - "${ROCM_SYSTEMS_SRC_DIR}/projects/rccl-tests/src/CMakeLists.txt" <<'PYEOF'
import os
import sys

path = sys.argv[1]
src = open(path).read()
anchor = "add_library(rccl_common OBJECT ${HIP_COMMON_SOURCES})"
if anchor not in src:
    sys.exit(f"rccl_common anchor not found in {path}")
targets = os.environ["RCCL_TESTS_GPU_TARGETS"].replace(";", ",")
addition = (
    anchor
    + f"\ntarget_compile_options(rccl_common PRIVATE -x hip --offload-arch={targets}"
    + " -include /tmp/rccl-tests-force-include.h)\n"
    # Directory-level options for the *_perf executables created later: the
    # hipified sources and the common object library both need HIP mode.
    + f"\nadd_compile_options(-x hip --offload-arch={targets}"
    + " -include /tmp/rccl-tests-force-include.h)\n"
)
src = src.replace(anchor, addition, 1)
open(path, "w").write(src)
PYEOF
    if ! grep -q 'target_compile_options(rccl_common' "${ROCM_SYSTEMS_SRC_DIR}/projects/rccl-tests/src/CMakeLists.txt"; then
        echo "ERROR: rccl-tests CMakeLists patch not applied" >&2
        exit 1
    fi
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
    pushd "/tmp/osu-micro-benchmarks-${OSU_VERSION}" > /dev/null || return 1
    CC="${OMPI_PREFIX}/bin/mpicc" \
    CXX="${OMPI_PREFIX}/bin/mpicxx" \
    CFLAGS="-O3" \
    ./configure \
        --prefix=/usr/local \
        --enable-rocm \
        --with-rocm="${ROCM_BUILD_PREFIX}"
    make -j"$(make_jobs)"
    make install
    popd > /dev/null || return 1
    rm -rf "/tmp/osu-micro-benchmarks-${OSU_VERSION}" /tmp/osu.tar.gz "${ROCM_SYSTEMS_SRC_DIR:-/tmp/rocm-systems}" "${RCCL_BUILDDIR:-/tmp/rccl-build}"
    ldconfig
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    die "This file provides ROCm stack build functions; run install-alps-rocm-stack.sh instead."
fi
