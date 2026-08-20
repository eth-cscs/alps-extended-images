#!/usr/bin/env bash
set -euo pipefail

# Depends on skopeo.sh being sourced for registry/marker helpers,
# and IMAGE_PREFIX being set in CI variables.

# Canonical ref and hash helpers shared by CI and manual builds.
#
# Tag model:
# - canonical tags include a deterministic content hash and are build outputs;
# - tested marker tags prove a canonical digest passed the current validation;
# - stable tags are promotion targets and must never be used as app build bases.
#
# Content hashes intentionally exclude timestamps and commit SHAs. OCI labels may
# carry CI metadata, but refs should only change when logical build inputs change.

# Iterate paths
_paths_iter() {
  # usage: _paths_iter "path1 path2 ..."
  local paths="${1:-}"
  # split on whitespace
  for p in $paths; do
    printf '%s\n' "$p"
  done
}

# Emits sha256sum lines for all files under paths in stable order.
# usage: hash_paths_stream "path1 path2 ..."
hash_paths_stream() {
  local paths="${1:?paths required}"

  # Expand directories to files, keep a stable list, then hash.
  # sort -u ensures stable uniqueness if two roots overlap.
  _paths_iter "$paths" | while read -r p; do
    if [[ -d "$p" ]]; then
      find "$p" -type f -print
    else
      printf '%s\n' "$p"
    fi
  done | sed '/^$/d' | sort -u | while read -r f; do
    [[ -f "$f" ]] || continue
    sha256sum "$f"
  done
}

# usage: vars_blob "VAR1 VAR2 ..."
vars_blob() {
  local names="${1:?var list required}"
  for n in $names; do
    printf '%s=%s\n' "$n" "${!n-}"
  done | sort
}

# Hash file inputs and selected variables into the short tag hash format used by
# both content hashes and validation hashes.
# usage: short_input_hash "paths..." ["vars..."]
short_input_hash() {
  local paths="${1:?paths required}"
  local vars_to_hash="${2:-}"

  {
    hash_paths_stream "$paths"
    if [[ -n "$vars_to_hash" ]]; then
      vars_blob "$vars_to_hash"
    fi
  } | sha256sum | awk '{print $1}' | cut -c1-16
}

# Hash content that affects image build output. The stream includes file hashes
# in stable path order and selected logical variables, then truncates the sha256
# for tag readability.
# usage: content_hash "paths..." "vars..."
content_hash() {
  short_input_hash "${1:?paths required}" "${2:?vars required}"
}

validation_hash() {
  short_input_hash "${1:?paths required}" "${2:-}"
}

# Validation hashes are separate from image content hashes. They cover test
# declarations and the templates/helpers that implement testing. This lets CI
# re-test an unchanged canonical image when validation logic changes, without
# forcing a rebuild of the image itself.
base_validation_hash() {
  local family="${1:?family required}"
  local name="${2:?name required}"
  local variant="${3:?variant required}"
  local ci_file

  case "$family" in
    cuda) ci_file="Alps-Images/NGC/ci.yaml" ;;
    rocm) ci_file="Alps-Images/ROCm/ci.yaml" ;;
    *) echo "ERROR: unsupported base family for validation hash: $family" >&2; return 1 ;;
  esac

  validation_hash "$ci_file ci-pipelines/child-templates.yaml ci-pipelines/helpers/generate-child-pipeline.py ci-pipelines/helpers/vetnode-config.yaml" "family name variant"
}

app_validation_hash() {
  local app_name="${1:?app name required}"
  local app_variant="${2:?app variant required}"
  local app_dir="Alps-Images/apps/${app_name}"
  local ci_file="${app_dir}/ci.yaml"
  local profile_file="${app_dir}/profile.env"
  local paths="$ci_file ci-pipelines/child-templates.yaml ci-pipelines/helpers/generate-child-pipeline.py ci-pipelines/helpers/vetnode-config.yaml"

  [[ -f "$ci_file" ]] || { echo "ERROR: missing $ci_file" >&2; return 1; }
  [[ -f "$profile_file" ]] || { echo "ERROR: missing $profile_file" >&2; return 1; }

  local APP_VARIANTS="" COMMON_TEST_DIR="" COMMON_PATCH_DIR=""
  local variant_upper="${app_variant^^}"
  local test_dir_var="${variant_upper}_TEST_DIR"
  local patch_dir_var="${variant_upper}_PATCH_DIR"
  local "$test_dir_var" "$patch_dir_var"
  # shellcheck disable=SC1090
  source "$profile_file"

  if [[ -n "$COMMON_TEST_DIR" ]]; then
    paths="$paths ${app_dir}/${COMMON_TEST_DIR}"
  fi
  if [[ -n "${!test_dir_var-}" ]]; then
    paths="$paths ${app_dir}/${!test_dir_var}"
  fi
  if [[ -n "$COMMON_PATCH_DIR" ]]; then
    paths="$paths ${app_dir}/${COMMON_PATCH_DIR}"
  fi
  if [[ -n "${!patch_dir_var-}" ]]; then
    paths="$paths ${app_dir}/${!patch_dir_var}"
  fi

  validation_hash "$paths" "app_name app_variant"
}

# usage: canon_tag_for TAG HASH
canon_tag_for() {
  printf '%s-%s\n' "$1" "$2"
}

# usage: img_ref NAME TAG
img_ref() {
  : "${IMAGE_PREFIX:?IMAGE_PREFIX must be set}"
  printf '%s/%s:%s\n' "$IMAGE_PREFIX" "$1" "$2"
}

# usage: image_refs NAME TAG HASH
# Returns: CANON_REF STABLE_REF
image_refs() {
  local name="${1:?name required}"
  local tag="${2:?tag required}"
  local hash="${3:?hash required}"
  local canon_tag

  canon_tag="$(canon_tag_for "$tag" "$hash")"
  printf '%s %s\n' "$(img_ref "$name" "$canon_tag")" "$(img_ref "$name" "$tag")"
}

# Parse BASE_IMAGE like:
#   pytorch-cuda:25.12-py3 -> prints "ngc pytorch 25.12-py3"
#   pytorch-rocm:rocm7.14-ubuntu24.04-py3.12-torch2.11 -> prints "rocm pytorch rocm7.14-ubuntu24.04-py3.12-torch2.11"
parse_base_image() {
  local base="${1:?BASE_IMAGE required}"
  local repo="${base%%:*}" tag="${base#*:}"

  case "$repo" in
    *-cuda)
      printf 'ngc %s %s\n' "${repo%-cuda}" "$tag"
      ;;
    *-rocm)
      printf 'rocm %s %s\n' "${repo%-rocm}" "$tag"
      ;;
    *)
      echo "ERROR: expected BASE_IMAGE like <name>-cuda:<tag> or <name>-rocm:<tag>, got: $base" >&2
      return 1
      ;;
  esac
}

load_base_ref_vars() {
  local family="${1:?family required}"
  local name="${2:?name required}"
  local variant="${3:?variant required}"

  case "$family" in
    ngc)
      read -r BASE_IMAGE_REF REMOVE_HPCX_DIRS_B64 DOCKERFILE CANON_IMAGE_REF STABLE_IMAGE_REF < <(ngc_base_refs "$name" "$variant")
      ;;
    rocm)
      REMOVE_HPCX_DIRS_B64=""
      read -r BASE_IMAGE_REF DOCKERFILE CANON_IMAGE_REF STABLE_IMAGE_REF < <(rocm_base_refs "$name" "$variant")
      ;;
    *)
      echo "ERROR: unsupported base image family: $family" >&2
      return 1
      ;;
  esac
}

rocm_profile_file() {
  local rocm_name="${1:?rocm_name required}"
  local rocm_variant="${2:?rocm_variant required}"

  printf 'Alps-Images/ROCm/%s-%s/profile.env\n' "$rocm_name" "$rocm_variant"
}

rocm_version_from_variant() {
  local rocm_variant="${1:?rocm_variant required}"

  if [[ "$rocm_variant" =~ ^rocm([0-9]+\.[0-9]+)- ]]; then
    printf '%s.0\n' "${BASH_REMATCH[1]}"
    return 0
  fi

  echo "ERROR: can not derive ROCm version from variant: ${rocm_variant}" >&2
  return 1
}

rocm_base_image_ref() {
  local rocm_name="${1:?rocm_name required}"
  local rocm_variant="${2:?rocm_variant required}"

  case "$rocm_name" in
    pytorch)
      if [[ "$rocm_variant" =~ ^rocm([0-9]+\.[0-9]+)-ubuntu([0-9]+\.[0-9]+)-py([0-9]+\.[0-9]+)-torch([0-9]+\.[0-9]+)$ ]]; then
        printf 'docker.io/rocm/pytorch:rocm%s_ubuntu%s_py%s_pytorch_release_%s.0\n' \
          "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}" "${BASH_REMATCH[3]}" "${BASH_REMATCH[4]}"
        return 0
      fi
      ;;
  esac

  echo "ERROR: can not derive ROCm base image ref from ${rocm_name}:${rocm_variant}" >&2
  return 1
}

validate_rocm_profile() {
  local profile_file="${1:?profile_file required}"

  [[ -n "${ROCM_PYPI_INDEX_URL:-}" ]] || { echo "ERROR: ROCM_PYPI_INDEX_URL must be set in $profile_file" >&2; return 1; }
  [[ "${ROCM_REBUILD_RCCL:-0}" == "0" || "${ROCM_REBUILD_RCCL:-0}" == "1" ]] || { echo "ERROR: ROCM_REBUILD_RCCL must be 0 or 1 in $profile_file" >&2; return 1; }
  [[ -n "${ROCM_SYSTEMS_REPO:-}" ]] || { echo "ERROR: ROCM_SYSTEMS_REPO must be set in $profile_file" >&2; return 1; }
  [[ -n "${ROCM_SYSTEMS_COMMIT:-}" ]] || { echo "ERROR: ROCM_SYSTEMS_COMMIT must be set in $profile_file" >&2; return 1; }
  [[ -n "${RCCL_GPU_TARGETS:-}" ]] || { echo "ERROR: RCCL_GPU_TARGETS must be set in $profile_file" >&2; return 1; }
  [[ -n "${RCCL_TESTS_GPU_TARGETS:-}" ]] || { echo "ERROR: RCCL_TESTS_GPU_TARGETS must be set in $profile_file" >&2; return 1; }
}

load_rocm_profile() {
  local profile_file="${1:?profile_file required}"

  ROCM_PYPI_INDEX_URL=""
  ROCM_REBUILD_RCCL="0"
  ROCM_SYSTEMS_REPO=""
  ROCM_SYSTEMS_COMMIT=""
  RCCL_GPU_TARGETS=""
  RCCL_TESTS_GPU_TARGETS=""
  # shellcheck disable=SC1090
  source "$profile_file"
  validate_rocm_profile "$profile_file"
}

rocm_base_build_args() {
  local rocm_variant_dir="${1:?rocm_variant_dir required}"

  cat <<EOF
  --build-arg ROCM_VARIANT_DIR="$rocm_variant_dir" \\
  --build-arg ROCM_VERSION="$ROCM_VERSION" \\
  --build-arg ROCM_REBUILD_RCCL="$ROCM_REBUILD_RCCL" \\
  --build-arg ROCM_SYSTEMS_REPO="$ROCM_SYSTEMS_REPO" \\
  --build-arg ROCM_SYSTEMS_COMMIT="$ROCM_SYSTEMS_COMMIT" \\
  --build-arg RCCL_GPU_TARGETS="$RCCL_GPU_TARGETS" \\
  --build-arg RCCL_TESTS_GPU_TARGETS="$RCCL_TESTS_GPU_TARGETS" \\
  --build-arg ROCM_PYPI_INDEX_URL="$ROCM_PYPI_INDEX_URL" \\
EOF
}

rocm_base_dotenv() {
  local rocm_variant_dir="${1:?rocm_variant_dir required}"

  cat <<EOF
ROCM_VARIANT_DIR=$rocm_variant_dir
ROCM_VERSION=$ROCM_VERSION
ROCM_REBUILD_RCCL=$ROCM_REBUILD_RCCL
ROCM_SYSTEMS_REPO=$ROCM_SYSTEMS_REPO
ROCM_SYSTEMS_COMMIT=$ROCM_SYSTEMS_COMMIT
RCCL_GPU_TARGETS=$RCCL_GPU_TARGETS
RCCL_TESTS_GPU_TARGETS=$RCCL_TESTS_GPU_TARGETS
ROCM_PYPI_INDEX_URL=$ROCM_PYPI_INDEX_URL
EOF
}

# Usage: rocm_base_refs NAME ROCM_VARIANT
# Returns: BASE_IMAGE_REF DOCKERFILE CANON_REF STABLE_REF
rocm_base_refs() {
  local rocm_name="${1:?rocm_name required}"       # e.g. pytorch
  local rocm_variant="${2:?rocm_variant required}" # e.g. rocm7.14-ubuntu24.04-py3.12-torch2.11

  : "${ALPS_REV:?ALPS_REV must be set}"
  : "${CSCS_CI_ORIG_CLONE_URL:?CSCS_CI_ORIG_CLONE_URL must be set}"

  local image_dir="Alps-Images/ROCm/${rocm_name}-${rocm_variant}"
  local profile_file
  profile_file="$(rocm_profile_file "$rocm_name" "$rocm_variant")"
  local dockerfile="Alps-Images/ROCm/Containerfile.rocm-alps"
  local rocm_installer="Alps-Images/ROCm/install-alps-rocm-stack.sh"
  local rocm_components="Alps-Images/ROCm/install-alps-rocm-components.sh"
  local rocm_runtime_env="Alps-Images/ROCm/alps-runtime.rocm.env"
  local common_dir="Alps-Images/common"
  local patches_dir="Alps-Images/patches"

  [[ -d "$image_dir" ]]    || { echo "ERROR: missing $image_dir" >&2; return 1; }
  [[ -f "$profile_file" ]] || { echo "ERROR: missing $profile_file" >&2; return 1; }
  [[ -f "$dockerfile" ]]   || { echo "ERROR: missing $dockerfile" >&2; return 1; }
  [[ -f "$rocm_installer" ]] || { echo "ERROR: missing $rocm_installer" >&2; return 1; }
  [[ -f "$rocm_components" ]] || { echo "ERROR: missing $rocm_components" >&2; return 1; }
  [[ -f "$rocm_runtime_env" ]] || { echo "ERROR: missing $rocm_runtime_env" >&2; return 1; }
  [[ -d "$common_dir" ]]   || { echo "ERROR: missing $common_dir" >&2; return 1; }
  [[ -d "$patches_dir" ]]  || { echo "ERROR: missing $patches_dir" >&2; return 1; }

  local ROCM_VERSION ROCM_PYPI_INDEX_URL ROCM_REBUILD_RCCL
  local ROCM_SYSTEMS_REPO ROCM_SYSTEMS_COMMIT RCCL_GPU_TARGETS RCCL_TESTS_GPU_TARGETS
  load_rocm_profile "$profile_file"
  ROCM_VERSION="$(rocm_version_from_variant "$rocm_variant")"
  local base_image_ref
  base_image_ref="$(rocm_base_image_ref "$rocm_name" "$rocm_variant")"

  local hash_paths="$dockerfile $rocm_installer $rocm_components $rocm_runtime_env $common_dir $patches_dir $image_dir"
  local name="${rocm_name}-rocm"
  local tag="${rocm_variant}-${ALPS_REV}"
  local h
  h="$(content_hash "$hash_paths" "name tag base_image_ref ROCM_VERSION ROCM_PYPI_INDEX_URL ROCM_REBUILD_RCCL ROCM_SYSTEMS_REPO ROCM_SYSTEMS_COMMIT RCCL_GPU_TARGETS RCCL_TESTS_GPU_TARGETS CSCS_CI_ORIG_CLONE_URL")"
  local canon_ref stable_ref
  read -r canon_ref stable_ref < <(image_refs "$name" "$tag" "$h")

  printf '%s %s %s %s\n' \
    "$base_image_ref" "$dockerfile" "$canon_ref" "$stable_ref"
}

# Usage: ngc_base_refs NGC_NAME NGC_TAG
ngc_base_refs() {
  local ngc_name="${1:?ngc_name required}"   # e.g. pytorch
  local ngc_tag="${2:?ngc_tag required}"     # e.g. 25.12-py3

  : "${ALPS_REV:?ALPS_REV must be set}"
  : "${CSCS_CI_ORIG_CLONE_URL:?CSCS_CI_ORIG_CLONE_URL must be set}"

  local image_dir="Alps-Images/NGC/${ngc_name}-${ngc_tag}"
  local profile_file="${image_dir}/profile.env"
  local dockerfile="Alps-Images/NGC/Containerfile.ngc-alps"
  local ngc_installer="Alps-Images/NGC/install-alps-cuda-stack.sh"
  local ngc_components="Alps-Images/NGC/install-alps-cuda-components.sh"
  local ngc_runtime_env="Alps-Images/NGC/alps-runtime.cuda.env"
  local common_dir="Alps-Images/common"
  local patches_dir="Alps-Images/patches"

  [[ -d "$image_dir" ]]    || { echo "ERROR: missing $image_dir" >&2; return 1; }
  [[ -f "$profile_file" ]] || { echo "ERROR: missing $profile_file" >&2; return 1; }
  [[ -f "$dockerfile" ]]   || { echo "ERROR: missing $dockerfile" >&2; return 1; }
  [[ -f "$ngc_installer" ]] || { echo "ERROR: missing $ngc_installer" >&2; return 1; }
  [[ -f "$ngc_components" ]] || { echo "ERROR: missing $ngc_components" >&2; return 1; }
  [[ -f "$ngc_runtime_env" ]] || { echo "ERROR: missing $ngc_runtime_env" >&2; return 1; }
  [[ -d "$common_dir" ]]   || { echo "ERROR: missing $common_dir" >&2; return 1; }
  [[ -d "$patches_dir" ]]  || { echo "ERROR: missing $patches_dir" >&2; return 1; }

  # Load optional NGC variant settings from profile file.
  local REMOVE_HPCX_DIRS=""
  local NVCR_PREFIX="nvidia"
  # shellcheck disable=SC1090
  source "$profile_file"
  REMOVE_HPCX_DIRS_B64="$(printf '%s' "$REMOVE_HPCX_DIRS" | base64 -w0)"
  # Keep the space-separated helper record parseable when the override is empty.
  # Command substitution strips the decoded newline, yielding an empty value in CI.
  [[ -n "$REMOVE_HPCX_DIRS_B64" ]] || REMOVE_HPCX_DIRS_B64="Cg=="

  # Some nvcr images have a different repo structure, e.g. physicsnemo:
  # nvcr.io/nvidia/physicsnemo/physicsnemo:25.11
  # but most are like nvcr.io/nvidia/pytorch:25.12-py3, so we need to handle both cases.
  # NVCR_PREFIX defaults to "nvidia" unless overridden by the profile file.

  # BASE IMAGE points to NGC image via remote proxy (jfrog) (speed up downloads
  # in CI and avoid hitting NGC rate limits)
  local base_image_ref="jfrog.svc.cscs.ch/nvcr/${NVCR_PREFIX}/${ngc_name}:${ngc_tag}"

  # Compute canonical tag from hashed content
  local hash_paths="$dockerfile $ngc_installer $ngc_components $ngc_runtime_env $common_dir $patches_dir $image_dir"
  local name="${ngc_name}-cuda"
  local tag="${ngc_tag}-${ALPS_REV}"
  local h
  h="$(content_hash "$hash_paths" "name tag CSCS_CI_ORIG_CLONE_URL")"
  local canon_ref stable_ref
  read -r canon_ref stable_ref < <(image_refs "$name" "$tag" "$h")

  printf '%s %s %s %s %s\n' \
    "$base_image_ref" "$REMOVE_HPCX_DIRS_B64" "$dockerfile" "$canon_ref" "$stable_ref"
}

validate_app_variant_name() {
  local app_variant="${1:?app variant required}"
  local context="${2:-app variant}"

  [[ "$app_variant" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || {
    echo "ERROR: ${context} '${app_variant}' must match [A-Za-z_][A-Za-z0-9_]* because app metadata uses shell variable names" >&2
    return 1
  }
}

write_base_build_env() {
  local output_file="${1:?output file required}"
  local family="${2:?family required}"
  local name="${3:?name required}"
  local variant="${4:?variant required}"
  local base_image_ref dockerfile canon_image_ref stable_image_ref tested_image_ref image_description validation_hash_value
  local family_variant_dir="" family_dotenv=""

  : "${GHCR_IMAGE_PREFIX:?GHCR_IMAGE_PREFIX must be set}"
  : "${CI_COMMIT_SHORT_SHA:?CI_COMMIT_SHORT_SHA must be set}"

  case "$family" in
    cuda|ngc)
      read -r base_image_ref REMOVE_HPCX_DIRS_B64 dockerfile canon_image_ref stable_image_ref < <(ngc_base_refs "$name" "$variant")
      local remove_hpcx_dirs
      remove_hpcx_dirs="$(printf '%s' "$REMOVE_HPCX_DIRS_B64" | base64 -d)"
      family="cuda"
      image_description="This image extends ${base_image_ref} with a fully-optimized HPC networking stack tailored for the Alps supercomputer (legacy HPC-X components might be replaced)."
      family_variant_dir="NGC_VARIANT_DIR=${name}-${variant}"
      family_dotenv="REMOVE_HPCX_DIRS=${remove_hpcx_dirs}"
      ;;
    rocm)
      read -r base_image_ref dockerfile canon_image_ref stable_image_ref < <(rocm_base_refs "$name" "$variant")
      local profile_file ROCM_VERSION ROCM_PYPI_INDEX_URL ROCM_REBUILD_RCCL
      local ROCM_SYSTEMS_REPO ROCM_SYSTEMS_COMMIT RCCL_GPU_TARGETS RCCL_TESTS_GPU_TARGETS
      profile_file="$(rocm_profile_file "$name" "$variant")"
      load_rocm_profile "$profile_file"
      ROCM_VERSION="$(rocm_version_from_variant "$variant")"
      image_description="This image extends ${base_image_ref} with a fully-optimized ROCm HPC networking stack tailored for the Alps supercomputer."
      family_dotenv="$(rocm_base_dotenv "${name}-${variant}")"
      ;;
    *)
      echo "ERROR: unsupported base family: $family" >&2
      return 1
      ;;
  esac

  # The generated child pipeline consumes one env record per variant. Keeping
  # this assembly in shell preserves identical canonical refs between CI,
  # publishing, and manual-build script generation.
  validation_hash_value="$(base_validation_hash "$family" "$name" "$variant")"
  tested_image_ref="$(tested_ref_for "$canon_image_ref" "$validation_hash_value")"

  {
    printf '%s\n' \
      "FAMILY=$family" \
      "NAME=$name" \
      "VARIANT=$variant" \
      "DOCKERFILE=$dockerfile" \
      "CANON_IMAGE_REF=$canon_image_ref" \
      "STABLE_IMAGE_REF=$stable_image_ref" \
      "TESTED_IMAGE_REF=$tested_image_ref" \
      "VALIDATION_HASH=$validation_hash_value" \
      "BASE_IMAGE=$base_image_ref" \
      "$family_variant_dir"
    printf '%s\n' "$family_dotenv"
    # shellcheck source=Alps-Images/common/alps-stack-versions.env
    source Alps-Images/common/alps-stack-versions.env
    vars_blob "BOOST_VER BOOST_BUILD_JOBS ALPS_BUILD_JOBS ALPS_CMAKE_BUILD_JOBS XPMEM_REF CASSINI_HEADERS_VERSION CXI_DRIVER_VERSION LIBCXI_VERSION LIBFABRIC_COMMIT LIBFABRIC_PATCH UCX_VERSION UCC_VERSION OMPI_VER AWS_OFI_NCCL_REPO AWS_OFI_NCCL_COMMIT AWS_OFI_NCCL_PATCH OSU_VERSION"
    printf '%s\n' \
      "OCI_SOURCE=${CSCS_CI_ORIG_CLONE_URL}" \
      "OCI_REVISION=${CI_COMMIT_SHA:-$CI_COMMIT_SHORT_SHA}" \
      "OCI_DESCRIPTION=$image_description" \
      "CSCS_ALPS_GIT_COMMIT_SHORT=${CI_COMMIT_SHORT_SHA}" \
      "GHCR_STABLE_IMAGE_REF=${GHCR_IMAGE_PREFIX}${stable_image_ref#"$IMAGE_PREFIX"}"
  } | sed '/^$/d' > "$output_file"
}

write_app_build_env() {
  local output_file="${1:?output file required}"
  local app_name="${2:?app name required}"
  local app_variant="${3:?app variant required}"
  local base_image_ref dockerfile canon_image_ref stable_image_ref tested_image_ref image_description validation_hash_value

  : "${GHCR_IMAGE_PREFIX:?GHCR_IMAGE_PREFIX must be set}"
  : "${CI_COMMIT_SHORT_SHA:?CI_COMMIT_SHORT_SHA must be set}"

  # App canonical hashes include the canonical base ref, so app images rebuild
  # automatically when their selected base content changes.
  read -r base_image_ref dockerfile canon_image_ref stable_image_ref < <(app_refs "$app_name" "$app_variant")
  validation_hash_value="$(app_validation_hash "$app_name" "$app_variant")"
  tested_image_ref="$(tested_ref_for "$canon_image_ref" "$validation_hash_value")"
  image_description="This image extends ${base_image_ref} with application software for Alps."

  cat > "${output_file}" <<EOF
DOCKERFILE=$dockerfile
CANON_IMAGE_REF=$canon_image_ref
STABLE_IMAGE_REF=$stable_image_ref
TESTED_IMAGE_REF=$tested_image_ref
VALIDATION_HASH=$validation_hash_value
BASE_IMAGE=$base_image_ref
OCI_SOURCE=${CSCS_CI_ORIG_CLONE_URL}
OCI_REVISION=${CI_COMMIT_SHA:-$CI_COMMIT_SHORT_SHA}
OCI_DESCRIPTION=$image_description
CSCS_ALPS_GIT_COMMIT_SHORT=${CI_COMMIT_SHORT_SHA}
GHCR_STABLE_IMAGE_REF=${GHCR_IMAGE_PREFIX}${stable_image_ref#"$IMAGE_PREFIX"}
EOF
}

# Usage: app_refs APP_NAME APP_VARIANT
# Returns a space-separated record:
#   BASE_IMAGE_REF DOCKERFILE CANON_REF STABLE_REF
app_refs() {
  local app_name="${1:?name required}"       # e.g. apertus-2
  local app_variant="${2:?app variant required}" # e.g. cuda

  : "${ALPS_REV:?ALPS_REV must be set}"
  : "${CSCS_CI_ORIG_CLONE_URL:?CSCS_CI_ORIG_CLONE_URL must be set}"

  local app_dir="Alps-Images/apps/${app_name}"
  local profile_file="${app_dir}/profile.env"
  local package_helpers="Alps-Images/common/package-helpers.sh"
  local sources_dir="${app_dir}/sources"
  validate_app_variant_name "$app_variant" "requested app variant"
  local variant_upper="${app_variant^^}"
  local APP_VARIANTS=""
  local COMMON_CONTAINERFILE=""
  local COMMON_TEST_DIR=""
  local COMMON_PATCH_DIR=""
  local base_image_var="${variant_upper}_BASE_IMAGE"
  local dockerfile_var="${variant_upper}_CONTAINERFILE"
  local test_dir_var="${variant_upper}_TEST_DIR"
  local patch_dir_var="${variant_upper}_PATCH_DIR"
  local "$base_image_var" "$dockerfile_var" "$test_dir_var" "$patch_dir_var"

  [[ -d "$app_dir" ]]      || { echo "ERROR: missing $app_dir" >&2; return 1; }
  [[ -f "$profile_file" ]] || { echo "ERROR: missing $profile_file" >&2; return 1; }

  # Load variant metadata from profile file.
  # shellcheck disable=SC1090
  source "$profile_file"

  local declared_variant
  for declared_variant in $APP_VARIANTS; do
    validate_app_variant_name "$declared_variant" "declared app variant in ${profile_file}"
  done

  case " ${APP_VARIANTS} " in
    *" ${app_variant} "*) ;;
    *) echo "ERROR: app variant ${app_variant} is not declared in ${profile_file}" >&2; return 1;;
  esac

  local base_image="${!base_image_var-}"
  local dockerfile_rel="${!dockerfile_var:-${COMMON_CONTAINERFILE:-Containerfile}}"
  local test_dir_rel="${!test_dir_var-}"
  local patch_dir_rel="${!patch_dir_var-}"
  local dockerfile test_dirs="" patch_dirs=""

  [[ -n "$base_image" ]] || { echo "ERROR: ${base_image_var} must be set in ${profile_file}" >&2; return 1; }
  dockerfile="${app_dir}/${dockerfile_rel}"
  [[ -f "$dockerfile" ]] || { echo "ERROR: missing $dockerfile" >&2; return 1; }

  if [[ -n "$COMMON_TEST_DIR" ]]; then
    local common_test_dir="${app_dir}/${COMMON_TEST_DIR}"
    [[ -d "$common_test_dir" ]] || { echo "ERROR: missing declared common test dir $common_test_dir" >&2; return 1; }
    test_dirs="$test_dirs $common_test_dir"
  fi
  if [[ -n "$test_dir_rel" ]]; then
    local test_dir="${app_dir}/${test_dir_rel}"
    [[ -d "$test_dir" ]] || { echo "ERROR: missing declared test dir $test_dir" >&2; return 1; }
    test_dirs="$test_dirs $test_dir"
  fi
  if [[ -n "$COMMON_PATCH_DIR" ]]; then
    local common_patch_dir="${app_dir}/${COMMON_PATCH_DIR}"
    [[ -d "$common_patch_dir" ]] || { echo "ERROR: missing declared common patch dir $common_patch_dir" >&2; return 1; }
    patch_dirs="$patch_dirs $common_patch_dir"
  fi
  if [[ -n "$patch_dir_rel" ]]; then
    local patch_dir="${app_dir}/${patch_dir_rel}"
    [[ -d "$patch_dir" ]] || { echo "ERROR: missing declared patch dir $patch_dir" >&2; return 1; }
    patch_dirs="$patch_dirs $patch_dir"
  fi

  # Compute canonical ref of base
  local base_family base_name base_variant
  read -r base_family base_name base_variant < <(parse_base_image "$base_image")
  # load_base_ref_vars fills the shared base-ref variables, but app hashing only
  # needs CANON_IMAGE_REF from that record.
  # shellcheck disable=SC2034
  local BASE_IMAGE_REF REMOVE_HPCX_DIRS_B64 DOCKERFILE CANON_IMAGE_REF STABLE_IMAGE_REF
  load_base_ref_vars "$base_family" "$base_name" "$base_variant"
  local base_canon_ref="$CANON_IMAGE_REF"

  # Compute canonical tag from hashed content
  local hash_paths="$dockerfile $profile_file $patch_dirs $test_dirs"
  if [[ -d "$sources_dir" ]]; then
    hash_paths="$hash_paths $sources_dir"
  fi
  if grep -Fq "$package_helpers" "$dockerfile"; then
    [[ -f "$package_helpers" ]] || { echo "ERROR: missing $package_helpers" >&2; return 1; }
    hash_paths="$hash_paths $package_helpers"
  fi
  local image_name="${app_name}-${app_variant}"
  local tag="${ALPS_REV}"
  local h
  h="$(content_hash "$hash_paths" "app_name app_variant image_name tag base_canon_ref CSCS_CI_ORIG_CLONE_URL")"
  local canon_ref stable_ref
  read -r canon_ref stable_ref < <(image_refs "$image_name" "$tag" "$h")

  printf '%s %s %s %s\n' "$base_canon_ref" "$dockerfile" "$canon_ref" "$stable_ref"
}
