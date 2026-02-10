#!/usr/bin/env bash
set -euo pipefail

# Depends on skopeo.sh being sourced (for img_exists),
# and IMAGE_PREFIX being set in CI variables.

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

# usage: content_hash "paths..." "vars..."
content_hash() {
  local paths="${1:?paths required}"
  local vars_to_hash="${2:?vars required}"
  {
    hash_paths_stream "$paths"
    vars_blob "$vars_to_hash"
  } | sha256sum | awk '{print $1}' | cut -c1-16
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

# Parse BASE_IMAGE like: ngc-pytorch:25.12-py3  -> prints "pytorch 25.12-py3"
parse_ngc_base_image() {
  local base="${1:?BASE_IMAGE required}"
  local repo="${base%%:*}" tag="${base#*:}"
  [[ "$repo" == ngc-* ]] || { echo "ERROR: expected BASE_IMAGE like ngc-<name>:<tag>, got: $base" >&2; return 1; }
  printf '%s %s\n' "${repo#ngc-}" "$tag"
}

# Usage: base_refs NGC_NAME NGC_TAG
# Returns a space-separated record:
#   BASE_IMAGE_REF REOMVE_HPCX_DIRS_B64 DOCKERFILE CANON_REF TEST_REF STABLE_REF
base_refs() {
  local ngc_name="${1:?ngc_name required}"   # e.g. pytorch
  local ngc_tag="${2:?ngc_tag required}"     # e.g. 25.12-py3

  : "${ALPS_REV:?ALPS_REV must be set}"
  : "${CI_COMMIT_SHORT_SHA:?CI_COMMIT_SHORT_SHA must be set}"

  local profile_file="Alps-Images/NGC/${ngc_name}-${ngc_tag}.env"
  local dockerfile="Alps-Images/NGC/Containerfile.ngc-alps"
  local install_sh="Alps-Images/common/install-alps-hpc-stack.sh"
  local patches_dir="Alps-Images/patches"

  [[ -f "$profile_file" ]]  || { echo "ERROR: missing $profile_file" >&2; return 1; }
  [[ -f "$dockerfile" ]]    || { echo "ERROR: missing $dockerfile" >&2; return 1; }
  [[ -f "$install_sh" ]]    || { echo "ERROR: missing $install_sh" >&2; return 1; }
  [[ -d "$patches_dir" ]]   || { echo "ERROR: missing $patches_dir" >&2; return 1; }

  # Load REMOVE_HPCX_DIRS from profile file
  # shellcheck disable=SC1090
  source "$profile_file"
  : "${REMOVE_HPCX_DIRS:?REMOVE_HPCX_DIRS must be set in ${profile_file}}"
  REMOVE_HPCX_DIRS_B64="$(printf '%s' "$REMOVE_HPCX_DIRS" | base64 -w0)"

  # BASE_IMAGE points to the NGC image we build on top of
  local base_image_ref="nvcr.io/nvidia/${ngc_name}:${ngc_tag}"

  # Compute canonical tag from hashed content
  local hash_paths="$dockerfile $install_sh $patches_dir $profile_file"
  local name="ngc-${ngc_name}"
  local tag="${ngc_tag}-${ALPS_REV}"
  local h="$(content_hash "$hash_paths" "name tag")"
  local canon_tag="$(canon_tag_for "$tag" "$h")"

  local canon_ref="$(img_ref "$name" "$canon_tag")"
  local test_ref="$(img_ref "$name" "${tag}-${CI_COMMIT_SHORT_SHA}")"
  local stable_ref="$(img_ref "$name" "$tag")"

  printf '%s %s %s %s %s %s %s\n' \
    "$base_image_ref" "$REMOVE_HPCX_DIRS_B64" "$dockerfile" "$canon_ref" "$test_ref" "$stable_ref"
}

# Usage: app_refs APP_NAME
# Returns a space-separated record:
#   BASE_IMAGE_REF DOCKERFILE CANON_REF TEST_REF STABLE_REF
app_refs() {
  local name="${1:?name required}"           # e.g. apertus-2

  : "${ALPS_REV:?ALPS_REV must be set}"
  : "${CI_COMMIT_SHORT_SHA:?CI_COMMIT_SHORT_SHA must be set}"

  local app_dir="Alps-Images/apps/${name}"
  local dockerfile="${app_dir}/Containerfile"
  local test_dir="${app_dir}/tests"
  local profile_file="${app_dir}/profile.env"

  [[ -d "$app_dir" ]]      || { echo "ERROR: missing $app_dir" >&2; return 1; }
  [[ -f "$dockerfile" ]]   || { echo "ERROR: missing $dockerfile" >&2; return 1; }
  [[ -f "$profile_file" ]] || { echo "ERROR: missing $profile_file" >&2; return 1; }
  # tests dir is optional but usually present
  [[ -d "$test_dir" ]] || test_dir=""

  # Load BASE_IMAGE from profile file
  # shellcheck disable=SC1090
  source "$profile_file"
  : "${BASE_IMAGE:?BASE_IMAGE must be set in ${profile_file}}"

  # Compute canonical ref of base
  local ngc_name ngc_tag
  read -r ngc_name ngc_tag < <(parse_ngc_base_image "$BASE_IMAGE")
  local _base_image_ref _remove_hpcx_dirs _base_dockerfile base_canon_ref _base_test_ref _base_stable_ref
  read -r _base_image_ref _remove_hpcx_dirs _base_dockerfile base_canon_ref _base_test_ref _base_stable_ref < <(base_refs "$ngc_name" "$ngc_tag")

  # Compute canonical tag from hashed content
  local hash_paths="$dockerfile $profile_file $test_dir"
  local tag="${ALPS_REV}"
  local h="$(content_hash "$hash_paths" "name tag base_canon_ref")"
  local canon_tag="$(canon_tag_for "$tag" "$h")"
  local canon_ref="$(img_ref "$name" "$canon_tag")"
  local test_ref="$(img_ref "$name" "${tag}-${CI_COMMIT_SHORT_SHA}")"
  local stable_ref="$(img_ref "$name" "$tag")"

  printf '%s %s %s %s %s\n' "$base_canon_ref" "$dockerfile" "$canon_ref" "$test_ref" "$stable_ref"
}
