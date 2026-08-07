#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  manual-build/manual-build.sh base FAMILY NAME VARIANT [OUTPUT]
  manual-build/manual-build.sh app APP_NAME APP_VARIANT [OUTPUT]

Emits a standalone podman build script that uses the same image reference
derivation as CI. If OUTPUT is omitted, the script is written to stdout.

Environment overrides:
  IMAGE_PREFIX              default: localhost/alps-images
  ALPS_REV                  default: parsed from CI YAML
  CI_COMMIT_SHORT_SHA       default: current git short SHA
  CSCS_CI_ORIG_CLONE_URL    default: current branch remote URL or local path
EOF
}

repo_root() {
  git rev-parse --show-toplevel 2>/dev/null || pwd
}

default_remote_url() {
  local root branch remote url
  root="$(repo_root)"
  branch="$(git -C "$root" branch --show-current 2>/dev/null || true)"

  if [[ -n "$branch" ]]; then
    remote="$(git -C "$root" config --get "branch.${branch}.remote" 2>/dev/null || true)"
    if [[ -n "$remote" && "$remote" != "." ]]; then
      url="$(git -C "$root" config --get "remote.${remote}.url" 2>/dev/null || true)"
      if [[ -n "$url" ]]; then
        printf '%s\n' "$url"
        return 0
      fi
    fi
  fi

  printf '%s\n' "$root"
}

default_alps_rev() {
  local root="${1:?root required}"
  local ci_yaml="${root}/ci-pipelines/build-alps-extended-images.yaml"
  local line value in_variables=0

  [[ -f "$ci_yaml" ]] || return 1

  while IFS= read -r line; do
    if [[ "$line" =~ ^variables:[[:space:]]*$ ]]; then
      in_variables=1
      continue
    fi

    if [[ "$in_variables" == 1 && "$line" =~ ^[^[:space:]#][^:]*: ]]; then
      break
    fi

    if [[ "$in_variables" == 1 && "$line" =~ ^[[:space:]]+ALPS_REV:[[:space:]]*(.*)$ ]]; then
      value="${BASH_REMATCH[1]}"
      value="${value%%#*}"
      value="${value#"${value%%[![:space:]]*}"}"
      value="${value%"${value##*[![:space:]]}"}"
      if [[ "${#value}" -ge 2 ]]; then
        if [[ "${value:0:1}" == '"' && "${value: -1}" == '"' ]] || [[ "${value:0:1}" == "'" && "${value: -1}" == "'" ]]; then
          value="${value:1:${#value}-2}"
        fi
      fi
      [[ -n "$value" ]] || return 1
      printf '%s\n' "$value"
      return 0
    fi
  done <"$ci_yaml"

  return 1
}

emit_script() {
  local output="${1:-}"
  if [[ -n "$output" ]]; then
    mkdir -p "$(dirname "$output")"
    cat >"$output"
    chmod +x "$output"
    printf 'Wrote %s\n' "$output" >&2
  else
    cat
  fi
}

emit_base_script() {
  local family="${1:?family required}"
  local name="${2:?name required}"
  local variant="${3:?variant required}"
  local root image_description variant_dir family_build_args
  root="$(repo_root)"

  source "$root/ci-pipelines/helpers/meta.sh"
  case "$family" in
    ngc)
      read -r BASE_IMAGE_REF REMOVE_HPCX_DIRS_B64 DOCKERFILE CANON_IMAGE_REF TEST_IMAGE_REF STABLE_IMAGE_REF < <(ngc_base_refs "$name" "$variant")
      local remove_hpcx_dirs
      remove_hpcx_dirs="$(printf '%s' "$REMOVE_HPCX_DIRS_B64" | base64 -d)"
      variant_dir="${name}-${variant}"
      image_description="This image extends ${BASE_IMAGE_REF} with a fully-optimized HPC networking stack tailored for the Alps supercomputer."
      family_build_args="  --build-arg NGC_VARIANT_DIR=\"$variant_dir\" \\
  --build-arg REMOVE_HPCX_DIRS=\"$remove_hpcx_dirs\" \\
"
      ;;
    rocm)
      read -r BASE_IMAGE_REF DOCKERFILE CANON_IMAGE_REF TEST_IMAGE_REF STABLE_IMAGE_REF < <(rocm_base_refs "$name" "$variant")
      variant_dir="${name}-${variant}"
      local profile_file ROCM_BASE_IMAGE_REF ROCM_VERSION ROCM_PYPI_INDEX_URL ROCM_REBUILD_RCCL
      local ROCM_SYSTEMS_REPO ROCM_SYSTEMS_COMMIT RCCL_GPU_TARGETS RCCL_TESTS_GPU_TARGETS
      profile_file="$root/$(rocm_profile_file "$name" "$variant")"
      load_rocm_profile "$profile_file"
      image_description="This image extends ${BASE_IMAGE_REF} with a fully-optimized ROCm HPC networking stack tailored for the Alps supercomputer."
      family_build_args="$(rocm_base_build_args "$variant_dir")"$'\n'
      ;;
    *)
      printf 'ERROR: unsupported base image family for manual builds: %s\n' "$family" >&2
      return 1
      ;;
  esac

  cat <<EOF
#!/usr/bin/env bash
set -euo pipefail

cd "$root"

podman build --format docker \\
  -f "$DOCKERFILE" \\
  -t "$CANON_IMAGE_REF" \\
  -t "$STABLE_IMAGE_REF" \\
  --build-arg BASE_IMAGE="$BASE_IMAGE_REF" \\
${family_build_args}  --build-arg OCI_SOURCE="$CSCS_CI_ORIG_CLONE_URL" \\
  --build-arg OCI_REVISION="${CI_COMMIT_SHA:-$CI_COMMIT_SHORT_SHA}" \\
  --build-arg OCI_CREATED="$(date -u +"%Y-%m-%dT%H:%M:%SZ")" \\
  --build-arg OCI_DESCRIPTION="$image_description" \\
  --build-arg CSCS_ALPS_GIT_COMMIT_SHORT="$CI_COMMIT_SHORT_SHA" \\
  .

printf 'Built %s\n' "$CANON_IMAGE_REF"
printf 'Also tagged %s\n' "$STABLE_IMAGE_REF"
EOF
}

emit_app_script() {
  local app_name="${1:?app_name required}"
  local app_variant="${2:?app_variant required}"
  local root image_description
  root="$(repo_root)"

  source "$root/ci-pipelines/helpers/meta.sh"
  read -r BASE_IMAGE_REF DOCKERFILE CANON_IMAGE_REF TEST_IMAGE_REF STABLE_IMAGE_REF < <(app_refs "$app_name" "$app_variant")
  image_description="This image extends ${BASE_IMAGE_REF} with application software for Alps."

  cat <<EOF
#!/usr/bin/env bash
set -euo pipefail

cd "$root"

podman build --format docker \\
  -f "$DOCKERFILE" \\
  -t "$CANON_IMAGE_REF" \\
  -t "$STABLE_IMAGE_REF" \\
  --build-arg BASE_IMAGE="$BASE_IMAGE_REF" \\
  --build-arg OCI_SOURCE="$CSCS_CI_ORIG_CLONE_URL" \\
  --build-arg OCI_REVISION="${CI_COMMIT_SHA:-$CI_COMMIT_SHORT_SHA}" \\
  --build-arg OCI_CREATED="$(date -u +"%Y-%m-%dT%H:%M:%SZ")" \\
  --build-arg OCI_DESCRIPTION="$image_description" \\
  --build-arg CSCS_ALPS_GIT_COMMIT_SHORT="$CI_COMMIT_SHORT_SHA" \\
  .

printf 'Built %s\n' "$CANON_IMAGE_REF"
printf 'Also tagged %s\n' "$STABLE_IMAGE_REF"
EOF
}

main() {
  local root default_rev
  root="$(repo_root)"

  if [[ $# -lt 1 ]]; then
    usage >&2
    exit 2
  fi

  export IMAGE_PREFIX="${IMAGE_PREFIX:-localhost/alps-images}"
  if [[ -z "${ALPS_REV:-}" ]]; then
    default_rev="$(default_alps_rev "$root")" || {
      printf 'ERROR: ALPS_REV is not set and could not be parsed from CI YAML\n' >&2
      exit 1
    }
    export ALPS_REV="$default_rev"
  else
    export ALPS_REV
  fi
  export CI_COMMIT_SHORT_SHA="${CI_COMMIT_SHORT_SHA:-$(git rev-parse --short HEAD 2>/dev/null || printf manual)}"
  export CI_COMMIT_SHA="${CI_COMMIT_SHA:-$(git rev-parse HEAD 2>/dev/null || printf manual)}"
  export CSCS_CI_ORIG_CLONE_URL="${CSCS_CI_ORIG_CLONE_URL:-$(default_remote_url)}"

  local kind="$1"
  shift

  case "$kind" in
    base)
      [[ $# -ge 3 && $# -le 4 ]] || { usage >&2; exit 2; }
      emit_base_script "$1" "$2" "$3" | emit_script "${4:-}"
      ;;
    app)
      [[ $# -ge 2 && $# -le 3 ]] || { usage >&2; exit 2; }
      emit_app_script "$1" "$2" | emit_script "${3:-}"
      ;;
    -h|--help|help)
      usage
      ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
}

main "$@"
