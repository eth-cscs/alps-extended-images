#!/usr/bin/env bash
set -euo pipefail

skopeo_login() {
  : "${IMAGE_PREFIX:?IMAGE_PREFIX must be set}"
  : "${GITHUB_ACTOR:?GITHUB_ACTOR must be set}"
  : "${GITHUB_TOKEN:?GITHUB_TOKEN must be set}"
  local reg="${IMAGE_PREFIX%%/*}"
  skopeo login --username "${GITHUB_ACTOR}" --password "${GITHUB_TOKEN}" "${reg}" >/dev/null
}

# prints digest or empty string if missing/unreachable
# usage: img_digest REF
img_digest() {
  skopeo inspect --format '{{.Digest}}' "docker://$1" 2>/dev/null || true
}

# usage: img_exists REF
img_exists() {
  [[ -n "$(img_digest "$1")" ]]
}

_ref_url() {
  local ref="$1"
  if [[ "$ref" == docker://* ]]; then
    printf '%s\n' "$ref"
  else
    printf 'docker://%s\n' "$ref"
  fi
}

# copy only if dst is missing or points to a different digest
# usage: copy_if_needed SRC_REF DST_REF
copy_if_needed() {
  local src="$1" dst="$2"
  local src_digest="$(img_digest "$src")"
  [[ -n "$src_digest" ]] || { echo "ERROR: source image missing: $src" >&2; return 1; }
  local dst_digest="$(img_digest "$dst")"

  if [[ -n "$dst_digest" && "$dst_digest" == "$src_digest" ]]; then
    echo "No-op: $dst already points to $src_digest"
    return 0
  fi

  echo "Copy: $src -> $dst ($dst_digest -> $src_digest)"
  skopeo copy "$(_ref_url "$src")" "$(_ref_url "$dst")"
}

# Check whether a promotion would succeed, without copying.
# usage: promote_check_strict CANON_REF STABLE_REF
# - returns 0 if safe/no-op
# - returns 1 if it would fail (e.g. stable exists and differs)
# - returns 2 if canonical missing
promote_check_strict() {
  local canon="$1" stable="$2"
  local canon_digest="$(img_digest "$canon")" || true
  local stable_digest="$(img_digest "$stable")" || true

  if [[ -z "$canon_digest" ]]; then
    echo "PROMOTE-CHECK: canonical missing: $canon" >&2
    return 2
  fi

  if [[ -z "$stable_digest" ]]; then
    echo "PROMOTE-CHECK: stable missing -> would promote: $stable"
    return 0
  fi

  if [[ "$stable_digest" == "$canon_digest" ]]; then
    echo "PROMOTE-CHECK: no-op (stable already matches canonical): $stable"
    return 0
  fi

  echo "PROMOTE-CHECK: would FAIL (stable exists but differs): $stable" >&2
  return 1
}

# Real promotion:
# usage: promote_strict CANON_REF STABLE_REF
# - no-op if stable already matches canonical
# - fails if stable exists but differs
# - fails if canonical missing
promote_strict() {
  local canon="$1" stable="$2"

  promote_check_strict "$canon" "$stable"
  local rc=$?
  case "$rc" in
    0)
      # either no-op or safe to promote (stable missing)
      if ! img_exists "$stable"; then
        echo "PROMOTE: $canon -> $stable"
        skopeo copy "$(_ref_url "$canon")" "$(_ref_url "$stable")"
      fi
      return 0
      ;;
    1)
      echo "ERROR: refusing to overwrite existing stable tag with different digest: $stable" >&2
      return 1
      ;;
    2)
      echo "ERROR: canonical image missing, cannot promote: $canon" >&2
      return 1
      ;;
  esac
}
