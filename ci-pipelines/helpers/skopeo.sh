#!/usr/bin/env bash
set -euo pipefail

# Registry helpers used by both the generator and child pipeline jobs.
#
# The dynamic pipeline treats the registry as the source of truth for whether an
# image is already built, tested, and published. Canonical image tags include a
# content hash. Validation markers are extra tags that point at the same digest
# after tests pass. Stable tags are promotion targets and are never trusted as
# build inputs.

_skopeo_login() {
  local reg="${1:?registry required}"
  local user="${2:?username required}"
  local password="${3:?password required}"

  echo "login to: ${reg}"
  skopeo login --username "${user}" --password "${password}" "${reg}" >/dev/null
}

skopeo_login() {
  : "${IMAGE_PREFIX:?IMAGE_PREFIX must be set}"
  : "${JFROG_USER:?JFROG_USER must be set}"
  : "${JFROG_KEY:?JFROG_KEY must be set}"
  _skopeo_login "${IMAGE_PREFIX%%/*}" "$JFROG_USER" "$JFROG_KEY"
}

skopeo_login_ghcr() {
  : "${GHCR_IMAGE_PREFIX:?GHCR_IMAGE_PREFIX must be set}"
  : "${GITHUB_ACTOR:?GITHUB_ACTOR must be set}"
  : "${GITHUB_TOKEN:?GITHUB_TOKEN must be set}"
  _skopeo_login "${GHCR_IMAGE_PREFIX%%/*}" "$GITHUB_ACTOR" "$GITHUB_TOKEN"
}

# Print an image digest. Return an empty string only when the registry reports a
# true missing-image condition. Auth, network, and other unexpected skopeo errors
# return non-zero so the generator stops instead of skipping required work.
# usage: img_digest REF
img_digest() {
  local ref="${1:?image ref required}"
  local output status

  set +e
  output="$(skopeo inspect --format '{{.Digest}}' "docker://$ref" 2>&1)"
  status=$?
  set -e
  if [[ "$status" -eq 0 ]]; then
    printf '%s\n' "$output"
    return 0
  fi

  # Skopeo/JFrog do not guarantee one exit status for missing manifests. Only
  # explicit registry missing-manifest/name messages count as cache misses; auth,
  # network, and tool errors return non-zero so the generator stops.
  if [[ "$output" == *"manifest unknown"* || "$output" == *"name unknown"* || "$output" == *"The named manifest is not known to the registry"* ]]; then
    return 0
  fi

  echo "ERROR: failed to inspect image: $ref" >&2
  echo "$output" >&2
  return "$status"
}

# usage: tested_ref_for CANON_REF [VALIDATION_HASH]
tested_ref_for() {
  local canon="${1:?canonical image ref required}"
  local validation_hash="${2:-}"
  local repo="${canon%:*}"
  local tag="${canon##*:}"

  [[ "$repo" != "$canon" ]] || { echo "ERROR: image ref must include a tag: $canon" >&2; return 1; }
  if [[ -n "$validation_hash" ]]; then
    printf '%s:%s-tested-%s\n' "$repo" "$tag" "$validation_hash"
  else
    printf '%s:%s-tested\n' "$repo" "$tag"
  fi
}

# Tested refs are marker tags, not separate image builds. They have the form:
#   <canonical-tag>-tested-<validation-hash>
# The digest must equal the canonical digest. The validation hash covers test
# definitions/templates/helper code, so changing tests invalidates old markers
# without rebuilding unchanged canonical image content.

# usage: require_tested_marker_valid CANON_REF TESTED_REF
require_tested_marker_valid() {
  local canon="${1:?canonical image ref required}"
  local tested="${2:?tested marker ref required}"
  local canon_digest tested_digest

  canon_digest="$(img_digest "$canon")" || return $?
  tested_digest="$(img_digest "$tested")" || return $?
  [[ -n "$canon_digest" ]] || { echo "ERROR: canonical image missing: $canon" >&2; return 1; }
  [[ -n "$tested_digest" ]] || { echo "ERROR: tested marker missing: $tested" >&2; return 1; }
  if [[ "$tested_digest" != "$canon_digest" ]]; then
    echo "ERROR: tested marker points to a different digest: $tested" >&2
    echo "  canonical: $canon_digest" >&2
    echo "  marker:    $tested_digest" >&2
    return 1
  fi
}

# usage: mark_tested CANON_REF TESTED_REF
mark_tested() {
  local canon="${1:?canonical image ref required}"
  local tested="${2:?tested marker ref required}"
  local canon_digest tested_digest

  canon_digest="$(img_digest "$canon")" || return $?
  tested_digest="$(img_digest "$tested")" || return $?
  [[ -n "$canon_digest" ]] || { echo "ERROR: canonical image missing: $canon" >&2; return 1; }
  if [[ -n "$tested_digest" && "$tested_digest" != "$canon_digest" ]]; then
    echo "ERROR: tested marker points to a different digest: $tested" >&2
    echo "  canonical: $canon_digest" >&2
    echo "  marker:    $tested_digest" >&2
    return 1
  fi
  if [[ "$tested_digest" == "$canon_digest" ]]; then
    echo "No-op: tested marker already matches canonical image: $tested"
    return 0
  fi

  echo "Mark tested: $canon -> $tested"
  skopeo copy "$(_ref_url "$canon")" "$(_ref_url "$tested")"
}

_ref_url() {
  local ref="$1"
  if [[ "$ref" == docker://* ]]; then
    printf '%s\n' "$ref"
  else
    printf 'docker://%s\n' "$ref"
  fi
}

# Check whether a strict stable promotion would succeed, without copying.
# usage: promote_check_strict CANON_REF STABLE_REF
# - returns 0 if safe/no-op
# - returns 1 if it would fail (e.g. stable exists and differs)
# - returns 2 if canonical missing
promote_check_strict() {
  local canon="$1" stable="$2"
  local canon_digest stable_digest

  canon_digest="$(img_digest "$canon")" || return $?
  stable_digest="$(img_digest "$stable")" || return $?

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

# Real strict promotion for non-dev stable refs:
# usage: promote_strict CANON_REF STABLE_REF
# - no-op if stable already matches canonical
# - fails if stable exists but differs
# - fails if canonical missing
promote_strict() {
  local canon="$1" stable="$2"
  local rc=0 stable_digest

  promote_check_strict "$canon" "$stable" || rc=$?
  case "$rc" in
    0)
      # either no-op or safe to promote (stable missing)
      stable_digest="$(img_digest "$stable")" || return $?
      if [[ -z "$stable_digest" ]]; then
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
    *)
      return "$rc"
      ;;
  esac
}
