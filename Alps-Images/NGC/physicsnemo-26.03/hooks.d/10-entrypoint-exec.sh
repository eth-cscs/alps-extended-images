#!/usr/bin/env bash
set -euo pipefail

target="/opt/nvidia/physicsnemo_env.sh"
needle='exec /opt/nvidia/nvidia_entrypoint.sh'

if [[ -f "$target" ]]; then
  # only patch if not already patched
  if ! grep -qF "$needle" "$target"; then
    sed -i -e 's|/opt/nvidia/nvidia_entrypoint.sh|exec /opt/nvidia/nvidia_entrypoint.sh|' "$target"
  fi
else
  echo "WARN: $target not found; skipping PhysicsNeMo entrypoint fix" >&2
fi
