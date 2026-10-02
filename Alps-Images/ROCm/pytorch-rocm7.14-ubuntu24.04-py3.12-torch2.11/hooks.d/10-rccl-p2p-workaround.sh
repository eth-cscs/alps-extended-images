#!/usr/bin/env bash
set -euo pipefail

runtime_env="/opt/alps/env/alps-runtime.env"
[[ -f "${runtime_env}" ]] || { echo "ERROR: missing runtime env: ${runtime_env}" >&2; exit 1; }

cat >> "${runtime_env}" <<'EOF'

# ROCm 7.14 bundled RCCL is unreliable for 2-node MI300A collectives with P2P
# enabled on Beverin's current kernel. The flaw is enroot-specific (the identical
# workload passes with P2P under the Sarus suite/podman), so detect the runtime at
# container start: podman-based runtimes create /run/.containerenv, enroot does not.
# Users can override by setting NCCL_P2P_DISABLE before the runtime env loads.
if [ ! -f /run/.containerenv ]; then
  defvar NCCL_P2P_DISABLE "1"
fi
EOF
