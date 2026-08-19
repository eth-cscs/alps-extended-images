#!/usr/bin/env bash
set -euo pipefail

runtime_env="/opt/alps/env/alps-runtime.env"
[[ -f "${runtime_env}" ]] || { echo "ERROR: missing runtime env: ${runtime_env}" >&2; exit 1; }

cat >> "${runtime_env}" <<'EOF'

# ROCm 7.14 bundled RCCL is unreliable for 2-node MI300A collectives with P2P
# enabled on Beverin's current kernel. Keep this scoped to this image variant;
# users can override it by setting NCCL_P2P_DISABLE before the runtime env loads.
defvar NCCL_P2P_DISABLE "1"
EOF
