#!/usr/bin/env bash
set -euo pipefail
# Match the existing Alps ROCm default, preserving explicit user overrides.
cat >> /opt/alps/env/alps-runtime.env <<'EOF'

# Alps MI300A baseline; exporting any value (including empty) overrides it.
defvar NCCL_P2P_DISABLE "1"
EOF
