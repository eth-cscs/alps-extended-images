#!/usr/bin/env bash
set -euo pipefail
bash /opt/alps/install-alps-rocm-stack.sh purge
rm -rf /opt/openmpi /ompi-4.1.6 /opt/ucx /workspace/ucx-1.18.0/install
