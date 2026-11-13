#!/usr/bin/env bash
set -euo pipefail
[[ "${ROCM_VARIANT_DIR:-}" == primus-26.3 ]] || {
    echo 'ERROR: MPI4 installation is restricted to primus-26.3' >&2
    exit 1
}
# RCCL must supply the SMI dependency before any subsequent Torch import.
# shellcheck source=/dev/null
source /opt/alps/env/alps-rocm-build.env
python3 /opt/alps/primus/mpi4-relink-rccl-smi.py
bash /opt/alps/primus/mpi4-install.sh
