#!/usr/bin/env bash
# Preserve vendor SDK/MPI4 dependencies until the Alps MPI4 installation.
set -euo pipefail
[[ "${ROCM_VARIANT_DIR:-}" == primus-26.3 ]] || {
    echo 'ERROR: MPI4/bootstrap preservation is restricted to primus-26.3' >&2
    exit 1
}
# shellcheck source=Alps-Images/common/install-alps-hpc-stack.sh
source /opt/alps/install-alps-hpc-stack.sh
# shellcheck source=/dev/null
source /opt/alps/env/alps-rocm-build.env
mapfile -t inherited_packages < <(dpkg-query -W -f='${binary:Package} ${db:Status-Abbrev}\n' | awk '$2 == "ii" {print $1}')
[[ "${#inherited_packages[@]}" -gt 0 ]] || die 'No installed vendor packages found'
apt_mark manual "${inherited_packages[@]}"
# APT's dependency graph ties the system ROCm SDK and Torch's MPI4 loader
# dependencies to the vendor network packages. Keep package ownership metadata
# and MPI4 core libraries during this transition; shared cleanup still removes
# the vendor fabric/UCX/UCC files and bundled MPI prefixes. The newly built Alps
# MPI4 replaces the remaining system MPI libraries later in this image only.
apt_get() {
    case "${1:-}" in
        purge|autoremove)
            echo "Preserving vendor SDK/MPI4 APT dependencies; skipping apt-get $1"
            ;;
        *) die "Unexpected APT operation in preserve-only cleanup: ${1:-}" ;;
    esac
}
purge_preinstalled_network_stack
rm -rf /opt/openmpi /ompi-4.1.6 /opt/ucx /workspace/ucx-1.18.0/install
test -f "$ROCM_BUILD_PREFIX/include/hip/hip_runtime_api.h"
python3 /opt/alps/check-primus-compute.py --compare /opt/alps/primus-compute-before.json
