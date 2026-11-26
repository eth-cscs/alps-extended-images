#!/usr/bin/env bash
set -euo pipefail
[[ "${ROCM_VARIANT_DIR:-}" == primus-26.3 ]] || {
    echo 'ERROR: MPI4 installation is restricted to primus-26.3' >&2
    exit 1
}
# Shared build primitives/defaults are read, never modified.
# shellcheck source=Alps-Images/common/install-alps-hpc-stack.sh
source /opt/alps/install-alps-hpc-stack.sh
# shellcheck source=Alps-Images/ROCm/primus-26.3/sources/mpi4-versions.env
source /opt/alps/primus/mpi4-versions.env
# shellcheck source=Alps-Images/ROCm/install-alps-rocm-components.sh
source /opt/alps/install-alps-rocm-components.sh
load_rocm_sdk_env
init_hpc_stack_prefixes
export LDFLAGS="${LDFLAGS:-} -L${ROCM_BUILD_PREFIX}/lib"
export PKG_CONFIG_PATH="$LIBFABRIC_PREFIX/lib/pkgconfig:$UCX_PREFIX/lib/pkgconfig:$UCC_PREFIX/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
archive=/tmp/primus263-openmpi.tar.gz
source_dir="/tmp/openmpi-${PRIMUS263_OMPI_VERSION}"
curl -fsSL "https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-${PRIMUS263_OMPI_VERSION}.tar.gz" -o "$archive"
printf '%s  %s\n' "$PRIMUS263_OMPI_SHA256" "$archive" | sha256sum -c -
tar --no-same-owner -C /tmp -xzf "$archive"
# New libfabric no longer exports the private container_of convenience macro.
# Keep equivalent pointer arithmetic local to MPI4's OFI request conversion.
patch --batch --fuzz=0 -p1 -d "$source_dir" -i /opt/alps/patches/openmpi/mpi4-ofi-container.patch
# Negotiate ROCm device-buffer capability with implicit-registration CXI.
# This is confined to MPI4; shared MPI5 and runtime transport defaults stay intact.
patch --batch --fuzz=0 -p1 -d "$source_dir" -i /opt/alps/patches/openmpi/mpi4-ofi-hmem.patch
(
    cd "$source_dir"
    ./configure --prefix="$OMPI_PREFIX" --libdir="$OMPI_PREFIX/lib" \
        --with-ofi="$LIBFABRIC_PREFIX" --with-ucx="$UCX_PREFIX" --with-ucc="$UCC_PREFIX" \
        --enable-mpi-cxx --enable-oshmem --without-cuda
    make -j"$(make_jobs)"
    make install
)
test -e "$OMPI_PREFIX/lib/libmpi_cxx.so.40"
# Remove the inherited MPI4 implementation, not any compute libraries. Only
# this image uses these paths; MPI5's shared installer and defaults are untouched.
for libdir in /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib; do
    [[ -d "$libdir" ]] || continue
    find "$libdir" -maxdepth 1 \( -type f -o -type l \) \( \
        -name 'libmpi.so*' -o -name 'libmpi_cxx.so*' -o -name 'libmpi_mpifh.so*' -o \
        -name 'libmpi_usempi_ignore_tkr.so*' -o -name 'libmpi_usempif08.so*' -o \
        -name 'libopen-pal.so*' -o -name 'libopen-rte.so*' -o -name 'liboshmem.so*' \) -print -delete
done
rm -rf /usr/local/mpi
ln -s "$OMPI_PREFIX" /usr/local/mpi
printf '%s\n' "$OMPI_PREFIX/lib" > /etc/ld.so.conf.d/99-alps-primus263-mpi4.conf
ldconfig
record_alps_version_var OMPI_VERSION "$PRIMUS263_OMPI_VERSION"
python3 - <<'PYTHON'
from pathlib import Path
import subprocess
import torch
library=Path(torch.__file__).parent/'lib/libtorch_global_deps.so'
text=subprocess.check_output(['ldd',str(library)],text=True)
assert 'not found' not in text,text
for name in ('libmpi_cxx.so.40','libmpi.so.40','libopen-pal.so.40','libopen-rte.so.40'):
    path=next(line.split()[2] for line in text.splitlines() if line.strip().startswith(name+' '))
    assert Path(path).resolve().is_relative_to('/opt/hpcx/ompi'),(name,path)
print('Unchanged vendor Torch loads with Alps-built MPI4; no vendor MPI core selected')
PYTHON
rm -rf "$source_dir"
rm -f "$archive"
