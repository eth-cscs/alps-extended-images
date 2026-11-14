#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=Alps-Images/common/package-helpers.sh
source /opt/alps/package-helpers.sh
# shellcheck source=Alps-Images/apps/apertus-2/sources/rocm/versions.env
source /opt/alps/apertus2-rocm/versions.env

: "${UCCL_REPO:?UCCL_REPO must be set in versions.env}"
: "${UCCL_REF:?UCCL_REF must be set in versions.env}"
: "${ROCM_HOME:?ROCM_HOME must point to the Alps ROCm SDK}"
: "${PYTORCH_ROCM_ARCH:?PYTORCH_ROCM_ARCH must be set}"
: "${MAX_JOBS:?MAX_JOBS must be set}"
: "${PIP_CONSTRAINT:?PIP_CONSTRAINT must preserve the base torch pins}"

# Build only the native training transport. Do not invoke build_inner.sh: it
# also builds serving transports and can install a different torch/ROCm stack.
# The pinned 18b588a source keeps D2H handles in device memory so combine cannot
# fault when managed pages are evicted under training memory pressure.
git init -q /opt/uccl
git -C /opt/uccl remote add origin "$UCCL_REPO"
git -C /opt/uccl fetch --depth 1 origin "$UCCL_REF"
git -C /opt/uccl checkout -q --detach FETCH_HEAD
test "$(git -C /opt/uccl rev-parse HEAD)" = "$UCCL_REF"

for patch_file in /opt/alps/apertus2-rocm/patches/uccl/*.patch; do
    printf 'Applying UCCL patch: %s\n' "$patch_file"
    git -C /opt/uccl apply --check "$patch_file"
    git -C /opt/uccl apply "$patch_file"
done

# Resolve the canonical Alps libfabric instead of a bundled/system substitute.
# LIBRARY_PATH also covers distributions whose pkg-config libdir is lib64 or
# lib/<triplet>, whereas upstream ep/setup.py only adds $LIBFABRIC_HOME/lib.
fabric_prefix="$(pkg-config --variable=prefix libfabric)"
fabric_libdir="$(pkg-config --variable=libdir libfabric)"
test -n "$fabric_prefix"
test -f "$fabric_libdir/libfabric.so"
export LIBFABRIC_HOME="$fabric_prefix" \
    LIBRARY_PATH="${fabric_libdir}:${LIBRARY_PATH:-}" \
    USE_LIBFABRIC_CXI=1 \
    PER_EXPERT_BATCHING=1 \
    NUM_MAX_NVL_PEERS=4 \
    MAX_NUM_GPUS=4 \
    OMP_THREAD_LIMIT="$MAX_JOBS"

printf '%s\n' "$fabric_libdir" > /etc/ld.so.conf.d/98-apertus2-uccl.conf
ldconfig

# Explicit gfx942 is supplied by the image, so setup.py never needs a physical
# GPU or a spoofed GPU-discovery executable on the build runner. PyTorch's HIP
# build extension uses the existing ROCm torch protected by PIP_CONSTRAINT.
cd /opt/uccl/ep
python3 setup.py build --build-base /tmp/apertus2-uccl-ep
cp /tmp/apertus2-uccl-ep/lib*/ep*.so /opt/uccl/uccl/

# Package the native extension with upstream metadata, then install the real
# DeepEP wrapper (its buffer.py/utils.py symlink to the retained bench sources).
# No build isolation or dependency resolution may replace the base ROCm torch.
pip_install python3 --no-cache-dir --no-build-isolation --no-deps \
    /opt/uccl /opt/uccl/ep/deep_ep_wrapper

mkdir -p /opt/alps/apertus2-rocm
printf 'uccl %s %s\n' "$UCCL_REPO" "$UCCL_REF" \
    >> /opt/alps/apertus2-rocm/source-pins.txt

# Keep /opt/uccl/ep/bench and ep/tests for high-throughput dispatch/combine
# benchmarks. Runtime modules are site-packages/uccl/ep*.so and deep_ep/*.py;
# collectives continue using the canonical Alps aws-ofi-rccl plugin.
rm -rf /tmp/apertus2-uccl-ep
