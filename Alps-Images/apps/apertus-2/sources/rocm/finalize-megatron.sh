#!/usr/bin/env bash
set -euo pipefail

root=/opt/alps/apertus2-rocm
megatron=/opt/megatron-lm
patches="${root}/patches/megatron"

# Keep this late component separate from install.sh: changing these fixes must
# not invalidate the earlier UCCL, Transformer Engine or FLA build layers.
for patch_file in "${patches}"/*.patch; do
    git -C "$megatron" apply --check "$patch_file"
    git -C "$megatron" apply "$patch_file"
done
(
    cd "$patches"
    sha256sum -- *.patch
) > "${root}/megatron-patches.sha256"

# The dataset Makefile builds helpers.cpp, not this separate import-time library.
# Match the pinned CPU loader's compiler/flags and leave a fresh, loadable DSO.
g++ -O3 -shared -fPIC \
    -o "${megatron}/megatron/core/datasets/libbfd_pack.so" \
    "${megatron}/megatron/core/datasets/bfd_pack.cpp"
