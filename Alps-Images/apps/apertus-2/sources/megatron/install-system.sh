#!/usr/bin/env bash
set -euo pipefail

source /opt/alps/package-helpers.sh
source /opt/alps/megatron/versions.env

test "$(uname -m)" = aarch64

python3 - <<'PY'
import sys

assert sys.version_info[:2] == (3, 12), sys.version
PY

apt_get update

apt_get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    gettext-base \
    libelf-dev \
    libgflags-dev \
    libgfortran5 \
    libgomp1 \
    libgtest-dev \
    libhwloc-dev \
    libibverbs-dev \
    libnl-3-dev \
    libnl-route-3-dev \
    libnuma-dev \
    ninja-build \
    pkg-config \
    python3-dev \
    rdma-core \
    zlib1g-dev

rm -rf /var/lib/apt/lists/*

libdir="/usr/lib/$(dpkg-architecture -qDEB_HOST_MULTIARCH)"

if [[ ! -e "${libdir}/libibverbs.so" && -e "${libdir}/libibverbs.so.1" ]]; then
    ln -sfn libibverbs.so.1 "${libdir}/libibverbs.so"
fi

test -e "${libdir}/libibverbs.so"

python3 - <<'PY'
from importlib import metadata
from pathlib import Path

pins = []

for name in ("torch", "torchvision", "triton", "pytorch-triton", "numpy"):
    try:
        pins.append(f"{name}=={metadata.version(name)}")
    except metadata.PackageNotFoundError:
        if name in {"torch", "numpy"}:
            raise

Path("/opt/alps/megatron/base-pins.txt").write_text("\n".join(pins) + "\n")
PY

curl \
    --fail \
    --location \
    --retry 3 \
    --retry-all-errors \
    --connect-timeout 15 \
    --max-time 180 \
    "https://github.com/mikefarah/yq/releases/download/v${YQ_VERSION}/yq_linux_arm64" \
    --output /tmp/yq_linux_arm64

printf '%s  %s\n' "${YQ_SHA256}" /tmp/yq_linux_arm64 \
    | sha256sum --check --strict

install -m 0755 /tmp/yq_linux_arm64 /usr/local/bin/yq

rm /tmp/yq_linux_arm64

ln -sfn /usr/bin/envsubst /usr/local/bin/envsubst

yq --version

envsubst --version
