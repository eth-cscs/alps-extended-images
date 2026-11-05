#!/usr/bin/env bash
set -euo pipefail

source /opt/alps/megatron/versions.env

export PIP_INDEX_URL="$UV_DEFAULT_INDEX"
export PIP_CONSTRAINT=/opt/alps/megatron/base-pins.txt
export UV_PYTHON_DOWNLOADS=never

: "${UCCL_MAX_JOBS:=8}"
: "${TE_MAX_JOBS:=8}"
: "${KERNEL_MAX_JOBS:=8}"
: "${FA3_MAX_JOBS:=8}"
: "${FA3_PTXAS_OPT:=1}"
: "${FLASHINFER_CACHE_MAX_JOBS:=16}"

uv_install() {
    uv pip install \
        --system \
        --break-system-packages \
        --python python3 \
        --no-cache \
        --constraint "$PIP_CONSTRAINT" \
        "$@"
}

record_source() {
    printf '%s %s %s\n' "$1" "$2" "$3" \
        >> /opt/alps/megatron/source-pins.txt
}

checkout_source() {
    local repo=$1
    local ref=$2
    local dest=$3

    git init -q "$dest"
    git -C "$dest" remote add origin "$repo"
    git -C "$dest" fetch --depth 1 origin "$ref"
    git -C "$dest" checkout -q FETCH_HEAD

    test "$(git -C "$dest" rev-parse HEAD)" = "$ref"

    git -C "$dest" submodule update --init --recursive --depth 1
}

install_source() {
    local name=$1
    local repo=$2
    local ref=$3
    local dest="/tmp/megatron-$1"

    checkout_source "$repo" "$ref" "$dest"

    uv_install \
        --no-build-isolation \
        --no-deps \
        "$dest"

    record_source "$name" "$repo" "$ref"

    rm -rf "$dest"
}

case "${1:?kernel group required}" in
    uccl)
        hard_limit="$(ulimit -Hn)"
        [[ "$hard_limit" != unlimited ]] || hard_limit=65536
        ulimit -n "$hard_limit"

        uv pip uninstall \
            --system \
            --break-system-packages \
            --python python3 \
            uccl \
            deep_ep

        git clone --recursive "$UCCL_REPO" /io
        git -C /io checkout -q "$UCCL_REF"

        test "$(git -C /io rev-parse HEAD)" = "$UCCL_REF"

        git -C /io submodule update --init --recursive

        mkdir -p /tmp/gpustub /opt/wheels/uccl

        # UCCL probes the GPU model on the GPU-less build runner.
        cat > /tmp/gpustub/nvidia-smi <<'PY'
#!/usr/bin/python3
import sys

query = " ".join(sys.argv[1:])

if "query-gpu=name" in query:
    print("NVIDIA GH200 120GB")
elif "query-gpu=compute_cap" in query:
    print("9.0")
else:
    sys.exit(1)
PY

        chmod 0755 /tmp/gpustub/nvidia-smi

        fabric_prefix="$(pkg-config --variable=prefix libfabric)"
        fabric_libdir="$(pkg-config --variable=libdir libfabric)"

        test -n "$fabric_prefix"
        test -f "$fabric_libdir/libfabric.so"

        printf '%s\n' "$fabric_libdir" \
            > /etc/ld.so.conf.d/98-alps-libfabric.conf

        ldconfig

        cd /io

        sed -i \
            's|-Wl,-rpath=/usr/lib/x86_64-linux-gnu||g' \
            p2p/Makefile

        host_glibc_ver="$(python3 -c 'import platform; print(platform.libc_ver()[1])')"

        export PATH="/tmp/gpustub:${PATH}" \
            CUDA_HOME=/usr/local/cuda \
            LIBFABRIC_HOME="$fabric_prefix" \
            LIBRARY_PATH="${fabric_libdir}:/usr/local/cuda/lib64/stubs:${LIBRARY_PATH:-}" \
            LDFLAGS="-L${fabric_libdir} -Wl,-rpath,${fabric_libdir} ${LDFLAGS:-}" \
            TORCH_CUDA_ARCH_LIST=9.0 \
            NUM_MAX_NVL_PEERS=4 \
            MAX_JOBS="$UCCL_MAX_JOBS" \
            OMP_THREAD_LIMIT="$UCCL_MAX_JOBS" \
            USE_LIBFABRIC_CXI=1 \
            PER_EXPERT_BATCHING=1 \
            TARGET=cu13 \
            PY_VER=3.12 \
            ARCH=aarch64 \
            BUILD_TYPE=all \
            IS_EFA="" \
            WHEEL_DIR=wheelhouse-cu13 \
            HOST_GLIBC_VER="$host_glibc_ver" \
            UCCL_RETAG_TO_HOST_GLIBC=0 \
            UCCL_LOCAL_VERSION=cu13

        # OMP_THREAD_LIMIT also limits nproc in the upstream make commands.
        bash ./build_inner.sh

        uv_install \
            --no-deps \
            wheelhouse-cu13/uccl-*.whl

        make -C p2p \
            -j"$UCCL_MAX_JOBS" \
            install \
            PYTHON=python3 \
            LIBFABRIC_HOME="$fabric_prefix"

        site_dir="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"

        printf '%s\n%s\n' \
            /usr/local/lib \
            "$site_dir/uccl/lib" \
            > /etc/ld.so.conf.d/98-uccl.conf

        ldconfig

        cd ep/deep_ep_wrapper

        uv build \
            --wheel \
            --no-build-isolation \
            --no-cache \
            --python python3 \
            --out-dir /opt/wheels/uccl \
            .

        uv_install \
            --no-deps \
            /opt/wheels/uccl/deep_ep-*.whl

        mkdir -p /opt/uccl/ep/bench
        cp -aL /io/ep/bench/. /opt/uccl/ep/bench/

        record_source uccl "$UCCL_REPO" "$UCCL_REF"

        cd /
        rm -rf /tmp/gpustub /io /opt/wheels/uccl
        ;;

    transformer-engine)
        uv pip uninstall \
            --system \
            --break-system-packages \
            --python python3 \
            transformer_engine \
            transformer_engine_cu13 \
            transformer_engine_torch

        MAX_JOBS="$TE_MAX_JOBS" \
            NVTE_BUILD_MAX_JOBS="$TE_MAX_JOBS" \
            CMAKE_BUILD_PARALLEL_LEVEL="$TE_MAX_JOBS" \
            NVTE_CUDA_ARCHS=90 \
            NVTE_FRAMEWORK=pytorch \
            NVTE_WITH_NCCL_EP=0 \
            install_source \
                transformer-engine \
                https://github.com/NVIDIA/TransformerEngine.git \
                "$TE_REF"
        ;;

    training)
        export MAX_JOBS="$KERNEL_MAX_JOBS" \
            TORCH_CUDA_ARCH_LIST=9.0 \
            CUDA_HOME=/usr/local/cuda

        uv pip uninstall \
            --system \
            --break-system-packages \
            --python python3 \
            deep_gemm \
            grouped_gemm \
            nv-grouped-gemm \
            emerging-optimizers \
            fla-core \
            flash-linear-attention \
            nvidia-resiliency-ext

        DG_FORCE_BUILD=1 \
            DG_USE_LOCAL_VERSION=1 \
            install_source \
                deep-gemm \
                "$DEEPGEMM_REPO" \
                "$DEEPGEMM_REF"

        GROUPED_GEMM_FORCE_BUILD=TRUE \
            install_source \
                nv-grouped-gemm \
                "$GROUPED_GEMM_REPO" \
                "$GROUPED_GEMM_REF"

        install_source \
            emerging-optimizers \
            "$EMERGING_OPTIMIZERS_REPO" \
            "$EMERGING_OPTIMIZERS_REF"

        install_source \
            flash-linear-attention \
            "$FLA_REPO" \
            "$FLA_REF"

        uv_install \
            --no-deps \
            "nvidia-resiliency-ext==$NVRESILIENCY_VERSION"
        ;;

    mamba)
        export MAX_JOBS="$KERNEL_MAX_JOBS" \
            TORCH_CUDA_ARCH_LIST=9.0 \
            CUDA_HOME=/usr/local/cuda

        uv pip uninstall \
            --system \
            --break-system-packages \
            --python python3 \
            causal-conv1d \
            mamba-ssm

        CAUSAL_CONV1D_FORCE_BUILD=TRUE \
            install_source \
                causal-conv1d \
                "$CAUSAL_CONV1D_REPO" \
                "$CAUSAL_CONV1D_REF"

        MAMBA_FORCE_BUILD=TRUE \
            uv_install \
                --no-build-isolation \
                --no-deps \
                --no-binary=mamba-ssm \
                "mamba-ssm==$MAMBA_VERSION"
        ;;

    hadamard)
        export MAX_JOBS="$KERNEL_MAX_JOBS" \
            TORCH_CUDA_ARCH_LIST=9.0 \
            CUDA_HOME=/usr/local/cuda

        HADAMARD_FORCE_BUILD=TRUE \
            install_source \
                fast-hadamard-transform \
                "$HADAMARD_REPO" \
                "$HADAMARD_REF"
        ;;

    flash-attention-3)
        checkout_source "$FA3_REPO" "$FA3_REF" /tmp/megatron-fa3

        # Rebuild FA3 against this image's Torch and CUDA.
        uv pip uninstall \
            --system \
            --break-system-packages \
            --python python3 \
            flash-attn-3

        cd /tmp/megatron-fa3/hopper

        test -f /usr/local/cuda/include/cccl/cuda/std/utility

        FLASH_ATTENTION_FORCE_BUILD=TRUE \
            FLASH_ATTENTION_DISABLE_SM80=TRUE \
            CPATH="/usr/local/cuda/include/cccl${CPATH:+:$CPATH}" \
            PYTORCH_NVCC=/usr/local/cuda/bin/nvcc \
            NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:+$NVCC_APPEND_FLAGS }-Xptxas=-O${FA3_PTXAS_OPT}" \
            MAX_JOBS="$FA3_MAX_JOBS" \
            NVCC_THREADS=2 \
            CUDA_HOME=/usr/local/cuda \
            uv_install \
                --verbose \
                --no-build-isolation \
                --no-deps \
                .

        site_dir="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"

        mkdir -p "$site_dir/flash_attn_3"
        touch "$site_dir/flash_attn_3/__init__.py"

        # TE imports these interfaces through the flash_attn_3 package.
        for module in flash_attn_interface flash_attn_config; do
            if [[ ! -f "$site_dir/flash_attn_3/$module.py" ]]; then
                printf 'import sys\nimport %s as _impl\nsys.modules[__name__] = _impl\n' \
                    "$module" \
                    > "$site_dir/flash_attn_3/$module.py"
            fi
        done

        record_source flash-attn-3 "$FA3_REPO" "$FA3_REF"

        cd /
        rm -rf /tmp/megatron-fa3
        ;;

    flashinfer)
        uv_install \
            --no-deps \
            "flashinfer-python==$FLASHINFER_VERSION"

        checkout_source \
            "$FLASHINFER_REPO" \
            "$FLASHINFER_REF" \
            /tmp/megatron-flashinfer

        cd /tmp/megatron-flashinfer/flashinfer-jit-cache

        mkdir -p /opt/wheels/flashinfer

        FLASHINFER_CUDA_ARCH_LIST=9.0a \
            CUDA_HOME=/usr/local/cuda \
            MAX_JOBS="$FLASHINFER_CACHE_MAX_JOBS" \
            FLASHINFER_NVCC_THREADS=1 \
            python3 -m pip wheel \
                --verbose \
                --no-cache-dir \
                --no-deps \
                --no-build-isolation \
                --wheel-dir /opt/wheels/flashinfer \
                .

        uv_install \
            --no-deps \
            /opt/wheels/flashinfer/flashinfer_jit_cache-*.whl

        record_source flashinfer "$FLASHINFER_REPO" "$FLASHINFER_REF"

        cd /
        rm -rf /tmp/megatron-flashinfer /opt/wheels/flashinfer
        ;;

    *)
        printf 'Unknown kernel group: %s\n' "$1" >&2
        exit 2
        ;;
esac
