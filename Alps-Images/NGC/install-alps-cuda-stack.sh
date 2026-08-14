#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/install-alps-hpc-stack.sh" ]]; then
    source "${SCRIPT_DIR}/install-alps-hpc-stack.sh"
    source "${SCRIPT_DIR}/install-alps-cuda-components.sh"
else
    source "${SCRIPT_DIR}/../common/install-alps-hpc-stack.sh"
    source "${SCRIPT_DIR}/install-alps-cuda-components.sh"
fi

CUDA_STACK_STEPS=(
    bootstrap
    build-deps
    purge
    boost
    xpmem
    gdrcopy
    cxi-bits
    libfabric
    nccl
    ucx
    ucc
    ompi
    aws-ofi
    nvshmem
    tests
    cleanup
)

init_cuda_stack() {
    CUDA_DIR="$(detect_cuda_dir)" || die "Could not determine CUDA directory..."
    export CUDA_DIR
    export CUDA_HOME="${CUDA_HOME:-$CUDA_DIR}"
    export CUDA_PATH="${CUDA_PATH:-$CUDA_DIR}"
    init_hpc_stack_prefixes
}

run_cuda_stack_step() {
    local step="${1:?step required}"

    case "${step}" in
        bootstrap)
            ;;
        build-deps)
            apt_install_build_deps
            ;;
        purge)
            purge_preinstalled_network_stack
            remove_hpcx_plugins
            ;;
        boost)
            build_boost
            ;;
        xpmem)
            build_xpmem
            ;;
        gdrcopy)
            build_gdrcopy
            ;;
        cxi-bits)
            build_cxi_bits
            ;;
        libfabric)
            build_libfabric
            ;;
        nccl)
            build_nccl_deb
            ;;
        ucx)
            build_ucx
            ;;
        ucc)
            build_ucc
            ;;
        ompi)
            build_ompi5
            ;;
        aws-ofi)
            build_aws_ofi_nccl
            ;;
        nvshmem)
            build_nvshmem
            ;;
        tests)
            build_nccl_tests
            build_osu
            ;;
        cleanup)
            clean_up
            ;;
    esac
}

run_step() {
    local step="${1:?step required}"

    init_cuda_stack

    if [[ "${step}" == "all" ]]; then
        run_stack_step_sequence run_cuda_stack_step "${CUDA_STACK_STEPS[@]}"
        return 0
    fi

    ensure_known_stack_step CUDA "${step}" "${CUDA_STACK_STEPS[@]}"
    run_cuda_stack_step "${step}"
}

main() {
    run_step "${1:-all}"
}

main "$@"
