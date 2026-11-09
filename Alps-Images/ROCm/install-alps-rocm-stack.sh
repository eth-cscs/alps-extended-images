#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/install-alps-hpc-stack.sh" ]]; then
    source "${SCRIPT_DIR}/install-alps-hpc-stack.sh"
    source "${SCRIPT_DIR}/install-alps-rocm-components.sh"
else
    source "${SCRIPT_DIR}/../common/install-alps-hpc-stack.sh"
    source "${SCRIPT_DIR}/install-alps-rocm-components.sh"
fi

ROCM_STACK_STEPS=(
    bootstrap
    purge
    boost
    xpmem
    cxi-bits
    libfabric
    rccl
    ucx
    ucc
    ompi
    aws-ofi
    tests
    cleanup
)

step_needs_rocm_env() {
    case "${1:?step required}" in
        cxi-bits|libfabric|rccl|ucx|ucc|ompi|aws-ofi|tests)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

run_rocm_stack_step() {
    local step="${1:?step required}"

    case "${step}" in
        bootstrap)
            apt_install_build_deps
            bootstrap_rocm_sdk
            ;;
        build-deps)
            apt_install_build_deps
            ;;
        purge)
            purge_preinstalled_network_stack
            ;;
        boost)
            build_boost
            ;;
        xpmem)
            build_xpmem
            ;;
        cxi-bits)
            build_cxi_bits
            ;;
        libfabric)
            build_libfabric
            ;;
        rccl)
            configure_rccl
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
            build_aws_ofi_rccl
            ;;
        tests)
            build_rccl_tests
            build_osu
            ;;
        cleanup)
            clean_up
            ;;
    esac
}

run_step() {
    local step="${1:?step required}"

    init_hpc_stack_prefixes

    if [[ "${step}" == "all" ]]; then
        run_stack_step_sequence run_rocm_stack_step "${ROCM_STACK_STEPS[@]}"
        return 0
    fi

    ensure_known_stack_step ROCm "${step}" build-deps "${ROCM_STACK_STEPS[@]}"

    if step_needs_rocm_env "${step}"; then
        load_rocm_sdk_env
    fi

    run_rocm_stack_step "${step}"
}

main() {
    run_step "${1:-all}"
}

main "$@"
