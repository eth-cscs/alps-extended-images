#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://jfrog.svc.cscs.ch/artifactory/api/pypi/pypi-remote/simple}"

apt_cmd() {
    apt "$@" && return 0
    local status=$?
    printf 'apt failed with status %s; retrying with APT::Sandbox::User=root\n' "$status" >&2
    apt -o APT::Sandbox::User=root "$@"
}

apt_get() {
    apt-get "$@" && return 0
    local status=$?
    printf 'apt-get failed with status %s; retrying with APT::Sandbox::User=root\n' "$status" >&2
    apt-get -o APT::Sandbox::User=root "$@"
}

apt_mark() {
    apt-mark "$@"
}

snapshot_apt_packages() {
    local snapshot_file="${1:?snapshot file required}"

    dpkg-query -W -f='${binary:Package}\n' > "${snapshot_file}"
}

cleanup_apt_build_deps() {
    local packages=("$@")
    local hold_packages=()
    local held_packages=()
    local pkg
    local status=0

    read -r -a hold_packages <<< "${APT_CLEANUP_HOLD_PACKAGES:-libibverbs-dev}"

    if [[ "${#packages[@]}" -eq 0 ]]; then
        return 0
    fi

    printf 'Packages cleanup...\n'
    for pkg in "${hold_packages[@]}"; do
        if dpkg-query -W -f='${db:Status-Abbrev}\n' "$pkg" 2>/dev/null | grep -q '^.i '; then
            printf 'Marking package to hold: %s\n' "$pkg"
            apt_mark hold "$pkg"
            held_packages+=("$pkg")
        fi
    done

    printf 'Removing build packages: %s\n' "${packages[*]}"
    apt_get remove --purge -y "${packages[@]}" || status=$?
    if [[ "$status" -eq 0 ]]; then
        printf 'Running autoremove...\n'
        apt_get autoremove -y || status=$?
    fi
    rm -rf /var/lib/apt/lists/*

    if [[ "${#held_packages[@]}" -gt 0 ]]; then
        printf 'Unholding packages: %s\n' "${held_packages[*]}"
        apt_mark unhold "${held_packages[@]}" || status=$?
    fi

    return "$status"
}

cleanup_new_apt_build_deps() {
    local snapshot_file="${1:?snapshot file required}"
    shift

    local packages=("$@")
    local cleanup_packages=()
    local pkg

    [[ -f "${snapshot_file}" ]] || {
        printf 'ERROR: missing apt package snapshot: %s\n' "${snapshot_file}" >&2
        return 1
    }

    for pkg in "${packages[@]}"; do
        if ! grep -Fxq "$pkg" "${snapshot_file}"; then
            cleanup_packages+=("$pkg")
        fi
    done

    cleanup_apt_build_deps "${cleanup_packages[@]}"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    cleanup_apt_build_deps "$@"
fi
