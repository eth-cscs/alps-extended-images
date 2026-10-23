#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

pip_install() {
    local python_bin="${1:?python executable required}"
    local index_url="${PIP_INDEX_URL:-${CSCS_PYPI_INDEX_URL:-https://jfrog.svc.cscs.ch/artifactory/api/pypi/pypi-remote/simple}}"
    shift

    PIP_INDEX_URL="${index_url}" "${python_bin}" -m pip install "$@"
}

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

# Purge only the requested build packages, including ones inherited from a base.
# Refuse reverse-dependency removals and never run autoremove in this step.
purge_apt_build_deps_exact() {
    local -a packages=("$@")
    [[ "${#packages[@]}" -gt 0 ]] || return 0

    local plan action package rest requested allowed
    plan="$(LC_ALL=C apt_get --simulate --no-auto-remove purge "${packages[@]}")" || return 1
    printf '%s\n' "${plan}"
    while read -r action package rest; do
        case "${action}" in
            Remv|Purg)
                allowed=0
                for requested in "${packages[@]}"; do
                    if [[ "${package}" == "${requested}" ]] \
                        || [[ "${requested}" != *:* && "${package%%:*}" == "${requested}" ]]; then
                        allowed=1
                        break
                    fi
                done
                if [[ "${allowed}" == 0 ]]; then
                    printf 'ERROR: refusing build-package purge that also removes %s\n' "${package}" >&2
                    return 1
                fi
                ;;
            Inst|Conf)
                printf 'ERROR: refusing build-package purge that also installs/configures %s\n' "${package}" >&2
                return 1
                ;;
        esac
    done <<< "${plan}"
    apt_get --no-auto-remove purge -y "${packages[@]}"
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

    # Build lists use unqualified names; dpkg may append a Multi-Arch suffix.
    local installed_package
    local -A installed_packages=()
    while IFS= read -r installed_package || [[ -n "$installed_package" ]]; do
        [[ -n "$installed_package" ]] || continue
        installed_packages["$installed_package"]=1
        installed_packages["${installed_package%%:*}"]=1
    done < "${snapshot_file}"

    for pkg in "${packages[@]}"; do
        if [[ -z "${installed_packages[$pkg]+set}" ]]; then
            cleanup_packages+=("$pkg")
        fi
    done

    cleanup_apt_build_deps "${cleanup_packages[@]}"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    cleanup_apt_build_deps "$@"
fi
