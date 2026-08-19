#!/usr/bin/env bash
set -euo pipefail

variant_dir="${1:?variant directory required}"
variant_label="${2:?variant label required}"

if [[ -d "${variant_dir}/hooks.d" ]]; then
    for hook in "${variant_dir}"/hooks.d/*.sh; do
        [[ -e "${hook}" ]] || break
        echo "Running variant hook: ${hook}"
        chmod +x "${hook}"
        "${hook}"
    done
else
    echo "No variant hooks for ${variant_label}"
fi
