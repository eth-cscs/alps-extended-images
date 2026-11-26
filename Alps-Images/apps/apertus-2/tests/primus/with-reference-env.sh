#!/usr/bin/env bash
# Explicit reference-path selection for image-local acceptance checks.
set -euo pipefail
# shellcheck source=Alps-Images/common/source-alps-env.sh
source /opt/alps/env/source-alps-env.sh
# The startup marker is exported, but defvar is shell-local. A child bash can
# inherit the marker and skip the hook, so reload the unset-only defaults here.
if ! declare -F defvar >/dev/null; then
    # shellcheck disable=SC1091
    source /opt/alps/env/alps-runtime.env
fi
# shellcheck source=Alps-Images/apps/apertus-2/sources/primus/reference.env
source /opt/alps/apertus2-primus/reference.env
exec "$@"
