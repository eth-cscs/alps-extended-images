#!/usr/bin/env bash
# Explicit reference-path selection for image-local acceptance checks.
set -euo pipefail
# shellcheck source=Alps-Images/common/source-alps-env.sh
source /opt/alps/env/source-alps-env.sh
# shellcheck source=Alps-Images/apps/apertus-2/sources/primus/reference.env
source /opt/alps/apertus2-primus/reference.env
exec "$@"
