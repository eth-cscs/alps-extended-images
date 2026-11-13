#!/usr/bin/env bash
set -euo pipefail
root=/opt/alps/apertus2-primus
# shellcheck source=Alps-Images/common/package-helpers.sh
source /opt/alps/package-helpers.sh
# shellcheck source=Alps-Images/apps/apertus-2/sources/primus/versions.env
source "$root/versions.env"
# shellcheck source=/dev/null
source /opt/alps/env/alps-rocm-build.env
export PIP_CONSTRAINT="$root/base-pins.txt"
export MAX_JOBS="${APERTUS_BUILD_JOBS:-4}" CMAKE_BUILD_PARALLEL_LEVEL="${APERTUS_BUILD_JOBS:-4}"
export ROCM_HOME="$ROCM_BUILD_PREFIX"
export PATH="$ROCM_SDK_BIN:$ROCM_BUILD_PREFIX/llvm/bin:$PATH"
case "${1:?installation group}" in
    dependencies)
        python3 - <<'PYTHON'
import json
from pathlib import Path
# Reuse the base's verified inventory rather than maintaining a second package list.
packages = json.loads(Path('/opt/alps/primus-compute-before.json').read_text())['packages']
pins = [f'{name}=={version}' for name, version in sorted(packages.items())]
Path('/opt/alps/apertus2-primus/base-pins.txt').write_text('\n'.join(pins) + '\n')
PYTHON
        pip_install python3 --no-cache-dir --no-deps --force-reinstall --require-hashes -r "$root/requirements.lock"
        python3 - <<'PYTHON'
import hashlib
import importlib.util
from pathlib import Path
root=Path('/opt/alps/apertus2-primus')
gate=root/'kda-gate.py'
assert hashlib.sha256(gate.read_bytes()).hexdigest()=='b350410cc605d1b3fac082d1e8384d7e9897e3aaa36c00f3d01d8d7fd48320ae'
package=Path(next(iter(importlib.util.find_spec('fla').submodule_search_locations)))
(package/'ops/kda/gate.py').write_bytes(gate.read_bytes())
PYTHON
        git init -q /tmp/apertus2-reference-optimizer
        git -C /tmp/apertus2-reference-optimizer remote add origin "$EMERGING_OPTIMIZERS_REPO"
        git -C /tmp/apertus2-reference-optimizer fetch --depth 1 origin "$EMERGING_OPTIMIZERS_REF"
        git -C /tmp/apertus2-reference-optimizer checkout -q --detach FETCH_HEAD
        test "$(git -C /tmp/apertus2-reference-optimizer rev-parse HEAD)" = "$EMERGING_OPTIMIZERS_REF"
        pip_install python3 --no-cache-dir --no-deps --no-build-isolation --force-reinstall /tmp/apertus2-reference-optimizer
        printf '%s %s\n' "$EMERGING_OPTIMIZERS_REPO" "$EMERGING_OPTIMIZERS_REF" > "$root/optimizer-source.txt"
        rm -rf /tmp/apertus2-reference-optimizer
        ;;
    megatron)
        python3 "$root/install-source.py" "$root/vendor" /opt/megatron-lm
        python3 "$root/prepare-source.py"
        pip_install python3 --no-cache-dir --no-deps --no-build-isolation -e /opt/megatron-lm
        make -C /opt/megatron-lm/megatron/core/datasets -j"$MAX_JOBS"
        g++ -O3 -shared -fPIC -o /opt/megatron-lm/megatron/core/datasets/libbfd_pack.so /opt/megatron-lm/megatron/core/datasets/bfd_pack.cpp
        # CPU-only extension, rebuilt independently for Torch2.10 and Torch2.14.
        MAX_JOBS=2 python3 "$root/build-share-storage.py"
        ;;
    reference-config)
        python3 - <<'PYTHON'
from pathlib import Path
root=Path('/opt/alps/apertus2-primus')
from importlib import metadata
if metadata.version('triton')=='3.6.0' and metadata.version('torch').startswith('2.10.0+git94c6e04'):
    with (root/'reference.env').open('a') as stream:
        stream.write('defvar FLA_CACHE_MODE "fuzzy"\n')
        stream.write('defvar FLA_CONFIG_DIR "/opt/alps/apertus2-primus/configs/fla-triton36"\n')
        stream.write('defvar HIPBLASLT_TUNING_OVERRIDE_FILE "/opt/alps/apertus2-primus/configs/hipblaslt-mid4b.txt"\n')
PYTHON
        python3 -m pip freeze > "$root/installed-packages.txt"
        ;;
    *) echo "Unknown Primus app installation group: $1" >&2; exit 2 ;;
esac
