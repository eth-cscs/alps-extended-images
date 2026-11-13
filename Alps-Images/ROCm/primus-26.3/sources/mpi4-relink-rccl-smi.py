#!/usr/bin/env python3
"""Restore Torch2.10's vendor-RCCL transitive SMI dependency without editing Torch."""
import os
from pathlib import Path
import shlex
import subprocess


def add_smi_dependency(command, prefix):
    """Add the required SMI dependency to the pinned RCCL final link command."""
    # This final link command is generated from the pinned RCCL CMake recipe.
    # Do not rerun CMake configuration: hipify regenerates sources, invalidating
    # device objects even when only the shared-library link flags change.
    if command.count(' -shared ') != 1 or ' -o librccl.so.1.0 ' not in command:
        raise RuntimeError('Unexpected pinned RCCL shared-library link command')
    flags=' -shared -Wl,--no-as-needed -L'+shlex.quote(str(prefix/'lib'))+' -lrocm_smi64 -Wl,--as-needed '
    return command.replace(' -shared ',flags,1)


def main():
    if os.environ.get('ROCM_VARIANT_DIR') != 'primus-26.3':
        raise RuntimeError('RCCL SMI compatibility is restricted to primus-26.3')
    build=Path('/tmp/rccl-build')
    prefix=Path(os.environ['ROCM_BUILD_PREFIX'])
    assert (prefix/'lib/librocm_smi64.so.1').is_file()
    commands=subprocess.check_output(['ninja','-t','commands','librccl.so.1.0'],cwd=build,text=True).splitlines()
    command=add_smi_dependency(commands[-1],prefix)
    print('Relinking Alps RCCL with the retained ROCm SMI dependency',flush=True)
    subprocess.run(['/bin/bash','-c',command],cwd=build,check=True)
    subprocess.run(['cmake','--install',str(build)],check=True)
    dynamic=subprocess.check_output(['readelf','-d','/opt/alps/rocm/rccl/lib/librccl.so.1'],text=True)
    assert '(NEEDED)' in dynamic and '[librocm_smi64.so.1]' in dynamic,dynamic
    subprocess.run(['python3','-c','import torch; print("Unchanged vendor Torch imports with explicit RCCL SMI dependency",torch.__version__)'],check=True)


if __name__ == '__main__':
    main()
