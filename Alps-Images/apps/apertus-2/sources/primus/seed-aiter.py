#!/usr/bin/env python3
"""Bake the installed Primus AITER core into a new, read-only runtime seed (no GPU)."""

import hashlib
from importlib import metadata, util
import json
import os
from pathlib import Path
import shutil
import sys


def main():
    """Compile only the core using the installed JIT sources, as in the vLLM recipe."""
    assert metadata.version('amd-aiter') == '0.1.22.post1'
    assert metadata.version('torch') == '2.14.0+rocm10.1.0'
    package = Path(next(iter(util.find_spec('aiter').submodule_search_locations)))
    seed = Path(os.environ['AITER_JIT_DIR'])
    # Refuse to destroy inherited caches; this layer owns only a fresh seed.
    seed.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(package))
    from jit import core

    args = core.get_args_of_build('module_aiter_core')
    keys = ('srcs', 'flags_extra_cc', 'flags_extra_hip', 'blob_gen_cmd',
            'extra_include', 'extra_ldflags', 'torch_exclude', 'third_party')
    core.build_module(md_name='module_aiter_core', **{key: args[key] for key in keys},
                      verbose=False, is_python_module=True, is_standalone=False)
    modules = sorted(seed.glob('module_aiter_core*.so'))
    assert modules, f'No core module built in {seed}'
    # Build trees can contain absolute paths, locks and large compiler intermediates.
    shutil.rmtree(seed / 'build')
    (seed / 'build').mkdir()
    manifest = dict(
        packages={name: metadata.version(name) for name in ('amd-aiter', 'torch', 'triton')},
        gpu_archs=os.environ['GPU_ARCHS'], cu_num=int(os.environ['CU_NUM']),
        core_source_sha256=hashlib.sha256((package / 'jit/core.py').read_bytes()).hexdigest(),
        modules={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in modules},
        scope='Core import seed only; attention and other workload kernels remain runtime JIT.')
    (seed / 'seed.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    # Runtime copies to private caches, never compiles into the baked seed itself.
    for path in sorted(seed.rglob('*'), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    seed.chmod(0o555)
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
