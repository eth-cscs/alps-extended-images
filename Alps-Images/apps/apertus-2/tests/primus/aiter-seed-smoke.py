#!/usr/bin/env python3
"""Copy the baked AITER core to a private cache, check ABI/import reuse, optionally run RMSNorm."""

import argparse
import hashlib
import importlib
from importlib import metadata, util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile


def main():
    """Never write to the seed; all imports and optional GPU JIT use a private copy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata-only', action='store_true')
    args = parser.parse_args()
    seed = Path('/opt/aiter-jit')
    manifest = json.loads((seed / 'seed.json').read_text())
    assert manifest['gpu_archs'] == 'gfx942' and manifest['cu_num'] == 228
    for name, version in manifest['packages'].items():
        assert metadata.version(name) == version, (name, version, metadata.version(name))
    package = Path(next(iter(util.find_spec('aiter').submodule_search_locations)))
    assert hashlib.sha256((package / 'jit/core.py').read_bytes()).hexdigest() == manifest['core_source_sha256']
    originals = {name: ((seed / name).stat().st_mtime_ns, hashlib.sha256((seed / name).read_bytes()).hexdigest())
                 for name in manifest['modules']}
    assert {name: value[1] for name, value in originals.items()} == manifest['modules']
    with tempfile.TemporaryDirectory(prefix='apertus2-aiter-seed-') as private:
        cache = Path(private) / 'aiter'
        # copyfile intentionally does not propagate the immutable seed's modes.
        shutil.copytree(seed, cache, copy_function=shutil.copyfile)
        for path in [cache, *cache.rglob('*')]:
            path.chmod(0o700 if path.is_dir() else 0o600)
        os.environ.update(AITER_JIT_DIR=str(cache), GPU_ARCHS='gfx942', CU_NUM='228',
                          AITER_USE_SYSTEM_TRITON='1')
        before = {name: (cache / name).stat().st_mtime_ns for name in manifest['modules']}
        import torch

        sys.path.insert(0, str(cache))
        core = importlib.import_module('module_aiter_core')
        assert Path(core.__file__).resolve().is_relative_to(cache)
        if not args.metadata_only:
            local_rank = int(os.environ.get('SLURM_LOCALID', '0'))
            torch.cuda.set_device(local_rank)
            properties = torch.cuda.get_device_properties(local_rank)
            assert properties.gcnArchName.startswith('gfx942')
            assert properties.multi_processor_count == 228
            import aiter

            x = torch.randn(64, 4096, device='cuda', dtype=torch.bfloat16)
            weight = torch.ones(4096, device='cuda', dtype=torch.bfloat16)
            actual = aiter.rms_norm(x, weight, 1e-6)
            expected = torch.nn.functional.rms_norm(x.float(), (4096,), weight.float(), 1e-6)
            torch.testing.assert_close(actual.float(), expected, rtol=0.02, atol=0.02)
            torch.cuda.synchronize()
        for name, checksum in manifest['modules'].items():
            assert (cache / name).stat().st_mtime_ns == before[name], f'Core rebuilt: {name}'
            assert hashlib.sha256((cache / name).read_bytes()).hexdigest() == checksum
            assert ((seed / name).stat().st_mtime_ns, hashlib.sha256((seed / name).read_bytes()).hexdigest()) == originals[name]
    scope = 'Private-copy core ABI/import without recompilation; no GPU acceptance.' if args.metadata_only else \
            'Private-copy core import reuse plus single-GPU RMSNorm; attention/training acceptance is separate.'
    print(json.dumps(dict(status='passed', seed=str(seed), scope=scope), indent=2))


if __name__ == '__main__':
    main()
