#!/usr/bin/env python3
"""Fingerprint the vendor compute stack around Alps network installation (no GPU init)."""
import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path

PACKAGES = ('torch', 'torchvision', 'torchaudio', 'triton', 'pytorch-triton-rocm',
            'triton-rocm', 'numpy', 'transformer-engine', 'transformer_engine_torch',
            'transformer_engine_rocm10', 'amd-aiter', 'primus-turbo', 'apex',
            'causal-conv1d', 'mamba-ssm', 'rocm-sdk-core', 'rocm-sdk-libraries', 'rocm-sdk-devel')


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def snapshot():
    import torch
    result = dict(packages={}, files={}, torch_hip=torch.version.hip)
    for name in PACKAGES:
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        result['packages'][name] = dist.version
        # SDK distributions deliberately replace RCCL only. Hash their main
        # compute libraries below, not RECORD entries claiming old RCCL bytes.
        if name.startswith('rocm-sdk-'):
            continue
        for entry in dist.files or []:
            path = Path(dist.locate_file(entry))
            if not path.is_file() or path.name.startswith('librccl'):
                continue
            if path.suffix not in ('.so', '.py') and path.name not in ('METADATA', 'direct_url.json'):
                continue
            result['files'][str(path)] = digest(path)
    prefixes = [Path(os.environ.get('ROCM_PATH', '/opt/rocm')), Path('/opt/rocm')]
    for name in ('rocm-sdk-core', 'rocm-sdk-libraries', 'rocm-sdk-devel'):
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        for entry in dist.files or []:
            if Path(entry).name.startswith(('libamdhip64.so', 'libhsa-runtime64.so', 'librocblas.so', 'libhipblaslt.so', 'libamd_comgr.so')):
                path = Path(dist.locate_file(entry))
                if path.is_file(): result['files'][str(path.resolve())] = digest(path)
    for prefix in prefixes:
        for name in ('libamdhip64.so', 'libhsa-runtime64.so', 'librocblas.so', 'libhipblaslt.so', 'libamd_comgr.so'):
            for libdir in ('lib', 'lib64'):
                for path in (prefix/libdir).glob(name+'*'):
                    if path.is_file(): result['files'][str(path.resolve())] = digest(path)
    if not result['torch_hip'] or 'torch' not in result['packages']:
        raise RuntimeError('Primus must provide a ROCm-enabled Torch compute stack')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', type=Path)
    parser.add_argument('--compare', type=Path)
    args = parser.parse_args()
    if bool(args.write) == bool(args.compare): parser.error('Choose --write or --compare')
    actual = snapshot()
    if args.write:
        args.write.write_text(json.dumps(actual, indent=2, sort_keys=True)+'\n')
    else:
        expected = json.loads(args.compare.read_text())
        if actual != expected:
            differences = {key: dict(expected=expected.get(key), actual=actual.get(key))
                           for key in ('packages', 'torch_hip') if expected.get(key) != actual.get(key)}
            changed = [key for key in set(expected['files']) | set(actual['files'])
                       if expected['files'].get(key) != actual['files'].get(key)]
            raise RuntimeError(f'Vendor compute stack changed: {differences}; changed files: {changed}')
    print(f'Primus compute fingerprint verified: {len(actual["packages"])} packages / {len(actual["files"])} files; HIP {actual["torch_hip"]}')


if __name__ == '__main__': main()
