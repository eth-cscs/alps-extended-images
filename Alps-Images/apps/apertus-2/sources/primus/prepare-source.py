#!/usr/bin/env python3
"""Apply the audited Turbo 0.5.1 adapter and record installed source identity."""
import hashlib
from importlib import metadata
import json
from pathlib import Path
import subprocess


def prepare(root, source, turbo_version, torch_version):
    """Apply the adapter for a supported version pair and verify every source file."""
    identity=json.loads((root/'vendor/manifest.json').read_text())
    identity['applied_patches']=[]
    if turbo_version=='0.5.1.dev7' and torch_version=='2.14.0+rocm10.1.0':
        patch=root/'patches/0001-turbo051-private-api-imports.patch'
        subprocess.run(['git','apply','--check',str(patch.resolve())],cwd=source,check=True)
        subprocess.run(['git','apply',str(patch.resolve())],cwd=source,check=True)
        changed=json.loads((root/'patches/primus268-source-hashes.json').read_text())
        identity['files'].update(changed)
        identity['applied_patches'].append(dict(file=patch.name,sha256=hashlib.sha256(patch.read_bytes()).hexdigest()))
    elif not (turbo_version=='0.2.0+3cd482d' and torch_version=='2.10.0+git94c6e04'):
        raise RuntimeError(f'Unaudited Primus Torch/Turbo pair: {torch_version}/{turbo_version}')
    for name,sha in identity['files'].items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest()==sha,name
    (root/'source-identity.json').write_text(json.dumps(identity,indent=2)+'\n')


if __name__=='__main__':
    prepare(Path('/opt/alps/apertus2-primus'),Path('/opt/megatron-lm'),
            metadata.version('primus-turbo'),metadata.version('torch'))
