#!/usr/bin/env python3
"""Install the archived reference commit, including licenses, without scratch access."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile


def install(source, destination):
    """Extract a verified archive into a new directory and check its manifest."""
    manifest = json.loads((source/'manifest.json').read_text())
    archive = source/manifest['archive']
    if hashlib.sha256(archive.read_bytes()).hexdigest() != manifest['archive_sha256']:
        raise RuntimeError('Megatron source archive checksum mismatch')
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive) as stream:
        members = stream.getmembers()
        seen = set()
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or '..' in path.parts or member.name in seen:
                raise RuntimeError('Unsafe/duplicate source archive path')
            seen.add(member.name)
            if member.isfile():
                if member.name not in manifest['files']:
                    raise RuntimeError('Unmanifested source file')
            elif member.issym():
                if manifest.get('symlinks', {}).get(member.name) != member.linkname:
                    raise RuntimeError('Unmanifested source symlink')
            elif not member.isdir():
                raise RuntimeError('Unsupported source archive member')
        stream.extractall(destination, filter='data')
    for name, expected in manifest['files'].items():
        path = destination/name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Source file checksum mismatch: {name}')
    for name, target in manifest.get('symlinks', {}).items():
        path = destination/name
        if not path.is_symlink() or str(path.readlink()) != target:
            raise RuntimeError(f'Source symlink mismatch: {name}')
    print(f'Installed exact Megatron {manifest["revision"]}: {len(manifest["files"])} files')


if __name__ == '__main__': install(Path(sys.argv[1]), Path(sys.argv[2]))
