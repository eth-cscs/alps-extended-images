#!/usr/bin/env python3
"""Check the combined candidate's UCCL native component and ownership corrections, no GPU."""

import hashlib
from importlib import metadata
import inspect
import json
import os
from pathlib import Path
import subprocess


def linked_dependencies(extension, torch_lib):
    """Resolve a fresh ldd process against the same installed Torch as the prior import."""
    torch_lib = torch_lib.resolve()
    assert torch_lib.is_dir(), torch_lib
    # Importing Torch preloads its libraries in this process, not in the ldd
    # child. Scope the matching search directory to this diagnostic only.
    inherited = os.environ.get('LD_LIBRARY_PATH', '')
    search = str(torch_lib) + (os.pathsep + inherited if inherited else '')
    linkage = subprocess.check_output(['ldd', str(extension)], text=True,
                                      env=dict(os.environ, LD_LIBRARY_PATH=search))
    assert 'not found' not in linkage, linkage
    for name in ('libc10.so', 'libtorch.so'):
        matches = [line.strip() for line in linkage.splitlines() if line.strip().startswith(name + ' => ')]
        assert len(matches) == 1, linkage
        resolved = Path(matches[0].split('=>', 1)[1].split()[0]).resolve()
        assert resolved == (torch_lib / name).resolve(), linkage
    return linkage


def main():
    """Verify installed native code, source/patch identity and canonical fabric linkage."""
    import torch
    from uccl import ep
    from deep_ep import Buffer, Config

    source = Path('/opt/uccl')
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    assert revision == '52d4184c89deb5631be6dac6a9b1c40896c9e26e', revision
    extension = Path(ep.__file__).resolve()
    assert 'site-packages/uccl/' in str(extension), extension
    assert extension.read_bytes() == (source / 'uccl' / extension.name).read_bytes()
    assert 'comm_stream_ptr' in ep.Buffer.__init__.__doc__
    wrapper = inspect.getsource(Buffer)
    assert 'self._comm_stream = torch.cuda.Stream(device=device_index, priority=-1)' in wrapper
    assert 'comm_stream_ptr=int(self._comm_stream.cuda_stream)' in wrapper
    assert 'return self._comm_stream' in inspect.getsource(Buffer.get_comm_stream)
    assert '_record_stream_safe' in wrapper
    proxy = (source / 'ep/src/uccl_proxy.cpp').read_text()
    assert 'proxy_->destroy(false);' in proxy
    native = (source / 'ep/src/uccl_ep.cc').read_text()
    assert 'cudaMalloc(&d_handles,' in native
    assert 'if (owns_comm_stream && comm_stream != nullptr)' in native
    assert native.count('cudaStreamDestroy(comm_stream)') == 1
    assert 'nb::arg("num_local_ranks") = -1, nb::arg("comm_stream_ptr") = 0' in native
    cxi = (source / 'ep/src/cxi_transport.cpp').read_text()
    assert 'FI_HMEM_ROCR' in cxi and 'g_cxi_setup_mu' in cxi
    assert cxi.count('select_cxi_info(info, cxi_local_rank)') == 2
    # Every retained patch must be applied, not just copied into the image.
    patches = sorted(Path('/opt/alps/apertus2-rocm/patches/uccl').glob('*.patch'))
    for patch in patches:
        subprocess.run(['git', '-C', str(source), 'apply', '--reverse', '--check', str(patch)], check=True)
    # This profile removes the older compiler/kernel rewrites and extra host
    # barrier; hipify generates separate files, leaving these tracked inputs intact.
    subprocess.run(['git', '-C', str(source), 'diff', '--quiet', 'HEAD', '--',
                    'ep/bench/utils.py', 'ep/include/ep_configs.cuh',
                    'ep/src/internode.cu', 'ep/src/internode_ll.cu'], check=True)
    torch_lib = Path(torch.__file__).resolve().parent / 'lib'
    linkage = linked_dependencies(extension, torch_lib)
    fabric_dir = Path('/etc/ld.so.conf.d/98-apertus2-uccl.conf').read_text().strip()
    fabric_lines = [line.strip() for line in linkage.splitlines() if 'libfabric.so' in line]
    assert fabric_lines and all(f'=> {fabric_dir}/' in line for line in fabric_lines), linkage
    assert Config(64, 8, 512, 16, 512).num_sms == 64
    assert torch.version.hip
    print(json.dumps(dict(status='passed', uccl_revision=revision, extension=str(extension),
        packages={name: metadata.version(name) for name in ('uccl', 'torch', 'amd-aiter')},
        patches={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in patches},
        torch_library_directory=str(torch_lib),
        fabric=fabric_lines,
        scope='Native import, pins, applied corrections and linkage only; GPU/CXI lifetime awaits scheduled smoke.'), indent=2))


if __name__ == '__main__':
    main()
