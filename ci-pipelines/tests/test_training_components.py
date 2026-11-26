"""CPU contracts for the combined Primus26.8 UCCL/AITER candidate; not GPU correctness."""

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from test_primus import ROOT, bash


APP = ROOT / 'Alps-Images/apps/apertus-2'


def seed_module():
    """Load the GPU-free seed entry point without importing vendor GPU libraries."""
    spec = importlib.util.spec_from_file_location('seed_aiter', APP / 'sources/primus/seed-aiter.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('failure', [None, 'version', 'existing', 'compile'])
def test_aiter_seed_preserves_vendor_sources_and_fails_closed(tmp_path, monkeypatch, failure):
    module = seed_module()
    package = tmp_path / 'site-packages/aiter'
    (package / 'jit').mkdir(parents=True)
    source = package / 'jit/core.py'
    source.write_text('# vendor source, do not rewrite\n')
    seed = tmp_path / 'seed'
    monkeypatch.setenv('AITER_JIT_DIR', str(seed))
    monkeypatch.setenv('GPU_ARCHS', 'gfx942')
    monkeypatch.setenv('CU_NUM', '228')
    versions = {'amd-aiter': 'wrong' if failure == 'version' else '0.1.22.post1',
                'torch': '2.14.0+rocm10.1.0', 'triton': 'vendor-triton'}
    monkeypatch.setattr(module.metadata, 'version', versions.__getitem__)
    monkeypatch.setattr(module.util, 'find_spec', lambda name: SimpleNamespace(submodule_search_locations=[str(package)]))
    builds = []
    jit = ModuleType('jit')
    keys = ('srcs', 'flags_extra_cc', 'flags_extra_hip', 'blob_gen_cmd',
            'extra_include', 'extra_ldflags', 'torch_exclude', 'third_party')
    def build(**kwargs):
        builds.append(kwargs)
        if failure == 'compile':
            raise RuntimeError('compiler failure')
        (seed / 'build').mkdir()
        (seed / 'build/intermediate.o').write_bytes(b'compiler intermediates')
        (seed / 'module_aiter_core.so').write_bytes(b'fake extension for orchestration test only')
    jit.core = SimpleNamespace(get_args_of_build=lambda name: dict.fromkeys(keys, []), build_module=build)
    monkeypatch.setitem(sys.modules, 'jit', jit)
    old_path = list(sys.path)
    if failure == 'existing':
        seed.mkdir()
        (seed / 'user-cache').write_text('preserve me')
    try:
        if failure:
            exception = {'version': AssertionError, 'existing': FileExistsError, 'compile': RuntimeError}[failure]
            with pytest.raises(exception):
                module.main()
            assert not (seed / 'seed.json').exists()
            assert bool(builds) == (failure == 'compile')
            if failure == 'version':
                assert not seed.exists()
            if failure == 'existing':
                assert (seed / 'user-cache').read_text() == 'preserve me'
        else:
            module.main()
            manifest = json.loads((seed / 'seed.json').read_text())
            assert manifest['packages'] == versions
            assert manifest['gpu_archs'] == 'gfx942' and manifest['cu_num'] == 228
            assert list(manifest['modules']) == ['module_aiter_core.so']
            assert not list((seed / 'build').iterdir())
            assert seed.stat().st_mode & 0o222 == 0
            assert builds[0]['md_name'] == 'module_aiter_core'
        assert source.read_text() == '# vendor source, do not rewrite\n'
    finally:
        sys.path[:] = old_path
        # This test owns the read-only seed, not any external image or cache.
        if seed.exists():
            for path in [seed, *seed.rglob('*')]:
                path.chmod(0o700 if path.is_dir() else 0o600)


def test_combined_candidate_does_not_replace_vendor_packages_or_network_policy():
    recipe = (APP / 'Containerfile.megatron.primus268').read_text()
    assert recipe.index('install-components.sh uccl') < recipe.index('install-components.sh aiter')
    assert 'seed-aiter.py' in recipe and 'check-primus-compute.py --compare' in recipe
    assert 'NCCL_P2P_DISABLE' not in recipe and 'UCCL_EP_TRANSPORT=' not in recipe
    assert 'FROM ${BASE_IMAGE}' in recipe
    installer = (APP / 'sources/primus/install-components.sh').read_text()
    assert '--no-deps' in installer and 'PIP_CONSTRAINT=' in installer
    seed = (APP / 'sources/primus/seed-aiter.py').read_text()
    assert 'pip_install' not in seed and 'exist_ok=False' in seed
    patch = (APP / 'patches/rocm/uccl/0001-alps-cxi-rocm-ep.patch').read_text()
    assert '+  static std::mutex cxi_init_mutex;' not in patch
    assert patch.count('+    fi_info* selected = select_cxi_info(info, cxi_local_rank);') == 1
    assert '+      selected = select_cxi_info(info, cxi_local_rank);' in patch


@pytest.mark.parametrize('existing', [False, True])
@pytest.mark.parametrize('failure', [False, True])
def test_uccl_component_installs_pkg_config_and_purges_only_added_packages(tmp_path, existing, failure):
    """Exercise the minimal base's missing pkg-config and preserve pre-existing tools."""
    alps = tmp_path / 'alps'
    (alps / 'env').mkdir(parents=True)
    (alps / 'apertus2-rocm').mkdir()
    ready = tmp_path / 'pkg-config-ready'
    purged = tmp_path / 'purged'
    compiled = tmp_path / 'compiled'
    if existing:
        ready.touch()
    (alps / 'env/alps-rocm-build.env').write_text('ROCM_BUILD_PREFIX=/sdk\nROCM_SDK_BIN=/sdk/bin\n')
    (alps / 'package-helpers.sh').write_text(f'''
apt_get() {{
    if [[ "$1" = install && " $* " = *" pkg-config "* ]]; then touch "{ready}"; fi
}}
pip_install() {{ :; }}
dpkg-query() {{
    if [[ "${{@: -1}}" = pkg-config && ! -e "{ready}" ]]; then return 1; fi
    printf 'ii'
}}
purge_apt_build_deps_exact() {{ printf '%s\\n' "$@" > "{purged}"; }}
''')
    (alps / 'apertus2-rocm/install-uccl.sh').write_text(
        f'test -e "{ready}" || exit 127\ntouch "{compiled}"\nexit {7 if failure else 0}\n')
    installer = (APP / 'sources/primus/install-components.sh').read_text()
    installer = installer.replace('/opt/alps', str(alps)).replace('rm -rf /var/lib/apt/lists/*', ':')
    isolated = tmp_path / 'install-components.sh'
    isolated.write_text(installer)
    result = bash(f'bash "{isolated}" uccl', check=False)
    assert result.returncode == (7 if failure else 0), result.stdout + result.stderr
    assert compiled.exists()
    if failure:
        assert not purged.exists()
    else:
        removed = purged.read_text().split()
        assert ('pkg-config' in removed) == (not existing)
        assert not any(name.startswith('lib') for name in removed)


def test_history_corrections_preserve_cleanup_order_and_native_positional_api():
    """Reject the disproven extra barrier, split NIC affinity and shifted byte counts."""
    patches = APP / 'patches/rocm/uccl'
    cxi = (patches / '0001-alps-cxi-rocm-ep.patch').read_text()
    ownership = (patches / '0002-preserve-torch-resource-ownership.patch').read_text()
    assert 'diff --git a/ep/bench/utils.py' not in cxi
    assert '+    dist.barrier(group)' not in cxi
    assert '+    ctx_.numa_node = cxi_nic_numa_node(cfg_.nic_local_rank);' in cxi
    assert '+              cfg_.nic_local_rank, cfg_.thread_idx);' in cxi
    assert '+            comm_stream_ptr=int(self._comm_stream.cuda_stream),' in ownership
    assert '+            int(self._comm_stream.cuda_stream),' not in ownership
    assert '+         std::uintptr_t comm_stream_ptr = 0)' in ownership
    assert '+           nb::arg("num_local_ranks") = -1, nb::arg("comm_stream_ptr") = 0)' in ownership
    assert '+    if (owns_comm_stream && comm_stream != nullptr)' in ownership
    assert '+  bool owns_comm_stream{false};' in ownership
    assert '+  proxy_->destroy(false);' in ownership


def test_primus268_uses_minimal_patch_without_changing_legacy_compiler_support():
    """The exact10.1 codegen ablation justifies removal only from this selected target."""
    recipe = (APP / 'Containerfile.megatron.primus268').read_text()
    assert 'COPY Alps-Images/apps/apertus-2/patches/rocm/uccl-primus268/' in recipe
    assert 'COPY Alps-Images/apps/apertus-2/patches/rocm/uccl/ /' not in recipe
    selected = (APP / 'patches/rocm/uccl-primus268/0001-alps-cxi-rocm-ep.patch').read_text()
    legacy = (APP / 'patches/rocm/uccl/0001-alps-cxi-rocm-ep.patch').read_text()
    for name in ('ep/bench/utils.py', 'ep/include/ep_configs.cuh',
                 'ep/src/internode.cu', 'ep/src/internode_ll.cu'):
        assert f'diff --git a/{name}' not in selected
    assert '+  attr.iface = FI_HMEM_ROCR;' in selected
    assert '+  CUDA_CHECK(hipLaunchKernelEx' in selected
    assert '+    ctx_.numa_node = cxi_nic_numa_node(cfg_.nic_local_rank);' in selected
    assert 'diff --git a/ep/include/ep_configs.cuh' in legacy


def component_module():
    """Load the CPU-side linkage gate without importing Torch or native GPU modules."""
    spec = importlib.util.spec_from_file_location('check_components', APP / 'tests/primus/check-components.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('failure', [None, 'c10', 'torch', 'other', 'wrong-torch'])
def test_component_ldd_uses_installed_torch_without_changing_runtime_env(tmp_path, monkeypatch, failure):
    module = component_module()
    torch_lib = tmp_path / 'torch/lib'
    torch_lib.mkdir(parents=True)
    monkeypatch.setenv('LD_LIBRARY_PATH', '/canonical/network/lib')
    lines = [f'libc10.so => {torch_lib}/libc10.so (0x1)',
             f'libtorch.so => {torch_lib}/libtorch.so (0x2)',
             'libfabric.so.1 => /usr/lib/libfabric.so.1 (0x3)']
    if failure in {'c10', 'torch'}:
        index = 0 if failure == 'c10' else 1
        lines[index] = ('libc10.so' if index == 0 else 'libtorch.so') + ' => not found'
    elif failure == 'other':
        lines.append('libnuma.so.1 => not found')
    elif failure == 'wrong-torch':
        lines[1] = 'libtorch.so => /unrelated/libtorch.so (0x2)'

    def ldd(argv, **kwargs):
        assert argv == ['ldd', str(tmp_path / 'ep.so')]
        assert kwargs['env']['LD_LIBRARY_PATH'] == f'{torch_lib}:/canonical/network/lib'
        return '\n'.join(lines)

    monkeypatch.setattr(module.subprocess, 'check_output', ldd)
    if failure:
        with pytest.raises(AssertionError):
            module.linked_dependencies(tmp_path / 'ep.so', torch_lib)
    else:
        assert 'libfabric.so.1 => /usr/lib/' in module.linked_dependencies(tmp_path / 'ep.so', torch_lib)
    assert module.os.environ['LD_LIBRARY_PATH'] == '/canonical/network/lib'


def test_preloaded_library_import_and_fresh_ldd_have_distinct_search_scopes(tmp_path, monkeypatch):
    """Exercise real ELF loading with tiny CPU fixtures, not Torch/GPU acceptance."""
    compiler = shutil.which('cc')
    if compiler is None or shutil.which('ldd') is None:
        pytest.skip('CPU ELF compiler and ldd required')
    torch_lib = tmp_path / 'torch/lib'
    torch_lib.mkdir(parents=True)
    sources = dict(c10='int c10_value(void) { return 7; }',
                   torch='extern int c10_value(void); int torch_value(void) { return c10_value(); }',
                   ep='extern int c10_value(void); extern int torch_value(void); int result(void) { return c10_value()+torch_value(); }')
    for name, source in sources.items():
        (tmp_path / f'{name}.c').write_text(source)
    for name in ('c10', 'torch'):
        command = [compiler, '-shared', '-fPIC', f'-Wl,-soname,lib{name}.so',
                   str(tmp_path / f'{name}.c'), '-o', str(torch_lib / f'lib{name}.so')]
        if name == 'torch':
            command += ['-L', str(torch_lib), '-lc10']
        subprocess.run(command, check=True, capture_output=True)
    extension = tmp_path / 'ep.so'
    subprocess.run([compiler, '-shared', '-fPIC', str(tmp_path / 'ep.c'), '-L', str(torch_lib),
                    '-ltorch', '-lc10', '-o', str(extension)], check=True, capture_output=True)
    monkeypatch.delenv('LD_LIBRARY_PATH', raising=False)
    script = ('import ctypes; '
              f'ctypes.CDLL({str(torch_lib / "libc10.so")!r}, mode=ctypes.RTLD_GLOBAL); '
              f'ctypes.CDLL({str(torch_lib / "libtorch.so")!r}, mode=ctypes.RTLD_GLOBAL); '
              f'assert ctypes.CDLL({str(extension)!r}).result() == 14')
    subprocess.run([sys.executable, '-c', script], check=True, capture_output=True)
    raw = subprocess.check_output(['ldd', str(extension)], text=True)
    assert 'not found' in raw
    assert 'not found' not in component_module().linked_dependencies(extension, torch_lib)
    assert 'LD_LIBRARY_PATH' not in component_module().os.environ


@pytest.mark.parametrize('relative', ['patches/rocm/uccl/0002-preserve-torch-resource-ownership.patch',
                                      'sources/primus/seed-aiter.py',
                                      'sources/primus/install-components.sh',
                                      'tests/primus/check-components.py'])
def test_combined_component_inputs_invalidate_app_not_base(tmp_path, relative):
    for name in ('Alps-Images', 'ci-pipelines'):
        shutil.copytree(ROOT / name, tmp_path / name, ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copyfile(ROOT / '.dockerignore', tmp_path / '.dockerignore')
    command = 'source ci-pipelines/helpers/meta.sh; app_refs apertus-2 megatron-primus268-rocm'
    before = bash(command, cwd=tmp_path).stdout.split()
    path = tmp_path / 'Alps-Images/apps/apertus-2' / relative
    path.write_text(path.read_text() + '\n# component hash probe\n')
    after = bash(command, cwd=tmp_path).stdout.split()
    assert before[0] == after[0]
    assert before[2] != after[2]
