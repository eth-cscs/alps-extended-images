"""Primus uses the ordinary ROCm graph, with pinned upstreams and SDK layout guards."""
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def bash(command, cwd=ROOT, check=True):
    env = dict(os.environ, IMAGE_PREFIX='localhost/alps-images', GHCR_IMAGE_PREFIX='localhost/ghcr',
               ALPS_REV='alps7-dev', CI_COMMIT_SHA='a'*40, CI_COMMIT_SHORT_SHA='aaaaaaa',
               CSCS_CI_ORIG_CLONE_URL='https://example.invalid/repo.git')
    value = subprocess.run(['bash', '-c', command], cwd=cwd, env=env, text=True, capture_output=True)
    if check: assert value.returncode == 0, value.stdout + value.stderr
    return value


@pytest.mark.parametrize('version,rocm,layout', [('26.3','7.2.1','system'),('26.8','10.1.0','wheel')])
def test_primus_base_refs_manual_and_ci_args_agree(tmp_path, version, rocm, layout):
    value = bash(f'source ci-pipelines/helpers/meta.sh; rocm_base_refs primus {version}').stdout.split()
    assert value[0].startswith('docker.io/rocm/primus@sha256:')
    assert value[1] == 'Alps-Images/ROCm/Containerfile.primus-alps'
    assert value[2].startswith(f'localhost/alps-images/primus-rocm:{version}-alps7-dev-')
    script = bash(f'manual-build/manual-build.sh base rocm primus {version}').stdout
    assert f'--build-arg ROCM_VERSION="{rocm}"' in script
    assert f'--build-arg ALPS_ROCM_VERSION="{rocm}"' in script
    assert f'--build-arg ROCM_SDK_LAYOUT="{layout}"' in script
    assert '--build-arg ROCM_REBUILD_RCCL="1"' in script
    assert '--build-arg ROCM_PRESERVE_COMPUTE="1"' in script
    assert value[2] in script
    path = tmp_path/'build.env'
    bash(f'source ci-pipelines/helpers/skopeo.sh; source ci-pipelines/helpers/meta.sh; write_base_build_env {path} rocm primus {version}')
    variables = dict(line.split('=',1) for line in path.read_text().splitlines())
    assert variables['CANON_IMAGE_REF'] == value[2]
    assert variables['ROCM_VERSION'] == rocm
    assert variables['ALPS_ROCM_VERSION'] == rocm
    template = yaml.safe_load((ROOT/'ci-pipelines/child-templates.yaml').read_text())['.child-rocm-base-build-template']
    args = json.loads(template['variables']['DOCKER_BUILD_ARGS'])
    assert {'ALPS_ROCM_VERSION','ROCM_SDK_LAYOUT','ROCM_PRESERVE_COMPUTE'} <= set(args)


def test_system_bootstrap_uses_existing_sdk_without_pip(tmp_path):
    root = tmp_path/'rocm'
    for relative in ('include/hip/hip_runtime_api.h','include/hsa/hsa.h','.info/version'):
        path = root/relative;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('7.2.1-12345' if relative.endswith('version') else '')
    value = bash(f'set -euo pipefail; source Alps-Images/ROCm/install-alps-rocm-components.sh; '
                 f'ROCM_VERSION=7.2.1; ROCM_PATH={root}; '
                 'register_rocm_sdk_ldconfig() { :; }; persist_rocm_sdk_env() { :; }; record_alps_version_var() { :; }; '
                 'bootstrap_rocm_system; printf "PREFIX=%s\n" "$ROCM_BUILD_PREFIX"')
    assert f'PREFIX={root}' in value.stdout
    wrong = bash(f'set -euo pipefail; source Alps-Images/ROCm/install-alps-rocm-components.sh; '
                 f'ROCM_VERSION=7.14.0; ROCM_PATH={root}; die() {{ echo "$*" >&2; exit 1; }}; bootstrap_rocm_system',check=False)
    assert wrong.returncode != 0
    assert 'differs from requested' in wrong.stderr


def test_preserved_wheel_bootstrap_cannot_install_missing_runtime():
    value = bash('set -euo pipefail; source Alps-Images/ROCm/install-alps-rocm-components.sh; '
                 'die() { echo "$*" >&2; exit 1; }; '
                 'rocm_sdk_distribution_present() { return 1; }; '
                 'ROCM_VERSION=10.1.0; ROCM_PYTHON=/does/not/exist; '
                 'ROCM_PRESERVE_COMPUTE=1; RCCL_GPU_TARGETS=gfx942; '
                 'ROCM_LIBRARIES_REPO=; ROCM_LIBRARIES_COMMIT=; install_rocm_sdk_wheels',check=False)
    assert value.returncode != 0
    assert 'refusing to install/replace' in value.stderr


def test_primus_profile_changes_invalidate_the_selected_base(tmp_path):
    for name in ('Alps-Images','ci-pipelines'):
        shutil.copytree(ROOT/name,tmp_path/name,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copyfile(ROOT/'.dockerignore',tmp_path/'.dockerignore')
    command='source ci-pipelines/helpers/meta.sh; rocm_base_refs primus 26.3'
    before=bash(command,cwd=tmp_path).stdout.split()[2]
    profile=tmp_path/'Alps-Images/ROCm/primus-26.3/profile.env'
    profile.write_text(profile.read_text().replace('ROCM_PRESERVE_COMPUTE="1"','ROCM_PRESERVE_COMPUTE="0"'))
    assert bash(command,cwd=tmp_path).stdout.split()[2] != before


def source_installer():
    import importlib.util
    path = ROOT/'Alps-Images/apps/apertus-2/sources/primus/install-source.py'
    spec = importlib.util.spec_from_file_location('primus_source', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reference_archive_is_self_contained_and_exact(tmp_path):
    import hashlib
    vendor = ROOT/'Alps-Images/apps/apertus-2/sources/primus/vendor'
    destination = tmp_path/'megatron'
    source_installer().install(vendor, destination)
    manifest = json.loads((vendor/'manifest.json').read_text())
    assert manifest['revision'] == 'd1277d86e0c1925f8438fd7630d86d05ab398dcd'
    assert (destination/'LICENSE').is_file()
    assert len(manifest['files']) > 2000
    for name, sha in manifest['files'].items():
        assert hashlib.sha256((destination/name).read_bytes()).hexdigest() == sha


@pytest.mark.parametrize('unsafe', ['../escape', '/absolute', 'duplicate'])
def test_reference_archive_rejects_unsafe_paths(tmp_path, unsafe):
    import hashlib
    import io
    import tarfile
    archive = tmp_path/'source.tar.gz'
    with tarfile.open(archive, 'w:gz') as stream:
        for name in (['duplicate', 'duplicate'] if unsafe == 'duplicate' else [unsafe]):
            member = tarfile.TarInfo(name); member.size = 1
            stream.addfile(member, io.BytesIO(b'x'))
    (tmp_path/'manifest.json').write_text(json.dumps(dict(archive=archive.name,
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        files={unsafe: hashlib.sha256(b'x').hexdigest()})))
    with pytest.raises(RuntimeError, match='Unsafe/duplicate'):
        source_installer().install(tmp_path, tmp_path/'destination')


def test_reference_environment_preserves_empty_and_explicit_overrides():
    runtime = 'Alps-Images/apps/apertus-2/sources/primus/reference.env'
    value = bash('source Alps-Images/common/alps-runtime.common.env; '
                 'BEVERIN_T6_EP=; BEVERIN_TURBO_GG=custom; NCCL_P2P_DISABLE=1; '
                 f'source {runtime}; '
                 '[[ "$BEVERIN_T6_EP" == "" && "$BEVERIN_TURBO_GG" == custom && "$NCCL_P2P_DISABLE" == 1 ]]')
    assert value.returncode == 0
    text = (ROOT/runtime).read_text()
    assert not re.search(r'^defvar (?:NCCL_|FI_|HSA_)', text, flags=re.M)


@pytest.mark.parametrize('inherited', [False, True])
@pytest.mark.parametrize('exit_code', [0, 23])
def test_reference_wrapper_in_child_shell(tmp_path, inherited, exit_code):
    common = ROOT / 'Alps-Images/common'
    app = ROOT / 'Alps-Images/apps/apertus-2'
    runtime = tmp_path / 'env'
    runtime.mkdir()
    # Use the installed runtime layout without touching host /opt or requiring GPUs.
    fragments = [common / 'alps-runtime.common.env', common / 'alps-runtime.nccl-like.env',
                 ROOT / 'Alps-Images/ROCm/alps-runtime.rocm.env']
    (runtime / 'alps-runtime.env').write_text(''.join(path.read_text() for path in fragments))
    hook = runtime / 'source-alps-env.sh'
    hook.write_text((common / hook.name).read_text().replace('/opt/alps/env', str(runtime)))
    reference = tmp_path / 'reference.env'
    shutil.copyfile(app / 'sources/primus/reference.env', reference)
    wrapper = tmp_path / 'with-reference-env.sh'
    wrapper.write_text((app / 'tests/primus' / wrapper.name).read_text()
                       .replace('/opt/alps/env', str(runtime))
                       .replace('/opt/alps/apertus2-primus/reference.env', str(reference)))
    probe = tmp_path / 'probe.sh'
    probe.write_text('set -euo pipefail\n'
                     '[[ "$BEVERIN_T6_EP" == "" && "$BEVERIN_TURBO_GG" == custom ]]\n'
                     '[[ "$BEVERIN_AITER_ATTN" == 1 && "$FI_PROVIDER" == cxi ]]\n'
                     '[[ "$NCCL_NET" == "AWS Libfabric" && "$NCCL_P2P_DISABLE" == 1 ]]\n'
                     '[[ "$#" == 1 && "$1" == "argument with spaces" ]]\n'
                     'echo probe-reached\n'
                     f'exit {exit_code}\n')
    # BASH_ENV runs before the wrapper, just as it does in the container.
    # Only the exported marker and defaults survive exec into a fresh bash.
    setup = f'source "{hook}"; declare -F defvar >/dev/null; ' if inherited else ''
    value = bash('unset ALPS_EXTENDED_IMAGES_ENV_SOURCED; unset -f defvar; '
                 'export BEVERIN_T6_EP= BEVERIN_TURBO_GG=custom NCCL_P2P_DISABLE=1; '
                 + setup + f'BASH_ENV="{hook}" bash "{wrapper}" bash "{probe}" "argument with spaces"',
                 check=False)
    assert value.returncode == exit_code, value.stdout + value.stderr
    assert value.stdout.strip() == 'probe-reached'


@pytest.mark.parametrize('historical', [False, True])
def test_reference_configuration_installs_no_automatic_hooks(tmp_path, monkeypatch, historical):
    from importlib import metadata
    import site

    app = ROOT / 'Alps-Images/apps/apertus-2/sources/primus'
    root = tmp_path / 'apertus2-primus'
    root.mkdir()
    shutil.copyfile(app / 'reference.env', root / 'reference.env')
    runtime = tmp_path / 'env/alps-runtime.env'
    runtime.parent.mkdir()
    runtime.write_text('# base runtime only\n')
    packages = tmp_path / 'site-packages'
    packages.mkdir()
    monkeypatch.setattr(site, 'getsitepackages', lambda: [str(packages)])
    versions = {'triton': '3.6.0' if historical else '3.8.0',
                'torch': '2.10.0+git94c6e04' if historical else '2.14.0+rocm10.1.0'}
    monkeypatch.setattr(metadata, 'version', versions.__getitem__)
    # Execute the real configuration step with all image paths redirected.
    step = (app / 'install.sh').read_text().split('    reference-config)', 1)[1]
    code = step.split("<<'PYTHON'\n", 1)[1].split('\nPYTHON', 1)[0]
    exec(compile(code.replace('/opt/alps', str(tmp_path)), 'reference-config', 'exec'), {})
    assert runtime.read_text() == '# base runtime only\n'
    assert not list(packages.iterdir())
    config = (root / 'reference.env').read_text()
    assert 'B2_AUTOTUNE' not in config
    assert ('defvar FLA_CONFIG_DIR' in config) == historical
    assert ('defvar HIPBLASLT_TUNING_OVERRIDE_FILE' in config) == historical


@pytest.mark.parametrize('suffix,version', [('263','26.3'),('268','26.8')])
def test_primus_app_uses_canonical_base_and_self_contained_tests(suffix, version):
    variant = f'megatron-primus{suffix}-rocm'
    base = bash(f'source ci-pipelines/helpers/meta.sh; rocm_base_refs primus {version}').stdout.split()[2]
    fields = bash(f'source ci-pipelines/helpers/meta.sh; app_refs apertus-2 {variant}').stdout.split()
    assert fields[0] == base
    assert fields[1] == 'Alps-Images/apps/apertus-2/Containerfile.megatron.primus'
    script = bash(f'manual-build/manual-build.sh app apertus-2 {variant}').stdout
    assert f'--build-arg BASE_IMAGE="{base}"' in script
    tests = yaml.safe_load((ROOT/'Alps-Images/apps/apertus-2/ci.yaml').read_text())['tests'][variant]
    assert {test['name'] for test in tests} == {'stack-smoke','train-resume','pipeline-resume'}
    assert all('/opt/tests/apertus-2/primus/' in test['script'][0] for test in tests)
    assert [(test['variables']['APERTUS_TEST_EP'],test['variables']['APERTUS_TEST_PP'])
            for test in tests if 'APERTUS_TEST_EP' in test['variables']] == [('8','1'),('4','2')]


def test_restore_observer_requires_native_checkpoint_ancestry():
    import ast
    import sys
    from types import SimpleNamespace
    path = ROOT/'Alps-Images/apps/apertus-2/tests/primus/train-entry.py'
    tree = ast.parse(path.read_text())
    observer = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'observe')
    namespace = dict(sys=sys, restore={'checkpoint': [], 'optimizer': [], 'rng': []})
    exec(compile(ast.Module(body=[observer], type_ignores=[]), str(path), 'exec'), namespace)
    def native_function(text, filename):
        scope = {}
        exec(compile(text, filename, 'exec'), scope)
        return scope
    optimizer = native_function('def load_state_dict_from_file(filename): return None',
        '/opt/megatron-lm/megatron/core/optimizer/layer_wise_optimizer.py')['load_state_dict_from_file']
    cpu = native_function('def set_rng_state(state): return None', '/opt/venv/torch/random.py')['set_rng_state']
    cuda = native_function('def set_rng_state(state): return None', '/opt/venv/torch/cuda/random.py')['set_rng_state']
    scope = dict(SimpleNamespace=SimpleNamespace, optimizer=optimizer, cpu=cpu, cuda=cuda)
    exec(compile('def load_checkpoint():\n'
        '    args=SimpleNamespace(no_load_optim=False,no_load_rng=False,finetune=False)\n'
        '    optimizer("sidecar.pt");cpu(None);cuda(None)\n'
        '    return (2, 123)\n', '/opt/megatron-lm/megatron/training/checkpointing.py', 'exec'), scope)
    previous = sys.getprofile()
    try:
        sys.setprofile(namespace['observe'])
        cpu(None);cuda(None);optimizer('not-a-restore.pt')
        assert namespace['restore'] == {'checkpoint': [], 'optimizer': [], 'rng': []}
        scope['load_checkpoint']()
        assert sys.getprofile() is None
    finally:
        sys.setprofile(previous)
    assert namespace['restore']['checkpoint'][0]['result'] == (2, 123)
    assert namespace['restore']['optimizer'] == ['sidecar.pt']
    assert namespace['restore']['rng'] == ['cpu', 'cuda']


@pytest.mark.parametrize('turbo,torch,modified', [('0.2.0+3cd482d','2.10.0+git94c6e04',False),
                                               ('0.5.1.dev7','2.14.0+rocm10.1.0',True)])
def test_reference_adapter_is_exact_and_version_gated(tmp_path, turbo, torch, modified):
    import importlib.util
    app = ROOT/'Alps-Images/apps/apertus-2'
    root = tmp_path/'install'
    root.mkdir()
    shutil.copytree(app/'sources/primus/vendor',root/'vendor')
    shutil.copytree(app/'patches/primus',root/'patches')
    source=tmp_path/'megatron'
    source_installer().install(root/'vendor',source)
    path=app/'sources/primus/prepare-source.py'
    spec=importlib.util.spec_from_file_location('prepare_primus',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.prepare(root,source,turbo,torch)
    identity=json.loads((root/'source-identity.json').read_text())
    original=json.loads((root/'vendor/manifest.json').read_text())
    changed={name for name,sha in identity['files'].items() if sha != original['files'][name]}
    assert changed == (set(json.loads((root/'patches/primus268-source-hashes.json').read_text())) if modified else set())
    assert len(identity['applied_patches']) == int(modified)
    with pytest.raises(RuntimeError,match='Unaudited'):
        module.prepare(root,source,'0.6.0',torch)


def test_primus_app_hash_covers_archive_gate_patch_and_tests(tmp_path):
    for name in ('Alps-Images','ci-pipelines'):
        shutil.copytree(ROOT/name,tmp_path/name,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copyfile(ROOT/'.dockerignore',tmp_path/'.dockerignore')
    command='source ci-pipelines/helpers/meta.sh; app_refs apertus-2 megatron-primus268-rocm'
    before=bash(command,cwd=tmp_path).stdout.split()
    paths=('sources/primus/vendor/megatron-source.tar.gz','sources/primus/kda-gate.py',
           'patches/primus/0001-turbo051-private-api-imports.patch','tests/primus/check-stack.py')
    for name in paths:
        path=tmp_path/'Alps-Images/apps/apertus-2'/name
        original=path.read_bytes()
        try:
            path.write_bytes(original+b'\n# content-invalidation-probe\n')
            changed=bash(command,cwd=tmp_path).stdout.split()
            assert changed[0]==before[0], 'App-only changes must not alter canonical base'
            assert changed[2]!=before[2], name
        finally:
            path.write_bytes(original)


@pytest.mark.parametrize('broken', [None,'nan-loss','inf-norm','skipped-update','nan-update','missing-rank','wrong-restore-owner','missing-rng','wrong-source','missing-banner','wrong-banner-iteration','metric-rank-only-banner'])
@pytest.mark.parametrize('check_rank', ['all','7'])
def test_checkpoint_acceptance_rejects_incomplete_or_false_evidence(tmp_path, monkeypatch, broken, check_rank):
    import sys
    from types import ModuleType, SimpleNamespace
    ranks=tmp_path/'ranks';ranks.mkdir()
    checkpoints=tmp_path/'checkpoints';checkpoints.mkdir()
    manifest=tmp_path/'source-identity.json'
    manifest.write_text(json.dumps({'files':{'source.py':'source-sha'}}))
    for stage,iteration in enumerate((2,4)):
        for rank in range(8):
            model=checkpoints/f'iter_{iteration:07d}'/f'mp_rank_{rank}'/'model_optim_rng.pt'
            model.parent.mkdir(parents=True)
            model.write_text('model-state')
            optimizer=model.with_name(f'layer_wise_optimizer_{rank}.pt')
            optimizer.write_text('optimizer-state')
            loaded=(checkpoints/'iter_0000002'/f'mp_rank_{rank}'/optimizer.name)
            restore={'checkpoint':[{'result':[2,0],'no_load_optim':False,'no_load_rng':False,'finetune':False}],
                     'optimizer':[str(loaded)],'rng':['cpu','cuda']}
            record=dict(rank=rank,world=8,iteration=iteration,dp_rank=rank,
                        model_checkpoint=str(model),optimizer_checkpoint=str(optimizer),
                        loaded_megatron={'source.py':'source-sha'},restore=restore)
            if rank==7 and stage==1:
                if broken=='wrong-restore-owner': restore['optimizer']=[str(loaded.with_name('layer_wise_optimizer_0.pt'))]
                if broken=='missing-rng': restore['rng']=['cpu']
                if broken=='wrong-source': record['loaded_megatron']['source.py']='different-sha'
            (ranks/f'{rank}-stage{stage}.json').write_text(json.dumps(record))
            text=''
            if rank==7:
                for update in ([1,2] if stage==0 else [3,4]):
                    loss='NaN' if broken=='nan-loss' and stage==1 else '1.25'
                    norm='inf' if broken=='inf-norm' and stage==1 else '1.2'
                    skipped=int(broken=='skipped-update' and stage==1)
                    nan=int(broken=='nan-update' and stage==1)
                    text+=f'iteration {update} / 4 | lm loss: {loss} | grad norm: {norm} | number of skipped iterations: {skipped} | number of nan iterations: {nan}\n'
            if stage and rank==0 and broken not in ('missing-banner','metric-rank-only-banner'):
                loaded_iteration=4 if broken=='wrong-banner-iteration' else 2
                text+=f'successfully loaded checkpoint from checkpoints at iteration {loaded_iteration}\n'
            if stage and rank==7 and broken=='metric-rank-only-banner':
                text+='successfully loaded checkpoint from checkpoints at iteration 2\n'
            (ranks/f'{rank}-stage{stage}.log').write_text(text)
    (checkpoints/'latest_checkpointed_iteration.txt').write_text('4\n')
    if broken=='missing-rank': (ranks/'7-stage1.json').unlink()
    def load(path, **kwargs):
        iteration=2 if 'iter_0000002' in str(path) else 4
        return {'iteration':iteration,'model':{},'optimizer':{},'opt_param_scheduler':{},'rng_state':{},
                'args':SimpleNamespace(consumed_train_samples=iteration*8)}
    torch=ModuleType('torch');torch.load=load
    monkeypatch.setitem(sys.modules,'torch',torch)
    monkeypatch.setenv('WORLD_SIZE','8')
    monkeypatch.setattr(sys,'argv',['check-checkpoint.py',str(tmp_path),check_rank,'all'])
    path=ROOT/'Alps-Images/apps/apertus-2/tests/primus/check-checkpoint.py'
    code=path.read_text().replace('/opt/alps/apertus2-primus/source-identity.json',str(manifest))
    if broken is None:
        exec(compile(code,str(path),'exec'),{})
    else:
        with pytest.raises((AssertionError,FileNotFoundError)):
            exec(compile(code,str(path),'exec'),{})


def test_primus_system_aql_policy_is_late_and_not_shared_network_policy():
    text=(ROOT/'Alps-Images/ROCm/Containerfile.primus-alps').read_text()
    policy=text.index('ENV DEBUG_CLR_AQL_DEV_QUEUE=0')
    assert text.index('--compare /opt/alps/primus-compute-before.json') < policy
    assert policy < text.index('ARG OCI_SOURCE=')
    assert 'DEBUG_CLR_AQL_DEV_QUEUE' not in (ROOT/'Alps-Images/common/alps-runtime.common.env').read_text()
    assert 'DEBUG_CLR_AQL_DEV_QUEUE' not in (ROOT/'Alps-Images/ROCm/Containerfile.rocm-alps').read_text()
    assert 'GPU_USE_DEVICE_QUEUE=' not in text



def test_primus_build_stages_keep_component_cache_boundaries():
    recipe=(ROOT/'Alps-Images/ROCm/Containerfile.primus-alps').read_text()
    assert recipe.index('sources/purge-network.sh') < recipe.index('install-alps-rocm-stack.sh boost')
    assert recipe.index('install-alps-rocm-stack.sh ucc') < recipe.index('sources/mpi*')
    assert recipe.index('install-alps-rocm-stack.sh aws-ofi') < recipe.index('install-alps-rocm-stack.sh tests')
    for version in ('26.3', '26.8'):
        for helper in ('purge-network.sh', 'mpi-install.sh'):
            assert (ROOT/f'Alps-Images/ROCm/primus-{version}/sources/{helper}').is_file()
    cleanup=(ROOT/'Alps-Images/ROCm/primus-26.3/sources/purge-network.sh').read_text()
    assert cleanup.index('apt_mark manual') < cleanup.index('purge_preinstalled_network_stack\n')
    assert '--compare /opt/alps/primus-compute-before.json' in cleanup


@pytest.mark.parametrize('helper', ['mpi-install.sh','mpi4-install.sh','purge-network.sh'])
@pytest.mark.parametrize('variant', ['', 'primus-26.8','pytorch-rocm10.0-ubuntu24.04-py3.12-torch2.13'])
def test_private_mpi4_helpers_reject_other_images_before_any_operation(helper, variant):
    value=bash(f'ROCM_VARIANT_DIR="{variant}" bash Alps-Images/ROCm/primus-26.3/sources/{helper}',check=False)
    assert value.returncode != 0
    assert 'restricted to primus-26.3' in value.stderr
    assert '/opt/alps' not in value.stderr


def test_only_primus263_selects_mpi4_and_common_default_remains_mpi5():
    selectors=yaml.safe_load((ROOT/'Alps-Images/ROCm/ci.yaml').read_text())['variants']
    for selector in selectors:
        name,version=selector['name'],selector['variant']
        recipe=bash(f'source ci-pipelines/helpers/meta.sh; rocm_base_refs {name} {version}').stdout.split()[1]
        text=(ROOT/recipe).read_text()
        if name=='primus':
            assert 'sources/mpi*' in text
            installer=(ROOT/f'Alps-Images/ROCm/primus-{version}/sources/mpi-install.sh').read_text()
            if version=='26.3':
                assert 'mpi4-install.sh' in installer
                assert 'install-alps-rocm-stack.sh ompi' not in installer
            else:
                assert 'install-alps-rocm-stack.sh ompi' in installer
        else:
            assert 'primus263' not in text
            assert '/opt/alps/install-alps-rocm-stack.sh\n' in text
    stack=(ROOT/'Alps-Images/ROCm/install-alps-rocm-stack.sh').read_text()
    assert re.search(r'ompi\)\s+build_ompi',stack)
    default=bash('unset OMPI_VER; source Alps-Images/common/alps-stack-versions.env; printf "%s" "$OMPI_VER"').stdout
    assert default.startswith('5.')
    common=(ROOT/'Alps-Images/common/install-alps-hpc-stack.sh').read_text()
    assert 'release/open-mpi/v5.0/' in common
    assert 'PRIMUS263' not in common


def test_primus_inputs_follow_existing_hash_scopes(tmp_path):
    for name in ('Alps-Images','ci-pipelines'):
        shutil.copytree(ROOT/name,tmp_path/name,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copyfile(ROOT/'.dockerignore',tmp_path/'.dockerignore')
    commands=[f'rocm_base_refs {v["name"]} {v["variant"]}' for v in
              yaml.safe_load((ROOT/'Alps-Images/ROCm/ci.yaml').read_text())['variants']]
    commands += ['app_refs apertus-2 '+v for v in ('megatron-primus263-rocm','megatron-primus268-rocm','megatron-rocm')]
    def refs():
        return {cmd:bash('source ci-pipelines/helpers/meta.sh; '+cmd,cwd=tmp_path).stdout.split()[2] for cmd in commands}
    before=refs()
    inputs = [(path, {'rocm_base_refs primus 26.3', 'app_refs apertus-2 megatron-primus263-rocm'})
              for path in (tmp_path/'Alps-Images/ROCm/primus-26.3/sources').rglob('*')]
    # Base hashes deliberately cover the complete shared patch tree.
    inputs += [(path, set(commands)) for path in (tmp_path/'Alps-Images/patches/openmpi').glob('*.patch')]
    inputs.append((tmp_path/'Alps-Images/ROCm/sources/primus/check-compute.py',
                   {'rocm_base_refs primus 26.3', 'rocm_base_refs primus 26.8',
                    'app_refs apertus-2 megatron-primus263-rocm', 'app_refs apertus-2 megatron-primus268-rocm'}))
    for path, expected in inputs:
        if not path.is_file():
            continue
        original=path.read_text()
        try:
            path.write_text(original+'\n# private MPI4 invalidation probe\n')
            after=refs()
            changed={cmd for cmd in commands if before[cmd]!=after[cmd]}
            assert changed == expected, path
        finally:
            path.write_text(original)


def test_rccl_test_hip_options_survive_cmake_deduplication(tmp_path):
    import shlex
    if not shutil.which('cmake') or not shutil.which('c++'):
        pytest.skip('CMake and a C++ compiler are needed to generate real compile commands')
    source=tmp_path/'projects/rccl-tests/src';source.mkdir(parents=True)
    (source/'common.cpp').write_text('int common() { return 0; }\n')
    (source/'perf.cpp').write_text('int main() { return 0; }\n')
    (source/'CMakeLists.txt').write_text(
        'cmake_minimum_required(VERSION 3.16)\nproject(rccl_flag_regression LANGUAGES CXX)\n'
        # Reproduce upstream HIP's existing language selection and duplicated options.
        'add_compile_options(-x hip)\nset(HIP_COMMON_SOURCES common.cpp)\n'
        'add_library(rccl_common OBJECT ${HIP_COMMON_SOURCES})\n'
        'add_executable(all_reduce_perf perf.cpp)\n')
    # Keep this regression entirely within pytest temporary storage.
    installer=(ROOT/'Alps-Images/ROCm/install-alps-rocm-components.sh').read_text()
    installer=installer.replace('/tmp/rccl-tests-force-include.h',str(tmp_path/'force-include.h'))
    helper=tmp_path/'components.sh';helper.write_text(installer)
    bash(f'source {helper}; ROCM_VERSION=10.1.0; RCCL_TESTS_GPU_TARGETS=gfx942; '
         f'ROCM_SYSTEMS_SRC_DIR={tmp_path}; patch_rocm_rccl_test_sources')
    subprocess.run(['cmake','-S',str(source),'-B',str(tmp_path/'build'),
                    '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'],check=True,capture_output=True,text=True)
    commands=json.loads((tmp_path/'build/compile_commands.json').read_text())
    assert len(commands)==2
    for item in commands:
        args=shlex.split(item['command'])
        assert '-xhip' in args and '--offload-arch=gfx942' in args
        assert all(args[i+1]=='hip' for i,arg in enumerate(args) if arg=='-x'),args


def test_private_mpi4_ofi_patch_applies_and_preserves_pointer_conversion(tmp_path):
    if not shutil.which('patch') or not shutil.which('cc'):
        pytest.skip('patch and a C compiler are needed')
    header=tmp_path/'ompi/mca/mtl/ofi/mtl_ofi_request.h'
    header.parent.mkdir(parents=True)
    header.write_text('#ifndef OMPI_MTL_OFI_REQUEST_H\n#define OMPI_MTL_OFI_REQUEST_H\n\n'
        '#include "mtl_ofi.h"\n\n#define TO_OFI_REQ(_ptr_ctx) \\\n'
        '    container_of((_ptr_ctx), struct ompi_mtl_ofi_request_t, ctx)\n\n'
        'typedef enum {\n    OMPI_MTL_OFI_SEND,\n    OMPI_MTL_OFI_RECV\n} kind;\n'
        'struct ompi_mtl_ofi_request_t { int leading; int ctx; };\n#endif\n')
    (header.parent/'mtl_ofi.h').write_text('/* libfabric no longer supplies container_of */\n')
    patch=ROOT/'Alps-Images/patches/openmpi/mpi4-ofi-container.patch'
    subprocess.run(['patch','--batch','--fuzz=0','-p1','-d',str(tmp_path),'-i',str(patch)],check=True,capture_output=True,text=True)
    source=tmp_path/'check.c'
    source.write_text('#include <assert.h>\n#include "ompi/mca/mtl/ofi/mtl_ofi_request.h"\n'
        'static int calls; static int *once(int *p) { ++calls; return p; }\n'
        'int main(void) { struct ompi_mtl_ofi_request_t r = {1,2};\n'
        'assert(TO_OFI_REQ(once(&r.ctx)) == &r); assert(calls == 1); return 0; }\n')
    subprocess.run(['cc','-std=c11','-Wall','-Wextra','-Werror',str(source),'-o',str(tmp_path/'check')],check=True,capture_output=True,text=True)
    subprocess.run([str(tmp_path/'check')],check=True)


def test_private_rccl_smi_relink_changes_only_final_link_and_rejects_other_images():
    import importlib.util
    path=ROOT/'Alps-Images/ROCm/primus-26.3/sources/mpi4-relink-rccl-smi.py'
    spec=importlib.util.spec_from_file_location('private_rccl_smi',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    original=': && /opt/rocm/bin/amdclang++ -shared -o librccl.so.1.0 objects.o -lhip && :'
    result=module.add_smi_dependency(original,Path('/opt/rocm sdk'))
    assert result.replace(" -Wl,--no-as-needed -L'/opt/rocm sdk/lib' -lrocm_smi64 -Wl,--as-needed",'')==original
    for broken in ('clang++ -o main object.o',original.replace('librccl.so.1.0','libother.so')):
        with pytest.raises(RuntimeError,match='Unexpected pinned'):
            module.add_smi_dependency(broken,Path('/opt/rocm'))
    for variant in ('','primus-26.8','pytorch-rocm10.0-ubuntu24.04-py3.12-torch2.13'):
        value=bash(f'ROCM_VARIANT_DIR="{variant}" python3 {path}',check=False)
        assert value.returncode!=0 and 'restricted to primus-26.3' in value.stderr


def attention_accuracy_policy():
    import runpy
    return runpy.run_path(str(ROOT/'Alps-Images/apps/apertus-2/tests/primus/attention-accuracy.py'))


def test_attention_policy_is_release_pinned_and_adds_global_error_guard():
    policy=attention_accuracy_policy()
    assert policy['UPSTREAM_REVISION']=='b4d9154d125e09efbe098d986e40fea3549c1244'
    result=policy['validate_gradient_error'](0.009,0.002,0.003,0.001)
    assert result['max_abs_bound']==0.02
    assert result['relative_l2_bound']==2**-7


@pytest.mark.parametrize('metrics,message', [
    ((float('nan'),0.002,0.003,0.001),'Nonfinite'),
    ((0.009,0.002,float('inf'),0.001),'Nonfinite'),
    ((-0.1,0.002,0.003,0.001),'negative'),
    ((0.009,0.002,0.02,0.001),'AITER gradient exceeds precision'),
    ((0.009,0.002,0.003,0.02),'math control exceeds precision'),
    ((0.1,0.002,0.003,0.001),'upstream BF16-control'),
])
def test_attention_policy_rejects_nonfinite_widespread_and_outlier_errors(metrics,message):
    with pytest.raises(AssertionError,match=message):
        attention_accuracy_policy()['validate_gradient_error'](*metrics)


def test_private_mpi4_hmem_patch_requires_implicit_device_registration():
    sources=ROOT/'Alps-Images/ROCm/primus-26.3/sources'
    patch=(ROOT/'Alps-Images/patches/openmpi/mpi4-ofi-hmem.patch').read_text()
    additions='\n'.join(line[1:] for line in patch.splitlines() if line.startswith('+') and not line.startswith('+++'))
    assert 'FI_DIRECTED_RECV | FI_HMEM' in additions
    assert 'fi_version = FI_VERSION(1, 18);' in additions
    assert 'hints->domain_attr->mr_mode = FI_MR_ALLOCATED;' in additions
    # MPI4 has no device MR callbacks; never claim explicit-MR support.
    assert not re.search(r'^[^/]*mr_mode.*FI_MR_HMEM',additions,flags=re.M)
    installer=(sources/'mpi4-install.sh').read_text()
    assert '--fuzz=0' in installer and '-i /opt/alps/patches/openmpi/mpi4-ofi-hmem.patch' in installer


def test_primus_training_launcher_clears_only_conflicting_vendor_attention_flags(tmp_path):
    probe=tmp_path/'probe.sh'
    probe.write_text('test -z "${NVTE_FLASH_ATTN+x}${NVTE_FUSED_ATTN+x}${NVTE_UNFUSED_ATTN+x}"\n'
                     'test "$NCCL_P2P_DISABLE" = 1\n')
    tests=yaml.safe_load((ROOT/'Alps-Images/apps/apertus-2/ci.yaml').read_text())['tests']
    for variant in ('megatron-primus263-rocm','megatron-primus268-rocm'):
        for test in tests[variant]:
            if test['name']=='stack-smoke':continue
            command=test['script'][0]
            prefix='bash /opt/tests/apertus-2/primus/with-reference-env.sh '
            assert command.startswith(prefix)
            command=command.removeprefix(prefix)
            assert 'env -u NVTE_FLASH_ATTN -u NVTE_FUSED_ATTN -u NVTE_UNFUSED_ATTN bash ' in command
            command=command.replace('/opt/tests/apertus-2/primus/train-resume.sh',str(probe))
            env=dict(os.environ,NVTE_FLASH_ATTN='0',NVTE_FUSED_ATTN='1',NVTE_UNFUSED_ATTN='0',NCCL_P2P_DISABLE='1')
            result=subprocess.run(['bash','-c',command],env=env,text=True,capture_output=True)
            assert result.returncode==0,result.stdout+result.stderr
