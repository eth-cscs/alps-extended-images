import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]


def run_bash(script, *, env=None, check=True, cwd=ROOT):
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    result = subprocess.run(
        ["bash", "-c", script],
        cwd=cwd,
        env=merged_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and result.returncode != 0:
        raise AssertionError(f"command failed: {script}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    return result


def fake_skopeo_env(tmp_path, digests):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    copy_log = tmp_path / "copy.log"
    login_log = tmp_path / "login.log"
    skopeo = bin_dir / "skopeo"
    skopeo.write_text(
        r"""#!/usr/bin/env python3
import json
import os
import sys

args = sys.argv[1:]
digests = json.loads(os.environ.get('FAKE_SKOPEO_DIGESTS', '{}'))
if args[0] == 'login':
    with open(os.environ['FAKE_SKOPEO_LOGIN_LOG'], 'a', encoding='utf-8') as handle:
        handle.write('ARGS=' + ' '.join(args[1:]) + '\n')
        handle.write('STDIN=' + sys.stdin.read() + '\n')
    raise SystemExit(0)
if args[0] == 'inspect':
    ref = args[-1]
    if ref.startswith('docker://'):
        ref = ref[len('docker://'):]
    digest = digests.get(ref, '')
    if digest == '__ERROR__':
        print('unauthorized: authentication required', file=sys.stderr)
        raise SystemExit(7)
    if digest == '__MISSING_FATAL__':
        print('time="2026-08-20T08:09:50Z" level=fatal msg="Error parsing image name \\\"docker://' + ref + '\\\": reading manifest tag in repo: manifest unknown: The named manifest is not known to the registry."', file=sys.stderr)
        raise SystemExit(2)
    if digest:
        print(digest)
        raise SystemExit(0)
    print('manifest unknown', file=sys.stderr)
    raise SystemExit(1)
if args[0] == 'copy':
    with open(os.environ['FAKE_SKOPEO_COPY_LOG'], 'a', encoding='utf-8') as handle:
        handle.write(' '.join(args[1:]) + '\n')
    raise SystemExit(0)
print('unsupported skopeo call: ' + ' '.join(args), file=sys.stderr)
raise SystemExit(2)
"""
    )
    skopeo.chmod(0o755)
    return {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "FAKE_SKOPEO_DIGESTS": json.dumps(digests),
        "FAKE_SKOPEO_COPY_LOG": str(copy_log),
        "FAKE_SKOPEO_LOGIN_LOG": str(login_log),
    }, copy_log


def source_skopeo(command):
    return f"source ci-pipelines/helpers/skopeo.sh; {command}"


def meta_env(**extra):
    env = {
        "IMAGE_PREFIX": "localhost/alps-images",
        "GHCR_IMAGE_PREFIX": "localhost/ghcr",
        "ALPS_REV": "alps7-dev",
        "CI_COMMIT_SHORT_SHA": "abc123",
        "CI_COMMIT_SHA": "abc123",
        "CSCS_CI_ORIG_CLONE_URL": "https://example.invalid/repo.git",
    }
    env.update(extra)
    return env


def test_skopeo_img_digest_returns_digest_for_existing_ref(tmp_path):
    env, _ = fake_skopeo_env(tmp_path, {"registry/image:tag": "sha256:123"})
    result = run_bash(source_skopeo("img_digest registry/image:tag"), env=env)
    assert result.stdout.strip() == "sha256:123"


def test_skopeo_img_digest_returns_empty_for_missing_ref(tmp_path):
    env, _ = fake_skopeo_env(tmp_path, {})
    result = run_bash(source_skopeo("img_digest registry/missing:tag"), env=env)
    assert result.stdout == ""


def test_skopeo_img_digest_returns_empty_for_jfrog_missing_manifest(tmp_path):
    env, _ = fake_skopeo_env(tmp_path, {"registry/missing:tag": "__MISSING_FATAL__"})
    result = run_bash(source_skopeo("img_digest registry/missing:tag"), env=env)
    assert result.stdout == ""


def test_skopeo_img_digest_fails_closed_on_unexpected_error(tmp_path):
    env, _ = fake_skopeo_env(tmp_path, {"registry/private:tag": "__ERROR__"})
    result = run_bash(source_skopeo("img_digest registry/private:tag"), env=env, check=False)
    assert result.returncode != 0
    assert "failed to inspect image" in result.stderr


def test_skopeo_login_passes_password_on_stdin(tmp_path):
    env, _ = fake_skopeo_env(tmp_path, {})
    env.update({"IMAGE_PREFIX": "registry.example/ns", "JFROG_USER": "user", "JFROG_KEY": "secret-token"})

    run_bash(source_skopeo("skopeo_login"), env=env)

    login_log = Path(env["FAKE_SKOPEO_LOGIN_LOG"]).read_text()
    assert "--password-stdin" in login_log
    assert "--password secret-token" not in login_log
    assert "STDIN=secret-token" in login_log


@pytest.mark.parametrize(
    ("digest", "expected_status", "build_expected"),
    [("", 0, True), ("sha256:existing", 0, False), ("__ERROR__", 7, False)],
)
def test_build_preflight_preserves_existing_images_and_fails_closed(tmp_path, digest, expected_status, build_expected):
    canon = "registry/image:canon"
    env, _ = fake_skopeo_env(tmp_path, {canon: digest} if digest else {})
    build_output = tmp_path / "build-output"
    env.update({
        "IMAGE_PREFIX": "registry/images",
        "JFROG_USER": "user",
        "JFROG_KEY": "test-token",
        "CANON_IMAGE_REF": canon,
        "BUILD_OUTPUT": str(build_output),
    })
    templates = yaml.safe_load((ROOT / "ci-pipelines/child-templates.yaml").read_text())
    preflight = "\n".join(templates[".child-build-images-template"]["before_script"])
    result = run_bash(
        preflight + '\ntouch "$BUILD_OUTPUT"',
        env=env,
        check=False,
    )
    assert result.returncode == expected_status
    assert build_output.exists() is build_expected


def test_skopeo_require_tested_marker_valid_requires_marker_exists(tmp_path):
    env, _ = fake_skopeo_env(tmp_path, {"registry/image:canon": "sha256:123"})
    result = run_bash(
        source_skopeo("require_tested_marker_valid registry/image:canon registry/image:canon-tested-hash"),
        env=env,
        check=False,
    )
    assert result.returncode != 0
    assert "tested marker missing" in result.stderr


def test_skopeo_require_tested_marker_valid_rejects_mismatch(tmp_path):
    env, _ = fake_skopeo_env(tmp_path, {"registry/image:canon": "sha256:123", "registry/image:canon-tested-hash": "sha256:old"})
    result = run_bash(
        source_skopeo("require_tested_marker_valid registry/image:canon registry/image:canon-tested-hash"),
        env=env,
        check=False,
    )
    assert result.returncode != 0
    assert "different digest" in result.stderr


def test_skopeo_mark_tested_creates_missing_marker(tmp_path):
    env, copy_log = fake_skopeo_env(tmp_path, {"registry/image:canon": "sha256:123"})
    run_bash(source_skopeo("mark_tested registry/image:canon registry/image:canon-tested-hash"), env=env)
    assert copy_log.read_text().strip() == "docker://registry/image:canon docker://registry/image:canon-tested-hash"


def test_skopeo_ref_url_preserves_existing_prefix():
    result = run_bash(source_skopeo("_ref_url docker://registry/image:tag"))
    assert result.stdout.strip() == "docker://registry/image:tag"


def test_skopeo_ref_url_adds_docker_prefix():
    result = run_bash(source_skopeo("_ref_url registry/image:tag"))
    assert result.stdout.strip() == "docker://registry/image:tag"


def test_skopeo_tested_ref_for_accepts_registry_port():
    result = run_bash(source_skopeo("tested_ref_for registry:5000/ns/image:tag hash"))
    assert result.stdout.strip() == "registry:5000/ns/image:tag-tested-hash"


def test_skopeo_tested_ref_for_rejects_untagged_registry_port():
    result = run_bash(source_skopeo("tested_ref_for registry:5000/ns/image hash"), check=False)
    assert result.returncode != 0
    assert "must include a tag" in result.stderr


def test_skopeo_mark_tested_noops_matching_marker(tmp_path):
    env, copy_log = fake_skopeo_env(tmp_path, {"registry/image:canon": "sha256:123", "registry/image:canon-tested-hash": "sha256:123"})
    result = run_bash(source_skopeo("mark_tested registry/image:canon registry/image:canon-tested-hash"), env=env)
    assert "No-op" in result.stdout
    assert not copy_log.exists()


def test_skopeo_mark_tested_rejects_mismatch(tmp_path):
    env, _ = fake_skopeo_env(tmp_path, {"registry/image:canon": "sha256:123", "registry/image:canon-tested-hash": "sha256:old"})
    result = run_bash(source_skopeo("mark_tested registry/image:canon registry/image:canon-tested-hash"), env=env, check=False)
    assert result.returncode != 0
    assert "different digest" in result.stderr


def test_skopeo_promote_strict_copies_missing_stable(tmp_path):
    env, copy_log = fake_skopeo_env(tmp_path, {"registry/image:canon": "sha256:123"})
    run_bash(source_skopeo("promote_strict registry/image:canon registry/image:stable"), env=env)
    assert copy_log.read_text().strip() == "docker://registry/image:canon docker://registry/image:stable"


def test_skopeo_promote_strict_noops_matching_stable(tmp_path):
    env, copy_log = fake_skopeo_env(tmp_path, {"registry/image:canon": "sha256:123", "registry/image:stable": "sha256:123"})
    result = run_bash(source_skopeo("promote_strict registry/image:canon registry/image:stable"), env=env)
    assert "no-op" in result.stdout
    assert not copy_log.exists()


def test_skopeo_promote_strict_rejects_different_stable(tmp_path):
    env, _ = fake_skopeo_env(tmp_path, {"registry/image:canon": "sha256:123", "registry/image:stable": "sha256:old"})
    result = run_bash(source_skopeo("promote_strict registry/image:canon registry/image:stable"), env=env, check=False)
    assert result.returncode != 0
    assert "refusing to overwrite" in result.stderr


def run_meta(command, *, check=True, cwd=ROOT, **extra_env):
    return run_bash(
        f"source ci-pipelines/helpers/skopeo.sh; source ci-pipelines/helpers/meta.sh; {command}",
        env=meta_env(**extra_env), check=check, cwd=cwd,
    )


def test_meta_base_refs_return_expected_fields():
    ngc_fields = run_meta("ngc_base_refs pytorch 25.12-py3").stdout.split()
    rocm_fields = run_meta("rocm_base_refs pytorch rocm7.14-ubuntu24.04-py3.12-torch2.11").stdout.split()
    assert len(ngc_fields) == 5
    assert len(rocm_fields) == 4


@pytest.mark.parametrize("helper", ["rocm_base_refs pytorch", "write_base_build_env"])
def test_rocm_base_helpers_preserve_optional_library_variable_scope(tmp_path, helper):
    variant = "rocm7.14-ubuntu24.04-py3.12-torch2.11"
    output = tmp_path / "base.env"
    command = (
        f"rocm_base_refs pytorch {variant}"
        if helper == "rocm_base_refs pytorch"
        else f"write_base_build_env {output} rocm pytorch {variant}"
    )
    result = run_meta(
        f'{command} && printf "\\nLIBRARY_SCOPE=%s,%s\\n" "$ROCM_LIBRARIES_REPO" "$ROCM_LIBRARIES_COMMIT"',
        ROCM_LIBRARIES_REPO="outer-repo",
        ROCM_LIBRARIES_COMMIT="outer-commit",
    )

    assert result.stdout.splitlines()[-1] == "LIBRARY_SCOPE=outer-repo,outer-commit"
    if output.exists():
        data = dict(line.split("=", 1) for line in output.read_text().splitlines())
        profile = f"Alps-Images/ROCm/pytorch-{variant}/profile.env"
        for key in ("ROCM_LIBRARIES_REPO", "ROCM_LIBRARIES_COMMIT"):
            assert data[key] == run_meta(f"profile_value {profile} {key}").stdout.strip()


@pytest.mark.parametrize(
    "family,helper,canonical_index",
    [("cuda", "ngc_base_refs", 3), ("rocm", "rocm_base_refs", 2)],
)
def test_meta_app_refs_use_canonical_base_ref(family, helper, canonical_index):
    base = run_meta(f"profile_value Alps-Images/apps/vllm/profile.env {family.upper()}_BASE_IMAGE").stdout.strip()
    image, variant = base.split(":", 1)
    name = image.removesuffix(f"-{family}")
    canonical = run_meta(f"{helper} {name} {variant}").stdout.split()[canonical_index]
    fields = run_meta(f"app_refs vllm {family}").stdout.split()
    assert len(fields) == 5
    assert fields[0] == canonical
    assert fields[4] == family


def test_meta_parse_base_image_rejects_unsupported_format():
    result = run_meta("parse_base_image pytorch:plain", check=False)
    assert result.returncode != 0
    assert "expected BASE_IMAGE" in result.stderr


def test_meta_validate_rocm_profile_rejects_invalid_rebuild_flag(tmp_path):
    profile = tmp_path / "profile.env"
    profile.write_text(
        "\n".join(
            [
                'ROCM_PYPI_INDEX_URL="https://example.invalid"',
                'ROCM_REBUILD_RCCL="maybe"',
                'ROCM_SYSTEMS_REPO="https://example.invalid/repo.git"',
                'ROCM_SYSTEMS_COMMIT="abc123"',
                'RCCL_GPU_TARGETS="gfx942"',
                'RCCL_TESTS_GPU_TARGETS="gfx942"',
            ]
        )
    )
    result = run_meta(f"load_rocm_profile {profile}", check=False)
    assert result.returncode != 0
    assert "ROCM_REBUILD_RCCL must be 0 or 1" in result.stderr


def test_meta_profile_value_accepts_quoted_value_with_inline_comment(tmp_path):
    profile = tmp_path / "profile.env"
    profile.write_text('APP_VARIANTS="cuda rocm" # supported variants\n')

    result = run_meta(f"profile_value {profile} APP_VARIANTS")

    assert result.stdout.strip() == "cuda rocm"


def test_meta_refs_are_deterministic():
    first = run_meta("app_refs vllm cuda").stdout
    second = run_meta("app_refs vllm cuda").stdout
    assert first == second


def test_meta_commit_sha_does_not_change_canonical_refs():
    first = run_meta("ngc_base_refs pytorch 25.12-py3", CI_COMMIT_SHORT_SHA="one", CI_COMMIT_SHA="one").stdout
    second = run_meta("ngc_base_refs pytorch 25.12-py3", CI_COMMIT_SHORT_SHA="two", CI_COMMIT_SHA="two").stdout
    assert first == second


@pytest.fixture()
def isolated_meta_repo(tmp_path):
    import shutil

    shutil.copytree(ROOT / "Alps-Images", tmp_path / "Alps-Images", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "ci-pipelines/helpers", tmp_path / "ci-pipelines/helpers", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(ROOT / "ci-pipelines/child-templates.yaml", tmp_path / "ci-pipelines")
    shutil.copy2(ROOT / ".dockerignore", tmp_path)

    def metadata(command):
        result = subprocess.run(
            ["bash", "-c", "source ci-pipelines/helpers/skopeo.sh; source ci-pipelines/helpers/meta.sh; " + command],
            cwd=tmp_path,
            env={**os.environ, **meta_env()},
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return result.stdout

    return tmp_path, metadata


@pytest.mark.parametrize(
    ("relative_path", "affected"),
    [
        ("patches/rocm/aiter/0001-mi300a-xcd-count.patch", {"rocm"}),
        ("patches/rocm/uccl/0001-rocm-use-rocr-libfabric-hmem.patch", {"rocm"}),
        ("patches/rocm/vllm/0001-rocm10-include-cstdint.patch", {"rocm"}),
        ("patches/cuda/uccl/0001-cuda-allow-explicit-grace-hopper-target.patch", {"cuda"}),
        ("sources/install-aiter.sh", {"rocm"}),
        ("sources/build-rocm-env.sh", {"rocm"}),
        ("sources/build-cuda-env.sh", {"cuda"}),
        ("sources/install-mooncake.sh", {"cuda", "rocm"}),
    ],
)
def test_app_inputs_invalidate_only_consuming_accelerators(isolated_meta_repo, relative_path, affected):
    root, metadata = isolated_meta_repo

    def hashes():
        lines = metadata('for variant in cuda rocm; do app_refs vllm "$variant"; app_validation_hash vllm "$variant"; done').splitlines()
        return {variant: (lines[index].split()[2], lines[index + 1]) for variant, index in (("cuda", 0), ("rocm", 2))}

    before = hashes()
    path = root / "Alps-Images/apps/vllm" / relative_path
    path.write_text(path.read_text() + "\n")
    after = hashes()
    for variant in ("cuda", "rocm"):
        if variant in affected:
            assert before[variant][0] != after[variant][0]
            if relative_path.startswith("patches/"):
                assert before[variant][1] != after[variant][1]
        else:
            assert before[variant] == after[variant]


def test_app_profile_changes_do_not_invalidate_other_accelerators(isolated_meta_repo):
    root, metadata = isolated_meta_repo
    before = {variant: metadata(f"app_refs vllm {variant}") for variant in ("cuda", "rocm")}
    profile = root / "Alps-Images/apps/vllm/profile.env"
    profile.write_text(profile.read_text().replace("CUDA_BASE_IMAGE=pytorch-cuda:26.07-py3", "CUDA_BASE_IMAGE=pytorch-cuda:26.06-py3"))
    assert metadata("app_refs vllm cuda") != before["cuda"]
    assert metadata("app_refs vllm rocm") == before["rocm"]


@pytest.mark.parametrize("patch_rel", ["nccl/net.cc.2.29.7-1.patch", "new-component/nested/fix.patch"])
def test_base_hashes_cover_all_patch_directories(isolated_meta_repo, patch_rel):
    root, metadata = isolated_meta_repo
    cuda_command = "ngc_base_refs pytorch 26.07-py3"
    rocm_command = "rocm_base_refs pytorch rocm10.0-ubuntu24.04-py3.12-torch2.13"
    before_cuda, before_rocm = metadata(cuda_command).split()[3], metadata(rocm_command).split()[2]
    patch = root / "Alps-Images/patches" / patch_rel
    patch.parent.mkdir(parents=True, exist_ok=True)
    patch.write_text((patch.read_text() if patch.exists() else "") + "\n")
    assert metadata(cuda_command).split()[3] != before_cuda
    assert metadata(rocm_command).split()[2] != before_rocm


def test_generated_and_editor_files_do_not_change_image_refs(isolated_meta_repo):
    root, metadata = isolated_meta_repo
    command = "ngc_base_refs pytorch 26.07-py3; rocm_base_refs pytorch rocm10.0-ubuntu24.04-py3.12-torch2.13; app_refs vllm cuda; app_refs vllm rocm"
    before = metadata(command)
    common = root / "Alps-Images/common"
    for directory in ("__pycache__", ".pytest_cache", ".git"):
        target = common / directory
        target.mkdir(exist_ok=True)
        (target / "generated").write_text("transient data")
    (common / "source-alps-env.sh.swp").write_text("editor state")
    assert metadata(command) == before
    source = common / "source-alps-env.sh"
    source.write_text(source.read_text() + "\n")
    assert metadata(command) != before


def test_apps_without_source_lists_still_hash_sources(isolated_meta_repo):
    root, metadata = isolated_meta_repo
    before = metadata("app_refs sfttrainer cuda")
    requirements = root / "Alps-Images/apps/sfttrainer/sources/requirements.txt"
    requirements.write_text(requirements.read_text() + "\n")
    assert metadata("app_refs sfttrainer cuda") != before


def test_meta_ngc_canonical_ref_changes_with_profile_inputs():
    profile = ROOT / "Alps-Images" / "NGC" / "pytorch-25.12-py3" / "profile.env"
    original = profile.read_text()
    baseline = run_meta("ngc_base_refs pytorch 25.12-py3").stdout.split()
    try:
        profile.write_text(original + '\nREMOVE_HPCX_DIRS="${REMOVE_HPCX_DIRS} /tmp/extra-hpcx"\n')
        changed_remove_hpcx = run_meta("ngc_base_refs pytorch 25.12-py3").stdout.split()
        profile.write_text(original + '\nNVCR_PREFIX=custom\n')
        changed_nvcr_prefix = run_meta("ngc_base_refs pytorch 25.12-py3").stdout.split()
    finally:
        profile.write_text(original)

    assert baseline[3] != changed_remove_hpcx[3]
    assert baseline[3] != changed_nvcr_prefix[3]


def test_meta_app_profile_is_not_executed(tmp_path):
    profile = ROOT / "Alps-Images" / "apps" / "vllm" / "profile.env"
    marker = tmp_path / "profile-executed"
    original = profile.read_text()
    try:
        profile.write_text(original + f"\ntouch {marker}\n")
        run_meta("app_refs vllm cuda")
        run_meta("app_validation_hash vllm cuda")
    finally:
        profile.write_text(original)

    assert not marker.exists()


def test_meta_write_build_env_includes_validation_marker_refs(tmp_path):
    output = tmp_path / "build.env"
    run_meta(f"write_base_build_env {output} cuda pytorch 25.12-py3")
    base_env = output.read_text()
    run_meta(f"write_app_build_env {output} vllm cuda")
    app_env = output.read_text()
    assert "VALIDATION_HASH=" in base_env
    assert "TESTED_IMAGE_REF=" in base_env
    assert "OCI_CREATED=" not in base_env
    assert "VALIDATION_HASH=" in app_env
    assert "TESTED_IMAGE_REF=" in app_env
    assert "OCI_CREATED=" not in app_env


def test_meta_validation_hashes_include_helper_scripts():
    result = run_meta("validation_helper_paths")
    paths = result.stdout.split()
    assert "ci-pipelines/helpers/meta.sh" in paths
    assert "ci-pipelines/helpers/skopeo.sh" in paths


def generate_manual(*args, output):
    env = meta_env()
    run_bash(f"manual-build/manual-build.sh {' '.join(args)} {output}", env=env)
    run_bash(f"bash -n {output}")
    return output.read_text()


def test_manual_build_generates_valid_cuda_base_script(tmp_path):
    script = generate_manual("base", "cuda", "pytorch", "25.12-py3", output=tmp_path / "base-cuda.sh")
    assert "podman build" in script
    assert "NGC_VARIANT_DIR" in script


def test_manual_build_generates_valid_rocm_base_script(tmp_path):
    script = generate_manual("base", "rocm", "pytorch", "rocm7.14-ubuntu24.04-py3.12-torch2.11", output=tmp_path / "base-rocm.sh")
    assert "podman build" in script
    assert "ROCM_VARIANT_DIR" in script
    assert re.search(r'--build-arg ROCM_SYSTEMS_COMMIT="[0-9a-f]{40}"', script)
    assert "ROCM_LIBRARIES_COMMIT" in script


def test_manual_build_generates_valid_cuda_app_script(tmp_path):
    script = generate_manual("app", "vllm", "cuda", output=tmp_path / "app-cuda.sh")
    assert "Alps-Images/apps/vllm/Containerfile" in script
    assert re.search(r'--build-arg BASE_IMAGE="localhost/alps-images/pytorch-cuda:.*-[0-9a-f]{32}"', script)
    assert '-t "localhost/alps-images/vllm-cuda:alps7-dev" \\' in script


def test_manual_build_generates_valid_rocm_app_script(tmp_path):
    script = generate_manual("app", "vllm", "rocm", output=tmp_path / "app-rocm.sh")
    assert "Alps-Images/apps/vllm/Containerfile.rocm" in script
    assert re.search(r'--build-arg BASE_IMAGE="localhost/alps-images/pytorch-rocm:.*-[0-9a-f]{32}"', script)
    assert '-t "localhost/alps-images/vllm-rocm:alps7-dev" \\' in script


@pytest.fixture()
def named_app_root(tmp_path):
    (tmp_path / "ci-pipelines").symlink_to(ROOT / "ci-pipelines", target_is_directory=True)
    images = tmp_path / "Alps-Images"
    images.mkdir()
    for name in ("NGC", "ROCm", "common", "patches"):
        (images / name).symlink_to(ROOT / "Alps-Images" / name, target_is_directory=True)
    app = images / "apps" / "apertus-2"
    app.mkdir(parents=True)
    targets = {
        "megatron-base-cuda": ("pytorch-cuda:25.12-py3", "Containerfile.megatron-base", "tests/megatron-base"),
        "megatron-cuda": ("pytorch-cuda:26.06-py3", "Containerfile.megatron", "tests/megatron"),
        "megatron-rocm": (
            "pytorch-rocm:rocm10.0-ubuntu24.04-py3.12-torch2.13",
            "Containerfile.megatron.rocm",
            "tests/rocm",
        ),
    }
    profile = [f'APP_VARIANTS="{" ".join(targets)}"']
    for target, (base, recipe, test_dir) in targets.items():
        prefix = target.upper().replace("-", "_")
        profile.extend([
            f"{prefix}_BASE_IMAGE={base}",
            f"{prefix}_CONTAINERFILE={recipe}",
            f"{prefix}_TEST_DIR={test_dir}",
        ])
        (app / recipe).write_text("ARG BASE_IMAGE\nFROM ${BASE_IMAGE}\n")
        tests = app / test_dir
        tests.mkdir(parents=True)
        (tests / "check-stack.py").write_text("assert True\n")
    (app / "profile.env").write_text("\n".join(profile) + "\n")
    (app / "ci.yaml").write_text("tests: {}\n")
    return tmp_path


def write_named_target_profile(root, targets):
    profile = root / "Alps-Images/apps/apertus-2/profile.env"
    lines = [f'APP_VARIANTS="{" ".join(targets)}"']
    for target, base in targets.items():
        prefix = target.upper().replace("-", "_")
        lines.extend([
            f"{prefix}_BASE_IMAGE={base}",
            f"{prefix}_CONTAINERFILE=Containerfile.megatron",
            f"{prefix}_TEST_DIR=tests/megatron",
        ])
    profile.write_text("\n".join(lines) + "\n")


def test_meta_named_targets_use_the_selected_base(named_app_root):
    targets = {
        "megatron-base-cuda": ("cuda", "ngc_base_refs pytorch 25.12-py3", 3, "Containerfile.megatron-base"),
        "megatron-cuda": ("cuda", "ngc_base_refs pytorch 26.06-py3", 3, "Containerfile.megatron"),
        "megatron-rocm": (
            "rocm",
            "rocm_base_refs pytorch rocm10.0-ubuntu24.04-py3.12-torch2.13",
            2,
            "Containerfile.megatron.rocm",
        ),
    }
    for target, (family, command, ref_index, recipe) in targets.items():
        output = named_app_root / f"{target}.env"
        run_meta(f"write_app_build_env {output} apertus-2 {target}", cwd=named_app_root)
        data = dict(line.split("=", 1) for line in output.read_text().splitlines())
        base_ref = run_meta(command, cwd=named_app_root).stdout.split()[ref_index]
        fields = run_meta(f"app_refs apertus-2 {target}", cwd=named_app_root).stdout.split()
        assert fields == [
            base_ref, f"Alps-Images/apps/apertus-2/{recipe}",
            data["CANON_IMAGE_REF"], data["STABLE_IMAGE_REF"], family,
        ]

        assert data["FAMILY"] == family
        assert data["VARIANT"] == target
        assert data["BASE_IMAGE"] == base_ref
        assert data["DOCKERFILE"] == f"Alps-Images/apps/apertus-2/{recipe}"
        assert f"apertus-2-{target}:" in data["CANON_IMAGE_REF"]
        assert data["TESTED_IMAGE_REF"] == f'{data["CANON_IMAGE_REF"]}-tested-{data["VALIDATION_HASH"]}'


@pytest.mark.parametrize(
    "variants",
    ["megatron-base-cuda megatron_base-cuda", "megatron-stage-cuda megatron_stage-cuda", "megatron-cuda megatron-cuda"],
)
@pytest.mark.parametrize("helper", ["app_refs", "app_validation_hash"])
def test_meta_rejects_colliding_target_prefixes(named_app_root, variants, helper):
    write_named_target_profile(named_app_root, dict.fromkeys(variants.split(), "pytorch-cuda:26.06-py3"))
    profile = named_app_root / "Alps-Images/apps/apertus-2/profile.env"
    lines = profile.read_text().splitlines()
    lines[0] = f'APP_VARIANTS="{variants}"'
    profile.write_text("\n".join(lines) + "\n")

    result = run_meta(f"{helper} apertus-2 {variants.split()[0]}", cwd=named_app_root, check=False)

    assert result.returncode != 0
    assert "share profile prefix" in result.stderr


@pytest.mark.parametrize(
    "target",
    ["megatron", "megatron-base", "megatron-tpu", "cuda-debug", "-cuda", "megatron.rocm"],
)
def test_meta_app_refs_reject_invalid_target_names(named_app_root, target):
    write_named_target_profile(named_app_root, {target: "pytorch-cuda:26.06-py3"})

    result = run_meta(f"app_refs apertus-2 {target}", cwd=named_app_root, check=False)

    assert result.returncode != 0
    assert target in result.stderr


@pytest.mark.parametrize("target", [
    "Megatron-cuda", "MEGATRON-cuda", "megatron-CUDA", "CUDA",
    "_team-stage-cuda", "_team-stage-rocm",
    "a_-cuda", "a-_stage-cuda", "a___stage-cuda",
    "a_-stage-rocm", "a--_stage-rocm",
])
@pytest.mark.parametrize("helper", ["app_refs", "app_validation_hash"])
def test_meta_consumers_reject_invalid_target_syntax(named_app_root, target, helper):
    base = "pytorch-rocm:rocm10.0-ubuntu24.04-py3.12-torch2.13" if target.endswith("-rocm") else "pytorch-cuda:26.06-py3"
    write_named_target_profile(named_app_root, {target: base})

    result = run_meta(f"{helper} apertus-2 {target}", cwd=named_app_root, check=False)

    assert result.returncode != 0
    assert result.stdout == ""
    assert target in result.stderr


@pytest.mark.parametrize(
    "target",
    [
        "cuda-rocm", "rocm-cuda", "cuda-cuda", "rocm-rocm",
        "cuda-stage-rocm", "rocm-stage-cuda",
        "team-cuda-stage-rocm", "team-rocm-stage-cuda",
        "team_cuda-rocm", "team_rocm-cuda",
    ],
)
@pytest.mark.parametrize("helper", ["app_refs", "app_validation_hash"])
def test_meta_consumers_reject_family_tokens_in_unselected_variant_names(named_app_root, target, helper):
    family = target.rsplit("-", 1)[-1]
    base = "pytorch-cuda:26.06-py3" if family == "cuda" else "pytorch-rocm:rocm10.0-ubuntu24.04-py3.12-torch2.13"
    write_named_target_profile(
        named_app_root,
        {"megatron-cuda": "pytorch-cuda:26.06-py3", target: base},
    )

    result = run_meta(f"{helper} apertus-2 megatron-cuda", cwd=named_app_root, check=False)

    assert result.returncode != 0
    assert result.stdout == ""
    assert target in result.stderr
    assert "must not contain cuda or rocm variant components" in result.stderr



@pytest.mark.parametrize("helper", ["app_refs", "app_validation_hash"])
def test_meta_consumers_reject_undeclared_target(named_app_root, helper):
    write_named_target_profile(named_app_root, {"megatron-cuda": "pytorch-cuda:26.06-py3"})

    result = run_meta(f"{helper} apertus-2 megatron-rocm", cwd=named_app_root, check=False)

    assert result.returncode != 0
    assert result.stdout == ""
    assert "megatron-rocm" in result.stderr


@pytest.mark.parametrize(
    "target,family,base",
    [
        ("cuda", "cuda", "pytorch-cuda:26.06-py3"),
        ("rocm", "rocm", "pytorch-rocm:rocm10.0-ubuntu24.04-py3.12-torch2.13"),
        ("team-long-stage-cuda", "cuda", "pytorch-cuda:26.06-py3"),
        ("team-long-stage-rocm", "rocm", "pytorch-rocm:rocm10.0-ubuntu24.04-py3.12-torch2.13"),
        ("cudagraph-rocm", "rocm", "pytorch-rocm:rocm10.0-ubuntu24.04-py3.12-torch2.13"),
        ("team_rocmish-cuda", "cuda", "pytorch-cuda:26.06-py3"),
        ("team__stage-cuda", "cuda", "pytorch-cuda:26.06-py3"),
        ("team--stage-rocm", "rocm", "pytorch-rocm:rocm10.0-ubuntu24.04-py3.12-torch2.13"),
    ],
)
def test_meta_build_env_accepts_family_only_and_multi_part_targets(named_app_root, target, family, base):
    write_named_target_profile(named_app_root, {target: base})
    output = named_app_root / "build.env"

    run_meta(f"write_app_build_env {output} apertus-2 {target}", cwd=named_app_root)

    data = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert data["VARIANT"] == target
    assert data["FAMILY"] == family
    assert f"/pytorch-{family}:" in data["BASE_IMAGE"]
    assert f"/apertus-2-{target}:" in data["CANON_IMAGE_REF"]
    assert data["VALIDATION_HASH"]
    assert data["TESTED_IMAGE_REF"] == f'{data["CANON_IMAGE_REF"]}-tested-{data["VALIDATION_HASH"]}'


@pytest.mark.parametrize(
    "helper,target,base",
    [
        ("app_refs", "stage-rocm", "pytorch-cuda:26.06-py3"),
        ("app_refs", "cuda", "pytorch-rocm:rocm10.0-ubuntu24.04-py3.12-torch2.13"),
        ("app_validation_hash", "stage-cuda", "pytorch-rocm:rocm10.0-ubuntu24.04-py3.12-torch2.13"),
        ("app_validation_hash", "rocm", "pytorch-cuda:26.06-py3"),
    ],
)
def test_meta_consumers_reject_mismatches_in_unselected_targets(named_app_root, helper, target, base):
    write_named_target_profile(
        named_app_root,
        {"megatron-cuda": "pytorch-cuda:26.06-py3", target: base},
    )

    result = run_meta(f"{helper} apertus-2 megatron-cuda", cwd=named_app_root, check=False)

    assert result.returncode != 0
    assert target in result.stderr
    assert base in result.stderr


@pytest.mark.parametrize(
    "target,base,error",
    [
        ("Megatron-rocm", "pytorch-rocm:rocm10.0-ubuntu24.04-py3.12-torch2.13", "must match"),
        ("megatron-rocm", "", "MEGATRON_ROCM_BASE_IMAGE must be set"),
        ("megatron-rocm", "pytorch:plain", "expected BASE_IMAGE"),
    ],
)
def test_meta_build_env_rejects_malformed_unselected_declarations(named_app_root, target, base, error):
    write_named_target_profile(
        named_app_root,
        {"megatron-cuda": "pytorch-cuda:26.06-py3", target: base},
    )
    output = named_app_root / "build.env"

    result = run_meta(f"write_app_build_env {output} apertus-2 megatron-cuda", cwd=named_app_root, check=False)

    assert result.returncode != 0
    assert result.stdout == ""
    assert error in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("failure", ["base-family-mismatch", "missing-validation-input"])
def test_meta_write_app_build_env_propagates_failures(named_app_root, failure):
    if failure == "base-family-mismatch":
        write_named_target_profile(
            named_app_root,
            {"megatron-cuda": "pytorch-rocm:rocm10.0-ubuntu24.04-py3.12-torch2.13"},
        )
    else:
        (named_app_root / "Alps-Images/apps/apertus-2/ci.yaml").unlink()
    output = named_app_root / "build.env"

    result = run_meta(f"write_app_build_env {output} apertus-2 megatron-cuda", cwd=named_app_root, check=False)

    assert result.returncode != 0
    assert result.stderr
    assert not output.exists()


@pytest.mark.parametrize(
    "target,recipe,test_dir",
    [
        ("megatron-base-cuda", "Containerfile.megatron-base", "tests/megatron-base"),
        ("megatron-cuda", "Containerfile.megatron", "tests/megatron"),
        ("megatron-rocm", "Containerfile.megatron.rocm", "tests/rocm"),
    ],
)
def test_megatron_hashes_use_the_selected_recipe_and_test_group(named_app_root, target, recipe, test_dir):
    app = named_app_root / "Alps-Images/apps/apertus-2"
    targets = ("megatron-base-cuda", "megatron-cuda", "megatron-rocm")
    refs = {name: run_meta(f"app_refs apertus-2 {name}", cwd=named_app_root).stdout for name in targets}
    validations = {
        name: run_meta(f"app_validation_hash apertus-2 {name}", cwd=named_app_root).stdout
        for name in targets
    }
    for relative in (recipe, f"{test_dir}/check-stack.py"):
        path = app / relative
        original = path.read_text()
        path.write_text(original + "\n# Changed hash input.\n")
        for name in targets:
            current_ref = run_meta(f"app_refs apertus-2 {name}", cwd=named_app_root).stdout
            current_validation = run_meta(f"app_validation_hash apertus-2 {name}", cwd=named_app_root).stdout
            assert (current_ref != refs[name]) == (name == target)
            assert (current_validation != validations[name]) == (name == target and relative.startswith("tests/"))
        path.write_text(original)


@pytest.mark.parametrize(
    "source,affected",
    [
        ("shared.sh", {"megatron-base-cuda", "megatron-cuda", "megatron-rocm"}),
        ("base.sh", {"megatron-base-cuda"}),
        ("cuda.sh", {"megatron-cuda"}),
        ("rocm.sh", {"megatron-rocm"}),
        ("unused.sh", set()),
    ],
)
def test_named_target_source_lists_hash_only_selected_inputs(named_app_root, source, affected):
    app = named_app_root / "Alps-Images/apps/apertus-2"
    sources = app / "sources"
    sources.mkdir()
    for name in ("shared.sh", "base.sh", "cuda.sh", "rocm.sh", "unused.sh"):
        (sources / name).write_text(f"# {name}\n")
    profile = app / "profile.env"
    profile.write_text(
        profile.read_text()
        + "COMMON_SOURCE_FILES=sources/shared.sh\n"
        + "MEGATRON_BASE_CUDA_SOURCE_FILES=sources/base.sh\n"
        + "MEGATRON_CUDA_SOURCE_FILES=sources/cuda.sh\n"
        + "MEGATRON_ROCM_SOURCE_FILES=sources/rocm.sh\n"
    )
    targets = ("megatron-base-cuda", "megatron-cuda", "megatron-rocm")
    before = {target: run_meta(f"app_refs apertus-2 {target}", cwd=named_app_root).stdout for target in targets}
    (sources / source).write_text("# changed source\n")

    for target in targets:
        after = run_meta(f"app_refs apertus-2 {target}", cwd=named_app_root).stdout
        assert (after != before[target]) == (target in affected)


def test_named_target_hashes_use_effective_selected_profile_values(named_app_root):
    app = named_app_root / "Alps-Images/apps/apertus-2"
    profile = app / "profile.env"
    original = profile.read_text()
    command = "app_refs apertus-2 megatron-cuda"
    before = run_meta(command, cwd=named_app_root).stdout
    profile.write_text(
        original.replace(
            'APP_VARIANTS="megatron-base-cuda megatron-cuda megatron-rocm"',
            'APP_VARIANTS="megatron-rocm megatron-cuda"',
        ).replace(
            "MEGATRON_CUDA_CONTAINERFILE=Containerfile.megatron",
            "COMMON_CONTAINERFILE=Containerfile.megatron",
        ).replace(
            "MEGATRON_ROCM_BASE_IMAGE=pytorch-rocm:rocm10.0-ubuntu24.04-py3.12-torch2.13",
            "MEGATRON_ROCM_BASE_IMAGE=pytorch-rocm:rocm7.14-ubuntu24.04-py3.12-torch2.11",
        )
    )

    assert run_meta(command, cwd=named_app_root).stdout == before
    profile.write_text(
        profile.read_text().replace(
            "MEGATRON_CUDA_BASE_IMAGE=pytorch-cuda:26.06-py3",
            "MEGATRON_CUDA_BASE_IMAGE=pytorch-cuda:25.12-py3",
        )
    )
    assert run_meta(command, cwd=named_app_root).stdout != before


def test_named_target_recursive_sources_exclude_transients_and_cover_dockerignore(named_app_root):
    app = named_app_root / "Alps-Images/apps/apertus-2"
    sources = app / "sources"
    sources.mkdir()
    nested = sources / "nested"
    nested.mkdir()
    source = nested / "install.sh"
    source.write_text("# source\n")
    dockerignore = named_app_root / ".dockerignore"
    dockerignore.write_text(".git\n__pycache__\n")
    command = "app_refs apertus-2 megatron-cuda"
    before = run_meta(command, cwd=named_app_root).stdout
    for directory in (".git", "__pycache__", ".pytest_cache"):
        transient = sources / directory
        transient.mkdir()
        (transient / "generated").write_text("cache\n")
    for name in ("install.pyc", "install.pyo", "install.sh.swp", "install.sh.swo", "install.sh~"):
        (nested / name).write_text("transient\n")

    assert run_meta(command, cwd=named_app_root).stdout == before
    source.write_text("# changed source\n")
    changed_source = run_meta(command, cwd=named_app_root).stdout
    assert changed_source != before
    dockerignore.write_text(dockerignore.read_text() + ".pytest_cache\n")
    assert run_meta(command, cwd=named_app_root).stdout != changed_source


def source_rocm(command):
    return (
        "set -e; source Alps-Images/ROCm/install-alps-rocm-components.sh; "
        'die() { printf "%s\\n" "$*" >&2; exit 1; }; '
        + command
    )


@pytest.mark.parametrize("kind", ["directory", "symlink", "dangling-symlink", "marker-symlink"])
def test_rocm_compat_root_refuses_unmanaged_layout(tmp_path, kind):
    root = tmp_path / "rocm"
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    sentinel = vendor / "preserve"
    sentinel.write_text("vendor data")
    if kind == "directory":
        root.mkdir()
        (root / "preserve").write_text("user data")
    elif kind == "marker-symlink":
        root.mkdir()
        marker = tmp_path / "marker"
        marker.write_text("alps-rocm-compat-v1\n")
        (root / ".alps-rocm-compat").symlink_to(marker)
    else:
        root.symlink_to(vendor if kind == "symlink" else tmp_path / "absent")
    result = run_bash(source_rocm('recreate_rocm_compat_root "$TEST_ROOT"'), env={"TEST_ROOT": str(root)}, check=False)
    assert result.returncode != 0
    assert "Refusing to replace unmanaged ROCm" in result.stderr
    assert sentinel.read_text() == "vendor data"
    if kind == "directory":
        assert (root / "preserve").read_text() == "user data"
    elif kind.endswith("symlink") and kind != "marker-symlink":
        assert root.is_symlink()


def test_rocm_compat_root_recreates_only_owned_tree(tmp_path):
    root = tmp_path / "rocm"
    root.mkdir()
    (root / ".alps-rocm-compat").write_text("alps-rocm-compat-v1\n")
    (root / "obsolete").write_text("owned")
    run_bash(source_rocm('recreate_rocm_compat_root "$TEST_ROOT"'), env={"TEST_ROOT": str(root)})
    assert not (root / "obsolete").exists()
    assert (root / ".alps-rocm-compat").read_text() == "alps-rocm-compat-v1\n"
    assert (root / "include").is_dir()


def fake_rocm_distributions(tmp_path, versions):
    site = tmp_path / "site"
    site.mkdir()
    for name, version in versions.items():
        dist = site / f"{name.replace('-', '_')}-{version}.dist-info"
        dist.mkdir()
        (dist / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
    return {"ROCM_PYTHON": sys.executable, "PYTHONPATH": str(site), "ROCM_VERSION": "7.14.0"}


@pytest.mark.parametrize(
    ("installed", "status"),
    [("7.14.0", 0), ("7.14.0+git.abc123", 0), ("7.14.0a20261015", 0), ("7.14.0.dev0+abc123", 0), ("7.14.0rc2", 0), ("7.13.0", 2), ("7.14.1", 2)],
)
def test_rocm_sdk_validates_distribution_release(tmp_path, installed, status):
    env = fake_rocm_distributions(tmp_path, {"rocm-sdk-core": installed})
    result = run_bash(source_rocm("rocm_sdk_distribution_present rocm-sdk-core"), env=env, check=False)
    assert result.returncode == status
    if status:
        assert f"installed {installed}, expected ROCm 7.14.0" in result.stderr


def test_rocm_sdk_detects_device_distribution_without_importable_module(tmp_path):
    env = fake_rocm_distributions(tmp_path, {"rocm-sdk-device-gfx942": "7.14.0"})
    result = run_bash(source_rocm("rocm_sdk_distribution_present rocm-sdk-device-gfx942"), env=env)
    assert result.returncode == 0
    missing = run_bash(source_rocm("rocm_sdk_distribution_present rocm-sdk-device-gfx90a"), env=env, check=False)
    assert missing.returncode == 1


@pytest.mark.parametrize("fail_install", [False, True])
def test_rocm_sdk_installs_absent_devel_and_propagates_network_failure(tmp_path, fail_install):
    versions = {name: "7.14.0" for name in ("rocm", "rocm-sdk-core", "rocm-sdk-libraries", "rocm-sdk-device-gfx942")}
    env = fake_rocm_distributions(tmp_path, versions)
    site = Path(env["PYTHONPATH"])
    pip = site / "pip"
    pip.mkdir()
    (pip / "__init__.py").write_text("")
    (pip / "__main__.py").write_text(
        "import os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['PIP_LOG']).write_text(' '.join(sys.argv[1:]))\n"
        "if os.environ['FAIL_INSTALL'] == '1':\n"
        "    sys.exit('network unavailable')\n"
        "site = Path(os.environ['PYTHONPATH'])\n"
        "dist = site / 'rocm_sdk_devel-7.14.0.dist-info'\n"
        "dist.mkdir()\n"
        "(dist / 'METADATA').write_text('Metadata-Version: 2.1\\nName: rocm-sdk-devel\\nVersion: 7.14.0\\n')\n"
    )
    log = tmp_path / "pip.log"
    env.update({
        "RCCL_GPU_TARGETS": "gfx942",
        "ROCM_PYPI_INDEX_URL": "https://example.invalid",
        "ROCM_LIBRARIES_REPO": "",
        "ROCM_LIBRARIES_COMMIT": "",
        "PIP_LOG": str(log),
        "FAIL_INSTALL": "1" if fail_install else "0",
    })
    result = run_bash(source_rocm('install_rocm_sdk_wheels; printf "devel=%s\\n" "$ROCM_HAVE_DEVEL"'), env=env, check=False)
    assert "rocm-sdk-devel==7.14.0" in log.read_text()
    if fail_install:
        assert result.returncode != 0
        assert "network unavailable" in result.stderr
        assert "Failed to install required ROCm SDK wheels" in result.stderr
    else:
        assert result.returncode == 0
        assert "devel=yes" in result.stdout


def test_rocm_sdk_source_pins_explicitly_allow_missing_devel(tmp_path):
    versions = {name: "7.14.0" for name in ("rocm", "rocm-sdk-core", "rocm-sdk-libraries", "rocm-sdk-device-gfx942")}
    env = fake_rocm_distributions(tmp_path, versions)
    env.update({"RCCL_GPU_TARGETS": "gfx942", "ROCM_LIBRARIES_REPO": "https://example.invalid/pinned.git", "ROCM_LIBRARIES_COMMIT": "a" * 40})
    result = run_bash(source_rocm('install_rocm_sdk_wheels; printf "devel=%s\\n" "$ROCM_HAVE_DEVEL"'), env=env)
    assert "devel=no" in result.stdout


@pytest.fixture
def miopen_headers_runtime(tmp_path):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("C compiler required for the MIOpen runtime ABI fixture")
    source = tmp_path / "sources"
    project = source / "projects/miopen"
    headers = project / "include/miopen"
    headers.mkdir(parents=True)
    (project / "CMakeLists.txt").write_text("rocm_setup_version(VERSION 3.6.0)\n")
    (headers / "miopen.h").write_text("/* public C API */\n")
    (headers / "config.h.in").write_text("/* upstream build configuration */\n")
    (headers / "version.h.in").write_text("".join(
        f"#define MIOPEN_VERSION_{name} @MIOpen_VERSION_{name}@\n"
        for name in ("MAJOR", "MINOR", "PATCH", "TWEAK")
    ))
    sdk = tmp_path / "sdk"
    (sdk / "lib").mkdir(parents=True)
    runtime = tmp_path / "miopen.c"
    runtime.write_text(
        "#include <stddef.h>\n"
        "int miopenGetVersion(size_t *major, size_t *minor, size_t *patch) {\n"
        "    *major = 3; *minor = 6; *patch = 0; return 0;\n"
        "}\n"
    )
    subprocess.run([compiler, "-shared", "-fPIC", str(runtime), "-o", str(sdk / "lib/libMIOpen.so.1")], check=True)
    env = {"SOURCE": str(source), "SDK": str(sdk), "ROCM_PYTHON": sys.executable,
           "ROCM_CORE_PREFIX": str(sdk), "ROCM_LIBRARIES_PREFIX": str(sdk)}
    return project, sdk, env


@pytest.mark.parametrize("declaration", [
    "rocm_setup_version(VERSION 3.6.0)",
    'rocm_setup_version ( VERSION "3.6.0" )',
    'ROCM_SETUP_VERSION(\\n\\tVERSION\\t"3.6.0"\\n)',
])
def test_miopen_public_headers_accept_cmake_version_layouts(miopen_headers_runtime, declaration):
    project, sdk, env = miopen_headers_runtime
    (project / "CMakeLists.txt").write_text(declaration.replace("\\n", "\n").replace("\\t", "\t") + "\n")
    run_bash(source_rocm('install_rocm_miopen_headers "$SOURCE" "$SDK"'), env=env)
    headers = sdk / "include/miopen"
    assert {path.name for path in headers.iterdir()} == {"miopen.h", "config.h", "export.h", "version.h"}
    version = (headers / "version.h").read_text()
    assert "#define MIOPEN_VERSION_MAJOR 3\n" in version
    assert "#define MIOPEN_VERSION_MINOR 6\n" in version
    assert "@" not in version


@pytest.mark.parametrize("failure", ["missing-runtime", "version-mismatch", "private-header", "private-directory"])
def test_miopen_public_headers_refuse_invalid_payloads(miopen_headers_runtime, failure):
    project, sdk, env = miopen_headers_runtime
    if failure == "missing-runtime":
        (sdk / "lib/libMIOpen.so.1").unlink()
    elif failure == "version-mismatch":
        (project / "CMakeLists.txt").write_text('rocm_setup_version ( VERSION "3.7.0" )\n')
    elif failure == "private-header":
        (project / "include/miopen/internal.hpp").write_text("/* private */\n")
    else:
        (project / "include/miopen/internal").mkdir()
    result = run_bash(source_rocm('install_rocm_miopen_headers "$SOURCE" "$SDK"'), env=env, check=False)
    assert result.returncode != 0
    assert not (sdk / "include/miopen").exists()
    if failure == "missing-runtime":
        assert str(sdk / "lib/libMIOpen.so.1") in result.stderr
        assert "OSError" not in result.stderr
    elif failure == "version-mismatch":
        assert "(3, 7, 0)" in result.stderr and "(3, 6, 0)" in result.stderr
    else:
        assert ("internal.hpp" if failure == "private-header" else "internal") in result.stderr


def test_rocm_build_prefix_uses_development_view_without_reordering_runtime_libraries(tmp_path):
    core = tmp_path / "core"
    libraries = tmp_path / "libraries"
    devel = tmp_path / "devel"
    # A runtime tree may have enough HIP/HSA files to pass the discovery
    # probes, but extension builds still need the complete development view.
    for prefix in (core, devel):
        for header in (
            "hip/hip_runtime_api.h", "hip/hip_runtime.h", "hip/hip_version.h",
            "hsa/hsa.h", "hsa/hsa_ext_amd.h",
        ):
            path = prefix / "include" / header
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("/* public header */\n")
        (prefix / "lib").mkdir()
        for name in ("libamdhip64.so", "libhsa-runtime64.so", "libhsakmt.so"):
            (prefix / "lib" / name).write_text("runtime")
    (libraries / "lib").mkdir(parents=True)
    (libraries / "lib" / "librccl.so").write_text("runtime")
    env = {
        "ROCM_CORE_PREFIX": str(core),
        "ROCM_LIBRARIES_PREFIX": str(libraries),
        "ROCM_DEVEL_PREFIX": str(devel),
        "ROCM_CORE_DIR": str(core),
        "ROCM_LIBRARIES_DIR": str(libraries),
        "ROCM_DEVEL_DIR": str(devel),
        "ROCM_SDK_ROOT": str(devel),
        "ROCM_SDK_BIN": str(devel / "bin"),
        "ROCM_SDK_CMAKE": str(devel / "lib/cmake"),
        "ROCM_PYTHON": sys.executable,
        "ROCM_SDK": str(devel / "bin/rocm-sdk"),
    }
    selected = run_bash(source_rocm("discover_rocm_build_prefix"), env=env)
    assert Path(selected.stdout.strip()) == devel
    runtime_dirs = run_bash(source_rocm("rocm_sdk_ldconfig_dirs"), env=env).stdout.splitlines()
    assert runtime_dirs.index(str(core / "lib")) < runtime_dirs.index(str(devel / "lib"))
    assert runtime_dirs.index(str(libraries / "lib")) < runtime_dirs.index(str(devel / "lib"))
    # Simulate bootstrap's selection, persist it, then load it in a fresh shell.
    alps_env = tmp_path / "env"
    component = tmp_path / "components.sh"
    component.write_text(
        (ROOT / "Alps-Images/ROCm/install-alps-rocm-components.sh").read_text()
        .replace("/opt/alps/env", str(alps_env))
    )
    run_bash(
        f'set -eu; source "{component}"; '
        'ROCM_BUILD_PREFIX="$(discover_rocm_build_prefix)"; persist_rocm_sdk_env',
        env=env,
    )
    loaded = run_bash(
        f'set -eu; source "{component}"; unset ROCM_BUILD_PREFIX; load_rocm_sdk_env; '
        'printf "%s\\n" "$ROCM_BUILD_PREFIX" "$ROCM_HOME" "$ROCM_PATH" "$HIP_PATH"',
        env=env,
    )
    assert loaded.stdout.splitlines() == [str(devel)] * 4


def test_vllm_rocm_build_setup_preserves_runtime_linker_order(tmp_path):
    core = tmp_path / "z-core"
    libraries = tmp_path / "zz-libraries"
    devel = tmp_path / "a-devel"
    for prefix in (core, libraries, devel):
        (prefix / "lib").mkdir(parents=True)
        runtime = "libamdhip64.so.7" if prefix == core else "librccl.so.1"
        (prefix / "lib" / runtime).write_text("runtime")
        (prefix / "blas").mkdir()
        (prefix / "blas" / "librocm-openblas.so.0").write_text("openblas")
    alps = tmp_path / "alps"
    (alps / "env").mkdir(parents=True)
    (alps / "env" / "alps-rocm-build.env").write_text(
        f"export ROCM_CORE_PREFIX={core}\n"
        f"export ROCM_LIBRARIES_PREFIX={libraries}\n"
        f"export ROCM_DEVEL_PREFIX={devel}\n"
        f"export ROCM_BUILD_PREFIX={devel}\n"
    )
    (alps / "package-helpers.sh").write_text(
        'snapshot_apt_packages() { printf "inherited:amd64\\n" > "$1"; }\n'
        "apt_get() { :; }\n"
    )
    (alps / "install-alps-hpc-stack.sh").write_text("")
    (alps / "install-alps-rocm-components.sh").write_text(
        (ROOT / "Alps-Images/ROCm/install-alps-rocm-components.sh").read_text()
        .replace("/opt/rocm", str(tmp_path / "compat"))
    )
    ldconfig = tmp_path / "ldconfig"
    ldconfig.mkdir()
    # Relocate container filesystem paths only; execute the real setup and SDK
    # discovery functions, with package installation isolated from the host.
    helper = tmp_path / "build-rocm-env.sh"
    helper.write_text(
        (ROOT / "Alps-Images/apps/vllm/sources/build-rocm-env.sh").read_text()
        .replace("/opt/alps", str(alps))
        .replace("/etc/ld.so.conf.d", str(ldconfig))
        .replace("/tmp/vllm-apt-before.txt", str(tmp_path / "apt-before.txt"))
        .replace("/var/lib/apt/lists", str(tmp_path / "apt-lists"))
    )
    run_bash(f'set -euo pipefail; source "{helper}"; vllm_build_begin')
    runtime_dirs = (ldconfig / "99-alps-vllm-rocm-sdk.conf").read_text().splitlines()
    assert runtime_dirs == [
        str(core / "lib"), str(libraries / "lib"), str(devel / "lib"),
        str(core / "blas"), str(libraries / "blas"), str(devel / "blas"),
    ]


def test_rocm_714_rccl_test_sources_remain_unchanged(tmp_path):
    source = tmp_path / "projects" / "rccl-tests" / "src"
    source.mkdir(parents=True)
    cmake = source / "CMakeLists.txt"
    original = "add_library(rccl_common OBJECT ${HIP_COMMON_SOURCES})\n"
    cmake.write_text(original)
    run_bash(source_rocm("patch_rocm_rccl_test_sources"), env={"ROCM_VERSION": "7.14.0", "ROCM_SYSTEMS_SRC_DIR": str(tmp_path)})
    assert cmake.read_text() == original


def test_rocm_component_versions_come_from_source_templates(tmp_path):
    source = tmp_path / "source"
    root = tmp_path / "rocm"
    templates = {
        "hipblas": ("library/include/hipblas-version.h.in", "hipblas/hipblas-version.h"),
        "hipblaslt": ("library/include/hipblaslt-version.h.in", "hipblaslt/hipblaslt-version.h"),
        "hipsparselt": ("library/include/hipsparselt-version.h.in", "hipsparselt/hipsparselt-version.h"),
        "hipsparse": ("library/include/hipsparse-version.h.in", "hipsparse/hipsparse-version.h"),
        "hipsolver": ("library/include/internal/hipsolver-version.h.in", "hipsolver/internal/hipsolver-version.h"),
        "rocprim": ("rocprim/include/rocprim/rocprim_version.hpp.in", "rocprim/rocprim_version.hpp"),
        "hipcub": ("hipcub/include/hipcub/hipcub_version.hpp.in", "hipcub/hipcub_version.hpp"),
        "rocthrust": ("thrust/rocthrust_version.hpp.in", "thrust/rocthrust_version.hpp"),
        "rocrand": ("library/include/rocrand/rocrand_version.h.in", "rocrand/rocrand_version.h"),
        "hiprand": ("library/include/hiprand/hiprand_version.h.in", "hiprand/hiprand_version.h"),
    }
    for package in [*templates, "rocblas", "rocsolver", "miopen", "hipfft"]:
        project = source / "projects" / package
        project.mkdir(parents=True)
        declaration = 'set(VERSION_STRING "8.2.3")\n'
        if package == "hipsparselt":
            declaration = "project(hipsparselt VERSION 0.2.9 LANGUAGES CXX)\n"
        elif package == "hipblaslt":
            declaration = 'set(HIPBLASLT_PROJECT_VERSION "1.4.1" CACHE STRING "version")\n'
        elif package == "miopen":
            declaration = "rocm_setup_version(VERSION 3.6.0)\n"
        elif package == "hipfft":
            declaration = '\n'.join(f'set(HIPFFT_VERSION_{key} "{value}")' for key, value in zip(("MAJOR", "MINOR", "PATCH"), (1, 0, 25)))
        if package == "hipcub":
            declaration += "\n".join(f"set(HIPCUB_CCCL_VERSION_{key} {value})" for key, value in zip(("MAJOR", "MINOR", "PATCH"), (2, 8, 2)))
        if package == "hipsparse":
            declaration += '\noption(HIPSPARSE_ENABLE_SPMV_BSR "feature" ON)\n'
        (project / "CMakeLists.txt").write_text(declaration)
        if package in templates:
            template = project / templates[package][0]
            template.parent.mkdir(parents=True, exist_ok=True)
            content = f"version=@{package}_VERSION_MAJOR@.@{package}_VERSION_MINOR@.@{package}_VERSION_PATCH@\nnumber=@{package}_VERSION_NUMBER@\n"
            if package == "hipcub":
                content += "cccl=@HIPCUB_CCCL_VERSION_NUMBER@\n"
            if package == "hipsparse":
                content += "#cmakedefine HIPSPARSE_WITH_SPMV_BSR\n"
                (template.parent / "CMakeLists.txt").write_text("set(HIPSPARSE_WITH_SPMV_BSR ${HIPSPARSE_ENABLE_SPMV_BSR})\n")
            template.write_text(content)
    env = {"SOURCE": str(source), "TEST_ROOT": str(root), "ROCM_PYTHON": sys.executable}
    result = run_bash(source_rocm('generate_rocm_component_metadata "$SOURCE" "$TEST_ROOT"'), env=env)
    assert "hipsparselt=0.2.9" in result.stdout
    assert "hipblaslt=1.4.1" in result.stdout
    assert "hipfft=1.0.25" in result.stdout
    assert "number=800203" in (root / "include/rocprim/rocprim_version.hpp").read_text()
    assert "cccl=200802" in (root / "include/hipcub/hipcub_version.hpp").read_text()
    assert "#define HIPSPARSE_WITH_SPMV_BSR" in (root / "include/hipsparse/hipsparse-version.h").read_text()
    template = source / "projects/hipblas" / templates["hipblas"][0]
    template.write_text("@UNKNOWN_VERSION_FIELD@\n")
    failed = run_bash(source_rocm('generate_rocm_component_metadata "$SOURCE" "$TEST_ROOT"'), env=env, check=False)
    assert failed.returncode != 0
    assert "Unknown version template field" in failed.stderr


def test_rocm_development_link_is_created_at_real_wheel_location(tmp_path):
    wheel = tmp_path / "wheel" / "lib"
    compat = tmp_path / "rocm" / "lib"
    wheel.mkdir(parents=True)
    compat.mkdir(parents=True)
    library = wheel / "libamdhip64.so.7.0"
    library.write_text("runtime library")
    (wheel / "libamdhip64.so.7").symlink_to(library.name)
    runtime_link = compat / "libamdhip64.so.7"
    runtime_link.symlink_to(wheel / "libamdhip64.so.7")
    dev_link = compat / "libamdhip64.so"
    run_bash(
        source_rocm('link_rocm_development_library "$RUNTIME_LINK" "$DEV_LINK"'),
        env={"RUNTIME_LINK": str(runtime_link), "DEV_LINK": str(dev_link)},
    )
    assert dev_link.resolve() == library
    assert (wheel / "libamdhip64.so").resolve() == library


@pytest.mark.parametrize("mismatch", ["rocm", "rocm-sdk-core", "rocm-sdk-libraries", "rocm-sdk-device-gfx942"])
def test_rocm_sdk_refuses_mismatched_preinstalled_wheels(tmp_path, mismatch):
    versions = {name: "7.14.0" for name in ("rocm", "rocm-sdk-core", "rocm-sdk-libraries", "rocm-sdk-device-gfx942")}
    versions[mismatch] = "7.13.0"
    env = fake_rocm_distributions(tmp_path, versions)
    env.update({"RCCL_GPU_TARGETS": "gfx942", "ROCM_LIBRARIES_REPO": "https://example.invalid/pinned.git", "ROCM_LIBRARIES_COMMIT": "a" * 40})
    result = run_bash(source_rocm("install_rocm_sdk_wheels"), env=env, check=False)
    assert result.returncode != 0
    assert f"Incompatible {mismatch}: installed 7.13.0, expected ROCm 7.14.0" in result.stderr


def test_exact_build_package_purge_refuses_reverse_dependency_removal(tmp_path):
    import shutil

    apt_get = shutil.which("apt-get")
    if apt_get is None:
        pytest.skip("requires the real APT dependency solver")
    status = tmp_path / "status"
    status.write_text(
        "\n".join(
            f"Package: {name}\nStatus: install ok installed\n"
            "Architecture: all\nVersion: 1.0\n"
            f"{dependencies}Description: isolated cleanup regression fixture\n"
            for name, dependencies in (
                ("libibverbs1", ""),
                ("librdmacm1", "Depends: libibverbs1\n"),
                ("libibverbs-dev", "Depends: libibverbs1\n"),
                ("librdmacm-dev", "Depends: libibverbs-dev, librdmacm1\n"),
                ("doca-base", "Depends: libibverbs-dev\n"),
            )
        )
    )
    result = run_bash(
        r'''
source Alps-Images/common/package-helpers.sh
apt_get() {
    if [[ " $* " != *" --simulate "* ]]; then
        echo "Unexpected attempt to mutate packages" >&2
        return 99
    fi
    "$REAL_APT_GET" \
        -o Debug::NoLocking=1 \
        -o Dir::State::status="$STATUS_FILE" \
        -o Dir::State::extended_states="$STATE_FILE" \
        -o Dir::Cache::pkgcache= \
        -o Dir::Cache::srcpkgcache= \
        -o Dir::Etc::sourcelist=/dev/null \
        -o Dir::Etc::sourceparts=- \
        "$@"
}
purge_apt_build_deps_exact libibverbs-dev librdmacm-dev
''',
        env={
            "REAL_APT_GET": apt_get,
            "STATUS_FILE": str(status),
            "STATE_FILE": str(tmp_path / "extended_states"),
        },
        check=False,
    )
    assert result.returncode != 0
    assert "refusing build-package purge that also removes doca-base" in result.stderr
    assert "Unexpected attempt to mutate packages" not in result.stderr


def test_rocm_device_bitcode_is_reachable_from_explicit_core_root(tmp_path):
    core = tmp_path / "core"
    compat = tmp_path / "compat"
    bitcode = core / "lib" / "llvm" / "amdgcn" / "bitcode"
    bitcode.mkdir(parents=True)
    compat.mkdir()
    for name in ("ocml.bc", "ockl.bc"):
        (bitcode / name).write_text(name)
    run_bash(
        source_rocm('link_rocm_device_libraries "$COMPAT_ROOT"; link_rocm_device_libraries "$COMPAT_ROOT"'),
        env={"ROCM_CORE_PREFIX": str(core), "COMPAT_ROOT": str(compat)},
    )
    # ROCM_PATH explicitly selects the core prefix, not the compatibility root.
    for prefix in (core, compat):
        for name in ("ocml.bc", "ockl.bc"):
            discovered = prefix / "amdgcn" / "bitcode" / name
            assert discovered.samefile(bitcode / name)
            assert discovered.read_text() == name


@pytest.mark.parametrize("missing", ["ocml.bc", "ockl.bc"])
def test_rocm_device_layout_rejects_incomplete_bitcode_without_partial_links(tmp_path, missing):
    core = tmp_path / "core"
    compat = tmp_path / "compat"
    bitcode = core / "lib" / "llvm" / "amdgcn" / "bitcode"
    bitcode.mkdir(parents=True)
    compat.mkdir()
    for name in ("ocml.bc", "ockl.bc"):
        if name != missing:
            (bitcode / name).write_text(name)
    result = run_bash(
        source_rocm('link_rocm_device_libraries "$COMPAT_ROOT"'),
        env={"ROCM_CORE_PREFIX": str(core), "COMPAT_ROOT": str(compat)},
        check=False,
    )
    assert result.returncode != 0
    assert str(bitcode) in result.stderr
    assert not (core / "amdgcn").is_symlink()
    assert not (compat / "amdgcn").is_symlink()


def test_rocm_device_layout_preserves_existing_vendor_directory(tmp_path):
    core = tmp_path / "core"
    compat = tmp_path / "compat"
    bitcode = core / "lib" / "llvm" / "amdgcn" / "bitcode"
    bitcode.mkdir(parents=True)
    compat.mkdir()
    for name in ("ocml.bc", "ockl.bc"):
        (bitcode / name).write_text(name)
    vendor = core / "amdgcn"
    vendor.mkdir()
    sentinel = vendor / "vendor-owned"
    sentinel.write_text("preserve")
    result = run_bash(
        source_rocm('link_rocm_device_libraries "$COMPAT_ROOT"'),
        env={"ROCM_CORE_PREFIX": str(core), "COMPAT_ROOT": str(compat)},
        check=False,
    )
    assert result.returncode != 0
    assert sentinel.read_text() == "preserve"
    assert not vendor.is_symlink()


@pytest.mark.parametrize("devel_executable", [True, False])
def test_rocm_hip_compiler_prefers_devel_and_falls_back_to_wheel(tmp_path, devel_executable):
    sdk_bin = tmp_path / "sdk" / "bin"
    wheel_bin = tmp_path / "core" / "lib" / "llvm" / "bin"
    sdk_bin.mkdir(parents=True)
    wheel_bin.mkdir(parents=True)
    devel = sdk_bin / "amdclang++"
    wheel = wheel_bin / "clang++"
    for compiler in (devel, wheel):
        compiler.write_text("#!/bin/sh\nexit 0\n")
        compiler.chmod(0o755)
    if not devel_executable:
        devel.chmod(0o644)
    result = run_bash(
        source_rocm("rocm_hip_cxx_compiler"),
        env={"ROCM_SDK_BIN": str(sdk_bin), "ROCM_CORE_PREFIX": str(tmp_path / "core")},
    )
    assert Path(result.stdout.strip()) == (devel if devel_executable else wheel)


def test_rocm_hip_compiler_rejects_missing_sdk_tools(tmp_path):
    result = run_bash(
        source_rocm("rocm_hip_cxx_compiler"),
        env={"ROCM_SDK_BIN": str(tmp_path / "sdk" / "bin"), "ROCM_CORE_PREFIX": str(tmp_path / "core")},
        check=False,
    )
    assert result.returncode != 0
    assert "No HIP C++ compiler found in ROCm SDK" in result.stderr
    assert str(tmp_path / "core" / "lib" / "llvm" / "bin") in result.stderr
