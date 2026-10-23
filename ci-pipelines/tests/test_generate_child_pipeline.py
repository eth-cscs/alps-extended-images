import importlib.util
import re
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "ci-pipelines" / "helpers" / "generate-child-pipeline.py"


@pytest.fixture()
def gen(monkeypatch):
    spec = importlib.util.spec_from_file_location("generate_child_pipeline", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.DIGEST_CACHE.clear()
    monkeypatch.setattr(module, "ROOT", ROOT)
    return module


def fake_registry(monkeypatch, gen, digests):
    monkeypatch.setattr(gen, "img_digest", lambda ref: digests.get(ref, ""))


def base_image(**overrides):
    data = {
        "FAMILY": "cuda",
        "NAME": "pytorch",
        "VARIANT": "26.06-py3",
        "DOCKERFILE": "Alps-Images/NGC/Containerfile.ngc-alps",
        "CANON_IMAGE_REF": "registry/pytorch-cuda:canon",
        "TESTED_IMAGE_REF": "registry/pytorch-cuda:canon-tested-hash",
        "STABLE_IMAGE_REF": "registry/pytorch-cuda:stable",
        "GHCR_STABLE_IMAGE_REF": "ghcr/pytorch-cuda:stable",
        "BASE_IMAGE": "registry/base:pytorch",
        "NGC_VARIANT_DIR": "pytorch-26.06-py3",
    }
    data.update(overrides)
    return data


def app_image(**overrides):
    data = {
        "FAMILY": "cuda",
        "VARIANT": "cuda",
        "NAME": "vllm",
        "DOCKERFILE": "Alps-Images/apps/vllm/Containerfile",
        "CANON_IMAGE_REF": "registry/vllm-cuda:canon",
        "TESTED_IMAGE_REF": "registry/vllm-cuda:canon-tested-hash",
        "STABLE_IMAGE_REF": "registry/vllm-cuda:stable",
        "GHCR_STABLE_IMAGE_REF": "ghcr/vllm-cuda:stable",
        "BASE_IMAGE": "registry/pytorch-cuda:canon",
        "OCI_SOURCE": "https://example.invalid/repo.git",
        "OCI_REVISION": "abc123",
        "OCI_CREATED": "2026-08-17T00:00:00Z",
        "OCI_DESCRIPTION": "test image",
        "CSCS_ALPS_GIT_COMMIT_SHORT": "abc123",
    }
    data.update(overrides)
    return data


def base_tests():
    return [{"name": "env"}]


def app_tests():
    return [{"name": "vetnode", "kind": "cuda-vetnode"}]


def job_names(child):
    return {name for name, value in child.data.items() if isinstance(value, dict) and not name.startswith(".")}


def test_generate_no_work_emits_noop(tmp_path, monkeypatch, gen):
    output = tmp_path / "child.yaml"
    monkeypatch.setattr(gen, "OUTPUT", output)
    child = gen.Child()
    child.write()

    data = yaml.safe_load(output.read_text())
    assert data["no-work-required"] == {"extends": [".child-noop-template"]}


def test_child_write_does_not_fold_long_values(tmp_path, monkeypatch, gen):
    output = tmp_path / "child.yaml"
    long_value = " ".join(["value"] * 80)
    monkeypatch.setattr(gen, "OUTPUT", output)
    child = gen.Child()
    child.add_job("long-value-job", ".template", variables={"LONG_VALUE": long_value})

    child.write()

    generated = output.read_text()
    assert f"LONG_VALUE: {long_value}" in generated
    assert yaml.safe_load(generated)["long-value-job"]["variables"]["LONG_VALUE"] == long_value


def test_main_generates_base_and_app_graph(tmp_path, monkeypatch, gen):
    output = tmp_path / "child.yaml"
    base = base_image()
    app = app_image(BASE_IMAGE=base["CANON_IMAGE_REF"])
    monkeypatch.setattr(gen, "OUTPUT", output)
    monkeypatch.setattr(gen, "discover_bases", lambda: [(base, base_tests())])
    monkeypatch.setattr(gen, "discover_apps", lambda: [(app, app_tests())])
    fake_registry(monkeypatch, gen, {})

    gen.main()

    data = yaml.safe_load(output.read_text())
    assert "build-base-pytorch-cuda-26-06-py3" in data
    assert data["mark-base-pytorch-cuda-26-06-py3-tested"]["needs"] == ["test-base-pytorch-cuda-26-06-py3-env"]
    assert data["build-app-vllm-cuda"]["needs"] == ["mark-base-pytorch-cuda-26-06-py3-tested"]
    assert data["mark-app-vllm-cuda-tested"]["needs"] == ["test-app-vllm-cuda-vetnode"]
    assert data["publish-app-vllm-cuda"]["needs"] == ["mark-app-vllm-cuda-tested"]


def test_slug_rejects_empty_result(gen):
    with pytest.raises(RuntimeError, match="slug produced empty string"):
        gen.slug("___")


def test_app_variants_from_profile_parses_without_executing(gen, tmp_path):
    profile = tmp_path / "profile.env"
    marker = tmp_path / "executed"
    profile.write_text(
        'export APP_VARIANTS="cuda rocm" # supported targets\n'
        'CUDA_BASE_IMAGE=pytorch-cuda:26.06-py3\n'
        'ROCM_BASE_IMAGE=pytorch-rocm:rocm10.0-ubuntu24.04-py3.12-torch2.13\n'
        f'touch {marker}\n'
    )

    assert gen.app_variants_from_profile(profile) == ["cuda", "rocm"]
    assert not marker.exists()


@pytest.mark.parametrize("profile_text", ["", 'APP_VARIANTS="   "\n'])
def test_app_variants_rejects_missing_or_empty_target_list(gen, tmp_path, profile_text):
    profile = tmp_path / "profile.env"
    profile.write_text(profile_text)

    with pytest.raises(RuntimeError):
        gen.app_variants_from_profile(profile)


@pytest.mark.parametrize(
    "variants",
    ["megatron-base-cuda megatron_base-cuda", "megatron-cuda MEGATRON-cuda", "megatron-cuda megatron-cuda"],
)
def test_app_variants_rejects_profile_prefix_collisions(gen, tmp_path, variants):
    profile = tmp_path / "profile.env"
    profile.write_text(f'APP_VARIANTS="{variants}"\n')

    with pytest.raises(RuntimeError, match="share profile prefix"):
        gen.app_variants_from_profile(profile)


@pytest.mark.parametrize(
    "target",
    ["megatron", "megatron-base", "megatron-tpu", "cuda-debug", "-cuda", "megatron.rocm", "2stage-cuda"],
)
def test_app_variants_rejects_invalid_target_name(gen, tmp_path, target):
    profile = tmp_path / "profile.env"
    profile.write_text(f'APP_VARIANTS="{target}"\n')

    with pytest.raises(RuntimeError) as error:
        gen.app_variants_from_profile(profile)

    assert target in str(error.value)


def test_app_variants_accepts_family_only_and_multi_part_targets(gen, tmp_path):
    profile = tmp_path / "profile.env"
    variants = ["cuda", "rocm", "_team-rocm-stage-cuda", "team-cuda-stage-rocm"]
    lines = [f'APP_VARIANTS="{" ".join(variants)}"']
    for target in variants:
        prefix = target.upper().replace("-", "_")
        family = target.rsplit("-", 1)[-1]
        lines.append(f"{prefix}_BASE_IMAGE=pytorch-{family}:test")
    profile.write_text("\n".join(lines) + "\n")

    assert gen.app_variants_from_profile(profile) == variants


def test_generate_missing_base_emits_build_tests_marker_publish(monkeypatch, gen):
    base = base_image()
    fake_registry(monkeypatch, gen, {})
    child = gen.Child()

    mark = gen.add_base(child, base, base_tests(), valid=False)

    assert mark == "mark-base-pytorch-cuda-26-06-py3-tested"
    assert "build-base-pytorch-cuda-26-06-py3" in child.data
    assert "OCI_CREATED" not in child.data["build-base-pytorch-cuda-26-06-py3"]["variables"]
    assert child.data["test-base-pytorch-cuda-26-06-py3-env"]["needs"] == ["build-base-pytorch-cuda-26-06-py3"]
    assert child.data[mark]["needs"] == ["test-base-pytorch-cuda-26-06-py3-env"]
    assert child.data["publish-base-pytorch-cuda-26-06-py3"]["needs"] == [mark]


@pytest.mark.parametrize("kind", ["base", "app"])
def test_build_lock_tracks_full_canonical_ref_not_job_or_commit(monkeypatch, gen, kind):
    fake_registry(monkeypatch, gen, {})

    def lock_for(ref, name="original", revision="first"):
        child = gen.Child()
        if kind == "base":
            base = base_image(NAME=name, CANON_IMAGE_REF=ref, OCI_REVISION=revision)
            gen.add_base(child, base, base_tests(), valid=False)
        else:
            app = app_image(NAME=name, CANON_IMAGE_REF=ref, OCI_REVISION=revision)
            gen.add_app(child, app, app_tests(), {app["BASE_IMAGE"]: None}, {app["BASE_IMAGE"]: True})
        build = next(job for name, job in child.data.items() if name.startswith("build-"))
        return build["resource_group"]

    canonical = "registry.example:5000/ns/image:content-one"
    lock = lock_for(canonical)
    assert lock_for(canonical, name="another-job", revision="another-commit") == lock
    assert re.fullmatch(r"[a-z0-9-]+", lock)
    for different_ref in (
        "registry.example:5000/ns/image:content-two",
        "registry.example:5000/ns/image:content_one",
        "registry.example:5000/ns/other-image:content-one",
        "other.example:5000/ns/image:content-one",
    ):
        assert lock_for(different_ref) != lock


def test_generate_existing_base_missing_marker_emits_tests_marker_publish(monkeypatch, gen):
    base = base_image()
    fake_registry(
        monkeypatch,
        gen,
        {
            base["CANON_IMAGE_REF"]: "sha256:base",
            base["STABLE_IMAGE_REF"]: "sha256:base",
            base["GHCR_STABLE_IMAGE_REF"]: "sha256:base",
        },
    )
    child = gen.Child()

    mark = gen.add_base(child, base, base_tests(), valid=False)

    assert "build-base-pytorch-cuda-26-06-py3" not in child.data
    assert "test-base-pytorch-cuda-26-06-py3-env" in child.data
    assert child.data[mark]["needs"] == ["test-base-pytorch-cuda-26-06-py3-env"]
    assert child.data["publish-base-pytorch-cuda-26-06-py3"]["needs"] == [mark]


def test_generate_valid_base_with_stale_stable_emits_publish_only(monkeypatch, gen):
    base = base_image()
    fake_registry(
        monkeypatch,
        gen,
        {
            base["CANON_IMAGE_REF"]: "sha256:base",
            base["TESTED_IMAGE_REF"]: "sha256:base",
            base["STABLE_IMAGE_REF"]: "sha256:old",
            base["GHCR_STABLE_IMAGE_REF"]: "sha256:base",
        },
    )
    child = gen.Child()

    assert gen.add_base(child, base, base_tests(), valid=True) is None

    assert set(child.data) >= {"publish-base-pytorch-cuda-26-06-py3"}
    assert child.data["publish-base-pytorch-cuda-26-06-py3"]["needs"] == []
    assert "test-base-pytorch-cuda-26-06-py3-env" not in child.data
    assert "mark-base-pytorch-cuda-26-06-py3-tested" not in child.data


def test_stable_publish_check_false_when_canonical_missing(monkeypatch, gen):
    base = base_image()
    fake_registry(monkeypatch, gen, {})

    assert not gen.stable_refs_need_publish_for_existing_canonical(base)


def test_tested_marker_check_false_when_canonical_missing(monkeypatch, gen):
    base = base_image()
    fake_registry(monkeypatch, gen, {base["TESTED_IMAGE_REF"]: "sha256:base"})

    assert not gen.tested_marker_matches_existing_canonical(base["CANON_IMAGE_REF"], base["TESTED_IMAGE_REF"])


def test_add_publish_respects_force(monkeypatch, gen):
    base = base_image()
    fake_registry(
        monkeypatch,
        gen,
        {
            base["CANON_IMAGE_REF"]: "sha256:base",
            base["STABLE_IMAGE_REF"]: "sha256:base",
            base["GHCR_STABLE_IMAGE_REF"]: "sha256:base",
        },
    )
    child = gen.Child()

    gen.add_publish(child, base, [], "base-pytorch", force=False)
    assert "publish-base-pytorch" not in child.data

    gen.add_publish(child, base, [], "base-pytorch", force=True)
    assert child.data["publish-base-pytorch"]["needs"] == []


def test_generate_missing_app_emits_build_tests_marker_publish(monkeypatch, gen):
    app = app_image()
    fake_registry(monkeypatch, gen, {})
    child = gen.Child()

    gen.add_app(child, app, app_tests(), {app["BASE_IMAGE"]: None}, {app["BASE_IMAGE"]: True})

    assert "build-app-vllm-cuda" in child.data
    assert "OCI_CREATED" not in child.data["build-app-vllm-cuda"]["variables"]
    assert child.data["test-app-vllm-cuda-vetnode"]["needs"] == ["build-app-vllm-cuda"]
    assert child.data["mark-app-vllm-cuda-tested"]["needs"] == ["test-app-vllm-cuda-vetnode"]
    assert child.data["publish-app-vllm-cuda"]["needs"] == ["mark-app-vllm-cuda-tested"]


def test_generate_app_waits_for_base_marker_when_base_revalidated(monkeypatch, gen):
    app = app_image()
    fake_registry(monkeypatch, gen, {})
    child = gen.Child()

    gen.add_app(
        child,
        app,
        app_tests(),
        {app["BASE_IMAGE"]: "mark-base-pytorch-cuda-tested"},
        {app["BASE_IMAGE"]: False},
    )

    assert child.data["build-app-vllm-cuda"]["needs"] == ["mark-base-pytorch-cuda-tested"]


def test_generate_app_rejects_missing_base_metadata(monkeypatch, gen):
    app = app_image(BASE_IMAGE="registry/missing-base:canon")
    fake_registry(monkeypatch, gen, {})
    child = gen.Child()

    with pytest.raises(RuntimeError, match="base metadata missing"):
        gen.add_app(child, app, app_tests(), {}, {})


def test_generate_custom_app_test_propagates_fields(monkeypatch, gen):
    app = app_image()
    fake_registry(monkeypatch, gen, {})
    child = gen.Child()
    tests = [
        {
            "name": "ray-smoke",
            "runner": "rocm",
            "timeout": "30m",
            "variables": {"SLURM_NTASKS": "2"},
            "script": ["/opt/tests/vllm/ray_nccl_smoke.sh"],
        }
    ]

    test_job = gen.add_app_test(child, app, tests[0], ["build-app-vllm-cuda"])

    assert test_job == "test-app-vllm-cuda-ray-smoke"
    assert child.data[test_job]["extends"] == [".child-rocm-custom-test-template"]
    assert child.data[test_job]["timeout"] == "30m"
    assert child.data[test_job]["variables"] == {"SLURM_NTASKS": "2"}
    assert child.data[test_job]["script"] == ["/opt/tests/vllm/ray_nccl_smoke.sh"]


def test_generate_app_has_no_base_need_when_base_already_valid(monkeypatch, gen):
    app = app_image()
    fake_registry(monkeypatch, gen, {})
    child = gen.Child()

    gen.add_app(child, app, app_tests(), {app["BASE_IMAGE"]: None}, {app["BASE_IMAGE"]: True})

    assert child.data["build-app-vllm-cuda"]["needs"] == []


def test_generate_valid_app_with_stale_stable_emits_immediate_publish(monkeypatch, gen):
    app = app_image()
    fake_registry(
        monkeypatch,
        gen,
        {
            app["CANON_IMAGE_REF"]: "sha256:app",
            app["TESTED_IMAGE_REF"]: "sha256:app",
            app["STABLE_IMAGE_REF"]: "sha256:old",
            app["GHCR_STABLE_IMAGE_REF"]: "sha256:app",
        },
    )
    child = gen.Child()

    gen.add_app(child, app, app_tests(), {app["BASE_IMAGE"]: None}, {app["BASE_IMAGE"]: True})

    assert child.data["publish-app-vllm-cuda"]["needs"] == []


def test_generate_marker_mismatch_fails_closed(monkeypatch, gen):
    base = base_image()
    fake_registry(monkeypatch, gen, {base["CANON_IMAGE_REF"]: "sha256:new", base["TESTED_IMAGE_REF"]: "sha256:old"})

    with pytest.raises(RuntimeError, match="tested marker digest mismatch"):
        gen.tested_marker_matches_existing_canonical(base["CANON_IMAGE_REF"], base["TESTED_IMAGE_REF"])


def test_generate_needs_reference_existing_jobs(monkeypatch, gen):
    base = base_image()
    app = app_image(BASE_IMAGE=base["CANON_IMAGE_REF"])
    fake_registry(monkeypatch, gen, {})
    child = gen.Child()
    base_mark = gen.add_base(child, base, base_tests(), valid=False)
    gen.add_app(child, app, app_tests(), {base["CANON_IMAGE_REF"]: base_mark}, {base["CANON_IMAGE_REF"]: False})

    names = job_names(child)
    for name in names:
        for need in child.data[name].get("needs", []):
            assert need in names


def test_generate_jobs_use_expected_templates(monkeypatch, gen):
    base = base_image()
    fake_registry(monkeypatch, gen, {})
    child = gen.Child()

    gen.add_base(child, base, base_tests(), valid=False)

    assert child.data["build-base-pytorch-cuda-26-06-py3"]["extends"] == [".child-cuda-base-build-template"]
    assert child.data["test-base-pytorch-cuda-26-06-py3-env"]["extends"] == [".child-cuda-base-env-test-template"]
    assert child.data["mark-base-pytorch-cuda-26-06-py3-tested"]["extends"] == [".child-mark-base-tested-template"]
    assert child.data["publish-base-pytorch-cuda-26-06-py3"]["extends"] == [".child-publish-template"]


def test_schema_base_rejects_empty_tests(gen):
    with pytest.raises(RuntimeError, match="tests must not be empty"):
        gen.validate_base_ci(Path("base.yaml"), {"family": "cuda", "variants": [{"name": "pytorch", "variant": "x"}], "tests": []})


def test_schema_base_rejects_rocm_vetnode(gen):
    with pytest.raises(RuntimeError, match="unsupported rocm base test name"):
        gen.validate_base_ci(
            Path("base.yaml"),
            {"family": "rocm", "variants": [{"name": "pytorch", "variant": "x"}], "tests": [{"name": "vetnode"}]},
        )


def test_schema_app_rejects_empty_variant_tests(gen):
    with pytest.raises(RuntimeError, match="tests.cuda must not be empty"):
        gen.validate_app_ci(Path("app.yaml"), ["cuda"], {"tests": {"cuda": []}}, {"cuda": "cuda"})


def test_schema_app_rejects_custom_test_without_script(gen):
    with pytest.raises(RuntimeError, match="script must be a list"):
        gen.validate_app_ci(
            Path("app.yaml"), ["cuda"], {"tests": {"cuda": [{"name": "smoke", "runner": "cuda"}]}}, {"cuda": "cuda"}
        )


def test_schema_app_rejects_unknown_runner(gen):
    with pytest.raises(RuntimeError, match="unsupported app test runner"):
        gen.validate_app_ci(
            Path("app.yaml"),
            ["cuda"],
            {"tests": {"cuda": [{"name": "smoke", "runner": "unknown", "script": ["true"]}]}},
            {"cuda": "cuda"},
        )


def test_schema_app_rejects_extra_cuda_vetnode_fields(gen):
    with pytest.raises(RuntimeError, match="cuda-vetnode app tests do not support: script, timeout"):
        gen.validate_app_ci(
            Path("app.yaml"),
            ["cuda"],
            {"tests": {"cuda": [{"name": "vetnode", "kind": "cuda-vetnode", "timeout": "1h", "script": ["true"]}]}},
            {"cuda": "cuda"},
        )


def test_schema_app_accepts_explicit_custom_kind(gen):
    gen.validate_app_ci(
        Path("app.yaml"),
        ["cuda"],
        {"tests": {"cuda": [{"name": "smoke", "kind": "custom", "runner": "cuda", "script": ["true"]}]}},
        {"cuda": "cuda"},
    )


def test_schema_app_requires_base_derived_family(gen):
    with pytest.raises(KeyError):
        gen.validate_app_ci(
            Path("app.yaml"),
            ["cuda"],
            {"tests": {"cuda": [{"name": "smoke", "runner": "cuda", "script": ["true"]}]}},
            {},
        )


@pytest.fixture()
def named_targets(tmp_path, monkeypatch, gen):
    for key, value in {
        "IMAGE_PREFIX": "registry/alps",
        "GHCR_IMAGE_PREFIX": "ghcr/alps",
        "ALPS_REV": "alps7-dev",
        "CI_COMMIT_SHORT_SHA": "abc123",
        "CI_COMMIT_SHA": "abc123",
        "CSCS_CI_ORIG_CLONE_URL": "https://example.invalid/repo.git",
    }.items():
        monkeypatch.setenv(key, value)
    (tmp_path / "ci-pipelines").symlink_to(ROOT / "ci-pipelines", target_is_directory=True)
    images = tmp_path / "Alps-Images"
    images.mkdir()
    for name in ("NGC", "ROCm", "common", "patches"):
        (images / name).symlink_to(ROOT / "Alps-Images" / name, target_is_directory=True)
    app = images / "apps" / "apertus-2"
    app.mkdir(parents=True)
    bases = {
        "megatron-base-cuda": ("cuda", "25.12-py3"),
        "megatron-cuda": ("cuda", "26.06-py3"),
        "megatron-rocm": ("rocm", "rocm10.0-ubuntu24.04-py3.12-torch2.13"),
    }
    paths = {
        "megatron-base-cuda": ("Containerfile.megatron-base", "tests/megatron-base"),
        "megatron-cuda": ("Containerfile.megatron", "tests/megatron"),
        "megatron-rocm": ("Containerfile.megatron.rocm", "tests/rocm"),
    }
    profile = [f'APP_VARIANTS="{" ".join(bases)}"']
    tests = {}
    for target, (family, version) in bases.items():
        prefix = target.upper().replace("-", "_")
        base = f"pytorch-{family}"
        recipe, test_dir = paths[target]
        profile.extend([
            f"{prefix}_BASE_IMAGE={base}:{version}",
            f"{prefix}_CONTAINERFILE={recipe}",
            f"{prefix}_TEST_DIR={test_dir}",
        ])
        (app / recipe).write_text("ARG BASE_IMAGE\nFROM ${BASE_IMAGE}\n")
        (app / test_dir).mkdir(parents=True)
        (app / test_dir / "check-stack.py").write_text("assert True\n")
        tests[target] = (
            [{"name": "vetnode", "kind": "cuda-vetnode"}]
            if family == "cuda"
            else [{"name": "stack", "script": ["/opt/tests/apertus-2/rocm/check-stack.py"]}]
        )
    (app / "profile.env").write_text("\n".join(profile) + "\n")
    (app / "ci.yaml").write_text(yaml.safe_dump({"tests": tests}))
    monkeypatch.setattr(gen, "ROOT", tmp_path)
    return app, bases


def test_named_targets_use_selected_base_and_distinct_jobs(monkeypatch, gen, named_targets):
    _, bases = named_targets
    fake_registry(monkeypatch, gen, {})
    child = gen.Child()
    base_marks = {}
    base_valid = {}
    expected_bases = {}
    for family, version in set(bases.values()):
        base = gen.base_env(family, "pytorch", version)
        ref = base["CANON_IMAGE_REF"]
        base_marks[ref] = gen.add_base(child, base, base_tests(), valid=False)
        base_valid[ref] = False
        expected_bases[(family, version)] = ref

    apps = gen.discover_apps()
    assert {data["VARIANT"] for data, _ in apps} == set(bases)
    for app, tests in apps:
        target = app["VARIANT"]
        family, version = bases[target]
        base_ref = expected_bases[(family, version)]
        assert app["FAMILY"] == family
        assert app["BASE_IMAGE"] == base_ref
        recipe = {
            "megatron-base-cuda": "Containerfile.megatron-base",
            "megatron-cuda": "Containerfile.megatron",
            "megatron-rocm": "Containerfile.megatron.rocm",
        }[target]
        assert app["DOCKERFILE"] == f"Alps-Images/apps/apertus-2/{recipe}"
        assert f"/apertus-2-{target}:alps7-dev-" in app["CANON_IMAGE_REF"]
        gen.add_app(child, app, tests, base_marks, base_valid)

        prefix = f"app-apertus-2-{target}"
        test_job = f"test-{prefix}-{tests[0]['name']}"
        build = child.data[f"build-{prefix}"]
        assert build["variables"]["BASE_IMAGE"] == base_ref
        assert build["extends"] == [f".child-{family}-app-build-template"]
        assert build["needs"] == [base_marks[base_ref]]
        assert child.data[test_job]["image"] == app["CANON_IMAGE_REF"]
        assert child.data[test_job]["needs"] == [f"build-{prefix}"]
        template = ".child-cuda-app-vetnode-test-template" if family == "cuda" else ".child-rocm-custom-test-template"
        assert child.data[test_job]["extends"] == [template]
        assert child.data[f"mark-{prefix}-tested"]["needs"] == [test_job]
        assert child.data[f"publish-{prefix}"]["needs"] == [f"mark-{prefix}-tested"]


def test_discovery_rejects_target_base_family_mismatch(gen, named_targets):
    app, _ = named_targets
    profile = app / "profile.env"
    wrong_base = "pytorch-cuda:26.06-py3"
    profile.write_text(
        profile.read_text().replace(
            "MEGATRON_ROCM_BASE_IMAGE=pytorch-rocm:rocm10.0-ubuntu24.04-py3.12-torch2.13",
            f"MEGATRON_ROCM_BASE_IMAGE={wrong_base}",
        )
    )

    with pytest.raises(RuntimeError) as error:
        gen.discover_apps()

    assert "megatron-rocm" in str(error.value)
    assert wrong_base in str(error.value)


@pytest.mark.parametrize("runner", [None, "cuda"])
def test_discovery_rejects_cuda_vetnode_on_named_rocm_target(gen, named_targets, runner):
    app, _ = named_targets
    cfg = gen.load_yaml(app / "ci.yaml")
    test = {"name": "vetnode", "kind": "cuda-vetnode"}
    if runner is not None:
        test["runner"] = runner
    cfg["tests"]["megatron-rocm"] = [test]
    (app / "ci.yaml").write_text(yaml.safe_dump(cfg))

    with pytest.raises(RuntimeError, match="only supported for cuda"):
        gen.discover_apps()


def test_cuda_vetnode_rejects_a_named_rocm_target(gen):
    app = app_image(FAMILY="rocm", VARIANT="megatron-rocm")
    test = {"name": "vetnode", "kind": "cuda-vetnode", "runner": "cuda"}

    with pytest.raises(RuntimeError, match="requires a CUDA base"):
        gen.add_app_test(gen.Child(), app, test, [])


def test_schema_current_repo_metadata_is_valid(gen, monkeypatch):
    for key, value in {
        "IMAGE_PREFIX": "registry/alps",
        "GHCR_IMAGE_PREFIX": "ghcr/alps",
        "ALPS_REV": "alps7-dev",
        "CI_COMMIT_SHORT_SHA": "abc123",
        "CSCS_CI_ORIG_CLONE_URL": "https://example.invalid/repo.git",
    }.items():
        monkeypatch.setenv(key, value)
    for path in [ROOT / "Alps-Images" / "NGC" / "ci.yaml", ROOT / "Alps-Images" / "ROCm" / "ci.yaml"]:
        gen.validate_base_ci(path, gen.load_yaml(path))
    for path in sorted((ROOT / "Alps-Images" / "apps").glob("*/ci.yaml")):
        variants = gen.app_variants_from_profile(path.parent / "profile.env")
        families = {variant: gen.app_env(path.parent.name, variant)["FAMILY"] for variant in variants}
        gen.validate_app_ci(path, variants, gen.load_yaml(path), families)
