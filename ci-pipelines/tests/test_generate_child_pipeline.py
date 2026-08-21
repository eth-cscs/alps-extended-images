import importlib.util
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
    profile.write_text(f'APP_VARIANTS="cuda rocm"\ntouch {marker}\n')

    assert gen.app_variants_from_profile(profile) == ["cuda", "rocm"]
    assert not marker.exists()


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
        gen.validate_app_ci(Path("app.yaml"), ["cuda"], {"tests": {"cuda": []}})


def test_schema_app_rejects_custom_test_without_script(gen):
    with pytest.raises(RuntimeError, match="script must be a list"):
        gen.validate_app_ci(Path("app.yaml"), ["cuda"], {"tests": {"cuda": [{"name": "smoke", "runner": "cuda"}]}})


def test_schema_app_rejects_unknown_runner(gen):
    with pytest.raises(RuntimeError, match="unsupported app test runner"):
        gen.validate_app_ci(
            Path("app.yaml"),
            ["cuda"],
            {"tests": {"cuda": [{"name": "smoke", "runner": "unknown", "script": ["true"]}]}},
        )


def test_schema_app_rejects_extra_cuda_vetnode_fields(gen):
    with pytest.raises(RuntimeError, match="cuda-vetnode app tests do not support: script, timeout"):
        gen.validate_app_ci(
            Path("app.yaml"),
            ["cuda"],
            {"tests": {"cuda": [{"name": "vetnode", "kind": "cuda-vetnode", "timeout": "1h", "script": ["true"]}]}},
        )


def test_schema_app_accepts_explicit_custom_kind(gen):
    gen.validate_app_ci(
        Path("app.yaml"),
        ["cuda"],
        {"tests": {"cuda": [{"name": "smoke", "kind": "custom", "runner": "cuda", "script": ["true"]}]}},
    )


def test_schema_current_repo_metadata_is_valid(gen):
    for path in [ROOT / "Alps-Images" / "NGC" / "ci.yaml", ROOT / "Alps-Images" / "ROCm" / "ci.yaml"]:
        gen.validate_base_ci(path, gen.load_yaml(path))
    for path in sorted((ROOT / "Alps-Images" / "apps").glob("*/ci.yaml")):
        variants = gen.app_variants_from_profile(path.parent / "profile.env")
        gen.validate_app_ci(path, variants, gen.load_yaml(path))
