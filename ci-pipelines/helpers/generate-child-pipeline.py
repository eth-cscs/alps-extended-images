#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "generated-child-pipeline.yaml"
DIGEST_CACHE: dict[str, str] = {}


def run_bash(script: str) -> str:
    proc = subprocess.run(
        ["bash", "-lc", script],
        cwd=ROOT,
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"command failed:\n{script}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    return proc.stdout.strip()


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise RuntimeError(f"expected mapping in {path}")
    return data


def load_env_text(text: str) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in text.splitlines():
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            data[key] = value
    return data


def source_profile_value(profile: Path, key: str) -> str:
    return run_bash(f"source {shell_quote(str(profile))} && printf '%s\\n' \"${{{key}:-}}\"")


def img_digest(ref: str) -> str:
    if ref not in DIGEST_CACHE:
        DIGEST_CACHE[ref] = run_bash(f"source ci-pipelines/helpers/skopeo.sh && img_digest {shell_quote(ref)}")
    return DIGEST_CACHE[ref]


def marker_valid(canon_ref: str, tested_ref: str) -> bool:
    canon_digest = img_digest(canon_ref)
    marker_digest = img_digest(tested_ref)
    if marker_digest and canon_digest and marker_digest != canon_digest:
        raise RuntimeError(
            f"tested marker digest mismatch for {tested_ref}: canonical={canon_digest} marker={marker_digest}"
        )
    return bool(canon_digest and marker_digest and canon_digest == marker_digest)


def publish_needed(image: dict[str, str]) -> bool:
    canon_digest = img_digest(image["CANON_IMAGE_REF"])
    if not canon_digest:
        return False
    return img_digest(image["STABLE_IMAGE_REF"]) != canon_digest or img_digest(image["GHCR_STABLE_IMAGE_REF"]) != canon_digest


def base_env(family: str, name: str, variant: str) -> dict[str, str]:
    out = run_bash(
        "source ci-pipelines/helpers/skopeo.sh && "
        "source ci-pipelines/helpers/meta.sh && "
        f"tmp=$(mktemp) && write_base_build_env \"$tmp\" {shell_quote(family)} {shell_quote(name)} {shell_quote(variant)} && cat \"$tmp\" && rm -f \"$tmp\""
    )
    return load_env_text(out)


def app_env(app: str, variant: str) -> dict[str, str]:
    out = run_bash(
        "source ci-pipelines/helpers/skopeo.sh && "
        "source ci-pipelines/helpers/meta.sh && "
        f"tmp=$(mktemp) && write_app_build_env \"$tmp\" {shell_quote(app)} {shell_quote(variant)} && cat \"$tmp\" && rm -f \"$tmp\""
    )
    data = load_env_text(out)
    data.update({"FAMILY": variant, "NAME": app, "VARIANT": variant})
    return data


class Child:
    def __init__(self) -> None:
        self.data: dict[str, Any] = {
            "include": [{"local": "ci-pipelines/child-templates.yaml"}],
            "stages": ["build-base", "test-base", "mark-base-tested", "build-apps", "test-apps", "mark-app-tested", "publish"],
            "variables": {
                "IMAGE_PREFIX": os.environ.get("IMAGE_PREFIX", ""),
                "GHCR_IMAGE_PREFIX": os.environ.get("GHCR_IMAGE_PREFIX", ""),
                "ALPS_REV": os.environ.get("ALPS_REV", ""),
            },
        }
        self.jobs: set[str] = set()

    def add_job(
        self,
        name: str,
        extends: str,
        image: str | None = None,
        needs: list[str] | None = None,
        variables: dict[str, str] | None = None,
        timeout: str | None = None,
        script: list[str] | None = None,
    ) -> None:
        if name in self.jobs:
            raise RuntimeError(f"duplicate generated job: {name}")
        self.jobs.add(name)
        job: dict[str, Any] = {"extends": [extends]}
        if image:
            job["image"] = image
        if timeout:
            job["timeout"] = timeout
        if needs:
            job["needs"] = needs
        if variables:
            job["variables"] = {key: str(value) for key, value in sorted(variables.items())}
        if script:
            job["script"] = script
        self.data[name] = job

    def add_noop(self) -> None:
        self.data["no-work-required"] = {"extends": [".child-noop-template"]}

    def write(self) -> None:
        if not self.jobs:
            self.add_noop()
        OUTPUT.write_text(yaml.safe_dump(self.data, sort_keys=False, width=120))


BASE_TEST_TEMPLATES = {
    "cuda-base-env": ".child-cuda-base-env-test-template",
    "cuda-base-collectives": ".child-cuda-base-collectives-test-template",
    "cuda-vetnode": ".child-cuda-vetnode-test-template",
    "rocm-base-env": ".child-rocm-base-env-test-template",
    "rocm-base-collectives": ".child-rocm-base-collectives-test-template",
}

APP_TEST_KINDS = {"", "cuda-vetnode"}
APP_TEST_RUNNERS = {"cuda", "rocm"}


def require_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{context} must be a mapping")
    return value


def require_list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise RuntimeError(f"{context} must be a list")
    return value


def validate_base_ci(path: Path, cfg: dict[str, Any]) -> None:
    family = cfg.get("family")
    if family not in {"cuda", "rocm"}:
        raise RuntimeError(f"{path}: family must be cuda or rocm")
    for idx, variant in enumerate(require_list(cfg.get("variants"), f"{path}: variants")):
        item = require_mapping(variant, f"{path}: variants[{idx}]")
        if not item.get("name") or not item.get("variant"):
            raise RuntimeError(f"{path}: variants[{idx}] must define name and variant")
    for idx, test in enumerate(require_list(cfg.get("tests"), f"{path}: tests")):
        item = require_mapping(test, f"{path}: tests[{idx}]")
        if not item.get("name") or not item.get("kind"):
            raise RuntimeError(f"{path}: tests[{idx}] must define name and kind")
        if item["kind"] not in BASE_TEST_TEMPLATES:
            raise RuntimeError(f"{path}: unsupported base test kind: {item['kind']}")


def validate_app_ci(path: Path, variants: list[str], cfg: dict[str, Any]) -> None:
    tests_by_variant = require_mapping(cfg.get("tests"), f"{path}: tests")
    for variant in variants:
        tests = require_list(tests_by_variant.get(variant), f"{path}: tests.{variant}")
        if not tests:
            raise RuntimeError(f"{path}: tests.{variant} must not be empty")
        for idx, test in enumerate(tests):
            item = require_mapping(test, f"{path}: tests.{variant}[{idx}]")
            if not item.get("name"):
                raise RuntimeError(f"{path}: tests.{variant}[{idx}] must define name")
            kind = str(item.get("kind", ""))
            if kind not in APP_TEST_KINDS:
                raise RuntimeError(f"{path}: unsupported app test kind: {kind}")
            if kind == "cuda-vetnode":
                continue
            runner = str(item.get("runner", variant))
            if runner not in APP_TEST_RUNNERS:
                raise RuntimeError(f"{path}: unsupported app test runner: {runner}")
            require_list(item.get("script"), f"{path}: tests.{variant}[{idx}].script")
            variables = item.get("variables", {})
            require_mapping(variables, f"{path}: tests.{variant}[{idx}].variables")


def discover_bases() -> list[tuple[dict[str, str], list[dict[str, Any]]]]:
    result = []
    for path in [ROOT / "Alps-Images" / "NGC" / "ci.yaml", ROOT / "Alps-Images" / "ROCm" / "ci.yaml"]:
        cfg = load_yaml(path)
        validate_base_ci(path, cfg)
        family = str(cfg["family"])
        tests = cfg.get("tests") or []
        for variant in cfg.get("variants") or []:
            data = base_env(family, str(variant["name"]), str(variant["variant"]))
            result.append((data, tests))
    return result


def discover_apps() -> list[tuple[dict[str, str], list[dict[str, Any]]]]:
    apps = []
    for profile in sorted((ROOT / "Alps-Images" / "apps").glob("*/profile.env")):
        app = profile.parent.name
        variants = source_profile_value(profile, "APP_VARIANTS").split()
        ci_file = profile.parent / "ci.yaml"
        cfg = load_yaml(ci_file)
        validate_app_ci(ci_file, variants, cfg)
        tests_by_variant = cfg.get("tests") or {}
        for variant in variants:
            tests = tests_by_variant.get(variant) or []
            data = app_env(app, variant)
            apps.append((data, tests))
    return apps


def add_publish(child: Child, image: dict[str, str], needs: list[str], prefix: str, force: bool = False) -> None:
    if not force and not publish_needed(image):
        return
    child.add_job(
        f"publish-{prefix}",
        ".child-publish-template",
        needs=needs,
        variables={
            "CANON_IMAGE_REF": image["CANON_IMAGE_REF"],
            "STABLE_IMAGE_REF": image["STABLE_IMAGE_REF"],
            "GHCR_STABLE_IMAGE_REF": image["GHCR_STABLE_IMAGE_REF"],
            "TESTED_IMAGE_REF": image["TESTED_IMAGE_REF"],
        },
    )


def add_base(child: Child, base: dict[str, str], tests: list[dict[str, Any]]) -> str | None:
    prefix = slug(f"base-{base['NAME']}-{base['FAMILY']}-{base['VARIANT']}")
    build = f"build-{prefix}"
    exists = bool(img_digest(base["CANON_IMAGE_REF"]))
    if not exists:
        build_vars = {
            key: value
            for key, value in base.items()
            if key not in {"FAMILY", "NAME", "VARIANT", "TEST_IMAGE_REF", "TESTED_IMAGE_REF", "VALIDATION_HASH", "STABLE_IMAGE_REF", "GHCR_STABLE_IMAGE_REF"}
        }
        child.add_job(build, f".child-{base['FAMILY']}-base-build-template", variables=build_vars)
    valid = marker_valid(base["CANON_IMAGE_REF"], base["TESTED_IMAGE_REF"])
    if valid:
        add_publish(child, base, [], prefix)
        return None
    test_needs = [build] if not exists else []
    test_jobs: list[str] = []
    for test in tests:
        kind = str(test["kind"])
        job = f"test-{prefix}-{slug(str(test['name']))}"
        child.add_job(job, BASE_TEST_TEMPLATES[kind], image=base["CANON_IMAGE_REF"], needs=test_needs)
        test_jobs.append(job)
    mark = f"mark-{prefix}-tested"
    child.add_job(mark, ".child-mark-base-tested-template", needs=test_jobs, variables={"CANON_IMAGE_REF": base["CANON_IMAGE_REF"], "TESTED_IMAGE_REF": base["TESTED_IMAGE_REF"]})
    add_publish(child, base, [mark], prefix, force=True)
    return mark


def add_app_test(child: Child, app: dict[str, str], test: dict[str, Any], needs: list[str]) -> str:
    prefix = slug(f"app-{app['NAME']}-{app['FAMILY']}")
    name = f"test-{prefix}-{slug(str(test['name']))}"
    kind = str(test.get("kind", ""))
    runner = str(test.get("runner", app["FAMILY"]))
    if kind == "cuda-vetnode":
        child.add_job(name, ".child-cuda-app-vetnode-test-template", image=app["CANON_IMAGE_REF"], needs=needs)
        return name
    template = ".child-rocm-custom-test-template" if runner == "rocm" else ".child-cuda-custom-test-template"
    variables = {str(k): str(v) for k, v in (test.get("variables") or {}).items()}
    child.add_job(
        name,
        template,
        image=app["CANON_IMAGE_REF"],
        needs=needs,
        variables=variables,
        timeout=str(test.get("timeout")) if test.get("timeout") else None,
        script=[str(command) for command in test.get("script") or []],
    )
    return name


def add_app(child: Child, app: dict[str, str], tests: list[dict[str, Any]], base_marks: dict[str, str | None], base_valid: dict[str, bool]) -> None:
    prefix = slug(f"app-{app['NAME']}-{app['FAMILY']}")
    base_ref = app["BASE_IMAGE"]
    base_needs = [base_marks[base_ref]] if base_marks.get(base_ref) else []
    if not base_valid.get(base_ref, False) and base_ref not in base_marks:
        raise RuntimeError(f"base metadata missing for {app['NAME']}/{app['FAMILY']}: {base_ref}")

    valid = marker_valid(app["CANON_IMAGE_REF"], app["TESTED_IMAGE_REF"])
    if valid:
        add_publish(child, app, base_needs, prefix)
        return

    build_job: str | None = None
    if not img_digest(app["CANON_IMAGE_REF"]):
        build_job = f"build-{prefix}"
        child.add_job(
            build_job,
            f".child-{app['FAMILY']}-app-build-template",
            needs=base_needs,
            variables={
                "DOCKERFILE": app["DOCKERFILE"],
                "CANON_IMAGE_REF": app["CANON_IMAGE_REF"],
                "BASE_IMAGE": app["BASE_IMAGE"],
                "OCI_SOURCE": app["OCI_SOURCE"],
                "OCI_REVISION": app["OCI_REVISION"],
                "OCI_CREATED": app["OCI_CREATED"],
                "OCI_DESCRIPTION": app["OCI_DESCRIPTION"],
                "CSCS_ALPS_GIT_COMMIT_SHORT": app["CSCS_ALPS_GIT_COMMIT_SHORT"],
            },
        )

    if not tests:
        raise RuntimeError(f"no tests declared for {app['NAME']}/{app['FAMILY']}")
    test_needs = [build_job] if build_job else base_needs
    test_jobs = [add_app_test(child, app, test, test_needs) for test in tests]
    mark = f"mark-{prefix}-tested"
    child.add_job(mark, ".child-mark-app-tested-template", needs=test_jobs, variables={"CANON_IMAGE_REF": app["CANON_IMAGE_REF"], "TESTED_IMAGE_REF": app["TESTED_IMAGE_REF"]})
    add_publish(child, app, [mark], prefix, force=True)


def main() -> None:
    child = Child()
    base_marks: dict[str, str | None] = {}
    base_valid: dict[str, bool] = {}
    for base, tests in discover_bases():
        base_valid[base["CANON_IMAGE_REF"]] = marker_valid(base["CANON_IMAGE_REF"], base["TESTED_IMAGE_REF"])
        base_marks[base["CANON_IMAGE_REF"]] = add_base(child, base, tests)
    for app, tests in discover_apps():
        add_app(child, app, tests, base_marks, base_valid)
    child.write()


if __name__ == "__main__":
    main()
