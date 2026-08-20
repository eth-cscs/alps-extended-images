#!/usr/bin/env python3
"""Generate the registry-dependent child pipeline for Alps images.

The static parent pipeline only runs this script and then triggers the generated
child pipeline. This indirection is deliberate: the correct CI graph depends on
registry state that GitLab cannot know while parsing a static YAML file.

For each declared base/app variant the generator asks meta.sh for canonical refs
and validation-marker refs, then inspects the registries with skopeo.sh:

* missing canonical image -> emit a build job;
* missing/stale tested marker -> emit tests and a marker job;
* validated image with missing/stale stable refs -> emit publish job;
* everything current -> emit nothing for that image.

The child jobs use hidden templates from ci-pipelines/child-templates.yaml. This
keeps dynamic data in generated YAML while runner selection and scripts remain in
reviewed static templates.
"""
from __future__ import annotations

import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "generated-child-pipeline.yaml"
DIGEST_CACHE: dict[str, str] = {}


def run_bash(script: str) -> str:
    """Run repo shell helpers and return stdout.

    Ref/hash derivation remains in shell because meta.sh is shared with manual
    builds and CI jobs. Calling it here avoids duplicating canonical-ref logic in
    Python.
    """
    proc = subprocess.run(
        ["bash", "-c", script],
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
    """Quote a value for a Bash command string."""
    return shlex.quote(value)


def slug(value: str) -> str:
    """Convert a value to a GitLab-safe job-name fragment."""
    result = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not result:
        raise RuntimeError(f"slug produced empty string for {value!r}")
    return result


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping from path."""
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise RuntimeError(f"expected mapping in {path}")
    return data


def load_env_text(text: str) -> dict[str, str]:
    """Parse KEY=VALUE lines emitted by shell env helpers."""
    data: dict[str, str] = {}
    for line in text.splitlines():
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            data[key] = value
    return data


def app_variants_from_profile(profile: Path) -> list[str]:
    """Parse APP_VARIANTS from profile.env without executing the profile."""
    value = None
    for line in profile.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].lstrip()
        if not stripped.startswith("APP_VARIANTS="):
            continue
        raw_value = stripped.split("=", 1)[1]
        try:
            value = " ".join(shlex.split(raw_value, comments=True, posix=True))
        except ValueError as exc:
            raise RuntimeError(f"{profile}: failed to parse APP_VARIANTS: {exc}") from exc
    if value is None:
        raise RuntimeError(f"{profile}: APP_VARIANTS must be set")
    variants = value.split()
    if not variants:
        raise RuntimeError(f"{profile}: APP_VARIANTS must not be empty")
    return variants


def img_digest(ref: str) -> str:
    """Return the registry digest for ref, caching repeated skopeo lookups."""
    if ref not in DIGEST_CACHE:
        DIGEST_CACHE[ref] = run_bash(f"source ci-pipelines/helpers/skopeo.sh && img_digest {shell_quote(ref)}")
    return DIGEST_CACHE[ref]


def tested_marker_matches_existing_canonical(canon_ref: str, tested_ref: str) -> bool:
    """Return whether a tested marker points to the canonical digest.

    A missing canonical digest is not fatal here: the generator uses False to
    mean "schedule build/test/mark work" for images whose canonical ref does not
    exist yet. Registry lookup failures still raise through img_digest().
    """
    canon_digest = img_digest(canon_ref)
    marker_digest = img_digest(tested_ref)
    if marker_digest and canon_digest and marker_digest != canon_digest:
        raise RuntimeError(
            f"tested marker digest mismatch for {tested_ref}: canonical={canon_digest} marker={marker_digest}"
        )
    return bool(canon_digest and marker_digest and canon_digest == marker_digest)


def stable_refs_need_publish_for_existing_canonical(image: dict[str, str]) -> bool:
    """Return true when either stable destination is missing or stale."""
    canon_digest = img_digest(image["CANON_IMAGE_REF"])
    if not canon_digest:
        return False
    stable_digest = img_digest(image["STABLE_IMAGE_REF"])
    ghcr_digest = img_digest(image["GHCR_STABLE_IMAGE_REF"])
    return stable_digest != canon_digest or ghcr_digest != canon_digest


def helper_env(function: str, *args: str) -> dict[str, str]:
    """Call a meta.sh env writer and parse its output."""
    quoted_args = " ".join(shell_quote(arg) for arg in args)
    out = run_bash(
        "source ci-pipelines/helpers/skopeo.sh && "
        "source ci-pipelines/helpers/meta.sh && "
        f"tmp=$(mktemp) && trap 'rm -f \"$tmp\"' EXIT && {function} \"$tmp\" {quoted_args} && cat \"$tmp\""
    )
    return load_env_text(out)


def base_env(family: str, name: str, variant: str) -> dict[str, str]:
    """Return generated-job variables for one base variant."""
    return helper_env("write_base_build_env", family, name, variant)


def app_env(app: str, variant: str) -> dict[str, str]:
    """Return generated-job variables for one app variant."""
    data = helper_env("write_app_build_env", app, variant)
    # App variants are accelerator-family names today; validate_app_variant_name
    # in meta.sh keeps them safe as generated-job metadata keys.
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
        """Add a concrete generated job that extends one hidden template."""
        if name in self.jobs:
            raise RuntimeError(f"duplicate generated job: {name}")
        self.jobs.add(name)
        job: dict[str, Any] = {"extends": [extends]}
        if image:
            job["image"] = image
        if timeout:
            job["timeout"] = timeout
        if needs is not None:
            job["needs"] = list(needs)
        if variables:
            job["variables"] = {key: str(value) for key, value in sorted(variables.items())}
        if script:
            job["script"] = script
        self.data[name] = job

    def add_noop(self) -> None:
        """Add the schedulable fallback job used when there is no work."""
        # GitLab requires at least one job in a pipeline. Use a real runner
        # template so the no-work pipeline can complete instead of waiting for an
        # untagged/default runner that does not exist for this project.
        self.data["no-work-required"] = {"extends": [".child-noop-template"]}

    def write(self) -> None:
        """Write the generated child pipeline YAML."""
        if not self.jobs:
            self.add_noop()
        OUTPUT.write_text(yaml.safe_dump(self.data, sort_keys=False, width=120))


BASE_TEST_TEMPLATES = {
    "cuda": {
        "env": ".child-cuda-base-env-test-template",
        "collectives": ".child-cuda-base-collectives-test-template",
        "vetnode": ".child-cuda-vetnode-test-template",
    },
    "rocm": {
        "env": ".child-rocm-base-env-test-template",
        "collectives": ".child-rocm-base-collectives-test-template",
    },
}
# Adding a new accelerator family requires registering its base test templates
# here and its ref/hash plumbing in meta.sh.
BASE_TEST_KINDS_BY_FAMILY = {
    family: set(templates)
    for family, templates in BASE_TEST_TEMPLATES.items()
}

APP_TEST_KINDS = {"custom", "cuda-vetnode"}
APP_TEST_RUNNERS = {"cuda", "rocm"}
APP_CUDA_VETNODE_KEYS = {"name", "kind"}


def require_mapping(value: Any, context: str) -> dict[str, Any]:
    """Require a YAML value to be a mapping."""
    if not isinstance(value, dict):
        raise RuntimeError(f"{context} must be a mapping")
    return value


def require_list(value: Any, context: str) -> list[Any]:
    """Require a YAML value to be a list."""
    if not isinstance(value, list):
        raise RuntimeError(f"{context} must be a list")
    return value


def validate_base_ci(path: Path, cfg: dict[str, Any]) -> None:
    """Validate one family base ci.yaml file."""
    family = cfg.get("family")
    if not isinstance(family, str) or family not in BASE_TEST_TEMPLATES:
        supported = ", ".join(sorted(BASE_TEST_TEMPLATES))
        raise RuntimeError(f"{path}: unsupported base family: {family!r}; supported families: {supported}")
    for idx, variant in enumerate(require_list(cfg.get("variants"), f"{path}: variants")):
        item = require_mapping(variant, f"{path}: variants[{idx}]")
        if not item.get("name") or not item.get("variant"):
            raise RuntimeError(f"{path}: variants[{idx}] must define name and variant")
    tests = require_list(cfg.get("tests"), f"{path}: tests")
    if not tests:
        raise RuntimeError(f"{path}: tests must not be empty")
    for idx, test in enumerate(tests):
        item = require_mapping(test, f"{path}: tests[{idx}]")
        if not item.get("name"):
            raise RuntimeError(f"{path}: tests[{idx}] must define name")
        name = str(item["name"])
        if name not in BASE_TEST_KINDS_BY_FAMILY[family]:
            raise RuntimeError(f"{path}: unsupported {family} base test name: {name}")


def validate_app_ci(path: Path, variants: list[str], cfg: dict[str, Any]) -> None:
    """Validate one app ci.yaml file against declared variants."""
    tests_by_variant = require_mapping(cfg.get("tests"), f"{path}: tests")
    for variant in variants:
        tests = require_list(tests_by_variant.get(variant), f"{path}: tests.{variant}")
        if not tests:
            raise RuntimeError(f"{path}: tests.{variant} must not be empty")
        for idx, test in enumerate(tests):
            item = require_mapping(test, f"{path}: tests.{variant}[{idx}]")
            if not item.get("name"):
                raise RuntimeError(f"{path}: tests.{variant}[{idx}] must define name")
            kind = str(item.get("kind", "custom"))
            if kind not in APP_TEST_KINDS:
                raise RuntimeError(f"{path}: unsupported app test kind: {kind}")
            if kind == "cuda-vetnode":
                if variant != "cuda":
                    raise RuntimeError(f"{path}: cuda-vetnode is only supported for cuda app tests")
                extra_keys = set(item) - APP_CUDA_VETNODE_KEYS
                if extra_keys:
                    extras = ", ".join(sorted(extra_keys))
                    raise RuntimeError(f"{path}: cuda-vetnode app tests do not support: {extras}")
                continue
            runner = str(item.get("runner", variant))
            if runner not in APP_TEST_RUNNERS:
                raise RuntimeError(f"{path}: unsupported app test runner: {runner}")
            require_list(item.get("script"), f"{path}: tests.{variant}[{idx}].script")
            variables = item.get("variables", {})
            require_mapping(variables, f"{path}: tests.{variant}[{idx}].variables")


def discover_bases() -> list[tuple[dict[str, str], list[dict[str, Any]]]]:
    """Load base variants/tests from family-local ci.yaml files."""
    result = []
    for path in sorted((ROOT / "Alps-Images").glob("*/ci.yaml")):
        if path.parent.name in {"apps", "common"}:
            continue
        cfg = load_yaml(path)
        if "family" not in cfg:
            continue
        validate_base_ci(path, cfg)
        family = str(cfg["family"])
        tests = cfg.get("tests") or []
        for variant in cfg.get("variants") or []:
            data = base_env(family, str(variant["name"]), str(variant["variant"]))
            result.append((data, tests))
    return result


def discover_apps() -> list[tuple[dict[str, str], list[dict[str, Any]]]]:
    """Load app variants from profile.env and tests from app-local ci.yaml."""
    apps = []
    for profile in sorted((ROOT / "Alps-Images" / "apps").glob("*/profile.env")):
        app = profile.parent.name
        variants = app_variants_from_profile(profile)
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
    """Emit promotion only when stable refs need updating, unless forced.

    force=True is used after validation work because publish must re-check the
    new tested marker and stable-tag policy at execution time.
    """
    if not force and not stable_refs_need_publish_for_existing_canonical(image):
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


def add_base(child: Child, base: dict[str, str], tests: list[dict[str, Any]], valid: bool) -> str | None:
    """Emit the minimal build/test/mark/publish graph for one base image."""
    prefix = slug(f"base-{base['NAME']}-{base['FAMILY']}-{base['VARIANT']}")
    build = f"build-{prefix}"
    exists = bool(img_digest(base["CANON_IMAGE_REF"]))
    if not exists:
        build_vars = {
            key: value
            for key, value in base.items()
            if key not in {"FAMILY", "NAME", "VARIANT", "TESTED_IMAGE_REF", "VALIDATION_HASH", "STABLE_IMAGE_REF", "GHCR_STABLE_IMAGE_REF", "OCI_CREATED"}
        }
        child.add_job(build, f".child-{base['FAMILY']}-base-build-template", variables=build_vars)
    if valid:
        add_publish(child, base, [], prefix)
        return None
    test_needs = [build] if not exists else []
    test_jobs: list[str] = []
    for test in tests:
        name = str(test["name"])
        job = f"test-{prefix}-{slug(name)}"
        child.add_job(job, BASE_TEST_TEMPLATES[base["FAMILY"]][name], image=base["CANON_IMAGE_REF"], needs=test_needs)
        test_jobs.append(job)
    mark = f"mark-{prefix}-tested"
    child.add_job(mark, ".child-mark-base-tested-template", needs=test_jobs, variables={"CANON_IMAGE_REF": base["CANON_IMAGE_REF"], "TESTED_IMAGE_REF": base["TESTED_IMAGE_REF"]})
    add_publish(child, base, [mark], prefix, force=True)
    return mark


def add_app_test(child: Child, app: dict[str, str], test: dict[str, Any], needs: list[str]) -> str:
    """Emit one generated app test job and return its name."""
    prefix = slug(f"app-{app['NAME']}-{app['FAMILY']}")
    name = f"test-{prefix}-{slug(str(test['name']))}"
    kind = str(test.get("kind", "custom"))
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
    """Emit the minimal build/test/mark/publish graph for one app image.

    App builds depend on base validation, not just base build completion. This
    keeps app images from being built on top of an untested base digest.
    """
    prefix = slug(f"app-{app['NAME']}-{app['FAMILY']}")
    base_ref = app["BASE_IMAGE"]
    base_needs = [base_marks[base_ref]] if base_marks.get(base_ref) else []
    if not base_valid.get(base_ref, False) and base_ref not in base_marks:
        raise RuntimeError(f"base metadata missing for {app['NAME']}/{app['FAMILY']}: {base_ref}")

    valid = tested_marker_matches_existing_canonical(app["CANON_IMAGE_REF"], app["TESTED_IMAGE_REF"])
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
    """Generate the child pipeline for current metadata and registry state."""
    child = Child()
    base_marks: dict[str, str | None] = {}
    base_valid: dict[str, bool] = {}
    for base, tests in discover_bases():
        canon_ref = base["CANON_IMAGE_REF"]
        valid = tested_marker_matches_existing_canonical(canon_ref, base["TESTED_IMAGE_REF"])
        base_valid[canon_ref] = valid
        base_marks[canon_ref] = add_base(child, base, tests, valid)
    for app, tests in discover_apps():
        add_app(child, app, tests, base_marks, base_valid)
    child.write()


if __name__ == "__main__":
    main()
