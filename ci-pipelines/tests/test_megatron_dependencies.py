import runpy
from importlib import metadata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
STACK_CHECK = ROOT / "Alps-Images/apps/apertus-2/tests/megatron/check-stack.py"


@pytest.fixture()
def dependency_check():
    return runpy.run_path(str(STACK_CHECK))["check_dependencies"]


def test_dependency_conflict_is_rejected(monkeypatch, dependency_check):
    monkeypatch.setattr(metadata, "requires", lambda name: ["apache-tvm-ffi<=0.1.9"])
    monkeypatch.setattr(metadata, "version", lambda name: "0.1.11")

    with pytest.raises(AssertionError, match=r"requires apache-tvm-ffi<=0\.1\.9"):
        dependency_check(["example"], {})


def test_dependency_uses_ngc_triton_metadata(monkeypatch, dependency_check):
    monkeypatch.setattr(metadata, "requires", lambda name: ["triton>=3.5.0"])

    def version(name):
        assert name == "pytorch-triton"
        return "3.7.0"

    monkeypatch.setattr(metadata, "version", version)

    dependency_check(["example"], {"pytorch-triton": "3.7.0"})


def test_optional_dependency_is_not_required(monkeypatch, dependency_check):
    monkeypatch.setattr(metadata, "requires", lambda name: ['addon; extra == "optional"'])
    monkeypatch.setattr(metadata, "version", lambda name: pytest.fail(f"Unexpected dependency: {name}"))

    dependency_check(["example"], {})


def test_missing_dependency_is_rejected(monkeypatch, dependency_check):
    monkeypatch.setattr(metadata, "requires", lambda name: ["missing-package"])

    def version(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", version)

    with pytest.raises(metadata.PackageNotFoundError):
        dependency_check(["example"], {})
