"""Exercise Mooncake distribution replacement without compiling its native stack."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "Alps-Images/apps/vllm"


def run(*args, cwd=None):
    return subprocess.run(
        args, cwd=cwd, check=True, text=True, capture_output=True,
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
    ).stdout


def wheel_source(root, files):
    """Create a small wheel fixture with the pinned upstream metadata patch context."""
    root.mkdir()
    (root / "pyproject.toml").write_text(
        '[build-system]\nrequires = ["setuptools", "wheel"]\n'
        'build-backend = "setuptools.build_meta"\n\n'
        '[project]\nname = "mooncake-transfer-engine"\n'
        'version = "0.3.13.post1"\n'
        'description = "A KVCache-centric Disaggregated Architecture for large-scale LLM inference and training."\n'
        '[tool.setuptools.packages.find]\ninclude = ["mooncake*"]\n'
        '[tool.setuptools.package-data]\nmooncake = ["*.so"]\n'
    )
    for name, content in {"__init__.py": "", **files}.items():
        path = root / "mooncake" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def build_wheel(source, destination):
    run(sys.executable, "-m", "pip", "--isolated", "wheel", "--no-index",
        "--no-deps", "--no-build-isolation", "--wheel-dir", str(destination), str(source))
    return next(destination.glob("*.whl"))


@pytest.mark.parametrize("accel", ["cuda", "rocm"])
def test_cxi_wheel_replaces_upstream_distribution_and_stale_modules(tmp_path, accel):
    # Use the Containerfile's actual COPY list to select packaging patches.
    containerfile = APP / ("Containerfile.rocm" if accel == "rocm" else "Containerfile")
    patch_dirs = [
        ROOT / line.split()[1]
        for line in containerfile.read_text().splitlines()
        if line.startswith("COPY ") and line.split()[-1] == "/opt/alps/mooncake-patches"
    ]
    source = tmp_path / "mooncake-wheel"
    wheel_source(source, {"engine.py": 'BUILD = "CXI"\n', "store.so": "CXI store",
                          "libasio.so": "pinned ASIO"})
    for patch_dir in patch_dirs:
        for patch in sorted(patch_dir.glob("*.patch")):
            run("git", "apply", "--include=mooncake-wheel/pyproject.toml", str(patch), cwd=tmp_path)
    local_wheel = build_wheel(source, tmp_path / "local-dist")
    expected_name = "mooncake-transfer-engine" + ("-rocm" if accel == "rocm" else "")

    upstream = tmp_path / "upstream"
    wheel_source(upstream, {
        "engine.py": 'BUILD = "upstream"\n', "libasio.so": "newer ASIO",
        "store/__init__.py": "from mooncake import _store\n",
        "_store.cpython-312-x86_64-linux-gnu.so": "newer store ABI",
    })
    metadata = upstream / "pyproject.toml"
    metadata.write_text(metadata.read_text().replace('name = "mooncake-transfer-engine"',
                        f'name = "{expected_name}"').replace('version = "0.3.13.post1"',
                        'version = "0.3.14"'))
    upstream_wheel = build_wheel(upstream, tmp_path / "upstream-dist")

    venv = tmp_path / "venv"
    run(sys.executable, "-m", "venv", "--without-pip", str(venv))
    python = str(venv / "bin/python")
    for wheel in (upstream_wheel, local_wheel):
        run(sys.executable, "-m", "pip", "--isolated", "--python", python,
            "install", "--no-index", "--no-deps", str(wheel))

    installed = json.loads(run(python, "-I", "-c", '''
import importlib.metadata as m, importlib.util, json
from pathlib import Path
import mooncake, mooncake.engine
root = Path(mooncake.__file__).parent
# Restrict discovery to the package directory: some venvs put both lib and its
# lib64 alias on sys.path, which otherwise reports the same distribution twice.
owners = [d for d in m.distributions(path=[str(root.parent)])
          if any(str(f).startswith("mooncake/") for f in d.files or [])]
print(json.dumps({
    "owners": [[d.metadata["Name"], d.version] for d in owners],
    "store": Path(importlib.util.find_spec("mooncake.store").origin).name,
    "stale": [str(p.relative_to(root)) for p in root.rglob("*") if p.name.startswith("_store") or p.name == "store"],
    "engine": mooncake.engine.BUILD,
    "asio": (root / "libasio.so").read_text(),
}))
'''))
    assert installed == {
        "owners": [[expected_name, "0.3.13.post1"]],
        "store": "store.so", "stale": [], "engine": "CXI", "asio": "pinned ASIO",
    }
