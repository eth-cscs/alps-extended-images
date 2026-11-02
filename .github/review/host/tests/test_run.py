import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


HELPER = Path(__file__).resolve().parents[1] / "run.py"


@pytest.fixture
def trusted_runner(tmp_path):
    root = tmp_path / "default-branch"
    root.mkdir()
    instructions = root / "AGENTS.md"
    instructions.write_text("trusted repository instructions\n")
    return HELPER, instructions


@pytest.fixture
def checkout(tmp_path):
    checkout = tmp_path / "pr-source"
    checkout.mkdir()
    (checkout / "app.py").write_text("important source\n")
    environment = dict(os.environ, GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)
    for args in [
        ["init", "-b", "main"], ["add", "app.py"],
        ["-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
         "commit", "-m", "temporary fixture"],
    ]:
        subprocess.run(["git", "-C", str(checkout), *args], env=environment,
                       check=True, capture_output=True)
    return checkout


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(HELPER.parent))
    spec = importlib.util.spec_from_file_location("review_runner", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("target", ["checkout-alias", "context", "instructions-alias"])
def test_runner_cannot_overwrite_inputs(tmp_path, trusted_runner, checkout, target):
    helper, instructions = trusted_runner
    source = checkout / "app.py"
    environment = dict(os.environ, CSCS_INFERENCE_API_KEY="local-path-safety-test")
    context = tmp_path / "context.md"
    context.write_text("important context\n")
    if target == "checkout-alias":
        alias = tmp_path / "looks-private"
        alias.symlink_to(checkout, target_is_directory=True)
        output = alias / "app.py"
    elif target == "context":
        output = context
    else:
        output = tmp_path / "instruction-alias"
        output.symlink_to(instructions)
    # /bin/false is a real failing executable, not a simulated container. If the
    # path guard regresses, opening stdout truncates the protected file before
    # that executable fails. No container or model credentials are needed.
    result = subprocess.run([
        sys.executable, str(helper), "--engine", "/bin/false", "--image", "unused",
        "--checkout", str(checkout), "--context", str(context), "--events", str(output),
        "--instructions", str(instructions),
    ], env=environment, text=True, capture_output=True)
    assert result.returncode == 1
    assert source.read_text() == "important source\n"
    assert context.read_text() == "important context\n"
    assert instructions.read_text() == "trusted repository instructions\n"


@pytest.mark.parametrize("alias", [False, True])
def test_runner_rejects_pr_supplied_instructions(tmp_path, checkout, alias):
    instructions = checkout / "AGENTS.md"
    instructions.write_text("untrusted PR instructions\n")
    if alias:
        link = tmp_path / "looks-trusted"
        link.symlink_to(instructions)
        instructions = link
    context = tmp_path / "context.md"
    context.write_text("review input\n")
    output = tmp_path / "events.jsonl"
    environment = dict(os.environ, CSCS_INFERENCE_API_KEY="local-path-safety-test")
    # A successful real executable makes acceptance observable if the path guard regresses.
    result = subprocess.run([
        sys.executable, str(HELPER), "--engine", "/bin/true", "--image", "unused",
        "--checkout", str(checkout), "--context", str(context), "--events", str(output),
        "--instructions", str(instructions),
    ], env=environment, text=True, capture_output=True)
    assert result.returncode == 1
    assert not output.exists()
    assert instructions.read_text() == "untrusted PR instructions\n"


@pytest.mark.parametrize("changed_source", [False, True])
def test_cleanup_timeout_preserves_failure_and_checks_source(
        tmp_path, trusted_runner, checkout, runner, monkeypatch, capsys, changed_source):
    _, instructions = trusted_runner
    context = tmp_path / "context.md"
    context.write_text("review input\n")
    monkeypatch.setenv("CSCS_INFERENCE_API_KEY", "local-cleanup-test")
    real_run = subprocess.run

    def run(command, *args, **kwargs):
        if command[:2] == ["/bin/false", "rm"]:
            if changed_source:
                (checkout / "app.py").write_text("changed during cleanup\n")
            # Force the timeout without sleeping or requiring a container daemon.
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(runner.subprocess, "run", run)
    expected = "checkout" if changed_source else "exit 1"
    with pytest.raises(ValueError, match=expected):
        runner.run_review("/bin/false", "unused", checkout, context,
                          tmp_path / "events.jsonl", instructions)
    assert "container cleanup failed" in capsys.readouterr().err


def test_progress_reports_silence_without_exposing_event_text(tmp_path, runner, capsys):
    private_event = b'{"type":"reasoning","text":"private reasoning and credential marker"}\n'
    with (tmp_path / "events.jsonl").open("w+b") as output:
        class Reports:
            count = 0

            def wait(self, interval):
                self.count += 1
                if self.count == 2:
                    os.write(output.fileno(), private_event)
                return self.count > 2

        # Drive a quiet tick and an output tick without waiting for wall-clock time.
        runner._report_progress(output, Reports(), runner.time.monotonic())
    progress = capsys.readouterr().out
    assert "0 bytes (+0)" in progress
    assert f"{len(private_event)} bytes (+{len(private_event)})" in progress
    assert "private reasoning" not in progress
    assert "credential marker" not in progress
