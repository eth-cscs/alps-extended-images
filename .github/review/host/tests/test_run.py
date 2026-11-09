import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

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
    diagnostic = json.loads((tmp_path / "events.diagnostics.json").read_text())
    assert diagnostic["status"] == "failed"
    assert diagnostic["failure"] == ("checkout_changed" if changed_source else "reviewer_exit")
    assert diagnostic["cleanup"] == "failed"
    assert diagnostic["exit_code"] == 1


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
    assert "reasoning:1" in progress
    assert "private reasoning" not in progress
    assert "credential marker" not in progress


def test_progress_waits_for_complete_events_and_counts_them_once(tmp_path, runner, capsys):
    with (tmp_path / "events.jsonl").open("w+b") as output:
        class Reports:
            count = 0

            def wait(self, interval):
                self.count += 1
                if self.count == 1:
                    os.write(output.fileno(), b'{"type":"step_finish","part":{"reason":"length')
                elif self.count == 2:
                    os.write(output.fileno(), b'"}}\n')
                return self.count > 2

        runner._report_progress(output, Reports(), runner.time.monotonic())
    lines = capsys.readouterr().out.splitlines()
    assert "events=none" in lines[0]
    assert "step_finish:1" in lines[1] and "last_finish=length" in lines[1]
    assert "step_finish:1" in lines[2] and "step_finish:2" not in lines[2]
    assert "invalid" not in "\n".join(lines)


def test_progress_summarizes_final_errors_without_exposing_unknown_fields(tmp_path, runner, capsys):
    with (tmp_path / "events.jsonl").open("w+b") as output:
        os.write(output.fileno(),
                 b'{"type":"error","error":{"message":"private error marker"}}\n'
                 b'{"type":"private event marker"}\n'
                 b'{"type":"review_phase","phase":"private phase marker"}\n'
                 b'{"type":"step_finish","part":{"reason":"private finish marker"}}')
        stopped = runner.threading.Event()
        stopped.set()
        runner._report_progress(output, stopped, runner.time.monotonic())
    progress = capsys.readouterr().out
    assert "error:1" in progress and "unknown:1" in progress
    assert "step_finish:1" in progress and "last_finish=other" in progress
    assert "phase=other" in progress
    assert "private" not in progress


@pytest.mark.parametrize(("tool", "state", "result"), [
    ("read", {"status": "completed"}, "read/completed"),
    ("read", {"status": "error"}, "read/error"),
    ("execute", {"status": "completed", "metadata": {"metadata": {"error": True}}}, "execute/error"),
    ("shell", {"status": "completed", "metadata": {"metadata": {"exit": 128}}}, "shell/error"),
    ("shell", {"status": "completed", "metadata": {"metadata": {"exit": 0}}}, "shell/completed"),
    ("private tool marker", {"status": "private status marker"}, "other/other"),
])
def test_progress_classifies_tool_results_without_exposing_payload(
        tmp_path, runner, capsys, tool, state, result):
    state = dict(state, input={"command": "private command marker"},
                 output="private output marker", error="private error marker")
    event = {"type": "tool_use", "part": {"tool": tool, "state": state}}
    with (tmp_path / "events.jsonl").open("w+b") as output:
        os.write(output.fileno(), json.dumps(event).encode() + b"\n")
        stopped = runner.threading.Event()
        stopped.set()
        runner._report_progress(output, stopped, runner.time.monotonic())
    progress = capsys.readouterr().out
    assert f"tools={result}:1" in progress
    assert "private" not in progress


@pytest.mark.parametrize("exit_code", [0, 7])
def test_runner_retains_private_events_and_sanitized_diagnostics(
        tmp_path, trusted_runner, checkout, runner, monkeypatch, capfd, exit_code):
    _, instructions = trusted_runner
    context = tmp_path / "context.md"
    context.write_text("private context marker\n")
    engine = tmp_path / "engine"
    cleanup = tmp_path / "cleanup"
    engine.write_text(
        f"#!{sys.executable}\n"
        "import json, pathlib, sys\n"
        "if sys.argv[1] == 'rm':\n"
        f"    pathlib.Path({str(cleanup)!r}).touch()\n"
        "    sys.exit(0)\n"
        "print(json.dumps({'type': 'reasoning', 'text': 'private model marker'}))\n"
        "print(json.dumps({'type': 'private event marker'}))\n"
        "print('private stderr credential marker', file=sys.stderr)\n"
        f"sys.exit({exit_code})\n"
    )
    engine.chmod(0o700)
    monkeypatch.setenv("CSCS_INFERENCE_API_KEY", "private credential marker")
    events = tmp_path / "events.jsonl"
    if exit_code:
        with pytest.raises(ValueError, match=f"exit {exit_code}"):
            runner.run_review(str(engine), "unused", checkout, context, events, instructions)
    else:
        runner.run_review(str(engine), "unused", checkout, context, events, instructions)
    diagnostic_path = tmp_path / "events.diagnostics.json"
    raw = diagnostic_path.read_text()
    diagnostic = json.loads(raw)
    assert diagnostic["status"] == ("failed" if exit_code else "success")
    assert diagnostic["failure"] == ("reviewer_exit" if exit_code else "none")
    assert diagnostic["exit_code"] == exit_code
    assert diagnostic["events"] == {"reasoning": 1, "unknown": 1}
    assert diagnostic["summary_complete"] is True
    assert diagnostic["elapsed_seconds"] >= 0
    assert diagnostic["cleanup"] == ("completed" if exit_code else "not_needed")
    assert cleanup.exists() == bool(exit_code)
    assert diagnostic_path.stat().st_mode & 0o777 == 0o600
    assert "private model marker" in events.read_text()
    stderr_path = tmp_path / "events.stderr.log"
    assert "private stderr credential marker" in stderr_path.read_text()
    assert stderr_path.stat().st_mode & 0o777 == 0o600
    captured = capfd.readouterr()
    assert "private" not in raw + captured.out + captured.err


def test_hanging_reviewer_is_killed_and_container_cleanup_is_bounded(
        tmp_path, trusted_runner, checkout, runner, monkeypatch):
    _, instructions = trusted_runner
    context = tmp_path / "context.md"
    context.write_text("review input\n")
    engine = tmp_path / "engine"
    cleanup = tmp_path / "cleanup"
    engine.write_text(
        f"#!{sys.executable}\n"
        "import pathlib, signal, sys, time\n"
        "if sys.argv[1] == 'rm':\n"
        f"    pathlib.Path({str(cleanup)!r}).touch()\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "while True:\n"
        "    time.sleep(60)\n"
    )
    engine.chmod(0o700)
    monkeypatch.setenv("CSCS_INFERENCE_API_KEY", "local-deadline-test")
    monkeypatch.setattr(runner, "_REVIEW_TIMEOUT", 0.5)
    monkeypatch.setattr(runner, "_KILL_GRACE", 0.1)
    monkeypatch.setattr(runner, "_CLEANUP_TIMEOUT", 0.5)
    started = time.monotonic()
    with pytest.raises(ValueError, match="hard deadline"):
        runner.run_review(str(engine), "unused", checkout, context,
                          tmp_path / "events.jsonl", instructions)
    assert time.monotonic() - started < 5
    diagnostic = json.loads((tmp_path / "events.diagnostics.json").read_text())
    assert diagnostic["status"] == "failed"
    assert diagnostic["failure"] == "timeout"
    assert diagnostic["exit_code"] == -signal.SIGKILL
    assert diagnostic["process_cleanup"] == "completed"
    assert diagnostic["cleanup"] == "failed"
    assert cleanup.exists()
    assert (checkout / "app.py").read_text() == "important source\n"


@pytest.mark.parametrize("suffix", ["diagnostics.json", "stderr.log"])
def test_private_outputs_cannot_overwrite_inputs(
        tmp_path, trusted_runner, checkout, runner, monkeypatch, suffix):
    _, instructions = trusted_runner
    context = tmp_path / "context.md"
    context.write_text("review input\n")
    (tmp_path / f"events.{suffix}").symlink_to(context)
    monkeypatch.setenv("CSCS_INFERENCE_API_KEY", "local-path-safety-test")
    with pytest.raises(ValueError, match=suffix.split(".")[0]):
        runner.run_review("/bin/true", "unused", checkout, context,
                          tmp_path / "events.jsonl", instructions)
    assert context.read_text() == "review input\n"
    assert not (tmp_path / "events.jsonl").exists()


