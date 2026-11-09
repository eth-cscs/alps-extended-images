#!/usr/bin/env python3
"""Run the trusted reviewer without exposing host credentials or writable source."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid

from context import git, worktree_root
from extract import EVENT_TYPES


_PROGRESS_INTERVAL = 30
_REVIEW_TIMEOUT = 17 * 60
_KILL_GRACE = 5
_CLEANUP_TIMEOUT = 30
_FINISH_REASONS = {"stop", "end_turn", "length", "tool-calls", "content-filter", "error", "unknown"}
_TOOL_NAMES = {"execute", "glob", "grep", "question", "read", "shell",
               "skill", "subagent", "webfetch", "websearch"}
_TOOL_STATUSES = {"pending", "running", "completed", "error"}


def _tool_result(part):
    if not isinstance(part, dict):
        return "other/other"
    name = part.get("tool")
    name = name if isinstance(name, str) and name in _TOOL_NAMES else "other"
    state = part.get("state")
    if not isinstance(state, dict):
        return f"{name}/other"
    status = state.get("status")
    status = status if isinstance(status, str) and status in _TOOL_STATUSES else "other"
    metadata = state.get("metadata")
    details = metadata.get("metadata") if isinstance(metadata, dict) else None
    if isinstance(details, dict):
        # Code Mode and shell failures can arrive with status=completed.
        exit_code = details.get("exit")
        if details.get("error") is True or (type(exit_code) is int and exit_code != 0):
            status = "error"
    return f"{name}/{status}"


def _report_progress(output, stopped, started, diagnostic=None):
    previous = 0
    counts = Counter()
    tool_counts = Counter()
    last_finish = "none"
    phase = "none"
    # A separate descriptor keeps reads from moving the container's stdout write position.
    with open(output.name, "rb") as events:
        while True:
            finished = stopped.wait(_PROGRESS_INTERVAL)
            # Bound each scan so a busy producer cannot delay cancellation.
            size = os.stat(output.name).st_size
            while events.tell() < size:
                position = events.tell()
                line = events.readline(size - position)
                if not line:
                    break
                if not finished and not line.endswith(b"\n"):
                    # A write may stop mid-event; retry that line on the next report.
                    events.seek(position)
                    break
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    counts["invalid"] += 1
                    continue
                kind = event.get("type") if isinstance(event, dict) else None
                if not isinstance(kind, str) or kind not in EVENT_TYPES:
                    counts["unknown"] += 1
                    continue
                counts[kind] += 1
                if kind == "step_finish":
                    part = event.get("part")
                    reason = part.get("reason") if isinstance(part, dict) else None
                    last_finish = reason if isinstance(reason, str) and reason in _FINISH_REASONS else "other"
                elif kind == "tool_use":
                    tool_counts[_tool_result(event.get("part"))] += 1
                elif kind == "review_phase":
                    value = event.get("phase")
                    phase = value if isinstance(value, str) and value in ("inspection", "final") else "other"
            # Only fixed labels reach logs, never arguments, errors, output, or model text.
            summary = "; ".join(
                f"{label}=" + (", ".join(f"{name}:{count}" for name, count in sorted(counter.items())) or "none")
                for label, counter in (("events", counts), ("tools", tool_counts))
            )
            if diagnostic is not None:
                diagnostic.update(events=dict(counts), tools=dict(tool_counts), phase=phase,
                                  last_finish=last_finish, stdout_bytes=size,
                                  summary_complete=finished)
            if finished:
                print(f"review runner: final {summary}; phase={phase}; last_finish={last_finish}", flush=True)
                return
            elapsed = int(time.monotonic() - started)
            print(f"review runner: running; elapsed={elapsed}s; stdout={size} bytes (+{size - previous}); "
                  f"{summary}; phase={phase}; last_finish={last_finish}", flush=True)
            previous = size

def _stop_process(process):
    """Bound termination of the engine client and any local descendants."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return "completed"
    try:
        process.wait(timeout=_KILL_GRACE)
    except subprocess.TimeoutExpired:
        pass
    # Kill surviving descendants even if their process-group leader exited.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=_KILL_GRACE)
    except subprocess.TimeoutExpired:
        return "failed"
    return "completed"


def _write_diagnostic(path, diagnostic):
    with path.open("w", encoding="utf-8") as output:
        os.fchmod(output.fileno(), 0o600)
        json.dump(diagnostic, output, sort_keys=True)
        output.write("\n")



def checkout_state(checkout):
    return (git(checkout, "rev-parse", "--verify", "HEAD"),
            git(checkout, "status", "--porcelain=v1", "--untracked-files=all"))


def run_review(engine, image, checkout, context, events, instructions):
    checkout = worktree_root(checkout)
    context = Path(context).resolve(strict=True)
    events = Path(events).resolve()
    # The workflow selects this file from its default-branch checkout.
    instructions = Path(instructions).resolve(strict=True)
    if not instructions.is_file():
        raise ValueError("trusted AGENTS.md must be a file")
    if checkout in instructions.parents:
        raise ValueError("instructions must be outside the PR checkout")
    if not context.is_file():
        raise ValueError("context must be a file")
    if events == checkout or checkout in events.parents or events in (context, instructions):
        raise ValueError("events output must be outside the checkout, context and instructions files")
    diagnostics = events.with_suffix(".diagnostics.json").resolve()
    if (diagnostics == events or diagnostics == checkout or checkout in diagnostics.parents
            or diagnostics in (context, instructions)):
        raise ValueError("diagnostics output must not overwrite review inputs or events")
    stderr = events.with_suffix(".stderr.log").resolve()
    if (stderr == checkout or checkout in stderr.parents
            or stderr in (events, diagnostics, context, instructions)):
        raise ValueError("stderr output must not overwrite review inputs, events or diagnostics")
    if not os.environ.get("CSCS_INFERENCE_API_KEY"):
        raise ValueError("CSCS_INFERENCE_API_KEY is required")
    executable = shutil.which(engine)
    if executable is None:
        raise ValueError("container engine not found")
    before = checkout_state(checkout)
    events.parent.mkdir(parents=True, exist_ok=True)
    # Private host staging permits the unprivileged container UID to read context
    # even when the caller's RUNNER_TEMP files have restrictive owner-only modes.
    with tempfile.TemporaryDirectory(prefix="opencode-context-", dir=events.parent) as staging:
        staged_context = Path(staging) / "context.md"
        staged_instructions = Path(staging) / "AGENTS.md"
        for source, destination in ((context, staged_context), (instructions, staged_instructions)):
            shutil.copyfile(source, destination)
            destination.chmod(0o444)
        name = "opencode-review-" + uuid.uuid4().hex
        command = [
            executable, "run", "--rm", "--name", name, "--pull=never", "--read-only",
            "--cap-drop=ALL", "--security-opt=no-new-privileges",
            "--user=1000:1000", "--workdir=/work",
            "--tmpfs=/work:rw,nosuid,nodev,mode=1777",
            "--tmpfs=/tmp:rw,nosuid,nodev,mode=1777",
            "--volume", f"{checkout}:/input/pr-source:ro",
            "--volume", f"{staged_context}:/input/context.md:ro",
            "--volume", f"{staged_instructions}:/instructions/AGENTS.md:ro",
            # Deliberately do not forward GitHub/Actions tokens, SSH agents,
            # Docker sockets, host HOME, or any other environment variable.
            "--env", "CSCS_INFERENCE_API_KEY", image,
        ]
        started = time.monotonic()
        diagnostic = {
            "status": "running", "failure": "none", "exit_code": None,
            "cleanup": "not_needed", "process_cleanup": "not_needed",
            "events": {}, "tools": {}, "phase": "none", "last_finish": "none",
            "stdout_bytes": 0, "summary_complete": False, "elapsed_seconds": 0,
        }
        _write_diagnostic(diagnostics, diagnostic)
        try:
            with events.open("w", encoding="utf-8") as output, stderr.open("w", encoding="utf-8") as errors:
                os.fchmod(output.fileno(), 0o600)
                os.fchmod(errors.fileno(), 0o600)
                print(f"review runner: starting isolated reviewer; deadline={_REVIEW_TIMEOUT}s; "
                      f"progress every {_PROGRESS_INTERVAL}s "
                      "(byte counts and fixed event/tool summaries only)", flush=True)
                stopped = threading.Event()
                reporter = threading.Thread(target=_report_progress,
                                            args=(output, stopped, started, diagnostic), daemon=True)
                reporter.start()
                process = None
                try:
                    # Raw stderr may contain source, model text, or credentials; never log it.
                    process = subprocess.Popen(command, stdout=output, stderr=errors,
                                               start_new_session=True)
                    diagnostic["exit_code"] = process.wait(timeout=_REVIEW_TIMEOUT)
                except BaseException as error:
                    diagnostic["failure"] = (
                        "timeout" if isinstance(error, subprocess.TimeoutExpired)
                        else "interrupted" if isinstance(error, KeyboardInterrupt) else "runner_error")
                    if process is not None:
                        diagnostic["process_cleanup"] = _stop_process(process)
                        diagnostic["exit_code"] = process.returncode
                    if isinstance(error, subprocess.TimeoutExpired):
                        raise ValueError("isolated reviewer exceeded the 17-minute hard deadline") from error
                    raise
                finally:
                    stopped.set()
                    reporter.join(timeout=_KILL_GRACE)
                print(f"review runner: exited; status={diagnostic['exit_code']}; "
                      f"elapsed={int(time.monotonic() - started)}s; "
                      f"stdout={os.fstat(output.fileno()).st_size} bytes", flush=True)
            if diagnostic["exit_code"]:
                diagnostic["failure"] = "reviewer_exit"
                raise ValueError(f"isolated reviewer failed (exit {diagnostic['exit_code']})")
            diagnostic["status"] = "success"
        except BaseException:
            diagnostic["status"] = "failed"
            if diagnostic["failure"] == "none":
                diagnostic["failure"] = "runner_error"
            # Cancellation also needs bounded container cleanup, independent of the engine client.
            try:
                subprocess.run([executable, "rm", "--force", name], check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               timeout=_CLEANUP_TIMEOUT)
                diagnostic["cleanup"] = "completed"
            except (OSError, subprocess.SubprocessError):
                diagnostic["cleanup"] = "failed"
                print("review runner: container cleanup failed", file=sys.stderr)
            raise
        finally:
            # Cleanup errors must not skip the source-state check or diagnostic persistence.
            try:
                if checkout_state(checkout) != before:
                    diagnostic.update(status="failed", failure="checkout_changed")
                    raise ValueError("review changed the source checkout or HEAD")
            except BaseException:
                if diagnostic["failure"] != "checkout_changed":
                    diagnostic.update(status="failed", failure="checkout_check_error")
                raise
            finally:
                diagnostic["elapsed_seconds"] = int(time.monotonic() - started)
                _write_diagnostic(diagnostics, diagnostic)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", default="docker", help="Docker-compatible engine executable")
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--instructions", required=True, help="AGENTS.md from the default-branch checkout")
    parser.add_argument("--events", required=True)
    args = parser.parse_args(argv)

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    previous_handler = signal.signal(signal.SIGTERM, interrupted)
    try:
        run_review(args.engine, args.image, args.checkout, args.context, args.events, args.instructions)
    except (OSError, ValueError) as error:
        parser.exit(1, f"review runner: {error}\n")
    except KeyboardInterrupt:
        parser.exit(130, "review runner: interrupted\n")
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


if __name__ == "__main__":
    main()
