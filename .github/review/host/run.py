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
_FINISH_REASONS = {"stop", "end_turn", "length", "tool-calls", "content-filter", "error", "unknown"}


def _report_progress(output, stopped, started):
    previous = 0
    counts = Counter()
    last_finish = "none"
    # A separate descriptor keeps reads from moving the container's stdout write position.
    with open(output.name, "rb") as events:
        while True:
            finished = stopped.wait(_PROGRESS_INTERVAL)
            # Bound each scan so a busy producer cannot delay cancellation.
            size = os.fstat(output.fileno()).st_size
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
            # Only fixed event names and finish reasons reach logs, never payload fields.
            summary = ", ".join(f"{kind}:{count}" for kind, count in sorted(counts.items())) or "none"
            if finished:
                print(f"review runner: final events={summary}; last_finish={last_finish}", flush=True)
                return
            elapsed = int(time.monotonic() - started)
            print(f"review runner: running; elapsed={elapsed}s; stdout={size} bytes (+{size - previous}); "
                  f"events={summary}; last_finish={last_finish}", flush=True)
            previous = size


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
        try:
            with events.open("w", encoding="utf-8") as output:
                os.fchmod(output.fileno(), 0o600)
                started = time.monotonic()
                print(f"review runner: starting isolated reviewer; progress every {_PROGRESS_INTERVAL}s "
                      "(byte counts and event types only)", flush=True)
                stopped = threading.Event()
                reporter = threading.Thread(target=_report_progress,
                                            args=(output, stopped, started), daemon=True)
                reporter.start()
                try:
                    result = subprocess.run(command, stdout=output, check=False)
                finally:
                    # Stop and join before closing the event file or removing the container.
                    stopped.set()
                    reporter.join()
                elapsed = int(time.monotonic() - started)
                size = os.fstat(output.fileno()).st_size
                print(f"review runner: exited; status={result.returncode}; elapsed={elapsed}s; "
                      f"stdout={size} bytes", flush=True)
            if result.returncode:
                raise ValueError(f"isolated reviewer failed (exit {result.returncode})")
        except BaseException:
            # Cancellation also needs cleanup; re-raise the original failure afterward.
            try:
                subprocess.run([executable, "rm", "--force", name], check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
            except (OSError, subprocess.SubprocessError) as error:
                print(f"review runner: container cleanup failed: {error}", file=sys.stderr)
            raise
        finally:
            # Cleanup errors must not skip the source-state check.
            if checkout_state(checkout) != before:
                raise ValueError("review changed the source checkout or HEAD")


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
