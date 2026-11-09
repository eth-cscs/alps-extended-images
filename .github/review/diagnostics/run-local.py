#!/usr/bin/env python3
"""Run an existing reviewer image with private, persistent local diagnostics."""

import argparse
import io
import json
import os
from pathlib import Path
import shutil
import shlex
import signal
import subprocess
import sys
import threading
import tarfile
import uuid
from urllib.parse import urlsplit, urlunsplit


HOST = Path(__file__).resolve().parents[1] / "host"
sys.path.insert(0, str(HOST))
from run import checkout_state, worktree_root


class LaunchError(Exception):
    def __init__(self, message, status=1):
        super().__init__(message)
        self.status = status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", default="docker", help="Docker or Podman executable")
    parser.add_argument("--image", required=True, help="Architecture-compatible prebuilt image")
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--instructions", required=True)
    parser.add_argument("--output", required=True, help="New or empty directory outside the checkout")
    args = parser.parse_args(argv)

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        launch(args)
    except KeyboardInterrupt:
        parser.exit(130, "local reviewer: interrupted; diagnostics retained\n")
    except LaunchError as error:
        parser.exit(error.status, f"local reviewer: {error}\n")
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        parser.exit(1, f"local reviewer: {error}\n")
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def launch(args):
    checkout = worktree_root(args.checkout)
    context = Path(args.context).resolve(strict=True)
    instructions = Path(args.instructions).resolve(strict=True)
    output = Path(args.output).resolve()
    if not context.is_file() or not instructions.is_file():
        raise ValueError("context and instructions must be files")
    if output == checkout or checkout in output.parents:
        raise ValueError("output must be outside the source checkout")
    secret = os.environ.get("CSCS_INFERENCE_API_KEY")
    if not secret:
        raise ValueError("CSCS_INFERENCE_API_KEY is required")
    executable = shutil.which(args.engine)
    if executable is None:
        raise ValueError("container engine not found")
    before = checkout_state(checkout)
    # Refuse reuse before creating any files: earlier captures may be the only
    # evidence of a failed run. Private parent permissions also protect /work.
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("output directory must be new or empty")
    output.chmod(0o700)
    staging = output / "staging"
    work = output / "work"
    recording = output / "recording"
    for directory in (staging, work, recording, staging / "bin"):
        directory.mkdir(mode=0o700)

    def redact(text):
        return text.replace(secret, "[REDACTED]")

    def save_json(path, value, *, mask=True):
        text = json.dumps(value, indent=2) + "\n"
        path.write_text(redact(text) if mask else text, encoding="utf-8")
        path.chmod(0o600)

    invocation = {
        "image": args.image, "checkout": str(checkout), "checkout_head": before[0],
        "context": str(context), "instructions": str(instructions),
        "environment": {"CSCS_INFERENCE_API_KEY": "[REDACTED]"}, "commands": [],
    }

    def record(command):
        invocation["commands"].append(command)
        save_json(output / "invocation.json", invocation)
        try:
            print("local reviewer: " + redact(shlex.join(command)), file=sys.stderr, flush=True)
        except BrokenPipeError:
            pass

    def log_bytes(stream, data):
        with (output / f"engine.{stream}.log").open("ab") as log:
            log.write(data.replace(secret.encode(), b"[REDACTED]"))

    def command(command, *, check=True, timeout=None, log_stdout=True):
        record(command)
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                check=False, timeout=timeout)
        if log_stdout:
            log_bytes("stdout", result.stdout)
        log_bytes("stderr", result.stderr)
        if check and result.returncode:
            sys.stderr.buffer.write(result.stderr.replace(secret.encode(), b"[REDACTED]"))
            sys.stderr.buffer.flush()
            raise LaunchError(f"command failed (exit {result.returncode}); "
                              f"see {output / 'engine.stderr.log'}", exit_status(result.returncode))
        return result

    # CSCS rootless Podman has a single mapped host user, not a subordinate UID
    # range. Docker instead uses the host UID/GID for writable bind ownership.
    podman = "podman" in Path(executable).name.lower()
    identity = (["--uidmap=1000:0:1", "--gidmap=1000:0:1", "--user=1000:1000"]
                if podman else [f"--user={os.getuid()}:{os.getgid()}"])
    extraction_name = "opencode-local-config-" + uuid.uuid4().hex
    reviewer_name = "opencode-local-review-" + uuid.uuid4().hex
    owner = uuid.uuid4().hex
    owner_label = "local.opencode-review.launch"
    owned = []
    active = None
    pumps = []
    try:
        inspected = command([executable, "image", "inspect", args.image], log_stdout=False)
        image_info = json.loads(inspected.stdout)
        if not isinstance(image_info, list) or len(image_info) != 1:
            raise ValueError("image inspect did not identify exactly one image")
        image_id = image_info[0].get("Id") or image_info[0].get("ID")
        if not isinstance(image_id, str) or not image_id:
            raise ValueError("image inspect did not return an image ID")
        # Keep the full image metadata; redact only the inference key used by this run.
        save_json(output / "image.json", image_info)
        invocation["image_id"] = image_id
        owned.append(extraction_name)
        command([executable, "create", "--name", extraction_name, "--pull=never",
                 "--label", f"{owner_label}={owner}", *identity, image_id])
        original = staging / "opencode.original.json"
        copied = command([executable, "cp",
                          f"{extraction_name}:/opt/reviewer/config/opencode/opencode.json", "-"],
                         log_stdout=False)
        # Read the cp archive in memory, without executing the image or letting
        # archive paths/ownership affect the host filesystem.
        with tarfile.open(fileobj=io.BytesIO(copied.stdout)) as archive:
            files = [member for member in archive.getmembers() if member.isfile()]
            if len(files) != 1:
                raise ValueError("image configuration archive must contain one file")
            with archive.extractfile(files[0]) as source:
                config_text = source.read().decode("utf-8")
        original.write_text(redact(config_text), encoding="utf-8")
        original.chmod(0o600)
        config = json.loads(config_text)
        provider = config["model"].split("/", 1)[0]
        settings = config["providers"][provider]["settings"]
        upstream = settings["baseURL"]
        parsed = urlsplit(upstream)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("baked provider baseURL must be HTTPS without credentials")
        # The baked config is authoritative. Change only its provider endpoint;
        # model, step allowances, token limits and tool settings stay untouched.
        settings["baseURL"] = urlunsplit(parsed._replace(scheme="http", netloc="127.0.0.1:18080"))
        # Preserve authentication exactly in the private runtime config, not its log copy.
        save_json(staging / "opencode.json", config, mask=False)
        command([executable, "rm", extraction_name])
        owned.remove(extraction_name)

        for source, name in ((context, "context.md"), (instructions, "AGENTS.md"),
                             (Path(__file__).with_name("provider-trace.mjs"), "provider-trace.mjs")):
            shutil.copyfile(source, staging / name)
            (staging / name).chmod(0o400)
        # Global logging flags were verified against the image CLI. A PATH
        # wrapper leaves the baked entrypoint and phase runner unmodified.
        wrapper = staging / "bin" / "opencode"
        wrapper.write_text('#!/bin/sh\nexec /opt/reviewer/bin/opencode --log-level all --print-logs "$@"\n',
                           encoding="utf-8")
        wrapper.chmod(0o500)
        debug_entrypoint = staging / "entrypoint.sh"
        debug_entrypoint.write_text('#!/bin/sh\nexport PATH="/opt/local-diagnostics/bin:$PATH"\n'
                                    'exec /opt/reviewer/entrypoint.sh\n', encoding="utf-8")
        debug_entrypoint.chmod(0o500)
        run = [
            executable, "run", "--name", reviewer_name, "--pull=never", "--read-only",
            "--label", f"{owner_label}={owner}",
            "--cap-drop=ALL", "--security-opt=no-new-privileges", *identity,
            "--workdir=/work", "--tmpfs=/tmp:rw,nosuid,nodev,mode=1777",
            "--volume", f"{checkout}:/input/pr-source:ro",
            "--volume", f"{staging / 'context.md'}:/input/context.md:ro",
            "--volume", f"{staging / 'AGENTS.md'}:/instructions/AGENTS.md:ro",
            "--volume", f"{staging}:/opt/local-diagnostics:ro",
            "--volume", f"{staging / 'opencode.json'}:/opt/reviewer/config/opencode/opencode.json:ro",
            "--volume", f"{work}:/work:rw", "--volume", f"{recording}:/diagnostics/recording:rw",
            "--env", "CSCS_INFERENCE_API_KEY", "--entrypoint", "node", image_id,
            "/opt/local-diagnostics/provider-trace.mjs", "--upstream", upstream,
            "--output", "/diagnostics/recording", "--port", "18080", "--verbose", "--",
            "/opt/local-diagnostics/entrypoint.sh",
        ]
        record(run)
        owned.append(reviewer_name)
        print(f"local reviewer: full private diagnostics in {output}", file=sys.stderr, flush=True)
        active = subprocess.Popen(run, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        def pump(source, stream, terminal):
            # Even if a terminal/pipe closes, keep draining to the persistent log.
            with (output / f"engine.{stream}.log").open("ab", buffering=0) as log:
                while chunk := source.read1(65536):
                    log.write(chunk)
                    if terminal is not None:
                        try:
                            terminal.write(chunk)
                            terminal.flush()
                        except (BrokenPipeError, OSError):
                            terminal = None
            source.close()

        for source, stream, terminal in ((active.stdout, "stdout", sys.stdout.buffer),
                                         (active.stderr, "stderr", sys.stderr.buffer)):
            thread = threading.Thread(target=pump, args=(source, stream, terminal), daemon=True)
            thread.start()
            pumps.append(thread)
        status = active.wait()
        for thread in pumps:
            thread.join()
        if status:
            raise LaunchError(f"reviewer failed (exit {status}); diagnostics retained in {output}",
                              exit_status(status))
        extraction = [sys.executable, str(HOST / "extract.py"), "--events",
                      str(recording / "opencode.events.jsonl"), "--output", str(output / "review.md")]
        command(extraction)
    finally:
        # Ignore further interrupts only during bounded cleanup. Stop gracefully
        # before forced removal so the recorder can flush its final captures.
        handlers = {sig: signal.signal(sig, signal.SIG_IGN) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            for name in reversed(owned):
                try:
                    # A failed/interrupted create might not have made a
                    # container. Never remove a same-named container we do not
                    # own, even in the unlikely event of a name collision.
                    found = command([executable, "inspect", "--type=container", "--format",
                                     '{{index .Config.Labels "' + owner_label + '"}}', name],
                                    check=False, timeout=15)
                    if found.returncode or found.stdout.decode().strip() != owner:
                        continue
                    if name == reviewer_name:
                        try:
                            command([executable, "stop", "--time", "10", name], check=False, timeout=20)
                        except (OSError, subprocess.SubprocessError) as error:
                            print(f"local reviewer: graceful stop failed: {error}", file=sys.stderr)
                    result = command([executable, "rm", "--force", name], check=False, timeout=30)
                    if result.returncode:
                        print(f"local reviewer: cleanup failed for {name}; see engine.stderr.log", file=sys.stderr)
                except (OSError, subprocess.SubprocessError) as error:
                    print(f"local reviewer: cleanup failed for {name}: {error}", file=sys.stderr)
            if active is not None:
                if active.poll() is None:
                    active.terminate()
                try:
                    active.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    active.kill()
                    active.wait()
                for thread in pumps:
                    thread.join()
            if checkout_state(checkout) != before:
                raise ValueError("review changed the source checkout or HEAD")
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


def exit_status(code):
    return 128 - code if code < 0 else min(code, 255)


if __name__ == "__main__":
    main()
