#!/usr/bin/env python3
"""Validate local GitLab CI wiring that YAML parsing alone cannot catch."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml


def load_pipeline(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"ERROR: {path} did not parse to a YAML mapping")
    return data


def need_job_name(need: object) -> str | None:
    if isinstance(need, str):
        return need
    if isinstance(need, dict):
        job = need.get("job")
        if isinstance(job, str):
            return job
    return None


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    pipeline_path = repo / "ci-pipelines" / "build-alps-extended-images.yaml"
    pipeline = load_pipeline(pipeline_path)

    test_jobs = {
        name
        for name, value in pipeline.items()
        if isinstance(value, dict)
        and isinstance(name, str)
        and name.startswith("test-")
        and value.get("stage") in {"test-base", "test-apps"}
    }

    publish_gate = pipeline.get("publish-gate")
    if not isinstance(publish_gate, dict):
        raise SystemExit("ERROR: missing publish-gate job")

    needs = publish_gate.get("needs")
    if not isinstance(needs, list):
        raise SystemExit("ERROR: publish-gate.needs must be a list")

    gate_jobs = {job for need in needs if (job := need_job_name(need))}

    missing = sorted(test_jobs - gate_jobs)
    stale = sorted(job for job in gate_jobs - test_jobs if job.startswith("test-"))

    if missing or stale:
        if missing:
            print("ERROR: publish-gate.needs is missing test jobs:", file=sys.stderr)
            for job in missing:
                print(f"  - {job}", file=sys.stderr)
        if stale:
            print("ERROR: publish-gate.needs references unknown test jobs:", file=sys.stderr)
            for job in stale:
                print(f"  - {job}", file=sys.stderr)
        return 1

    print(f"CI wiring OK: publish-gate.needs covers {len(test_jobs)} test jobs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
