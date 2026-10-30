#!/usr/bin/env python3
"""Collect RL training run files from the run tracker directory into public run records.

Training jobs launched from the FirecREST UI AI Training page write files into a shared run
tracker directory on the cluster: <job id>-run.json when they start, <job id>-metrics.json with
the metrics so far every few minutes while they run, and <job id>-result.json when they exit.
This script reads them through FirecREST and turns them into one record per run,
<recipe id>/<cluster>-<job id>.json, plus an index.json per recipe, in a checkout of the records
branch.

A job can't always report how it ended: preemption or a cancellation may kill it before its exit
handler runs, or leave it to record the signal as an ordinary failure. So for recent runs the
final state is also asked from Slurm, which then decides the status (e.g. preempted).
It runs in two phases, so that tracker files are only deleted once the records are pushed:

  collect_rl_runs.py collect --runs-dir <checkout> --manifest <file>
  collect_rl_runs.py delete --manifest <file> [--dry-run]

Configuration (environment): FIRECREST_URL, FIRECREST_TOKEN_URL, FIRECREST_CLIENT_ID,
FIRECREST_CLIENT_SECRET, FIRECREST_SYSTEM and RUN_TRACKER_DIR.

Records are public:
- only an allowlist of fields is kept, each checked against a strict format, whatever else a file
  contains (everyone can write to the tracker directory, so files are untrusted)
- the cluster and Slurm job id are kept, so that the UI can link a run to its job, and the
  partition, account and monitoring settings (e.g. the W&B project), so that it can clone the
  run. No user name or path is, though whoever can see the job in Slurm can see whose it is
- dates are kept to the day; durations are kept exactly
Output is deterministic, so a pass with nothing new leaves nothing to commit.

Workflow logs are public too: they show counts, validation reasons and record paths, but never
file contents.

Standard library only, so the workflow needs no install step.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# Must match the schemaVersion the UI writes into tracker files
RUN_RECORD_SCHEMA_VERSION = 1

# Invalid files are kept for a while in case they are still being fixed, then deleted
INVALID_FILE_MAX_AGE = timedelta(hours=24)
# Run files are a few kB; anything much larger is not one
MAX_TRACKER_FILE_SIZE = 1024 * 1024
# A run without a result this long after its time limit ran out is reported as unknown. Start
# days are only known to the day, so the margin covers a full day more.
OVERDUE_MARGIN = timedelta(days=2)
# Ceiling for runs whose time limit isn't known (launches always record one, so only for
# malformed files)
FALLBACK_TIME_LIMIT = timedelta(days=7)

# ASCII only (\d also matches other scripts' digits), and \Z rather than $, which also matches
# before a trailing newline
TRACKER_FILE_PATTERN = re.compile(r"^(\d{1,20})-(run|metrics|result)\.json\Z", re.ASCII)
SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z", re.ASCII)
# <cluster>-<job id>
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}-\d{1,20}\Z", re.ASCII)
PARAMETER_KEY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z", re.ASCII)
# Metric names may also use ':' as a separator, e.g. val:accuracy
METRIC_KEY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}\Z", re.ASCII)
# Short tokens only: no spaces and no slashes, so no free text and no paths
PARAMETER_STRING_PATTERN = re.compile(r"^[A-Za-z0-9._:+-]{0,64}\Z", re.ASCII)
# Slurm time limits are at most a few years; three digits of days is plenty
TIME_LIMIT_PATTERN = re.compile(r"^(?:(\d{1,3})-)?(\d{1,2}):(\d{2}):(\d{2})\Z", re.ASCII)
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}", re.ASCII)
# Monitoring settings (W&B project, entity, mode, server URL, ...) for cloning a run: short
# values without spaces or quotes. The launcher never records secrets or file paths.
MONITORING_SETTING_PATTERN = re.compile(r"^[A-Za-z0-9._:/+-]{0,256}\Z", re.ASCII)
MAX_MONITORING_SETTINGS = 20
# Link to the run in W&B, recorded only when chosen at launch:
# https://<host>[:<port>][/<path>]/<entity>/<project>/runs/<run id>
MONITORING_URL_PATTERN = re.compile(
    r"^https://[A-Za-z0-9.-]{1,253}(?::\d{1,5})?(?:/[A-Za-z0-9._-]{1,128}){2,6}/runs/([A-Za-z0-9._-]{1,128})\Z",
    re.ASCII,
)
MAX_PARAMETERS = 100
MAX_METRICS = 50
RESULT_STATUSES = ("completed", "failed", "terminated")
# Slurm states of jobs that have ended, and the run status each means. Other states (pending,
# running, requeued, ...) mean the job hasn't ended.
SCHEDULER_END_STATES = {
    "COMPLETED": "completed",
    "FAILED": "failed",
    "NODE_FAIL": "failed",
    "OUT_OF_MEMORY": "failed",
    "BOOT_FAIL": "failed",
    "PREEMPTED": "preempted",
    "CANCELLED": "terminated",
    "TIMEOUT": "terminated",
    "DEADLINE": "terminated",
}
# Runs are checked with Slurm until it confirms how they ended, for at most this long; Slurm
# forgets old jobs anyway
SCHEDULER_CHECK_MAX_AGE = timedelta(days=30)
RESULT_SIGNALS = ("TERM", "INT")

REQUEST_TIMEOUT_SECONDS = 60
# Gateway errors, rate limiting and dropped connections are usually brief: each request is tried
# again after these delays before it counts as failed
RETRY_DELAYS_SECONDS = (5, 15, 45)
RETRY_HTTP_CODES = frozenset({429, 502, 503, 504})

Json = dict[str, Any]


# ----- Validation and records (pure) -----


@dataclass
class TrackerFile:
    name: str
    # Bytes, as listed by FirecREST
    size: int | None = None
    last_modified: str | None = None


@dataclass
class ParsedTrackerFile:
    kind: str
    job_id: str
    cluster: str
    recipe_id: str
    started_on: str | None
    # The public run settings (kind "run"), metrics so far (kind "metrics") or result (kind
    # "result")
    data: Json


@dataclass
class CollectionPlan:
    # Records to write, by path in the records checkout
    records: dict[str, Json] = field(default_factory=dict)
    # Tracker files to delete once the records are pushed
    processed: list[str] = field(default_factory=list)
    # Invalid files old enough to be deleted, with the reason
    expired: list[tuple[str, str]] = field(default_factory=list)
    # Files left in place for now, with the reason
    skipped: list[tuple[str, str]] = field(default_factory=list)


def parse_tracker_file_name(name: str) -> tuple[str, str] | None:
    """Returns (job id, kind) for <job id>-run.json, -metrics.json and -result.json, else None."""
    match = TRACKER_FILE_PATTERN.match(name)
    return (match.group(1), match.group(2)) if match else None


def is_safe_id(value: Any) -> bool:
    return isinstance(value, str) and SAFE_ID_PATTERN.match(value) is not None


def get_run_id(cluster: str, job_id: str) -> str:
    # Job ids are only unique within a cluster
    return f"{cluster}-{job_id}"


def get_record_path(recipe_id: str, run_id: str) -> str:
    return f"{recipe_id}/{run_id}.json"


def _safe_id(value: Any) -> str | None:
    return value if is_safe_id(value) else None


def _id_of(value: Any) -> str | None:
    return _safe_id(value.get("id")) if isinstance(value, dict) else None


def _is_number(value: Any) -> bool:
    # bool is an int in Python, but true and false aren't numbers in JSON
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _to_integer(value: Any, minimum: int, maximum: int) -> int | None:
    if not _is_number(value) or value != int(value):
        return None
    return int(value) if minimum <= value <= maximum else None


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _to_day(value: Any) -> str | None:
    if not isinstance(value, str) or not DATE_PATTERN.match(value):
        return None
    parsed = _parse_time(value)
    return parsed.astimezone(timezone.utc).date().isoformat() if parsed else None


def _to_hyperparameters(value: Any) -> Json:
    if not isinstance(value, dict):
        return {}
    keys = [key for key in value if PARAMETER_KEY_PATTERN.match(key)][:MAX_PARAMETERS]
    hyperparameters = {}
    for key in keys:
        parameter = value[key]
        if (
            _is_number(parameter)
            or isinstance(parameter, bool)
            or (isinstance(parameter, str) and PARAMETER_STRING_PATTERN.match(parameter))
        ):
            hyperparameters[key] = parameter
    return hyperparameters


def _to_metrics(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    keys = [key for key in value if METRIC_KEY_PATTERN.match(key) and _is_number(value[key])]
    return {key: value[key] for key in keys[:MAX_METRICS]}


def _to_monitoring_settings(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    keys = [
        key
        for key in value
        if PARAMETER_KEY_PATTERN.match(key)
        and isinstance(value[key], str)
        and MONITORING_SETTING_PATTERN.match(value[key])
    ]
    return {key: value[key] for key in keys[:MAX_MONITORING_SETTINGS]}


def _to_monitoring_url(value: Any, run_id: str) -> str | None:
    # Only links to this run's id, so a record can't show another W&B run as its own
    match = MONITORING_URL_PATTERN.match(value) if isinstance(value, str) else None
    return value if match and match.group(1) == run_id else None


def _to_public_run(launch: Json, recipe_id: str, run_id: str) -> Json:
    job = launch.get("job")
    job = job if isinstance(job, dict) else {}
    recipe = launch.get("recipe")
    recipe = recipe if isinstance(recipe, dict) else {}
    monitoring = launch.get("monitoring")
    monitoring = monitoring if isinstance(monitoring, dict) else {}
    metrics = launch.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    time_limit = job.get("timeLimit")
    return {
        "recipe": {"id": recipe_id, "version": _safe_id(recipe.get("version"))},
        "model": _id_of(launch.get("model")),
        "environment": _id_of(launch.get("environment")),
        "preset": _id_of(launch.get("preset")),
        "hyperparameters": _to_hyperparameters(launch.get("hyperparameters")),
        "monitoring": _safe_id(monitoring.get("destination")),
        "monitoringSettings": _to_monitoring_settings(monitoring.get("settings")),
        "monitoringUrl": _to_monitoring_url(monitoring.get("url"), run_id),
        "nodes": _to_integer(job.get("nodes"), 1, 100000),
        "gpusPerNode": _to_integer(job.get("gpusPerNode"), 0, 64),
        "timeLimit": (
            time_limit if isinstance(time_limit, str) and TIME_LIMIT_PATTERN.match(time_limit) else None
        ),
        "partition": _safe_id(job.get("partition")),
        "account": _safe_id(job.get("account")),
        # How often the job reports its metrics while running, when it does
        "metricsIntervalSeconds": _to_integer(metrics.get("intervalSeconds"), 1, 24 * 3600),
    }


def _to_public_result(data: Json) -> Json:
    signal = data.get("signal")
    return {
        "status": data["status"],
        "exitCode": _to_integer(data.get("exitCode"), 0, 255),
        "signal": signal if signal in RESULT_SIGNALS else None,
        "endedOn": _to_day(data.get("endedAt")),
        "elapsedSeconds": _to_integer(data.get("elapsedSeconds"), 0, 10 * 365 * 24 * 3600),
        "metrics": _to_metrics(data.get("metrics")),
    }


def parse_tracker_file(name: str, content: str) -> ParsedTrackerFile | str:
    """Validates a tracker file and reduces it to what may be published, or returns an error.

    Only the fields needed to place the record are required; any other field that is missing or
    malformed is left out.
    """
    parsed_name = parse_tracker_file_name(name)
    if not parsed_name:
        return "not a tracker file name"
    job_id, kind = parsed_name
    try:
        data = json.loads(content)
    except ValueError:
        return "not valid JSON"
    if not isinstance(data, dict):
        return "not a JSON object"
    errors = []
    schema_version = data.get("schemaVersion")
    if not _is_number(schema_version) or schema_version != RUN_RECORD_SCHEMA_VERSION:
        errors.append(f"unsupported schemaVersion {schema_version}")
    if data.get("jobId") != job_id:
        errors.append("jobId does not match the file name")
    if not is_safe_id(data.get("cluster")):
        errors.append("missing or invalid cluster")
    launch = data.get("launch")
    if not isinstance(launch, dict):
        launch = None
    if kind == "run":
        recipe_id = _id_of(launch.get("recipe")) if launch else None
    else:
        recipe_id = _safe_id(data.get("recipeId"))
    if not recipe_id:
        errors.append("missing or invalid recipe id")
    if kind == "run" and not launch:
        errors.append("missing launch")
    if kind == "result" and data.get("status") not in RESULT_STATUSES:
        errors.append("missing or invalid status")
    if errors:
        return "; ".join(errors)

    return ParsedTrackerFile(
        kind=kind,
        job_id=job_id,
        cluster=data["cluster"],
        recipe_id=recipe_id,
        started_on=_to_day(data.get("startedAt")),
        data=(
            _to_public_run(launch, recipe_id, get_run_id(data["cluster"], job_id))
            if kind == "run"
            else _to_public_result(data)
            if kind == "result"
            else {"metrics": _to_metrics(data.get("metrics"))}
        ),
    )


def merge_into_record(existing: Json | None, parsed: ParsedTrackerFile, run_id: str) -> Json:
    existing = existing or {}
    if parsed.kind == "run":
        started_on = parsed.started_on or existing.get("startedOn")
    else:
        # The run file's start is authoritative; a result arriving first gives a provisional one
        started_on = existing.get("startedOn") or parsed.started_on
    record = {
        "schemaVersion": RUN_RECORD_SCHEMA_VERSION,
        "runId": run_id,
        "recipeId": parsed.recipe_id,
        "cluster": parsed.cluster,
        "jobId": parsed.job_id,
        "startedOn": started_on,
        "run": parsed.data if parsed.kind == "run" else existing.get("run"),
        "result": parsed.data if parsed.kind == "result" else existing.get("result"),
    }
    # Metrics while the run goes on; only in records that got some, so others don't change
    progress = parsed.data if parsed.kind == "metrics" else existing.get("progress")
    if progress:
        record["progress"] = progress
    return record


def plan_collection(
    files: list[TrackerFile],
    contents: dict[str, str],
    existing: dict[str, Json],
    now: datetime,
) -> CollectionPlan:
    """Plans one collection pass.

    `contents` holds the downloaded content of each listed file (missing when it could not be
    read) and `existing` the records already in the records checkout, by path.
    """
    plan = CollectionPlan()
    # A run's two files must name the same recipe: files are untrusted, and a result naming
    # another recipe would otherwise create a record of its own there
    recipe_by_run_id = {record["runId"]: record["recipeId"] for record in existing.values()}

    def handle_invalid(file: TrackerFile, reason: str) -> None:
        modified = _parse_time(file.last_modified)
        if modified and now - modified > INVALID_FILE_MAX_AGE:
            plan.expired.append((file.name, reason))
        else:
            plan.skipped.append((file.name, reason))

    tracker_files = [
        (file, parsed_name)
        for file in files
        if (parsed_name := parse_tracker_file_name(file.name)) is not None
    ]
    # A run's settings are merged before its metrics and result
    kind_order = {"run": 0, "metrics": 1, "result": 2}
    tracker_files.sort(key=lambda entry: (entry[1][0], kind_order[entry[1][1]]))
    for file, _ in tracker_files:
        if file.size is not None and file.size > MAX_TRACKER_FILE_SIZE:
            handle_invalid(file, "file too large")
            continue
        content = contents.get(file.name)
        if content is None:
            # A read failure says nothing about the file, so it is retried rather than expired
            plan.skipped.append((file.name, "could not be read"))
            continue
        parsed = parse_tracker_file(file.name, content)
        if isinstance(parsed, str):
            handle_invalid(file, parsed)
            continue
        run_id = get_run_id(parsed.cluster, parsed.job_id)
        if recipe_by_run_id.setdefault(run_id, parsed.recipe_id) != parsed.recipe_id:
            handle_invalid(file, "recipe does not match the run's other file")
            continue
        path = get_record_path(parsed.recipe_id, run_id)
        current = plan.records.get(path) or existing.get(path)
        plan.records[path] = merge_into_record(current, parsed, run_id)
        plan.processed.append(file.name)
    return plan


def needs_scheduler_check(record: Json, system: str, now: datetime) -> bool:
    """Whether Slurm should be asked how the record's job ended.

    Only jobs on the system the collector reads through, until Slurm has confirmed their end.
    """
    result = record.get("result") or {}
    started_on = _parse_time(record.get("startedOn"))
    return (
        record.get("cluster") == system
        and isinstance(record.get("jobId"), str)
        and "schedulerState" not in result
        and started_on is not None
        and now - started_on <= SCHEDULER_CHECK_MAX_AGE
    )


def apply_scheduler_state(record: Json, job: Any) -> Json | None:
    """The record updated with Slurm's account of its job, or None when the job hasn't ended.

    Slurm's state decides the status. What the job reported itself (exit code, metrics) is
    kept; for a job that couldn't report, the end and duration come from Slurm.
    """
    if not isinstance(job, dict):
        return None
    status = job.get("status") if isinstance(job.get("status"), dict) else {}
    state = status.get("state")
    # e.g. "CANCELLED by 1234"
    state = state.split()[0].upper() if isinstance(state, str) and state.strip() else None
    if state not in SCHEDULER_END_STATES:
        return None
    times = job.get("time") if isinstance(job.get("time"), dict) else {}
    end = _to_integer(times.get("end"), 1, 2**40)
    reported = record.get("result")
    result = dict(reported) if reported is not None else {
        "status": None,
        "exitCode": _to_integer(status.get("exitCode"), 0, 255),
        "signal": None,
        "endedOn": datetime.fromtimestamp(end, timezone.utc).date().isoformat() if end else None,
        "elapsedSeconds": _to_integer(times.get("elapsed"), 0, 10 * 365 * 24 * 3600),
        "metrics": None,
    }
    result["status"] = SCHEDULER_END_STATES[state]
    result["schedulerState"] = state
    return {**record, "result": result}


def parse_time_limit(time_limit: Any) -> timedelta | None:
    """[D-]HH:MM:SS, as accepted at launch."""
    if not isinstance(time_limit, str):
        return None
    match = TIME_LIMIT_PATTERN.match(time_limit)
    if not match:
        return None
    days, hours, minutes, seconds = match.groups()
    try:
        return timedelta(days=int(days or 0), hours=int(hours), minutes=int(minutes), seconds=int(seconds))
    except OverflowError:
        return None


def get_run_status(record: Json, now: datetime) -> str:
    if record.get("result"):
        return record["result"]["status"]
    started_on = _parse_time(record.get("startedOn"))
    time_limit = parse_time_limit((record.get("run") or {}).get("timeLimit")) or FALLBACK_TIME_LIMIT
    if started_on and now > started_on + time_limit + OVERDUE_MARGIN:
        return "unknown"
    return "running"


def build_run_index(recipe_id: str, records: list[Json], now: datetime) -> Json:
    runs = []
    for record in records:
        if record.get("recipeId") != recipe_id:
            continue
        run = record.get("run") or {}
        result = record.get("result") or {}
        runs.append(
            {
                "runId": record["runId"],
                "cluster": record["cluster"],
                "jobId": record["jobId"],
                "path": get_record_path(record["recipeId"], record["runId"]),
                "status": get_run_status(record, now),
                "startedOn": record.get("startedOn"),
                "endedOn": result.get("endedOn"),
                "elapsedSeconds": result.get("elapsedSeconds"),
                "recipeVersion": (run.get("recipe") or {}).get("version"),
                "model": run.get("model"),
                "environment": run.get("environment"),
                "preset": run.get("preset"),
                "hyperparameters": run.get("hyperparameters") or {},
                "monitoring": run.get("monitoring"),
                "monitoringSettings": run.get("monitoringSettings") or {},
                "monitoringUrl": run.get("monitoringUrl"),
                "nodes": run.get("nodes"),
                "gpusPerNode": run.get("gpusPerNode"),
                "timeLimit": run.get("timeLimit"),
                "partition": run.get("partition"),
                "account": run.get("account"),
                "metricsIntervalSeconds": run.get("metricsIntervalSeconds"),
                # The job's final metrics, else the last ones it reported while running
                "metrics": result.get("metrics") or (record.get("progress") or {}).get("metrics"),
            }
        )
    # Newest first; the run id only makes the order stable within a day
    runs.sort(key=lambda entry: entry["runId"])
    runs.sort(key=lambda entry: entry["startedOn"] or "", reverse=True)
    return {"schemaVersion": RUN_RECORD_SCHEMA_VERSION, "recipeId": recipe_id, "runs": runs}


# ----- FirecREST and files -----


def fail(message: str) -> None:
    print(f"::error::{message}", file=sys.stderr)
    sys.exit(1)


@dataclass
class Config:
    firecrest_url: str
    token_url: str
    client_id: str
    client_secret: str
    system: str
    tracker_dir: str


def get_config() -> Config:
    """Reads the configuration from the environment, where the workflow passes its secrets.

    Only main() calls it; everything else is given the configuration explicitly.
    """
    names = [
        "FIRECREST_URL",
        "FIRECREST_TOKEN_URL",
        "FIRECREST_CLIENT_ID",
        "FIRECREST_CLIENT_SECRET",
        "FIRECREST_SYSTEM",
        "RUN_TRACKER_DIR",
    ]
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        fail(f"Missing configuration: {', '.join(missing)}")
    return Config(
        firecrest_url=os.environ["FIRECREST_URL"].rstrip("/"),
        token_url=os.environ["FIRECREST_TOKEN_URL"],
        client_id=os.environ["FIRECREST_CLIENT_ID"],
        client_secret=os.environ["FIRECREST_CLIENT_SECRET"],
        system=os.environ["FIRECREST_SYSTEM"],
        tracker_dir=os.environ["RUN_TRACKER_DIR"].rstrip("/"),
    )


class FirecrestError(RuntimeError):
    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def _open(request: urllib.request.Request) -> bytes:
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        return response.read()


def send(request: urllib.request.Request) -> bytes:
    """The response body, after retrying temporary failures; raises the last error otherwise."""
    for delay in RETRY_DELAYS_SECONDS:
        try:
            return _open(request)
        except urllib.error.HTTPError as error:
            if error.code not in RETRY_HTTP_CODES:
                raise
        except OSError:
            # URLError, timeouts and connection errors
            pass
        time.sleep(delay)
    # The last attempt, whose error is not retried
    return _open(request)


def get_access_token(config: Config) -> str:
    body = urllib.parse.urlencode(
        {
            "grant_type": "client_credentials",
            "client_id": config.client_id,
            "client_secret": config.client_secret,
        }
    ).encode()
    request = urllib.request.Request(
        config.token_url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    # Errors are reported without details: neither the request nor the response is logged
    try:
        data = json.loads(send(request))
    except urllib.error.HTTPError as error:
        fail(f"Token request failed with HTTP {error.code}")
    except (OSError, ValueError) as error:
        fail(f"Token request failed: {type(error).__name__}")
    access_token = data.get("access_token") if isinstance(data, dict) else None
    if not isinstance(access_token, str) or not access_token:
        fail("Token response has no access_token")
    return access_token


def firecrest(config: Config, access_token: str, method: str, operation: str, path: str) -> bytes:
    system = urllib.parse.quote(config.system, safe="")
    query = urllib.parse.urlencode({"path": path})
    request = urllib.request.Request(
        f"{config.firecrest_url}/filesystem/{system}/ops/{operation}?{query}",
        method=method,
        headers={"Authorization": f"Bearer {access_token}"},
    )
    # FirecREST v2 returns the raw file content for ops/download (not base64 in JSON as v1 did)
    try:
        return send(request)
    except urllib.error.HTTPError as error:
        raise FirecrestError(
            f"{method} ops/{operation} failed with HTTP {error.code}", error.code
        ) from None
    except OSError as error:
        # URLError, timeouts and connection errors
        raise FirecrestError(f"{method} ops/{operation} failed: {type(error).__name__}") from None


def get_scheduler_job(config: Config, access_token: str, job_id: str) -> Any:
    """Slurm's record of a job, or None when Slurm doesn't know it (anymore).

    Only jobs the collector's FirecREST user may see are found: those of its own accounts.
    """
    system = urllib.parse.quote(config.system, safe="")
    request = urllib.request.Request(
        f"{config.firecrest_url}/compute/{system}/jobs/{urllib.parse.quote(job_id, safe='')}",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    try:
        response = json.loads(send(request))
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise FirecrestError(f"GET compute/jobs failed with HTTP {error.code}", error.code) from None
    except OSError as error:
        raise FirecrestError(f"GET compute/jobs failed: {type(error).__name__}") from None
    except ValueError:
        raise FirecrestError("GET compute/jobs returned an unexpected response") from None
    jobs = response.get("jobs") if isinstance(response, dict) else None
    return jobs[0] if isinstance(jobs, list) and jobs else None


def check_scheduler(
    config: Config, access_token: str, records: dict[str, Json], now: datetime
) -> tuple[dict[str, Json], int, int]:
    """Records whose job Slurm reports as ended, updated, with the number checked and failed."""
    updated: dict[str, Json] = {}
    candidates = [path for path in sorted(records) if needs_scheduler_check(records[path], config.system, now)]
    failures = 0
    for path in candidates:
        try:
            job = get_scheduler_job(config, access_token, records[path]["jobId"])
        except FirecrestError:
            # Asked again next pass
            failures += 1
            continue
        record = apply_scheduler_state(records[path], job)
        if record:
            updated[path] = record
    return updated, len(candidates), failures


def parse_listing(content: bytes) -> list[TrackerFile]:
    """The tracker files in an ops/ls response, which is {"output": [File, ...]}."""
    listing = json.loads(content)
    entries = listing.get("output") if isinstance(listing, dict) else None
    if not isinstance(entries, list):
        raise ValueError("unexpected ops/ls response")
    files = []
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("type") != "-":
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not parse_tracker_file_name(name):
            continue
        size = entry.get("size")
        last_modified = entry.get("lastModified")
        files.append(
            TrackerFile(
                name=name,
                size=int(size) if isinstance(size, (int, str)) and str(size).isdigit() else None,
                last_modified=last_modified if isinstance(last_modified, str) else None,
            )
        )
    return files


def list_tracker_files(config: Config, access_token: str) -> list[TrackerFile]:
    try:
        return parse_listing(firecrest(config, access_token, "GET", "ls", config.tracker_dir))
    except FirecrestError as error:
        fail(f"Listing the tracker directory failed: {error}")
    except ValueError:
        fail("Listing the tracker directory returned an unexpected response")


def load_records(runs_dir: Path) -> dict[str, Json]:
    """Existing records in the records checkout, by path relative to it.

    A record that can't be read is left out with a warning; collecting a file of its run again
    rewrites it.
    """
    if not runs_dir.is_dir():
        fail(f"The records checkout {runs_dir} does not exist")
    records = {}
    unreadable = 0
    for recipe_dir in sorted(runs_dir.iterdir()):
        if not recipe_dir.is_dir() or not is_safe_id(recipe_dir.name):
            continue
        for record_file in sorted(recipe_dir.glob("*.json")):
            if not RUN_ID_PATTERN.match(record_file.stem):
                continue
            try:
                record = json.loads(record_file.read_text())
            except (OSError, ValueError):
                record = None
            if (
                isinstance(record, dict)
                and record.get("runId") == record_file.stem
                and record.get("recipeId") == recipe_dir.name
                and isinstance(record.get("cluster"), str)
                and isinstance(record.get("jobId"), str)
                and get_run_id(record["cluster"], record["jobId"]) == record_file.stem
            ):
                records[f"{recipe_dir.name}/{record_file.name}"] = record
            else:
                unreadable += 1
    if unreadable:
        print(f"::warning::{unreadable} existing records could not be read and were left out")
    return records


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def log_reasons(title: str, files: list[tuple[str, str]]) -> None:
    if files:
        print(f"  {title}:")
    for reason, count in Counter(reason for _, reason in files).items():
        print(f"    {count} × {reason}")


def collect(config: Config, runs_dir: Path, manifest: Path) -> None:
    access_token = get_access_token(config)
    files = list_tracker_files(config, access_token)
    contents = {}
    read_failures = 0
    for file in files:
        try:
            content = firecrest(
                config, access_token, "GET", "download", f"{config.tracker_dir}/{file.name}"
            )
        except FirecrestError:
            # A failed download says nothing about the file, so it isn't expired or deleted:
            # plan_collection skips it and the next pass tries again
            read_failures += 1
            continue
        # Bytes that aren't UTF-8 can't be a valid tracker file. Replacing them, rather than
        # treating the file as unreadable, lets validation reject it so that it expires instead
        # of being retried forever; replaced characters never pass the field checks.
        contents[file.name] = content.decode("utf-8", errors="replace")

    existing = load_records(runs_dir)
    now = datetime.now(timezone.utc)
    plan = plan_collection(files, contents, existing, now)
    # How jobs ended, from Slurm, also for runs collected in earlier passes
    ended, checked, check_failures = check_scheduler(
        config, access_token, {**existing, **plan.records}, now
    )
    plan.records.update(ended)
    for record_path, record in plan.records.items():
        write_json(runs_dir / record_path, record)
    # Indexes are rebuilt for every recipe, so that statuses age (running -> unknown) even
    # without new files
    all_records = list({**existing, **plan.records}.values())
    for recipe_id in sorted({record["recipeId"] for record in all_records}):
        write_json(runs_dir / recipe_id / "index.json", build_run_index(recipe_id, all_records, now))

    write_json(
        manifest,
        {
            "system": config.system,
            "trackerDir": config.tracker_dir,
            "delete": plan.processed + [name for name, _ in plan.expired],
        },
    )
    print(
        f"Tracker files: {len(files)} listed, {len(plan.processed)} collected, "
        f"{len(plan.expired)} invalid and expired, {len(plan.skipped)} skipped"
    )
    print(f"Slurm: {checked} runs checked, {len(ended)} confirmed as ended")
    for record_path in sorted(plan.records):
        print(f"  updated {record_path}")
    log_reasons("expired", plan.expired)
    log_reasons("skipped", plan.skipped)
    if read_failures:
        print(f"::warning::{read_failures} tracker files could not be downloaded; the next pass retries them")
    if check_failures:
        print(f"::warning::{check_failures} runs could not be checked with Slurm; the next pass retries them")


def delete_processed(config: Config, manifest_path: Path, dry_run: bool) -> None:
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError):
        fail("The manifest of collected files can't be read")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("delete"), list):
        fail("The manifest of collected files is malformed")
    if manifest.get("system") != config.system or manifest.get("trackerDir") != config.tracker_dir:
        fail("The manifest was written for another system or tracker directory")
    # Only tracker file names, directly in the tracker directory
    names = [
        name
        for name in manifest["delete"]
        if isinstance(name, str) and parse_tracker_file_name(name)
    ]
    if dry_run:
        print(f"Would delete {len(names)} tracker files")
        return
    access_token = get_access_token(config) if names else ""
    failures = 0
    for name in names:
        try:
            firecrest(config, access_token, "DELETE", "rm", f"{config.tracker_dir}/{name}")
        except FirecrestError as error:
            if error.status == 404:
                # Already gone, e.g. deleted by an attempt whose response was lost
                continue
            # Collecting again is idempotent, so the file is simply processed again next time
            failures += 1
    print(f"Deleted {len(names) - failures} of {len(names)} tracker files")
    if failures:
        # The records are pushed, so nothing is lost, but the files would pile up unnoticed
        fail(f"{failures} tracker files could not be deleted; the next pass retries them")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    collect_parser = commands.add_parser("collect", help="collect tracker files into records")
    collect_parser.add_argument("--runs-dir", type=Path, required=True)
    collect_parser.add_argument("--manifest", type=Path, required=True)
    delete_parser = commands.add_parser("delete", help="delete the collected tracker files")
    delete_parser.add_argument("--manifest", type=Path, required=True)
    delete_parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = get_config()
    if args.command == "collect":
        collect(config, args.runs_dir, args.manifest)
    else:
        delete_processed(config, args.manifest, args.dry_run)


if __name__ == "__main__":
    main()
