import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest


COLLECTOR = Path(__file__).resolve().parents[1] / "collect_rl_runs.py"
spec = importlib.util.spec_from_file_location("collect_rl_runs", COLLECTOR)
crr = importlib.util.module_from_spec(spec)
# Registered before running it, as dataclasses look their module up
sys.modules[spec.name] = crr
spec.loader.exec_module(crr)

NOW = datetime(2026, 10, 23, 12, 0, tzinfo=timezone.utc)
RECENT = "2026-10-23T11:00:00Z"
OLD = "2026-10-20T11:00:00Z"
RECIPE = "nemo-rl-apertus-gsm8k"


def run_file(job_id, **overrides):
    data = {
        "schemaVersion": crr.RUN_RECORD_SCHEMA_VERSION,
        "jobId": job_id,
        "cluster": "clariden",
        "startedAt": "2026-10-23T08:13:27Z",
        "launch": {
            "recipe": {"id": RECIPE, "version": "0f34d2a5d94a6be6c679e69270433dd3af51c80d"},
            "model": {"id": "apertus-2-chonk"},
            "environment": {"id": "gsm8k"},
            "preset": {"id": "muon"},
            "hyperparameters": {"MAX_NUM_STEPS": 58, "BASE_LR": 2e-7, "USE_FOO": True},
            "monitoring": {"destination": "wandb"},
            "job": {"nodes": 20, "gpusPerNode": 4, "timeLimit": "07:00:00"},
        },
    }
    data.update(overrides)
    return data


def result_file(job_id, **overrides):
    data = {
        "schemaVersion": crr.RUN_RECORD_SCHEMA_VERSION,
        "jobId": job_id,
        "cluster": "clariden",
        "recipeId": RECIPE,
        "status": "completed",
        "exitCode": 0,
        "signal": "",
        "startedAt": "2026-10-23T08:13:27Z",
        "endedAt": "2026-10-23T09:33:41Z",
        "elapsedSeconds": 4814,
        "metrics": None,
    }
    data.update(overrides)
    return data


def collect(files, contents, existing=None):
    return crr.plan_collection(
        [crr.TrackerFile(name=name, size=size, last_modified=modified) for name, modified, size in files],
        {name: json.dumps(data) if not isinstance(data, str) else data for name, data in contents.items()},
        existing or {},
        NOW,
    )


def collect_one(name, data):
    return collect([(name, RECENT, None)], {name: data})


class TestTrackerFiles:
    def test_only_accepts_job_id_run_and_result_names(self):
        assert crr.parse_tracker_file_name("123-run.json") == ("123", "run")
        assert crr.parse_tracker_file_name("123-result.json") == ("123", "result")
        assert crr.parse_tracker_file_name(".123-run.json.tmp") is None
        assert crr.parse_tracker_file_name("../123-run.json") is None
        assert crr.parse_tracker_file_name("abc-run.json") is None
        # Python's $ would accept a trailing newline, and \d other scripts' digits
        assert crr.parse_tracker_file_name("123-run.json\n") is None
        assert crr.parse_tracker_file_name("١٢٣-run.json") is None

    def test_rejects_files_that_could_not_be_placed_safely(self):
        def parse(name, data):
            return crr.parse_tracker_file(name, json.dumps(data))

        assert parse("1-run.json", run_file("1")).kind == "run"
        bad_recipe = run_file("1", launch={"recipe": {"id": "../../x"}})
        assert isinstance(parse("1-run.json", bad_recipe), str)
        assert isinstance(parse("1-result.json", result_file("1", cluster="a/b")), str)
        assert isinstance(parse("1-result.json", result_file("1", cluster="clariden\n")), str)
        assert isinstance(parse("1-result.json", result_file("1", status="pwned")), str)
        assert isinstance(parse("1-run.json", run_file("1", schemaVersion=True)), str)
        assert parse("1-run.json", run_file("2")) == "jobId does not match the file name"
        assert crr.parse_tracker_file("1-run.json", '{"truncated":') == "not valid JSON"


class TestPublishedRecords:
    def test_keep_only_allowlisted_fields_whatever_else_a_file_contains(self):
        launch = run_file("7")["launch"]
        plan = collect_one(
            "7-run.json",
            run_file(
                "7",
                user="alice",
                account="a01",
                workingDirectory="/users/alice/runs",
                launch={
                    **launch,
                    "recipe": {"id": RECIPE, "version": None, "name": "Alice's run"},
                    "model": {"id": "apertus-2-chonk", "modelPath": "/users/alice/model"},
                    "hyperparameters": {
                        "MAX_NUM_STEPS": 58,
                        "DATA_PATH": "/users/alice/data",
                        "NOTE": "contact alice@example.com",
                        "MODE": "fast",
                        "bad key": 1,
                        "NAN": None,
                        "TRAILING": "fast\n",
                    },
                    "monitoring": {"destination": "wandb", "environment": {"WANDB_ENTITY": "alice"}},
                    "job": {
                        "name": "alice-test",
                        "nodes": 20,
                        "gpusPerNode": 4,
                        "timeLimit": "07:00:00",
                        "partition": "normal",
                        "account": "../alice",
                    },
                },
            ),
        )
        record = plan.records[f"{RECIPE}/clariden-7.json"]
        assert record == {
            "schemaVersion": crr.RUN_RECORD_SCHEMA_VERSION,
            "runId": "clariden-7",
            "recipeId": RECIPE,
            "cluster": "clariden",
            "jobId": "7",
            "startedOn": "2026-10-23",
            "run": {
                "recipe": {"id": RECIPE, "version": None},
                "model": "apertus-2-chonk",
                "environment": "gsm8k",
                "preset": "muon",
                "hyperparameters": {"MAX_NUM_STEPS": 58, "MODE": "fast"},
                "monitoring": "wandb",
                "monitoringSettings": {},
                "monitoringUrl": None,
                "nodes": 20,
                "gpusPerNode": 4,
                "timeLimit": "07:00:00",
                "partition": "normal",
                "account": None,
            },
            "result": None,
        }
        published = json.dumps(plan.records)
        for leaked in ("alice", "08:13"):
            assert leaked not in published

    def test_keep_the_partition_and_account_for_cloning(self):
        launch = run_file("7")["launch"]
        launch = {**launch, "job": {**launch["job"], "partition": "debug", "account": "a-csstaff"}}
        plan = collect_one("7-run.json", run_file("7", launch=launch))
        run = plan.records[f"{RECIPE}/clariden-7.json"]["run"]
        assert (run["partition"], run["account"]) == ("debug", "a-csstaff")
        index = crr.build_run_index(RECIPE, list(plan.records.values()), NOW)
        assert (index["runs"][0]["partition"], index["runs"][0]["account"]) == ("debug", "a-csstaff")

    def test_keep_short_monitoring_settings_for_cloning(self):
        settings = {
            "WANDB_PROJECT": "chonk-grpo",
            "WANDB_ENTITY": "my-team",
            "wandb.host": "https://forge.example.org/wandb",
            "wandb.publishLink": "true",
            "NOTE": "contact me",
            "QUOTE": 'a"b',
            "NUMBER": 3,
            "bad key": "x",
        }
        launch = {**run_file("7")["launch"], "monitoring": {"destination": "wandb", "settings": settings}}
        plan = collect_one("7-run.json", run_file("7", launch=launch))
        assert plan.records[f"{RECIPE}/clariden-7.json"]["run"]["monitoringSettings"] == {
            "WANDB_PROJECT": "chonk-grpo",
            "WANDB_ENTITY": "my-team",
            "wandb.host": "https://forge.example.org/wandb",
            "wandb.publishLink": "true",
        }

    def test_keep_the_monitoring_link_only_when_it_points_to_the_run(self):
        def url_of(url):
            launch = {**run_file("7")["launch"], "monitoring": {"destination": "wandb", "url": url}}
            plan = collect_one("7-run.json", run_file("7", launch=launch))
            return plan.records[f"{RECIPE}/clariden-7.json"]["run"]["monitoringUrl"]

        url = "https://forge.example.org/wandb/team/project/runs/clariden-7"
        assert url_of(url) == url
        assert url_of("https://wandb.ai/team/project/runs/clariden-7") is not None
        for bad in (
            "https://wandb.ai/team/project/runs/clariden-8",
            "http://wandb.ai/team/project/runs/clariden-7",
            "https://wandb.ai/project/runs/clariden-7",
            "https://wandb.ai/team/project/runs/clariden-7?x=1",
            "https://wandb.ai/te am/project/runs/clariden-7",
            "javascript:alert(1)//runs/clariden-7",
            None,
            42,
        ):
            assert url_of(bad) is None, bad

    def test_keep_results_to_status_exit_day_duration_and_numeric_metrics(self):
        plan = collect_one(
            "7-result.json",
            result_file(
                "7",
                signal="TERM",
                status="terminated",
                exitCode=143,
                metrics={"val:accuracy": 0.907, "note": "great run by alice", "steps": 58, "flag": True},
                user="alice",
            ),
        )
        assert plan.records[f"{RECIPE}/clariden-7.json"]["result"] == {
            "status": "terminated",
            "exitCode": 143,
            "signal": "TERM",
            "endedOn": "2026-10-23",
            "elapsedSeconds": 4814,
            "metrics": {"val:accuracy": 0.907, "steps": 58},
        }

    def test_booleans_are_not_numbers(self):
        plan = collect_one("7-result.json", result_file("7", exitCode=True, elapsedSeconds=False))
        result = plan.records[f"{RECIPE}/clariden-7.json"]["result"]
        assert result["exitCode"] is None
        assert result["elapsedSeconds"] is None


class TestPlanCollection:
    def test_merges_a_run_and_its_result_and_marks_both_for_deletion(self):
        plan = collect(
            [("7-result.json", RECENT, None), ("7-run.json", RECENT, None)],
            {"7-run.json": run_file("7"), "7-result.json": result_file("7")},
        )
        record = plan.records[f"{RECIPE}/clariden-7.json"]
        assert record["run"]["preset"] == "muon"
        assert record["result"]["status"] == "completed"
        assert sorted(plan.processed) == ["7-result.json", "7-run.json"]

    def test_adds_a_later_result_to_the_existing_record(self):
        first = collect_one("7-run.json", run_file("7"))
        plan = collect(
            [("7-result.json", RECENT, None)],
            {"7-result.json": result_file("7", status="failed", exitCode=1)},
            first.records,
        )
        record = plan.records[f"{RECIPE}/clariden-7.json"]
        assert record["run"]["preset"] == "muon"
        assert record["result"]["status"] == "failed"

    def test_keeps_the_same_job_id_on_another_cluster_apart(self):
        plan = collect(
            [("7-run.json", RECENT, None), ("7-result.json", RECENT, None)],
            {"7-run.json": run_file("7"), "7-result.json": result_file("7", cluster="daint")},
        )
        assert sorted(plan.records) == [
            f"{RECIPE}/clariden-7.json",
            f"{RECIPE}/daint-7.json",
        ]

    def test_keeps_recent_invalid_or_unreadable_files_and_expires_old_invalid_ones(self):
        plan = collect(
            [
                ("1-run.json", RECENT, None),
                ("2-run.json", OLD, None),
                ("3-run.json", OLD, None),
                ("4-run.json", OLD, 10 * 1024 * 1024),
                ("notes.txt", OLD, None),
            ],
            {"1-run.json": "garbage", "2-run.json": "garbage", "4-run.json": run_file("4")},
        )
        assert plan.records == {}
        assert plan.processed == []
        assert [name for name, _ in plan.expired] == ["2-run.json", "4-run.json"]
        # 3 could not be read: retried, never expired; other files are ignored
        assert [name for name, _ in plan.skipped] == ["1-run.json", "3-run.json"]

    def test_rejects_a_result_naming_another_recipe_than_its_run(self):
        plan = collect(
            [("7-run.json", RECENT, None), ("7-result.json", OLD, None)],
            {"7-run.json": run_file("7"), "7-result.json": result_file("7", recipeId="verl-grpo")},
        )
        assert list(plan.records) == [f"{RECIPE}/clariden-7.json"]
        assert plan.records[f"{RECIPE}/clariden-7.json"]["result"] is None
        assert plan.expired == [("7-result.json", "recipe does not match the run's other file")]

    def test_checks_the_recipe_against_records_from_earlier_passes(self):
        first = collect_one("7-run.json", run_file("7"))
        plan = collect(
            [("7-result.json", RECENT, None)],
            {"7-result.json": result_file("7", recipeId="verl-grpo")},
            first.records,
        )
        assert plan.records == {}
        assert [name for name, _ in plan.skipped] == ["7-result.json"]


class TestRunIndex:
    @staticmethod
    def record(job_id, started_at, with_result):
        files = [(f"{job_id}-run.json", RECENT, None)]
        contents = {f"{job_id}-run.json": run_file(job_id, startedAt=started_at)}
        if with_result:
            files.append((f"{job_id}-result.json", RECENT, None))
            contents[f"{job_id}-result.json"] = result_file(job_id)
        return next(iter(collect(files, contents).records.values()))

    def test_reports_runs_without_result_as_running_then_unknown_after_their_time_limit(self):
        assert crr.parse_time_limit("1-02:00:00").total_seconds() == 26 * 3600
        assert crr.parse_time_limit("999999999999-00:00:00") is None
        assert crr.get_run_status(self.record("1", "2026-10-23T08:00:00Z", False), NOW) == "running"
        assert crr.get_run_status(self.record("1", "2026-10-19T08:00:00Z", False), NOW) == "unknown"
        assert crr.get_run_status(self.record("1", "2026-10-19T08:00:00Z", True), NOW) == "completed"

    def test_lists_the_runs_of_one_recipe_newest_first_without_its_own_timestamp(self):
        records = [
            self.record("1", "2026-10-22T08:00:00Z", True),
            self.record("2", "2026-10-23T08:00:00Z", False),
            {**self.record("3", "2026-10-23T09:00:00Z", False), "recipeId": "verl-grpo"},
        ]
        index = crr.build_run_index(RECIPE, records, NOW)
        assert [run["runId"] for run in index["runs"]] == ["clariden-2", "clariden-1"]
        assert index["runs"][1]["path"] == f"{RECIPE}/clariden-1.json"
        assert index["runs"][1]["status"] == "completed"
        assert index["runs"][1]["preset"] == "muon"
        assert index["runs"][1]["hyperparameters"]["MAX_NUM_STEPS"] == 58
        assert index["runs"][1]["elapsedSeconds"] == 4814
        # Rebuilding with nothing new gives the same content, so there is nothing to commit
        assert crr.build_run_index(RECIPE, records, NOW) == index


def test_run_ids_are_the_cluster_and_job_id():
    assert crr.get_run_id("clariden", "501") == "clariden-501"
    assert crr.RUN_ID_PATTERN.match("clariden-501")
    assert not crr.RUN_ID_PATTERN.match("index")


class TestSchedulerState:
    RECORD = {"runId": "clariden-7", "cluster": "clariden", "jobId": "7", "startedOn": "2026-10-23", "result": None}

    @pytest.mark.parametrize(
        ("state", "status"),
        [
            ("COMPLETED", "completed"),
            ("FAILED", "failed"),
            ("OUT_OF_MEMORY", "failed"),
            ("PREEMPTED", "preempted"),
            ("TIMEOUT", "terminated"),
            ("CANCELLED by 1234", "terminated"),
        ],
    )
    def test_maps_end_states_to_run_statuses(self, state, status):
        job = {"status": {"state": state, "exitCode": 1}, "time": {"end": 1793102823, "elapsed": 287}}
        result = crr.apply_scheduler_state(self.RECORD, job)["result"]
        assert result["status"] == status
        # A job that couldn't report gets its end and duration from Slurm
        assert result["endedOn"] == "2026-10-27"
        assert result["elapsedSeconds"] == 287
        assert result["exitCode"] == 1
        assert result["metrics"] is None

    @pytest.mark.parametrize(
        "job",
        [
            {"status": {"state": "RUNNING"}},
            {"status": {"state": "PENDING"}},
            {"status": {"state": "REQUEUED"}},
            {"status": {"state": ""}},
            {"status": "PREEMPTED"},
            None,
            "garbage",
        ],
    )
    def test_jobs_that_have_not_ended_change_nothing(self, job):
        assert crr.apply_scheduler_state(self.RECORD, job) is None

    def test_only_recent_runs_on_this_system_without_a_confirmed_end_are_checked(self):
        now = datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc)
        assert crr.needs_scheduler_check(self.RECORD, "clariden", now)
        assert not crr.needs_scheduler_check(self.RECORD, "daint", now)
        assert not crr.needs_scheduler_check({**self.RECORD, "startedOn": "2026-09-01"}, "clariden", now)
        confirmed = {**self.RECORD, "result": {"status": "preempted", "schedulerState": "PREEMPTED"}}
        assert not crr.needs_scheduler_check(confirmed, "clariden", now)
        # A result the job wrote itself is still confirmed with Slurm
        reported = {**self.RECORD, "result": {"status": "failed"}}
        assert crr.needs_scheduler_check(reported, "clariden", now)


def test_runs_without_a_known_time_limit_eventually_become_unknown():
    record = {"startedOn": "2026-10-01", "run": {"timeLimit": None}, "result": None}
    assert crr.get_run_status(record, NOW) == "unknown"
    assert crr.get_run_status({**record, "startedOn": "2026-10-22"}, NOW) == "running"


class TestFirecrestErrors:
    CONFIG = crr.Config(
        firecrest_url="https://firecrest.invalid",
        token_url="https://auth.invalid/token",
        client_id="client",
        client_secret="not-a-real-secret",
        system="clariden",
        tracker_dir="/tracker",
    )

    @pytest.fixture(autouse=True)
    def no_retry_delays(self, monkeypatch):
        self.delays = []
        monkeypatch.setattr(crr.time, "sleep", self.delays.append)

    @staticmethod
    def respond_with(monkeypatch, outcome):
        class Response:
            def __init__(self, body):
                self.body = body

            def read(self):
                return self.body

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        # One outcome for every attempt, or a list with one per attempt
        outcomes = list(outcome) if isinstance(outcome, list) else None

        def urlopen(request, timeout):
            current = outcomes.pop(0) if outcomes is not None else outcome
            if isinstance(current, Exception):
                raise current
            return Response(current)

        monkeypatch.setattr(crr.urllib.request, "urlopen", urlopen)

    @pytest.mark.parametrize(
        "outcome",
        [
            crr.urllib.error.URLError("no route"),
            TimeoutError(),
            b"<html>gateway error</html>",
            b"[]",
            b'{"token_type": "bearer"}',
        ],
    )
    def test_token_failures_end_the_pass_with_a_short_message(self, monkeypatch, capsys, outcome):
        self.respond_with(monkeypatch, outcome)
        with pytest.raises(SystemExit):
            crr.get_access_token(self.CONFIG)
        error = capsys.readouterr().err
        assert error.startswith("::error::Token")
        assert "not-a-real-secret" not in error and "Traceback" not in error

    @pytest.mark.parametrize(
        "outcome",
        [crr.urllib.error.URLError("no route"), b'{"errorType": "error", "message": "x"}', b"oops"],
    )
    def test_listing_failures_end_the_pass_with_a_short_message(self, monkeypatch, capsys, outcome):
        self.respond_with(monkeypatch, outcome)
        with pytest.raises(SystemExit):
            crr.list_tracker_files(self.CONFIG, "token")
        assert capsys.readouterr().err.startswith("::error::Listing the tracker directory")

    def test_listing_keeps_only_well_formed_tracker_file_entries(self):
        listing = {
            "output": [
                {"name": "1-run.json", "type": "-", "size": "120", "lastModified": RECENT},
                {"name": "2-run.json", "type": "-", "size": 99},
                {"name": "3-run.json", "type": "d"},
                {"name": "notes.txt", "type": "-"},
                {"type": "-"},
                "garbage",
            ]
        }
        files = crr.parse_listing(json.dumps(listing).encode())
        assert [(f.name, f.size, f.last_modified) for f in files] == [
            ("1-run.json", 120, RECENT),
            ("2-run.json", 99, None),
        ]

    def test_reads_a_job_from_slurm(self, monkeypatch):
        self.respond_with(monkeypatch, b'{"jobs": [{"jobId": "7", "status": {"state": "PREEMPTED"}}]}')
        assert crr.get_scheduler_job(self.CONFIG, "token", "7")["status"]["state"] == "PREEMPTED"

    @pytest.mark.parametrize("outcome", [b'{"jobs": []}', b"{}"])
    def test_jobs_slurm_does_not_know_are_none(self, monkeypatch, outcome):
        self.respond_with(monkeypatch, outcome)
        assert crr.get_scheduler_job(self.CONFIG, "token", "7") is None

    def test_a_job_not_found_is_none_and_other_errors_raise(self, monkeypatch):
        self.respond_with(monkeypatch, self.http_error(404))
        assert crr.get_scheduler_job(self.CONFIG, "token", "7") is None
        self.respond_with(monkeypatch, self.http_error(403))
        with pytest.raises(crr.FirecrestError, match="HTTP 403"):
            crr.get_scheduler_job(self.CONFIG, "token", "7")
        self.respond_with(monkeypatch, b"<html>")
        with pytest.raises(crr.FirecrestError, match="unexpected response"):
            crr.get_scheduler_job(self.CONFIG, "token", "7")

    def test_download_failures_are_retried_rather_than_crashing(self, monkeypatch):
        self.respond_with(monkeypatch, TimeoutError())
        with pytest.raises(RuntimeError, match="GET ops/download failed: TimeoutError"):
            crr.firecrest(self.CONFIG, "token", "GET", "download", "/tracker/1-run.json")
        assert self.delays == list(crr.RETRY_DELAYS_SECONDS)

    @staticmethod
    def http_error(code):
        return crr.urllib.error.HTTPError("https://firecrest.example", code, "error", {}, None)

    def test_temporary_failures_are_tried_again(self, monkeypatch):
        listing = b'{"output": [{"name": "1-run.json", "type": "-"}]}'
        self.respond_with(
            monkeypatch,
            [self.http_error(504), crr.urllib.error.URLError("reset"), self.http_error(503), listing],
        )
        files = crr.list_tracker_files(self.CONFIG, "token")
        assert [f.name for f in files] == ["1-run.json"]
        assert self.delays == list(crr.RETRY_DELAYS_SECONDS)

    def test_gateway_errors_fail_after_the_last_attempt(self, monkeypatch, capsys):
        self.respond_with(monkeypatch, self.http_error(504))
        with pytest.raises(SystemExit):
            crr.list_tracker_files(self.CONFIG, "token")
        assert "HTTP 504" in capsys.readouterr().err
        assert self.delays == list(crr.RETRY_DELAYS_SECONDS)

    def test_files_already_gone_count_as_deleted(self, monkeypatch, tmp_path, capsys):
        manifest = TestDeletePhase.manifest(tmp_path, delete=["1-run.json", "1-result.json"])
        monkeypatch.setattr(crr, "get_access_token", lambda config: "token")
        # The first delete times out at the gateway after removing the file
        self.respond_with(monkeypatch, [self.http_error(504), self.http_error(404), b"{}"])
        crr.delete_processed(self.CONFIG, manifest, dry_run=False)
        assert "Deleted 2 of 2" in capsys.readouterr().out

    @pytest.mark.parametrize("code", [401, 403, 404, 500])
    def test_other_http_errors_are_not_tried_again(self, monkeypatch, code):
        self.respond_with(monkeypatch, self.http_error(code))
        with pytest.raises(RuntimeError, match=f"HTTP {code}"):
            crr.firecrest(self.CONFIG, "token", "GET", "download", "/tracker/1-run.json")
        assert self.delays == []


def test_huge_time_limits_are_dropped_instead_of_crashing_the_pass():
    plan = collect_one("7-run.json", run_file("7", launch={**run_file("7")["launch"], "job": {"timeLimit": "999999999999-00:00:00"}}))
    record = plan.records[f"{RECIPE}/clariden-7.json"]
    assert record["run"]["timeLimit"] is None
    assert crr.build_run_index(RECIPE, [record], NOW)["runs"][0]["status"] == "running"


class TestLoadRecords:
    def test_skips_unreadable_or_misplaced_records_with_a_warning(self, tmp_path, capsys):
        good = collect_one("7-run.json", run_file("7")).records
        for path, record in good.items():
            crr.write_json(tmp_path / path, record)
        (tmp_path / RECIPE / "clariden-1.json").write_text("{not json")
        # A record whose file name does not match its run
        crr.write_json(tmp_path / RECIPE / "clariden-2.json", {**next(iter(good.values())), "runId": "clariden-2"})
        (tmp_path / RECIPE / "index.json").write_text("{}")
        assert crr.load_records(tmp_path) == good
        assert "2 existing records could not be read" in capsys.readouterr().out

    def test_a_missing_checkout_ends_the_pass_cleanly(self, tmp_path):
        with pytest.raises(SystemExit):
            crr.load_records(tmp_path / "missing")


class TestDeletePhase:
    CONFIG = crr.Config(
        firecrest_url="https://firecrest.invalid",
        token_url="https://auth.invalid/token",
        client_id="client",
        client_secret="not-a-real-secret",
        system="clariden",
        tracker_dir="/tracker",
    )

    @pytest.fixture()
    def calls(self, monkeypatch):
        calls = []
        monkeypatch.setattr(crr, "get_access_token", lambda config: "token")

        def firecrest(config, access_token, method, operation, path):
            calls.append((method, operation, path))
            if path.endswith("/9-run.json"):
                raise crr.FirecrestError("DELETE ops/rm failed with HTTP 500", 500)
            return b""

        monkeypatch.setattr(crr, "firecrest", firecrest)
        return calls

    @staticmethod
    def manifest(tmp_path, **overrides):
        path = tmp_path / "manifest.json"
        data = {"system": "clariden", "trackerDir": "/tracker", "delete": ["7-run.json", "7-result.json"]}
        crr.write_json(path, {**data, **overrides})
        return path

    def test_deletes_only_the_listed_tracker_files(self, tmp_path, calls, capsys):
        names = ["7-run.json", "../7-run.json", "notes.txt", 42, "7-result.json"]
        crr.delete_processed(self.CONFIG, self.manifest(tmp_path, delete=names), dry_run=False)
        assert calls == [
            ("DELETE", "rm", "/tracker/7-run.json"),
            ("DELETE", "rm", "/tracker/7-result.json"),
        ]
        assert "Deleted 2 of 2 tracker files" in capsys.readouterr().out

    def test_dry_run_deletes_nothing(self, tmp_path, calls, capsys):
        crr.delete_processed(self.CONFIG, self.manifest(tmp_path), dry_run=True)
        assert calls == []
        assert "Would delete 2 tracker files" in capsys.readouterr().out

    @pytest.mark.parametrize(
        "overrides",
        [{"system": "daint"}, {"trackerDir": "/elsewhere"}, {"delete": "7-run.json"}],
    )
    def test_refuses_a_manifest_for_another_system_or_directory(self, tmp_path, calls, overrides):
        with pytest.raises(SystemExit):
            crr.delete_processed(self.CONFIG, self.manifest(tmp_path, **overrides), dry_run=False)
        assert calls == []

    def test_fails_the_step_when_files_could_not_be_deleted(self, tmp_path, calls, capsys):
        with pytest.raises(SystemExit) as exit_info:
            crr.delete_processed(self.CONFIG, self.manifest(tmp_path, delete=["9-run.json", "7-run.json"]), dry_run=False)
        assert exit_info.value.code == 1
        assert len(calls) == 2
        assert "1 tracker files could not be deleted" in capsys.readouterr().err


class TestCollectPhase:
    CONFIG = TestDeletePhase.CONFIG

    @pytest.fixture()
    def tracker(self, monkeypatch):
        """Tracker files by name: bytes, or an exception their download raises."""
        files = {}
        monkeypatch.setattr(crr, "get_access_token", lambda config: "token")
        monkeypatch.setattr(
            crr,
            "list_tracker_files",
            lambda config, token: [crr.TrackerFile(name, None, OLD) for name in files],
        )

        def firecrest(config, access_token, method, operation, path):
            content = files[path.rsplit("/", 1)[1]]
            if isinstance(content, Exception):
                raise content
            return content

        monkeypatch.setattr(crr, "firecrest", firecrest)
        # Slurm's records by job id: a job dict, or an exception the request raises
        self.jobs = {}
        self.job_requests = []

        def get_scheduler_job(config, access_token, job_id):
            self.job_requests.append(job_id)
            job = self.jobs.get(job_id)
            if isinstance(job, Exception):
                raise job
            return job

        monkeypatch.setattr(crr, "get_scheduler_job", get_scheduler_job)
        return files

    @staticmethod
    def started_today():
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def test_slurm_decides_how_a_job_ended(self, tmp_path, tracker, capsys):
        # The job was killed by preemption and recorded the signal as an ordinary failure
        tracker["7-run.json"] = json.dumps(run_file("7", startedAt=self.started_today())).encode()
        tracker["7-result.json"] = json.dumps(
            result_file("7", status="failed", exitCode=143, startedAt=self.started_today())
        ).encode()
        self.jobs["7"] = {"status": {"state": "PREEMPTED", "exitCode": 0}, "time": {"elapsed": 287}}
        crr.collect(self.CONFIG, tmp_path, tmp_path / "manifest.json")
        record = json.loads((tmp_path / RECIPE / "clariden-7.json").read_text())
        assert record["result"]["status"] == "preempted"
        assert record["result"]["schedulerState"] == "PREEMPTED"
        # What the job reported itself is kept
        assert record["result"]["exitCode"] == 143
        index = json.loads((tmp_path / RECIPE / "index.json").read_text())
        assert index["runs"][0]["status"] == "preempted"
        assert "Slurm: 1 runs checked, 1 confirmed as ended" in capsys.readouterr().out

        # Once confirmed, a run isn't asked about again
        self.job_requests.clear()
        tracker.clear()
        crr.collect(self.CONFIG, tmp_path, tmp_path / "manifest.json")
        assert self.job_requests == []

    def test_runs_still_going_or_unknown_to_slurm_keep_waiting(self, tmp_path, tracker, capsys):
        tracker["7-run.json"] = json.dumps(run_file("7", startedAt=self.started_today())).encode()
        tracker["8-run.json"] = json.dumps(run_file("8", startedAt=self.started_today())).encode()
        tracker["9-run.json"] = json.dumps(run_file("9", startedAt=self.started_today())).encode()
        self.jobs["7"] = {"status": {"state": "RUNNING"}}
        self.jobs["9"] = crr.FirecrestError("GET compute/jobs failed with HTTP 500", 500)
        crr.collect(self.CONFIG, tmp_path, tmp_path / "manifest.json")
        index = json.loads((tmp_path / RECIPE / "index.json").read_text())
        assert [run["status"] for run in index["runs"]] == ["running"] * 3
        output = capsys.readouterr().out
        assert "::warning::1 runs could not be checked with Slurm" in output

    def test_retries_files_that_could_not_be_downloaded_and_says_so(self, tmp_path, tracker, capsys):
        tracker["7-run.json"] = json.dumps(run_file("7")).encode()
        tracker["8-run.json"] = RuntimeError("GET ops/download failed: TimeoutError")
        crr.collect(self.CONFIG, tmp_path, tmp_path / "manifest.json")
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        # Even though it is old, the file that couldn't be read is kept for the next pass
        assert manifest["delete"] == ["7-run.json"]
        output = capsys.readouterr().out
        assert "1 skipped" in output
        assert "::warning::1 tracker files could not be downloaded" in output

    def test_expires_old_files_that_are_not_utf8_instead_of_retrying_them_forever(
        self, tmp_path, tracker
    ):
        tracker["9-run.json"] = b"\xff\xfe{garbage"
        crr.collect(self.CONFIG, tmp_path, tmp_path / "manifest.json")
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest["delete"] == ["9-run.json"]
