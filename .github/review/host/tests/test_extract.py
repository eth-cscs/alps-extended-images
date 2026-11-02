import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("review_extract", Path(__file__).resolve().parents[1] / "extract.py")
extract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extract)


def event(kind, message="final", session="parent", **part):
    return json.dumps({"type": kind, "sessionID": session,
                       "part": {"messageID": message, **part}})


def text(value, message="final", session="parent", part_id="text", **changes):
    return event("text", message, session, id=part_id, text=value, time={"end": 1}, **changes)


def test_only_final_parent_response_is_published():
    lines = [
        event("step_start", "inspection"),
        text("I will inspect this.", "inspection"),
        event("tool_use", "inspection", state={"output": "not review text"}),
        event("step_finish", "inspection", reason="tool-calls"),
        event("step_start"),
        event("step_start", "child", "child-session"),
        text("Child opinion", "child", "child-session"),
        event("reasoning", text="Private reasoning"),
        text("## Findings", part_id="heading"),
        text("- P1: incorrect boundary."),
    ]
    assert extract.extract_review(lines) == "## Findings\n\n- P1: incorrect boundary.\n"


@pytest.mark.parametrize("inspection", [
    [],
    [text("I will inspect this.", "inspection"), event("tool_use", "inspection")],
])
def test_native_response_without_step_events_publishes_only_final_text(inspection):
    assert extract.extract_review([*inspection, text("Final findings")]) == "Final findings\n"


def test_updated_text_part_is_not_duplicated():
    assert extract.extract_review([
        event("step_start"), text("Partial"), text("Complete review"),
    ]) == "Complete review\n"


@pytest.mark.parametrize("suffix", [
    [event("tool_use"), event("step_finish", reason="tool-calls")],
    [event("step_finish", reason="length")],
    [event("step_start", "unfinished")],
    [json.dumps({"type": "error", "error": {"message": "provider failed"}})],
])
def test_partial_reviews_are_not_published(suffix):
    with pytest.raises(ValueError):
        extract.extract_review([event("step_start"), text("Intermediate text"), *suffix])


@pytest.mark.parametrize("lines", [
    [],
    [event("step_start"), text("  ")],
    ["not JSON"],
    ["[]"],
    [json.dumps({"type": "unexpected"})],
    [event("step_start"), event("text", id="part", text="Incomplete", time={"start": 1})],
])
def test_empty_corrupt_and_incomplete_output_is_rejected(lines):
    with pytest.raises(ValueError):
        extract.extract_review(lines)


def test_failed_extraction_preserves_existing_output(tmp_path):
    events = tmp_path / "events.jsonl"
    events.write_text(json.dumps({"type": "error"}) + "\n")
    output = tmp_path / "review.md"
    output.write_text("existing review\n")
    with pytest.raises(SystemExit) as error:
        extract.main(["--events", str(events), "--output", str(output)])
    assert error.value.code == 1
    assert output.read_text() == "existing review\n"


def test_cli_cannot_overwrite_event_stream(tmp_path):
    events = tmp_path / "events.jsonl"
    original = event("step_start") + "\n" + text("A review") + "\n"
    events.write_text(original)
    alias = tmp_path / "alias"
    alias.symlink_to(events)
    with pytest.raises(SystemExit) as error:
        extract.main(["--events", str(events), "--output", str(alias)])
    assert error.value.code == 1
    assert events.read_text() == original
