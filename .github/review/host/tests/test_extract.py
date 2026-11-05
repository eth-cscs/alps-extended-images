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


def phase(name, state, session="parent"):
    return json.dumps({"type": "review_phase", "phase": name, "state": state, "sessionID": session})


def inspection():
    return [phase("inspection", "start"), text("Private inspection notes", "inspection"),
            event("tool_use", "inspection"), phase("inspection", "end")]


def review(final):
    return [*inspection(), phase("final", "start"), *final, phase("final", "end")]


def test_only_explicit_final_parent_response_is_published():
    lines = review([
        event("step_start"),
        text("Child opinion", "child", "child-session"),
        event("reasoning", text="Private reasoning"),
        text("## Findings", part_id="heading"),
        text("- P1: incorrect boundary."),
    ])
    assert extract.extract_review(lines) == "## Findings\n\n- P1: incorrect boundary.\n"


def test_native_final_response_without_step_events_is_accepted():
    assert extract.extract_review(review([text("No actionable findings.")])) == "No actionable findings.\n"


def test_updated_text_part_is_not_duplicated():
    assert extract.extract_review(review([
        event("step_start"), text("Partial"), text("Complete review"),
    ])) == "Complete review\n"


@pytest.mark.parametrize("lines", [
    inspection(),
    [*inspection(), phase("final", "start"), text("Looks complete but was interrupted")],
    [phase("inspection", "start"), text("An inspection review"), phase("final", "start")],
    [*inspection(), phase("final", "start", "different-session"), text("Wrong session")],
    review([text("A final review")]) + [text("Unexpected extra output")],
    [json.dumps({"type": "review_phase", "phase": [], "state": "start"})],
    # An unmarked old-style stream must never bypass explicit finalization.
    [event("step_start"), text("Inspection masquerading as a final review")],
])
def test_missing_reordered_or_mismatched_phases_are_rejected(lines):
    with pytest.raises(ValueError):
        extract.extract_review(lines)


@pytest.mark.parametrize("suffix", [
    [event("tool_use"), text("A later response must not hide the tool attempt", "later")],
    [event("step_finish", reason="length")],
    [event("step_start", "unfinished")],
    [json.dumps({"type": "error", "error": {"message": "provider failed"}})],
])
def test_partial_or_tool_using_final_responses_are_not_published(suffix):
    with pytest.raises(ValueError):
        extract.extract_review(review([event("step_start"), text("Intermediate text"), *suffix]))


def test_inspection_error_cannot_be_hidden_by_a_final_response():
    lines = review([text("An otherwise complete review")])
    lines.insert(2, json.dumps({"type": "error"}))
    with pytest.raises(ValueError):
        extract.extract_review(lines)


@pytest.mark.parametrize("final", [
    [],
    [text("  ")],
    ["not JSON"],
    ["[]"],
    [json.dumps({"type": "unexpected"})],
    [json.dumps({"type": []})],
    [event("text", id="part", text="Incomplete", time={"start": 1})],
])
def test_empty_corrupt_and_incomplete_final_output_is_rejected(final):
    with pytest.raises(ValueError):
        extract.extract_review(review(final))


def test_failed_extraction_preserves_existing_output(tmp_path):
    events = tmp_path / "events.jsonl"
    events.write_text("\n".join(inspection()) + "\n")
    output = tmp_path / "review.md"
    output.write_text("existing review\n")
    with pytest.raises(SystemExit) as error:
        extract.main(["--events", str(events), "--output", str(output)])
    assert error.value.code == 1
    assert output.read_text() == "existing review\n"


def test_cli_cannot_overwrite_event_stream(tmp_path):
    events = tmp_path / "events.jsonl"
    original = "\n".join(review([text("A review")])) + "\n"
    events.write_text(original)
    alias = tmp_path / "alias"
    alias.symlink_to(events)
    with pytest.raises(SystemExit) as error:
        extract.main(["--events", str(events), "--output", str(alias)])
    assert error.value.code == 1
    assert events.read_text() == original
