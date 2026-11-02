#!/usr/bin/env python3
"""Extract the final assistant message from native OpenCode v2 JSONL output."""

import argparse
import json
from pathlib import Path


EVENT_TYPES = {"step_start", "step_finish", "tool_use", "text", "reasoning", "error"}


def extract_review(lines):
    session = None
    message = None
    parts = {}
    has_tools = False
    finish_reason = None
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid OpenCode JSON on line {line_number}") from error
        if not isinstance(event, dict) or event.get("type") not in EVENT_TYPES:
            raise ValueError(f"unsupported OpenCode event on line {line_number}")
        kind = event["type"]
        if kind == "error":
            raise ValueError("OpenCode reported an error; refusing to publish a partial review")
        part = event.get("part")
        if not isinstance(part, dict):
            raise ValueError(f"missing OpenCode event part on line {line_number}")
        event_session = event.get("sessionID")
        if not isinstance(event_session, str) or not event_session:
            raise ValueError("missing OpenCode session ID")
        if session is None:
            session = event_session
        if event_session != session:
            # Child-agent results must not replace the parent review.
            continue
        message_id = part.get("messageID")
        if not isinstance(message_id, str) or not message_id:
            raise ValueError("missing OpenCode message ID")
        # Message IDs delimit responses when the CLI omits explicit step events.
        if kind == "step_start" or message_id != message:
            message = message_id
            parts = {}
            has_tools = False
            finish_reason = None
        if kind == "step_start":
            continue
        if kind == "tool_use":
            has_tools = True
        elif kind == "step_finish":
            finish_reason = part.get("reason")
        elif kind == "text":
            part_id = part.get("id")
            text = part.get("text")
            end = part.get("time", {}).get("end") if isinstance(part.get("time"), dict) else None
            if not isinstance(part_id, str) or not isinstance(text, str):
                raise ValueError("malformed OpenCode text part")
            if type(end) not in (int, float) or end <= 0:
                raise ValueError("OpenCode text part is incomplete")
            parts[part_id] = text
    # Native text parts carry completion times; the runner must also have exited successfully.
    if has_tools or finish_reason not in (None, "stop", "end_turn"):
        raise ValueError("OpenCode stopped before a final review response")
    review = "\n\n".join(parts.values()).strip()
    if not review:
        raise ValueError("OpenCode produced no final review text")
    return review + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.events.resolve() == args.output.resolve():
            raise ValueError("review output must not overwrite the event stream")
        with args.events.open(encoding="utf-8") as events:
            review = extract_review(events)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(review, encoding="utf-8")
    except (OSError, ValueError) as error:
        parser.exit(1, f"review extraction: {error}\n")


if __name__ == "__main__":
    main()
