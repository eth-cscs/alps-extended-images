"""Marked-answer reward for the Apertus GRPO benchmarks (verl side).

Kept in lockstep with the NeMo-RL side of the framework benchmark
(nemo_rl/environments/bracket_math_reward.py in Alvorecer721/Nemo-RL): the final answer is
the last marker span in the text OUTSIDE the model's native deliberation
(<|inner_prefix|> ... <|inner_suffix|>), numbers are compared after comma stripping and float
normalisation, a 0.1 format bonus rewards any marker, and a length penalty of up to -0.2 ramps
from 2000 to 4000 words of the whole response. An unfinished or malformed deliberation span
(e.g. a response truncated at the length cap while still thinking) earns neither outcome nor
format reward.

The marker is chosen per data_source: the AIME-2024 evaluation set (data_source "aime_2024",
see BOXED_DATA_SOURCES) is prompted for and scored with \\boxed{answer}, the form Apertus
produces on its own; every other source (gsm8k, dapo_math training rows) uses [[[answer]]],
the benchmark's original convention, unless ANSWER_MARKER=boxed is set in the container env
(like ENABLE_THINKING / BENCHMARK). dataset_prepare.py builds the matching SYSTEM_PROMPT per
split -- keep both in sync and bump DATASET_PROMPT_VERSION in the launch script on any prompt
change.

verl calls compute_reward(); it returns {"score": shaped training reward, "acc": exact-match
0/1} so val-core/<benchmark>/acc/* is a clean accuracy while "score" drives GRPO.
"""

import math
import os
import re
from typing import Optional

ANSWER_MARKER = os.environ.get("ANSWER_MARKER", "bracket").strip().lower()  # default marker
if ANSWER_MARKER not in ("bracket", "boxed"):
    raise ValueError(f"ANSWER_MARKER must be 'bracket' or 'boxed', got {ANSWER_MARKER!r}")
# data_sources that are always prompted for / scored with \boxed{} (the AIME-2024 eval set).
BOXED_DATA_SOURCES = frozenset(
    x.strip() for x in os.environ.get("BOXED_DATA_SOURCES", "aime_2024").split(",") if x.strip()
)


def marker_for(data_source) -> str:
    return "boxed" if str(data_source) in BOXED_DATA_SOURCES else ANSWER_MARKER

FORMAT_REWARD = 0.1
OUTCOME_REWARD = 1.0
LENGTH_PENALTY_MAX = 0.2
LENGTH_PENALTY_START_WORDS = 2000
LENGTH_PENALTY_SPAN_WORDS = 2000

_BRACKET_ANSWER = re.compile(r"\[\[\[(.*?)\]\]\]", re.DOTALL)
_BOXED = "\\boxed"
_LATEX_WRAPPERS = re.compile(r"\\(?:text|textbf|mathrm|mathbf)\{([^{}]*)\}")
_LATEX_NOISE = re.compile(r"\\left|\\right|\\[$%,;!]|[$~]")
_THINK_SPLIT = re.compile(r"(<\|inner_prefix\|>|<\|inner_suffix\|>)")


def normalize_number(raw: str) -> str:
    raw = raw.strip().replace(",", "")
    try:
        value = float(raw)
    except ValueError:
        return raw
    if not math.isfinite(value):
        return str(value)
    return str(int(value)) if value == int(value) else str(value)


def extract_bracket_answer(response: str) -> Optional[str]:
    matches = _BRACKET_ANSWER.findall(response)
    return normalize_number(matches[-1]) if matches else None


def _last_boxed_content(response: str) -> Optional[str]:
    start = response.rfind(_BOXED + "{")
    if start < 0:
        return None
    open_brace = response.find("{", start)
    if open_brace < 0:
        return None
    depth = 0
    for index in range(open_brace, len(response)):
        if response[index] == "{":
            depth += 1
        elif response[index] == "}":
            depth -= 1
            if depth == 0:
                return response[open_brace + 1 : index]
    return None


def _unwrap_numeric_latex(match: "re.Match[str]") -> str:
    """Keep wrapped numbers while dropping textual unit suffixes."""
    content = match.group(1)
    try:
        float(content.replace(",", ""))
    except ValueError:
        return ""
    return content


def extract_boxed_answer(response: str) -> Optional[str]:
    content = _last_boxed_content(response)
    if content is None:
        return None
    content = _LATEX_NOISE.sub("", content)
    content = _LATEX_WRAPPERS.sub(_unwrap_numeric_latex, content)
    return normalize_number(content)


def has_marker(response: str, marker: str = ANSWER_MARKER) -> bool:
    if marker == "bracket":
        return "[[[" in response and "]]]" in response
    return _BOXED + "{" in response


def extract_answer(response: str, marker: str = ANSWER_MARKER) -> Optional[str]:
    if marker == "bracket":
        return extract_bracket_answer(response)
    return extract_boxed_answer(response)


def completed_final_text(response: str) -> Optional[str]:
    """Return text outside completed native deliberation, rejecting bad spans."""
    visible = []
    in_thinking = False
    for part in _THINK_SPLIT.split(response):
        if part == "<|inner_prefix|>":
            if in_thinking:
                return None
            in_thinking = True
        elif part == "<|inner_suffix|>":
            if not in_thinking:
                return None
            in_thinking = False
        elif not in_thinking:
            visible.append(part)
    return None if in_thinking else "".join(visible)


def score_marked_answer(response: str, ground_truth: str, marker: str = ANSWER_MARKER) -> dict:
    final_text = completed_final_text(response)
    extracted = extract_answer(final_text, marker) if final_text is not None else None
    format_reward = FORMAT_REWARD if final_text is not None and has_marker(final_text, marker) else 0.0
    outcome = (
        OUTCOME_REWARD
        if extracted is not None and extracted == normalize_number(str(ground_truth))
        else 0.0
    )
    words = len(response.split())
    overflow = (words - LENGTH_PENALTY_START_WORDS) / LENGTH_PENALTY_SPAN_WORDS
    length_penalty = -LENGTH_PENALTY_MAX * min(1.0, max(0.0, overflow))
    return {
        "reward": outcome + format_reward + length_penalty,
        "outcome": outcome,
        "format": format_reward,
        "length_penalty": length_penalty,
        "extracted_answer": extracted,
    }


def compute_reward(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> dict:
    s = score_marked_answer(solution_str, ground_truth, marker_for(data_source))
    return {"score": s["reward"], "acc": s["outcome"]}
