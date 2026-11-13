"""Boxed-answer reward for the Apertus GRPO benchmarks (verl side).

Simplified 2026-09-09 (user request) to a single answer convention, \\boxed{answer}, for BOTH
training and evaluation rows -- the form Apertus produces on its own (confirmed this session:
the model never adopts a bracket-only marker even when RL-trained on it, and the earlier
bracket-vs-boxed train/eval mismatch was found NOT to explain the AIME regression; the real
cause is degenerate repetition on hard problems). dataset_prepare.py builds every split
(gsm8k, dapo_math, aime_2024) with the matching \\boxed{} SYSTEM_PROMPT -- keep the two files in
sync and bump DATASET_PROMPT_VERSION in the launch script on any prompt change.

The final answer is the last \\boxed{...} span in the text OUTSIDE the model's native
deliberation (<|inner_prefix|> ... <|inner_suffix|>), numbers are compared after comma
stripping and float normalisation. An unfinished or malformed deliberation span (e.g. a
response truncated at the length cap while still thinking) earns no reward -- there is no
answer to score, in either mode below.

REWARD_MODE (env, 2026-09-09: A/B test for whether reward SHAPING itself is teaching the model
to answer more confidently-but-wrongly on hard problems -- see the SFT-baseline vs. post-RL
AIME comparison in the session that motivated this):
  shaped (default): outcome (1.0 correct / 0.0 wrong) + a 0.1 format bonus for any \\boxed{}
                     + a length penalty of up to -0.2 ramping from 2000 to 4000 words of the
                     whole response. This is what every run up to and including 3330187/3331427
                     used.
  binary:           outcome only (1.0 / 0.0), no format bonus, no length penalty -- the
                     simplest possible outcome-only RLVR reward. The \\boxed{} extraction logic
                     is unchanged in both modes (it is the only way to know if the answer is
                     correct, not a "format" reward in the shaping sense).

verl calls compute_reward(); it returns {"score": training reward (== outcome in binary mode,
outcome+format+length_penalty in shaped mode), "acc": exact-match 0/1} so val-core/<benchmark>/
acc/* is a clean accuracy in both modes while "score" drives GRPO.
"""

import math
import os
import re
from typing import Optional

REWARD_MODE = os.environ.get("REWARD_MODE", "shaped")  # "shaped" | "binary"
FORMAT_REWARD = 0.1
OUTCOME_REWARD = 1.0
LENGTH_PENALTY_MAX = 0.2
LENGTH_PENALTY_START_WORDS = 2000
LENGTH_PENALTY_SPAN_WORDS = 2000

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


def has_marker(response: str) -> bool:
    return _BOXED + "{" in response


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


def score_boxed_answer(response: str, ground_truth: str) -> dict:
    final_text = completed_final_text(response)
    extracted = extract_boxed_answer(final_text) if final_text is not None else None
    outcome = (
        OUTCOME_REWARD
        if extracted is not None and extracted == normalize_number(str(ground_truth))
        else 0.0
    )
    if REWARD_MODE == "binary":
        return {
            "reward": outcome,
            "outcome": outcome,
            "format": 0.0,
            "length_penalty": 0.0,
            "extracted_answer": extracted,
        }
    format_reward = FORMAT_REWARD if final_text is not None and has_marker(final_text) else 0.0
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
    s = score_boxed_answer(solution_str, ground_truth)
    return {"score": s["reward"], "acc": s["outcome"]}
