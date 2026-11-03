"""Build the train/test parquet files for the Apertus GRPO benchmarks.

Selected by the BENCHMARK env var (set by the launch script):
  gsm8k      (default) openai/gsm8k main split -> data/gsm8k/{train,test}.parquet
  dapo-math  BytedTsinghua-SIA/DAPO-Math-17k (train) + BytedTsinghua-SIA/AIME-2024 (test)
             -> data/dapo-math/{train,test}.parquet

Simplified 2026-09-09 (user request) to a single \\boxed{N} SYSTEM_PROMPT for every split
(gsm8k, dapo_math training rows, aime_2024 eval rows) -- matching reward.py, which now scores
only \\boxed{}. The earlier bracket/boxed marker split was found not to explain the AIME
regression (the model never adopted the bracket marker even after bracket-only RL training).
The DAPO parquets on the Hub are pre-replicated for DAPO's multi-epoch sampling (DAPO-Math-17k
has ~1.79M rows for ~17k problems; AIME-2024 has 960 rows for 30 problems), so both are
de-duplicated by problem text here; per-problem repeats for the AIME avg@k evaluation come from
rollout.val_kwargs.n in the launch script instead. Every DAPO/AIME row wraps the problem in a fixed
"Answer: $Answer" instruction, which is stripped so only the problem statement goes into the
user turn.
"""
import re
import os
import pandas as pd
from pathlib import Path

# Single answer marker, all splits (2026-09-09; was per-split/env-selectable). Bump
# DATASET_PROMPT_VERSION in the launch script on any prompt change.
SYSTEM_PROMPT = """You are a precise math solver.
Solve the problem step by step, then give your final answer inside \\boxed{}.

Example:
\\boxed{42}"""

DAPO_PREFIX = (
    "Solve the following math problem step by step. The last line of your response should be of "
    "the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
)
DAPO_SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'


def extract_ground_truth(solution: str) -> str:
    """Pull the number after #### from a GSM8K solution string."""
    match = re.search(r"####\s*([\d,\-\.]+)", solution)
    return match.group(1).replace(",", "").strip() if match else ""


def make_prompt(question: str, system_prompt: str = SYSTEM_PROMPT) -> list:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": question},
    ]


def dapo_problem_text(item: dict) -> str:
    """Recover the bare problem statement from a DAPO-Math / AIME-2024 row."""
    raw = (item.get("extra_info") or {}).get("raw_problem")
    if raw:
        return raw.strip()
    content = item["prompt"][0]["content"]
    if content.startswith(DAPO_PREFIX):
        content = content[len(DAPO_PREFIX):]
    if content.endswith(DAPO_SUFFIX):
        content = content[: -len(DAPO_SUFFIX)]
    return content.strip()


def dapo_rows(items, data_source: str, system_prompt: str = SYSTEM_PROMPT) -> tuple[list, int, int]:
    """Convert DAPO-style rows to the benchmark format, de-duplicated by problem text.

    Returns (rows, n_duplicates_dropped, n_skipped_no_answer)."""
    rows, seen, dups, skipped = [], set(), 0, 0
    for item in items:
        problem = dapo_problem_text(item)
        gt = str((item.get("reward_model") or {}).get("ground_truth", "")).strip()
        if not problem or not gt:
            skipped += 1
            continue
        if problem in seen:
            dups += 1
            continue
        seen.add(problem)
        rows.append({
            "prompt": make_prompt(problem, system_prompt),
            "data_source": data_source,
            "reward_model": {"ground_truth": gt},
        })
    return rows, dups, skipped


def write_parquet(rows: list, output_path: str, label: str, note: str = "") -> None:
    df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"[{label}] Saved {len(df)} rows -> {output_path}{note}")


def prepare_gsm8k(split: str, output_path: str):
    import datasets

    training_home = os.environ.get("TRAINING_HOME", ".")
    raw_path = os.path.join(training_home, "data/gsm8k_raw")

    if os.path.exists(raw_path):
        print(f"Loading {split} from local cache: {raw_path}")
        ds = datasets.load_from_disk(raw_path)[split]
    else:
        print(f"Downloading {split} from HuggingFace...")
        ds = datasets.load_dataset("openai/gsm8k", "main", split=split)

    rows = []
    skipped = 0
    for item in ds:
        gt = extract_ground_truth(item["answer"])
        if not gt:
            skipped += 1
            continue
        rows.append({
            "prompt": make_prompt(item["question"]),
            "data_source": "gsm8k",
            "reward_model": {"ground_truth": gt},
        })
    write_parquet(rows, output_path, split, f" (skipped {skipped})")


def prepare_dapo_math(split: str, output_path: str):
    import datasets

    if split == "train":
        hub, data_source = "BytedTsinghua-SIA/DAPO-Math-17k", "dapo_math"
    else:
        hub, data_source = "BytedTsinghua-SIA/AIME-2024", "aime_2024"
    print(f"Downloading {hub} from HuggingFace...")
    ds = datasets.load_dataset(hub, split="train")
    rows, dups, skipped = dapo_rows(ds, data_source)
    write_parquet(rows, output_path, split, f" ({data_source}; {dups} replicated rows dropped, {skipped} skipped)")


if __name__ == "__main__":
    training_home = os.environ.get("TRAINING_HOME", ".")
    benchmark = os.environ.get("BENCHMARK", "gsm8k")
    if benchmark == "gsm8k":
        prepare_gsm8k("train", os.path.join(training_home, "data/gsm8k/train.parquet"))
        prepare_gsm8k("test",  os.path.join(training_home, "data/gsm8k/test.parquet"))
    elif benchmark == "dapo-math":
        prepare_dapo_math("train", os.path.join(training_home, "data/dapo-math/train.parquet"))
        prepare_dapo_math("test",  os.path.join(training_home, "data/dapo-math/test.parquet"))
    else:
        raise SystemExit(f"unknown BENCHMARK {benchmark!r} (expected gsm8k or dapo-math)")
