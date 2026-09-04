import os
import re
import math
from typing import Optional


_ENABLE_THINKING = os.environ.get("ENABLE_THINKING", "False").strip().lower() == "true"
LENGTH_PENALTY_RAMP_START_WORDS = 2000 if _ENABLE_THINKING else 1400
LENGTH_PENALTY_MAX_AT_WORDS = 4000 if _ENABLE_THINKING else 2800

def _norm(raw: str) -> str:
   raw = raw.strip().replace(",", "")
   try:
       val = float(raw)
       return str(val) if not math.isfinite(val) else (str(int(val)) if val == int(val) else str(val))
   except ValueError:
       return raw


def extract_model_answer(response: str) -> Optional[str]:
   matches = re.findall(r"\[\[\[(.*?)\]\]\]", response, re.DOTALL)
   return _norm(matches[-1]) if matches else None




def compute_reward(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> dict:
   # Returning a dict (score + acc) instead of a bare float, so verl's naive
   # reward manager (verl/experimental/reward_loop/reward_manager/naive.py)
   # logs a genuinely clean 0/1 "acc" alongside the shaped training reward,
   # instead of aliasing the whole shaped score as "acc" (its behavior for a
   # bare-float return) -- val-core/{data_source}/acc/mean@1 was showing
   # values like 1.1/0.9/-0.005 because of that, not comparable to a
   # normalized eval like NeMo's. "score" (unchanged) still drives GRPO/PPO
   # advantage computation; "acc" is exact-match-only, for eval comparability.
   if "<inner_prefix>" in solution_str and "<inner_suffix>" not in solution_str:
       return {"score": 0.0, "acc": 0.0}


   model_ans = extract_model_answer(solution_str)
   has_answer = "[[[" in solution_str and "]]]" in solution_str
   format_reward  = 0.1 if has_answer else 0.0
   outcome_reward = 1.0 if (model_ans is not None and model_ans == _norm(str(ground_truth))) else 0.0
   words = len(solution_str.split())
   length_penalty = -0.2 * min(1.0, max(0.0, (words - LENGTH_PENALTY_RAMP_START_WORDS)
                                         / (LENGTH_PENALTY_MAX_AT_WORDS - LENGTH_PENALTY_RAMP_START_WORDS)))

   return {"score": outcome_reward + format_reward + length_penalty, "acc": outcome_reward}
