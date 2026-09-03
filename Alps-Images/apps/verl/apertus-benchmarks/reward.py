import re
import math
from typing import Optional


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




def compute_reward(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> float:
   if "<inner_prefix>" in solution_str and "<inner_suffix>" not in solution_str:
       return 0.0


   model_ans = extract_model_answer(solution_str)
   has_answer = "[[[" in solution_str and "]]]" in solution_str
   format_reward  = 0.1 if has_answer else 0.0
   outcome_reward = 1.0 if (model_ans is not None and model_ans == _norm(str(ground_truth))) else 0.0
   words = len(solution_str.split())

   LENGTH_PENALTY_RAMP_START_WORDS = 2000
   LENGTH_PENALTY_MAX_AT_WORDS = 4000
   length_penalty = -0.2 * min(1.0, max(0.0, (words - LENGTH_PENALTY_RAMP_START_WORDS)
                                         / (LENGTH_PENALTY_MAX_AT_WORDS - LENGTH_PENALTY_RAMP_START_WORDS)))

   return outcome_reward + format_reward + length_penalty
