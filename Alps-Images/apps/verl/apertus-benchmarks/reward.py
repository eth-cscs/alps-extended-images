import re
import math
from typing import Optional


def extract_model_answer(response: str) -> Optional[str]:
   matches = re.findall(r"\[\[\[\(.*?)\]\]\]", response, re.DOTALL)
   if not matches:
       return None
   raw = matches[-1].strip().replace(",", "")
   try:
       val = float(raw)
       return str(val) if not math.isfinite(val) else (str(int(val)) if val == int(val) else str(val))
   except ValueError:
       return raw




def compute_reward(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> float:
   if "<think>" in solution_str and "</think>" not in solution_str:
       return 0.0

   model_ans = extract_model_answer(solution_str)
   has_answer = "[[[" in solution_str and "]]]" in solution_str
   format_reward  = 0.1 if has_answer else 0.0
   outcome_reward = 1.0 if (model_ans is not None and model_ans == str(ground_truth)) else 0.0
   words = len(solution_str.split())
   length_penalty = -0.2 * min(1.0, max(0.0, (words - 350) / 350))

   return outcome_reward + format_reward + length_penalty
