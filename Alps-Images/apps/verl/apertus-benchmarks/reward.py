import os
import re
import math
from typing import Optional

# Tied to the launch script's ENABLE_THINKING / MAX_RESPONSE_LENGTH (passed
# through via env.toml's [env] table, read back here from the container env):
#   thinking OFF -> max_response_length 500 tokens (~350-400 words max) ->
#     350/700-word thresholds (validated run 3283187: 46/46 steps, healthy).
#   thinking ON  -> max_response_length 8192 tokens (room for a <think> turn)
#     -> 2000/4000-word thresholds (validated run 3284608: 92/92 steps, penalty
#     engaged on genuinely long/degenerate responses without being inert).
_ENABLE_THINKING = os.environ.get("ENABLE_THINKING", "False").strip().lower() == "true"
LENGTH_PENALTY_RAMP_START_WORDS = 2000 if _ENABLE_THINKING else 350
LENGTH_PENALTY_MAX_AT_WORDS = 4000 if _ENABLE_THINKING else 700


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
   length_penalty = -0.2 * min(1.0, max(0.0, (words - LENGTH_PENALTY_RAMP_START_WORDS)
                                         / (LENGTH_PENALTY_MAX_AT_WORDS - LENGTH_PENALTY_RAMP_START_WORDS)))

   return outcome_reward + format_reward + length_penalty
