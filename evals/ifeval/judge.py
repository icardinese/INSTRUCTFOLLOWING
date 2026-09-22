"""Programmatic IFEval scoring -- wraps Google's official evaluation_main.py (vendored under
third_party/microsoft_llm_steer_instruct/ifeval_scripts/), not an LLM judge. Deterministic
instruction-following checks, not a rubric call -- genuinely different scoring MECHANISM from
caveman/judge.py's LLM call, which is exactly why this whole layer is built around a
task-agnostic score_response(row, response) -> dict contract instead of assuming every task uses
an LLM.
"""
import sys
from pathlib import Path

_IFEVAL_SCRIPTS_PARENT = Path("third_party/microsoft_llm_steer_instruct")
if str(_IFEVAL_SCRIPTS_PARENT) not in sys.path:
    sys.path.insert(0, str(_IFEVAL_SCRIPTS_PARENT))

from ifeval_scripts.evaluation_main import InputExample, test_instruction_following_loose  # noqa: E402

# IFEval is NOT a length task -- the instruction is a format/content constraint, and compliance
# is the programmatic checker's verdict. Ranking by avg_tokens (the previous unconditional
# behaviour of select_constrained_survivors) would have selected whichever variant happened to
# emit the shortest text, which is unrelated to following the constraint.
RANK_BY = "primary_desc"     # descending on follow_all_instructions
PRIMARY_FIELD_MAX = 1        # follow_all_instructions is 0/1, not 0-2

SCORE_FIELDS = ["follow_all_instructions", "n_followed", "n_total"]  # exactly what score_response
# returns -- see evals/caveman/judge.py's SCORE_FIELDS for why this is needed.


def score_response(row: dict, response: str) -> dict:
    """row must have "instruction_id_list", "prompt", "kwargs" -- IFEval's real schema, unchanged
    from Google's own format so the vendored harness can consume it directly, no translation layer."""
    inp = InputExample(
        key=row.get("key", row.get("id", 0)),
        instruction_id_list=row["instruction_id_list"],
        prompt=row["prompt"],
        kwargs=row["kwargs"],
    )
    prompt_to_response = {row["prompt"]: response}
    output = test_instruction_following_loose(inp, prompt_to_response)
    return {
        "follow_all_instructions": output.follow_all_instructions,
        # Graded count alongside the boolean -- a partial-credit signal, same reason
        # bootstrap_analysis.py's mean_score beat a pure binary full_correct for statistical power.
        "n_followed": sum(output.follow_instruction_list),
        "n_total": len(output.follow_instruction_list),
    }
