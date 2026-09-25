"""Triage scoring: label parse + exact match. ZERO API CALLS, ever.

Same role as evals/caveman/judge.py and evals/ifeval/judge.py, and exposes the same two names
(score_response, SCORE_FIELDS) so evals/run_judge.py needs no changes.

WHY THERE IS NO SECOND CRITERION. caveman needs two (correctness as a floor, brevity as
compliance) because brevity can be bought by being wrong, so a tokens-only view hides the
tradeoff -- that was finding F4's correction. Triage collapses them: the instruction being
followed IS the correct classification, and a degenerate or unparseable output simply fails the
match. So accuracy is self-guarding and a separate floor would be measuring the same thing twice.
This is a real structural difference from the other two tasks and should be stated in the paper
rather than presented as a uniform two-criterion design.

UNPARSEABLE COUNTS AS WRONG, deliberately. A stripped prompt that has lost the instruction may
well produce something that isn't a label at all; scoring that as missing-data would silently
drop the hardest cases and flatter the method.
"""
import re

# Triage has NO length axis at all: the instruction governs a classification decision, so the
# only thing worth ranking on is whether the label is right. Ranking by avg_tokens here would
# select the variant producing the shortest REASONING -- unrelated to accuracy, and actively
# rewarding degenerate output that stops early. See select_constrained_survivors.
RANK_BY = "primary_desc"     # descending on "correct"
PRIMARY_FIELD_MAX = 1        # exact match is 0/1

# NUMERIC fields only. evals/summarize.py, evals/bootstrap_analysis.py and evals/plotting.py sum every
# SCORE_FIELDS entry, so the string label "predicted" crashed all three (int + str). score_response
# still returns "predicted" for inspection; it just isn't declared as a score. It can always be
# recomputed from the stored response with parse_label().
SCORE_FIELDS = ["correct", "parsed"]

_LABELS = ("no", "email", "notify")
# Anchored on the required final line first. The fallback scans for a bare label at the very end,
# which catches models that drop the "Triage:" prefix but still answer.
_FINAL_LINE = re.compile(r"triage\s*:\s*[`\"']?\s*(no|email|notify)\b", re.IGNORECASE)
_TRAILING_LABEL = re.compile(r"\b(no|email|notify)\b[\s.`\"']*$", re.IGNORECASE)


def parse_label(response: str) -> str | None:
    """The LAST `Triage: <label>` in the response, or a trailing bare label. Last rather than
    first because the reasoning preamble frequently mentions the label names while thinking out
    loud ("this isn't a notify..."), and the final line is the actual answer."""
    if not response:
        return None
    matches = _FINAL_LINE.findall(response)
    if matches:
        return matches[-1].lower()
    tail = _TRAILING_LABEL.search(response.strip())
    if tail:
        return tail.group(1).lower()
    return None


def score_response(row: dict, response: str) -> dict:
    """row needs "gold" (one of no/email/notify), written by scripts/build_triage_data.py."""
    predicted = parse_label(response)
    gold = (row.get("gold") or "").strip().lower()
    if gold not in _LABELS:
        raise ValueError(
            f"row {row.get('id')!r} has gold={row.get('gold')!r}; expected one of {_LABELS}. "
            f"Run scripts/build_triage_data.py to populate gold labels."
        )
    return {
        "correct": 1 if (predicted is not None and predicted == gold) else 0,
        "parsed": 1 if predicted is not None else 0,
        "predicted": predicted,
    }
