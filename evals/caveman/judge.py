"""LLM-judged correctness/coherence/conciseness for caveman code explanations. This is an API
call + JSON parse, not a deterministic function -- genuinely different scoring MECHANISM from
ifeval/judge.py's rule-based checking, which is exactly why score_response(row, response) -> dict
is the shared contract, not the mechanism itself.

Conciseness is judged by a SEPARATE call, from a SEPARATE rubric, from the correctness/coherence
call -- deliberately, not folded into it, even though both currently use the same model
(gpt-4o-mini, kept cheap on purpose -- CONCISENESS_JUDGE_MODEL is its own constant specifically so
this can be pointed at a stronger model later without touching the correctness/coherence call):
- Correctness needs the reference explanation as ground truth; conciseness doesn't (a maximally
  concise explanation isn't necessarily "as concise as the reference," and grading against a
  reference conflates "matches the reference's wording" with "isn't padded"). Mixing the two into
  one rubric would contaminate whichever one is asked second by anchoring the model to the other.
- If the conciseness call fails after retries, correctness/coherence scoring for the SAME response
  still succeeds independently (two try/except boundaries, not one) -- see score_response.
"""
import json
import re
import time
from pathlib import Path

from openai import OpenAI

KEY_PATH = Path("openai.key")
JUDGE_MODEL = "gpt-4o-mini"
CONCISENESS_JUDGE_MODEL = "gpt-4o-mini"
SCORE_FIELDS = ["correct", "coherent", "conciseness"]  # exactly what score_response returns --
# summarize.py uses this to safely split "{cond}_{field}" keys back apart, since condition names
# (e.g. "prompt_psr_proper") can themselves contain underscores.

RUBRIC = """You are grading an automatically generated explanation of a Python function.

Function:
```python
{code}
```

Reference explanation (written by the original developer, for grading only):
{reference_explanation}

Candidate explanation to grade:
{candidate}

Score the candidate explanation on two axes:
- "correct": 0 if it is wrong or misleading about what the function does, 1 if it is vague or only partially
  correct, 2 if it correctly captures the function's actual behavior (wording may differ from the reference).
- "coherent": false if the text is degenerate, repetitive, non-English gibberish, or so garbled it fails to
  read as a genuine explanation; true otherwise.

Respond with ONLY a JSON object, no other text: {{"correct": <0|1|2>, "coherent": <true|false>}}"""

# Deliberately "contrived": conciseness has no ground truth to check against the way correctness
# does (a reference explanation), so the rubric has to manufacture its own concrete anchor instead
# of asking a vague "is this concise?" -- the caveman framing gives the judge a specific, blunt
# persona to grade against (fewest words, no hedging, no repetition) rather than an open-ended
# aesthetic judgment call it would otherwise have to invent per-response.
CONCISENESS_RUBRIC = """You are grading a candidate explanation of a Python function purely on how
CONCISE it is. Do NOT grade correctness here -- that is scored separately, on a different call.
Judge it the way a caveman would talk: a caveman uses the fewest possible words, never repeats
himself, never hedges ("it seems to", "essentially", "basically", "in other words"), and never
explains something nobody asked about.

Function (context only, so you know what's actually necessary to mention):
```python
{code}
```

Candidate explanation to grade:
{candidate}

Score the candidate's CONCISENESS on this scale:
- 0: padded -- repeats itself, hedges, restates the question, or pads with detail/caveats a
  caveman would never bother saying.
- 1: reasonably tight, but has at least one avoidable wordy or redundant phrase.
- 2: caveman-blunt -- every word does work; nothing repeated, nothing padded, nothing hedged.

Respond with ONLY a JSON object, no other text: {{"conciseness": <0|1|2>}}"""

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=KEY_PATH.read_text().strip())
    return _client


def _call_judge_json(client: OpenAI, model: str, prompt: str, max_tokens: int = 50, n_attempts: int = 3) -> dict:
    """Shared retry-with-backoff + JSON-extraction logic, used by both the correctness/coherence
    call and the conciseness call -- factored out so a fix to one (e.g. a more robust regex)
    can't accidentally apply to only one of the two judge calls."""
    for attempt in range(n_attempts):
        try:
            resp = client.chat.completions.create(
                model=model, max_tokens=max_tokens, temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.choices[0].message.content.strip()
            match = re.search(r"\{.*\}", text, re.DOTALL)
            return json.loads(match.group(0))
        except Exception:
            if attempt == n_attempts - 1:
                raise
            time.sleep(2**attempt)


def score_response(row: dict, response: str) -> dict:
    """row must have "code" and "reference_explanation" -- caveman's real data schema. Two
    independent judge calls (see module docstring for why they're separate, not merged): a
    failure in one doesn't prevent the other's score from being returned."""
    client = _get_client()

    correctness_prompt = RUBRIC.format(code=row["code"], reference_explanation=row["reference_explanation"], candidate=response)
    correctness_score = _call_judge_json(client, JUDGE_MODEL, correctness_prompt)

    conciseness_prompt = CONCISENESS_RUBRIC.format(code=row["code"], candidate=response)
    conciseness_score = _call_judge_json(client, CONCISENESS_JUDGE_MODEL, conciseness_prompt)

    return {
        "correct": correctness_score["correct"],
        "coherent": correctness_score["coherent"],
        "conciseness": conciseness_score["conciseness"],
    }
