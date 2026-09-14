"""LLM-judged correctness/coherence for caveman code explanations. This is an API call + JSON
parse, not a deterministic function -- genuinely different scoring MECHANISM from ifeval/judge.py's
rule-based checking, which is exactly why score_response(row, response) -> dict is the shared
contract, not the mechanism itself.
"""
import json
import re
import time
from pathlib import Path

from openai import OpenAI

KEY_PATH = Path("openai.key")
JUDGE_MODEL = "gpt-4o-mini"
SCORE_FIELDS = ["correct", "coherent"]  # exactly what score_response returns -- summarize.py
# uses this to safely split "{cond}_{field}" keys back apart, since condition names (e.g.
# "prompt_psr_proper") can themselves contain underscores.

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

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=KEY_PATH.read_text().strip())
    return _client


def score_response(row: dict, response: str) -> dict:
    """row must have "code" and "reference_explanation" -- caveman's real data schema."""
    client = _get_client()
    prompt = RUBRIC.format(code=row["code"], reference_explanation=row["reference_explanation"], candidate=response)
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                max_tokens=50,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.choices[0].message.content.strip()
            match = re.search(r"\{.*\}", text, re.DOTALL)
            score = json.loads(match.group(0))
            return {"correct": score["correct"], "coherent": score["coherent"]}
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2**attempt)
