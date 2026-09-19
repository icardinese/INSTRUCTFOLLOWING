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
import os
import re
import time
from pathlib import Path

from openai import OpenAI

KEY_PATH = Path("openai.key")
JUDGE_MODEL = "gpt-4o-mini"
CONCISENESS_JUDGE_MODEL = "gpt-4o-mini"
# Halves API usage when set: score_response makes TWO sequential calls per response (correctness
# +coherence, then conciseness), so skipping the second is an exact 2x on every judged run. Safe
# for most of this project's purposes because avg_tokens is the primary length metric and the
# judged conciseness score was measured to SATURATE at n=20 (many layers tying at exactly 1.0 with
# zero variance -- the failure that made the tiered search pick wrong layers). Set
# JUDGE_SKIP_CONCISENESS=1 to drop it.
#
# Deliberately NOT merged into one call instead: the conciseness rubric says "Do NOT grade
# correctness here" and is not shown the reference explanation, so merging would condition
# conciseness on the reference and bias it. Skipping is safe; merging is not.
SKIP_CONCISENESS = os.environ.get("JUDGE_SKIP_CONCISENESS", "0") == "1"

SCORE_FIELDS = (["correct", "coherent"] if SKIP_CONCISENESS
                else ["correct", "coherent", "conciseness"])  # exactly what score_response returns --
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

_clients = None


def _load_keys() -> list[str]:
    """Every API key available, in order. Reads openai.key plus any openai.key2, openai.key3...
    and OPENAI_API_KEYS (comma-separated).

    Rate limits are per-ORGANIZATION, not per-machine or per-process, so running the judge
    elsewhere does not raise the ceiling -- but a key from a DIFFERENT org has its own quota, and
    rotating across keys genuinely multiplies the daily request budget. Keys from the same org
    give no benefit; that is not detectable here, so it's on the caller to know.
    """
    keys, seen = [], set()

    def add(k: str) -> None:
        k = k.strip()
        if k and k not in seen:
            seen.add(k)
            keys.append(k)

    if KEY_PATH.exists():
        add(KEY_PATH.read_text())
    for i in range(2, 9):
        p = Path(f"openai.key{i}")
        if p.exists():
            add(p.read_text())
    for k in os.environ.get("OPENAI_API_KEYS", "").split(","):
        add(k)
    if not keys:
        raise FileNotFoundError("no API key found (looked for openai.key, openai.key2..., OPENAI_API_KEYS)")
    return keys


def _get_clients() -> list[OpenAI]:
    global _clients
    if _clients is None:
        ks = _load_keys()
        _clients = [OpenAI(api_key=k) for k in ks]
        if len(_clients) > 1:
            print(f"[judge] {len(_clients)} API keys loaded -- rotating on rate-limit errors")
    return _clients


def _get_client() -> OpenAI:
    """Kept for callers that just want one client; returns the first."""
    return _get_clients()[0]


def _suggested_wait(err: Exception) -> float | None:
    """Pulls OpenAI's own suggested wait out of a rate-limit error message ("Please try again in
    8.64s" / "in 1m26.4s"). Using the server's number instead of a guess matters because the two
    limit types behave completely differently: a TOKENS-PER-MINUTE limit refills within ~60s, so
    waiting works; a REQUESTS-PER-DAY limit is a rolling 24h window where the suggestion is just
    "time until one slot frees" and retrying in a loop will not get you far."""
    msg = str(err)
    m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", msg)
    if not m:
        return None
    minutes = float(m.group(1)) if m.group(1) else 0.0
    return minutes * 60.0 + float(m.group(2))


def _is_rate_limit(err: Exception) -> bool:
    return err.__class__.__name__ == "RateLimitError" or "rate_limit" in str(err).lower()


def _call_judge_json(client: OpenAI, model: str, prompt: str, max_tokens: int = 50,
                      n_attempts: int = 3, rate_limit_attempts: int = 8) -> dict:
    """Shared retry-with-backoff + JSON-extraction logic, used by both the correctness/coherence
    call and the conciseness call -- factored out so a fix to one (e.g. a more robust regex)
    can't accidentally apply to only one of the two judge calls.

    Rate limits get their OWN, much more patient retry budget, separate from the malformed-response
    budget. The original 3 attempts at 2**attempt seconds (1s, 2s) was calibrated for a flaky JSON
    response and is hopeless against a tokens-per-minute cap, which needs up to a full minute to
    refill: it would exhaust all three attempts in three seconds and then raise, killing a run
    that would have succeeded after one short wait. Malformed-JSON failures still fail fast --
    there's no reason to wait a minute for a response that was simply unparseable."""
    rate_limit_hits = 0
    attempt = 0
    key_idx = 0
    while True:
        try:
            resp = client.chat.completions.create(
                model=model, max_tokens=max_tokens, temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.choices[0].message.content.strip()
            match = re.search(r"\{.*\}", text, re.DOTALL)
            return json.loads(match.group(0))
        except Exception as err:
            if _is_rate_limit(err):
                rate_limit_hits += 1
                if rate_limit_hits >= rate_limit_attempts:
                    raise
                # Try the NEXT key before sleeping: if it belongs to a different organization it
                # has an independent quota, so the wait may be unnecessary entirely. Only sleep
                # once every key has been tried for this attempt.
                clients = _get_clients()
                if len(clients) > 1:
                    key_idx = (key_idx + 1) % len(clients)
                    client = clients[key_idx]
                    if rate_limit_hits % len(clients) != 0:
                        continue
                # Trust the server's suggestion when it gives one; otherwise assume a per-minute
                # window and wait out a full one. The +2s margin avoids retrying a hair early and
                # burning another attempt on the same window.
                wait = _suggested_wait(err)
                wait = min(wait + 2.0, 90.0) if wait is not None else min(15.0 * rate_limit_hits, 75.0)
                time.sleep(wait)
                continue
            attempt += 1
            if attempt >= n_attempts:
                raise
            time.sleep(2**attempt)


def score_response(row: dict, response: str) -> dict:
    """row must have "code" and "reference_explanation" -- caveman's real data schema. Two
    independent judge calls (see module docstring for why they're separate, not merged): a
    failure in one doesn't prevent the other's score from being returned."""
    client = _get_client()

    correctness_prompt = RUBRIC.format(code=row["code"], reference_explanation=row["reference_explanation"], candidate=response)
    correctness_score = _call_judge_json(client, JUDGE_MODEL, correctness_prompt)

    if SKIP_CONCISENESS:
        return {"correct": correctness_score["correct"], "coherent": correctness_score["coherent"]}

    conciseness_prompt = CONCISENESS_RUBRIC.format(code=row["code"], candidate=response)
    conciseness_score = _call_judge_json(client, CONCISENESS_JUDGE_MODEL, conciseness_prompt)

    return {
        "correct": correctness_score["correct"],
        "coherent": correctness_score["coherent"],
        "conciseness": conciseness_score["conciseness"],
    }
