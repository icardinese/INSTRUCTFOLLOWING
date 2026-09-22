"""Judge via OpenAI's Batch API instead of synchronous calls.

WHY. The Batch endpoint draws on a SEPARATE rate-limit pool from synchronous RPD/TPM, and costs
50% less. Since judging in this project is already decoupled from generation (responses are saved
to disk and scored later), a batch turnaround costs nothing structurally.

WHAT IT IS NOT. This is not free the way the GPU batching is. generate_batched_uniform is
mathematically identical to looping one row at a time -- a decoder-only transformer has no
cross-row interaction, so batching there is pure throughput with no tradeoff. The Batch API trades
real things:

  - LATENCY: up to 24h. A batch that doesn't finish in its window returns PARTIAL results.
  - ITS OWN LIMIT: there is an enqueued-token cap per model, so a very large submission can be
    rejected. --max-requests splits the work if needed.
  - MAPPING RISK: this is the one that actually matters. Results come back in arbitrary order and
    must be rejoined to rows via custom_id. A silent mis-join would attach the wrong score to the
    wrong response -- far worse than a rate limit, because nothing would look broken. So --collect
    verifies that every submitted custom_id came back, refuses to write partial results unless
    told to, and parses indices strictly rather than by position.

ONLY CORRECTNESS IS SUBMITTED. The conciseness call is skipped here (see
evals/caveman/judge.py's SKIP_CONCISENESS) -- that's the 2x, and it matters more than the batch
discount.

USAGE
  python3 scripts/judge_via_batch.py --submit     # writes requests, uploads, returns a batch id
  python3 scripts/judge_via_batch.py --status     # check progress
  python3 scripts/judge_via_batch.py --collect    # download, verify, write scores back
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters.registry import get_adapter
from evals.bootstrap_analysis import bootstrap_ci
from adapters.registry import TASK_CHOICES

SOURCE = "token_distributions.json"      # canonical record produced by scripts/regen_token_cis.py
STATE = "batch_judge_state.json"         # batch ids + the custom_id manifest
REQUESTS_FILE = "batch_requests.jsonl"
CUSTOM_ID_RE = re.compile(r"^(?P<key>.+)::(?P<idx>\d+)$")


def _client():
    from evals.caveman.judge import _get_client
    return _get_client()


def _rubric_prompt(row, response):
    from evals.caveman.judge import RUBRIC
    return RUBRIC.format(code=row["code"],
                          reference_explanation=row["reference_explanation"],
                          candidate=response)


def submit(task: str, max_requests: int, model: str) -> None:
    adapter = get_adapter(task)
    src = adapter.RESULTS_DIR / SOURCE
    if not src.exists():
        raise FileNotFoundError(f"{src} not found -- run scripts/regen_token_cis.py first.")
    with src.open() as f:
        payload = json.load(f)

    rows_cache = {}
    requests, manifest = [], {}
    for key, entry in payload.items():
        if "correct" in entry:
            continue                       # already judged
        responses = entry.get("responses")
        if not responses:
            continue
        n = len(responses)
        if n not in rows_cache:
            rows_cache[n] = adapter.load_rows("test")[:n]
        rows = rows_cache[n]
        manifest[key] = n
        for i, (row, resp) in enumerate(zip(rows, responses)):
            # custom_id must round-trip to (condition, row index) exactly. "::" is used as the
            # separator because condition keys contain single underscores throughout.
            requests.append({
                "custom_id": f"{key}::{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {"model": model, "temperature": 0, "max_tokens": 50,
                          "messages": [{"role": "user", "content": _rubric_prompt(row, resp)}]},
            })

    if not requests:
        print("nothing to submit -- every condition already has correctness scores")
        return
    print(f"{len(requests)} requests across {len(manifest)} conditions")

    client = _client()
    chunks = [requests[i:i + max_requests] for i in range(0, len(requests), max_requests)]
    batch_ids = []
    for ci, chunk in enumerate(chunks):
        path = adapter.RESULTS_DIR / f"{REQUESTS_FILE}.{ci}"
        with path.open("w") as f:
            for r in chunk:
                f.write(json.dumps(r) + "\n")
        with path.open("rb") as f:
            uploaded = client.files.create(file=f, purpose="batch")
        batch = client.batches.create(input_file_id=uploaded.id,
                                       endpoint="/v1/chat/completions",
                                       completion_window="24h")
        batch_ids.append(batch.id)
        print(f"  chunk {ci}: {len(chunk)} requests -> batch {batch.id}")

    with (adapter.RESULTS_DIR / STATE).open("w") as f:
        json.dump({"task": task, "model": model, "batch_ids": batch_ids,
                    "manifest": manifest, "n_requests": len(requests)}, f, indent=2)
    print(f"\nwrote {adapter.RESULTS_DIR / STATE}")
    print("check with --status, then --collect once completed")


def _load_state(adapter):
    p = adapter.RESULTS_DIR / STATE
    if not p.exists():
        raise FileNotFoundError(f"{p} not found -- run --submit first.")
    with p.open() as f:
        return json.load(f)


def status(task: str) -> None:
    adapter = get_adapter(task)
    state = _load_state(adapter)
    client = _client()
    for bid in state["batch_ids"]:
        b = client.batches.retrieve(bid)
        c = b.request_counts
        print(f"{bid}  status={b.status:<12} completed={c.completed}/{c.total}  failed={c.failed}")


def collect(task: str, allow_partial: bool) -> None:
    adapter = get_adapter(task)
    state = _load_state(adapter)
    client = _client()

    results = {}
    incomplete = []
    for bid in state["batch_ids"]:
        b = client.batches.retrieve(bid)
        if b.status != "completed":
            incomplete.append(f"{bid} ({b.status})")
        if b.error_file_id:
            err = client.files.content(b.error_file_id).text
            n_err = sum(1 for line in err.splitlines() if line.strip())
            print(f"{bid}: {n_err} request(s) errored (see the error file on the API side)")
        if not b.output_file_id:
            continue
        for line in client.files.content(b.output_file_id).text.splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            cid = rec.get("custom_id", "")
            m = CUSTOM_ID_RE.match(cid)
            if not m:
                print(f"  WARNING: unparseable custom_id {cid!r} -- skipped")
                continue
            body = (rec.get("response") or {}).get("body") or {}
            choices = body.get("choices") or []
            if not choices:
                continue
            text = choices[0]["message"]["content"].strip()
            jm = re.search(r"\{.*\}", text, re.DOTALL)
            if not jm:
                print(f"  WARNING: no JSON in response for {cid} -- skipped")
                continue
            results[(m.group("key"), int(m.group("idx")))] = json.loads(jm.group(0))

    if incomplete:
        print(f"\n{len(incomplete)} batch(es) not completed: {', '.join(incomplete)}")
        if not allow_partial:
            print("Refusing to write partial results. Re-run --collect later, or pass "
                  "--allow-partial to write only the fully-covered conditions.")
            return

    src = adapter.RESULTS_DIR / SOURCE
    with src.open() as f:
        payload = json.load(f)

    written, skipped = 0, []
    for key, n in state["manifest"].items():
        if key not in payload:
            skipped.append(f"{key} (no longer in {SOURCE})")
            continue
        # STRICT COVERAGE CHECK: every index 0..n-1 must be present. A missing index would
        # otherwise shift nothing (dict lookup, not positional) but WOULD silently shorten the
        # score list relative to avg_tokens, breaking the pairing this whole file exists to fix.
        got = [results.get((key, i)) for i in range(n)]
        missing = [i for i, g in enumerate(got) if g is None]
        if missing:
            skipped.append(f"{key} ({len(missing)}/{n} missing)")
            continue
        scores = [g["correct"] for g in got]
        pt, lo, hi = bootstrap_ci(scores)
        payload[key].update(correct=pt, correct_ci_lo=lo, correct_ci_hi=hi,
                             correctness_scores=scores,
                             coherent_rate=sum(1.0 if g.get("coherent") else 0.0 for g in got) / n,
                             judged_via="batch_api")
        written += 1
        print(f"{key:<38} {payload[key]['avg_tokens']:6.1f} tok   correct={pt:.3f} [{lo:.3f}, {hi:.3f}]")

    with src.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote correctness for {written}/{len(state['manifest'])} conditions into {src}")
    if skipped:
        print("not written (incomplete coverage -- rerun --collect when the batch finishes):")
        for s in skipped:
            print(f"  {s}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="caveman", choices=TASK_CHOICES)
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--allow-partial", action="store_true",
                     help="with --collect, write conditions that are fully covered even if some batch is still running")
    ap.add_argument("--max-requests", type=int, default=20000,
                     help="split into multiple batches at this size (the Batch endpoint has its own enqueued limit)")
    ap.add_argument("--model", default=None, help="defaults to evals/caveman/judge.py's JUDGE_MODEL")
    a = ap.parse_args()

    model = a.model
    if model is None:
        from evals.caveman.judge import JUDGE_MODEL
        model = JUDGE_MODEL

    if a.submit:
        submit(a.task, a.max_requests, model)
    elif a.status:
        status(a.task)
    elif a.collect:
        collect(a.task, a.allow_partial)
    else:
        ap.error("pass one of --submit / --status / --collect")
