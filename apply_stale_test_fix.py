#!/usr/bin/env python3
"""Fixes the 7 stale `avg_tokens` KeyError failures in tests/test_layer_hparam_search.py --
handoff v3 section 5's "possibly-stale tests" item.

THE CAUSE: on 2026-09-17, select_constrained_survivors and mark_pareto_frontier switched their
ranking/frontier axis from optimize_score (judged conciseness) to avg_tokens, per finding F6 --
judged conciseness saturates at n=20, ties are the common case, and avg_tokens is continuous and
free. The production code was updated; these tests' hand-built fake candidate dicts were not, so
they still omit the avg_tokens key the code now reads.

Nothing about the production behaviour is wrong here. Only the fixtures are out of date.

THE FIX: give every fake candidate an avg_tokens derived INVERSELY from its optimize_score
(1000 / (1 + optimize_score)), so the two keys rank in the same direction. That matters because
each of these tests asserts that the candidate its own inline comments describe as "more concise"
is the one that wins -- if avg_tokens disagreed with optimize_score, the fixtures would contradict
their stated intent and the tests would be asserting the opposite of what they document.

Worth fixing rather than ignoring: with these 7 failing, a genuine regression introduced by a real
run is invisible inside known noise.

Idempotent. Run from the repo root:

    python3 apply_stale_test_fix.py
    python3 -m pytest tests/test_layer_hparam_search.py -q
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent
TARGET = ROOT / "tests" / "test_layer_hparam_search.py"

if not TARGET.exists():
    sys.exit(f"missing {TARGET.relative_to(ROOT)} -- aborting without changes")

lines = TARGET.read_text().split("\n")
patched = 0
out = []
for line in lines:
    if '"optimize_score":' in line and "avg_tokens" not in line:
        m = re.search(r'"optimize_score": ([0-9.]+)', line)
        if m:
            score = float(m.group(1))
            avg = round(1000.0 / (1.0 + score), 2)
            line = line.replace(m.group(0), f'{m.group(0)}, "avg_tokens": {avg}', 1)
            patched += 1
    out.append(line)

if patched == 0:
    print("no un-patched candidate dicts found -- already fixed, nothing to do")
else:
    TARGET.write_text("\n".join(out))
    print(f"added avg_tokens to {patched} fake candidate dict(s) in "
          f"{TARGET.relative_to(ROOT)}")

print("\nverify with:")
print("  python3 -m pytest tests/test_layer_hparam_search.py -q")
print("  python3 -m pytest tests/ -q --ignore=tests/test_caveman_judge.py")
