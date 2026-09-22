#!/usr/bin/env python3
"""Makes Tier 1 / Tier 2 survivor ranking TASK-DECLARED instead of always avg_tokens.

THE BUG: select_constrained_survivors ranked gate-passing candidates by
`(r["avg_tokens"], -r["optimize_score"])` unconditionally. That is correct for caveman -- whose
instruction IS about brevity, making output length its compliance signal -- and wrong for every
task added since:

  - triage has no length axis at all. Ranking by avg_tokens selects whichever variant produced
    the SHORTEST REASONING, which is unrelated to classification accuracy and actively rewards
    degenerate early-stopping output.
  - ifeval is also not a length task: the instruction is a format/content constraint and
    compliance is the programmatic checker's verdict. It has been silently mis-ranked too.

SECOND BUG, same root cause: DEFAULT_PRIMARY_FIELD_MAX = 2 is caveman's 0/1/2 correctness scale.
triage's "correct" and ifeval's "follow_all_instructions" are both 0/1, so _fully_correct_rate
counted scores equal to 2 and returned identically 0 for both.

THE FIX: each eval adapter declares RANK_BY and PRIMARY_FIELD_MAX; layer_hparam_search reads them
via getattr with caveman's historical values as the default, so caveman's completed results are
not retroactively reinterpreted and any adapter predating these attributes is unchanged.

Idempotent. Run from the repo root:

    python3 apply_ranking_axis_fix.py
    python3 -m pytest tests/ -q --ignore=tests/test_caveman_judge.py
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent


def edit(rel_path: str, old: str, new: str, label: str) -> None:
    path = ROOT / rel_path
    if not path.exists():
        sys.exit(f"missing {rel_path} -- aborting without changes")
    text = path.read_text()
    if new.strip().split("\n")[0] in text:
        print(f"  {rel_path}: {label} already present, skipping")
        return
    if old not in text:
        sys.exit(
            f"could not find the expected anchor in {rel_path} for '{label}'.\n"
            f"Your copy of this file differs from what this fix expects -- paste its contents "
            f"back rather than forcing the edit."
        )
    path.write_text(text.replace(old, new, 1))
    print(f"  {rel_path}: {label}")


print("1/4 declaring RANK_BY + PRIMARY_FIELD_MAX per eval adapter")

edit("evals/caveman/judge.py",
     'SCORE_FIELDS = (["correct", "coherent"] if SKIP_CONCISENESS',
     '''# How Tier 1 survivors are ranked once the correctness gates have run, and the top of the
# primary field's scale. Declared per task because the right ranking axis is a property OF THE
# TASK: caveman's instruction is ABOUT brevity, so brevity is its compliance signal and fewer
# tokens legitimately wins. No other task works that way -- see evals/triage/judge.py.
RANK_BY = "avg_tokens"       # ascending; fewer tokens wins
PRIMARY_FIELD_MAX = 2        # "correct" is judged 0/1/2

SCORE_FIELDS = (["correct", "coherent"] if SKIP_CONCISENESS''',
     "RANK_BY=avg_tokens, PRIMARY_FIELD_MAX=2")

edit("evals/ifeval/judge.py",
     'SCORE_FIELDS = ["follow_all_instructions", "n_followed", "n_total"]',
     '''# IFEval is NOT a length task -- the instruction is a format/content constraint, and compliance
# is the programmatic checker's verdict. Ranking by avg_tokens (the previous unconditional
# behaviour of select_constrained_survivors) would have selected whichever variant happened to
# emit the shortest text, which is unrelated to following the constraint.
RANK_BY = "primary_desc"     # descending on follow_all_instructions
PRIMARY_FIELD_MAX = 1        # follow_all_instructions is 0/1, not 0-2

SCORE_FIELDS = ["follow_all_instructions", "n_followed", "n_total"]''',
     "RANK_BY=primary_desc, PRIMARY_FIELD_MAX=1")

edit("evals/triage/judge.py",
     'SCORE_FIELDS = ["correct", "parsed", "predicted"]',
     '''# Triage has NO length axis at all: the instruction governs a classification decision, so the
# only thing worth ranking on is whether the label is right. Ranking by avg_tokens here would
# select the variant producing the shortest REASONING -- unrelated to accuracy, and actively
# rewarding degenerate output that stops early. See select_constrained_survivors.
RANK_BY = "primary_desc"     # descending on "correct"
PRIMARY_FIELD_MAX = 1        # exact match is 0/1

SCORE_FIELDS = ["correct", "parsed", "predicted"]''',
     "RANK_BY=primary_desc, PRIMARY_FIELD_MAX=1")

print("2/4 adding rank_by parameter to select_constrained_survivors")

edit("evals/layer_hparam_search.py",
     "def select_constrained_survivors(results: list[dict], top_k: int) -> list[dict]:",
     '''RANK_BY_AVG_TOKENS = "avg_tokens"
RANK_BY_PRIMARY_DESC = "primary_desc"
RANK_BY_CHOICES = (RANK_BY_AVG_TOKENS, RANK_BY_PRIMARY_DESC)


def select_constrained_survivors(results: list[dict], top_k: int,
                                 rank_by: str = RANK_BY_AVG_TOKENS) -> list[dict]:''',
     "rank_by parameter")

edit("evals/layer_hparam_search.py",
     '''    RANKING KEY among gate-2 survivors, changed 2026-09-17: avg_tokens (ascending -- fewer wins)''',
     '''    RANKING AXIS IS TASK-DECLARED (rank_by), read from the eval adapter's RANK_BY. This is not a
    style preference -- the right axis is a property of the task, and using the wrong one silently
    selects on something unrelated to the instruction being studied:

      - "avg_tokens" (caveman ONLY): caveman's instruction is ABOUT brevity, so output length IS
        its compliance signal and fewer tokens legitimately wins. Rationale below.
      - "primary_desc" (ifeval, triage): descending on the primary judged field. Neither task has
        a length dimension -- IFEval's instruction is a format constraint and triage's is a
        classification rule. Ranking either by avg_tokens would pick whichever variant emitted the
        least text, which for triage means preferring the SHORTEST REASONING and actively
        rewarding degenerate early-stopping output.

    Before 2026-09-22 this function ranked by avg_tokens unconditionally, which was correct for
    the only task that then existed and wrong for both tasks added since.

    THE avg_tokens RATIONALE (caveman): avg_tokens (ascending -- fewer wins)''',
     "docstring for the task-declared axis")

edit("evals/layer_hparam_search.py",
     '''    if prompt_gate_survivors:
        winners = sorted(prompt_gate_survivors, key=lambda r: (r["avg_tokens"], -r["optimize_score"]))[:top_k]
        return [{**r, "prompt_floor_fallback": False} for r in winners]''',
     '''    if rank_by not in RANK_BY_CHOICES:
        raise ValueError(f"unknown rank_by {rank_by!r}; expected one of {list(RANK_BY_CHOICES)}")

    if prompt_gate_survivors:
        if rank_by == RANK_BY_AVG_TOKENS:
            key = lambda r: (r["avg_tokens"], -r["optimize_score"])  # noqa: E731
        else:
            # No length axis exists for this task. Rank on the judged primary field, breaking
            # ties with the stricter fully-correct rate rather than anything length-derived.
            key = lambda r: (-r["judged_score"], -r.get("fully_correct_rate", 0.0))  # noqa: E731
        winners = sorted(prompt_gate_survivors, key=key)[:top_k]
        return [{**r, "prompt_floor_fallback": False} for r in winners]''',
     "the ranking itself")

print("3/4 adding adapter-reading helpers")

edit("evals/layer_hparam_search.py",
     "def load_existing_tiered_results",
     '''def _rank_by_for(eval_adapter) -> str:
    """The task's declared ranking axis. Defaults to avg_tokens -- the historical behaviour, and
    correct for caveman -- so an adapter without RANK_BY is unchanged."""
    return getattr(eval_adapter, "RANK_BY", RANK_BY_AVG_TOKENS)


def _primary_field_max_for(eval_adapter) -> float:
    """Top of the primary judged field's scale, needed by _fully_correct_rate. caveman judges
    correctness 0/1/2; ifeval and triage are both 0/1, so using caveman's 2 for them made
    fully_correct_rate always 0 (nothing ever equals 2)."""
    return getattr(eval_adapter, "PRIMARY_FIELD_MAX", DEFAULT_PRIMARY_FIELD_MAX)


def load_existing_tiered_results''',
     "_rank_by_for / _primary_field_max_for")

print("4/4 threading through call sites")

for old, new, label in [
    ("    survivors = select_constrained_survivors(tier1_results, top_k_layers)",
     "    survivors = select_constrained_survivors(tier1_results, top_k_layers, _rank_by_for(eval_adapter))",
     "tier1 ranking"),
    ("    winner_list = select_constrained_survivors(tier2_results, top_k=1)",
     "    winner_list = select_constrained_survivors(tier2_results, top_k=1, rank_by=_rank_by_for(eval_adapter))",
     "tier2 ranking"),
    ("    winner_list = select_constrained_survivors(tier1_results, top_k=1)",
     "    winner_list = select_constrained_survivors(tier1_results, top_k=1, rank_by=_rank_by_for(eval_adapter))",
     "training-free ranking"),
    ("    primary_field_max: float = DEFAULT_PRIMARY_FIELD_MAX,",
     "    primary_field_max: float | None = None,",
     "primary_field_max default"),
    ("    primary_field = eval_adapter.SCORE_FIELDS[0]",
     '''    # None means "ask the task" -- caveman judges 0/1/2, ifeval and triage are both 0/1, so a
    # single hardcoded max made fully_correct_rate identically zero for the latter two.
    if primary_field_max is None:
        primary_field_max = _primary_field_max_for(eval_adapter)
    primary_field = eval_adapter.SCORE_FIELDS[0]''',
     "primary_field_max resolution"),
]:
    edit("evals/layer_hparam_search.py", old, new, label)

print("\ndone. Verify with:")
print("  python3 -m pytest tests/ -q --ignore=tests/test_caveman_judge.py")
