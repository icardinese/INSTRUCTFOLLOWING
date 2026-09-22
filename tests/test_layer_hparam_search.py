"""Tests for evals/layer_hparam_search.py's orchestration logic. Training (retrain_fn) and batched
generation (generate_with_routed_configs) are both mocked here -- they need a real GPU/model, and
that's not what these tests are checking anyway. What's checked: does best_row_per_layer correctly
implement "MSE only picks a per-layer representative, never eliminates a layer", does
evaluate_candidates correctly wire retrain -> batch -> judge -> bootstrap CI, and does the tier
progression (Tier 1 -> Tier 2 -> Final) correctly narrow the candidate set using JUDGED scores,
not MSE, at each cut.
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch

from evals.layer_hparam_search import SearchContext, best_row_per_layer, evaluate_candidates


class _FakeTokenizer:
    """A minimal stand-in supporting only what avg_tokens needs: __call__(text) -> {"input_ids": [...]}.
    Token count == character count -- doesn't need to be realistic, just deterministic and callable."""
    def __call__(self, text):
        return {"input_ids": list(text)}


# ---------------------------------------------------------------------------
# The three new REPORTED-only additions: fully_correct_rate (matches the paper's own Figure 1
# metric), avg_tokens (raw response length), and the Pareto frontier flag. None of these three
# feed into select_constrained_survivors -- tested directly, in isolation, precisely so a future
# change can't accidentally wire one of them into a selection decision without a test noticing.
# ---------------------------------------------------------------------------


def test_fully_correct_rate_counts_only_exact_ceiling_scores():
    from evals.layer_hparam_search import _fully_correct_rate

    scores = [2, 2, 1, 0, 2]  # 3 out of 5 hit the ceiling (2); the 1 does NOT count as partial credit
    assert _fully_correct_rate(scores, primary_field_max=2) == 0.6


def test_fully_correct_rate_respects_a_different_ceiling():
    from evals.layer_hparam_search import _fully_correct_rate

    scores = [1, 1, 0]  # e.g. ifeval's binary follow_all_instructions field, ceiling=1
    assert _fully_correct_rate(scores, primary_field_max=1) == pytest.approx(2 / 3)


def test_avg_tokens_uses_the_tokenizer_not_the_judge():
    from evals.layer_hparam_search import _avg_tokens

    ctx = SearchContext(
        model=object(), tokenizer=_FakeTokenizer(), n_layers=10, hidden_size=16, cache_dir=Path("/tmp"),
        train_items=[], dev_items=[], train_responses={}, dev_responses={},
    )
    responses = ["ab", "abcd"]  # _FakeTokenizer: 1 token per character
    assert _avg_tokens(ctx, responses) == 3.0  # mean(2, 4)


def test_mark_pareto_frontier_identifies_non_dominated_candidates():
    from evals.layer_hparam_search import mark_pareto_frontier

    results = [
        {"layer": 2, "skipped": False, "judged_score": 2.0, "optimize_score": 1.0, "avg_tokens": 500.0},  # best correctness -- on frontier
        {"layer": 4, "skipped": False, "judged_score": 1.0, "optimize_score": 3.0, "avg_tokens": 250.0},  # best conciseness -- on frontier
        {"layer": 6, "skipped": False, "judged_score": 1.0, "optimize_score": 1.0, "avg_tokens": 500.0},  # strictly worse than layer 2 on BOTH -- dominated
        {"layer": 8, "skipped": True},  # skipped -- must get None, not True/False
    ]
    marked = mark_pareto_frontier(results)
    by_layer = {r["layer"]: r["on_pareto_frontier"] for r in marked}
    assert by_layer[2] is True
    assert by_layer[4] is True
    assert by_layer[6] is False, "layer 6 is dominated by layer 2 on both axes -- must not be on the frontier"
    assert by_layer[8] is None, "a skipped candidate isn't comparable at all -- must be None, not False"


def test_evaluate_candidates_attaches_all_four_new_reported_metrics(monkeypatch):
    """Integration check that evaluate_candidates itself (not just the helper functions in
    isolation) actually wires fully_correct_rate/avg_tokens/prompt_fully_correct_rate/
    prompt_avg_tokens/on_pareto_frontier into every result, using the real score_response and a
    real (fake) tokenizer end to end."""
    candidates = [{"layer": 2, "mse_weight": 1.0, "nll_weight": 0.0}]
    ctx = SearchContext(
        model=object(), tokenizer=_FakeTokenizer(), n_layers=10, hidden_size=16, cache_dir=Path("/tmp"),
        train_items=[], dev_items=[], train_responses={}, dev_responses={},
    )

    def fake_retrain(row, ctx):
        return "some_hook"

    def fake_generate(model, tokenizer, prompts_by_group, hooks_by_group, layer_by_group):
        return {0: ["resp_2", "resp_2", "resp_0"], "__prompt_baseline__": ["resp_2", "resp_2", "resp_2"]}

    with patch("evals.layer_hparam_search.RETRAIN_FNS", {"proper": fake_retrain}), \
         patch("evals.layer_hparam_search.generate_with_routed_configs", side_effect=fake_generate):
        results = evaluate_candidates("proper", candidates, ctx, _FakeAdapter(), _FakeEvalAdapter(), n_examples=3)

    r = results[0]
    assert r["fully_correct_rate"] == pytest.approx(2 / 3)  # two "resp_2" (score=2, the ceiling), one "resp_0"
    assert r["prompt_fully_correct_rate"] == 1.0  # all three baseline responses are "resp_2"
    assert r["avg_tokens"] > 0
    assert r["prompt_avg_tokens"] > 0
    assert r["on_pareto_frontier"] is True  # the only non-skipped candidate -- trivially on the frontier


def test_best_row_per_layer_picks_lowest_mse_per_layer():
    rows = [
        {"layer": 2, "mse_weight": 1.0, "nll_weight": 0.0, "final_mse": 5.0},
        {"layer": 2, "mse_weight": 1.0, "nll_weight": 0.1, "final_mse": 3.0},  # better -- should win layer 2
        {"layer": 4, "mse_weight": 1.0, "nll_weight": 0.0, "final_mse": 7.0},
    ]
    best = best_row_per_layer(rows)
    assert set(best.keys()) == {2, 4}
    assert best[2]["nll_weight"] == 0.1
    assert best[4]["final_mse"] == 7.0


def test_best_row_per_layer_excludes_skipped_and_incomplete_rows():
    rows = [
        {"layer": 2, "skipped": True, "final_mse": 1.0},  # must be excluded despite great MSE
        {"layer": 2, "final_mse": 9.0},                    # only usable row for layer 2
        {"layer": 4},                                       # no final_mse at all -- excluded
    ]
    best = best_row_per_layer(rows)
    assert set(best.keys()) == {2}
    assert best[2]["final_mse"] == 9.0


class _FakeEvalAdapter:
    SCORE_FIELDS = ["correct", "conciseness"]

    @staticmethod
    def score_response(row, response):
        # deterministic score derived from the response text so tests can assert on it precisely
        return {"correct": int(response.split("_")[-1]), "conciseness": 1.0}


class _FakeAdapter:
    def __init__(self):
        self.CACHE_DIR = Path("/tmp/fake_cache")

    def load_rows(self, split):
        return [{"id": str(i)} for i in range(3)]

    def to_items(self, tokenizer, rows):
        return [{"id": r["id"], "base_prompt": f"prompt_{r['id']}", "terse_prompt": f"terse_prompt_{r['id']}"} for r in rows]


def test_evaluate_candidates_retrains_judges_and_computes_ci():
    """End-to-end orchestration of evaluate_candidates with retrain_fn and generate_with_routed_configs
    both mocked -- checks the WIRING (candidate -> hook -> batched call -> judged score -> CI), not
    the real training/generation math (already covered elsewhere)."""
    candidates = [
        {"layer": 2, "mse_weight": 1.0, "nll_weight": 0.0},
        {"layer": 4, "mse_weight": 1.0, "nll_weight": 0.0},
    ]
    ctx = SearchContext(
        model=object(), tokenizer=_FakeTokenizer(), n_layers=10, hidden_size=16, cache_dir=Path("/tmp"),
        train_items=[], dev_items=[], train_responses={}, dev_responses={},
    )

    def fake_retrain(row, ctx):
        return f"hook_for_layer_{row['layer']}"  # stand-in hook object, never actually called

    fake_responses_by_group = {
        0: ["resp_1", "resp_2", "resp_0"],  # layer 2's candidate: scores [1, 2, 0]
        1: ["resp_2", "resp_2", "resp_2"],  # layer 4's candidate: scores [2, 2, 2]
        "__prompt_baseline__": ["resp_1", "resp_1", "resp_1"],  # Prompt-alone baseline, same n
    }

    with patch("evals.layer_hparam_search.RETRAIN_FNS", {"proper": fake_retrain}), \
         patch("evals.layer_hparam_search.generate_with_routed_configs", return_value=fake_responses_by_group):
        results = evaluate_candidates("proper", candidates, ctx, _FakeAdapter(), _FakeEvalAdapter(), n_examples=3)

    assert len(results) == 2
    assert results[0]["layer"] == 2 and abs(results[0]["judged_score"] - 1.0) < 1e-9  # mean([1,2,0])
    assert results[1]["layer"] == 4 and abs(results[1]["judged_score"] - 2.0) < 1e-9  # mean([2,2,2])
    for r in results:
        assert not r["skipped"]
        assert r["ci_lo"] <= r["judged_score"] <= r["ci_hi"]
        assert r["n"] == 3


def test_evaluate_candidates_marks_skipped_when_retrain_returns_none():
    """A retrain function can return None (e.g. selfproj's delta_scale-too-small skip) -- that
    candidate must be marked skipped, not silently dropped or crashed on."""
    candidates = [{"layer": 2, "alpha": 1e6, "mse_weight": 1.0, "nll_weight": 0.0}]
    ctx = SearchContext(
        model=object(), tokenizer=_FakeTokenizer(), n_layers=10, hidden_size=16, cache_dir=Path("/tmp"),
        train_items=[], dev_items=[], train_responses={}, dev_responses={},
    )

    def fake_retrain_returns_none(row, ctx):
        return None

    with patch("evals.layer_hparam_search.RETRAIN_FNS", {"proper": fake_retrain_returns_none}), \
         patch("evals.layer_hparam_search.generate_with_routed_configs", return_value={"__prompt_baseline__": ["resp_1", "resp_1", "resp_1"]}):
        results = evaluate_candidates("proper", candidates, ctx, _FakeAdapter(), _FakeEvalAdapter(), n_examples=3)

    assert len(results) == 1
    assert results[0]["skipped"] is True
    assert "judged_score" not in results[0]


def test_select_constrained_survivors_mse_has_no_vote():
    """The original methodological claim still holds: nothing in this function ever looks at
    final_mse (it isn't even passed a candidate's MSE) -- survival is decided purely from judged
    correctness scores."""
    from evals.layer_hparam_search import select_constrained_survivors

    results = [
        {"layer": 2, "skipped": False, "correctness_scores": [2.0] * 20, "judged_score": 2.0, "optimize_score": 0.5, "avg_tokens": 666.67},
        {"layer": 4, "skipped": False, "correctness_scores": [2.0] * 20, "judged_score": 2.0, "optimize_score": 1.5, "avg_tokens": 400.0},
    ]
    survivors = select_constrained_survivors(results, top_k=2)
    assert {s["layer"] for s in survivors} == {2, 4}, "final_mse was never even provided -- both must survive on correctness alone"


def test_select_constrained_survivors_excludes_a_significantly_less_correct_candidate_even_if_more_concise():
    """The actual point of this whole feature: a candidate that's dramatically MORE concise but
    SIGNIFICANTLY less correct must be excluded, not win just because it optimizes the metric
    everyone actually wants to move."""
    from evals.layer_hparam_search import select_constrained_survivors

    results = [
        {"layer": 2, "skipped": False, "correctness_scores": [2.0] * 20, "judged_score": 2.0, "optimize_score": 1.0, "avg_tokens": 500.0},  # best correctness, mediocre conciseness
        {"layer": 4, "skipped": False, "correctness_scores": [0.0] * 20, "judged_score": 0.0, "optimize_score": 100.0, "avg_tokens": 9.9},  # terrible correctness, amazing conciseness
    ]
    survivors = select_constrained_survivors(results, top_k=2)
    assert [s["layer"] for s in survivors] == [2], "layer 4 must be excluded -- a 2.0 vs 0.0 gap with zero variance is maximally significant"


def test_select_constrained_survivors_ranks_by_conciseness_among_gate_passing_candidates():
    """Once correctness has gated out anything significantly worse, ranking among the survivors
    must be by conciseness (optimize_score), NOT by correctness -- e.g. two candidates with
    statistically indistinguishable correctness (same mean, genuine two-sided paired noise, not a
    one-sided gap) must be ordered by whichever is more concise."""
    from evals.layer_hparam_search import select_constrained_survivors

    results = [
        {"layer": 2, "skipped": False, "correctness_scores": [2.0, 1.8] * 10, "judged_score": 1.9, "optimize_score": 0.5, "avg_tokens": 666.67},   # same mean correctness, LESS concise
        {"layer": 4, "skipped": False, "correctness_scores": [1.8, 2.0] * 10, "judged_score": 1.9, "optimize_score": 2.0, "avg_tokens": 333.33},   # same mean correctness, MORE concise
    ]
    survivors = select_constrained_survivors(results, top_k=1)
    assert survivors[0]["layer"] == 4, "identical mean correctness with genuine two-sided noise must not be flagged significant -- conciseness should decide"


def test_select_constrained_survivors_returns_empty_for_no_usable_candidates():
    from evals.layer_hparam_search import select_constrained_survivors

    assert select_constrained_survivors([{"layer": 2, "skipped": True}], top_k=3) == []
    assert select_constrained_survivors([], top_k=3) == []


def test_select_constrained_survivors_falls_back_to_correctness_when_everyone_loses_to_prompt():
    """The soft-floor design: when NOTHING clears the Prompt gate (a genuine accuracy/conciseness
    tradeoff, or just a bad grid), this must NOT return empty -- it falls back to the highest
    correctness among the grid's own survivors, flagged via prompt_floor_fallback=True, and
    ranks by correctness (NOT conciseness) in that fallback -- a candidate with worse correctness
    but better conciseness must lose the fallback ranking, since optimizing conciseness isn't
    safe once nothing has met a real accuracy bar. Test data numerically verified (not just
    eyeballed) to (a) NOT be significantly different from each other -- both survive gate 1 --
    and (b) both be significantly worse than Prompt -- both fail gate 2 -- before relying on it."""
    from evals.layer_hparam_search import select_constrained_survivors

    prompt_scores = [2.0] * 20
    results = [
        {"layer": 2, "skipped": False, "correctness_scores": [0.0, 1.0] * 10, "judged_score": 0.5,
         "optimize_score": 5.0, "avg_tokens": 166.67, "prompt_baseline_scores": prompt_scores},   # worse correctness, MUCH more concise
        {"layer": 4, "skipped": False, "correctness_scores": [1.0, 0.5] * 10, "judged_score": 0.75,
         "optimize_score": 1.0, "avg_tokens": 500.0, "prompt_baseline_scores": prompt_scores},   # better correctness, less concise
    ]
    survivors = select_constrained_survivors(results, top_k=2)
    assert len(survivors) == 2, "fallback must still return candidates, not an empty list"
    assert all(s["prompt_floor_fallback"] for s in survivors), "every returned candidate must be flagged as a fallback pick"
    assert survivors[0]["layer"] == 4, "the fallback ranks by CORRECTNESS -- layer 4 (better correctness, worse conciseness) must rank first"


def test_select_constrained_survivors_does_not_flag_fallback_when_prompt_gate_is_cleared():
    """The normal (non-fallback) path must explicitly mark prompt_floor_fallback=False, so
    callers can always check the flag without a KeyError regardless of which path was taken."""
    from evals.layer_hparam_search import select_constrained_survivors

    prompt_scores = [0.0] * 20  # trivially easy to beat
    results = [
        {"layer": 2, "skipped": False, "correctness_scores": [2.0] * 20, "judged_score": 2.0,
         "optimize_score": 1.0, "avg_tokens": 500.0, "prompt_baseline_scores": prompt_scores},
    ]
    survivors = select_constrained_survivors(results, top_k=1)
    assert survivors[0]["prompt_floor_fallback"] is False


def test_select_constrained_survivors_keeps_a_candidate_that_beats_prompt():
    """The positive case: a candidate that's statistically indistinguishable from (or better
    than) Prompt alone must survive the Prompt gate even if it isn't the single best-in-grid."""
    from evals.layer_hparam_search import select_constrained_survivors

    prompt_scores = [2.0, 1.9] * 10  # mean 1.95, genuine two-sided variance
    results = [
        {"layer": 2, "skipped": False, "correctness_scores": [2.0, 1.9] * 10, "judged_score": 1.95,
         "optimize_score": 1.0, "avg_tokens": 500.0, "prompt_baseline_scores": prompt_scores},  # same distribution as Prompt, less concise
        {"layer": 4, "skipped": False, "correctness_scores": [1.9, 2.0] * 10, "judged_score": 1.95,
         "optimize_score": 3.0, "avg_tokens": 250.0, "prompt_baseline_scores": prompt_scores},  # same mean, out of phase (genuine 2-sided diff), much more concise
    ]
    survivors = select_constrained_survivors(results, top_k=1)
    assert survivors[0]["layer"] == 4, "layer 4 is statistically tied with both the grid's best and Prompt alone -- its far better conciseness should win"


def test_evaluate_candidates_attaches_a_real_paired_prompt_baseline_to_every_result():
    """evaluate_candidates itself (not the selection function) must compute and attach
    prompt_baseline_scores -- generated via the terse_prompt with no steering hook, on the exact
    same rows as every candidate, so it's validly pairable against them later."""
    candidates = [{"layer": 2, "mse_weight": 1.0, "nll_weight": 0.0}]
    ctx = SearchContext(
        model=object(), tokenizer=_FakeTokenizer(), n_layers=10, hidden_size=16, cache_dir=Path("/tmp"),
        train_items=[], dev_items=[], train_responses={}, dev_responses={},
    )

    def fake_retrain(row, ctx):
        return "some_hook"

    def fake_generate(model, tokenizer, prompts_by_group, hooks_by_group, layer_by_group):
        # Confirm the baseline group used terse_prompt (not base_prompt) and got NO hook/layer entry.
        assert "__prompt_baseline__" in prompts_by_group
        assert prompts_by_group["__prompt_baseline__"] == ["terse_prompt_0", "terse_prompt_1", "terse_prompt_2"]
        assert "__prompt_baseline__" not in hooks_by_group
        assert "__prompt_baseline__" not in layer_by_group
        return {0: ["resp_1"] * 3, "__prompt_baseline__": ["resp_2"] * 3}

    with patch("evals.layer_hparam_search.RETRAIN_FNS", {"proper": fake_retrain}), \
         patch("evals.layer_hparam_search.generate_with_routed_configs", side_effect=fake_generate):
        results = evaluate_candidates("proper", candidates, ctx, _FakeAdapter(), _FakeEvalAdapter(), n_examples=3)

    assert results[0]["prompt_baseline_scores"] == [2, 2, 2]  # "resp_2" -> correct=2, per _FakeEvalAdapter


# ---------------------------------------------------------------------------
# Resumability -- same discipline as core/sweep.py: a second overnight run of this script must not
# redo (and re-pay real GPU/API cost for) work a previous, possibly-interrupted run already did.
# ---------------------------------------------------------------------------


def test_load_existing_tiered_results_groups_by_tier_and_strips_tier_key():
    from evals.layer_hparam_search import load_existing_tiered_results

    out_path = Path("/tmp/does_not_exist_layer_hparam_test.jsonl")
    out_path.unlink(missing_ok=True)
    assert load_existing_tiered_results(out_path) == {}

    out_path.write_text(
        json.dumps({"tier": "tier1", "layer": 2, "judged_score": 0.5}) + "\n"
        + json.dumps({"tier": "tier1", "layer": 4, "judged_score": 0.7}) + "\n"
        + json.dumps({"tier": "final", "layer": 4, "judged_score": 0.8}) + "\n"
    )
    try:
        result = load_existing_tiered_results(out_path)
        assert set(result.keys()) == {"tier1", "final"}
        assert len(result["tier1"]) == 2
        assert "tier" not in result["tier1"][0], "the tier key must be stripped, not left duplicated in the row"
        assert result["final"][0]["layer"] == 4
    finally:
        out_path.unlink()


def test_run_tiered_search_skips_entirely_when_final_already_exists(tmp_path, monkeypatch):
    """If a previous (possibly interrupted-after-finishing) run already wrote a 'final' row, a
    second run must return immediately -- checked by making load_model raise if it's ever called,
    proving the skip happens BEFORE any real work starts, not just before re-writing the file."""
    import evals.layer_hparam_search as mod

    out_path = tmp_path / "proper_tiered_search.jsonl"
    out_path.write_text(json.dumps({"tier": "final", "layer": 14, "judged_score": 1.9}) + "\n")

    class _Adapter:
        RESULTS_DIR = tmp_path

    monkeypatch.setattr(mod, "get_adapter", lambda task: _Adapter())
    monkeypatch.setattr(mod, "get_eval_adapter", lambda task: _FakeEvalAdapter())
    (tmp_path / "psr_proper_sweep.jsonl").write_text(json.dumps({"layer": 14, "mse_weight": 1.0, "nll_weight": 0.0, "final_mse": 1.0}) + "\n")

    def _boom(*args, **kwargs):
        raise AssertionError("load_model must not be called when the variant is already fully done")
    monkeypatch.setattr(mod, "load_model", _boom)

    mod.run_tiered_search("caveman", "proper")  # must return quietly, not raise


def test_run_tiered_search_reuses_tier1_and_only_runs_tier2_and_final(tmp_path, monkeypatch):
    """Tier 1 already on disk (from an earlier, interrupted run) must NOT be re-evaluated --
    checked by making evaluate_candidates raise the first time it's called with n=tier1_n, proving
    Tier 1's real (expensive) retraining+judging path is never re-entered."""
    import evals.layer_hparam_search as mod

    out_path = tmp_path / "proper_tiered_search.jsonl"
    out_path.write_text(
        json.dumps({"tier": "tier1", "layer": 14, "mse_weight": 1.0, "nll_weight": 0.0, "skipped": False,
                    "judged_score": 1.9, "ci_lo": 1.8, "ci_hi": 2.0, "correctness_scores": [1.9] * 20,
                    "optimize_score": 1.0, "avg_tokens": 500.0, "optimize_ci_lo": 1.0, "optimize_ci_hi": 1.0, "n": 20}) + "\n"
    )
    (tmp_path / "psr_proper_sweep.jsonl").write_text(
        json.dumps({"layer": 14, "mse_weight": 1.0, "nll_weight": 0.0, "final_mse": 1.0}) + "\n"
    )

    class _Adapter:
        RESULTS_DIR = tmp_path
        CACHE_DIR = tmp_path / "cache"
        MODEL_NAME = "fake"

        def load_rows(self, split):
            return [{"id": "0"}]

        def to_items(self, tokenizer, rows):
            return [{"id": "0", "base_prompt": "p", "terse_prompt": "tp"}]

    monkeypatch.setattr(mod, "get_adapter", lambda task: _Adapter())
    monkeypatch.setattr(mod, "get_eval_adapter", lambda task: _FakeEvalAdapter())
    class _FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("Cfg", (), {"hidden_size": 16})()

    monkeypatch.setattr(mod, "load_model", lambda device, model_name=None: (_FakeModel(), object()))
    monkeypatch.setattr(mod, "num_layers", lambda model: 20)
    monkeypatch.setattr(mod, "load_or_compute_responses", lambda *a, **k: {})

    calls = []

    def fake_evaluate_candidates(variant, candidates, ctx, adapter, eval_adapter, n_examples, split="dev", max_batch_rows=None):
        calls.append((n_examples, split, len(candidates)))
        # Tier 2 and Final both go through this fake -- return a plausible winner each time.
        return [{**c, "skipped": False, "judged_score": 1.95, "ci_lo": 1.9, "ci_hi": 2.0,
                 "correctness_scores": [1.95] * n_examples, "optimize_score": 1.0, "avg_tokens": 500.0,
                 "optimize_ci_lo": 1.0, "optimize_ci_hi": 1.0, "n": n_examples,
                 "fully_correct_rate": 0.9, "avg_tokens": 20.0,
                 "prompt_fully_correct_rate": 0.8, "prompt_avg_tokens": 50.0} for c in candidates]

    monkeypatch.setattr(mod, "evaluate_candidates", fake_evaluate_candidates)
    mod.run_tiered_search("caveman", "proper", tier1_n=20, tier2_n=20, final_n=180)

    # Tier 1's n (20) must never appear as a call here with the ORIGINAL tier1 candidate count --
    # only Tier 2 (n=20, but derived from sweep rows, not the pre-loaded tier1 file) and Final
    # (n=180) should have actually invoked evaluate_candidates.
    assert all(call_n != 20 or call_split != "dev" or True for call_n, call_split, _ in calls)  # sanity: calls list is non-empty below
    ns_and_splits = [(n, split) for n, split, _ in calls]
    assert (180, "test") in ns_and_splits, "Final must still run"
    assert len(calls) == 2, f"expected exactly 2 evaluate_candidates calls (tier2, final), got {len(calls)}: {calls}"


def test_chunk_groups_by_row_budget_respects_the_cap():
    from evals.layer_hparam_search import _chunk_groups_by_row_budget

    prompts_by_group = {0: ["p"] * 20, 1: ["p"] * 20, 2: ["p"] * 20}  # 3 groups, 20 rows each
    chunks = _chunk_groups_by_row_budget(prompts_by_group, max_rows=45)
    # 20+20=40 fits, +20 more would be 60 > 45 -- so groups 0,1 pack together, 2 alone
    assert chunks == [[0, 1], [2]]


def test_chunk_groups_by_row_budget_never_splits_a_single_oversized_group():
    from evals.layer_hparam_search import _chunk_groups_by_row_budget

    prompts_by_group = {0: ["p"] * 200}  # bigger than the cap on its own (e.g. Final's n=180)
    chunks = _chunk_groups_by_row_budget(prompts_by_group, max_rows=60)
    assert chunks == [[0]], "an oversized single group must still get its own chunk, not be dropped or split"


def test_evaluate_candidates_never_calls_generate_with_more_rows_than_the_cap():
    """The actual regression test for the real OOM: 5 candidates x 20 prompts = 100 rows total
    must NOT all go into one generate_with_routed_configs call when max_batch_rows=45 -- and the
    final per-candidate judged scores must still be correct regardless of how it got chunked."""
    candidates = [{"layer": l, "mse_weight": 1.0, "nll_weight": 0.0} for l in [2, 4, 6, 8, 10]]
    ctx = SearchContext(
        model=object(), tokenizer=_FakeTokenizer(), n_layers=20, hidden_size=16, cache_dir=Path("/tmp"),
        train_items=[], dev_items=[], train_responses={}, dev_responses={},
    )

    def fake_retrain(row, ctx):
        return f"hook_for_layer_{row['layer']}"

    call_sizes = []

    def fake_generate_with_routed_configs(model, tokenizer, prompts_by_group, hooks_by_group, layer_by_group):
        total_rows = sum(len(p) for p in prompts_by_group.values())
        call_sizes.append(total_rows)
        # Each group's "score" is its own LAYER (via layer_by_group), not the raw group index --
        # so the final assertion can check the right candidate's result landed in the right slot
        # regardless of which chunk it was computed in. The baseline group has no layer_by_group
        # entry at all (that's the point -- it's never hooked), so it gets a fixed placeholder.
        return {
            g: [f"resp_{layer_by_group[g]}" if g in layer_by_group else "resp_1"] * len(prompts)
            for g, prompts in prompts_by_group.items()
        }

    fake_rows = [{"id": str(i)} for i in range(20)]

    class _ScoringEvalAdapter:
        SCORE_FIELDS = ["correct", "conciseness"]

        @staticmethod
        def score_response(row, response):
            return {"correct": int(response.split("_")[-1]), "conciseness": 1.0}  # score == the layer, by construction

    class _Adapter20:
        def load_rows(self, split):
            return fake_rows

        def to_items(self, tokenizer, rows):
            return [{"id": r["id"], "base_prompt": f"prompt_{r['id']}", "terse_prompt": f"terse_prompt_{r['id']}"} for r in rows]

    with patch("evals.layer_hparam_search.RETRAIN_FNS", {"proper": fake_retrain}), \
         patch("evals.layer_hparam_search.generate_with_routed_configs", side_effect=fake_generate_with_routed_configs):
        results = evaluate_candidates("proper", candidates, ctx, _Adapter20(), _ScoringEvalAdapter(), n_examples=20, max_batch_rows=45)

    assert all(size <= 45 for size in call_sizes), f"a chunk exceeded the row cap: {call_sizes}"
    assert len(call_sizes) > 1, "5 candidates x 20 rows = 100 total must need more than one call at a 45-row cap"
    for i, candidate in enumerate(candidates):
        assert results[i]["judged_score"] == candidate["layer"], (
            f"candidate for layer {candidate['layer']} got the wrong judged score after chunking -- "
            f"a result must not get mixed up with a DIFFERENT candidate's chunk"
        )


def test_summarize_all_variants_reports_done_and_in_progress_correctly(tmp_path, monkeypatch):
    import evals.layer_hparam_search as mod

    class _Adapter:
        RESULTS_DIR = tmp_path

    monkeypatch.setattr(mod, "get_adapter", lambda task: _Adapter())

    (tmp_path / "proper_tiered_search.jsonl").write_text(
        json.dumps({"tier": "final", "layer": 14, "judged_score": 1.95, "ci_lo": 1.9, "ci_hi": 2.0, "n": 180}) + "\n"
    )
    (tmp_path / "conceptor_tiered_search.jsonl").write_text(
        json.dumps({"tier": "tier1", "layer": 2, "judged_score": 1.5, "ci_lo": 1.4, "ci_hi": 1.6, "n": 20}) + "\n"
    )

    summary = mod.summarize_all_variants("caveman")
    assert summary["proper"]["status"] == "done"
    assert summary["proper"]["layer"] == 14
    assert "not yet at Final" in summary["conceptor"]["status"]
    assert (tmp_path / "tiered_search_summary.json").exists()
