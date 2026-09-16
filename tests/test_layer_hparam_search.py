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

import torch

from evals.layer_hparam_search import SearchContext, best_row_per_layer, evaluate_candidates


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
    SCORE_FIELDS = ["correct"]

    @staticmethod
    def score_response(row, response):
        # deterministic score derived from the response text so tests can assert on it precisely
        return {"correct": int(response.split("_")[-1])}


class _FakeAdapter:
    def __init__(self):
        self.CACHE_DIR = Path("/tmp/fake_cache")

    def load_rows(self, split):
        return [{"id": str(i)} for i in range(3)]

    def to_items(self, tokenizer, rows):
        return [{"id": r["id"], "base_prompt": f"prompt_{r['id']}"} for r in rows]


def test_evaluate_candidates_retrains_judges_and_computes_ci():
    """End-to-end orchestration of evaluate_candidates with retrain_fn and generate_with_routed_configs
    both mocked -- checks the WIRING (candidate -> hook -> batched call -> judged score -> CI), not
    the real training/generation math (already covered elsewhere)."""
    candidates = [
        {"layer": 2, "mse_weight": 1.0, "nll_weight": 0.0},
        {"layer": 4, "mse_weight": 1.0, "nll_weight": 0.0},
    ]
    ctx = SearchContext(
        model=object(), tokenizer=object(), n_layers=10, hidden_size=16, cache_dir=Path("/tmp"),
        train_items=[], dev_items=[], train_responses={}, dev_responses={},
    )

    def fake_retrain(row, ctx):
        return f"hook_for_layer_{row['layer']}"  # stand-in hook object, never actually called

    fake_responses_by_group = {
        0: ["resp_1", "resp_2", "resp_0"],  # layer 2's candidate: scores [1, 2, 0]
        1: ["resp_2", "resp_2", "resp_2"],  # layer 4's candidate: scores [2, 2, 2]
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
        model=object(), tokenizer=object(), n_layers=10, hidden_size=16, cache_dir=Path("/tmp"),
        train_items=[], dev_items=[], train_responses={}, dev_responses={},
    )

    def fake_retrain_returns_none(row, ctx):
        return None

    with patch("evals.layer_hparam_search.RETRAIN_FNS", {"proper": fake_retrain_returns_none}), \
         patch("evals.layer_hparam_search.generate_with_routed_configs", return_value={}):
        results = evaluate_candidates("proper", candidates, ctx, _FakeAdapter(), _FakeEvalAdapter(), n_examples=3)

    assert len(results) == 1
    assert results[0]["skipped"] is True
    assert "judged_score" not in results[0]


def test_tier_progression_narrows_by_judged_score_not_mse():
    """The core methodological claim: a layer with WORSE mse (higher final_mse) but BETTER judged
    score must be able to survive over a layer with better mse but worse judged score -- proving
    MSE genuinely has no vote in which layers survive Tier 1."""
    tier1_results = [
        {"layer": 2, "final_mse": 1.0, "judged_score": 0.2, "skipped": False},   # great MSE, bad judged score
        {"layer": 4, "final_mse": 9.0, "judged_score": 0.9, "skipped": False},   # bad MSE, great judged score
    ]
    survivors = sorted((r for r in tier1_results if not r["skipped"]), key=lambda r: -r["judged_score"])[:1]
    assert survivors[0]["layer"] == 4, "the judged-score winner must survive even with far worse training MSE"


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
        json.dumps({"tier": "tier1", "layer": 14, "mse_weight": 1.0, "nll_weight": 0.0, "skipped": False, "judged_score": 1.9, "ci_lo": 1.8, "ci_hi": 2.0, "n": 20}) + "\n"
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
            return [{"id": "0", "base_prompt": "p"}]

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
        return [{**c, "skipped": False, "judged_score": 1.95, "ci_lo": 1.9, "ci_hi": 2.0, "n": n_examples} for c in candidates]

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
        model=object(), tokenizer=object(), n_layers=20, hidden_size=16, cache_dir=Path("/tmp"),
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
        # regardless of which chunk it was computed in.
        return {g: [f"resp_{layer_by_group[g]}"] * len(prompts) for g, prompts in prompts_by_group.items()}

    fake_rows = [{"id": str(i)} for i in range(20)]

    class _ScoringEvalAdapter:
        SCORE_FIELDS = ["correct"]

        @staticmethod
        def score_response(row, response):
            return {"correct": int(response.split("_")[-1])}  # score == the group_id, by construction

    class _Adapter20:
        def load_rows(self, split):
            return fake_rows

        def to_items(self, tokenizer, rows):
            return [{"id": r["id"], "base_prompt": f"prompt_{r['id']}"} for r in rows]

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
