"""Covers the switchable intervention surface (R vs QR) and the triage adapter's structural
invariants -- particularly the system/user split, which several other things silently depend on.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters.registry import get_adapter, steering_location_for
from evals.registry import get_eval_adapter
from evals.triage.judge import parse_label, score_response
from steering.psr.gate import answer_only_mask
from steering.psr.spans import (
    ANSWER_ONLY,
    QUESTION_AND_ANSWER,
    prefill_tail_length,
    steering_mask,
    validate_location,
)


# --- surface switching -------------------------------------------------------------------------

def test_answer_only_is_response_plus_final_prompt_token():
    mask = steering_mask(10, 4, "cpu", ANSWER_ONLY).squeeze().tolist()
    assert mask == [False] * 5 + [True] * 5, "n_resp + 1 positions"


def test_qr_covers_everything_from_the_system_boundary():
    mask = steering_mask(10, 4, "cpu", QUESTION_AND_ANSWER, last_sys_idx=2).squeeze().tolist()
    assert mask == [False, False] + [True] * 8


def test_qr_strictly_contains_r():
    r = steering_mask(20, 5, "cpu", ANSWER_ONLY)
    qr = steering_mask(20, 5, "cpu", QUESTION_AND_ANSWER, last_sys_idx=3)
    assert torch.all(qr | r == qr), "QR must steer every position R steers, and more"


def test_answer_only_mask_wrapper_agrees_with_the_general_helper():
    """gate.answer_only_mask is a named wrapper; a divergence would split the R path in two."""
    assert torch.equal(answer_only_mask(12, 3, "cpu"), steering_mask(12, 3, "cpu", ANSWER_ONLY))


def test_qr_without_a_boundary_index_is_an_error_not_a_default():
    """Silently falling back to R would make a QR run secretly an R run."""
    with pytest.raises(ValueError, match="needs last_sys_idx"):
        steering_mask(10, 4, "cpu", QUESTION_AND_ANSWER)


def test_unknown_surface_rejected():
    with pytest.raises(ValueError, match="unknown steering_location"):
        validate_location("prompt_only")


def test_prefill_tail_is_one_for_r_and_the_question_span_for_qr():
    assert prefill_tail_length(50, ANSWER_ONLY) == 1
    assert prefill_tail_length(50, QUESTION_AND_ANSWER, last_sys_idx=20) == 30


def test_prefill_tail_never_zero():
    """A zero tail would skip prefill entirely, reintroducing the bug fixed on 2026-09-20."""
    assert prefill_tail_length(10, QUESTION_AND_ANSWER, last_sys_idx=10) >= 1


# --- per-task surface declaration --------------------------------------------------------------

@pytest.mark.parametrize("task,expected", [
    ("caveman", ANSWER_ONLY),
    ("ifeval", ANSWER_ONLY),
    ("triage", QUESTION_AND_ANSWER),
])
def test_each_task_declares_the_surface_its_instruction_requires(task, expected):
    """caveman/ifeval instructions act on generated tokens; triage's slots act on how the input
    is read, which response-only steering cannot reach."""
    assert steering_location_for(task) == expected


def test_adapters_without_an_explicit_surface_default_to_r():
    mod = get_adapter("caveman")
    assert not hasattr(mod, "STEERING_LOCATION")
    assert steering_location_for("caveman") == ANSWER_ONLY


# --- triage adapter structure ------------------------------------------------------------------

def _fake_tokenizer():
    import re

    from jinja2 import Template
    tmpl = Template(
        "{% for m in messages %}{{'<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>\n'}}"
        "{% endfor %}{% if add_generation_prompt %}{{'<|im_start|>assistant\n'}}{% endif %}"
    )

    class T:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            return tmpl.render(messages=messages, add_generation_prompt=add_generation_prompt)

        def __call__(self, text, add_special_tokens=False, return_tensors=None):
            return {"input_ids": re.findall(r"<\|[^|]*\|>|\S+", text)}

    return T()


_ROW = {
    "id": "x_inbox_1",
    "subject": "Limit breach on the West book",
    "email_body": "We went over VaR at close and are still over. Need your sign-off today.",
    "author": "unknown.sender@external.com",
    "to": "x@enron.com",
}


def test_user_span_is_identical_between_base_and_instructed():
    """THE load-bearing invariant. All removed slot content lives in the system turn, so the
    sequences are identical from the email onward and the MSE objective can align from the end
    with no segment-alignment machinery. If this breaks, the loss path needs rewriting."""
    A = get_adapter("triage")
    tok = _fake_tokenizer()
    items = A.to_items(tok, [_ROW])
    base, instr = items[0]["base_prompt"], items[0]["terse_prompt"]
    marker = "Please determine how to handle"
    assert base[base.index(marker):] == instr[instr.index(marker):]


def test_token_count_after_the_system_boundary_matches_across_both_prompts():
    A = get_adapter("triage")
    tok = _fake_tokenizer()
    it = A.to_items(tok, [_ROW])[0]
    n = lambda s: len(tok(s)["input_ids"])  # noqa: E731
    assert (n(it["base_prompt"]) - it["base_last_sys_idx"]) == (
        n(it["terse_prompt"]) - it["terse_last_sys_idx"]
    )


def test_instructed_prompt_is_substantially_longer_than_stripped():
    """The slot content is the instruction being substituted; if these were close, there would be
    nothing for steering to carry."""
    A = get_adapter("triage")
    tok = _fake_tokenizer()
    it = A.to_items(tok, [_ROW])[0]
    assert len(it["terse_prompt"]) > 3 * len(it["base_prompt"])


def test_items_carry_a_boundary_index_for_each_side():
    A = get_adapter("triage")
    it = A.to_items(_fake_tokenizer(), [_ROW])[0]
    assert it["terse_last_sys_idx"] > it["base_last_sys_idx"] > 0


def test_ablation_ladder_is_monotone_decreasing():
    """The sweep is the experiment: accuracy retained as a function of instruction withheld."""
    A = get_adapter("triage")
    cfg = A.load_config()
    order = ["none", "fewshot", "rules", "rules_fewshot", "all"]
    sizes = [len(A.build_system(cfg, A.SLOT_ABLATIONS[k])) for k in order]
    assert sizes == sorted(sizes, reverse=True), dict(zip(order, sizes))


def test_stripped_prompt_still_names_the_three_labels():
    """Removing the label DEFINITIONS is the manipulation; removing the output format would make
    this a test of whether steering can invent a format, which is a different question."""
    A = get_adapter("triage")
    stripped = A.build_system(A.load_config(), A.SLOT_ABLATIONS["all"])
    for label in A.LABELS:
        assert label in stripped


# --- triage judge ------------------------------------------------------------------------------

@pytest.mark.parametrize("response,expected", [
    ("Reasoning here.\nTriage: email", "email"),
    ("Triage: no", "no"),
    ("blah\ntriage:  notify", "notify"),
    ("Triage: `email`", "email"),
    ("This is clearly not a notify.\nTriage: email", "email"),
    ("I think email", "email"),
    ("", None),
    ("no idea what to do here with this one", None),
])
def test_label_parsing(response, expected):
    assert parse_label(response) == expected


def test_last_label_wins_over_labels_mentioned_while_reasoning():
    assert parse_label("Triage: no\nActually reconsidering.\nTriage: notify") == "notify"


def test_unparseable_scores_as_incorrect_rather_than_missing():
    out = score_response({"id": "a", "gold": "email"}, "...")
    assert out == {"correct": 0, "parsed": 0, "predicted": None}


def test_exact_match_scoring():
    assert score_response({"id": "a", "gold": "email"}, "Triage: email")["correct"] == 1
    assert score_response({"id": "a", "gold": "no"}, "Triage: email")["correct"] == 0


def test_missing_gold_fails_loudly():
    with pytest.raises(ValueError, match="expected one of"):
        score_response({"id": "a"}, "Triage: email")


def test_triage_judge_exposes_the_shared_eval_contract():
    mod = get_eval_adapter("triage")
    assert mod.SCORE_FIELDS and callable(mod.score_response)


# --- Final tier must honour max_batch_rows (the 2026-09-25 OOM) ---------------------------------

def _run_evaluate_with_fake_generate(monkeypatch, n_candidates, n_prompts, cap):
    """Drive the REAL evaluate_candidates with generate replaced by a recorder. Each response
    encodes (group, prompt) so ordering and routing can be checked exactly."""
    import evals.layer_hparam_search as mod

    calls = []

    def fake_generate(model, tok, prompts_by_group, hooks_by_group, layer_by_group):
        calls.append({"rows": sum(len(v) for v in prompts_by_group.values()),
                      "hooks": {k: hooks_by_group.get(k) for k in prompts_by_group},
                      "layers": {k: layer_by_group.get(k) for k in prompts_by_group}})
        return {k: [f"{k}|{p}" for p in v] for k, v in prompts_by_group.items()}

    monkeypatch.setattr(mod, "generate_with_routed_configs", fake_generate)
    monkeypatch.setattr(mod, "_score_all_concurrently",
                        lambda ea, rows, responses: [{"correct": 1, "parsed": 1} for _ in responses])
    monkeypatch.setattr(mod, "_avg_tokens", lambda ctx, responses: 1.0)

    class A:
        def load_rows(self, split): return [{"id": str(i), "gold": "no"} for i in range(n_prompts)]
        def to_items(self, tok, rows): return [{"id": r["id"], "base_prompt": f"B{r['id']}",
                                                "terse_prompt": f"T{r['id']}"} for r in rows]

    class E:
        SCORE_FIELDS = ["correct", "parsed"]; PRIMARY_FIELD_MAX = 1; RANK_BY = "primary_desc"

    hooks = {}
    def retrain(cand, ctx):
        hooks[cand["layer"]] = f"hook-L{cand['layer']}"
        return hooks[cand["layer"]]
    monkeypatch.setitem(mod.RETRAIN_FNS, "proper", retrain)
    ctx = type("C", (), {"model": None,
                         "tokenizer": staticmethod(lambda text, **k: {"input_ids": text.split()})})()
    cands = [{"layer": 10 + 2 * i, "mse_weight": 1.0, "nll_weight": 0.0} for i in range(n_candidates)]
    results = mod.evaluate_candidates("proper", cands, ctx, A(), E(), n_prompts, split="test",
                                      max_batch_rows=cap)
    return calls, results


def test_final_tier_never_exceeds_max_batch_rows(monkeypatch):
    """Final is ONE candidate x 180 prompts plus a 180-prompt baseline. The packer never split a
    group, so all 180 went through generate at once and OOM'd on triage's 1.7k-token prompts."""
    calls, _ = _run_evaluate_with_fake_generate(monkeypatch, n_candidates=1, n_prompts=180, cap=11)
    assert max(c["rows"] for c in calls) <= 11
    assert sum(c["rows"] for c in calls) == 360, "every prompt (candidate + baseline) generated exactly once"


def test_sliced_responses_reassemble_in_order_with_correct_routing(monkeypatch):
    calls, results = _run_evaluate_with_fake_generate(monkeypatch, n_candidates=2, n_prompts=25, cap=11)
    # every slice of group g carries group g's hook and layer, never a neighbour's
    for c in calls:
        for g, hook in c["hooks"].items():
            if isinstance(g, int):
                assert hook == f"hook-L{10 + 2 * g}" and c["layers"][g] == 10 + 2 * g
            else:
                assert hook is None and c["layers"][g] is None, "Prompt baseline must stay unsteered"
    assert max(c["rows"] for c in calls) <= 11


def test_tier1_shape_unchanged_when_groups_already_fit(monkeypatch):
    """Tier 1 packs several small groups per call; slicing must not change that behaviour."""
    calls, _ = _run_evaluate_with_fake_generate(monkeypatch, n_candidates=13, n_prompts=5, cap=11)
    assert max(c["rows"] for c in calls) <= 11
    assert sum(c["rows"] for c in calls) == 13 * 5 + 5


def test_triage_score_fields_are_numeric():
    """summarize / bootstrap_analysis / plotting sum every SCORE_FIELDS entry; a string field
    ('predicted') crashed all three with int + str."""
    import importlib
    j = importlib.import_module("evals.triage.judge")
    out = j.score_response({"id": "a", "gold": "email"}, "Triage: email")
    for f in j.SCORE_FIELDS:
        assert isinstance(out[f], (int, float)), f"SCORE_FIELDS entry {f!r} is not numeric"
    assert out["predicted"] == "email", "the label is still returned for inspection"


# --- every-point Tier 1: every sweep setting gets judged on generations -------------------------

def _run_search_capturing_tier1(tmp_path, monkeypatch, sweep_rows, variant, **kw):
    import json as _json
    import evals.layer_hparam_search as mod
    (tmp_path / f"psr_{variant}_sweep.jsonl").write_text("".join(_json.dumps(r) + "\n" for r in sweep_rows))

    class A:
        RESULTS_DIR = tmp_path; CACHE_DIR = tmp_path / "cache"; MODEL_NAME = "fake"
        def load_rows(self, split): return [{"id": "0"}]
        def to_items(self, tok, rows): return [{"id": "0", "base_prompt": "p", "terse_prompt": "tp"}]

    class E:
        SCORE_FIELDS = ["correct", "parsed"]; PRIMARY_FIELD_MAX = 1; RANK_BY = "primary_desc"

    monkeypatch.setattr(mod, "get_adapter", lambda t: A())
    monkeypatch.setattr(mod, "get_eval_adapter", lambda t: E())
    class _M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("C", (), {"hidden_size": 16})()
    monkeypatch.setattr(mod, "load_model", lambda device, model_name=None: (_M(), object()))
    monkeypatch.setattr(mod, "num_layers", lambda m: 28)
    monkeypatch.setattr(mod, "load_or_compute_responses", lambda *a, **k: {})
    seen = {}

    def fake_eval(variant_, cands, ctx, adapter, ea, n, split="dev", max_batch_rows=None):
        tier = {"dev": "tier1" if "tier1" not in seen else "tier2", "test": "final"}[split]
        seen[tier] = [dict(c) for c in cands]
        return [{**c, "skipped": False, "judged_score": 0.9, "ci_lo": 0.8, "ci_hi": 1.0,
                 "correctness_scores": [1.0] * n, "optimize_score": 1.0,
                 "optimize_ci_lo": 1.0, "optimize_ci_hi": 1.0, "n": n,
                 "fully_correct_rate": 0.9, "avg_tokens": 10.0,
                 "prompt_fully_correct_rate": 0.8, "prompt_avg_tokens": 12.0} for c in cands]

    monkeypatch.setattr(mod, "evaluate_candidates", fake_eval)
    mod.run_tiered_search("triage", variant, tier1_n=2, tier2_n=2, final_n=2, **kw)
    return seen


def _sweep(layers=(2, 4, 6), alphas=None):
    rows = []
    for L in layers:
        for a in (alphas or [None]):
            for m, n, mse in ((1.0, 0.0, 1.0), (0.0, 1.0, 9.0)):   # MSE config always has lower final_mse
                r = {"layer": L, "mse_weight": m, "nll_weight": n, "final_mse": mse + L / 100}
                if a is not None:
                    r["alpha"] = a
                rows.append(r)
    return rows


def test_default_prefilter_never_judges_an_nll_gate(tmp_path, monkeypatch):
    """Documents the problem being fixed: best_row_per_layer lets final_mse choose within a layer,
    and the MSE-trained config always wins that, so no NLL gate is ever generated from."""
    seen = _run_search_capturing_tier1(tmp_path, monkeypatch, _sweep(), "proper")
    assert all(c["nll_weight"] == 0 for c in seen["tier1"])


def test_every_point_nll_search_judges_every_nll_gate(tmp_path, monkeypatch):
    seen = _run_search_capturing_tier1(tmp_path, monkeypatch, _sweep(), "proper",
                                       tier1_all_points=True, objective="nll")
    got = sorted((c["layer"], c["mse_weight"], c["nll_weight"]) for c in seen["tier1"])
    assert got == [(2, 0.0, 1.0), (4, 0.0, 1.0), (6, 0.0, 1.0)]


def test_every_point_judges_every_alpha_not_just_the_mse_best(tmp_path, monkeypatch):
    seen = _run_search_capturing_tier1(tmp_path, monkeypatch, _sweep(alphas=[1.0, 2.0, 4.0]),
                                       "conceptor", tier1_all_points=True, objective="mse")
    assert len(seen["tier1"]) == 3 * 3, "every layer x every alpha for the objective"
    assert {c["alpha"] for c in seen["tier1"]} == {1.0, 2.0, 4.0}


def test_each_objective_writes_its_own_results_and_default_file_is_untouched(tmp_path, monkeypatch):
    _run_search_capturing_tier1(tmp_path, monkeypatch, _sweep(), "sg", tier1_all_points=True, objective="mse")
    _run_search_capturing_tier1(tmp_path, monkeypatch, _sweep(), "sg", tier1_all_points=True, objective="nll")
    assert (tmp_path / "sg_allpoints_mse_tiered_search.jsonl").exists()
    assert (tmp_path / "sg_allpoints_nll_tiered_search.jsonl").exists()
    assert not (tmp_path / "sg_tiered_search.jsonl").exists(), "every-point runs must not clobber the default file"


def test_regen_reports_each_objective_as_its_own_row():
    import importlib
    regen = importlib.import_module("scripts.regen_token_cis")
    for v in ("proper", "sg", "sg_clamp", "conceptor"):
        for obj in ("mse", "nll"):
            key = f"{v}_{obj}"
            assert regen.TIERED_ALLPOINTS[key] == (v, [f"{v}_allpoints_{obj}"])
    for v in ("sg_clamp", "const_resp", "stolfo_resp"):
        assert v in regen.TIERED, f"{v} winners would never reach token_distributions.json"
