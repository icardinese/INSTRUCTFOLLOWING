"""core/gate_store + evals/layer_hparam_search._stored: a gate is trained at most once, a stored gate
steers bit-identically to a freshly trained one, and a stale gate is never reused."""
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import gate_store  # noqa: E402

D = 16


@pytest.fixture(autouse=True)
def cuda_is_cpu(monkeypatch):
    """The real hook builders call .to("cuda"). Map that to CPU so the ACTUAL builders are tested."""
    real_to = torch.Tensor.to

    def to(self, *args, **kwargs):
        args = tuple("cpu" if a == "cuda" else a for a in args)
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return real_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", to)


def _gate_result(seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"weight": torch.randn(D, 1, generator=g), "bias": torch.randn(1, generator=g),
            "coeff_bias": torch.zeros(1), "direction": torch.randn(D, generator=g),
            "final_mse": 1.23, "final_nll": 0.45, "completed_epochs": 15}


def _clamp_result(layer, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"gates": {layer: {"weight": torch.randn(D, 1, generator=g), "bias": torch.randn(1, generator=g),
                              "coeff_bias": torch.zeros(1)}},
            "directions": {layer: torch.nn.functional.normalize(torch.randn(D, generator=g), dim=0)},
            "targets": {layer: 0.7}, "final_mse": 2.0, "completed_epochs": 15}


class _Ctx:
    def __init__(self, results_dir, seed=42, responses=None):
        self.results_dir = results_dir
        self.seed = seed
        self.train_items = [{"id": "a"}, {"id": "b"}]
        self.dev_items = [{"id": "d"}]
        self.train_responses = responses or {"a": {"text": "x", "finished": True},
                                             "b": {"text": "y", "finished": True}}
        self._fp = None

    def fingerprint(self):
        if self._fp is None:
            self._fp = gate_store.fingerprint(self.seed, self.train_items, self.train_responses, self.dev_items)
        return self._fp


def _counting(result_factory):
    calls = []

    def train(row, ctx):
        calls.append(dict(row))
        return result_factory()
    return train, calls


ROW = {"layer": 14, "mse_weight": 1.0, "nll_weight": 0.0}


# ---------------------------------------------------------------- trained at most once
def test_second_request_loads_instead_of_retraining(tmp_path):
    from evals.layer_hparam_search import _hook_gate, _stored
    train, calls = _counting(_gate_result)
    fn = _stored("proper", train, _hook_gate)
    ctx = _Ctx(tmp_path)
    fn(ROW, ctx)
    fn(ROW, ctx)   # Tier 2
    fn(ROW, ctx)   # Final
    assert len(calls) == 1, "Tier 2 and Final must load, not retrain"


def test_stored_gate_steers_bit_identically_to_fresh(tmp_path):
    """The whole point: loading is not an approximation of retraining."""
    from evals.layer_hparam_search import _hook_gate, _stored
    fresh = _gate_result(seed=3)
    train, _ = _counting(lambda: fresh)
    fn = _stored("proper", train, _hook_gate)
    ctx = _Ctx(tmp_path)
    hook_first = fn(ROW, ctx)           # trained + saved
    hook_loaded = fn(ROW, ctx)          # loaded from disk
    hook_direct = _hook_gate(fresh, ROW)
    torch.manual_seed(0)
    for shape in [(3, 1, D), (2, 7, D)]:  # decode step, and prefill
        h = torch.randn(*shape)
        a, b, c = hook_direct(h.clone()), hook_first(h.clone()), hook_loaded(h.clone())
        assert torch.equal(a, b) and torch.equal(a, c), f"hook output differs for shape {shape}"


def test_sg_clamp_stored_gate_steers_bit_identically(tmp_path):
    from evals.layer_hparam_search import _hook_sg_clamp, _stored
    L = 6
    fresh = _clamp_result(L, seed=5)
    train, calls = _counting(lambda: fresh)
    fn = _stored("sg_clamp", train, _hook_sg_clamp)
    row = {"layer": L, "mse_weight": 1.0, "nll_weight": 0.0}
    ctx = _Ctx(tmp_path)
    fn(row, ctx)
    loaded = fn(row, ctx)
    assert len(calls) == 1
    h = torch.randn(2, 1, D)
    assert torch.equal(_hook_sg_clamp(fresh, row)(h.clone()), loaded(h.clone()))


# ---------------------------------------------------------------- sweep save == search lookup
def test_point_saved_by_sweep_is_found_by_search_row(tmp_path):
    """A sweep saves under its in-memory point dict; the search looks up with a JSON-loaded sweep
    row that carries extra metric fields and possibly int-vs-float differences. Same file."""
    ctx = _Ctx(tmp_path)
    sweep_point = {"layer": 14, "alpha": 2, "mse_weight": 1, "nll_weight": 0}
    gate_store.save(gate_store.point_path(tmp_path, "conceptor", sweep_point, ctx.fingerprint()),
                    "conceptor", _gate_result())
    search_row = json.loads(json.dumps({"layer": 14, "alpha": 2.0, "mse_weight": 1.0, "nll_weight": 0.0,
                                        "final_mse": 9.9, "completed_epochs": 15}))
    assert gate_store.point_path(tmp_path, "conceptor", search_row, ctx.fingerprint()).exists()


# ---------------------------------------------------------------- staleness guard
@pytest.mark.parametrize("change", ["seed", "response_text", "finished_flag", "env"])
def test_changed_training_setup_never_reuses_old_gate(tmp_path, monkeypatch, change):
    from evals.layer_hparam_search import _hook_gate, _stored
    train, calls = _counting(_gate_result)
    fn = _stored("proper", train, _hook_gate)
    fn(ROW, _Ctx(tmp_path))
    if change == "seed":
        ctx = _Ctx(tmp_path, seed=7)
    elif change == "response_text":
        ctx = _Ctx(tmp_path, responses={"a": {"text": "CHANGED", "finished": True}, "b": {"text": "y", "finished": True}})
    elif change == "finished_flag":
        ctx = _Ctx(tmp_path, responses={"a": {"text": "x", "finished": False}, "b": {"text": "y", "finished": True}})
    else:
        monkeypatch.setenv("PSR_PROPER_LOSS_BALANCE", "persona")
        ctx = _Ctx(tmp_path)
    fn(ROW, ctx)
    assert len(calls) == 2, f"a {change} change must force a retrain, not reuse the old gate"


def test_different_points_never_collide(tmp_path):
    ctx = _Ctx(tmp_path)
    fp = ctx.fingerprint()
    paths = {gate_store.point_path(tmp_path, v, r, fp) for v, r in [
        ("proper", {"layer": 14, "mse_weight": 1.0, "nll_weight": 0.0}),
        ("proper", {"layer": 14, "mse_weight": 0.0, "nll_weight": 1.0}),
        ("proper", {"layer": 16, "mse_weight": 1.0, "nll_weight": 0.0}),
        ("sg", {"layer": 14, "mse_weight": 1.0, "nll_weight": 0.0}),
        ("conceptor", {"layer": 14, "alpha": 1.0, "mse_weight": 1.0, "nll_weight": 0.0}),
        ("conceptor", {"layer": 14, "alpha": 2.0, "mse_weight": 1.0, "nll_weight": 0.0}),
    ]}
    assert len(paths) == 6


# ---------------------------------------------------------------- robustness
def test_corrupt_store_file_is_retrained_not_trusted(tmp_path):
    from evals.layer_hparam_search import _hook_gate, _stored
    ctx = _Ctx(tmp_path)
    path = gate_store.point_path(tmp_path, "proper", ROW, ctx.fingerprint())
    path.parent.mkdir(parents=True)
    path.write_bytes(b"truncated garbage")
    train, calls = _counting(_gate_result)
    _stored("proper", train, _hook_gate)(ROW, ctx)
    assert len(calls) == 1
    assert gate_store.load(path) is not None, "the retrained gate must overwrite the corrupt file"


def test_no_results_dir_keeps_old_behaviour(tmp_path):
    from evals.layer_hparam_search import _hook_gate, _stored
    train, calls = _counting(_gate_result)
    fn = _stored("proper", train, _hook_gate)
    ctx = _Ctx(None)
    fn(ROW, ctx)
    fn(ROW, ctx)
    assert len(calls) == 2


def test_store_keeps_only_hook_fields_not_big_matrices(tmp_path):
    res = _gate_result()
    res["conceptor"] = torch.randn(512, 512)   # the kind of thing that must NOT be dragged along
    path = tmp_path / "x.pt"
    gate_store.save(path, "conceptor", res)
    assert "conceptor" not in gate_store.load(path)


# ---------------------------------------------------------------- legacy SG+Clamp probes
def _write_legacy(tmp_path, layer, epochs):
    from src.psr.clamp_gate.train import _save
    r = _clamp_result(layer, seed=9)
    r["completed_epochs"] = epochs
    _save(tmp_path / f"sg_clamp_L{layer}_probe_mse.pt", r, [layer], "triage", 1.0, 0.0)
    return r


def test_legacy_sg_clamp_probe_is_imported_without_retraining(tmp_path):
    from evals.layer_hparam_search import _hook_sg_clamp, _legacy_sg_clamp, _stored
    L = 8
    legacy = _write_legacy(tmp_path, L, epochs=15)
    train, calls = _counting(lambda: _clamp_result(L))
    fn = _stored("sg_clamp", train, _hook_sg_clamp, legacy_fn=_legacy_sg_clamp)
    row = {"layer": L, "mse_weight": 1.0, "nll_weight": 0.0}
    hook = fn(row, _Ctx(tmp_path))
    assert calls == [], "a valid legacy probe must be reused, not retrained"
    h = torch.randn(2, 1, D)
    assert torch.equal(hook(h.clone()), _hook_sg_clamp(legacy, row)(h.clone()))
    fn(row, _Ctx(tmp_path))
    assert calls == [], "after import it must come from the store"


def test_legacy_probe_from_old_epoch_budget_is_rejected(tmp_path):
    from evals.layer_hparam_search import _hook_sg_clamp, _legacy_sg_clamp, _stored
    L = 8
    _write_legacy(tmp_path, L, epochs=3)   # pre-fidelity budget
    train, calls = _counting(lambda: _clamp_result(L))
    _stored("sg_clamp", train, _hook_sg_clamp, legacy_fn=_legacy_sg_clamp)(
        {"layer": L, "mse_weight": 1.0, "nll_weight": 0.0}, _Ctx(tmp_path))
    assert len(calls) == 1, "a probe trained on a different epoch budget must not be passed off as current"


# ---------------------------------------------------------------- registry
def test_every_trained_variant_goes_through_the_store():
    from evals.layer_hparam_search import RETRAIN_FNS
    for v in ("proper", "sg", "sg_clamp", "conceptor", "conceptor_matrix", "conceptor_selfproj"):
        assert RETRAIN_FNS[v].__name__ == "wrapped", f"{v} bypasses the gate store"
    for v in ("const", "stolfo", "const_resp", "stolfo_resp"):
        assert RETRAIN_FNS[v].__name__ != "wrapped", f"{v} is training-free and needs no store"
