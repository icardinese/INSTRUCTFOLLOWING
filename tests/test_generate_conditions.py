"""Regression test for a real bug caught this session: load_const_condition checked only
const_steer_config.json's existence, then unconditionally loaded const_steer_directions.pt right
after -- crashing instead of gracefully skipping if the second file was missing while the first
existed. Every other condition loader in generate.py checks its one required file; const uniquely
needs two, which is exactly why it was the one that slipped through.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch


def test_const_condition_returns_none_when_directions_file_missing(tmp_path, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location("generate_under_test", Path(__file__).resolve().parent.parent / "src" / "generate.py")
    generate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generate)

    class FakeAdapter:
        RESULTS_DIR = tmp_path

    (tmp_path / "const_steer_config.json").write_text(json.dumps({"layer": 1, "coeff": 4}))
    # deliberately do NOT create const_steer_directions.pt

    result = generate.load_const_condition(FakeAdapter(), "cpu")
    assert result is None, "should gracefully return None, not crash, when directions.pt is missing"


def test_const_condition_works_when_both_files_present(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("generate_under_test2", Path(__file__).resolve().parent.parent / "src" / "generate.py")
    generate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generate)

    class FakeAdapter:
        RESULTS_DIR = tmp_path

    (tmp_path / "const_steer_config.json").write_text(json.dumps({"layer": 1, "coeff": 4}))
    torch.save({1: torch.randn(8)}, tmp_path / "const_steer_directions.pt")

    result = generate.load_const_condition(FakeAdapter(), "cpu")
    assert result is not None, "should succeed when both required files are present"


def _load_generate_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("generate_under_test3", Path(__file__).resolve().parent.parent / "src" / "generate.py")
    generate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generate)
    return generate


def test_matrix_condition_context_fn_carries_participation_ratio_from_checkpoint(tmp_path):
    """load_matrix_condition should prefer the value already logged at train time
    (ckpt["participation_ratio"]) over recomputing it -- this test uses a garbage conceptor
    matrix whose RECOMPUTED PR would be wrong, to prove the checkpoint's own value wins."""
    generate = _load_generate_module()

    class FakeAdapter:
        RESULTS_DIR = tmp_path

    hidden_size = 8
    torch.save({
        "weight": torch.zeros(hidden_size, 1), "bias": torch.zeros(1), "coeff_bias": torch.zeros(1),
        "conceptor": torch.eye(hidden_size),  # PR of an identity matrix would recompute to 8.0
        "mu_instr": torch.zeros(hidden_size), "delta_scale": torch.tensor(1.0),
        "layer": 1, "participation_ratio": 2.5,  # deliberately different from the recomputed value
    }, tmp_path / "psr_conceptor_matrix_probe.pt")

    context_fn = generate.load_matrix_condition(FakeAdapter(), "cpu")
    assert context_fn is not None
    assert context_fn.participation_ratio == 2.5


def test_matrix_condition_falls_back_to_recomputing_participation_ratio_when_missing(tmp_path):
    """Older checkpoints trained before participation_ratio existed shouldn't crash -- fall back
    to computing it fresh from the stored conceptor matrix."""
    generate = _load_generate_module()

    class FakeAdapter:
        RESULTS_DIR = tmp_path

    hidden_size = 8
    torch.save({
        "weight": torch.zeros(hidden_size, 1), "bias": torch.zeros(1), "coeff_bias": torch.zeros(1),
        "conceptor": torch.eye(hidden_size), "mu_instr": torch.zeros(hidden_size),
        "delta_scale": torch.tensor(1.0), "layer": 1,
        # no "participation_ratio" key
    }, tmp_path / "psr_conceptor_matrix_probe.pt")

    context_fn = generate.load_matrix_condition(FakeAdapter(), "cpu")
    assert context_fn is not None
    assert abs(context_fn.participation_ratio - hidden_size) < 1e-3, "PR of an identity matrix must equal its dimensionality"


def test_gate_condition_never_gets_a_participation_ratio_attribute(tmp_path):
    """load_gate_condition covers proper, fixed-vector conceptor, and the old S-PSR baseline --
    none of these are matrix-based at inference (see rank_diagnostic.py's module docstring), so
    their context_fn must NOT carry participation_ratio even if the checkpoint happens to include
    a conceptor matrix (the fixed-vector conceptor's checkpoint does, for provenance only)."""
    generate = _load_generate_module()

    class FakeAdapter:
        RESULTS_DIR = tmp_path

    hidden_size = 8
    torch.save({
        "weight": torch.zeros(hidden_size, 1), "bias": torch.zeros(1), "coeff_bias": torch.zeros(1),
        "direction": torch.randn(hidden_size), "conceptor": torch.eye(hidden_size),  # present but unused at inference
        "layer": 1,
    }, tmp_path / "psr_conceptor_probe.pt")

    context_fn = generate.load_gate_condition(FakeAdapter(), "psr_conceptor_probe.pt", "cpu")
    assert context_fn is not None
    assert not hasattr(context_fn, "participation_ratio")
