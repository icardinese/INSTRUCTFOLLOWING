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
