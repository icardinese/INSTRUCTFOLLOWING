"""Tests for adapters/registry.py and evals/registry.py -- the fail-fast contract checks that let
adding a new task/method never touch orchestration code.
"""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest


def test_adapters_registry_unknown_task_raises_clearly():
    from adapters.registry import get_adapter
    with pytest.raises(ValueError, match="unknown task"):
        get_adapter("nonexistent_task_xyz")


def test_adapters_registry_incomplete_adapter_raises_clearly():
    import adapters.registry as reg
    incomplete = types.ModuleType("fake_incomplete_adapter")
    sys.modules["fake_incomplete_adapter"] = incomplete
    reg._TASKS["_test_broken"] = "fake_incomplete_adapter"
    with pytest.raises(AttributeError, match="missing required attribute"):
        reg.get_adapter("_test_broken")
    del reg._TASKS["_test_broken"]


def test_adapters_registry_real_caveman_adapter_resolves():
    from adapters.registry import get_adapter
    adapter = get_adapter("caveman")
    for attr in ["DATA_DIR", "RESULTS_DIR", "CACHE_DIR", "MODEL_NAME", "load_rows", "to_items"]:
        assert hasattr(adapter, attr), f"real caveman adapter missing {attr}"


def test_adapters_registry_real_ifeval_adapter_resolves():
    from adapters.registry import get_adapter
    adapter = get_adapter("ifeval")
    for attr in ["DATA_DIR", "RESULTS_DIR", "CACHE_DIR", "MODEL_NAME", "load_rows", "to_items"]:
        assert hasattr(adapter, attr), f"real ifeval adapter missing {attr}"


def test_evals_registry_unknown_task_raises_clearly():
    from evals.registry import get_eval_adapter
    with pytest.raises(ValueError, match="unknown task"):
        get_eval_adapter("nonexistent_task_xyz")


def test_evals_registry_incomplete_adapter_raises_clearly():
    import evals.registry as reg
    incomplete = types.ModuleType("fake_incomplete_eval")
    sys.modules["fake_incomplete_eval"] = incomplete
    reg._TASKS["_test_broken"] = "fake_incomplete_eval"
    with pytest.raises(AttributeError, match="missing required attribute"):
        reg.get_eval_adapter("_test_broken")
    del reg._TASKS["_test_broken"]


def test_evals_registry_real_caveman_and_ifeval_judges_resolve():
    from evals.registry import get_eval_adapter
    caveman = get_eval_adapter("caveman")
    ifeval = get_eval_adapter("ifeval")
    assert callable(caveman.score_response)
    assert callable(ifeval.score_response)
    assert caveman.SCORE_FIELDS == ["correct", "coherent", "conciseness"]
    assert ifeval.SCORE_FIELDS == ["follow_all_instructions", "n_followed", "n_total"]
