"""Real IFEval harness integration test, and seed reproducibility -- both proven earlier this
session via one-off scripts, now permanent.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch


def test_ifeval_judge_correctly_distinguishes_compliant_response():
    """Uses the REAL vendored Google IFEval harness (third_party/microsoft_llm_steer_instruct/
    ifeval_scripts/), not a fake -- a comma-containing response must fail CommaChecker, a
    comma-free one must pass."""
    from evals.registry import get_eval_adapter
    eval_adapter = get_eval_adapter("ifeval")

    row = {
        "key": 1,
        "prompt": "Write a short bio. In your entire response, refrain from the use of any commas.",
        "instruction_id_list": ["punctuation:no_comma"],
        "kwargs": [{}],
    }
    compliant = "She trained for years and flew missions to Mars and beyond."
    noncompliant = "She trained for years, and flew missions to Mars, and beyond."

    assert eval_adapter.score_response(row, compliant)["follow_all_instructions"] is True
    assert eval_adapter.score_response(row, noncompliant)["follow_all_instructions"] is False


def test_set_seed_gives_bit_identical_results_same_seed():
    from core.reproducibility import set_seed
    set_seed(42)
    a = torch.randn(10)
    set_seed(42)
    b = torch.randn(10)
    assert torch.equal(a, b)


def test_set_seed_gives_different_results_different_seed():
    from core.reproducibility import set_seed
    set_seed(42)
    a = torch.randn(10)
    set_seed(123)
    b = torch.randn(10)
    assert not torch.equal(a, b)
