"""Regression test for a real bug: resuming a sweep (core.sweep.run_grid_sweep skips
already-completed grid points) where the OVERALL WINNER happens to be one of the already-completed
points, not one retrained this session -- the old code bailed with a "NOTE, delete this row and
rerun" message and wrote NO checkpoint at all, even though the sweep was functionally done. This
is exactly what happened on a real overnight run: 324/325 points done, only the LAST point missing,
and the actual winner was among the 324 -- resuming would have finished the sweep and then thrown
the result away. Fixed by retraining the winner once more if it's not already in memory.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

import core.model_common as model_common
import src.psr.proper.train as proper_train
from tests.fakes import make_fake_model_and_tokenizer

ITEMS = [
    {"id": "0", "base_prompt": "explain this", "terse_prompt": "explain this tersely"},
    {"id": "1", "base_prompt": "what does it do", "terse_prompt": "what does it do, briefly"},
]
RESPONSES = {"0": "it adds two numbers", "1": "it returns a value"}


class _FakeAdapter:
    MODEL_NAME = "fake"

    def __init__(self, results_dir):
        self.RESULTS_DIR = results_dir
        self.CACHE_DIR = results_dir / "cache"

    def load_rows(self, split):
        return [{"id": "0"}, {"id": "1"}]

    def to_items(self, tokenizer, rows):
        return ITEMS


def test_resumed_sweep_still_checkpoints_the_true_winner_even_if_not_retrained_this_session(tmp_path, monkeypatch):
    (tmp_path / "cache").mkdir()

    fake_model, fake_tokenizer = make_fake_model_and_tokenizer(d=16, n_layers=4)
    for p in fake_model.parameters():
        p.requires_grad_(False)

    monkeypatch.setattr(model_common, "load_model", lambda device, model_name=None, **kw: (fake_model, fake_tokenizer))
    monkeypatch.setattr(model_common, "generate_response", lambda model, tokenizer, prompt, **kw: "a fake response")
    monkeypatch.setattr(proper_train, "load_model", lambda device, model_name=None, **kw: (fake_model, fake_tokenizer))
    monkeypatch.setattr(proper_train, "generate_response", lambda model, tokenizer, prompt, **kw: "a fake response")
    monkeypatch.setattr(proper_train, "num_layers", lambda model: 4)
    monkeypatch.setattr(proper_train, "get_adapter", lambda task: _FakeAdapter(tmp_path))

    # Pre-populate the sweep file as if layers 1 and 2 were ALREADY done in an earlier session --
    # layer 1's final_mse is set essentially unbeatably low (not just "lower than layer 2") so it's
    # guaranteed to still be the true winner regardless of what the freshly-trained layer 3 achieves.
    sweep_path = tmp_path / "psr_proper_sweep.jsonl"
    with sweep_path.open("w") as f:
        f.write(json.dumps({"layer": 1, "mse_weight": 1.0, "nll_weight": 0.0, "baseline_mse": 99.0, "baseline_nll": 5.0, "final_mse": 1e-9, "final_nll": 4.0}) + "\n")
        f.write(json.dumps({"layer": 2, "mse_weight": 1.0, "nll_weight": 0.0, "baseline_mse": 99.0, "baseline_nll": 5.0, "final_mse": 5.0, "final_nll": 4.0}) + "\n")
    # Only layer 3 is "missing" -- resuming should retrain JUST layer 3, then realize layer 1 (the
    # real winner, final_mse=1.0) needs its own tensors regenerated since it wasn't retrained now.
    proper_train.sweep("caveman", layers=[1, 2, 3], loss_config_grid=[{"mse_weight": 1.0, "nll_weight": 0.0}], seed=0, device="cpu")

    probe_path = tmp_path / "psr_proper_probe.pt"
    assert probe_path.exists(), "a checkpoint must be written even though the true winner wasn't retrained this session"
    ckpt = torch.load(probe_path)
    assert ckpt["layer"] == 1, "the checkpoint must reflect the TRUE winner (layer 1, lowest final_mse), not layer 3 just because it was the only one retrained this session"
