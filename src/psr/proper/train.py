"""Proper PSR training. Task-agnostic by construction: --task selects an adapter at runtime, so this
is ONE script that runs against caveman OR ifeval, not two near-identical copies.

train_one_config() holds the actual "build a gate+direction for one (layer, loss-config) point and
train it" logic; main() (single run) and sweep() (grid over layer x loss-config) are both thin
wrappers around it, so the two paths can never silently drift apart -- a bug fix to how a gate gets
built only has to happen once.
"""
import argparse
import json
import os

import torch

from adapters.registry import get_adapter
from core.model_common import generate_response, load_model, num_layers
from core.generation_cache import load_or_compute_responses
from core.reproducibility import set_seed
from core.sweep import run_grid_sweep
from steering.psr.gate import forward_with_gate_hook, init_gate_state
from steering.psr.proper.direction import init_direction
from steering.psr.training_loop import train_gate

N_EPOCHS = 3
LR = float(os.environ.get("PSR_PROPER_LR", 1e-3))
WEIGHT_DECAY = 1e-4
REG_COEFF = float(os.environ.get("PSR_PROPER_REG_COEFF", 0.1))
# mse_weight/nll_weight are independent (see steering/psr/training_loop.py's module docstring for
# why): mse_weight=1.0, nll_weight=0.0 is pure MSE (the default -- every existing result
# reproduces unchanged unless these are explicitly set).
MSE_WEIGHT = float(os.environ.get("PSR_PROPER_MSE_WEIGHT", 1.0))
NLL_WEIGHT = float(os.environ.get("PSR_PROPER_NLL_WEIGHT", 0.0))
# Layer grid for sweep() -- a dense range, not just DEFAULT_LAYER_FRACTIONS's 4 points, matching
# the project's own earlier one-off "layers 2-26" sweep (see project handoff finding #4) now made
# a permanent, reusable, resumable script instead of a one-off notebook cell.
DEFAULT_SWEEP_LAYERS = list(range(2, 27, 2))
# Loss configurations to sweep, as (mse_weight, nll_weight) PAIRS rather than a full cartesian
# product of two independent grids -- most (mse_weight, nll_weight) combinations don't correspond
# to a meaningful experiment (e.g. mse_weight=0, nll_weight=0.01 has almost no training signal at
# all). These five points cover: pure MSE (Heyman & Vandeputte's "_MSE" variant, and this
# project's original design), three additive MSE+NLL blends at increasing strength (this
# project's own proposed extension -- see training_loop.py's docstring), and pure NLL (H&V's
# "_LL" variant -- the one that was missing until this point).
# 2026-09-17: cut back to just the two real endpoints per architect's directive -- the three
# intermediate MSE+NLL blends were shown (real sweep data, psr_proper_sweep.jsonl) to change
# final_mse by under 1% relative to pure MSE at every layer, and best_row_per_layer (Tier 1's
# candidate builder) always selects pure MSE anyway since it deterministically has the lowest
# final_mse -- the blends never influenced a single layer decision, only added dead sweep points.
# H&V's own paper trains MSE and LL as two mutually-exclusive alternatives, never blended; this
# matches that directly instead of also chasing this project's own since-abandoned additive-blend
# extension (see steering/psr/training_loop.py's module docstring for that extension's history).
DEFAULT_LOSS_CONFIG_GRID = [
    {"mse_weight": 1.0, "nll_weight": 0.0},   # pure MSE (H&V's "_MSE" variant)
    {"mse_weight": 0.0, "nll_weight": 1.0},   # pure NLL (H&V's "_LL" variant)
]


def train_one_config(
    model, tokenizer, layer_idx: int, seed: int, n_layers: int, hidden_size: int, device: str,
    train_items: list[dict], dev_items: list[dict], train_responses: dict, dev_responses: dict,
    lr: float = LR, weight_decay: float = WEIGHT_DECAY, reg_coeff: float = REG_COEFF,
    n_epochs: int = N_EPOCHS, mse_weight: float = MSE_WEIGHT, nll_weight: float = NLL_WEIGHT,
    on_epoch_end=None,
) -> dict:
    """Builds and trains one gate+direction at one layer. Returns everything a caller might need:
    scalar dev metrics (JSON-safe, for logs/sweep rows) AND the trained tensors (NOT JSON-safe --
    callers that only want the former, like sweep()'s per-point log row, must pick those keys out
    explicitly rather than dumping this whole dict).

    on_epoch_end(epoch, gate, direction, dev_metrics), if given, gets the LIVE gate/direction
    objects (not just metrics, unlike steering.psr.training_loop.train_gate's own on_epoch_end) --
    that's what lets a caller checkpoint mid-training without this function needing to know
    anything about checkpoint file format itself."""
    set_seed(seed)
    gate = init_gate_state(hidden_size, device)
    direction = init_direction(hidden_size, device)
    optimizer = torch.optim.Adam(gate.parameters() + [direction], lr=lr, weight_decay=weight_decay)
    forward_fn = lambda pair: forward_with_gate_hook(model, gate, direction, layer_idx, pair["full_base"], pair["n_resp"])

    wrapped_on_epoch_end = (lambda epoch, metrics: on_epoch_end(epoch, gate, direction, metrics)) if on_epoch_end else None
    baseline_metrics, final_metrics = train_gate(
        model, tokenizer, forward_fn, optimizer, layer_idx, n_layers,
        train_items, dev_items, train_responses, dev_responses, n_epochs, reg_coeff,
        mse_weight=mse_weight, nll_weight=nll_weight, on_epoch_end=wrapped_on_epoch_end,
    )
    return {
        "baseline_mse": baseline_metrics["mse"], "baseline_nll": baseline_metrics["nll"],
        "final_mse": final_metrics["mse"], "final_nll": final_metrics["nll"],
        "weight": gate.weight.detach().cpu(), "bias": gate.bias.detach().cpu(),
        "coeff_bias": gate.coeff_bias.detach().cpu(), "direction": direction.detach().cpu(),
    }


def main(task: str, layer_idx: int | None, seed: int = 42) -> None:
    adapter = get_adapter(task)
    out_path = adapter.RESULTS_DIR / "psr_proper_probe.pt"
    if out_path.exists() and os.environ.get("FORCE_RERUN", "0") != "1":
        print(f">>> {out_path} already exists, skipping")
        return

    device = "cuda"
    adapter.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    adapter.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model(device, model_name=adapter.MODEL_NAME)
    for p in model.parameters():
        p.requires_grad_(False)
    n_layers = num_layers(model)

    train_items = adapter.to_items(tokenizer, adapter.load_rows("train"))
    dev_items = adapter.to_items(tokenizer, adapter.load_rows("dev"))

    if layer_idx is None:
        with (adapter.RESULTS_DIR / "const_steer_config.json").open() as f:
            layer_idx = json.load(f)["layer"]

    train_responses = load_or_compute_responses(model, tokenizer, train_items, adapter.CACHE_DIR / "teacher_responses_train.json", generate_response)
    dev_responses = load_or_compute_responses(model, tokenizer, dev_items, adapter.CACHE_DIR / "teacher_responses_dev.json", generate_response)
    hidden_size = model.config.hidden_size

    def write_checkpoint_and_log(gate_weight, gate_bias, gate_coeff_bias, direction, baseline: dict, final: dict, completed_epochs) -> None:
        torch.save({
            "weight": gate_weight, "bias": gate_bias, "coeff_bias": gate_coeff_bias,
            "direction": direction, "layer": layer_idx, "task": task,
            "completed_epochs": N_EPOCHS if completed_epochs is None else completed_epochs,
        }, out_path)
        with (adapter.RESULTS_DIR / "psr_proper_train_log.json").open("w") as f:
            json.dump({
                "task": task, "layer": layer_idx, "seed": seed, "lr": LR, "weight_decay": WEIGHT_DECAY,
                "reg_coeff": REG_COEFF, "mse_weight": MSE_WEIGHT, "nll_weight": NLL_WEIGHT,
                "completed_epochs": N_EPOCHS if completed_epochs is None else completed_epochs,
                "baseline_dev_mse": baseline["mse"], "baseline_dev_nll": baseline["nll"],
                "dev_mse": final["mse"], "dev_nll": final["nll"],
            }, f, indent=2)

    def on_epoch_end(epoch, gate, direction, dev_metrics: dict) -> None:
        print(f"epoch={epoch} dev_mse={dev_metrics['mse']:.4f} dev_nll={dev_metrics['nll']:.4f} -- checkpointing")
        write_checkpoint_and_log(
            gate.weight.detach().cpu(), gate.bias.detach().cpu(), gate.coeff_bias.detach().cpu(),
            direction.detach().cpu(), dev_metrics, dev_metrics, epoch,
        )

    result = train_one_config(
        model, tokenizer, layer_idx, seed, n_layers, hidden_size, device,
        train_items, dev_items, train_responses, dev_responses,
        on_epoch_end=on_epoch_end,
    )
    print(f"task={task} layer={layer_idx} baseline_dev_mse={result['baseline_mse']:.4f} final_dev_mse={result['final_mse']:.4f}")
    write_checkpoint_and_log(
        result["weight"], result["bias"], result["coeff_bias"], result["direction"],
        {"mse": result["baseline_mse"], "nll": result["baseline_nll"]},
        {"mse": result["final_mse"], "nll": result["final_nll"]}, None,
    )


def sweep(
    task: str, layers: list[int] | None = None, loss_config_grid: list[dict] | None = None,
    seed: int = 42, device: str = "cuda",
) -> None:
    """Grid over (layer, loss-config), resumable via core.sweep.run_grid_sweep. Writes
    results/<task>/psr_proper_sweep.jsonl (every point's dev mse/nll) and, at the end,
    results/<task>/psr_proper_probe.pt for the single best (layer, loss-config) by final_mse --
    i.e. this REPLACES needing to already know layer=14 up front; main()/psr_proper_probe.pt is
    exactly what the rest of the pipeline (src/generate.py) already expects, unchanged.

    If the sweep is resumed across a restart and the eventual winner turns out to be one of the
    already-completed (skipped-on-resume) points, its tensors are regenerated with one more cheap
    retrain rather than leaving no checkpoint behind -- see the else branch below. Confirmed
    against a real overnight run where this exact situation happened (see
    tests/test_sweep_resume_winner_fix.py)."""
    layers = layers if layers is not None else DEFAULT_SWEEP_LAYERS
    loss_config_grid = loss_config_grid if loss_config_grid is not None else DEFAULT_LOSS_CONFIG_GRID
    adapter = get_adapter(task)
    adapter.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    adapter.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model(device, model_name=adapter.MODEL_NAME)
    for p in model.parameters():
        p.requires_grad_(False)
    n_layers = num_layers(model)
    hidden_size = model.config.hidden_size

    train_items = adapter.to_items(tokenizer, adapter.load_rows("train"))
    dev_items = adapter.to_items(tokenizer, adapter.load_rows("dev"))
    train_responses = load_or_compute_responses(model, tokenizer, train_items, adapter.CACHE_DIR / "teacher_responses_train.json", generate_response)
    dev_responses = load_or_compute_responses(model, tokenizer, dev_items, adapter.CACHE_DIR / "teacher_responses_dev.json", generate_response)

    checkpoints_by_point = {}

    def train_fn(point: dict) -> dict:
        result = train_one_config(
            model, tokenizer, point["layer"], seed, n_layers, hidden_size, device,
            train_items, dev_items, train_responses, dev_responses,
            mse_weight=point["mse_weight"], nll_weight=point["nll_weight"],
        )
        key = (point["layer"], point["mse_weight"], point["nll_weight"])
        checkpoints_by_point[key] = result
        print(f"layer={point['layer']} mse_weight={point['mse_weight']} nll_weight={point['nll_weight']} "
              f"final_mse={result['final_mse']:.4f} final_nll={result['final_nll']:.4f}")
        return {k: v for k, v in result.items() if k not in ("weight", "bias", "coeff_bias", "direction")}

    grid = [{"layer": l, **loss_cfg} for l in layers for loss_cfg in loss_config_grid]
    out_path = adapter.RESULTS_DIR / "psr_proper_sweep.jsonl"
    results, best = run_grid_sweep(grid, train_fn, out_path, key_fields=["layer", "mse_weight", "nll_weight"])
    print(f"\nwrote {len(results)} sweep rows to {out_path}")

    if best is None:
        print("WARNING: sweep produced no usable points -- no probe checkpoint written")
        return
    best_key = (best["layer"], best["mse_weight"], best["nll_weight"])
    if best_key in checkpoints_by_point:
        winner = checkpoints_by_point[best_key]
    else:
        # See src/psr/conceptor/matrix/train.py's sweep() for why this retrains rather than bails:
        # the winner may have been completed in an earlier (resumed) session.
        print(f"Best point {best_key} was completed in an earlier session -- retraining it once more.")
        winner = train_one_config(
            model, tokenizer, best["layer"], seed, n_layers, hidden_size, device,
            train_items, dev_items, train_responses, dev_responses,
            mse_weight=best["mse_weight"], nll_weight=best["nll_weight"],
        )
    out_probe_path = adapter.RESULTS_DIR / "psr_proper_probe.pt"
    torch.save({
        "weight": winner["weight"], "bias": winner["bias"], "coeff_bias": winner["coeff_bias"],
        "direction": winner["direction"], "layer": best["layer"], "task": task,
        "mse_weight": best["mse_weight"], "nll_weight": best["nll_weight"], "completed_epochs": N_EPOCHS,
    }, out_probe_path)
    print(f"wrote {out_probe_path} (best layer={best['layer']}, mse_weight={best['mse_weight']}, "
          f"nll_weight={best['nll_weight']}, final_mse={best['final_mse']:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["caveman", "ifeval"])
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sweep", action="store_true", help="run a layer x loss-config grid sweep instead of a single training run")
    parser.add_argument("--sweep-layers", type=str, default=None, help="comma-separated layer indices, e.g. 2,4,6,...,26")
    parser.add_argument("--loss-configs", type=str, default=None,
                         help='comma-separated "mse_weight:nll_weight" pairs, e.g. "1.0:0.0,0.0:1.0" for pure-MSE + pure-NLL only')
    args = parser.parse_args()
    if args.sweep:
        layers = [int(x) for x in args.sweep_layers.split(",")] if args.sweep_layers else None
        loss_configs = None
        if args.loss_configs:
            loss_configs = []
            for pair in args.loss_configs.split(","):
                mse_w, nll_w = pair.split(":")
                loss_configs.append({"mse_weight": float(mse_w), "nll_weight": float(nll_w)})
        sweep(args.task, layers, loss_configs, args.seed)
    else:
        main(args.task, args.layer, args.seed)
