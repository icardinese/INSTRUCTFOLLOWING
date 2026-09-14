"""Proper PSR training. Task-agnostic by construction: --task selects an adapter at runtime, so this
is ONE script that runs against caveman OR ifeval, not two near-identical copies.

train_one_config() holds the actual "build a gate+direction for one (layer, hyperparameter) point
and train it" logic; main() (single run) and sweep() (grid over layer x nll_weight) are both thin
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
# Auxiliary NLL loss weight (Eq. 4) -- 0.0 by default so every existing result reproduces
# unchanged unless this is explicitly set. See steering/psr/nll.py and steering/psr/training_loop.py.
NLL_WEIGHT = float(os.environ.get("PSR_PROPER_NLL_WEIGHT", 0.0))
# Layer grid for sweep() -- a dense range, not just DEFAULT_LAYER_FRACTIONS's 4 points, matching
# the project's own earlier one-off "layers 2-26" sweep (see project handoff finding #4) now made
# a permanent, reusable, resumable script instead of a one-off notebook cell.
DEFAULT_SWEEP_LAYERS = list(range(2, 27, 2))
DEFAULT_NLL_WEIGHT_GRID = [0.0, 0.01, 0.05, 0.1]


def train_one_config(
    model, tokenizer, layer_idx: int, seed: int, n_layers: int, hidden_size: int, device: str,
    train_items: list[dict], dev_items: list[dict], train_responses: dict, dev_responses: dict,
    lr: float = LR, weight_decay: float = WEIGHT_DECAY, reg_coeff: float = REG_COEFF,
    n_epochs: int = N_EPOCHS, nll_weight: float = NLL_WEIGHT, on_epoch_end=None,
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
        nll_weight=nll_weight, on_epoch_end=wrapped_on_epoch_end,
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
                "reg_coeff": REG_COEFF, "nll_weight": NLL_WEIGHT,
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


def sweep(task: str, layers: list[int] | None = None, nll_weight_grid: list[float] | None = None, seed: int = 42) -> None:
    """Grid over (layer, nll_weight), resumable via core.sweep.run_grid_sweep. Writes
    results/<task>/psr_proper_sweep.jsonl (every point's dev mse/nll) and, at the end,
    results/<task>/psr_proper_probe.pt for the single best (layer, nll_weight) by final_mse --
    i.e. this REPLACES needing to already know layer=14 up front; main()/psr_proper_probe.pt is
    exactly what the rest of the pipeline (src/generate.py) already expects, unchanged.

    Tradeoff, stated plainly: the winning checkpoint's tensors are only available to save if that
    grid point was actually (re)trained THIS session. If the sweep is resumed across a restart and
    the eventual winner was one of the already-completed (skipped-on-resume) points, this prints a
    clear instruction rather than silently writing a wrong or missing checkpoint -- see the NOTE
    branch below."""
    layers = layers if layers is not None else DEFAULT_SWEEP_LAYERS
    nll_weight_grid = nll_weight_grid if nll_weight_grid is not None else DEFAULT_NLL_WEIGHT_GRID
    adapter = get_adapter(task)
    device = "cuda"
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
            train_items, dev_items, train_responses, dev_responses, nll_weight=point["nll_weight"],
        )
        checkpoints_by_point[(point["layer"], point["nll_weight"])] = result
        print(f"layer={point['layer']} nll_weight={point['nll_weight']} "
              f"final_mse={result['final_mse']:.4f} final_nll={result['final_nll']:.4f}")
        return {k: v for k, v in result.items() if k not in ("weight", "bias", "coeff_bias", "direction")}

    grid = [{"layer": l, "nll_weight": w} for l in layers for w in nll_weight_grid]
    out_path = adapter.RESULTS_DIR / "psr_proper_sweep.jsonl"
    results, best = run_grid_sweep(grid, train_fn, out_path, key_fields=["layer", "nll_weight"])
    print(f"\nwrote {len(results)} sweep rows to {out_path}")

    if best is None:
        print("WARNING: sweep produced no usable points -- no probe checkpoint written")
        return
    best_key = (best["layer"], best["nll_weight"])
    if best_key not in checkpoints_by_point:
        print(f"NOTE: best point (layer={best['layer']}, nll_weight={best['nll_weight']}) was "
              f"already completed in a previous session, so its trained tensors aren't in memory "
              f"this run. Delete its row from {out_path} and rerun sweep() to regenerate a "
              f"checkpoint for it, or just call main(task, layer={best['layer']}) directly with "
              f"PSR_PROPER_NLL_WEIGHT={best['nll_weight']}.")
        return
    winner = checkpoints_by_point[best_key]
    out_probe_path = adapter.RESULTS_DIR / "psr_proper_probe.pt"
    torch.save({
        "weight": winner["weight"], "bias": winner["bias"], "coeff_bias": winner["coeff_bias"],
        "direction": winner["direction"], "layer": best["layer"], "task": task,
        "nll_weight": best["nll_weight"], "completed_epochs": N_EPOCHS,
    }, out_probe_path)
    print(f"wrote {out_probe_path} (best layer={best['layer']}, nll_weight={best['nll_weight']}, "
          f"final_mse={best['final_mse']:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["caveman", "ifeval"])
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sweep", action="store_true", help="run a layer x nll_weight grid sweep instead of a single training run")
    parser.add_argument("--sweep-layers", type=str, default=None, help="comma-separated layer indices, e.g. 2,4,6,...,26")
    parser.add_argument("--sweep-nll-weights", type=str, default=None, help="comma-separated nll_weight values, e.g. 0,0.01,0.1")
    args = parser.parse_args()
    if args.sweep:
        layers = [int(x) for x in args.sweep_layers.split(",")] if args.sweep_layers else None
        nll_weights = [float(x) for x in args.sweep_nll_weights.split(",")] if args.sweep_nll_weights else None
        sweep(args.task, layers, nll_weights, args.seed)
    else:
        main(args.task, args.layer, args.seed)
