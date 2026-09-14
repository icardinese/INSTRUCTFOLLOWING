"""Fixed-vector conceptor PSR: direction comes from projecting a diff-in-means vector through the
closed-form conceptor matrix, not gradient-trained. Reuses steering.psr.gate.forward_with_gate_hook
directly -- mechanically identical to proper/train.py's injection, differing only in where
`direction` comes from (see steering/psr/conceptor/direction.py).

No participation-ratio logging here (unlike conceptor/matrix/train.py and
conceptor/selfproj/train.py) -- the conceptor matrix C is used ONCE, offline, to build a single
fixed direction; the correction actually injected at inference is exactly rank-1 regardless of C's
effective dimensionality (see steering/psr/conceptor/rank_diagnostic.py's module docstring and
project finding #3). Attaching PR here would describe a property of an intermediate computation
the final method doesn't inherit.
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
from steering.psr.conceptor.direction import compute_conceptor, project_direction
from steering.psr.data import load_or_pool_separate_poles
from steering.psr.gate import forward_with_gate_hook, init_gate_state
from steering.psr.training_loop import train_gate

N_EPOCHS = 3
LR = float(os.environ.get("PSR_CONCEPTOR_LR", 1e-3))
WEIGHT_DECAY = 1e-4
REG_COEFF = float(os.environ.get("PSR_CONCEPTOR_REG_COEFF", 0.1))
ALPHA = float(os.environ.get("PSR_CONCEPTOR_ALPHA", 4.0))
OUT_TAG = os.environ.get("PSR_CONCEPTOR_OUT_TAG", "")  # e.g. "_alpha8" for one-off sweep runs
NLL_WEIGHT = float(os.environ.get("PSR_CONCEPTOR_NLL_WEIGHT", 0.0))
DEFAULT_SWEEP_LAYERS = list(range(2, 27, 2))
DEFAULT_ALPHA_GRID = [1.0, 2.0, 4.0, 8.0, 16.0]
DEFAULT_NLL_WEIGHT_GRID = [0.0, 0.01, 0.05, 0.1]


def train_one_config(
    model, tokenizer, layer_idx: int, alpha: float, seed: int, n_layers: int, device: str,
    train_items: list[dict], dev_items: list[dict], train_responses: dict, dev_responses: dict,
    cache_dir, lr: float = LR, weight_decay: float = WEIGHT_DECAY, reg_coeff: float = REG_COEFF,
    n_epochs: int = N_EPOCHS, nll_weight: float = NLL_WEIGHT, on_epoch_end=None,
) -> dict:
    """Pools (or loads cached pooled) activations at layer_idx, builds C and the projected fixed
    direction at the given alpha, then trains the gate on top of it. Returns dev metrics (JSON-safe)
    plus the trained/derived tensors (not JSON-safe -- see proper/train.py's train_one_config for
    the same split and why)."""
    set_seed(seed)
    base_pool, instr_pool = load_or_pool_separate_poles(model, tokenizer, train_items, layer_idx, train_responses, cache_dir)
    diff_mean_direction = instr_pool.mean(0) - base_pool.mean(0)
    diff_mean_direction = diff_mean_direction / diff_mean_direction.norm()

    conceptor = compute_conceptor(torch.cat([base_pool, instr_pool], dim=0), alpha=alpha)
    direction = project_direction(conceptor, diff_mean_direction)

    hidden_size = base_pool.shape[1]
    gate = init_gate_state(hidden_size, device)
    optimizer = torch.optim.Adam(gate.parameters(), lr=lr, weight_decay=weight_decay)
    forward_fn = lambda pair: forward_with_gate_hook(model, gate, direction, layer_idx, pair["full_base"], pair["n_resp"])

    wrapped_on_epoch_end = (lambda epoch, metrics: on_epoch_end(epoch, gate, direction, conceptor, metrics)) if on_epoch_end else None
    baseline_metrics, final_metrics = train_gate(
        model, tokenizer, forward_fn, optimizer, layer_idx, n_layers,
        train_items, dev_items, train_responses, dev_responses, n_epochs, reg_coeff,
        nll_weight=nll_weight, on_epoch_end=wrapped_on_epoch_end,
    )
    return {
        "baseline_mse": baseline_metrics["mse"], "baseline_nll": baseline_metrics["nll"],
        "final_mse": final_metrics["mse"], "final_nll": final_metrics["nll"],
        "weight": gate.weight.detach().cpu(), "bias": gate.bias.detach().cpu(),
        "coeff_bias": gate.coeff_bias.detach().cpu(),
        "conceptor": conceptor.detach().cpu(), "direction": direction.detach().cpu(),
    }


def main(task: str, layer_idx: int | None, seed: int = 42) -> None:
    adapter = get_adapter(task)
    out_path = adapter.RESULTS_DIR / f"psr_conceptor_probe{OUT_TAG}.pt"
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

    print(f"loading/pooling activations at layer {layer_idx} for the conceptor and diff-in-means direction")

    def write_checkpoint_and_log(gate_weight, gate_bias, gate_coeff_bias, conceptor, direction, baseline: dict, final: dict, completed_epochs) -> None:
        torch.save({
            "weight": gate_weight, "bias": gate_bias, "coeff_bias": gate_coeff_bias,
            "conceptor": conceptor, "direction": direction, "layer": layer_idx, "alpha": ALPHA, "task": task,
            "completed_epochs": N_EPOCHS if completed_epochs is None else completed_epochs,
        }, out_path)
        with (adapter.RESULTS_DIR / f"psr_conceptor_train_log{OUT_TAG}.json").open("w") as f:
            json.dump({
                "task": task, "layer": layer_idx, "alpha": ALPHA, "seed": seed, "lr": LR,
                "weight_decay": WEIGHT_DECAY, "reg_coeff": REG_COEFF, "nll_weight": NLL_WEIGHT,
                "completed_epochs": N_EPOCHS if completed_epochs is None else completed_epochs,
                "baseline_dev_mse": baseline["mse"], "baseline_dev_nll": baseline["nll"],
                "dev_mse": final["mse"], "dev_nll": final["nll"],
            }, f, indent=2)

    def on_epoch_end(epoch, gate, direction, conceptor, dev_metrics: dict) -> None:
        print(f"epoch={epoch} dev_mse={dev_metrics['mse']:.4f} dev_nll={dev_metrics['nll']:.4f} -- checkpointing")
        write_checkpoint_and_log(
            gate.weight.detach().cpu(), gate.bias.detach().cpu(), gate.coeff_bias.detach().cpu(),
            conceptor.detach().cpu(), direction.detach().cpu(), dev_metrics, dev_metrics, epoch,
        )

    result = train_one_config(
        model, tokenizer, layer_idx, ALPHA, seed, n_layers, device,
        train_items, dev_items, train_responses, dev_responses, adapter.CACHE_DIR,
        on_epoch_end=on_epoch_end,
    )
    print(f"task={task} layer={layer_idx} alpha={ALPHA} baseline_dev_mse={result['baseline_mse']:.4f} final_dev_mse={result['final_mse']:.4f}")
    write_checkpoint_and_log(
        result["weight"], result["bias"], result["coeff_bias"], result["conceptor"], result["direction"],
        {"mse": result["baseline_mse"], "nll": result["baseline_nll"]},
        {"mse": result["final_mse"], "nll": result["final_nll"]}, None,
    )


def sweep(
    task: str, layers: list[int] | None = None, alpha_grid: list[float] | None = None,
    nll_weight_grid: list[float] | None = None, seed: int = 42,
) -> None:
    """Grid over (layer, alpha, nll_weight). See src/psr/proper/train.py's sweep() docstring for
    the resumability/best-checkpoint tradeoff this shares (same core.sweep.run_grid_sweep call)."""
    layers = layers if layers is not None else DEFAULT_SWEEP_LAYERS
    alpha_grid = alpha_grid if alpha_grid is not None else DEFAULT_ALPHA_GRID
    nll_weight_grid = nll_weight_grid if nll_weight_grid is not None else DEFAULT_NLL_WEIGHT_GRID
    adapter = get_adapter(task)
    device = "cuda"
    adapter.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    adapter.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model(device, model_name=adapter.MODEL_NAME)
    for p in model.parameters():
        p.requires_grad_(False)
    n_layers = num_layers(model)

    train_items = adapter.to_items(tokenizer, adapter.load_rows("train"))
    dev_items = adapter.to_items(tokenizer, adapter.load_rows("dev"))
    train_responses = load_or_compute_responses(model, tokenizer, train_items, adapter.CACHE_DIR / "teacher_responses_train.json", generate_response)
    dev_responses = load_or_compute_responses(model, tokenizer, dev_items, adapter.CACHE_DIR / "teacher_responses_dev.json", generate_response)

    checkpoints_by_point = {}

    def train_fn(point: dict) -> dict:
        result = train_one_config(
            model, tokenizer, point["layer"], point["alpha"], seed, n_layers, device,
            train_items, dev_items, train_responses, dev_responses, adapter.CACHE_DIR,
            nll_weight=point["nll_weight"],
        )
        key = (point["layer"], point["alpha"], point["nll_weight"])
        checkpoints_by_point[key] = result
        print(f"layer={point['layer']} alpha={point['alpha']} nll_weight={point['nll_weight']} "
              f"final_mse={result['final_mse']:.4f} final_nll={result['final_nll']:.4f}")
        return {k: v for k, v in result.items() if k not in ("weight", "bias", "coeff_bias", "conceptor", "direction")}

    grid = [{"layer": l, "alpha": a, "nll_weight": w} for l in layers for a in alpha_grid for w in nll_weight_grid]
    out_path = adapter.RESULTS_DIR / "psr_conceptor_sweep.jsonl"
    results, best = run_grid_sweep(grid, train_fn, out_path, key_fields=["layer", "alpha", "nll_weight"])
    print(f"\nwrote {len(results)} sweep rows to {out_path}")

    if best is None:
        print("WARNING: sweep produced no usable points -- no probe checkpoint written")
        return
    best_key = (best["layer"], best["alpha"], best["nll_weight"])
    if best_key not in checkpoints_by_point:
        print(f"NOTE: best point {best_key} was already completed in a previous session -- its "
              f"trained tensors aren't in memory this run. Delete its row from {out_path} and "
              f"rerun sweep() to regenerate a checkpoint for it.")
        return
    winner = checkpoints_by_point[best_key]
    out_probe_path = adapter.RESULTS_DIR / "psr_conceptor_probe.pt"
    torch.save({
        "weight": winner["weight"], "bias": winner["bias"], "coeff_bias": winner["coeff_bias"],
        "conceptor": winner["conceptor"], "direction": winner["direction"],
        "layer": best["layer"], "alpha": best["alpha"], "nll_weight": best["nll_weight"],
        "task": task, "completed_epochs": N_EPOCHS,
    }, out_probe_path)
    print(f"wrote {out_probe_path} (best layer={best['layer']}, alpha={best['alpha']}, "
          f"nll_weight={best['nll_weight']}, final_mse={best['final_mse']:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["caveman", "ifeval"])
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--sweep-layers", type=str, default=None)
    parser.add_argument("--sweep-alphas", type=str, default=None)
    parser.add_argument("--sweep-nll-weights", type=str, default=None)
    args = parser.parse_args()
    if args.sweep:
        layers = [int(x) for x in args.sweep_layers.split(",")] if args.sweep_layers else None
        alphas = [float(x) for x in args.sweep_alphas.split(",")] if args.sweep_alphas else None
        nll_weights = [float(x) for x in args.sweep_nll_weights.split(",")] if args.sweep_nll_weights else None
        sweep(args.task, layers, alphas, nll_weights, args.seed)
    else:
        main(args.task, args.layer, args.seed)
