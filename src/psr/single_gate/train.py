"""SG -- single-gated steering with a FIXED diff-in-means direction.

The missing cell in the ablation taxonomy: one trained gate at one layer, scaling a closed-form
diff-in-means vector that is never updated by gradient descent.

WHY THIS FILE EXISTS. src/psr/old_baseline/train.py was previously labelled "S-PSR" and treated as
the diff-in-means baseline, but it is not comparable to anything else in this codebase: it does
OFFLINE regression against precomputed fixed activation pairs, for 200 epochs, with no
answer-only masking and no subsequent-layers loss (its own module docstring says so). So a gap
between it and S-PSR confounds FOUR differences at once -- direction source, offline vs. live
forward pass, single-layer vs. subsequent-layers objective, and 200 vs. 3 epochs -- and cannot be
read as evidence about the direction.

This file fixes that by construction. It calls the same steering.psr.training_loop.train_gate with
the same epochs, LR, weight decay, dead-gate regularization, answer-only masking and
subsequent-layers MSE as src/psr/proper/train.py (S-PSR) and src/psr/conceptor/train.py
(SG + Conceptor). The ONLY difference from S-PSR is that `direction` is a frozen tensor rather
than a trainable leaf in the optimizer; the only difference from SG + Conceptor is the absence of
the conceptor projection step. That makes two clean one-variable comparisons:

    SG  ->  S-PSR            direction source: frozen diff-in-means vs. gradient-trained
    SG  ->  SG + Conceptor   direction source: raw diff-in-means vs. conceptor-reshaped
    SG  ->  MG               gate structure: one layer vs. all layers (src/psr/all_layer/train.py
                             with --direction-source diff_in_means uses the same frozen DiM vector)

NO ALPHA DIMENSION. Unlike the conceptor variants there is no aperture hyperparameter here, so the
sweep grid is layers x loss-configs only -- 13 x 2 = 26 points, the same size as S-PSR's and a
fifth of SG + Conceptor's. This is the cheapest trainable variant in the project.

DIRECTION EXTRACTION. Prompt last-token (base vs. instructed), matching Const, Stolfo, and the
fixed-vector conceptor as of 2026-09-17. Deliberately NOT the response-token-pooled variant: that
was this project's earlier default and measured clearly worse for conceptor, so using it here
would make the SG-vs-conceptor comparison a rerun of the already-settled extraction-point question
instead of a clean test of the projection step.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from adapters.registry import get_adapter
from core.generation_cache import load_or_compute_responses
from core.model_common import generate_response, load_model, num_layers, generate_response_with_meta
from core.reproducibility import set_seed
from core.sweep import run_grid_sweep
from steering.psr.data import load_or_pool_prompt_last_token
from steering.psr.gate import forward_with_gate_hook, init_gate_state
from steering.psr.reference_config import (
    DEFAULT_LOSS_BALANCE,
    N_EPOCHS_MSE,
    WEIGHT_DECAY as REF_WEIGHT_DECAY,
    epochs_for,
    loss_balance,
)
from steering.psr.training_loop import train_gate
from adapters.registry import TASK_CHOICES

N_EPOCHS = N_EPOCHS_MSE  # reference: 15 for MSE, 7 for LL -- see epochs_for()
LR = float(os.environ.get("PSR_SG_LR", 1e-3))
WEIGHT_DECAY = REF_WEIGHT_DECAY  # 1e-6; the reference's dataclass default of 1e-4 is never used
LOSS_BALANCE = os.environ.get("PSR_SG_LOSS_BALANCE", DEFAULT_LOSS_BALANCE)
REG_COEFF = float(os.environ.get("PSR_SG_REG_COEFF", loss_balance(LOSS_BALANCE)["reg_coeff"]))
NORMALIZE_PSI = loss_balance(LOSS_BALANCE)["normalize_psi"]
MSE_WEIGHT = float(os.environ.get("PSR_SG_MSE_WEIGHT", 1.0))
NLL_WEIGHT = float(os.environ.get("PSR_SG_NLL_WEIGHT", 0.0))
OUT_TAG = os.environ.get("PSR_SG_OUT_TAG", "")

DEFAULT_SWEEP_LAYERS = list(range(2, 27, 2))
DEFAULT_LOSS_CONFIG_GRID = [
    {"mse_weight": 1.0, "nll_weight": 0.0},   # pure MSE (H&V's "_MSE" variant)
    {"mse_weight": 0.0, "nll_weight": 1.0},   # pure NLL (H&V's "_LL" variant)
]


def build_direction(model, tokenizer, train_items, layer_idx: int, cache_dir, device: str) -> torch.Tensor:
    """Frozen unit diff-in-means at layer_idx. Unit-normalized so the gate's learned coefficient is
    the only thing setting magnitude -- raw diff norms vary substantially with depth, which would
    otherwise silently rescale each layer's gate and make the layer sweep partly a comparison of
    direction magnitudes rather than of layers."""
    base_pool, instr_pool = load_or_pool_prompt_last_token(model, tokenizer, train_items, layer_idx, cache_dir)
    # load_or_pool_prompt_last_token returns CPU tensors (it pools on CPU to bound memory); the
    # correction has to live on the model's device or the hook hits the cuda-vs-cpu device mismatch
    # documented in src/psr/conceptor/train.py.
    diff = instr_pool.to(device).mean(0) - base_pool.to(device).mean(0)
    return (diff / diff.norm()).detach().requires_grad_(False)


def train_one_config(
    model, tokenizer, layer_idx: int, seed: int, n_layers: int, hidden_size: int, device: str,
    train_items: list[dict], dev_items: list[dict], train_responses: dict, dev_responses: dict,
    cache_dir, lr: float = LR, weight_decay: float = WEIGHT_DECAY, reg_coeff: float = REG_COEFF,
    n_epochs: int | None = None, mse_weight: float = MSE_WEIGHT, nll_weight: float = NLL_WEIGHT,
    normalize_psi: bool = NORMALIZE_PSI,
    on_epoch_end=None,
) -> dict:
    """Trains the gate on top of a frozen direction. Same return contract as proper/conceptor's
    train_one_config: JSON-safe dev metrics plus the trained tensors in one dict."""
    # Reference epoch budget is objective-dependent (15 MSE / 7 LL). Training both endpoints
    # for the same number of epochs confounds 'which objective wins' with 'which converged'.
    if n_epochs is None:
        n_epochs = epochs_for(mse_weight, nll_weight)
    set_seed(seed)
    gate = init_gate_state(hidden_size, device)
    direction = build_direction(model, tokenizer, train_items, layer_idx, cache_dir, device)

    # Only the gate's parameters go to the optimizer -- `direction` is frozen. This single line is
    # the entire difference from S-PSR, which additionally passes [direction] here.
    optimizer = torch.optim.AdamW(gate.parameters(), lr=lr, weight_decay=weight_decay)

    forward_fn = lambda pair: forward_with_gate_hook(
        model, gate, direction, layer_idx, pair["full_base"], pair["n_resp"]
    )
    wrapped_on_epoch_end = (lambda epoch, metrics: on_epoch_end(epoch, gate, direction, metrics)) if on_epoch_end else None
    baseline_metrics, final_metrics = train_gate(
        model, tokenizer, forward_fn, optimizer, layer_idx, n_layers,
        train_items, dev_items, train_responses, dev_responses, n_epochs, reg_coeff,
        mse_weight=mse_weight, nll_weight=nll_weight, on_epoch_end=wrapped_on_epoch_end,
        normalize_psi=normalize_psi,
    )
    return {
        "completed_epochs": n_epochs,
        "baseline_mse": baseline_metrics["mse"], "baseline_nll": baseline_metrics["nll"],
        "final_mse": final_metrics["mse"], "final_nll": final_metrics["nll"],
        "weight": gate.weight.detach().cpu(), "bias": gate.bias.detach().cpu(),
        "coeff_bias": gate.coeff_bias.detach().cpu(), "direction": direction.detach().cpu(),
    }


def _setup(task: str, device: str):
    adapter = get_adapter(task)
    adapter.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    adapter.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model(device, model_name=adapter.MODEL_NAME)
    for p in model.parameters():
        p.requires_grad_(False)
    train_items = adapter.to_items(tokenizer, adapter.load_rows("train"))
    dev_items = adapter.to_items(tokenizer, adapter.load_rows("dev"))
    train_responses = load_or_compute_responses(model, tokenizer, train_items, adapter.CACHE_DIR / "teacher_responses_train.json", generate_response_with_meta)
    dev_responses = load_or_compute_responses(model, tokenizer, dev_items, adapter.CACHE_DIR / "teacher_responses_dev.json", generate_response_with_meta)
    return adapter, model, tokenizer, num_layers(model), train_items, dev_items, train_responses, dev_responses


def _save(out_path, result: dict, layer_idx: int, task: str, mse_weight: float, nll_weight: float) -> None:
    # Same flat single-layer checkpoint shape proper/conceptor use ("direction" singular + "layer"),
    # so scripts/compute_cosine_similarities.py's single-layer loader reads it with no special case.
    torch.save({
        "weight": result["weight"], "bias": result["bias"], "coeff_bias": result["coeff_bias"],
        "direction": result["direction"], "layer": layer_idx, "task": task,
        "mse_weight": mse_weight, "nll_weight": nll_weight,
        "completed_epochs": result["completed_epochs"],
    }, out_path)


def main(task: str, layer_idx: int | None, seed: int = 42, device: str = "cuda") -> None:
    adapter, model, tokenizer, n_layers, train_items, dev_items, train_responses, dev_responses = _setup(task, device)
    out_path = adapter.RESULTS_DIR / f"psr_sg_probe{OUT_TAG}.pt"
    if out_path.exists() and os.environ.get("FORCE_RERUN", "0") != "1":
        print(f">>> {out_path} already exists, skipping")
        return
    layer_idx = layer_idx if layer_idx is not None else n_layers // 2
    print(f"SG: layer={layer_idx}, mse_weight={MSE_WEIGHT}, nll_weight={NLL_WEIGHT}")
    result = train_one_config(
        model, tokenizer, layer_idx, seed, n_layers, model.config.hidden_size, device,
        train_items, dev_items, train_responses, dev_responses, adapter.CACHE_DIR,
        mse_weight=MSE_WEIGHT, nll_weight=NLL_WEIGHT,
    )
    _save(out_path, result, layer_idx, task, MSE_WEIGHT, NLL_WEIGHT)
    print(f"task={task} layer={layer_idx} baseline_dev_mse={result['baseline_mse']:.4f} "
          f"final_dev_mse={result['final_mse']:.4f}")
    print(f"wrote {out_path}")


def sweep(task: str, layers: list[int] | None = None, loss_config_grid: list[dict] | None = None,
          seed: int = 42, device: str = "cuda") -> None:
    """Grid over (layer, loss-config) -- 13 x 2 = 26 points by default. Resumable via
    core.sweep.run_grid_sweep, same as every other variant."""
    layers = layers if layers is not None else DEFAULT_SWEEP_LAYERS
    loss_config_grid = loss_config_grid if loss_config_grid is not None else DEFAULT_LOSS_CONFIG_GRID
    adapter, model, tokenizer, n_layers, train_items, dev_items, train_responses, dev_responses = _setup(task, device)

    checkpoints_by_point = {}

    def train_fn(point: dict) -> dict:
        result = train_one_config(
            model, tokenizer, point["layer"], seed, n_layers, model.config.hidden_size, device,
            train_items, dev_items, train_responses, dev_responses, adapter.CACHE_DIR,
            mse_weight=point["mse_weight"], nll_weight=point["nll_weight"],
        )
        checkpoints_by_point[(point["layer"], point["mse_weight"], point["nll_weight"])] = result
        print(f"layer={point['layer']} mse_weight={point['mse_weight']} nll_weight={point['nll_weight']} "
              f"final_mse={result['final_mse']:.4f} final_nll={result['final_nll']:.4f}")
        return {k: v for k, v in result.items() if k not in ("weight", "bias", "coeff_bias", "direction")}

    grid = [{"layer": l, **cfg} for l in layers for cfg in loss_config_grid]
    out_path = adapter.RESULTS_DIR / "psr_sg_sweep.jsonl"
    results, best = run_grid_sweep(grid, train_fn, out_path, key_fields=["layer", "mse_weight", "nll_weight"])
    print(f"\nwrote {len(results)} sweep rows to {out_path}")

    if best is None:
        print("WARNING: sweep produced no usable points -- no probe checkpoint written")
        return
    best_key = (best["layer"], best["mse_weight"], best["nll_weight"])
    if best_key in checkpoints_by_point:
        winner = checkpoints_by_point[best_key]
    else:
        # Retrain rather than bail: a resumed sweep whose winner completed in an EARLIER session
        # has no in-memory tensors for it, and silently writing no checkpoint (the original bug in
        # this project's sweeps) loses the result entirely. See src/psr/proper/train.py's sweep().
        print(f"Best point {best_key} was completed in an earlier session -- retraining it once more.")
        winner = train_one_config(
            model, tokenizer, best["layer"], seed, n_layers, model.config.hidden_size, device,
            train_items, dev_items, train_responses, dev_responses, adapter.CACHE_DIR,
            mse_weight=best["mse_weight"], nll_weight=best["nll_weight"],
        )
    probe_path = adapter.RESULTS_DIR / "psr_sg_probe.pt"
    _save(probe_path, winner, best["layer"], task, best["mse_weight"], best["nll_weight"])
    print(f"wrote {probe_path} (layer={best['layer']}, mse_weight={best['mse_weight']}, "
          f"nll_weight={best['nll_weight']}, final_mse={best['final_mse']:.4f})")
    print("NOTE: this checkpoint is the lowest-final_mse point, which is NOT the same thing as the "
          "best generation quality -- run evals/layer_hparam_search.py --variant single_gate for "
          "the judged layer selection.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=TASK_CHOICES)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--sweep-layers", type=str, default=None)
    parser.add_argument("--loss-configs", type=str, default=None,
                         help='JSON list, e.g. \'[{"mse_weight":1.0,"nll_weight":0.0}]\'')
    args = parser.parse_args()
    sweep_layers = [int(x) for x in args.sweep_layers.split(",")] if args.sweep_layers else None
    loss_configs = json.loads(args.loss_configs) if args.loss_configs else None
    if args.sweep:
        sweep(args.task, sweep_layers, loss_configs, args.seed)
    else:
        main(args.task, args.layer, args.seed)
