"""Gated clamp: PSR's learned gate scaling Stolfo's closed-form projection shortfall.

    h' = h + w(h) * (tau_l - h.u_l) * u_l        at each hooked layer l

u_l and tau_l are closed-form and FROZEN (unit diff-in-means from the prompt's last token, and the
instructed pole's own mean projection onto it -- identical to what the ungated Stolfo baseline
uses, imported from the same helpers so they cannot drift apart). Only the gate trains.

ONE SCRIPT, BOTH CELLS, via --layers:
    --layers 16        -> SG+Clamp, single-gated
    (omit --layers)    -> MG+Clamp, all layers, jointly optimized in one shared forward pass

This is the direct test of "does PSR's gating add anything on top of Stolfo's functional form."
The three ingredients are now fully separable across the experiment suite:

    functional form   additive (h + c*u)  vs  clamp (h + (tau - h.u)*u)
    gating            none  vs  SG  vs  MG
    surface           all positions  vs  response-only

with the ungated all-positions clamp being Stolfo as published, and this file supplying the gated
response-only corners. Gating is only meaningful on the response-only surface: a gate trained with
answer_only_mask has no signal on prompt positions by construction, so gated + all-positions would
be a gate that is simply undefined over most of what it touches.

WHY THE GATE MIGHT NOT HELP, stated up front so the result is interpretable either way: an ungated
clamp already lands every position exactly on h.u == tau, which is the strongest possible
enforcement of that constraint. A gate can only scale that shortfall -- partially clamping
(w < 1), overshooting (w > 1), or skipping positions (w = 0). If the clamp's value comes from the
guarantee itself, gating can only weaken it, and the honest finding is "the clamp is already
saturated." If instead clamping uniformly over-constrains some positions, the gate should recover
real headroom. Both outcomes are publishable; neither is assumed here.
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
from steering.clamp.hooks import forward_with_gated_clamp_hook, forward_with_multi_gated_clamp_hook
from steering.const.direction import compute_diff_mean_direction
from steering.psr.data import load_or_pool_prompt_last_token_all_layers
from steering.psr.gate import init_gate_state
from steering.psr.reference_config import (
    DEFAULT_LOSS_BALANCE,
    N_EPOCHS_MSE,
    WEIGHT_DECAY as REF_WEIGHT_DECAY,
    epochs_for,
    loss_balance,
)
from steering.psr.training_loop import train_gate
from steering.stolfo.direction import compute_target_projection
from adapters.registry import TASK_CHOICES

N_EPOCHS = N_EPOCHS_MSE  # reference: 15 for MSE, 7 for LL -- see epochs_for()
LR = float(os.environ.get("PSR_CLAMP_LR", 1e-3))
WEIGHT_DECAY = REF_WEIGHT_DECAY  # 1e-6; the reference's dataclass default of 1e-4 is never used
LOSS_BALANCE = os.environ.get("PSR_CLAMP_LOSS_BALANCE", DEFAULT_LOSS_BALANCE)
REG_COEFF = float(os.environ.get("PSR_CLAMP_REG_COEFF", loss_balance(LOSS_BALANCE)["reg_coeff"]))
NORMALIZE_PSI = loss_balance(LOSS_BALANCE)["normalize_psi"]
MSE_WEIGHT = float(os.environ.get("PSR_CLAMP_MSE_WEIGHT", 1.0))
NLL_WEIGHT = float(os.environ.get("PSR_CLAMP_NLL_WEIGHT", 0.0))

DEFAULT_LOSS_CONFIG_GRID = [
    {"mse_weight": 1.0, "nll_weight": 0.0},
    {"mse_weight": 0.0, "nll_weight": 1.0},
]


def build_clamp_params(model, tokenizer, train_items, layer_indices, n_layers, cache_dir, device):
    """Frozen (direction, target) per layer, from one all-layers pooling pass. Uses the SAME
    compute_diff_mean_direction / compute_target_projection the ungated Stolfo baseline uses."""
    base_by_layer, instr_by_layer = load_or_pool_prompt_last_token_all_layers(
        model, tokenizer, train_items, n_layers, cache_dir)
    directions, targets = {}, {}
    for l in layer_indices:
        base, instr = base_by_layer[l].to(device), instr_by_layer[l].to(device)
        d = compute_diff_mean_direction(base, instr)
        directions[l] = d.detach().requires_grad_(False)
        targets[l] = compute_target_projection(instr, d)
    return directions, targets


def train_one_config(
    model, tokenizer, layer_indices: list[int], seed: int, n_layers: int, hidden_size: int,
    device: str, train_items: list[dict], dev_items: list[dict], train_responses: dict,
    dev_responses: dict, cache_dir, lr: float = LR, weight_decay: float = WEIGHT_DECAY,
    reg_coeff: float = REG_COEFF, n_epochs: int | None = None,
    normalize_psi: bool = NORMALIZE_PSI,
    mse_weight: float = MSE_WEIGHT, nll_weight: float = NLL_WEIGHT, on_epoch_end=None,
) -> dict:
    # Reference epoch budget is objective-dependent (15 MSE / 7 LL). Training both endpoints
    # for the same number of epochs confounds 'which objective wins' with 'which converged'.
    if n_epochs is None:
        n_epochs = epochs_for(mse_weight, nll_weight)
    set_seed(seed)
    gates = {l: init_gate_state(hidden_size, device) for l in layer_indices}
    directions, targets = build_clamp_params(
        model, tokenizer, train_items, layer_indices, n_layers, cache_dir, device)

    # Gates only -- directions and targets are closed-form and frozen.
    params = [p for l in layer_indices for p in gates[l].parameters()]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    if len(layer_indices) == 1:
        l = layer_indices[0]
        forward_fn = lambda pair: forward_with_gated_clamp_hook(
            model, gates[l], directions[l], targets[l], l, pair["full_base"], pair["n_resp"])
    else:
        forward_fn = lambda pair: forward_with_multi_gated_clamp_hook(
            model, gates, directions, targets, layer_indices, pair["full_base"], pair["n_resp"])

    # MSE over the earliest hooked layer and everything after it -- matches what each additive
    # counterpart does (single-layer variants pass their own layer; all-layer ones pass 0).
    mse_from = min(layer_indices)
    wrapped = (lambda e, m: on_epoch_end(e, gates, directions, targets, m)) if on_epoch_end else None
    baseline_metrics, final_metrics = train_gate(
        model, tokenizer, forward_fn, optimizer, mse_from, n_layers,
        train_items, dev_items, train_responses, dev_responses, n_epochs, reg_coeff,
        mse_weight=mse_weight, nll_weight=nll_weight, on_epoch_end=wrapped,
        normalize_psi=normalize_psi,
    )
    return {
        "completed_epochs": n_epochs,
        "baseline_mse": baseline_metrics["mse"], "baseline_nll": baseline_metrics["nll"],
        "final_mse": final_metrics["mse"], "final_nll": final_metrics["nll"],
        "gates": {l: {"weight": gates[l].weight.detach().cpu(), "bias": gates[l].bias.detach().cpu(),
                       "coeff_bias": gates[l].coeff_bias.detach().cpu()} for l in layer_indices},
        "directions": {l: directions[l].detach().cpu() for l in layer_indices},
        "targets": {l: float(targets[l]) for l in layer_indices},
    }


def _tag(layer_indices, n_layers):
    return "sg_clamp" if len(layer_indices) == 1 else "mg_clamp"


def _save(path, result, layer_indices, task, mse_weight, nll_weight):
    torch.save({
        "gates": result["gates"], "directions": result["directions"], "targets": result["targets"],
        "layer_indices": layer_indices, "direction_source": "clamp_dim",
        "task": task, "mse_weight": mse_weight, "nll_weight": nll_weight,
        "completed_epochs": result["completed_epochs"],
    }, path)


def _setup(task, device):
    adapter = get_adapter(task)
    adapter.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    adapter.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model(device, model_name=adapter.MODEL_NAME)
    for p in model.parameters():
        p.requires_grad_(False)
    train_items = adapter.to_items(tokenizer, adapter.load_rows("train"))
    dev_items = adapter.to_items(tokenizer, adapter.load_rows("dev"))
    tr = load_or_compute_responses(model, tokenizer, train_items, adapter.CACHE_DIR / "teacher_responses_train.json", generate_response_with_meta)
    dv = load_or_compute_responses(model, tokenizer, dev_items, adapter.CACHE_DIR / "teacher_responses_dev.json", generate_response_with_meta)
    return adapter, model, tokenizer, num_layers(model), train_items, dev_items, tr, dv


def sweep(task, layers=None, loss_config_grid=None, seed=42, device="cuda"):
    loss_config_grid = loss_config_grid or DEFAULT_LOSS_CONFIG_GRID
    adapter, model, tokenizer, n_layers, train_items, dev_items, tr, dv = _setup(task, device)
    layer_indices = layers if layers is not None else list(range(n_layers))
    tag = _tag(layer_indices, n_layers)
    print(f"{tag}: {len(layer_indices)} layer(s), {len(loss_config_grid)} loss configs")

    def train_fn(point):
        r = train_one_config(
            model, tokenizer, layer_indices, seed, n_layers, model.config.hidden_size, device,
            train_items, dev_items, tr, dv, adapter.CACHE_DIR,
            mse_weight=point["mse_weight"], nll_weight=point["nll_weight"])
        suffix = "_mse" if point["mse_weight"] else "_nll"
        _save(adapter.RESULTS_DIR / f"{tag}_probe{suffix}.pt", r, layer_indices, task,
              point["mse_weight"], point["nll_weight"])
        print(f"{tag} mse_w={point['mse_weight']} nll_w={point['nll_weight']} "
              f"final_mse={r['final_mse']:.4f} final_nll={r['final_nll']:.4f}")
        return {k: v for k, v in r.items() if k not in ("gates", "directions", "targets")}

    grid = [{"n_layers_hooked": len(layer_indices), **c} for c in loss_config_grid]
    out = adapter.RESULTS_DIR / f"{tag}_sweep.jsonl"
    results, _ = run_grid_sweep(grid, train_fn, out, key_fields=["mse_weight", "nll_weight"])
    print(f"\nwrote {len(results)} rows to {out}")
    print("checkpoints saved per loss config -- compare by JUDGED eval, not final_mse across objectives")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=TASK_CHOICES)
    ap.add_argument("--layers", type=str, default=None,
                     help="comma-separated. One layer = SG+Clamp; omit = MG+Clamp (all layers)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--loss-configs", type=str, default=None)
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")] if a.layers else None
    cfgs = json.loads(a.loss_configs) if a.loss_configs else None
    sweep(a.task, layers, cfgs, a.seed)
