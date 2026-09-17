"""Matrix-application conceptor PSR: correction(h) = coeff(h) * (C @ (mu_instr - h)) / scale. See
steering/psr/conceptor/matrix/logic.py for the formula itself -- this script wires it to real data
and a real optimizer, and (unlike the fixed-vector variant in conceptor/train.py) logs the
conceptor's participation ratio -- see steering/psr/conceptor/rank_diagnostic.py -- since C is
genuinely applied fresh at every position here, so its effective dimensionality is a real property
of the method, not just an intermediate computation. Logged in TWO places: this file's checkpoint
and train_log.json (so it's visible right after training, alongside every other hyperparameter),
and again by src/generate.py (attached to every generated row for this condition, for downstream
analysis/plotting -- see that file's load_matrix_condition).
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
from steering.psr.conceptor.direction import compute_conceptor
from steering.psr.conceptor.matrix.logic import compute_delta_scale, forward_with_gate_hook
from steering.psr.conceptor.rank_diagnostic import participation_ratio
from steering.psr.data import load_or_pool_separate_poles
from steering.psr.gate import init_gate_state
from steering.psr.training_loop import train_gate

N_EPOCHS = 3
LR = float(os.environ.get("PSR_CONCEPTOR_MATRIX_LR", 1e-3))
WEIGHT_DECAY = 1e-4
REG_COEFF = float(os.environ.get("PSR_CONCEPTOR_MATRIX_REG_COEFF", 0.1))
ALPHA = float(os.environ.get("PSR_CONCEPTOR_MATRIX_ALPHA", 4.0))
OUT_TAG = os.environ.get("PSR_CONCEPTOR_MATRIX_OUT_TAG", "")
# mse_weight/nll_weight are independent (see steering/psr/training_loop.py's module docstring).
MSE_WEIGHT = float(os.environ.get("PSR_CONCEPTOR_MATRIX_MSE_WEIGHT", 1.0))
NLL_WEIGHT = float(os.environ.get("PSR_CONCEPTOR_MATRIX_NLL_WEIGHT", 0.0))
DEFAULT_SWEEP_LAYERS = list(range(2, 27, 2))
DEFAULT_ALPHA_GRID = [1.0, 2.0, 4.0, 8.0, 16.0]
# See src/psr/proper/train.py's DEFAULT_LOSS_CONFIG_GRID docstring for why these are paired
# (mse_weight, nll_weight) points rather than a full cartesian product of two independent grids.
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
    model, tokenizer, layer_idx: int, alpha: float, seed: int, n_layers: int, device: str,
    train_items: list[dict], dev_items: list[dict], train_responses: dict, dev_responses: dict,
    cache_dir, lr: float = LR, weight_decay: float = WEIGHT_DECAY, reg_coeff: float = REG_COEFF,
    n_epochs: int = N_EPOCHS, mse_weight: float = MSE_WEIGHT, nll_weight: float = NLL_WEIGHT,
    on_epoch_end=None,
) -> dict:
    set_seed(seed)
    base_pool, instr_pool = load_or_pool_separate_poles(model, tokenizer, train_items, layer_idx, train_responses, cache_dir)
    # See src/psr/conceptor/train.py's train_one_config for why this .to(device) is necessary --
    # pool_separate_poles deliberately returns CPU tensors, and everything derived from them
    # (conceptor, mu_instr, delta_scale) needs to be on the live model's device before being
    # combined with real (CUDA) hidden states inside the hook.
    base_pool, instr_pool = base_pool.to(device), instr_pool.to(device)
    conceptor = compute_conceptor(torch.cat([base_pool, instr_pool], dim=0), alpha=alpha)
    mu_instr = instr_pool.mean(dim=0)
    delta_scale = compute_delta_scale(conceptor, mu_instr, base_pool)
    pr = participation_ratio(conceptor)
    print(f"layer={layer_idx} alpha={alpha} delta_scale={delta_scale.item():.2f} participation_ratio={pr:.3f}")

    hidden_size = base_pool.shape[1]
    gate = init_gate_state(hidden_size, device)
    optimizer = torch.optim.Adam(gate.parameters(), lr=lr, weight_decay=weight_decay)
    forward_fn = lambda pair: forward_with_gate_hook(model, gate, conceptor, mu_instr, delta_scale, layer_idx, pair["full_base"], pair["n_resp"])

    wrapped_on_epoch_end = (lambda epoch, metrics: on_epoch_end(epoch, gate, conceptor, mu_instr, delta_scale, metrics)) if on_epoch_end else None
    baseline_metrics, final_metrics = train_gate(
        model, tokenizer, forward_fn, optimizer, layer_idx, n_layers,
        train_items, dev_items, train_responses, dev_responses, n_epochs, reg_coeff,
        mse_weight=mse_weight, nll_weight=nll_weight, on_epoch_end=wrapped_on_epoch_end,
    )
    return {
        "baseline_mse": baseline_metrics["mse"], "baseline_nll": baseline_metrics["nll"],
        "final_mse": final_metrics["mse"], "final_nll": final_metrics["nll"],
        "participation_ratio": pr,
        "weight": gate.weight.detach().cpu(), "bias": gate.bias.detach().cpu(),
        "coeff_bias": gate.coeff_bias.detach().cpu(), "conceptor": conceptor.detach().cpu(),
        "mu_instr": mu_instr.detach().cpu(), "delta_scale": delta_scale.detach().cpu(),
    }


def main(task: str, layer_idx: int | None, seed: int = 42) -> None:
    adapter = get_adapter(task)
    out_path = adapter.RESULTS_DIR / f"psr_conceptor_matrix_probe{OUT_TAG}.pt"
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

    def write_checkpoint_and_log(gate_weight, gate_bias, gate_coeff_bias, conceptor, mu_instr, delta_scale, pr, baseline: dict, final: dict, completed_epochs) -> None:
        torch.save({
            "weight": gate_weight, "bias": gate_bias, "coeff_bias": gate_coeff_bias,
            "conceptor": conceptor, "mu_instr": mu_instr, "delta_scale": delta_scale,
            "layer": layer_idx, "alpha": ALPHA, "task": task, "participation_ratio": pr,
            "completed_epochs": N_EPOCHS if completed_epochs is None else completed_epochs,
        }, out_path)
        with (adapter.RESULTS_DIR / f"psr_conceptor_matrix_train_log{OUT_TAG}.json").open("w") as f:
            json.dump({
                "task": task, "layer": layer_idx, "alpha": ALPHA, "seed": seed, "lr": LR,
                "weight_decay": WEIGHT_DECAY, "reg_coeff": REG_COEFF,
                "mse_weight": MSE_WEIGHT, "nll_weight": NLL_WEIGHT,
                "completed_epochs": N_EPOCHS if completed_epochs is None else completed_epochs,
                "baseline_dev_mse": baseline["mse"], "baseline_dev_nll": baseline["nll"],
                "dev_mse": final["mse"], "dev_nll": final["nll"], "participation_ratio": pr,
            }, f, indent=2)

    def on_epoch_end(epoch, gate, conceptor, mu_instr, delta_scale, dev_metrics: dict) -> None:
        print(f"epoch={epoch} dev_mse={dev_metrics['mse']:.4f} dev_nll={dev_metrics['nll']:.4f} -- checkpointing")
        write_checkpoint_and_log(
            gate.weight.detach().cpu(), gate.bias.detach().cpu(), gate.coeff_bias.detach().cpu(),
            conceptor.detach().cpu(), mu_instr.detach().cpu(), delta_scale.detach().cpu(),
            participation_ratio(conceptor), dev_metrics, dev_metrics, epoch,
        )

    result = train_one_config(
        model, tokenizer, layer_idx, ALPHA, seed, n_layers, device,
        train_items, dev_items, train_responses, dev_responses, adapter.CACHE_DIR,
        on_epoch_end=on_epoch_end,
    )
    print(f"baseline_dev_mse={result['baseline_mse']:.4f} final_dev_mse={result['final_mse']:.4f} "
          f"participation_ratio={result['participation_ratio']:.3f}")
    write_checkpoint_and_log(
        result["weight"], result["bias"], result["coeff_bias"], result["conceptor"],
        result["mu_instr"], result["delta_scale"], result["participation_ratio"],
        {"mse": result["baseline_mse"], "nll": result["baseline_nll"]},
        {"mse": result["final_mse"], "nll": result["final_nll"]}, None,
    )


def sweep(
    task: str, layers: list[int] | None = None, alpha_grid: list[float] | None = None,
    loss_config_grid: list[dict] | None = None, seed: int = 42, device: str = "cuda",
) -> None:
    """Grid over (layer, alpha, loss-config). Every row in the resulting sweep JSONL carries
    participation_ratio alongside final_mse -- this is the cheap, dense dataset the project wants
    for "does accuracy [proxied here by dev MSE, no generation/judging needed] increase or
    decrease as the conceptor's effective dimensionality changes", see evals/plotting.py's
    plot_participation_ratio_vs_metric, which reads exactly this file."""
    layers = layers if layers is not None else DEFAULT_SWEEP_LAYERS
    alpha_grid = alpha_grid if alpha_grid is not None else DEFAULT_ALPHA_GRID
    loss_config_grid = loss_config_grid if loss_config_grid is not None else DEFAULT_LOSS_CONFIG_GRID
    adapter = get_adapter(task)
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
            mse_weight=point["mse_weight"], nll_weight=point["nll_weight"],
        )
        key = (point["layer"], point["alpha"], point["mse_weight"], point["nll_weight"])
        checkpoints_by_point[key] = result
        print(f"layer={point['layer']} alpha={point['alpha']} mse_weight={point['mse_weight']} "
              f"nll_weight={point['nll_weight']} final_mse={result['final_mse']:.4f} "
              f"participation_ratio={result['participation_ratio']:.3f}")
        return {k: v for k, v in result.items()
                if k not in ("weight", "bias", "coeff_bias", "conceptor", "mu_instr", "delta_scale")}

    grid = [{"layer": l, "alpha": a, **loss_cfg} for l in layers for a in alpha_grid for loss_cfg in loss_config_grid]
    out_path = adapter.RESULTS_DIR / "psr_conceptor_matrix_sweep.jsonl"
    results, best = run_grid_sweep(grid, train_fn, out_path, key_fields=["layer", "alpha", "mse_weight", "nll_weight"])
    print(f"\nwrote {len(results)} sweep rows to {out_path}")

    if best is None:
        print("WARNING: sweep produced no usable points -- no probe checkpoint written")
        return
    best_key = (best["layer"], best["alpha"], best["mse_weight"], best["nll_weight"])
    if best_key in checkpoints_by_point:
        winner = checkpoints_by_point[best_key]
    else:
        # The overall winner was completed in an EARLIER session (this run only retrained
        # whatever was still missing on resume) -- rather than bailing with nothing saved, just
        # retrain this ONE point again to get its tensors. Cheap: pooled activations are cached,
        # so this costs one training run, not a repool. Without this, a sweep that's 99% resumed
        # (like tonight's) would finish, correctly identify its own winner, and then throw that
        # information away instead of producing the checkpoint generate.py actually needs.
        print(f"Best point {best_key} was completed in an earlier session -- retraining it once "
              f"more to obtain its checkpoint tensors (cheap: pooling is cached).")
        winner = train_one_config(
            model, tokenizer, best["layer"], best["alpha"], seed, n_layers, device,
            train_items, dev_items, train_responses, dev_responses, adapter.CACHE_DIR,
            mse_weight=best["mse_weight"], nll_weight=best["nll_weight"],
        )
    out_probe_path = adapter.RESULTS_DIR / "psr_conceptor_matrix_probe.pt"
    torch.save({
        "weight": winner["weight"], "bias": winner["bias"], "coeff_bias": winner["coeff_bias"],
        "conceptor": winner["conceptor"], "mu_instr": winner["mu_instr"], "delta_scale": winner["delta_scale"],
        "layer": best["layer"], "alpha": best["alpha"],
        "mse_weight": best["mse_weight"], "nll_weight": best["nll_weight"],
        "participation_ratio": winner["participation_ratio"], "task": task, "completed_epochs": N_EPOCHS,
    }, out_probe_path)
    print(f"wrote {out_probe_path} (best layer={best['layer']}, alpha={best['alpha']}, "
          f"mse_weight={best['mse_weight']}, nll_weight={best['nll_weight']}, final_mse={best['final_mse']:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["caveman", "ifeval"])
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--sweep-layers", type=str, default=None)
    parser.add_argument("--sweep-alphas", type=str, default=None)
    parser.add_argument("--loss-configs", type=str, default=None,
                         help='comma-separated "mse_weight:nll_weight" pairs, e.g. "1.0:0.0,0.0:1.0"')
    args = parser.parse_args()
    if args.sweep:
        layers = [int(x) for x in args.sweep_layers.split(",")] if args.sweep_layers else None
        alphas = [float(x) for x in args.sweep_alphas.split(",")] if args.sweep_alphas else None
        loss_configs = None
        if args.loss_configs:
            loss_configs = []
            for pair in args.loss_configs.split(","):
                mse_w, nll_w = pair.split(":")
                loss_configs.append({"mse_weight": float(mse_w), "nll_weight": float(nll_w)})
        sweep(args.task, layers, alphas, loss_configs, args.seed)
    else:
        main(args.task, args.layer, args.seed)
