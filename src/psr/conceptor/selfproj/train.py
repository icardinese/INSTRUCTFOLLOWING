"""Self-projection conceptor PSR: correction(h) = coeff(h) * (C @ h - h) / scale. No external target.

Alpha is chosen ADAPTIVELY from the real eigenvalue spectrum (at fixed percentiles), not a guessed
absolute number -- mu_i = sigma_i / (sigma_i + alpha^-2) sits near 1 (C ~ identity, no real
filtering) whenever alpha^-2 << sigma_i for most eigenvalues, so a naively-guessed alpha can end up
never meaningfully differing from doing nothing. See steering/psr/conceptor/selfproj/logic.py's
compute_delta_scale docstring for what "too loose" looks like numerically. Because the eigenvalue
scale itself depends on the layer (residual-stream norms grow through the network), the adaptive
alpha grid is re-derived AT EACH layer sweep() visits, rather than sharing one fixed alpha grid
across layers the way conceptor/matrix/train.py's sweep does -- a raw alpha value that's meaningful
at layer 4 usually isn't at layer 20.

Like conceptor/matrix/train.py, this logs participation_ratio (steering/psr/conceptor/
rank_diagnostic.py) -- self-projection applies C fresh to every hidden state, so its effective
dimensionality is a real property of the method, not just an intermediate computation.
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
from steering.psr.conceptor.rank_diagnostic import participation_ratio
from steering.psr.conceptor.selfproj.logic import compute_delta_scale, forward_with_gate_hook
from steering.psr.data import load_or_pool_separate_poles
from steering.psr.gate import init_gate_state
from steering.psr.training_loop import train_gate

N_EPOCHS = 3
LR = float(os.environ.get("PSR_CONCEPTOR_SELFPROJ_LR", 1e-3))
WEIGHT_DECAY = 1e-4
REG_COEFF = float(os.environ.get("PSR_CONCEPTOR_SELFPROJ_REG_COEFF", 0.1))
# mse_weight/nll_weight are independent (see steering/psr/training_loop.py's module docstring).
MSE_WEIGHT = float(os.environ.get("PSR_CONCEPTOR_SELFPROJ_MSE_WEIGHT", 1.0))
NLL_WEIGHT = float(os.environ.get("PSR_CONCEPTOR_SELFPROJ_NLL_WEIGHT", 0.0))
PERCENTILES = [10, 30, 50, 70, 90]
DEFAULT_SWEEP_LAYERS = list(range(2, 27, 2))
# See src/psr/proper/train.py's DEFAULT_LOSS_CONFIG_GRID docstring for why these are paired
# (mse_weight, nll_weight) points rather than a full cartesian product of two independent grids.
DEFAULT_LOSS_CONFIG_GRID = [
    {"mse_weight": 1.0, "nll_weight": 0.0},   # pure MSE
    {"mse_weight": 1.0, "nll_weight": 0.01},  # MSE + light NLL blend
    {"mse_weight": 1.0, "nll_weight": 0.05},  # MSE + medium NLL blend
    {"mse_weight": 1.0, "nll_weight": 0.1},   # MSE + heavy NLL blend
    {"mse_weight": 0.0, "nll_weight": 1.0},   # pure NLL
]


def eigendecompose(pool: torch.Tensor):
    n, d = pool.shape
    r = (pool.T @ pool) / n
    eigvals, eigvecs = torch.linalg.eigh(r)
    order = torch.argsort(eigvals, descending=True)
    return eigvals[order], eigvecs[:, order], r


def adaptive_alpha_grid(eigvals: torch.Tensor, percentiles: list[int] = PERCENTILES) -> list[tuple[int, float]]:
    """alpha^-2 = eigenvalue at each percentile -> alpha = 1/sqrt(eigenvalue). Guarantees the grid
    spans from "barely filters anything" to "filters almost everything" regardless of the real
    data's scale, instead of guessing an absolute alpha value that might land nowhere meaningful."""
    d = eigvals.shape[0]
    grid = []
    for p in percentiles:
        idx = min(d - 1, int((100 - p) / 100 * d))
        eig_at_p = max(eigvals[idx].item(), 1e-6)
        grid.append((p, 1.0 / (eig_at_p ** 0.5)))
    return grid


def train_one_config(
    model, tokenizer, layer_idx: int, alpha: float, seed: int, n_layers: int, device: str,
    train_items: list[dict], dev_items: list[dict], train_responses: dict, dev_responses: dict,
    cache_dir, lr: float = LR, weight_decay: float = WEIGHT_DECAY, reg_coeff: float = REG_COEFF,
    n_epochs: int = N_EPOCHS, mse_weight: float = MSE_WEIGHT, nll_weight: float = NLL_WEIGHT,
    on_epoch_end=None,
) -> dict:
    """One (layer, alpha) point. Returns {"skipped": True, ...} without training anything if
    delta_scale is too small at this alpha (C indistinguishable from identity) -- same skip
    condition the original percentile loop used, now reusable from sweep() too."""
    set_seed(seed)
    base_pool, instr_pool = load_or_pool_separate_poles(model, tokenizer, train_items, layer_idx, train_responses, cache_dir)
    pool = torch.cat([base_pool, instr_pool], dim=0)
    hidden_size = pool.shape[1]
    identity = torch.eye(hidden_size, device=pool.device, dtype=pool.dtype)
    r = (pool.T @ pool) / pool.shape[0]
    conceptor = r @ torch.linalg.inv(r + (alpha ** -2) * identity)
    delta_scale = compute_delta_scale(conceptor, pool)
    pr = participation_ratio(conceptor)

    if delta_scale.item() < 1e-3:
        print(f"  SKIPPING layer={layer_idx} alpha={alpha:.4f}: delta_scale~0, C indistinguishable from identity")
        return {"skipped": True, "delta_scale": delta_scale.item(), "participation_ratio": pr}

    gate = init_gate_state(hidden_size, device)
    optimizer = torch.optim.Adam(gate.parameters(), lr=lr, weight_decay=weight_decay)
    forward_fn = lambda pair: forward_with_gate_hook(model, gate, conceptor, delta_scale, layer_idx, pair["full_base"], pair["n_resp"])

    wrapped_on_epoch_end = (lambda epoch, metrics: on_epoch_end(epoch, gate, conceptor, delta_scale, metrics)) if on_epoch_end else None
    baseline_metrics, final_metrics = train_gate(
        model, tokenizer, forward_fn, optimizer, layer_idx, n_layers,
        train_items, dev_items, train_responses, dev_responses, n_epochs, reg_coeff,
        mse_weight=mse_weight, nll_weight=nll_weight, on_epoch_end=wrapped_on_epoch_end,
    )
    return {
        "skipped": False,
        "baseline_mse": baseline_metrics["mse"], "baseline_nll": baseline_metrics["nll"],
        "final_mse": final_metrics["mse"], "final_nll": final_metrics["nll"],
        "delta_scale": delta_scale.item(), "participation_ratio": pr,
        "weight": gate.weight.detach().cpu(), "bias": gate.bias.detach().cpu(),
        "coeff_bias": gate.coeff_bias.detach().cpu(),
        "conceptor": conceptor.detach().cpu(), "delta_scale_tensor": delta_scale.detach().cpu(),
    }


def main(task: str, layer_idx: int | None, seed: int = 42) -> None:
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

    if layer_idx is None:
        with (adapter.RESULTS_DIR / "const_steer_config.json").open() as f:
            layer_idx = json.load(f)["layer"]

    train_responses = load_or_compute_responses(model, tokenizer, train_items, adapter.CACHE_DIR / "teacher_responses_train.json", generate_response)
    dev_responses = load_or_compute_responses(model, tokenizer, dev_items, adapter.CACHE_DIR / "teacher_responses_dev.json", generate_response)

    base_pool, instr_pool = load_or_pool_separate_poles(model, tokenizer, train_items, layer_idx, train_responses, adapter.CACHE_DIR)
    pool = torch.cat([base_pool, instr_pool], dim=0)
    eigvals, eigvecs, r = eigendecompose(pool)
    print(f"eigenvalue range: max={eigvals[0].item():.4f}, min={eigvals[-1].item():.4f}")

    grid = adaptive_alpha_grid(eigvals)
    out_path = adapter.RESULTS_DIR / "psr_conceptor_selfproj_results.jsonl"
    probe_path = adapter.RESULTS_DIR / "psr_conceptor_selfproj_probe.pt"
    results = []
    best = {"final_mse": float("inf")}

    for percentile, alpha in grid:
        print(f"\n=== percentile={percentile}, alpha={alpha:.4f} ===")
        result = train_one_config(
            model, tokenizer, layer_idx, alpha, seed, n_layers, device,
            train_items, dev_items, train_responses, dev_responses, adapter.CACHE_DIR,
        )
        row = {"task": task, "percentile": percentile, "alpha": alpha, "skipped": result["skipped"],
               "seed": seed, "lr": LR, "weight_decay": WEIGHT_DECAY, "reg_coeff": REG_COEFF,
               "mse_weight": MSE_WEIGHT, "nll_weight": NLL_WEIGHT,
               "participation_ratio": result.get("participation_ratio")}
        if not result["skipped"]:
            row.update({"baseline_mse": result["baseline_mse"], "final_mse": result["final_mse"],
                        "baseline_nll": result["baseline_nll"], "final_nll": result["final_nll"]})
            print(f"  baseline_dev_mse={result['baseline_mse']:.4f} final_dev_mse={result['final_mse']:.4f} "
                  f"participation_ratio={result['participation_ratio']:.3f}")
        results.append(row)
        with out_path.open("w") as f:
            for r_ in results:
                f.write(json.dumps(r_) + "\n")

        # Track the best alpha's trained state -- this is what src/generate.py actually loads.
        # Without this, the sweep only ever produced a metrics log, never anything generate.py
        # could use, no matter how good a given alpha's numbers were.
        if not result["skipped"] and result["final_mse"] < best["final_mse"]:
            best = {**result, "alpha": alpha, "percentile": percentile}

    print(f"\nwrote {len(results)} rows to {out_path}")

    if best["final_mse"] < float("inf"):
        torch.save({
            "weight": best["weight"], "bias": best["bias"], "coeff_bias": best["coeff_bias"],
            "conceptor": best["conceptor"], "delta_scale": best["delta_scale_tensor"],
            "participation_ratio": best["participation_ratio"],
            "layer": layer_idx, "task": task, "seed": seed, "lr": LR,
            "weight_decay": WEIGHT_DECAY, "reg_coeff": REG_COEFF,
            "mse_weight": MSE_WEIGHT, "nll_weight": NLL_WEIGHT,
        }, probe_path)
        print(f"wrote {probe_path} (best alpha={best['alpha']:.4f}, final_mse={best['final_mse']:.4f}, "
              f"participation_ratio={best['participation_ratio']:.3f})")
    else:
        print("WARNING: every alpha in the grid was skipped (delta_scale too small) -- "
              "no probe checkpoint written, generate.py will skip this condition")


def sweep(
    task: str, layers: list[int] | None = None, percentiles: list[int] | None = None,
    loss_config_grid: list[dict] | None = None, seed: int = 42,
) -> None:
    """Grid over (layer, alpha, loss-config) where, for EACH layer, alpha is re-derived from that
    layer's own eigenvalue spectrum at `percentiles` (see module docstring for why alpha isn't
    shared across layers here the way it is in conceptor/matrix/train.py's sweep). See
    src/psr/proper/train.py's sweep() docstring for the resumability/best-checkpoint tradeoff this
    shares (same core.sweep.run_grid_sweep call)."""
    layers = layers if layers is not None else DEFAULT_SWEEP_LAYERS
    percentiles = percentiles if percentiles is not None else PERCENTILES
    loss_config_grid = loss_config_grid if loss_config_grid is not None else DEFAULT_LOSS_CONFIG_GRID
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

    # alpha grid depends on the layer's own eigenvalue spectrum -- computed once per layer here,
    # not once per (layer, alpha, loss-config) grid point, since it only depends on layer.
    alpha_by_layer_and_percentile = {}
    for layer_idx in layers:
        base_pool, instr_pool = load_or_pool_separate_poles(model, tokenizer, train_items, layer_idx, train_responses, adapter.CACHE_DIR)
        eigvals, _, _ = eigendecompose(torch.cat([base_pool, instr_pool], dim=0))
        for percentile, alpha in adaptive_alpha_grid(eigvals, percentiles):
            alpha_by_layer_and_percentile[(layer_idx, percentile)] = alpha

    checkpoints_by_point = {}

    def train_fn(point: dict) -> dict:
        alpha = alpha_by_layer_and_percentile[(point["layer"], point["percentile"])]
        result = train_one_config(
            model, tokenizer, point["layer"], alpha, seed, n_layers, device,
            train_items, dev_items, train_responses, dev_responses, adapter.CACHE_DIR,
            mse_weight=point["mse_weight"], nll_weight=point["nll_weight"],
        )
        key = (point["layer"], point["percentile"], point["mse_weight"], point["nll_weight"])
        checkpoints_by_point[key] = result
        status = "SKIPPED" if result["skipped"] else f"final_mse={result['final_mse']:.4f}"
        print(f"layer={point['layer']} percentile={point['percentile']} (alpha={alpha:.4f}) "
              f"mse_weight={point['mse_weight']} nll_weight={point['nll_weight']}: {status} "
              f"participation_ratio={result.get('participation_ratio'):.3f}")
        out = {"alpha": alpha, "skipped": result["skipped"], "participation_ratio": result.get("participation_ratio")}
        if not result["skipped"]:
            out.update({"baseline_mse": result["baseline_mse"], "baseline_nll": result["baseline_nll"],
                        "final_mse": result["final_mse"], "final_nll": result["final_nll"]})
        return out

    grid = [{"layer": l, "percentile": p, **loss_cfg} for l in layers for p in percentiles for loss_cfg in loss_config_grid]
    out_path = adapter.RESULTS_DIR / "psr_conceptor_selfproj_sweep.jsonl"
    results, best = run_grid_sweep(grid, train_fn, out_path, key_fields=["layer", "percentile", "mse_weight", "nll_weight"])
    print(f"\nwrote {len(results)} sweep rows to {out_path}")

    if best is None:
        print("WARNING: sweep produced no usable (non-skipped) points -- no probe checkpoint written")
        return
    best_key = (best["layer"], best["percentile"], best["mse_weight"], best["nll_weight"])
    if best_key not in checkpoints_by_point:
        print(f"NOTE: best point {best_key} was already completed in a previous session -- its "
              f"trained tensors aren't in memory this run. Delete its row from {out_path} and "
              f"rerun sweep() to regenerate a checkpoint for it.")
        return
    winner = checkpoints_by_point[best_key]
    out_probe_path = adapter.RESULTS_DIR / "psr_conceptor_selfproj_probe.pt"
    torch.save({
        "weight": winner["weight"], "bias": winner["bias"], "coeff_bias": winner["coeff_bias"],
        "conceptor": winner["conceptor"], "delta_scale": winner["delta_scale_tensor"],
        "participation_ratio": winner["participation_ratio"],
        "layer": best["layer"], "alpha": best["alpha"],
        "mse_weight": best["mse_weight"], "nll_weight": best["nll_weight"],
        "task": task, "seed": seed,
    }, out_probe_path)
    print(f"wrote {out_probe_path} (best layer={best['layer']}, alpha={best['alpha']:.4f}, "
          f"mse_weight={best['mse_weight']}, nll_weight={best['nll_weight']}, final_mse={best['final_mse']:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["caveman", "ifeval"])
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sweep", action="store_true", help="sweep layer x adaptive-alpha x loss-config instead of a single-layer alpha sweep")
    parser.add_argument("--sweep-layers", type=str, default=None)
    parser.add_argument("--loss-configs", type=str, default=None,
                         help='comma-separated "mse_weight:nll_weight" pairs, e.g. "1.0:0.0,0.0:1.0"')
    args = parser.parse_args()
    if args.sweep:
        layers = [int(x) for x in args.sweep_layers.split(",")] if args.sweep_layers else None
        loss_configs = None
        if args.loss_configs:
            loss_configs = []
            for pair in args.loss_configs.split(","):
                mse_w, nll_w = pair.split(":")
                loss_configs.append({"mse_weight": float(mse_w), "nll_weight": float(nll_w)})
        sweep(args.task, layers, None, loss_configs, args.seed)
    else:
        main(args.task, args.layer, args.seed)
