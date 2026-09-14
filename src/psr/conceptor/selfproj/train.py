"""Self-projection conceptor PSR: correction(h) = coeff(h) * (C @ h - h) / scale. No external target.

Alpha is chosen ADAPTIVELY from the real eigenvalue spectrum (at fixed percentiles), not a guessed
absolute number -- mu_i = sigma_i / (sigma_i + alpha^-2) sits near 1 (C ~ identity, no real
filtering) whenever alpha^-2 << sigma_i for most eigenvalues, so a naively-guessed alpha can end up
never meaningfully differing from doing nothing. See steering/psr/conceptor/selfproj/logic.py's
compute_delta_scale docstring for what "too loose" looks like numerically.
"""
import argparse
import json
import os

import torch

from adapters.registry import get_adapter
from core.reproducibility import set_seed
from core.generation_cache import load_or_compute_responses
from core.model_common import generate_response, load_model, num_layers
from steering.psr.conceptor.selfproj.logic import compute_delta_scale, forward_with_gate_hook
from steering.psr.data import load_or_pool_separate_poles
from steering.psr.gate import init_gate_state
from steering.psr.training_loop import train_gate

N_EPOCHS = 3
LR = float(os.environ.get("PSR_CONCEPTOR_SELFPROJ_LR", 1e-3))
WEIGHT_DECAY = 1e-4
REG_COEFF = float(os.environ.get("PSR_CONCEPTOR_SELFPROJ_REG_COEFF", 0.1))
PERCENTILES = [10, 30, 50, 70, 90]


def eigendecompose(pool: torch.Tensor):
    n, d = pool.shape
    r = (pool.T @ pool) / n
    eigvals, eigvecs = torch.linalg.eigh(r)
    order = torch.argsort(eigvals, descending=True)
    return eigvals[order], eigvecs[:, order], r


def adaptive_alpha_grid(eigvals: torch.Tensor) -> list[tuple[int, float]]:
    """alpha^-2 = eigenvalue at each percentile -> alpha = 1/sqrt(eigenvalue). Guarantees the grid
    spans from "barely filters anything" to "filters almost everything" regardless of the real
    data's scale, instead of guessing an absolute alpha value that might land nowhere meaningful."""
    d = eigvals.shape[0]
    grid = []
    for p in PERCENTILES:
        idx = min(d - 1, int((100 - p) / 100 * d))
        eig_at_p = max(eigvals[idx].item(), 1e-6)
        grid.append((p, 1.0 / (eig_at_p ** 0.5)))
    return grid


def main(task: str, layer_idx: int | None, seed: int = 42) -> None:
    set_seed(seed)
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
    hidden_size = pool.shape[1]
    d = hidden_size
    results = []
    out_path = adapter.RESULTS_DIR / "psr_conceptor_selfproj_results.jsonl"
    probe_path = adapter.RESULTS_DIR / "psr_conceptor_selfproj_probe.pt"
    best = {"final_mse": float("inf")}

    for percentile, alpha in grid:
        identity = torch.eye(d, device=device, dtype=pool.dtype)
        conceptor = r @ torch.linalg.inv(r + (alpha ** -2) * identity)
        delta_scale = compute_delta_scale(conceptor, pool)
        mean_mu = (eigvals / (eigvals + alpha ** -2)).mean().item()
        print(f"\n=== percentile={percentile}, alpha={alpha:.4f}, mean(mu)={mean_mu:.4f}, delta_scale={delta_scale.item():.4f} ===")

        if delta_scale.item() < 1e-3:
            print("  SKIPPING: delta_scale ~0, C is indistinguishable from identity at this alpha")
            results.append({"task": task, "percentile": percentile, "alpha": alpha, "mean_mu": mean_mu, "skipped": True})
            continue

        gate = init_gate_state(hidden_size, device)
        optimizer = torch.optim.Adam(gate.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        forward_fn = lambda pair: forward_with_gate_hook(model, gate, conceptor, delta_scale, layer_idx, pair["full_base"], pair["n_resp"])

        baseline_mse, final_mse = train_gate(
            model, tokenizer, forward_fn, optimizer, layer_idx, n_layers,
            train_items, dev_items, train_responses, dev_responses, N_EPOCHS, REG_COEFF,
        )
        print(f"  baseline_dev_mse={baseline_mse:.4f} final_dev_mse={final_mse:.4f}")
        results.append({
            "task": task, "percentile": percentile, "alpha": alpha, "mean_mu": mean_mu, "skipped": False,
            "baseline_mse": baseline_mse, "final_mse": final_mse,
            "seed": seed, "lr": LR, "weight_decay": WEIGHT_DECAY, "reg_coeff": REG_COEFF,
        })
        with out_path.open("w") as f:
            for row in results:
                f.write(json.dumps(row) + "\n")

        # Track the best alpha's trained state -- this is what generate.py's src/generate.py
        # actually loads. Without this, the sweep only ever produced a metrics log, never
        # anything generate.py could use, no matter how good a given alpha's numbers were.
        if final_mse < best["final_mse"]:
            best = {
                "final_mse": final_mse, "alpha": alpha, "percentile": percentile,
                "weight": gate.weight.detach().cpu(), "bias": gate.bias.detach().cpu(),
                "coeff_bias": gate.coeff_bias.detach().cpu(),
                "conceptor": conceptor.detach().cpu(), "delta_scale": delta_scale.detach().cpu(),
            }

    print(f"\nwrote {len(results)} rows to {out_path}")

    if best["final_mse"] < float("inf"):
        torch.save({**best, "layer": layer_idx, "task": task, "seed": seed, "lr": LR,
                    "weight_decay": WEIGHT_DECAY, "reg_coeff": REG_COEFF}, probe_path)
        print(f"wrote {probe_path} (best alpha={best['alpha']:.4f}, final_mse={best['final_mse']:.4f})")
    else:
        print("WARNING: every alpha in the grid was skipped (delta_scale too small) -- "
              f"no probe checkpoint written, generate.py will skip this condition")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["caveman", "ifeval"])
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.task, args.layer, args.seed)
