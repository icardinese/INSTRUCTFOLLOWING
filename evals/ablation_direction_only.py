"""Ablation: strip the trained gate off ANY GateState-compatible checkpoint and apply its frozen
`direction` as plain constant additive steering instead -- h' = h + coeff * direction, same
coefficient at every response token, via steering.const.hooks.make_const_hook (the exact mechanism
the paper's "Constant Activation Addition" / Steer condition already uses). This directly answers
"does the vector alone carry the method's value, or does it need the token-position-adaptive gate
too" -- reuses the real injection mechanism rather than approximating it.

Works for ANY checkpoint with {"direction", "layer"} in GateState-compatible shape -- psr_proper_probe.pt,
psr_probe.pt (S-PSR), psr_conceptor_probe.pt, or a_psr_probe.pt at one chosen layer. This is
deliberately the SAME script for the two ablations discussed (PSR-Proper direction-only, and
Conceptor direction-only): the only thing that differs between them is --checkpoint.

`direction` is normalized to unit norm before sweeping coefficients (PSR-Proper's is a trained
leaf tensor with an arbitrary norm from gradient descent -- around 1.8 for the current checkpoint,
not the 1.0 every OTHER direction in this codebase already has by construction). Without this,
the coefficient grid below has no consistent "steering strength" meaning across checkpoints, and
isn't comparable to the Const method's own already-calibrated coefficients either.

Two stages, same discipline as evals/layer_hparam_search.py:
  Calibration (dev, small n): sweep COEFF_GRID, pick the lowest avg_tokens among coefficients
    whose real judged `coherent` rate stays above COHERENT_FLOOR -- a judged floor, not a
    heuristic degenerate-text regex, consistent with this project's "real judged evidence decides,
    not a proxy" stance elsewhere (see layer_hparam_search.py's own module docstring).
  Final (test, full n): the chosen coefficient, evaluated on BOTH the alone (base_prompt + steer)
    and combined (terse_prompt + steer, i.e. "Prompt+X") conditions, since the paper cares about
    both -- calibration itself is done on ALONE only (see CALIBRATE_ON below for why), matching
    the ablation's actual question ("does the vector alone move tokens, no textual instruction
    helping it"), not the combined condition's own separate optimum.
"""
import argparse
import json

import torch

from adapters.registry import get_adapter
from core.model_common import generate_response, load_model, token_count
from evals.bootstrap_analysis import bootstrap_ci, paired_bootstrap_diff
from evals.registry import get_eval_adapter
from steering.const.hooks import make_const_hook
from steering.hooks import steering_hook
from adapters.registry import TASK_CHOICES

COEFF_GRID = [6.0, 12.0, 20.0, 28.0, 36.0]  # same starting grid caveman-steer's Const calibration
# used for an already-unit-normalized direction on this model -- a reasonable prior, not a
# measured-for-this-checkpoint ceiling; widen it if the calibration picks an edge value.
COHERENT_FLOOR = 0.85  # mean judged `coherent` rate a coefficient must clear to be considered at
# all -- same spirit as caveman-steer's MAX_DEGENERATE_RATE=0.15 (1 - 0.15 = 0.85), but read off
# the real judge's `coherent` field instead of a heuristic regex-based degenerate-text check.
DEFAULT_CALIB_N = 20
DEFAULT_FINAL_N = 180


def load_direction_only(checkpoint_path, device: str) -> tuple[int, torch.Tensor, float]:
    """Returns (layer_idx, unit_direction, original_norm). Deliberately ignores weight/bias/
    coeff_bias entirely -- that IS the ablation. original_norm is returned purely for the log, so
    it's visible exactly how much re-normalizing changed the vector, not silently done."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    direction = ckpt["direction"].to(device)
    original_norm = direction.norm().item()
    unit_direction = direction / direction.norm()
    return ckpt["layer"], unit_direction, original_norm


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def evaluate_coeff(
    model, tokenizer, direction: torch.Tensor, layer_idx: int, coeff: float,
    rows: list[dict], items: list[dict], adapter, eval_adapter, alone_only: bool,
) -> dict:
    """Generates (and judges) the alone condition always; the combined (Prompt+X) condition only
    when alone_only is False -- calibration only ever needs the former, Final needs both."""
    alone_responses, combined_responses = [], []
    for item in items:
        with steering_hook(model, layer_idx, make_const_hook(direction, coeff)):
            alone_responses.append(generate_response(model, tokenizer, item["base_prompt"]))
        if not alone_only:
            with steering_hook(model, layer_idx, make_const_hook(direction, coeff)):
                combined_responses.append(generate_response(model, tokenizer, item["terse_prompt"]))

    primary_field = eval_adapter.SCORE_FIELDS[0]
    alone_scores = [eval_adapter.score_response(row, resp) for row, resp in zip(rows, alone_responses)]
    result = {
        "coeff": coeff,
        "alone_avg_tokens": _mean([token_count(tokenizer, r) for r in alone_responses]),
        "alone_correct": _mean([s[primary_field] for s in alone_scores]),
        "alone_coherent_rate": _mean([1.0 if s["coherent"] else 0.0 for s in alone_scores]),
        "alone_correctness_scores": [s[primary_field] for s in alone_scores],
    }
    if "conciseness" in eval_adapter.SCORE_FIELDS:
        result["alone_conciseness"] = _mean([s["conciseness"] for s in alone_scores])
    if not alone_only:
        combined_scores = [eval_adapter.score_response(row, resp) for row, resp in zip(rows, combined_responses)]
        result["combined_avg_tokens"] = _mean([token_count(tokenizer, r) for r in combined_responses])
        result["combined_correct"] = _mean([s[primary_field] for s in combined_scores])
        result["combined_coherent_rate"] = _mean([1.0 if s["coherent"] else 0.0 for s in combined_scores])
        result["combined_correctness_scores"] = [s[primary_field] for s in combined_scores]
        if "conciseness" in eval_adapter.SCORE_FIELDS:
            result["combined_conciseness"] = _mean([s["conciseness"] for s in combined_scores])
    return result


def run(task: str, checkpoint_name: str, coeff_grid: list[float], calib_n: int, final_n: int, seed: int = 42) -> None:
    adapter = get_adapter(task)
    eval_adapter = get_eval_adapter(task)
    device = "cuda"
    model, tokenizer = load_model(device, model_name=adapter.MODEL_NAME)

    layer_idx, direction, original_norm = load_direction_only(adapter.RESULTS_DIR / checkpoint_name, device)
    print(f"loaded {checkpoint_name}: layer={layer_idx}, ||direction||={original_norm:.4f} "
          f"(renormalized to 1.0 for this ablation)")

    dev_rows = adapter.load_rows("dev")[:calib_n]
    dev_items = adapter.to_items(tokenizer, dev_rows)

    print(f"\n== Calibration: sweeping {len(coeff_grid)} coefficients on dev (n={calib_n}), ALONE condition only ==")
    calib_results = []
    for coeff in coeff_grid:
        r = evaluate_coeff(model, tokenizer, direction, layer_idx, coeff, dev_rows, dev_items, adapter, eval_adapter, alone_only=True)
        print(f"  coeff={coeff:<6} alone_avg_tokens={r['alone_avg_tokens']:.1f}  "
              f"alone_correct={r['alone_correct']:.3f}  alone_coherent_rate={r['alone_coherent_rate']:.2f}")
        calib_results.append(r)

    eligible = [r for r in calib_results if r["alone_coherent_rate"] >= COHERENT_FLOOR]
    if not eligible:
        print(f"\nWARNING: no coefficient cleared the coherent-rate floor ({COHERENT_FLOOR}) -- "
              f"every candidate produced too much incoherent output. Widen COEFF_GRID downward "
              f"or inspect alone_responses manually before trusting any Final run here.")
        best_coeff = min(calib_results, key=lambda r: r["alone_avg_tokens"])["coeff"]
        print(f"Proceeding anyway with the lowest-avg_tokens coefficient ({best_coeff}) for visibility, "
              f"but treat its Final result as unreliable.")
    else:
        best_coeff = min(eligible, key=lambda r: r["alone_avg_tokens"])["coeff"]
        print(f"\nCalibration winner: coeff={best_coeff} (lowest alone_avg_tokens among candidates "
              f"clearing the {COHERENT_FLOOR} coherent-rate floor)")

    print(f"\n== Final: coeff={best_coeff}, test split (n={final_n}), BOTH alone and combined ==")
    test_rows = adapter.load_rows("test")[:final_n]
    test_items = adapter.to_items(tokenizer, test_rows)
    final = evaluate_coeff(model, tokenizer, direction, layer_idx, best_coeff, test_rows, test_items, adapter, eval_adapter, alone_only=False)

    alone_pt, alone_lo, alone_hi = bootstrap_ci(final["alone_correctness_scores"])
    combined_pt, combined_lo, combined_hi = bootstrap_ci(final["combined_correctness_scores"])
    print(f"\nAlone:    avg_tokens={final['alone_avg_tokens']:.1f}  "
          f"correct={alone_pt:.3f} [{alone_lo:.3f}, {alone_hi:.3f}]  coherent_rate={final['alone_coherent_rate']:.2f}")
    print(f"Combined: avg_tokens={final['combined_avg_tokens']:.1f}  "
          f"correct={combined_pt:.3f} [{combined_lo:.3f}, {combined_hi:.3f}]  coherent_rate={final['combined_coherent_rate']:.2f}")

    out_path = adapter.RESULTS_DIR / f"ablation_direction_only_{checkpoint_name.removesuffix('.pt')}.json"
    with out_path.open("w") as f:
        json.dump({
            "task": task, "checkpoint": checkpoint_name, "layer": layer_idx,
            "original_direction_norm": original_norm, "seed": seed,
            "calibration": calib_results, "chosen_coeff": best_coeff,
            "final": {k: v for k, v in final.items() if not k.endswith("_scores")},
            "final_correctness_scores": {
                "alone": final["alone_correctness_scores"], "combined": final["combined_correctness_scores"],
            },
        }, f, indent=2)
    print(f"\nwrote {out_path}")
    print(f"\nCompare this against the SAME checkpoint's real (gated) generation results to answer "
          f"the actual ablation question: does removing the gate hurt, help, or do nothing?")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=TASK_CHOICES)
    parser.add_argument("--checkpoint", required=True,
                         help="checkpoint filename under results/<task>/, e.g. psr_proper_probe.pt or psr_conceptor_probe.pt")
    parser.add_argument("--coeffs", type=str, default=None, help="comma-separated coefficient grid, e.g. '6,12,20,28,36'")
    parser.add_argument("--calib-n", type=int, default=DEFAULT_CALIB_N)
    parser.add_argument("--final-n", type=int, default=DEFAULT_FINAL_N)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    coeff_grid = [float(x) for x in args.coeffs.split(",")] if args.coeffs else COEFF_GRID
    run(args.task, args.checkpoint, coeff_grid, args.calib_n, args.final_n, args.seed)
