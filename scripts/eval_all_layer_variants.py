"""Judged evaluation of every all-layer checkpoint: A-PSR (trained directions) and Multi-Gate
(fixed diff-in-means), each under pure MSE and pure NLL -- 4 configs -- plus Base and Prompt
baselines, in both the alone (steering only) and combined (Prompt+Steer) conditions.

Why this is a standalone script rather than a --variant in evals/layer_hparam_search.py: that
module's entire structure is layer/hyperparameter SELECTION (Tier 1 narrows layers, Tier 2 narrows
hyperparameters, Final confirms one winner). A-PSR has no layer hyperparameter to select -- "all
layers" is definitional -- and only 2 loss configs, both of which are real experiments H&V report
separately rather than candidates to prune. So there is nothing to search: just load each
checkpoint and evaluate it at full n. Forcing that through the tiered-search machinery would mean
inventing a selection problem that doesn't exist.

Baselines are generated ONCE on the same rows and reused across all 4 configs (they don't depend
on any checkpoint), so this is 4 steered conditions x 2 prompt types + 2 baselines, all paired to
the same examples -- valid for the paired bootstrap comparisons at the bottom.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters.registry import get_adapter
from core.model_common import load_model, token_count
from evals.bootstrap_analysis import bootstrap_ci, paired_bootstrap_diff
from evals.layer_hparam_search import _score_all_concurrently
from evals.registry import get_eval_adapter
from steering.batch_routing import generate_batched_uniform
from steering.psr.gate import GateState
from steering.psr.multi_gate import make_multi_inference_hooks
from adapters.registry import TASK_CHOICES

DEFAULT_N = 180
MAX_BATCH_ROWS = 60  # same budget evals/layer_hparam_search.py uses -- an unbounded batch hit a
# real CUDA OOM on an 80GB A100. Lower it if a bigger model OOMs here.

# (checkpoint stem, human label). Both variants, both objectives.
CONFIGS = [
    ("psr_all_layer_probe_mse", "A-PSR (MSE)"),
    ("psr_all_layer_probe_nll", "A-PSR (NLL)"),
    ("psr_multi_gate_probe_mse", "Multi-Gate (MSE)"),
    ("psr_multi_gate_probe_nll", "Multi-Gate (NLL)"),
    ("psr_multi_gate_conceptor_probe_mse", "MG-Conceptor (MSE)"),
    ("psr_multi_gate_conceptor_probe_nll", "MG-Conceptor (NLL)"),
]


def _mean(values) -> float:
    return sum(values) / len(values)


def load_multi_checkpoint(path: Path, device: str):
    """Returns (gates, directions, layer_indices, meta) from an all-layer checkpoint written by
    src/psr/all_layer/train.py's _save_checkpoint."""
    ckpt = torch.load(path, map_location=device)
    layer_indices = ckpt["layer_indices"]
    gates = {
        l: GateState(
            weight=ckpt["gates"][l]["weight"].to(device),
            bias=ckpt["gates"][l]["bias"].to(device),
            coeff_bias=ckpt["gates"][l]["coeff_bias"].to(device),
        )
        for l in layer_indices
    }
    directions = {l: ckpt["directions"][l].to(device) for l in layer_indices}
    meta = {
        "direction_source": ckpt.get("direction_source"),
        "mse_weight": ckpt.get("mse_weight"), "nll_weight": ckpt.get("nll_weight"),
        "n_layers_hooked": len(layer_indices),
    }
    return gates, directions, layer_indices, meta


def run(task: str, n: int, device: str = "cuda") -> None:
    adapter = get_adapter(task)
    eval_adapter = get_eval_adapter(task)
    primary_field = eval_adapter.SCORE_FIELDS[0]

    available = [(stem, label) for stem, label in CONFIGS if (adapter.RESULTS_DIR / f"{stem}.pt").exists()]
    if not available:
        raise FileNotFoundError(
            f"no all-layer checkpoints found in {adapter.RESULTS_DIR}. Run:\n"
            f"  python3 src/psr/all_layer/train.py --task {task} --direction-source trained --sweep\n"
            f"  python3 src/psr/all_layer/train.py --task {task} --direction-source diff_in_means --sweep\n"
            f"  python3 src/psr/all_layer/train.py --task {task} --direction-source conceptor --sweep"
        )
    available_stems = {stem for stem, _ in available}
    missing = [stem for stem, _ in CONFIGS if stem not in available_stems]
    if missing:
        print(f"NOTE: evaluating {len(available)}/{len(CONFIGS)} configs -- missing checkpoints: {missing}")

    model, tokenizer = load_model(device, model_name=adapter.MODEL_NAME)
    for p in model.parameters():
        p.requires_grad_(False)

    test_rows = adapter.load_rows("test")[:n]
    test_items = adapter.to_items(tokenizer, test_rows)
    base_prompts = [it["base_prompt"] for it in test_items]
    terse_prompts = [it["terse_prompt"] for it in test_items]

    print(f"\ngenerating Base and Prompt baselines once (n={n}, shared across all configs) ...")
    with torch.no_grad():
        base_responses = generate_batched_uniform(model, tokenizer, base_prompts, hooks=None, max_batch_rows=MAX_BATCH_ROWS)
        prompt_responses = generate_batched_uniform(model, tokenizer, terse_prompts, hooks=None, max_batch_rows=MAX_BATCH_ROWS)

    def score(responses):
        # _score_all_concurrently, NOT a list comprehension: score_response is 2 blocking OpenAI
        # calls per response, and this script scores 10 conditions x n responses. Sequentially that
        # is the dominant cost of the whole run (the exact bottleneck that made Const's Tier 1 take
        # ~3h before evals/layer_hparam_search.py got this same fix).
        dicts = _score_all_concurrently(eval_adapter, test_rows, responses)
        return {
            "scores": [d[primary_field] for d in dicts],
            "coherent_rate": _mean([1.0 if d["coherent"] else 0.0 for d in dicts]),
            "avg_tokens": _mean([token_count(tokenizer, r) for r in responses]),
        }

    results = {"base": score(base_responses), "prompt": score(prompt_responses)}

    for stem, label in available:
        print(f"\n== {label} ==")
        gates, directions, layer_indices, meta = load_multi_checkpoint(adapter.RESULTS_DIR / f"{stem}.pt", device)
        print(f"   {meta['n_layers_hooked']} layers hooked, direction_source={meta['direction_source']}")
        hooks = make_multi_inference_hooks(gates, directions, layer_indices)
        with torch.no_grad():
            alone = generate_batched_uniform(model, tokenizer, base_prompts, hooks=hooks, max_batch_rows=MAX_BATCH_ROWS)
            combined = generate_batched_uniform(model, tokenizer, terse_prompts, hooks=hooks, max_batch_rows=MAX_BATCH_ROWS)
        results[f"{stem}_alone"] = {**score(alone), "label": f"{label} alone", "meta": meta}
        results[f"{stem}_combined"] = {**score(combined), "label": f"Prompt + {label}", "meta": meta}

    print(f"\n{'condition':<30}{'avg_tokens':>12}{'correct':>10}{'coherent':>11}")
    print("-" * 63)
    for key in ("base", "prompt"):
        pt, _, _ = bootstrap_ci(results[key]["scores"])
        print(f"{key:<30}{results[key]['avg_tokens']:>12.1f}{pt:>10.3f}{results[key]['coherent_rate']:>11.2f}")
    for stem, _label in available:
        for suffix in ("alone", "combined"):
            r = results[f"{stem}_{suffix}"]
            pt, _, _ = bootstrap_ci(r["scores"])
            print(f"{r['label']:<30}{r['avg_tokens']:>12.1f}{pt:>10.3f}{r['coherent_rate']:>11.2f}")

    print(f"\n== paired vs Prompt alone (n={n}) ==")
    for stem, _label in available:
        for suffix in ("alone", "combined"):
            r = results[f"{stem}_{suffix}"]
            # Candidate FIRST, prompt SECOND. paired_bootstrap_diff returns mean(a) - mean(b), so
            # this orientation makes a POSITIVE diff mean "candidate scored higher than Prompt",
            # which is what the verdict strings below claim. Reversing these two arguments silently
            # inverts every verdict while leaving the magnitudes correct -- which is exactly the bug
            # this line had on 2026-09-18, and it is invisible unless you cross-check the diff
            # against the two means by hand. evals/layer_hparam_search.py uses this same order.
            diff, lo, hi = paired_bootstrap_diff(r["scores"], results["prompt"]["scores"])
            verdict = "BEATS Prompt" if lo > 0 else "WORSE than Prompt" if hi < 0 else "ties Prompt"
            print(f"  {r['label']:<30} correctness diff={diff:+.4f} [{lo:+.4f}, {hi:+.4f}] -- {verdict}"
                  f"   (tokens {r['avg_tokens']:.1f} vs {results['prompt']['avg_tokens']:.1f})")

    out_path = adapter.RESULTS_DIR / "all_layer_variants_eval.json"
    with out_path.open("w") as f:
        json.dump({"task": task, "n": n, "results": results}, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="caveman", choices=TASK_CHOICES)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    args = parser.parse_args()
    run(args.task, args.n)
