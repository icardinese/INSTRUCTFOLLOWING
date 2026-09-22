"""Recover per-response token counts (and therefore CIs) for the winning configurations.

ZERO API CALLS. Token counting is the local tokenizer; nothing here is judged.

WHY A REGENERATION IS NEEDED AT ALL. Every producing script stored only the MEAN:
evals/layer_hparam_search.py's _avg_tokens returns sum(...)/len(...), run_prompt_plus.py saves
prompt_plus_steer_avg_tokens, eval_all_layer_variants.py's score() returns avg_tokens. A mean
cannot be turned back into a distribution, so the per-response counts have to be produced again.
They are worth having: for SG+Clamp alone the token sd is ~30 on a mean of ~44, so a CI-free mean
implies far more precision than the data supports.

WHY CORRECTNESS DOES NOT NEED RE-JUDGING. Generation is greedy (do_sample=False in
generate_batched_uniform), so the same hook on the same rows is deterministic and reproduces the
same text -- meaning the already-stored correctness_scores still describe these responses. That is
an assumption about batching though, not a guarantee: batched generation left-pads, so a different
batch composition can perturb results slightly. So this script VERIFIES it instead of asserting
it, by comparing the recomputed mean against the stored avg_tokens and reporting the drift per
condition. Small drift (< ~0.5 tokens) means the responses are effectively identical and the
stored correctness applies; large drift means that condition genuinely needs re-judging, and it
says so explicitly rather than leaving you to guess.

WINNERS ONLY. Each variant is regenerated at the exact (layer, coefficient, loss-config) its own
tiered search or sweep selected -- read from disk, never hardcoded. No grid is re-explored.
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters.registry import get_adapter
from core.generation_cache import load_or_compute_responses
from core.model_common import generate_response, load_model, num_layers, token_count, generate_response_with_meta
from evals.layer_hparam_search import (
    DEFAULT_MAX_BATCH_ROWS, RETRAIN_FNS, SearchContext, load_existing_tiered_results,
)
from steering.batch_routing import generate_batched_uniform
from adapters.registry import TASK_CHOICES

# variant key -> tiered-search filename stems to try (new convention first, then legacy)
TIERED = {
    "sg": ["sg_dim", "sg"],
    "conceptor": ["sg_conc", "conceptor"],
    "proper": ["sg_gradient_trained", "proper"],
    "conceptor_matrix": ["sg_conc_matrix", "conceptor_matrix"],
    "conceptor_selfproj": ["sg_conc_selfproj", "conceptor_selfproj"],
    "const": ["nogate_dim", "const"],
    "stolfo": ["nogate_clamp", "stolfo"],
}
# all-layer checkpoints: stem -> label. These carry their own winning loss config in the filename.
ALL_LAYER = {
    "mg_gradient_trained_probe_mse": "MG+GradientTrained (MSE)",
    "mg_gradient_trained_probe_nll": "MG+GradientTrained (NLL)",
    "mg_dim_probe_mse": "MG+DiM (MSE)",
    "mg_dim_probe_nll": "MG+DiM (NLL)",
    "mg_conc_probe_mse": "MG+Conc (MSE)",
    "mg_conc_probe_nll": "MG+Conc (NLL)",
    # legacy names, in case migrate_naming.py hasn't run
    "psr_all_layer_probe_mse": "MG+GradientTrained (MSE)",
    "psr_all_layer_probe_nll": "MG+GradientTrained (NLL)",
    "psr_multi_gate_probe_mse": "MG+DiM (MSE)",
    "psr_multi_gate_probe_nll": "MG+DiM (NLL)",
    "psr_multi_gate_conceptor_probe_mse": "MG+Conc (MSE)",
    "psr_multi_gate_conceptor_probe_nll": "MG+Conc (NLL)",
}
# Gated clamp checkpoints. Same dict-of-per-layer-gates shape as the all-layer ones, but they also
# store `targets`, so they need make_multi_gated_clamp_hooks rather than the additive builder.
CLAMP = {
    "sg_clamp_probe_mse": "SG+Clamp (MSE)",
    "sg_clamp_probe_nll": "SG+Clamp (NLL)",
    "mg_clamp_probe_mse": "MG+Clamp (MSE)",
    "mg_clamp_probe_nll": "MG+Clamp (NLL)",
}
OUT_NAME = "token_distributions.json"


def boot_ci(vals, n_boot=4000, seed=0):
    rng = random.Random(seed)
    n = len(vals)
    means = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def summarize(counts):
    m = sum(counts) / len(counts)
    lo, hi = boot_ci(counts)
    s = sorted(counts)
    sd = (sum((x - m) ** 2 for x in counts) / max(len(counts) - 1, 1)) ** 0.5
    return {"avg_tokens": m, "ci_lo": lo, "ci_hi": hi, "sd": sd, "n": len(counts),
            "median": s[len(s) // 2], "p25": s[len(s) // 4], "p75": s[3 * len(s) // 4],
            "token_counts": counts}


def _find_final(results_dir: Path, stems):
    for stem in stems:
        p = results_dir / f"{stem}_tiered_search.jsonl"
        if p.exists():
            finals = load_existing_tiered_results(p).get("final")
            if finals:
                return finals[0], p.name
    return None, None


def _stored_mean(results_dir: Path, variant: str, cond: str, winner: dict):
    """The previously reported mean for this condition, for the drift check."""
    if cond == "alone":
        return winner.get("avg_tokens")
    stem = {"sg": "sg_dim", "conceptor": "sg_conc", "proper": "sg_gradient_trained",
            "const": "nogate_dim", "stolfo": "nogate_clamp"}.get(variant, variant)
    for name in (f"{stem}_prompt_plus.json", f"prompt_plus_steer_{variant}.json"):
        p = results_dir / name
        if p.exists():
            with p.open() as f:
                return json.load(f).get("prompt_plus_steer_avg_tokens")
    return None


def main(task: str, n: int, seed: int, max_batch_rows: int, only: list[str] | None) -> None:
    adapter = get_adapter(task)
    device = "cuda"
    model, tokenizer = load_model(device, model_name=adapter.MODEL_NAME)
    for p in model.parameters():
        p.requires_grad_(False)

    train_items = adapter.to_items(tokenizer, adapter.load_rows("train"))
    dev_items = adapter.to_items(tokenizer, adapter.load_rows("dev"))
    tr = load_or_compute_responses(model, tokenizer, train_items, adapter.CACHE_DIR / "teacher_responses_train.json", generate_response_with_meta)
    dv = load_or_compute_responses(model, tokenizer, dev_items, adapter.CACHE_DIR / "teacher_responses_dev.json", generate_response_with_meta)
    ctx = SearchContext(model=model, tokenizer=tokenizer, n_layers=num_layers(model),
                         hidden_size=model.config.hidden_size, cache_dir=adapter.CACHE_DIR,
                         train_items=train_items, dev_items=dev_items,
                         train_responses=tr, dev_responses=dv, seed=seed)

    test_rows = adapter.load_rows("test")[:n]
    test_items = adapter.to_items(tokenizer, test_rows)
    prompts = {"alone": [it["base_prompt"] for it in test_items],
               "prompt": [it["terse_prompt"] for it in test_items]}

    out, drift_report = {}, []

    # ---- baselines: no steering at all, so they need no hook ----
    for key, cond in (("base", "alone"), ("prompt", "prompt")):
        if only and key not in only:
            continue
        with torch.no_grad():
            rs = generate_batched_uniform(model, tokenizer, prompts[cond], hooks=None,
                                           max_batch_rows=max_batch_rows)
        counts = [token_count(tokenizer, r) for r in rs]
        out[key] = {"label": key.capitalize(), "condition": cond,
                     **summarize(counts), "responses": rs}
        print(f"{key:<28} {out[key]['avg_tokens']:6.1f} "
              f"[{out[key]['ci_lo']:.1f}, {out[key]['ci_hi']:.1f}]  sd={out[key]['sd']:.1f}")

    # ---- single-layer variants, at their own tiered-search winner ----
    for variant, stems in TIERED.items():
        if only and variant not in only:
            continue
        winner, src = _find_final(adapter.RESULTS_DIR, stems)
        if winner is None:
            print(f"{variant:<28} SKIP (no tiered-search Final)")
            continue
        try:
            hook = RETRAIN_FNS[variant](winner, ctx)
        except Exception as e:
            print(f"{variant:<28} SKIP ({type(e).__name__}: {e})")
            continue
        if hook is None:
            print(f"{variant:<28} SKIP (winning config was a skipped point)")
            continue
        for cond in ("alone", "prompt"):
            with torch.no_grad():
                rs = generate_batched_uniform(model, tokenizer, prompts[cond],
                                               hooks={winner["layer"]: hook},
                                               max_batch_rows=max_batch_rows)
            counts = [token_count(tokenizer, r) for r in rs]
            s = summarize(counts)
            stored = _stored_mean(adapter.RESULTS_DIR, variant, cond, winner)
            drift = None if stored is None else s["avg_tokens"] - stored
            key = f"{variant}_{cond}"
            out[key] = {"label": f"{variant} ({cond})", "variant": variant, "condition": cond,
                         "layer": winner["layer"], "source": src,
                         "stored_avg_tokens": stored, "drift_vs_stored": drift, **s,
                         "responses": rs}
            dtxt = "   (no stored mean)" if drift is None else f"   drift={drift:+.2f}"
            print(f"{key:<28} {s['avg_tokens']:6.1f} [{s['ci_lo']:.1f}, {s['ci_hi']:.1f}]  "
                  f"sd={s['sd']:.1f}{dtxt}")
            if drift is not None:
                drift_report.append((key, drift))

    # ---- all-layer (MG) variants, from their saved checkpoints ----
    from steering.psr.gate import GateState
    from steering.psr.multi_gate import make_multi_inference_hooks
    done_labels = set()
    for stem, label in ALL_LAYER.items():
        if only and stem not in only:
            continue
        path = adapter.RESULTS_DIR / f"{stem}.pt"
        if not path.exists() or label in done_labels:
            continue
        done_labels.add(label)
        ckpt = torch.load(path, map_location=device)
        layers = [int(l) for l in ckpt["layer_indices"]]
        gates = {l: GateState(ckpt["gates"][l]["weight"].to(device),
                               ckpt["gates"][l]["bias"].to(device),
                               ckpt["gates"][l]["coeff_bias"].to(device)) for l in layers}
        dirs = {l: ckpt["directions"][l].to(device) for l in layers}
        hooks = make_multi_inference_hooks(gates, dirs, layers)
        for cond in ("alone", "prompt"):
            with torch.no_grad():
                rs = generate_batched_uniform(model, tokenizer, prompts[cond], hooks=hooks,
                                               max_batch_rows=max_batch_rows)
            counts = [token_count(tokenizer, r) for r in rs]
            s = summarize(counts)
            key = f"{stem}_{cond}"
            out[key] = {"label": f"{label} ({cond})", "condition": cond, "layers": layers,
                         "source": f"{stem}.pt", **s, "responses": rs}
            print(f"{key:<28} {s['avg_tokens']:6.1f} [{s['ci_lo']:.1f}, {s['ci_hi']:.1f}]  sd={s['sd']:.1f}")

    # ---- gated clamp variants ----
    from steering.clamp.hooks import make_multi_gated_clamp_hooks
    for stem, label in CLAMP.items():
        if only and stem not in only:
            continue
        path_ck = adapter.RESULTS_DIR / f"{stem}.pt"
        if not path_ck.exists():
            continue
        ckpt = torch.load(path_ck, map_location=device)
        layers = [int(l) for l in ckpt["layer_indices"]]
        gates = {l: GateState(ckpt["gates"][l]["weight"].to(device),
                               ckpt["gates"][l]["bias"].to(device),
                               ckpt["gates"][l]["coeff_bias"].to(device)) for l in layers}
        dirs = {l: ckpt["directions"][l].to(device) for l in layers}
        targets = {l: float(ckpt["targets"][l]) for l in layers}
        hooks = make_multi_gated_clamp_hooks(gates, dirs, targets, layers, response_only=True)
        for cond in ("alone", "prompt"):
            with torch.no_grad():
                rs = generate_batched_uniform(model, tokenizer, prompts[cond], hooks=hooks,
                                               max_batch_rows=max_batch_rows)
            # Skip NaN-poisoned runs (MG+Clamp) instead of recording 20.0 as a length result.
            # Same rule scripts/inventory.py's is_degenerate uses: a long run of one repeated
            # character is the argmax-over-NaN signature, and unique-count alone misses it because
            # the clean prefill still emits a real first token.
            runs = sum(1 for r in rs if re.search(r"(.)\1{9,}", r))
            if runs >= len(rs) * 0.5 or len(set(rs)) <= 2:
                print(f"{stem}_{cond:<22} SKIP (degenerate output -- NaN-poisoned checkpoint)")
                continue
            counts = [token_count(tokenizer, r) for r in rs]
            s = summarize(counts)
            key = f"{stem}_{cond}"
            out[key] = {"label": f"{label} ({cond})", "condition": cond, "layers": layers,
                         "source": f"{stem}.pt", **s, "responses": rs}
            print(f"{key:<28} {s['avg_tokens']:6.1f} [{s['ci_lo']:.1f}, {s['ci_hi']:.1f}]  sd={s['sd']:.1f}")

    path = adapter.RESULTS_DIR / OUT_NAME
    with path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {path}  ({len(out)} conditions)")

    # ---- the verification that decides whether re-judging is needed ----
    if drift_report:
        big = [(k, d) for k, d in drift_report if abs(d) > 0.5]
        print("\ndrift vs previously stored means:")
        print(f"  max |drift| = {max(abs(d) for _, d in drift_report):.2f} tokens "
              f"over {len(drift_report)} conditions")
        if big:
            print(f"  {len(big)} condition(s) drifted more than 0.5 tokens -- their regenerated")
            print("  responses differ from the originals, so the STORED correctness scores may not")
            print("  describe them. Re-judge these (or keep the original means for reporting):")
            for k, d in big:
                print(f"    {k}: {d:+.2f}")
        else:
            print("  all within 0.5 tokens -- generation reproduced, so the already-stored")
            print("  correctness_scores still apply and NO re-judging is needed.")


def judge(task: str) -> None:
    """Score the responses stored in token_distributions.json, making that file the single
    canonical record: tokens and correctness then describe THE SAME generated text.

    This matters beyond tidiness. Regeneration reproduces most responses exactly but not all (see
    the drift report), so pairing a freshly computed token CI with a correctness score from an
    earlier generation mixes two runs. Judging these responses removes that inconsistency
    entirely. Resumable -- already-judged conditions are skipped, so a rate-limit interruption
    costs only the condition in flight.
    """
    from evals.bootstrap_analysis import bootstrap_ci
    from evals.layer_hparam_search import _score_all_concurrently
    from evals.registry import get_eval_adapter

    adapter = get_adapter(task)
    eval_adapter = get_eval_adapter(task)
    primary = eval_adapter.SCORE_FIELDS[0]
    path = adapter.RESULTS_DIR / OUT_NAME
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run without --judge first.")
    with path.open() as f:
        payload = json.load(f)

    test_rows = None
    for key, entry in payload.items():
        if "correct" in entry:
            print(f"{key:<36} already judged")
            continue
        rs = entry.get("responses")
        if not rs:
            continue
        if test_rows is None or len(test_rows) != len(rs):
            test_rows = adapter.load_rows("test")[: len(rs)]
        dicts = _score_all_concurrently(eval_adapter, test_rows, rs)
        scores = [d[primary] for d in dicts]
        pt, lo, hi = bootstrap_ci(scores)
        entry.update(correct=pt, correct_ci_lo=lo, correct_ci_hi=hi, correctness_scores=scores,
                      coherent_rate=sum(1.0 if d["coherent"] else 0.0 for d in dicts) / len(dicts))
        if "conciseness" in eval_adapter.SCORE_FIELDS:
            cs = [d["conciseness"] for d in dicts]
            cpt, clo, chi = bootstrap_ci(cs)
            entry.update(conciseness=cpt, conciseness_ci_lo=clo, conciseness_ci_hi=chi,
                          conciseness_scores=cs)
        print(f"{key:<36} {entry['avg_tokens']:6.1f} tok [{entry['ci_lo']:.1f}, {entry['ci_hi']:.1f}]"
              f"   correct={pt:.3f} [{lo:.3f}, {hi:.3f}]")
        with path.open("w") as f:   # write after EACH condition, so an interruption loses nothing
            json.dump(payload, f, indent=2)
    print(f"\nupdated {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="caveman", choices=TASK_CHOICES)
    ap.add_argument("--n", type=int, default=180)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-batch-rows", type=int, default=DEFAULT_MAX_BATCH_ROWS)
    ap.add_argument("--only", type=str, default=None,
                     help="comma-separated subset of variant keys / checkpoint stems")
    ap.add_argument("--judge", action="store_true",
                     help="score the responses already in token_distributions.json (uses API budget)")
    a = ap.parse_args()
    only = [x.strip() for x in a.only.split(",")] if a.only else None
    if a.judge:
        judge(a.task)
    else:
        main(a.task, a.n, a.seed, a.max_batch_rows, only)
