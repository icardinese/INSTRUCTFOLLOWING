"""The ONE missing generation: Prompt+Steer for the corrected proper/conceptor winners. Everything
else in the final table below is already sitting on disk or in caveman-steer's old results --
this deliberately does NOT regenerate any of it.

Reused, not regenerated:
  - steer_alone avg_tokens/correctness for proper & conceptor: {variant}_tiered_search.jsonl's
    "final" row already has these (avg_tokens, correctness_scores), at n=180, TEST split.
  - prompt_alone avg_tokens/correctness for proper & conceptor: same Final row's prompt_avg_tokens/
    prompt_baseline_scores -- also already paired to the exact same 180 rows, so the comparison
    below is valid without a second generation pass.
  - Base, Prompt, Const, S-PSR, A-PSR, and the OLD (pre-fix) PSR-Proper/PSR-Conceptor: hardcoded
    from caveman-steer (ryan branch) results/summary_test_all12.json, n=180. Cited, not re-derived
    -- these are a DIFFERENT repo/implementation and this project's own numbers shouldn't be
    confused with them; kept here purely as reference context for how much the fix moved things.

Generated fresh here, and ONLY here: prompt_plus_steer for proper and for conceptor, at each
variant's real corrected winner (read from {variant}_tiered_search.jsonl's Final row, same as
combined_condition_final.py did before this rewrite -- but that version ALSO regenerated
base/prompt/steer_alone redundantly for both variants; this doesn't).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters.registry import get_adapter
from core.generation_cache import load_or_compute_responses
from core.model_common import generate_response, load_model, num_layers, token_count
from evals.bootstrap_analysis import bootstrap_ci, paired_bootstrap_diff
from evals.layer_hparam_search import RETRAIN_FNS, SearchContext, load_existing_tiered_results
from evals.registry import get_eval_adapter
from steering.hooks import steering_hook

TASK = "caveman"
N = 180

# caveman-steer (ryan branch) results/summary_test_all12.json, n=180 -- a DIFFERENT repo/
# implementation, kept only as reference context, never used in any comparison's statistics below
# (no per-example scores survived into that summary file, so no paired test is even possible
# against it -- point estimates only).
OLD_REPO_REFERENCE = {
    "base":              {"avg_tokens": 149.93},
    "prompt":            {"avg_tokens": 57.20},
    "const":             {"avg_tokens": 129.77}, "prompt_const":         {"avg_tokens": 20.20},
    "s_psr":             {"avg_tokens": 149.87}, "prompt_s_psr":         {"avg_tokens": 46.11},
    "a_psr":             {"avg_tokens": 132.48}, "prompt_a_psr":         {"avg_tokens": 28.37},
    "psr_conceptor_old": {"avg_tokens": 103.99}, "prompt_psr_conceptor_old": {"avg_tokens": 30.23},
    "psr_proper_old":    {"avg_tokens": 103.16}, "prompt_psr_proper_old":    {"avg_tokens": 38.01},
}


def _mean(values):
    return sum(values) / len(values)


def generate_prompt_plus_steer(variant: str) -> dict:
    adapter = get_adapter(TASK)
    eval_adapter = get_eval_adapter(TASK)
    tiered_path = adapter.RESULTS_DIR / f"{variant}_tiered_search.jsonl"
    existing = load_existing_tiered_results(tiered_path)
    if "final" not in existing:
        raise FileNotFoundError(f"{tiered_path} has no Final row.")
    winner = existing["final"][0]
    print(f"{variant} winner: layer={winner['layer']} alpha={winner.get('alpha', 'n/a')} "
          f"mse_weight={winner['mse_weight']} nll_weight={winner['nll_weight']}")

    device = "cuda"
    model, tokenizer = load_model(device, model_name=adapter.MODEL_NAME)
    for p in model.parameters():
        p.requires_grad_(False)

    train_items = adapter.to_items(tokenizer, adapter.load_rows("train"))
    dev_items = adapter.to_items(tokenizer, adapter.load_rows("dev"))
    train_responses = load_or_compute_responses(model, tokenizer, train_items, adapter.CACHE_DIR / "teacher_responses_train.json", generate_response)
    dev_responses = load_or_compute_responses(model, tokenizer, dev_items, adapter.CACHE_DIR / "teacher_responses_dev.json", generate_response)
    ctx = SearchContext(
        model=model, tokenizer=tokenizer, n_layers=num_layers(model), hidden_size=model.config.hidden_size,
        cache_dir=adapter.CACHE_DIR, train_items=train_items, dev_items=dev_items,
        train_responses=train_responses, dev_responses=dev_responses, seed=42,
    )
    hook_fn = RETRAIN_FNS[variant](winner, ctx)
    if hook_fn is None:
        raise RuntimeError(f"{variant}'s winning config was itself a skipped point -- can't generate with it.")

    test_rows = adapter.load_rows("test")[:N]  # SAME [:N] slice Final used -> identical rows, valid pairing
    test_items = adapter.to_items(tokenizer, test_rows)

    print(f"generating the ONE missing condition: Prompt+Steer, n={N} ...")
    responses = []
    for item in test_items:
        with steering_hook(model, winner["layer"], hook_fn):
            responses.append(generate_response(model, tokenizer, item["terse_prompt"]))

    primary_field = eval_adapter.SCORE_FIELDS[0]
    score_dicts = [eval_adapter.score_response(row, resp) for row, resp in zip(test_rows, responses)]
    correctness_scores = [d[primary_field] for d in score_dicts]
    avg_tokens = _mean([token_count(tokenizer, r) for r in responses])

    return {
        "variant": variant, "winner": winner, "n": N,
        "prompt_plus_steer_avg_tokens": avg_tokens,
        "prompt_plus_steer_correctness_scores": correctness_scores,
        "steer_alone_avg_tokens": winner["avg_tokens"],
        "steer_alone_correctness_scores": winner["correctness_scores"],
        "prompt_alone_avg_tokens": winner["prompt_avg_tokens"],
        "prompt_alone_correctness_scores": winner["prompt_baseline_scores"],
    }


def print_table(results: dict) -> None:
    print(f"\n{'condition':<28}{'avg_tokens':>12}{'correct':>10}")
    for name, v in OLD_REPO_REFERENCE.items():
        print(f"{'[old repo] ' + name:<28}{v['avg_tokens']:>12.1f}{'n/a':>10}")
    for variant, r in results.items():
        point_alone, _, _ = bootstrap_ci(r["steer_alone_correctness_scores"])
        point_combined, _, _ = bootstrap_ci(r["prompt_plus_steer_correctness_scores"])
        point_prompt, _, _ = bootstrap_ci(r["prompt_alone_correctness_scores"])
        print(f"{'[new fix] ' + variant + '_alone':<28}{r['steer_alone_avg_tokens']:>12.1f}{point_alone:>10.3f}")
        print(f"{'[new fix] prompt_' + variant:<28}{r['prompt_plus_steer_avg_tokens']:>12.1f}{point_combined:>10.3f}")
    # prompt_alone should match across variants (same rows, no steering) -- print once, and warn if it doesn't
    prompt_vals = {v["prompt_alone_avg_tokens"] for v in results.values()}
    if len(prompt_vals) > 1:
        print(f"\nWARNING: prompt_alone avg_tokens differs across variants ({prompt_vals}) -- "
              f"test split rows may not actually be identical, check before trusting the paired comparisons below.")
    print(f"{'[new fix] prompt (shared)':<28}{next(iter(prompt_vals)):>12.1f}{point_prompt:>10.3f}")

    print(f"\n== paired comparisons (n={N}) ==")
    for variant, r in results.items():
        diff, lo, hi = paired_bootstrap_diff(r["prompt_alone_correctness_scores"], r["prompt_plus_steer_correctness_scores"])
        verdict = "SIGNIFICANTLY BEATS Prompt" if lo > 0 else "SIGNIFICANTLY WORSE than Prompt" if hi < 0 else "not significantly different from Prompt"
        print(f"  prompt_{variant} vs prompt alone: diff={diff:.4f} [{lo:.4f}, {hi:.4f}] -- {verdict}  "
              f"(tokens: {r['prompt_plus_steer_avg_tokens']:.1f} vs {r['prompt_alone_avg_tokens']:.1f})")


if __name__ == "__main__":
    adapter = get_adapter(TASK)
    results = {}
    for variant in ("proper", "conceptor"):
        r = generate_prompt_plus_steer(variant)
        results[variant] = r
        out_path = adapter.RESULTS_DIR / f"prompt_plus_steer_{variant}.json"
        with out_path.open("w") as f:
            json.dump({k: v for k, v in r.items() if k != "winner"}, f, indent=2)
        print(f"wrote {out_path}")
    print_table(results)
