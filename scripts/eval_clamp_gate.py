"""Evaluate the gated-clamp checkpoints (SG+Clamp / MG+Clamp).

Splits the free work from the rate-limited work, because a daily request cap -- not GPU time -- is
currently the binding constraint. avg_tokens comes from the local tokenizer and costs nothing, and
it is this project's primary length metric, so the headline result is fully obtainable with no API
budget at all. Correctness is the only part that needs the judge, and it can be added later from
the saved responses without regenerating anything.

  default        generate both conditions, report avg_tokens, save responses.  ZERO API calls.
  --judge        additionally score correctness now (needs request budget).
  --judge-saved  score the responses from a previous run; no model load, no generation.

Reads whichever checkpoints exist: {sg,mg}_clamp_probe_{mse,nll}.pt as written by
src/psr/clamp_gate/train.py's sweep().
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
from evals.layer_hparam_search import DEFAULT_MAX_BATCH_ROWS, _score_all_concurrently
from evals.registry import get_eval_adapter
from steering.batch_routing import generate_batched_uniform
from steering.clamp.hooks import make_multi_gated_clamp_hooks
from steering.psr.gate import GateState

CONFIGS = [
    ("sg_clamp_probe_mse", "SG+Clamp (MSE)"),
    ("sg_clamp_probe_nll", "SG+Clamp (NLL)"),
    ("mg_clamp_probe_mse", "MG+Clamp (MSE)"),
    ("mg_clamp_probe_nll", "MG+Clamp (NLL)"),
]
OUT_NAME = "clamp_gate_eval.json"


def load_ckpt(path: Path, device: str):
    ckpt = torch.load(path, map_location=device)
    layers = [int(l) for l in ckpt["layer_indices"]]
    gates = {
        l: GateState(ckpt["gates"][l]["weight"].to(device),
                      ckpt["gates"][l]["bias"].to(device),
                      ckpt["gates"][l]["coeff_bias"].to(device))
        for l in layers
    }
    directions = {l: ckpt["directions"][l].to(device) for l in layers}
    targets = {l: float(ckpt["targets"][l]) for l in layers}
    return gates, directions, targets, layers, ckpt


def run(task: str, judge: bool, n: int, max_batch_rows: int) -> None:
    adapter = get_adapter(task)
    eval_adapter = get_eval_adapter(task)
    primary = eval_adapter.SCORE_FIELDS[0]

    available = [(s, lbl) for s, lbl in CONFIGS if (adapter.RESULTS_DIR / f"{s}.pt").exists()]
    if not available:
        raise FileNotFoundError(
            f"no gated-clamp checkpoints in {adapter.RESULTS_DIR}. Train them first:\n"
            f"  python3 src/psr/clamp_gate/train.py --task {task} --layers <L> --sweep   # SG+Clamp\n"
            f"  python3 src/psr/clamp_gate/train.py --task {task} --sweep                # MG+Clamp")
    print(f"found {len(available)}/{len(CONFIGS)} checkpoints: {', '.join(s for s, _ in available)}")

    device = "cuda"
    model, tokenizer = load_model(device, model_name=adapter.MODEL_NAME)
    for p in model.parameters():
        p.requires_grad_(False)

    test_rows = adapter.load_rows("test")[:n]
    test_items = adapter.to_items(tokenizer, test_rows)
    base_prompts = [it["base_prompt"] for it in test_items]
    terse_prompts = [it["terse_prompt"] for it in test_items]

    out = {}
    for stem, label in available:
        gates, directions, targets, layers, ckpt = load_ckpt(adapter.RESULTS_DIR / f"{stem}.pt", device)
        print(f"\n{label}: {len(layers)} layer(s) hooked {layers if len(layers) <= 4 else '(all)'}")
        # response_only=True: the gate was trained with answer_only_mask, so it has no gradient
        # signal on prompt positions -- applying it there would use a gate that is undefined over
        # most of what it touches. This also matches the PSR family's surface, which is the
        # comparison these cells exist to make.
        hooks = make_multi_gated_clamp_hooks(gates, directions, targets, layers, response_only=True)

        entry = {"label": label, "layers": layers, "n": n,
                 "mse_weight": ckpt.get("mse_weight"), "nll_weight": ckpt.get("nll_weight")}
        for cond, prompts in (("alone", base_prompts), ("prompt", terse_prompts)):
            with torch.no_grad():
                responses = generate_batched_uniform(
                    model, tokenizer, prompts, hooks=hooks, max_batch_rows=max_batch_rows)
            tokens = sum(token_count(tokenizer, r) for r in responses) / len(responses)
            e = {"avg_tokens": tokens, "responses": responses}
            if judge:
                dicts = _score_all_concurrently(eval_adapter, test_rows, responses)
                scores = [d[primary] for d in dicts]
                pt, lo, hi = bootstrap_ci(scores)
                e.update(correct=pt, ci_lo=lo, ci_hi=hi, scores=scores,
                          coherent_rate=sum(1.0 if d["coherent"] else 0.0 for d in dicts) / len(dicts))
                print(f"  {cond:<8} {tokens:6.1f} tok / {pt:.3f} [{lo:.3f}, {hi:.3f}]")
            else:
                print(f"  {cond:<8} {tokens:6.1f} tok   (judging skipped)")
            entry[cond] = e
        entry["judged"] = judge
        out[stem] = entry

    path = adapter.RESULTS_DIR / OUT_NAME
    with path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {path}")

    # Context the numbers are only meaningful against: the ungated clamp at the same surface, and
    # the best gated additive method. Read from disk rather than hardcoded so it can't go stale.
    print("\nreference points (response-only surface, steering alone):")
    for stem, lbl in (("nogate_clamp", "ungated clamp, ALL positions"),):
        p2 = adapter.RESULTS_DIR / f"{stem}_tiered_search.jsonl"
        if p2.exists():
            fin = [json.loads(l) for l in p2.open() if l.strip()]
            fin = [r for r in fin if r.get("tier") == "final"]
            if fin:
                print(f"  {lbl:<34} {fin[0]['avg_tokens']:6.1f} tok")
    sa = adapter.RESULTS_DIR / "surface_ablation.json"
    if sa.exists():
        with sa.open() as f:
            d = json.load(f)
        if "stolfo" in d:
            ro = d["stolfo"]["surfaces"].get("response_only", {}).get("avg_tokens")
            if ro is not None:
                print(f"  {'ungated clamp, response-only':<34} {ro:6.1f} tok  <-- beat this")
    if not judge:
        print(f"\nAdd correctness later without regenerating:\n"
              f"  python3 scripts/eval_clamp_gate.py --task {task} --judge-saved")


def judge_saved(task: str) -> None:
    adapter = get_adapter(task)
    eval_adapter = get_eval_adapter(task)
    primary = eval_adapter.SCORE_FIELDS[0]
    path = adapter.RESULTS_DIR / OUT_NAME
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run without --judge-saved first.")
    with path.open() as f:
        payload = json.load(f)
    test_rows = None

    for stem, entry in payload.items():
        if entry.get("judged"):
            print(f"{stem}: already judged, skipping")
            continue
        if test_rows is None:
            test_rows = adapter.load_rows("test")[: entry["n"]]
        print(f"\n{entry['label']}")
        for cond in ("alone", "prompt"):
            e = entry.get(cond)
            if not e or "scores" in e:
                continue
            dicts = _score_all_concurrently(eval_adapter, test_rows, e["responses"])
            scores = [d[primary] for d in dicts]
            pt, lo, hi = bootstrap_ci(scores)
            e.update(correct=pt, ci_lo=lo, ci_hi=hi, scores=scores,
                      coherent_rate=sum(1.0 if d["coherent"] else 0.0 for d in dicts) / len(dicts))
            print(f"  {cond:<8} {e['avg_tokens']:6.1f} tok / {pt:.3f} [{lo:.3f}, {hi:.3f}]")
        if all("scores" in entry[c] for c in ("alone", "prompt") if c in entry):
            entry["judged"] = True
        with path.open("w") as f:
            json.dump(payload, f, indent=2)
    print(f"\nupdated {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="caveman", choices=["caveman", "ifeval"])
    ap.add_argument("--judge", action="store_true", help="score correctness now (uses API budget)")
    ap.add_argument("--judge-saved", action="store_true", help="score a previous run's saved responses")
    ap.add_argument("--n", type=int, default=180)
    ap.add_argument("--max-batch-rows", type=int, default=DEFAULT_MAX_BATCH_ROWS)
    a = ap.parse_args()
    if a.judge_saved:
        judge_saved(a.task)
    else:
        run(a.task, a.judge, a.n, a.max_batch_rows)
