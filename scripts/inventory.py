"""Inventory of every result on disk, laid out as the Gate x Direction grid.

READ-ONLY. No GPU, no API, no writes except the CSV/markdown it emits. Run it any time to see
exactly what exists.

WHY THIS EXISTS. Nothing in a filename says which method it is: `psr_conceptor_probe.pt` doesn't
say "SG+Conc", and `ablation_direction_only_psr_conceptor_probe_tiered_winner.json` doesn't say
"NoGate+Conc". That made it genuinely unclear, in conversation and on disk, whether a given cell
had been run. This script is the single answer: it reads the actual result files, derives the
numbers from stored per-example scores, and prints each cell with the file it came from -- so no
claim about what's been run depends on anyone's memory.

NAMING. Every method is exactly two coordinates, Gate + Direction:

  Gate       -- how the per-token coefficient is produced
    NoGate   one fixed scalar at every position (calibrated once, not learned)
    SG       one trained probe relu(w.h + b), at ONE layer
    MG       trained probes at ALL layers, jointly optimized in one shared forward pass

  Direction  -- what that coefficient scales
    DiM              diff-in-means, closed form, frozen
    Conc             DiM reshaped once through that layer's conceptor matrix C, frozen
    GradientTrained  a leaf tensor optimized by gradient descent alongside the gate(s)
    Clamp            Stolfo's projection clamp -- forces h's component along u to a target; not a
                     (direction, coefficient) pair at all, so it sits OUTSIDE the grid rather than
                     being forced into a cell it doesn't fit

CONDITION is a third axis, not part of the name: `alone` (steering only) vs `+Prompt` (steering
applied on top of the real textual instruction). The same trained artifact is measured under both,
so folding it into the method name would conflate identity with measurement.

Paper-facing aliases, used in prose only, never as file or variant names:
    SG+GradientTrained = S-PSR (Heyman & Vandeputte)
    MG+GradientTrained = A-PSR (Heyman & Vandeputte)
    NoGate+DiM         = Const (Stolfo et al.'s additive baseline)
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.bootstrap_analysis import bootstrap_ci

GATES = ["NoGate", "SG", "MG"]
# Clamp is a full column as of 2026-09-19: it was reported outside the grid while only the
# ungated NoGate+Clamp existed (no (direction, coefficient) decomposition to place), but
# src/psr/clamp_gate/train.py adds SG+Clamp and MG+Clamp, so the gate axis is now populated for it
# exactly like the additive columns.
DIRECTIONS = ["DiM", "Conc", "GradientTrained", "Clamp"]

ALIASES = {
    ("SG", "GradientTrained"): "S-PSR",
    ("MG", "GradientTrained"): "A-PSR",
    ("NoGate", "DiM"): "Const",
}

# Sub-variants of the conceptor direction: these use C differently (an absolute mu_instr target,
# or pure self-projection) rather than reshaping a DiM vector, so they are NOT interchangeable with
# Conc and are reported separately instead of overwriting that cell.
SUBVARIANTS = {
    "conceptor_matrix": ("SG", "Conc-Matrix"),
    "conceptor_selfproj": ("SG", "Conc-SelfProj"),
}

# ---- how each file on disk maps to (gate, direction). The mapping IS the documentation. ----
# tiered-search files: {variant}_tiered_search.jsonl, 'final' row = n=180 confirmation, ALONE only
# (evals/layer_hparam_search.py never steers terse_prompt).
# Both old and new stems are listed so this script works before, during and after
# scripts/migrate_naming.py runs -- in --mode copy both names exist simultaneously, and reading
# whichever is present avoids a window where the inventory silently reports NOT RUN.
TIERED = {
    # new (Gate+Direction) names
    "sg_dim":                   ("SG", "DiM"),
    "sg_conc":                  ("SG", "Conc"),
    "sg_gradient_trained":      ("SG", "GradientTrained"),
    "sg_conc_matrix":           ("SG", "Conc-Matrix"),
    "sg_conc_selfproj":         ("SG", "Conc-SelfProj"),
    "nogate_dim":               ("NoGate", "DiM"),
    "nogate_clamp":             ("NoGate", "Clamp"),
    # legacy names
    "sg":                 ("SG", "DiM"),
    "conceptor":          ("SG", "Conc"),
    "proper":             ("SG", "GradientTrained"),
    "conceptor_matrix":   ("SG", "Conc-Matrix"),
    "conceptor_selfproj": ("SG", "Conc-SelfProj"),
    "const":              ("NoGate", "DiM"),
    "stolfo":             ("NoGate", "Clamp"),
}
# all_layer_variants_eval.json keys -> (gate, direction). Has BOTH conditions.
ALL_LAYER = {
    "mg_dim":                         ("MG", "DiM"),
    "mg_conc":                        ("MG", "Conc"),
    "mg_gradient_trained":            ("MG", "GradientTrained"),
    "psr_multi_gate_probe":           ("MG", "DiM"),
    "psr_multi_gate_conceptor_probe": ("MG", "Conc"),
    "psr_all_layer_probe":            ("MG", "GradientTrained"),
}
# prompt_plus_steer_{variant}.json -> the +Prompt condition for an SG method.
PROMPT_PLUS = {
    "proper":    ("SG", "GradientTrained"),
    # new-style: {gate}_{direction}_prompt_plus.json, handled by the loop below as well

    "conceptor": ("SG", "Conc"),
    "sg":        ("SG", "DiM"),
}
# ablation_direction_only_{checkpoint}.json -> gate STRIPPED, frozen direction at a constant
# coefficient. That is mechanically Const's mechanism with a different direction, so it lands in
# the NoGate row, inheriting whichever direction the source checkpoint had. Has BOTH conditions.
ABLATION = {
    "psr_proper_probe_tiered_winner":    ("NoGate", "GradientTrained"),
    "psr_conceptor_probe_tiered_winner": ("NoGate", "Conc"),
}
# post-migration filenames for the same three ablations
ABLATION_NEW = {
    "nogate_gradient_trained_eval.json": ("NoGate", "GradientTrained"),
    "nogate_conc_eval.json":             ("NoGate", "Conc"),
    "nogate_dim_from_sg_eval.json":      ("NoGate", "DiM"),
}


def is_degenerate(responses: list[str]) -> bool:
    """True if a run looks NaN-poisoned rather than merely terse.

    Two signatures, because either alone gives false negatives:
      - a long run of one repeated character ("The!!!!!!!!!!..."), which is what argmax over a NaN
        logit vector produces once it locks onto a token id;
      - near-total lack of distinct outputs across the whole set.
    Counting unique strings ALONE is not enough: a NaN run still emits a real first token from the
    clean prefill, so "Function!!!...", "Parse!!!..." differ as strings while carrying no content.
    That exact case slipped past an earlier unique-count-only check.
    """
    if not responses:
        return False
    runs = sum(1 for r in responses if re.search(r"(.)\1{9,}", r))
    if runs >= len(responses) * 0.5:
        return True
    return len(set(responses)) <= max(2, len(responses) // 50)


def _read_jsonl(p: Path):
    if not p.exists():
        return []
    with p.open() as f:
        return [json.loads(l) for l in f if l.strip()]


def _load(p: Path):
    if not p.exists():
        return None
    with p.open() as f:
        return json.load(f)


def _entry(tokens, scores=None, point=None, source="", extra="", conc=None, conc_lo=None, conc_hi=None):
    """One measured condition. `scores` (per-example) is preferred so a CI can be derived here
    rather than trusting a stored summary; `point` is the fallback when only a mean survived."""
    e = {"tokens": tokens, "source": source, "extra": extra}
    # Judged conciseness, when the producing script stored it. Only the tiered searches do
    # (as optimize_score + its CI); the per-example conciseness scores are not persisted anywhere,
    # so this is a mean with a CI rather than something re-derivable here.
    if conc is not None:
        e.update(conc=conc, conc_lo=conc_lo, conc_hi=conc_hi)
    if scores:
        pt, lo, hi = bootstrap_ci(scores)
        e.update(correct=pt, lo=lo, hi=hi, n=len(scores))
    elif point is not None:
        e.update(correct=point, lo=None, hi=None, n=None)
    return e


def collect(results_dir: Path):
    """Returns (cells, outside, baselines, unclassified).
    cells: {(gate, direction): {"alone": entry|None, "prompt": entry|None}}"""
    cells: dict[tuple, dict] = {}
    outside: dict[tuple, dict] = {}
    baselines: dict[str, dict] = {}
    surface_twins: dict[tuple, dict] = {}
    degenerate_runs: list[str] = []
    seen: set[str] = set()

    def put(gate, direction, cond, entry):
        cells.setdefault((gate, direction), {"alone": None, "prompt": None})[cond] = entry

    # --- tiered searches (alone) ---
    for variant, (gate, direction) in TIERED.items():
        p = results_dir / f"{variant}_tiered_search.jsonl"
        if not p.exists():
            continue
        seen.add(p.name)
        finals = [r for r in _read_jsonl(p) if r.get("tier") == "final"]
        if not finals:
            continue
        r = finals[0]
        extra = f"L{r.get('layer')}"
        if r.get("alpha") is not None:
            extra += f" a={r['alpha']}"
        if r.get("coeff") is not None:
            extra += f" c={r['coeff']}"
        if r.get("nll_weight"):
            extra += " NLL"
        elif r.get("mse_weight"):
            extra += " MSE"
        put(gate, direction, "alone",
            _entry(r.get("avg_tokens"), r.get("correctness_scores"), r.get("judged_score"), p.name, extra,
                   conc=r.get("optimize_score"), conc_lo=r.get("optimize_ci_lo"),
                   conc_hi=r.get("optimize_ci_hi")))
        # Every tiered final also carries the Prompt-alone baseline it was paired against.
        if r.get("prompt_baseline_scores") and "prompt" not in baselines:
            baselines["prompt"] = _entry(r.get("prompt_avg_tokens"), r["prompt_baseline_scores"],
                                          source=p.name)

    # --- all-layer eval (both conditions) ---
    p = results_dir / "mg_variants_eval.json"
    if not p.exists():
        p = results_dir / "all_layer_variants_eval.json"
    payload = _load(p)
    if payload:
        seen.add(p.name)
        res = payload.get("results", {})
        for key, r in res.items():
            if key in ("base", "prompt"):
                baselines[key] = _entry(r.get("avg_tokens"), r.get("scores"), source=p.name)
                continue
            cond = "prompt" if key.endswith("_combined") else "alone" if key.endswith("_alone") else None
            if cond is None:
                continue
            stem = key.rsplit("_", 1)[0]
            loss = ""
            for lw in ("mse", "nll"):
                if stem.endswith(f"_{lw}"):
                    loss, stem = lw.upper(), stem[: -(len(lw) + 1)]
            if stem not in ALL_LAYER:
                continue
            gate, direction = ALL_LAYER[stem]
            prev = (cells.get((gate, direction)) or {}).get(cond)
            cand = _entry(r.get("avg_tokens"), r.get("scores"), source=p.name, extra=f"all L, {loss}")
            # Two loss objectives share one cell; keep the fewer-token one and say so, rather than
            # letting dict order silently decide which is displayed.
            if prev is None or (cand["tokens"] or 1e9) < (prev["tokens"] or 1e9):
                put(gate, direction, cond, cand)

    # --- prompt_plus_steer (the +Prompt condition for SG methods) ---
    prompt_plus_files = {f"prompt_plus_steer_{v}.json": gd for v, gd in PROMPT_PLUS.items()}
    prompt_plus_files.update({
        "sg_dim_prompt_plus.json":              ("SG", "DiM"),
        "sg_conc_prompt_plus.json":             ("SG", "Conc"),
        "sg_gradient_trained_prompt_plus.json": ("SG", "GradientTrained"),
        "nogate_dim_prompt_plus.json":          ("NoGate", "DiM"),
        "nogate_clamp_prompt_plus.json":        ("NoGate", "Clamp"),
        "sg_conc_matrix_prompt_plus.json":      ("SG", "Conc-Matrix"),
        "sg_conc_selfproj_prompt_plus.json":    ("SG", "Conc-SelfProj"),
    })
    for fname, (gate, direction) in prompt_plus_files.items():
        p = results_dir / fname
        d = _load(p)
        if not d:
            continue
        seen.add(p.name)
        put(gate, direction, "prompt",
            _entry(d.get("prompt_plus_steer_avg_tokens"),
                   d.get("prompt_plus_steer_correctness_scores"), source=p.name))
        cur = (cells.get((gate, direction)) or {}).get("alone")
        if cur is None and d.get("steer_alone_correctness_scores"):
            put(gate, direction, "alone",
                _entry(d.get("steer_alone_avg_tokens"),
                       d["steer_alone_correctness_scores"], source=p.name))

    # --- gated clamp (SG+Clamp / MG+Clamp), both conditions ---
    p = results_dir / "clamp_gate_eval.json"
    d = _load(p)
    if d:
        seen.add(p.name)
        for stem, entry in d.items():
            gate = "SG" if stem.startswith("sg_") else "MG"
            loss = "NLL" if entry.get("nll_weight") else "MSE"
            degenerate = False
            for cond in ("alone", "prompt"):
                e = entry.get(cond) or {}
                # EXCLUDE NaN-poisoned runs rather than plotting 20.0 tokens as a length result.
                if is_degenerate(e.get("responses") or []):
                    degenerate = True
            if degenerate:
                degenerate_runs.append(f"{gate}+Clamp ({loss})")
                continue
            for cond in ("alone", "prompt"):
                e = entry.get(cond) or {}
                if e.get("avg_tokens") is None:
                    continue
                put(gate, "Clamp", cond,
                    _entry(e["avg_tokens"], e.get("scores"), e.get("correct"), p.name,
                           f"{len(entry.get('layers', []))}L, {loss}"))

    # --- surface-ablation twins: same method, response-only surface ---
    p = results_dir / "surface_ablation.json"
    d = _load(p)
    if d:
        seen.add(p.name)
        for variant, v in d.items():
            direction = "Clamp" if v.get("form") == "clamp" else "DiM"
            e = (v.get("surfaces") or {}).get("response_only") or {}
            if e.get("avg_tokens") is None:
                continue
            # Response-only is a different INTERVENTION SURFACE, not a different cell, so it is
            # tracked separately instead of overwriting the published all-positions result.
            surface_twins[("NoGate", direction)] = _entry(
                e["avg_tokens"], e.get("scores"), e.get("correct"), p.name, "response-only")

    # --- direction-only ablations (gate stripped -> NoGate row, both conditions) ---
    ablation_files = {f"ablation_direction_only_{s}.json": gd for s, gd in ABLATION.items()}
    ablation_files.update(ABLATION_NEW)
    for fname, (gate, direction) in ablation_files.items():
        p = results_dir / fname
        d = _load(p)
        if not d:
            continue
        seen.add(p.name)
        fin, sc = d.get("final", {}), d.get("final_correctness_scores", {})
        extra = f"L{d.get('layer')} coeff={d.get('chosen_coeff')}"
        put(gate, direction, "alone",
            _entry(fin.get("alone_avg_tokens"), sc.get("alone"), fin.get("alone_correct"), p.name, extra))
        put(gate, direction, "prompt",
            _entry(fin.get("combined_avg_tokens"), sc.get("combined"), fin.get("combined_correct"), p.name, extra))

    known = set(seen)
    unclassified = sorted(
        f.name for f in results_dir.iterdir()
        if f.suffix in (".json", ".jsonl") and f.name not in known
        and not f.name.endswith(".log")
    )
    return cells, outside, baselines, unclassified, surface_twins, degenerate_runs


def _fmt(e):
    if e is None:
        return "—"
    tok = f"{e['tokens']:.1f}" if e.get("tokens") is not None else "?"
    if e.get("correct") is None:
        return f"{tok} tok"
    ci = f" [{e['lo']:.3f},{e['hi']:.3f}]" if e.get("lo") is not None else ""
    return f"{tok} tok / {e['correct']:.3f}{ci}"


def main(task: str) -> None:
    from adapters.registry import get_adapter
    results_dir = get_adapter(task).RESULTS_DIR
    cells, outside, baselines, unclassified, surface_twins, degenerate_runs = collect(results_dir)

    print(f"\n{'=' * 78}\nRESULT INVENTORY -- {results_dir}\n{'=' * 78}")

    print("\nBASELINES")
    for k in ("base", "prompt"):
        if k in baselines:
            print(f"  {k.capitalize():<8} {_fmt(baselines[k]):<38} [{baselines[k]['source']}]")
        else:
            print(f"  {k.capitalize():<8} —  NOT FOUND")

    print(f"\nGATE x DIRECTION GRID   (each cell: alone  |  +Prompt)")
    print(f"{'-' * 78}")
    for gate in GATES:
        print(f"\n{gate}")
        for direction in DIRECTIONS:
            c = cells.get((gate, direction))
            alias = ALIASES.get((gate, direction))
            name = f"{gate}+{direction}" + (f"  ({alias})" if alias else "")
            if c is None:
                print(f"  {name:<34} NOT RUN")
                continue
            a, pp = c.get("alone"), c.get("prompt")
            print(f"  {name:<34}")
            # Config is printed PER CONDITION because the two conditions in one cell are selected
            # independently (lowest tokens wins), so they can come from different loss objectives.
            # Showing one config for the whole cell would imply they matched when they may not.
            for lbl, e in (("alone", a), ("+Prompt", pp)):
                cfg = f"   [{e['extra']}]" if e and e.get("extra") else ""
                print(f"    {lbl:<9} {_fmt(e)}{cfg}")
            if a and pp and a.get("extra") and pp.get("extra") and a["extra"] != pp["extra"]:
                print(f"    {'':<9} NOTE: conditions come from DIFFERENT configs above")
            src = {x["source"] for x in (a, pp) if x}
            print(f"    {'source':<9} {', '.join(sorted(src))}")

    subs = {k: v for k, v in cells.items() if k[1].startswith("Conc-")}
    if subs:
        print(f"\nCONCEPTOR SUB-VARIANTS  (use C differently -- not interchangeable with Conc)")
        print(f"{'-' * 78}")
        for (gate, direction), c in sorted(subs.items()):
            a, pp = c.get("alone"), c.get("prompt")
            print(f"  {gate}+{direction:<26} {(a or pp or {}).get('extra','')}")
            print(f"    {'alone':<9} {_fmt(a)}")
            print(f"    {'+Prompt':<9} {_fmt(pp)}")

    if outside:
        print(f"\nOUTSIDE THE GRID  (no (direction, coefficient) decomposition)")
        print(f"{'-' * 78}")
        for (gate, direction), c in sorted(outside.items()):
            a, pp = c.get("alone"), c.get("prompt")
            print(f"  {gate}+{direction:<26} {(a or pp or {}).get('extra','')}")
            print(f"    {'alone':<9} {_fmt(a)}")
            print(f"    {'+Prompt':<9} {_fmt(pp)}")

    if surface_twins:
        print(f"\nSURFACE TWINS  (same method, response-only surface instead of all positions)")
        print(f"{'-' * 78}")
        for (gate, direction), e in sorted(surface_twins.items()):
            print(f"  {gate}+{direction:<26} {_fmt(e)}   [{e['source']}]")
    if degenerate_runs:
        print(f"\nEXCLUDED -- DEGENERATE OUTPUT ({len(degenerate_runs)})")
        print(f"{'-' * 78}")
        for r in degenerate_runs:
            print(f"  {r}: collapsed to <=2 unique responses (NaN-poisoned checkpoint)")

    missing = [f"{g}+{d}" for g in GATES for d in DIRECTIONS if (g, d) not in cells]
    partial = [f"{g}+{d} (+Prompt)" for g in GATES for d in DIRECTIONS
               if (g, d) in cells and cells[(g, d)].get("prompt") is None]
    partial += [f"{g}+{d} (alone)" for g in GATES for d in DIRECTIONS
                if (g, d) in cells and cells[(g, d)].get("alone") is None]
    for (g, d), c in list(outside.items()) + [(k, v) for k, v in cells.items() if k[1].startswith("Conc-")]:
        if c.get("prompt") is None:
            partial.append(f"{g}+{d} (+Prompt)")
        if c.get("alone") is None:
            partial.append(f"{g}+{d} (alone)")

    print(f"\n{'=' * 78}\nGAPS\n{'=' * 78}")
    print(f"  cells with NO data ({len(missing)}): {', '.join(missing) or 'none'}")
    print(f"  cells missing one condition ({len(partial)}): {', '.join(sorted(set(partial))) or 'none'}")
    if unclassified:
        print(f"\n  UNCLASSIFIED FILES ({len(unclassified)}) -- present but not mapped to any cell.")
        print(f"  Either they're logs/summaries, or the mapping tables in this file need updating:")
        for f in unclassified:
            print(f"    {f}")

    out = results_dir / "inventory.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gate", "direction", "alias", "condition", "avg_tokens", "correct",
                     "ci_lo", "ci_hi", "n", "config", "source"])
        for coll in (cells, outside):
            for (gate, direction), c in sorted(coll.items()):
                for cond in ("alone", "prompt"):
                    e = c.get(cond)
                    if not e:
                        continue
                    w.writerow([gate, direction, ALIASES.get((gate, direction), ""),
                                 "alone" if cond == "alone" else "+Prompt",
                                 f"{e['tokens']:.1f}" if e.get("tokens") is not None else "",
                                 f"{e['correct']:.4f}" if e.get("correct") is not None else "",
                                 f"{e['lo']:.4f}" if e.get("lo") is not None else "",
                                 f"{e['hi']:.4f}" if e.get("hi") is not None else "",
                                 e.get("n") or "", e.get("extra", ""), e.get("source", "")])
    print(f"\nwrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="caveman", choices=["caveman", "ifeval"])
    main(ap.parse_args().task)
