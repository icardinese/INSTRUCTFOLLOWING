"""Presentation figure suite.

ENCODING -- three factors, three visual channels, so the eye reads categories rather than points:

    SHAPE  = gate structure        NoGate = square,  SG = circle,  MG = diamond
    HUE    = intervention          DiM = blue, Conc = orange, GradientTrained = green,
                                   Clamp = purple
    SHADE  = condition             alone = light fill, +Prompt = dark fill (same hue)

Shape and hue are independent channels, so "all diamonds" reads as multi-gated regardless of
technique, and "all blue" reads as diff-in-means regardless of gate. Light/dark within one hue
keeps each method's two conditions visually bound together instead of scattered -- which is the
point, since the alone-vs-+Prompt pairing is where the joint-intervention claim lives.

Colour is never the ONLY carrier: every hue has a distinct shape partner and every point is
directly labelled, so the figures survive greyscale printing and colour-vision deficiency. The
palette is Okabe-Ito-derived for the same reason.

DATA SOURCES. Everything comes from scripts/inventory.py's collect(), which is the single mapping
from files on disk to (gate, direction, condition) cells -- so these figures cannot disagree with
the inventory table about what exists. Nothing here re-runs a model or calls an API; all
statistics are re-derived from stored per-example scores.

CONFIDENCE INTERVALS. Drawn by default on every estimate that has stored per-example scores, and
their absence is stated rather than hidden (avg_tokens has no CI because only its mean was
persisted). --no-ci additionally writes a clean variant of the main frontier for slides where the
error bars crowd the labels; it is a second file, never a replacement.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evals.bootstrap_analysis import bootstrap_ci, paired_bootstrap_diff
from inventory import ALIASES, collect, is_degenerate

# ---------------------------------------------------------------- encoding

GATE_MARKER = {"NoGate": "s", "SG": "o", "MG": "D"}
GATE_SIZE = {"NoGate": 9, "SG": 9, "MG": 9.5}

# (light = alone, dark = +Prompt) per technique. Distinct hues, each with enough luminance
# separation between its pair that light/dark is readable in greyscale too.
HUE = {
    "DiM":             ("#8ECAE6", "#01579B"),
    "Conc":            ("#FFCC80", "#B35900"),
    "GradientTrained": ("#8FD9C0", "#00695C"),
    "Clamp":           ("#E3A8C8", "#8C2D5C"),
    "Conc-Matrix":     ("#FFE0A3", "#8A6100"),
    "Conc-SelfProj":   ("#D9C2E9", "#5B2C82"),
}
BASELINE_COLOR = "#222222"

DIR_ORDER = ["DiM", "Conc", "Conc-Matrix", "Conc-SelfProj", "GradientTrained", "Clamp"]
GATE_HATCH = {"NoGate": "", "SG": "//", "MG": "xx"}  # bars can't carry marker shape, so gate
# structure moves to hatch there -- same three-way distinction, same reading.
GATE_ORDER = ["NoGate", "SG", "MG"]


def style():
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 12.5, "axes.titleweight": "semibold", "axes.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#333333", "axes.linewidth": 0.9,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "legend.frameon": False, "legend.fontsize": 9,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def grid(ax, axis="both"):
    ax.grid(axis=axis, alpha=0.16, lw=0.6, color="#444444")
    ax.set_axisbelow(True)


def color_for(direction, condition):
    pair = HUE.get(direction)
    if pair is None:
        return "#888888"
    return pair[1] if condition == "prompt" else pair[0]


# Point labels only -- legends and tables keep the full technique names. "NoGate+GradientTrained"
# is 22 characters and collides with everything near the token ceiling, where several methods
# cluster.
SHORT_DIR = {"GradientTrained": "GradTr", "Conc-SelfProj": "ConcSP", "Conc-Matrix": "ConcMx"}


def label_for(gate, direction, condition):
    alias = ALIASES.get((gate, direction))
    name = alias if alias else f"{gate}+{SHORT_DIR.get(direction, direction)}"
    return f"P+{name}" if condition == "prompt" else name


# ---------------------------------------------------------------- rows

# Canonical-source loader -------------------------------------------------------------------
# token_distributions.json (scripts/regen_token_cis.py + judge_via_batch.py) is preferred over
# the per-script files that inventory.collect() stitches together, for one substantive reason:
# batched bf16 generation is NOT reproducible run to run (observed: the same condition's mean
# moved 0.2-2.2 tokens between two identical invocations). So a token CI from one generation and
# a correctness score from another describe DIFFERENT text. In this file both come from the same
# saved responses, which is the only way the pairing is guaranteed correct -- and it carries
# token CIs, which nothing else has.

CANON = "token_distributions.json"

# printed key -> (gate, direction). The keys are internal variant names, not the Gate+Direction
# convention, so this is where they get translated.
CANON_MAP = {
    "sg": ("SG", "DiM"), "conceptor": ("SG", "Conc"), "proper": ("SG", "GradientTrained"),
    "conceptor_matrix": ("SG", "Conc-Matrix"), "conceptor_selfproj": ("SG", "Conc-SelfProj"),
    "const": ("NoGate", "DiM"), "stolfo": ("NoGate", "Clamp"),
    "mg_gradient_trained_probe_mse": ("MG", "GradientTrained"),
    "mg_gradient_trained_probe_nll": ("MG", "GradientTrained"),
    "mg_dim_probe_mse": ("MG", "DiM"), "mg_dim_probe_nll": ("MG", "DiM"),
    "mg_conc_probe_mse": ("MG", "Conc"), "mg_conc_probe_nll": ("MG", "Conc"),
    "sg_clamp_probe_mse": ("SG", "Clamp"), "sg_clamp_probe_nll": ("SG", "Clamp"),
    "mg_clamp_probe_mse": ("MG", "Clamp"), "mg_clamp_probe_nll": ("MG", "Clamp"),
    "psr_all_layer_probe_mse": ("MG", "GradientTrained"),
    "psr_all_layer_probe_nll": ("MG", "GradientTrained"),
    "psr_multi_gate_probe_mse": ("MG", "DiM"), "psr_multi_gate_probe_nll": ("MG", "DiM"),
    "psr_multi_gate_conceptor_probe_mse": ("MG", "Conc"),
    "psr_multi_gate_conceptor_probe_nll": ("MG", "Conc"),
}


def build_rows_canonical(results_dir: Path):
    path = results_dir / CANON
    if not path.exists():
        return None
    with path.open() as f:
        payload = json.load(f)

    rows, base_rows = [], {}
    for key, e in payload.items():
        if key in ("base", "prompt"):
            base_rows[key] = {"label": key.capitalize(), "tokens": e.get("avg_tokens"),
                               "tokens_lo": e.get("ci_lo"), "tokens_hi": e.get("ci_hi"),
                               "correct": e.get("correct"), "lo": e.get("correct_ci_lo"),
                               "hi": e.get("correct_ci_hi"), "scores": e.get("correctness_scores"),
                               "conc": e.get("conciseness"), "conc_lo": e.get("conciseness_ci_lo"),
                               "conc_hi": e.get("conciseness_ci_hi")}
            continue
        cond = "prompt" if key.endswith("_prompt") else "alone" if key.endswith("_alone") else None
        if cond is None:
            continue
        stem = key[: -len(f"_{cond}")]
        if stem not in CANON_MAP:
            print(f"  NOTE: unmapped condition {key!r} -- add it to CANON_MAP")
            continue
        gate, direction = CANON_MAP[stem]
        loss = "NLL" if stem.endswith("_nll") else "MSE" if stem.endswith("_mse") else ""
        label = label_for(gate, direction, cond) + (f" [{loss}]" if loss else "")
        rows.append({
            "gate": gate, "direction": direction, "condition": cond, "label": label,
            "tokens": e.get("avg_tokens"), "tokens_lo": e.get("ci_lo"), "tokens_hi": e.get("ci_hi"),
            "correct": e.get("correct"), "lo": e.get("correct_ci_lo"), "hi": e.get("correct_ci_hi"),
            "scores": e.get("correctness_scores"),
            "conc": e.get("conciseness"), "conc_lo": e.get("conciseness_ci_lo"),
            "conc_hi": e.get("conciseness_ci_hi"),
            "color": color_for(direction, cond), "marker": GATE_MARKER.get(gate, "o"),
        })
    return rows, base_rows, {}


def build_rows(results_dir: Path):
    cells, outside, baselines, _, surface_twins, degenerate = collect(results_dir)
    if degenerate:
        print(f"  EXCLUDED (degenerate output): {', '.join(degenerate)}")
    rows = []
    for coll in (cells, outside):
        for (gate, direction), conds in coll.items():
            for cond in ("alone", "prompt"):
                e = conds.get(cond)
                if not e or e.get("tokens") is None:
                    continue
                rows.append({
                    "gate": gate, "direction": direction, "condition": cond,
                    "label": label_for(gate, direction, cond),
                    "tokens": e["tokens"], "correct": e.get("correct"),
                    "lo": e.get("lo"), "hi": e.get("hi"), "scores": e.get("scores"),
                    "conc": e.get("conc"), "conc_lo": e.get("conc_lo"), "conc_hi": e.get("conc_hi"),
                    "color": color_for(direction, cond),
                    "marker": GATE_MARKER.get(gate, "o"),
                })
    base_rows = {}
    for k, e in baselines.items():
        if e and e.get("tokens") is not None:
            base_rows[k] = {"label": k.capitalize(), "tokens": e["tokens"],
                             "correct": e.get("correct"), "lo": e.get("lo"), "hi": e.get("hi"),
                             "scores": e.get("scores")}
    return rows, base_rows, surface_twins


def fig_tokens_bars(rows, base_rows, surface_twins, out_dir):
    """Every condition ranked by avg_tokens. This is the figure that works with NO judge budget:
    avg_tokens comes from the local tokenizer, so conditions awaiting correctness still appear
    here rather than being silently dropped the way a 2D frontier has to drop them.

    Horizontal bars because the labels are long method names; hatch encodes gate structure since
    bars cannot carry marker shape, and fill colour keeps the technique/condition encoding
    identical to the scatter figures."""
    items = [r for r in rows if r["tokens"] is not None]
    if not items:
        return None
    for (gate, direction), e in surface_twins.items():
        items.append({"gate": gate, "direction": direction, "condition": "alone",
                       "label": f"{label_for(gate, direction, 'alone')} [resp-only]",
                       "tokens": e["tokens"], "correct": e.get("correct"),
                       "color": color_for(direction, "alone")})
    items.sort(key=lambda r: r["tokens"])

    fig, ax = plt.subplots(figsize=(8.6, 0.34 * len(items) + 1.9))
    ys = np.arange(len(items))
    ax.barh(ys, [r["tokens"] for r in items],
             color=[r["color"] for r in items], edgecolor="#333333", lw=0.7,
             hatch=None, height=0.72)
    for y, r in zip(ys, items):
        h = GATE_HATCH.get(r["gate"], "")
        if h:
            ax.patches[y].set_hatch(h)
        ax.text(r["tokens"] + max(r["tokens"] for r in items) * 0.012, y,
                f"{r['tokens']:.1f}", va="center", fontsize=8.4, color="#333333")
    ax.set_yticks(ys)
    ax.set_yticklabels([r["label"] for r in items], fontsize=8.6)
    ax.invert_yaxis()

    for key, ls, col in (("prompt", (0, (3, 3)), "#222222"), ("base", (0, (1, 2)), "#9E9E9E")):
        b = base_rows.get(key)
        if b:
            ax.axvline(b["tokens"], ls=ls, lw=1.1, color=col, zorder=3)
            ax.annotate(b["label"], (b["tokens"], len(items) - 0.2), textcoords="offset points",
                         xytext=(3, 0), fontsize=8.6, color=col, fontweight="semibold")
    ax.set_xlabel("Average response tokens   (lower = better)")
    ax.set_title("Response length, all conditions")
    grid(ax, "x")

    handles = [plt.Rectangle((0, 0), 1, 1, fc="#CCCCCC", ec="#333333", hatch=GATE_HATCH[g], label=g)
               for g in GATE_ORDER]
    # Outside the axes: the longest bars run to the right edge and the bottom rows are the
    # longest, so any in-axes placement collides with data.
    leg = ax.legend(handles=handles, title="Gate structure", loc="upper left",
                     bbox_to_anchor=(1.01, 1.0), alignment="left")
    leg.get_title().set_fontweight("semibold")
    fig.savefig(out_dir / "fig0_tokens_bars.png")
    plt.close(fig)
    print(f"  wrote {out_dir / 'fig0_tokens_bars.png'}")
    return True


def fig_length_distribution(results_dir, out_dir):
    """Per-response length distributions for the gated-clamp runs. The mean hides real structure
    here: SG+Clamp alone averages ~62 tokens but is BIMODAL -- roughly two thirds already-terse
    plus a third still at full length -- which is what a selective gate should produce and what a
    single mean actively misrepresents. Drawn from the saved raw responses, so no judging needed."""
    p = results_dir / "clamp_gate_eval.json"
    if not p.exists():
        return None
    with p.open() as f:
        payload = json.load(f)

    series = []
    for stem, entry in payload.items():
        gate = "SG" if stem.startswith("sg_") else "MG"
        loss = "NLL" if entry.get("nll_weight") else "MSE"
        for cond in ("alone", "prompt"):
            e = entry.get(cond) or {}
            rs = e.get("responses") or []
            if not rs or is_degenerate(rs):
                continue   # degenerate -- excluded by the same rule the inventory uses
            lengths = [len(r.split()) for r in rs]
            series.append((f"{gate}+Clamp ({loss})\n{'alone' if cond == 'alone' else '+Prompt'}",
                            lengths, color_for("Clamp", cond)))
    if not series:
        return None

    fig, ax = plt.subplots(figsize=(1.7 * len(series) + 2.6, 5.2))
    pos = np.arange(len(series))
    parts = ax.violinplot([s[1] for s in series], positions=pos, widths=0.78,
                           showmeans=False, showmedians=False, showextrema=False)
    for body, (_, _, c) in zip(parts["bodies"], series):
        body.set_facecolor(c)
        body.set_edgecolor("#333333")
        body.set_alpha(0.85)
        body.set_linewidth(0.8)
    bp = ax.boxplot([s[1] for s in series], positions=pos, widths=0.15, showfliers=False,
                     patch_artist=True, medianprops=dict(color="white", lw=1.8),
                     boxprops=dict(facecolor="#333333", ec="#333333"),
                     whiskerprops=dict(color="#333333"), capprops=dict(color="#333333"))
    for i, (_, lengths, _) in enumerate(series):
        ax.scatter([pos[i]], [np.mean(lengths)], marker="D", s=34, color="white",
                    edgecolor="#333333", zorder=5)
    ax.set_xticks(pos)
    ax.set_xticklabels([s[0] for s in series], fontsize=8.6)
    ax.set_ylabel("Response length (words)")
    ax.set_title("Length distributions: the mean hides bimodality\n"
                  "white line = median, white diamond = mean")
    grid(ax, "y")
    fig.savefig(out_dir / "fig7_length_distribution.png")
    plt.close(fig)
    print(f"  wrote {out_dir / 'fig7_length_distribution.png'}")
    return True


# ---------------------------------------------------------------- fig 1: frontier

def fig_frontier(rows, base_rows, out_dir, with_ci=True, x_key="tokens",
                  x_lo="tokens_lo", x_hi="tokens_hi",
                  x_label="Average response tokens   (lower = better)",
                  title="Correctness vs. response length", fname="fig1_frontier_tokens"):
    plottable = [r for r in rows if r["correct"] is not None and r.get(x_key) is not None]
    if not plottable:
        return None
    fig, ax = plt.subplots(figsize=(10.4, 6.6))

    prompt = base_rows.get("prompt")
    if prompt:
        # The "crosshair": everything in the paper is relative to prompting alone, so make that
        # explicit as a reference frame instead of a data point the reader has to locate.
        ax.axhline(prompt["correct"], ls=(0, (3, 3)), lw=1.0, color="#9E9E9E", zorder=0)
        if prompt.get(x_key) is not None:
            ax.axvline(prompt[x_key], ls=(0, (3, 3)), lw=1.0, color="#9E9E9E", zorder=0)

    for name, b in base_rows.items():
        if b["correct"] is None or b.get(x_key) is None:
            continue
        yerr = ([[b["correct"] - b["lo"]], [b["hi"] - b["correct"]]]
                if with_ci and b.get("lo") is not None else None)
        ax.errorbar(b[x_key], b["correct"], yerr=yerr, fmt="^", ms=11,
                     mfc="white", mec=BASELINE_COLOR, ecolor=BASELINE_COLOR,
                     mew=1.6, lw=1.1, capsize=2.5, zorder=5)
        ax.annotate(b["label"], (b[x_key], b["correct"]), textcoords="offset points",
                     xytext=(11, -4), fontsize=9.5, fontweight="semibold", color=BASELINE_COLOR)

    for r in plottable:
        yerr = ([[r["correct"] - r["lo"]], [r["hi"] - r["correct"]]]
                if with_ci and r.get("lo") is not None else None)
        xerr = ([[r[x_key] - r[x_lo]], [r[x_hi] - r[x_key]]]
                if with_ci and r.get(x_lo) is not None and r.get(x_hi) is not None else None)
        ax.errorbar(r[x_key], r["correct"], xerr=xerr, yerr=yerr, fmt=r["marker"],
                     ms=GATE_SIZE.get(r["gate"], 9), mfc=r["color"], mec="#333333", mew=0.8,
                     ecolor=r["color"], lw=1.2, capsize=2.5, alpha=0.95, zorder=4)

    ax.set_xlabel(x_label)
    ax.set_ylabel("Correctness, LLM-judged 0–2   (higher = better)")
    ax.set_title(title + ("" if with_ci else "  (point estimates)"))

    xs = [r[x_key] for r in plottable] + [b[x_key] for b in base_rows.values() if b.get(x_key) is not None]
    span = max(xs) - min(xs)
    ax.set_xlim(min(xs) - span * 0.20, max(xs) + span * 0.20)
    ys = [r["correct"] for r in plottable] + [b["correct"] for b in base_rows.values() if b["correct"]]
    los = [r["lo"] for r in plottable if r.get("lo")] or ys
    his = [r["hi"] for r in plottable if r.get("hi")] or ys
    lo, hi = min(min(ys), min(los)), max(max(ys), max(his))
    pad = (hi - lo) * 0.16 or 0.05
    ax.set_ylim(lo - pad, min(2.0, hi + pad) + (hi - lo) * 0.12)
    grid(ax)

    # Two legends: one per channel. A single combined legend would need gate x technique x
    # condition entries and become a table.
    shape_handles = [plt.Line2D([], [], ls="none", marker=GATE_MARKER[g], ms=9,
                                 mfc="#BDBDBD", mec="#333333", label=g) for g in GATE_ORDER
                     if any(r["gate"] == g for r in plottable)]
    shape_handles.append(plt.Line2D([], [], ls="none", marker="^", ms=10, mfc="white",
                                     mec=BASELINE_COLOR, label="Baseline"))
    hue_handles = []
    for d in DIR_ORDER:
        if not any(r["direction"] == d for r in plottable):
            continue
        hue_handles.append(plt.Line2D([], [], ls="none", marker="o", ms=9, mfc=HUE[d][0],
                                       mec="#333333", label=f"{d} (alone)"))
        if any(r["direction"] == d and r["condition"] == "prompt" for r in plottable):
            hue_handles.append(plt.Line2D([], [], ls="none", marker="o", ms=9, mfc=HUE[d][1],
                                           mec="#333333", label=f"{d} (+Prompt)"))
    l1 = ax.legend(handles=shape_handles, title="Gate structure  (shape)", loc="upper left",
                    bbox_to_anchor=(1.01, 1.0), alignment="left")
    l1.get_title().set_fontweight("semibold")
    l2 = ax.legend(handles=hue_handles, title="Intervention  (colour)", loc="upper left",
                    bbox_to_anchor=(1.01, 0.70), alignment="left")
    l2.get_title().set_fontweight("semibold")
    ax.add_artist(l1)
    # Points are unlabelled by design -- shape x colour x shade identifies each one uniquely, and
    # per-point text made the near-ceiling cluster unreadable. The composition rule and the paper
    # aliases are spelled out here so the legend is genuinely sufficient on its own.
    present_aliases = [f"{g}+{d} = {ALIASES[(g, d)]}" for (g, d) in ALIASES
                        if any(r["gate"] == g and r["direction"] == d for r in plottable)]
    key_lines = ["Read each point as", "shape + colour + shade:", "", "light fill  = steering alone",
                  "dark fill   = Prompt + steering"]
    if present_aliases:
        key_lines += ["", "Paper names:"] + ["  " + a for a in present_aliases]
    ax.text(1.02, 0.30, "\n".join(key_lines), transform=ax.transAxes, va="top", ha="left",
            fontsize=8.2, color="#444444", linespacing=1.5)
    note = "95% bootstrap CI" if with_ci else "point estimates; CIs omitted for legibility"
    # Inside the axes, bottom-right: the right margin is fully occupied by the two legends and the
    # reading key, so an out-of-axes note collides with the alias block.
    ax.text(0.99, 0.015, note, transform=ax.transAxes, fontsize=8, color="#999999",
            ha="right", va="bottom")

    name = f"{fname}.png" if with_ci else f"{fname}_noCI.png"
    fig.savefig(out_dir / name)
    plt.close(fig)
    print(f"  wrote {out_dir / name}")
    return out_dir / name


# ---------------------------------------------------------------- fig 2: gate ladder

def fig_gate_ladder(rows, base_rows, out_dir):
    """Gate capacity at fixed technique, as connected slopes. A slope chart rather than grouped
    bars because the claim is about the SHAPE of the NoGate -> SG -> MG progression within each
    technique, and bars force that comparison across non-adjacent groups."""
    alone = [r for r in rows if r["condition"] == "alone"]
    by_dir = {}
    for r in alone:
        by_dir.setdefault(r["direction"], {})[r["gate"]] = r
    by_dir = {d: v for d, v in by_dir.items() if len(v) >= 2}
    if not by_dir:
        return None

    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    xpos = {g: i for i, g in enumerate(GATE_ORDER)}
    for d in DIR_ORDER:
        if d not in by_dir:
            continue
        pts = [(xpos[g], by_dir[d][g]) for g in GATE_ORDER if g in by_dir[d]]
        ax.plot([p[0] for p in pts], [p[1]["tokens"] for p in pts], "-",
                 lw=2.0, color=HUE[d][1], alpha=0.85, zorder=2)
        for x, r in pts:
            ax.plot(x, r["tokens"], r["marker"], ms=11, mfc=HUE[d][0], mec=HUE[d][1],
                     mew=1.6, zorder=3)
        ax.annotate(f" {d}", (pts[-1][0], pts[-1][1]["tokens"]), textcoords="offset points",
                     xytext=(9, 0), fontsize=9.5, color=HUE[d][1], fontweight="semibold",
                     va="center")

    if base_rows.get("prompt"):
        ax.axhline(base_rows["prompt"]["tokens"], ls=(0, (3, 3)), lw=1.0, color="#9E9E9E")
        ax.annotate("Prompt alone", (-0.35, base_rows["prompt"]["tokens"]),
                     textcoords="offset points", xytext=(0, 5), fontsize=8.6, color="#666666")
    if base_rows.get("base"):
        ax.axhline(base_rows["base"]["tokens"], ls=(0, (1, 2)), lw=1.0, color="#BDBDBD")
        ax.annotate("Base (no steering)", (-0.35, base_rows["base"]["tokens"]),
                     textcoords="offset points", xytext=(0, 5), fontsize=8.6, color="#999999")

    ax.set_xticks(list(xpos.values()))
    ax.set_xticklabels([f"{g}\n({GATE_MARKER[g]})" if False else g for g in GATE_ORDER])
    ax.set_xlim(-0.45, len(GATE_ORDER) - 1 + 0.75)
    ax.set_xlabel("Gate structure")
    ax.set_ylabel("Average response tokens   (lower = better)")
    ax.set_title("Ablation: gate capacity at fixed intervention\n(steering alone, no prompt)")
    grid(ax, "y")
    fig.savefig(out_dir / "fig2_gate_ladder.png")
    plt.close(fig)
    print(f"  wrote {out_dir / 'fig2_gate_ladder.png'}")
    return True


# ---------------------------------------------------------------- fig 3: surface 2x2

def fig_surface(results_dir, out_dir):
    p = results_dir / "surface_ablation.json"
    if not p.exists():
        return None
    with p.open() as f:
        payload = json.load(f)

    forms = {"additive": "DiM", "clamp": "Clamp"}
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    xs = [0, 1]
    for variant, v in payload.items():
        form = v.get("form", "additive")
        hue = HUE[forms.get(form, "DiM")]
        s = v["surfaces"]
        ys = [s["all_positions"]["avg_tokens"], s["response_only"]["avg_tokens"]]
        ax.plot(xs, ys, "-", lw=2.2, color=hue[1], zorder=2)
        for x, y, key in zip(xs, ys, ("all_positions", "response_only")):
            e = s[key]
            yerr = None
            if e.get("ci_lo") is not None:
                pass  # CIs here are on correctness, not tokens -- not drawable on this axis
            ax.plot(x, y, "s" if form == "additive" else "P", ms=13, mfc=hue[0], mec=hue[1],
                     mew=1.8, zorder=3)
            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 13),
                         ha="center", fontsize=9.5, fontweight="semibold", color=hue[1])
        delta = ys[1] - ys[0]
        ax.annotate(f"{form}   ({delta:+.1f} tok)", (xs[-1], ys[-1]), textcoords="offset points",
                     xytext=(14, 0), fontsize=10, color=hue[1], fontweight="semibold", va="center")

    ax.set_xticks(xs)
    ax.set_xticklabels(["All positions\n(prompt + response)", "Response only\n(PSR's surface)"])
    ax.set_xlim(-0.3, 1.55)
    ax.set_ylabel("Average response tokens   (lower = better)")
    ax.set_title("Ablation: intervention surface, configuration held fixed\n"
                  "same layer, same coefficient, only the surface differs")
    grid(ax, "y")
    ax.text(0.99, 0.02, "no judge calls needed -- avg_tokens is computed locally",
            transform=ax.transAxes, ha="right", fontsize=8, color="#777777")
    fig.savefig(out_dir / "fig3_surface_ablation.png")
    plt.close(fig)
    print(f"  wrote {out_dir / 'fig3_surface_ablation.png'}")
    return True


# ---------------------------------------------------------------- fig 4: forest

def fig_forest(rows, base_rows, out_dir):
    prompt = base_rows.get("prompt")
    if not prompt or not prompt.get("scores"):
        return None
    cand = [r for r in rows if r.get("scores")]
    if not cand:
        return None
    for r in cand:
        d, lo, hi = paired_bootstrap_diff(r["scores"], prompt["scores"])
        r["d"], r["dlo"], r["dhi"] = d, lo, hi
    cand.sort(key=lambda r: r["d"])

    fig, ax = plt.subplots(figsize=(8.0, 0.40 * len(cand) + 2.1))
    for y, r in enumerate(cand):
        sig = r["dlo"] > 0 or r["dhi"] < 0
        ax.errorbar(r["d"], y, xerr=[[r["d"] - r["dlo"]], [r["dhi"] - r["d"]]],
                     fmt=r["marker"], ms=GATE_SIZE.get(r["gate"], 9),
                     mfc=r["color"] if sig else "white", mec=r["color"], mew=1.5,
                     ecolor=r["color"], lw=1.2, capsize=2.5)
    ax.axvline(0, color="#222222", lw=1.1)
    ax.set_yticks(range(len(cand)))
    ax.set_yticklabels([r["label"] for r in cand], fontsize=9)
    ax.set_xlabel("Δ correctness vs Prompt alone   (negative = worse than Prompt)")
    ax.set_title("Correctness relative to prompting alone")
    grid(ax, "x")
    ax.text(1.0, -0.13, "95% paired bootstrap CI; hollow marker = interval spans zero",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#777777")
    fig.savefig(out_dir / "fig4_forest.png")
    plt.close(fig)
    print(f"  wrote {out_dir / 'fig4_forest.png'}")
    return True


# ---------------------------------------------------------------- fig 5: cosine heatmap

def fig_cosine(results_dir, out_dir):
    p = results_dir / "cosine_similarity_matrix.json"
    if not p.exists():
        return None
    with p.open() as f:
        payload = json.load(f)
    names, mat = payload.get("names", []), np.array(payload.get("matrix", []), dtype=float)
    if not names or mat.size == 0:
        return None

    n = len(names)
    fig, ax = plt.subplots(figsize=(1.05 * n + 3.6, 0.92 * n + 2.6))
    # Diverging map centred at zero: cosine is signed, and "unrelated" should read as neutral
    # rather than as a low value on a sequential ramp.
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(n):
        for j in range(n):
            v = mat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.6,
                    fontweight="semibold" if i != j else "normal",
                    color="white" if abs(v) > 0.58 else "#1A1A1A")
    short = [s.replace("_probe", "").replace("psr_", "") for s in names]
    ax.set_xticks(range(n)); ax.set_xticklabels(short, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(n)); ax.set_yticklabels(short, fontsize=9)
    ax.set_title("Steering direction cosine similarity")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("cosine similarity", fontsize=9)
    cb.outline.set_visible(False)
    d = payload.get("dim", 3584)
    ax.text(1.0, -0.14, f"d={d}; |cos| below ~0.03 is indistinguishable from random",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#777777")
    fig.savefig(out_dir / "fig5_cosine_heatmap.png")
    plt.close(fig)
    print(f"  wrote {out_dir / 'fig5_cosine_heatmap.png'}")
    return True


# ---------------------------------------------------------------- fig 6: layer sweeps

def _read_jsonl(p):
    if not p.exists():
        return []
    with p.open() as f:
        return [json.loads(l) for l in f if l.strip()]


def fig_layer_sweep(results_dir, out_dir):
    stems = {
        "sg_gradient_trained": ("GradientTrained", "SG"), "proper": ("GradientTrained", "SG"),
        "sg_conc": ("Conc", "SG"), "conceptor": ("Conc", "SG"),
        "sg_dim": ("DiM", "SG"), "sg": ("DiM", "SG"),
        "nogate_dim": ("DiM", "NoGate"), "const": ("DiM", "NoGate"),
        "nogate_clamp": ("Clamp", "NoGate"), "stolfo": ("Clamp", "NoGate"),
    }
    seen, series = set(), []
    for stem, (direction, gate) in stems.items():
        rows = [r for r in _read_jsonl(results_dir / f"{stem}_tiered_search.jsonl")
                if r.get("tier") == "tier1" and not r.get("skipped")]
        if not rows or (direction, gate) in seen:
            continue
        seen.add((direction, gate))
        best = {}
        for r in rows:   # several coefficients per layer for additive -- keep the best per layer
            l = r["layer"]
            if l not in best or r["avg_tokens"] < best[l]["avg_tokens"]:
                best[l] = r
        series.append((direction, gate, [best[l] for l in sorted(best)]))
    if not series:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6), sharex=True)
    for direction, gate, rows in series:
        layers = [r["layer"] for r in rows]
        c = HUE[direction][1]
        lbl = f"{gate}+{direction}"
        axes[0].plot(layers, [r["avg_tokens"] for r in rows], "-", color=c, lw=1.8,
                      marker=GATE_MARKER[gate], ms=5.5, mfc=HUE[direction][0], mec=c, label=lbl)
        corr = [r.get("judged_score") for r in rows]
        if all(x is not None for x in corr):
            axes[1].plot(layers, corr, "-", color=c, lw=1.8, marker=GATE_MARKER[gate],
                          ms=5.5, mfc=HUE[direction][0], mec=c, label=lbl)
            lo = [r.get("ci_lo", r["judged_score"]) for r in rows]
            hi = [r.get("ci_hi", r["judged_score"]) for r in rows]
            axes[1].fill_between(layers, lo, hi, color=c, alpha=0.10, lw=0)

    anyr = series[0][2][0]
    if anyr.get("prompt_avg_tokens"):
        axes[0].axhline(anyr["prompt_avg_tokens"], ls=(0, (3, 3)), lw=1.0, color="#9E9E9E")
        axes[0].annotate("Prompt alone", (min(l for _, _, rs in series for l in [rs[0]["layer"]]),
                          anyr["prompt_avg_tokens"]), textcoords="offset points", xytext=(2, 5),
                          fontsize=8.6, color="#666666")
    axes[0].set_ylabel("Average response tokens")
    axes[0].set_title("Length reduction by layer")
    axes[1].set_ylabel("Correctness (0–2)")
    axes[1].set_title("Correctness by layer  (shaded = 95% CI)")
    axes[1].set_ylim(top=2.03)
    for a in axes:
        a.set_xlabel("Injection layer")
        grid(a, "y")
    axes[0].legend(loc="best", fontsize=8.6)
    fig.suptitle("Layer sweeps (Tier 1, judged, n=20)", y=1.02, fontsize=12.5, fontweight="semibold")
    fig.savefig(out_dir / "fig6_layer_sweep.png")
    plt.close(fig)
    print(f"  wrote {out_dir / 'fig6_layer_sweep.png'}")
    return True


# ---------------------------------------------------------------- main

def main(task: str, no_ci: bool) -> None:
    from adapters.registry import get_adapter
    results_dir = get_adapter(task).RESULTS_DIR
    out_dir = Path("figures") / task
    out_dir.mkdir(parents=True, exist_ok=True)
    style()

    canon = build_rows_canonical(results_dir)
    if canon is not None:
        rows, base_rows, surface_twins = canon
        print(f"  source: {CANON} (tokens+correctness from the SAME responses, both with CIs)")
    else:
        rows, base_rows, surface_twins = build_rows(results_dir)
        print(f"  source: per-script files via inventory.collect() -- no token CIs available. "
              f"Run scripts/regen_token_cis.py for {CANON}.")
    print(f"\n{len(rows)} method-conditions, {len(base_rows)} baselines from {results_dir}")
    print("\nbuilding figures:")

    if not fig_tokens_bars(rows, base_rows, surface_twins, out_dir):
        print("  SKIP fig0: no conditions with avg_tokens")
    if not fig_frontier(rows, base_rows, out_dir, with_ci=True):
        print("  SKIP fig1: no conditions with correctness scores")
    if no_ci:
        fig_frontier(rows, base_rows, out_dir, with_ci=False)

    # Correctness vs JUDGED CONCISENESS. Only the tiered searches persist this (as optimize_score
    # with a CI); the all-layer and clamp evals never stored per-example conciseness, so those
    # cells are absent rather than imputed. Stated explicitly when it happens.
    have_conc = [r for r in rows if r.get("conc") is not None and r["correct"] is not None]
    if have_conc:
        fig_frontier(rows, base_rows, out_dir, with_ci=True, x_key="conc",
                      x_lo="conc_lo", x_hi="conc_hi",
                      x_label="Judged conciseness, 0–2   (higher = better)",
                      title="Correctness vs. judged conciseness",
                      fname="fig1c_frontier_conciseness")
        missing = sorted({r["label"] for r in rows
                          if r["correct"] is not None and r.get("conc") is None})
        if missing:
            print(f"    note: {len(missing)} judged condition(s) have no stored conciseness "
                  f"and are absent from fig1c: {', '.join(missing[:6])}"
                  + (" ..." if len(missing) > 6 else ""))
        if no_ci:
            fig_frontier(rows, base_rows, out_dir, with_ci=False, x_key="conc",
                          x_lo="conc_lo", x_hi="conc_hi",
                          x_label="Judged conciseness, 0–2   (higher = better)",
                          title="Correctness vs. judged conciseness",
                          fname="fig1c_frontier_conciseness")
    else:
        print("  SKIP fig1c: no condition has a stored judged-conciseness score "
              "(only *_tiered_search.jsonl finals carry optimize_score)")
    if not fig_gate_ladder(rows, base_rows, out_dir):
        print("  SKIP fig2: need >=2 gate levels at one technique")
    if not fig_surface(results_dir, out_dir):
        print("  SKIP fig3: run scripts/surface_ablation.py")
    if not fig_forest(rows, base_rows, out_dir):
        print("  SKIP fig4: needs per-example scores + a Prompt baseline")
    if not fig_cosine(results_dir, out_dir):
        print("  SKIP fig5: run scripts/compute_cosine_similarities.py")
    if not fig_length_distribution(results_dir, out_dir):
        print("  SKIP fig7: no clamp_gate_eval.json with usable responses")
    if not fig_layer_sweep(results_dir, out_dir):
        print("  SKIP fig6: no tier1 rows in any *_tiered_search.jsonl")

    print(f"\nall figures in {out_dir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="caveman", choices=["caveman", "ifeval"])
    ap.add_argument("--no-ci", action="store_true",
                     help="also write a CI-free variant of the main frontier")
    a = ap.parse_args()
    main(a.task, a.no_ci)
