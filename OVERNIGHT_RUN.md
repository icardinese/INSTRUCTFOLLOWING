# Running the full overnight caveman sweep

## Before you start it

1. Apply this tarball on top of your real repo (excludes `data/`, `results/`, `third_party/` --
   unchanged, keep your existing copies).
2. Make sure `openai.key` exists at the repo root (plain text file, just the key). The judge now
   makes TWO calls per response (correctness/coherence, then conciseness) -- both on
   `gpt-4o-mini` as you set it, so cost is still low, but the call COUNT roughly doubles versus
   before this session.
3. **Run it somewhere that survives you disconnecting.** A plain `bash infra/run_caveman_overnight.sh`
   in a foreground SSH session dies the instant your laptop sleeps or the connection drops. Use:
   ```bash
   tmux new -s overnight
   # inside the tmux session:
   bash infra/run_caveman_overnight.sh
   # detach with Ctrl-b then d -- it keeps running. Reattach anytime with: tmux attach -t overnight
   ```
   or:
   ```bash
   nohup bash infra/run_caveman_overnight.sh > /dev/null 2>&1 &
   ```
   The script tees its own output to `results/caveman/overnight_<timestamp>.log` either way, so
   you have a real log to check in the morning regardless of which you use.

## What it actually runs

Caveman only. In order:

1. Pre-flight checks (openai.key present, CUDA available, deps importable) -- **fails fast, before
   touching the GPU**, if any of these are wrong. Nothing is trained if this step fails.
2. S-PSR baseline at layer 14 (this project's own earlier full layer sweep found 14 near-optimal
   for PSR-Proper -- used here as a reasonable single-layer default for the one variant that isn't
   itself being swept this round).
3. A-PSR baseline (automatic, its own multi-layer set, no layer argument needed).
4. **Every trainable variant's full sweep**, using each script's full default grid (nothing
   throttled):
   - PSR-Proper: 13 layers (2 to 26, step 2) x 4 `nll_weight` values = 52 training runs.
   - Conceptor (fixed-vector): 13 layers x 5 `alpha` values x 4 `nll_weight` values = 260 runs.
   - Conceptor/matrix: same grid = 260 runs.
   - Conceptor/selfproj: 13 layers x 5 adaptive-alpha percentiles x 4 `nll_weight` values = 260
     runs (alpha itself is re-derived from each layer's own eigenvalue spectrum, not shared).
   - **~832 individual training runs total.** I can't tell you exactly how long this takes on your
     GPU without knowing your dataset size and hardware -- that's the actual meaning of "10-12
     hours, idc" you gave me room for. If it's running long and you want to check on it without
     stopping it, `tmux attach -t overnight` or `tail -f results/caveman/overnight_*.log`.
5. Generate every available condition's responses (`src/generate.py`).
6. Judge every response -- correctness + coherence + conciseness (`evals/run_judge.py`).
7. Summarize + bootstrap 95% CIs (`evals/summarize.py`, `evals/bootstrap_analysis.py`).
8. **Every plot we can make** (`evals/plotting.py everything`) -- judged-result plots (score
   comparison per field, compression-vs-correctness frontier, compression-vs-conciseness
   frontier) plus, for EVERY sweep file that exists, a layer-vs-metric plot ungrouped and grouped
   by every hyperparameter that sweep varied, for both `final_mse` and `final_nll`, plus a
   participation-ratio-vs-metric scatter for the two matrix-based variants. On a simulated version
   of tonight's exact grid shapes this produced **33 PNGs** -- comfortably past "20+."

## What it deliberately skips (and why)

- **const steering calibration.** Picking its winning (layer, coefficient) needs a human to judge
  `sweep_dev.jsonl` -- always has, still does. `generate.py` already skips the `"const"` condition
  gracefully if `const_steer_config.json` doesn't exist, so nothing downstream breaks. Run
  `src/const/sweep.py` and write that config by hand separately if you want it included later.
- **ifeval.** Caveman only, per tonight's ask.

## If something fails partway through

Every step continues past a failure rather than aborting the whole run (deliberately not `set -e`
for the long section) -- a single crashed sweep point or a transient OpenAI API error costs you
that one step, not the other ~7+ hours of results. The final summary lists exactly which steps
failed. Every `*_sweep.jsonl` is independently resumable (`core/sweep.py`'s own resumability), so
**rerunning this exact same script tomorrow picks up where anything failed**, not from scratch --
already-completed grid points are skipped automatically.

## Shrinking the grid for a quick smoke test first

If you want to sanity-check the whole pipeline end-to-end on a tiny grid before committing to the
full ~832-run night:

```bash
SWEEP_LAYERS=10,14,18 SWEEP_ALPHAS=2,8 SWEEP_NLL_WEIGHTS=0,0.1 bash infra/run_caveman_overnight.sh
```

This overrides the layer/alpha/nll_weight grids for every variant's sweep in one shot (3 layers x
2 alphas x 2 nll_weights = 12 runs per conceptor variant, 6 for proper) without touching the
script itself.
