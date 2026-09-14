# Extension Changes: NLL loss, participation ratio, sweeps, plotting, conciseness score, overnight run

Applies on top of `psr_refactor_v10.tar.gz`. Verified with the real (non-mocked) test suite --
`python3 -m pytest tests/ -v` -- **76 passed** (22 original + 54 new), using `tests/fakes.py`'s
fake model and a fake OpenAI-shaped client, no GPU/real weights/API key needed. Ran this for real
in a sandboxed environment before every delivery, not just written-but-untested.

## Round 1: auxiliary NLL loss, participation ratio, sweeps, plotting

### NLL loss (Eq. 4, Heyman & Vandeputte 2026)
- **New:** `steering/psr/nll.py` -- `response_nll(logits, input_ids, n_resp)`, the literal equation.
- **Changed:** `steering/psr/gate.py`, `conceptor/matrix/logic.py`, `conceptor/selfproj/logic.py`
  -- `forward_with_gate_hook` now returns `(hidden_states, logits, fit)`, not `(hidden_states, fit)`.
- **Changed:** `steering/psr/training_loop.py` -- `train_gate(..., nll_weight=0.0, ...)`, loss is
  `mse + reg + nll_weight * nll`. Returns `({"mse","nll"}, {"mse","nll"})` dicts, not two floats.
- **Changed:** all four trainable variants' `train.py` -- new `PSR_<VARIANT>_NLL_WEIGHT` env var,
  default `0.0` (every existing result reproduces unchanged unless set).
- Verified end-to-end that `nll_weight=1.0` vs `0.0` from the same seed gives different trained
  gate weights -- the term actually participates in the gradient.

### Participation ratio (matrix-based methods only)
- **New:** `steering/psr/conceptor/rank_diagnostic.py` -- PR = (sum lambda)^2 / sum(lambda^2).
- Deliberately only for conceptor/matrix and conceptor/selfproj (C applied fresh every hidden
  state), NOT the fixed-vector conceptor variant (correction stays rank-1 regardless of C's PR).
- **Changed:** `src/generate.py` attaches it to matrix-based conditions' rows via `getattr`, no
  hardcoded condition list. `evals/run_judge.py`'s metadata carry-over generalized from a
  `_tokens`-only special case to any `{cond}_*` field -- **caught and fixed a real bug**: condition
  names that are prefixes of others (`"psr"` vs `"psr_proper"`) could cross-contaminate; fixed via
  longest-name-first + a claimed-keys set, regression-tested.
- **Changed:** `evals/summarize.py`/`bootstrap_analysis.py`/`plotting.py` all read this through one
  shared `evals.summarize.collect_raw_values_by_cond_field`.

### Layer + hyperparameter sweeps
- **New:** `core/sweep.py` -- generic, resumable grid-sweep driver with best-point tracking.
- **Changed:** all four trainable variants get a `sweep()` (`--sweep` CLI flag) alongside the
  unchanged `main()`, sharing a `train_one_config(...)` so the two paths can't drift apart.
  - proper/conceptor/matrix: layer x nll_weight (conceptor variants also x alpha).
  - selfproj: layer x nll_weight, with alpha re-derived from each layer's OWN eigenvalue spectrum
    (adaptive percentiles), not shared across layers.
  - Every sweep writes `results/<task>/psr_<variant>_sweep.jsonl` + the same `..._probe.pt`
    `main()` would for the best point.
  - Stated tradeoff: if a sweep resumes after a crash and the winner was completed in an earlier
    session, its tensors aren't in memory -- `sweep()` prints an explicit instruction rather than
    writing a wrong/missing checkpoint.
- **Changed:** `infra/run_gpu_pipeline.sh` -- `USE_SWEEP=1` (default off) runs `--sweep` instead of
  a single pre-chosen layer.

### Visualization
- **New:** `evals/plotting.py` -- score-comparison bars (95% CI), compression/quality frontier
  (95% CI both axes), layer-sweep lines, participation-ratio-vs-metric scatter. All reuse
  `bootstrap_ci`/the shared collector rather than re-deriving numbers.
- Fills the "plotting regression" flagged in the original handoff (summarize.py used to only
  print a table).

## Round 2: caveman conciseness score

- **Changed:** `evals/caveman/judge.py` -- `SCORE_FIELDS = ["correct", "coherent", "conciseness"]`.
  Conciseness is a separate API call, separate rubric, from correctness/coherence -- deliberately
  not folded in: correctness needs the reference explanation as ground truth, conciseness doesn't
  (grading against a reference would conflate "matches the reference" with "isn't padded"); and a
  failure in one call doesn't block the other's score for the same response.
- Rubric gives the judge a concrete caveman persona (0 = padded/hedging/repetitive, 1 = tight but
  one avoidable wordy phrase, 2 = every word does work) instead of an open-ended "is this
  concise?" question, since there's no reference to check conciseness against.
- On top of, not instead of, per-condition token counts.
- Zero changes needed to `run_judge.py`/`summarize.py`/`bootstrap_analysis.py` -- generic over
  whatever `SCORE_FIELDS` a judge returns, the whole point of the registry pattern.
- `evals/plotting.py`'s `plot_judged_results` produces an extra compression-vs-conciseness
  frontier plot when that field is present.
- **Assumption flagged, not silently decided:** kept `"coherent"` as a third axis rather than
  dropping it (your message read as exactly two axes) -- removing it would lose the signal that
  caught `Prompt+Const`'s coherent-but-vague failure mode in the original findings. Say so if you
  want it gone; one-line removal.

## Round 3: gpt-4o-mini for conciseness, `plot_everything`, unattended overnight run

- **Changed:** `evals/caveman/judge.py` -- `CONCISENESS_JUDGE_MODEL = "gpt-4o-mini"` (matches your
  local edit; was `"gpt-4o"` in the previous round). Kept as its own constant, separate from
  `JUDGE_MODEL`, so it can be pointed at a stronger model later without touching correctness/
  coherence. Fixed the corresponding tests: they previously keyed a fake client's replies by
  model name, which silently breaks now that both constants are the same string (two different
  expected replies collapse to one key) -- rewrote the fake client to consume replies in CALL
  ORDER instead, which is correct regardless of whether the two models happen to match.
- **New, in `evals/plotting.py`:** `discover_group_keys(sweep_rows)` -- auto-detects which keys in
  a sweep JSONL's rows are swept hyperparameters (vs. fixed bookkeeping fields like `layer`,
  `final_mse`, `participation_ratio`) purely from the data, no hardcoded per-variant list.
  `plot_sweep_file_everything(sweep_path, out_dir)` -- for every metric actually present
  (`final_mse`, `final_nll`) and every discovered hyperparameter, produces an ungrouped
  layer-sweep plot AND a grouped-by-that-hyperparameter plot, plus a PR-vs-metric scatter per
  metric where `participation_ratio` is present. `plot_everything(task, split)` -- combines the
  judged-result plots with `plot_sweep_file_everything` over every `*_sweep.jsonl` actually found
  in `results/<task>/`, however many that is. New `everything` CLI subcommand.
  - Verified against a full simulation of tonight's actual grid shapes (all four variants' real
    default layer/alpha/nll_weight grids): **33 PNGs**, comfortably past "20+."
- **New:** `infra/run_caveman_overnight.sh` -- the actual thing you asked for. Pre-flight checks
  (openai.key, CUDA, deps) that fail FAST before touching the GPU; S-PSR at layer 14 + A-PSR;
  every trainable variant's full `--sweep`; generate -> judge -> summarize -> bootstrap -> plot
  everything; a resilient `run_step` wrapper that logs and continues past a failure instead of
  aborting the whole night (every sweep is independently resumable, so rerunning tomorrow picks
  up where anything failed); a final summary (elapsed time, plot count, failed steps, log path).
  Deliberately skips const-steering calibration (needs a human judgment call, always has) and
  ifeval (caveman only, per this round's ask). See `OVERNIGHT_RUN.md` for how to actually run it
  (tmux/nohup so it survives you disconnecting) and the full grid-size breakdown (~832 training
  runs across the four variants at full default grids).
  - Dry-run tested the script's own control flow (pre-flight branches, failure bookkeeping,
    final summary) with the real training/judging commands stubbed out and one deliberately
    forced to fail, to confirm the bash logic itself (not just the Python underneath) is correct
    before handing it over.

## Testing added, all three rounds combined

54 new tests across 13 files, including three end-to-end fake-model integration tests (one per
proper/conceptor-matrix/conceptor-selfproj) that exercise the full train_gate loop with the new
combined loss, a fake-OpenAI-client-driven suite for the judge's dual-call logic, and a full
synthetic-data simulation of the plotting auto-discovery at the exact scale tonight's real run
will produce.

## Exact next steps

1. Apply this on top of your real repo (excludes `data/`, `results/`, `third_party/` -- unchanged).
2. `python3 -m pytest tests/ -v` on your GPU pod (should show 76 passing).
3. Read `OVERNIGHT_RUN.md`, confirm `openai.key` exists, start it in `tmux` or with `nohup` --
   **not** a plain foreground command you might disconnect from.
4. In the morning: `results/caveman/plots/*.png` for the graphs, `results/caveman/overnight_*.log`
   for exactly what happened and what (if anything) failed and needs a rerun.
