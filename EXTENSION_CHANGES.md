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

## Round 4: mse_weight -- pure MSE and pure NLL as true mutually-exclusive alternatives

Prompted by a direct question: does H&V's actual paper train MSE and loglikelihood additively, or
as separate alternatives? Checked against the real paper (Section 3.5): **alternatives** --
"Loglikelihood (LL). **As an alternative to MSE**..." Every results table in the paper reports
`_MSE` or `_LL` variants, never a combined one. The additive design from round 1
(`mse + reg + nll_weight * nll`) was faithful to *this project's own draft* ("we incorporate an
auxiliary log-likelihood objective... integrating this into the total loss"), which is a different,
additive proposal on top of H&V, not a replication of their ablation.

- **Changed:** `steering/psr/training_loop.py` -- `train_gate` now takes an independent
  `mse_weight` (default `1.0`) alongside `nll_weight` (default `0.0`). Loss is
  `mse_weight * MSE + reg + nll_weight * NLL`. `mse_weight=0.0` means the MSE term contributes
  **zero** gradient, not a small one -- verified in `tests/test_training_loop_loss_weights.py` by
  training pure-MSE vs. pure-NLL from the same seed and confirming they diverge.
- **Changed:** all four trainable variants -- new `PSR_<VARIANT>_MSE_WEIGHT` env var (default
  `1.0`, so nothing changes unless set), and the flat `nll_weight` sweep grid was replaced with a
  small set of **paired** `(mse_weight, nll_weight)` loss configurations (5 points: pure MSE,
  three MSE+NLL blends, pure NLL) rather than a full 2D cartesian product -- most
  `(mse_weight, nll_weight)` combinations aren't meaningful experiments (e.g. `mse_weight=0`
  with a small `nll_weight` has almost no training signal at all). New `--loss-configs
  "mse_weight:nll_weight,..."` CLI override on every variant's `--sweep` mode.
- **Changed:** `infra/run_caveman_overnight.sh` -- `SWEEP_NLL_WEIGHTS` env var replaced with
  `LOSS_CONFIGS` (same `mse_weight:nll_weight` pair format); total grid is now ~1040 runs (was
  ~832), since every variant now sweeps 5 loss configs instead of 4.
- No change needed in `evals/plotting.py` -- `mse_weight` is automatically picked up by
  `discover_group_keys` as another hyperparameter to plot, same as `alpha`/`nll_weight` already
  were, since it isn't in the fixed exclusion set. More plots per variant as a result.
- Diagnosed a real overnight-run crash report from `steering/psr/gate.py`'s `coefficient()` --
  traceback was truncated before the actual exception message, so the root cause is still
  unconfirmed; flagged for the user to supply the missing final lines before restarting.

## Round 5: the real overnight crash -- decoder layer output shape (transformers version compat)

Root cause, confirmed from the real traceback: `AttributeError: 'tuple' object has no attribute
'dtype'` inside Qwen2's decoder forward. This transformers version returns a decoder layer's
output as a **plain tensor**, not the classic `(hidden_states, ...)` tuple. Every hook in this
codebase assumed the tuple form unconditionally (`output[0]`, `tuple(output[1:])`) -- against a
plain tensor, `output[0]` silently mis-indexes along the batch dimension instead of raising, and
the hook then wraps the result BACK into a tuple, so the next layer receives a tuple where it
expects a tensor. This predates this session's extension entirely (it was in the original
`psr_refactor_v10` hook code) and could never have been caught by the existing fake-model tests,
because the fake model only ever returned tuples too.

Confirmed from the failure list: S-PSR and A-PSR actually succeeded (16 real minutes of training,
not in the failed-steps list) -- the crash is from the sweeps, the only steps that register
hooks; everything after that (generate/judge/summarize/bootstrap/plot) cascade-failed on missing
files as a pure consequence, not separate bugs.

- **New:** `steering/hooks.py` -- `unwrap_hidden(output)` / `rewrap_hidden(new_hidden, rest)`,
  handling both the tuple and plain-tensor conventions. Used in `steering_hook`/
  `multi_steering_hook` (this file) and all three `forward_with_gate_hook` implementations
  (`steering/psr/gate.py`, `conceptor/matrix/logic.py`, `conceptor/selfproj/logic.py`).
- **Changed:** `tests/fakes.py` -- `TinyLayer`/`TinyModel`/`make_fake_model_and_tokenizer` gained
  a `layer_returns_tuple` flag (default `True`, so every existing test is unaffected) so the
  plain-tensor code path is actually exercisable in tests, closing the exact blind spot that let
  this ship in the first place.
- **New tests**, in `tests/test_steering_psr_gate.py`: `forward_with_gate_hook` against a
  plain-tensor-returning fake model, and a same-correction-regardless-of-output-shape check.
  Verified these tests actually catch the bug -- reverted the fix locally, confirmed both new
  tests fail (with the fake-model analog of the real error), then restored the fix and confirmed
  all 82 tests pass.

## Round 6: unwrap_hidden hardened, and a verification step for you

Same reported error persisted after round 5's fix, byte-for-byte identical text/line, while
S-PSR/A-PSR (no hooks) kept succeeding and the sweeps (hooks) kept failing -- that pattern most
likely means the round-5 fix wasn't actually applied to the running checkout (only the `.sh`
script was updated, not the Python files). **Run this before anything else**:

```bash
grep -n "unwrap_hidden" steering/hooks.py steering/psr/gate.py
```

If empty, extract this tarball over the real repo's Python files (not just `infra/*.sh`) and rerun.

Made the fix itself more robust regardless, in case there's a second, subtler variant of the same
issue: `unwrap_hidden` now checks `isinstance(output, torch.Tensor)` (the unambiguous case)
instead of `isinstance(output, tuple)`. A tuple-only check would wrongly treat an HF
`ModelOutput`-style object (indexable via `output[0]`/`output[1:]`, but not a literal `tuple`
instance) as "plain tensor," returning the WHOLE container as if it were hidden_states -- a
second path to the identical downstream crash. New `tests/test_hooks.py` (7 tests) covers plain
tensor, tuple, list, and a fake ModelOutput-shaped object explicitly for this.

Still asked for (not yet resolved): the exact "3584 / 111" error text mentioned but not pasted --
these numbers look like Qwen2.5-7B's hidden_size (3584) and possibly a sequence length, but
without the actual error line I can't confirm what's mismatching. Search your saved log directly
rather than relying on terminal scrollback: `grep -n "3584" results/caveman/overnight_*.log`.
