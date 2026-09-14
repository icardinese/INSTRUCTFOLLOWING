# Architecture

This codebase is organized around one question, asked of every file: **does this know about a
specific task's data, or a specific method's math -- or neither?** That question is the entire
reason the five top-level directories exist, and it's the thing to check before adding new code
anywhere in this repo.

```
core/       task-agnostic AND method-agnostic. Model loading, hook mechanics, caching.
steering/   method-agnostic math (PSR, conceptor, const), but knows nothing about caveman/ifeval.
adapters/   task-specific data (one file per task: what a "prompt" and "row" mean for that task).
evals/      task-specific grading (one file per task: how a response gets scored).
src/        orchestration -- wires core + steering + adapters + evals together into runnable scripts.
```

If you're ever unsure where a new function belongs, ask: "would this still make sense if I deleted
every other task, or every other method?" If yes to both, it's `core/`. If it only survives without
other *methods* (not tasks), it's `steering/`. If it only survives without other *tasks* (not
methods), it's `adapters/` or `evals/`. Everything else is orchestration -- `src/`.

## The registry pattern

`adapters/registry.py` and `evals/registry.py` are the same pattern twice: a task name maps to a
module, and that module must expose a fixed, small contract. `get_adapter`/`get_eval_adapter` check
that contract explicitly and raise a clear, named error if anything is missing -- not a cryptic
`AttributeError` three function calls later.

This is *the* mechanism that makes adding a task cheap: `src/generate.py`, `evals/run_judge.py`, and
every training script never import a task's adapter directly. They call `get_adapter(args.task)`
and work with whatever comes back. Concretely proven this session: the exact same `evals/run_judge.py`
code path was run against caveman's LLM-judge scoring shape (`{correct, coherent, conciseness}`)
and IFEval's
programmatic scoring shape (`{follow_all_instructions, n_followed, n_total}`) with zero
special-casing needed.

### Adding a new task

1. Write `adapters/<task>_adapter.py` exposing `DATA_DIR`, `RESULTS_DIR`, `CACHE_DIR`, `MODEL_NAME`,
   `load_rows(split)`, `to_items(tokenizer, rows)`.
2. Write `evals/<task>/judge.py` exposing `SCORE_FIELDS` (a list of the field names your
   `score_response` returns) and `score_response(row, response) -> dict`.
3. Add one line to each registry's `_TASKS` dict.
4. Nothing else changes. Every training script, `generate.py`, `run_judge.py`, `summarize.py`, and
   `bootstrap_analysis.py` already work with your new task.

### Adding a new steering method

1. Write the method's math under `steering/<method>/` -- state (what gets trained) and logic (how a
   correction gets computed) only. No data loading, no task references.
2. If it shares gate mechanics with PSR (a token-specific coefficient gating a correction), reuse
   `steering/psr/gate.py`'s `GateState`/`coefficient`/`forward_with_gate_hook` rather than
   reimplementing them -- `psr/proper` and `psr/conceptor` both do exactly this, differing only in
   where `direction` comes from.
3. Write `src/<method>/train.py` following the existing scripts' shape: `--task` selects an adapter
   at runtime (never hardcode one), `--seed` defaults to 42, log every hyperparameter actually used
   into the output JSON (not just the ones that happened to seem important at the time -- this was
   a real gap fixed this session: `seed`/`lr`/`weight_decay` weren't recorded anywhere until it was).
4. Add a `load_<method>_condition(adapter, device)` function to `src/generate.py` returning
   `context_fn(model) -> context manager`, and a line in `CONDITION_LOADERS`. This is the *only*
   place `generate.py` needs to change -- the main generation loop never branches on method type.

## Why some things are NOT centralized

**Hyperparameters aren't in a config file.** Every train script has env-var-overridable constants
at the top (`PSR_CONCEPTOR_ALPHA`, etc.) instead of a Hydra/OmegaConf config layer. This was a
deliberate choice, not an oversight: the env-var pattern already does the actual job (overriding a
value per-run, including for sweeps), and introducing a second, parallel mechanism for the same
concern would add complexity without solving a real problem. If a genuine multi-axis sweep need
shows up later (not "would be nice to have," but "actually blocked without it"), that's the signal
to revisit this, not before.

**`core/`, not `steering/`, owns the hook *attachment* mechanism.** `steering/hooks.py`'s
`steering_hook`/`multi_steering_hook` register/remove PyTorch forward hooks -- they contain zero
method-specific math, but they only exist because steering methods need them, so they don't belong
in `core/` either (a project with zero steering methods would have no reason for them to exist).
They're the one thing in `steering/` that isn't any single method's math.

## Data flow, one full run

```
src/const/sweep.py                              -> results/<task>/{sweep_dev.jsonl, const_steer_directions.pt}
    (judge sweep_dev.jsonl, write const_steer_config.json by hand -- this step is NOT automated,
     matching the original pipeline design; it's a judgment call, not a computation)
src/psr/old_baseline/train.py                   -> results/<task>/psr_probe.pt          (S-PSR baseline)
src/psr/old_baseline/train_a_psr.py             -> results/<task>/a_psr_probe.pt        (A-PSR baseline)
src/psr/proper/train.py                         -> results/<task>/psr_proper_probe.pt
src/psr/conceptor/train.py                      -> results/<task>/psr_conceptor_probe.pt
src/psr/conceptor/matrix/train.py               -> results/<task>/psr_conceptor_matrix_probe.pt
src/psr/conceptor/selfproj/train.py             -> results/<task>/psr_conceptor_selfproj_probe.pt
src/generate.py                                 -> results/<task>/generations_test.jsonl
    (gracefully skips any condition whose checkpoint doesn't exist -- this file never needs
     editing as methods get trained; only checkpoints need to show up)
evals/run_judge.py                              -> results/<task>/judged_test.jsonl
evals/summarize.py                              -> results/<task>/summary_test.json
evals/bootstrap_analysis.py                     -> printed CIs, --compare for paired comparisons
evals/plotting.py                               -> results/<task>/plots/*.png
```

Each of proper/conceptor/conceptor-matrix/conceptor-selfproj's `train.py` ALSO exposes a `sweep()`
function (`--sweep` on the CLI) as an alternative to a single `main()` run:

```
src/psr/proper/train.py --sweep                     -> results/<task>/psr_proper_sweep.jsonl
src/psr/conceptor/train.py --sweep                  -> results/<task>/psr_conceptor_sweep.jsonl
src/psr/conceptor/matrix/train.py --sweep           -> results/<task>/psr_conceptor_matrix_sweep.jsonl
src/psr/conceptor/selfproj/train.py --sweep         -> results/<task>/psr_conceptor_selfproj_sweep.jsonl
```

Each grids over layer (+ alpha, for the conceptor variants) x nll_weight, resumable via
`core/sweep.py`, and writes the SAME `..._probe.pt` `main()` would have -- so nothing downstream
(`generate.py` etc.) needs to know whether a checkpoint came from a single run or the best point
of a sweep. `evals/plotting.py`'s `plot_sweep_file` turns any of these JSONLs into a
layer-vs-metric plot, plus a participation-ratio-vs-metric scatter for the two matrix-based
variants (see below).

`cache/<task>/` (teacher-forced responses, pooled activations) is separate from `results/<task>/`
on purpose: cache contents are deterministic and safe to delete anytime; results are not
guaranteed to regenerate identically (an LLM-judged sweep, for instance) and losing them is a real
loss, not a cache miss. See `.gitignore` -- both are excluded from git, but for different reasons.

## Auxiliary NLL loss and participation ratio

Every gate-based PSR variant's training loop (`steering/psr/training_loop.py`) supports an
optional auxiliary log-likelihood term (Eq. 4, Heyman & Vandeputte 2026 -- see
`steering/psr/nll.py`), controlled by each variant's own `PSR_<VARIANT>_NLL_WEIGHT` env var
(default `0.0`, so nothing changes unless explicitly opted into) and swept as a real hyperparameter
axis by `sweep()`. This required every `forward_with_gate_hook` (`gate.py`,
`conceptor/matrix/logic.py`, `conceptor/selfproj/logic.py`) to also return `logits` from the same
corrected forward pass (free -- the model already computes them), and `train_gate` to return
`{"mse": ..., "nll": ...}` dicts instead of a lone MSE float.

Matrix-based methods (conceptor/matrix and conceptor/selfproj -- the two that apply C fresh to
every hidden state, NOT the fixed-vector conceptor variant, whose injected correction stays
rank-1 regardless of C) log their conceptor's participation ratio
(`steering/psr/conceptor/rank_diagnostic.py`, PR = (sum lambda)^2 / sum(lambda^2)) at train time,
and again at generation time: `src/generate.py` attaches it to every row for that condition via a
`context_fn.participation_ratio` attribute, read back generically with `getattr` rather than a
hardcoded condition-name list, so a future matrix-based method gets this for free the moment its
own loader sets the same attribute. `evals/run_judge.py` forwards any such `{cond}_*` metadata
field into `judged_{split}.jsonl` automatically, by key prefix -- processing condition names
longest-first so a name like `"psr"` can't accidentally swallow `"psr_proper"`'s fields, a real
bug caught by `tests/test_run_judge_metadata_carryover.py`. `evals/summarize.py`,
`evals/bootstrap_analysis.py`, and `evals/plotting.py` all read this through one shared collector,
`evals.summarize.collect_raw_values_by_cond_field`, so a future metadata field needs changes only
in whichever `generate.py` loader sets it -- never in those three files.

## Caveman conciseness score

`evals/caveman/judge.py`'s `SCORE_FIELDS` is `["correct", "coherent", "conciseness"]` --
conciseness is a THIRD, independent LLM-judged call (own rubric, own model,
`CONCISENESS_JUDGE_MODEL = "gpt-4o"`, distinct from `JUDGE_MODEL = "gpt-4o-mini"` used for
correct/coherent), not folded into the existing rubric or call. It exists alongside, not instead
of, the per-row token counts every condition already gets (`{cond}_tokens`) -- the two measure
different things: token count is a blunt proxy for terseness, conciseness is a judged score of
whether the WORDING itself is padded (hedging, repetition, restating the question), so a response
can be short-but-padded or long-but-tight, and this is the metric that tells those apart. Because
this is a purely task-specific addition (the rubric only makes sense for caveman's code-explanation
setting), it required zero changes anywhere outside `evals/caveman/judge.py` itself --
`run_judge.py`, `summarize.py`, `bootstrap_analysis.py`, and `plotting.py` are all already generic
over whatever `SCORE_FIELDS` a task's judge returns, which is exactly what the registry pattern
promises ("Adding a new task" above). `evals/plotting.py`'s `plot_judged_results` does add one
small task-aware branch: an extra compression-vs-conciseness frontier plot specifically when
`"conciseness"` is present in `SCORE_FIELDS`, since that comparison (does the LLM-judged
conciseness score actually track raw token count, or diverge from it) is the direct reason this
metric exists.

## Testing

`tests/` is real pytest, not one-off verification scripts. Two things worth knowing before adding
to it:

- `tests/fakes.py`'s `TinyModel`/`TinyTokenizer` genuinely respect attention masks (masked running
  mean instead of no mixing at all). An earlier version without that made every batched-generation
  test meaningless -- padding tokens could silently corrupt real ones without any test noticing.
  Use `make_fake_model_and_tokenizer()` from there rather than building a new fake model per test.
- `tests/test_ifeval_integration_and_reproducibility.py` imports the REAL vendored Google IFEval
  harness (`third_party/microsoft_llm_steer_instruct/ifeval_scripts/`), not a mock. If that harness
  ever moves or gets updated, this is the test that will tell you.

Run everything with `python3 -m pytest tests/ -v` from the repo root (or via `PYTHONPATH` set as in
`infra/run_gpu_pipeline.sh`, which the setup script already runs once as a sanity check).
