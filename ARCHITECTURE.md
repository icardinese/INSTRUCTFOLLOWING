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
code path was run against caveman's LLM-judge scoring shape (`{correct, coherent}`) and IFEval's
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
```

`cache/<task>/` (teacher-forced responses, pooled activations) is separate from `results/<task>/`
on purpose: cache contents are deterministic and safe to delete anytime; results are not
guaranteed to regenerate identically (an LLM-judged sweep, for instance) and losing them is a real
loss, not a cache miss. See `.gitignore` -- both are excluded from git, but for different reasons.

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
