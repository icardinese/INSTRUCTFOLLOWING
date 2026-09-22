"""Maps a task name to its adapter module. Every adapter module (caveman_adapter.py, ifeval_adapter.py)
is expected to expose the same names -- DATA_DIR, RESULTS_DIR, CACHE_DIR, MODEL_NAME,
load_rows(split), to_items(tokenizer, rows) -- duck-typed, not enforced via inheritance, but checked explicitly here so
a missing piece fails loudly at startup with a clear message, not a mysterious AttributeError three
function calls into training.
"""
import importlib

_REQUIRED_ATTRS = ["DATA_DIR", "RESULTS_DIR", "CACHE_DIR", "MODEL_NAME", "load_rows", "to_items"]

# Optional, read via getattr with a default so existing adapters need no edits. An adapter that
# wants a non-default intervention surface declares it here rather than any call site special-
# casing the task name -- see steering/psr/spans.py for why the right surface is task-dependent.
_OPTIONAL_ATTRS = {"STEERING_LOCATION": "answer_only"}


def steering_location_for(task_name: str) -> str:
    """The intervention surface this task's removed instruction requires. caveman and ifeval use
    the default (R, response-only); triage declares QR because its slots govern how the input is
    read, which R cannot reach."""
    from steering.psr.spans import validate_location
    module = get_adapter(task_name)
    return validate_location(getattr(module, "STEERING_LOCATION", _OPTIONAL_ATTRS["STEERING_LOCATION"]))

_TASKS = {
    "caveman": "adapters.caveman_adapter",
    "ifeval": "adapters.ifeval_adapter",
    "triage": "adapters.triage_adapter",
}

# The single source of truth for --task choices=[...] across every script and trainer in
# this project. There were 28 hardcoded copies of ["caveman", "ifeval"] scattered
# across src/, evals/ and scripts/ before triage was added -- literals that predate this
# registry existing everywhere, never updated when a new task was registered here.
# Importing this instead of typing the list out makes that drift impossible: add a task
# to _TASKS above and every CLI picks it up with no second edit.
TASK_CHOICES = list(_TASKS)


def get_adapter(task_name: str):
    if task_name not in _TASKS:
        raise ValueError(f"unknown task {task_name!r}, available: {list(_TASKS)}")
    module = importlib.import_module(_TASKS[task_name])
    missing = [attr for attr in _REQUIRED_ATTRS if not hasattr(module, attr)]
    if missing:
        raise AttributeError(
            f"adapters.{task_name}_adapter is missing required attribute(s) {missing} -- "
            f"every adapter needs all of {_REQUIRED_ATTRS}."
        )
    return module
