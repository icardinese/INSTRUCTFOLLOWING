"""Maps a task name to its adapter module. Every adapter module (caveman_adapter.py, ifeval_adapter.py)
is expected to expose the same four names -- DATA_DIR, RESULTS_DIR, MODEL_NAME, load_rows(split),
to_items(tokenizer, rows) -- duck-typed, not enforced via inheritance, but checked explicitly here so
a missing piece fails loudly at startup with a clear message, not a mysterious AttributeError three
function calls into training.
"""
import importlib

_REQUIRED_ATTRS = ["DATA_DIR", "RESULTS_DIR", "CACHE_DIR", "MODEL_NAME", "load_rows", "to_items"]

_TASKS = {
    "caveman": "adapters.caveman_adapter",
    "ifeval": "adapters.ifeval_adapter",
}


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
