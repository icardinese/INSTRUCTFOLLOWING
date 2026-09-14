"""Maps a task name to its eval adapter module. Every eval adapter exposes exactly one thing --
score_response(row, response) -> dict -- the orchestration (reading generations, looping over
conditions, writing judged output) is fully generic and lives in run_judge.py, which is NEVER
touched when adding a new task's judging. Adding a new task's eval support is one new file plus
one line here.
"""
import importlib

_REQUIRED_ATTRS = ["score_response", "SCORE_FIELDS"]

_TASKS = {
    "caveman": "evals.caveman.judge",
    "ifeval": "evals.ifeval.judge",
}


def get_eval_adapter(task_name: str):
    if task_name not in _TASKS:
        raise ValueError(f"unknown task {task_name!r}, available: {list(_TASKS)}")
    module = importlib.import_module(_TASKS[task_name])
    missing = [attr for attr in _REQUIRED_ATTRS if not hasattr(module, attr)]
    if missing:
        raise AttributeError(
            f"evals.{task_name}.judge is missing required attribute(s) {missing} -- "
            f"every eval adapter needs {_REQUIRED_ATTRS}."
        )
    return module
