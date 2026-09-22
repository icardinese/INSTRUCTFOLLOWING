#!/usr/bin/env python3
"""Fixes the --task triage failure: 28 files hardcoded choices=["caveman", "ifeval"] in their
argparse setup, so --task triage failed validation before any of today's registry/adapter work
ever ran. Run this directly on your tree instead of a patch, since the transformation is a plain
literal substitution and an import insertion -- no diff context to mismatch.

Idempotent: safe to run twice. Run from the repo root:

    python3 apply_task_choices_fix.py

Then verify:

    grep -rn 'choices=\["caveman", "ifeval"\]' --include=*.py .   # should print nothing
    python3 -m pytest tests/ -q --ignore=tests/test_caveman_judge.py
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent

# 1. adapters/registry.py: add TASK_CHOICES, generated from _TASKS -- the single source of truth
#    every CLI's --task flag should read from, so a newly registered task never needs a second
#    edit anywhere else.
REGISTRY = ROOT / "adapters" / "registry.py"
registry_src = REGISTRY.read_text()
if "TASK_CHOICES" not in registry_src:
    marker = "}\n\n\ndef get_adapter"
    if marker not in registry_src:
        sys.exit(f"expected marker not found in {REGISTRY} -- paste this file's contents back, "
                  f"the automatic fix can't proceed safely without seeing the real layout")
    addition = (
        "}\n\n"
        "# The single source of truth for --task choices=[...] across every script and trainer "
        "in\n# this project. There were 28 hardcoded copies of [\"caveman\", \"ifeval\"] scattered\n"
        "# across src/, evals/ and scripts/ before triage was added -- literals that predate this\n"
        "# registry existing everywhere, never updated when a new task was registered here.\n"
        "# Importing this instead of typing the list out makes that drift impossible: add a task\n"
        "# to _TASKS above and every CLI picks it up with no second edit.\n"
        "TASK_CHOICES = list(_TASKS)\n\n\ndef get_adapter"
    )
    registry_src = registry_src.replace(marker, addition, 1)
    REGISTRY.write_text(registry_src)
    print(f"updated {REGISTRY.relative_to(ROOT)}")
else:
    print(f"{REGISTRY.relative_to(ROOT)} already has TASK_CHOICES, skipping")

# 2. Every file with the hardcoded literal: swap it for TASK_CHOICES, import if needed.
PATTERN = re.compile(r'choices=\["caveman", "ifeval"\]')
IMPORT_LINE = "from adapters.registry import TASK_CHOICES"

changed = []
SELF = pathlib.Path(__file__).resolve()
for path in ROOT.rglob("*.py"):
    if ".git" in path.parts or path.resolve() == SELF:
        continue
    text = path.read_text()
    if not PATTERN.search(text):
        continue
    new_text = PATTERN.sub("choices=TASK_CHOICES", text)
    if IMPORT_LINE not in new_text:
        lines = new_text.split("\n")
        last_import = 0
        for i, line in enumerate(lines):
            if line.startswith(("import ", "from ")) and "__future__" not in line:
                last_import = i
        lines.insert(last_import + 1, IMPORT_LINE)
        new_text = "\n".join(lines)
    if new_text != text:
        path.write_text(new_text)
        changed.append(str(path.relative_to(ROOT)))

print(f"\nupdated {len(changed)} files:")
for f in sorted(changed):
    print(" ", f)

remaining = subprocess_check = None
import subprocess
out = subprocess.run(["grep", "-rln", r'choices=\["caveman", "ifeval"\]', "--include=*.py", "."],
                     cwd=ROOT, capture_output=True, text=True)
if out.stdout.strip():
    print("\nWARNING -- still found in:")
    print(out.stdout)
else:
    print("\nno remaining hardcoded occurrences.")
