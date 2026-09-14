#!/usr/bin/env bash
# Rebuilds docs_html/ from the current docstrings. Run this after any significant change to
# core/, steering/, adapters/, or evals/ -- takes a few seconds, no network needed once pdoc is
# installed (pip install pdoc, already in requirements.txt).
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH="$(pwd):$(pwd)/src"
rm -rf docs_html
python3 -m pdoc --output-dir docs_html core steering adapters evals

echo "== docs built at docs_html/index.html =="
echo "   open it directly in a browser, or: python3 -m http.server -d docs_html 8000"
