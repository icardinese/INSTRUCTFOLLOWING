#!/usr/bin/env bash
# Run once per fresh Colab runtime, AFTER mounting Drive from a notebook cell:
#     from google.colab import drive; drive.mount('/content/drive')
# This script can't do that mount itself -- the auth handshake needs the Colab frontend.
set -euo pipefail

cd "$(dirname "$0")/.."

DRIVE_BACKUP_DIR="/content/drive/Shareddrives/Eric/LLM_STEER_BACKUP"
if [ ! -d "/content/drive/Shareddrives" ]; then
    echo "ERROR: /content/drive/Shareddrives not found. Mount Drive from a notebook cell first:"
    echo "  from google.colab import drive; drive.mount('/content/drive')"
    echo "(Shared Drives mount automatically under Shareddrives/ as long as your account has access."
    echo " If you're not using Drive backup right now, that's fine -- just run without this script"
    echo " and use a plain 'pip install -r requirements.txt' instead.)"
    exit 1
fi
mkdir -p "$DRIVE_BACKUP_DIR/results" "$DRIVE_BACKUP_DIR/hf_cache"

pip install --upgrade pip
pip install -r requirements.txt

# HF cache on Drive, not local Colab disk -- model weights only download once, ever, across every
# future Colab session instead of on every fresh runtime.
export HF_HOME="$DRIVE_BACKUP_DIR/hf_cache"
echo "export HF_HOME=\"$DRIVE_BACKUP_DIR/hf_cache\"" >> ~/.bashrc

# Restore any prior backup zips (see run_gpu_pipeline.sh's end-of-run zip step) -- extracts from
# the repo ROOT, not into results/ directly: the zip's internal paths already start with
# "results/<task>/...", so extracting into results/ would double it into results/results/<task>/.
# Verified against a synthetic zip built the same way earlier this session.
for task in caveman ifeval; do
    ZIP_PATH="$DRIVE_BACKUP_DIR/results/results_${task}_backup.zip"
    if [ -f "$ZIP_PATH" ]; then
        echo "== found backup for task=$task, restoring (won't overwrite anything already local) =="
        unzip -n "$ZIP_PATH" -d .
    fi
done

echo "== running the test suite as a setup sanity check =="
REPO_ROOT="$(pwd)"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/src:${PYTHONPATH:-}"
python3 -m pytest tests/ -q

echo "== setup done. Drive-backed results dir: $DRIVE_BACKUP_DIR/results =="
