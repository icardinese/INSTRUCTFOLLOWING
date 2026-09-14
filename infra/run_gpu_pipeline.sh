#!/usr/bin/env bash
# Full pipeline for one task: const calibration -> every PSR variant -> generation -> judging ->
# summary -> bootstrap CIs. Usage:
#   bash infra/run_gpu_pipeline.sh caveman
#   bash infra/run_gpu_pipeline.sh ifeval
#
# Drive backup is entirely optional -- if /content/drive/Shareddrives isn't mounted, this just
# skips backup and runs locally, no error. Where it IS mounted, sync is TIME-gated (every 5 min,
# background thread) rather than triggered by row/step count -- the same fix applied earlier this
# project to a script that was accidentally re-copying an ever-growing file on every Nth row,
# which quietly became an O(n^2) cost as the file grew across a long run.
set -euo pipefail

TASK="${1:?usage: run_gpu_pipeline.sh <caveman|ifeval> [layer_override]}"
LAYER_OVERRIDE="${2:-}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/src:${PYTHONPATH:-}"
cd "$REPO_ROOT"

RESULTS_DIR="results/$TASK"
DRIVE_BACKUP_DIR="/content/drive/Shareddrives/Eric/LLM_STEER_BACKUP/results/$TASK"
FORCE_RERUN="${FORCE_RERUN:-0}"

if [ -d "/content/drive/Shareddrives" ]; then
    mkdir -p "$DRIVE_BACKUP_DIR"
    sync_to_drive() {
        while true; do
            rsync -a --exclude 'pooled_*' "$RESULTS_DIR/" "$DRIVE_BACKUP_DIR/" 2>/dev/null || true
            sleep 300
        done
    }
    sync_to_drive &
    SYNC_PID=$!
    trap 'kill "$SYNC_PID" 2>/dev/null || true' EXIT
    echo "== background Drive sync started for task=$TASK (pid $SYNC_PID, every 5 min) =="
else
    echo "== Shared Drive not mounted -- running local-only, no backup. Mount it from a notebook"
    echo "   cell first if you want backup: from google.colab import drive; drive.mount('/content/drive') =="
fi

echo "== const steering calibration sweep (task=$TASK) =="
FORCE_RERUN="$FORCE_RERUN" python3 -u src/const/sweep.py --task "$TASK"

if [ -z "$LAYER_OVERRIDE" ] && [ ! -f "$RESULTS_DIR/const_steer_config.json" ]; then
    echo "== const_steer_config.json not found. Judge results/$TASK/sweep_dev.jsonl and write this"
    echo "   file with the chosen (layer, coefficient) before continuing -- every PSR variant below"
    echo "   needs it (or pass an explicit layer as this script's 2nd argument to skip that step):"
    echo "   bash infra/run_gpu_pipeline.sh $TASK 14"
    exit 1
fi

LAYER_ARGS=()
[ -n "$LAYER_OVERRIDE" ] && LAYER_ARGS=(--layer "$LAYER_OVERRIDE")

echo "== training every PSR variant (task=$TASK) =="
FORCE_RERUN="$FORCE_RERUN" python3 -u src/psr/old_baseline/train.py --task "$TASK" "${LAYER_ARGS[@]}"
FORCE_RERUN="$FORCE_RERUN" python3 -u src/psr/old_baseline/train_a_psr.py --task "$TASK"
FORCE_RERUN="$FORCE_RERUN" python3 -u src/psr/proper/train.py --task "$TASK" "${LAYER_ARGS[@]}"
FORCE_RERUN="$FORCE_RERUN" python3 -u src/psr/conceptor/train.py --task "$TASK" "${LAYER_ARGS[@]}"
FORCE_RERUN="$FORCE_RERUN" python3 -u src/psr/conceptor/matrix/train.py --task "$TASK" "${LAYER_ARGS[@]}"
python3 -u src/psr/conceptor/selfproj/train.py --task "$TASK" "${LAYER_ARGS[@]}"  # own alpha-sweep resumability, no FORCE_RERUN check

echo "== generating every available condition (task=$TASK) =="
python3 -u src/generate.py --task "$TASK" --split test

echo "== judging + summarizing (task=$TASK) =="
python3 -u evals/run_judge.py --task "$TASK" --split test
python3 -u evals/summarize.py --task "$TASK" --split test
python3 -u evals/bootstrap_analysis.py --task "$TASK" --split test

if [ -d "/content/drive/Shareddrives" ]; then
    echo "== zipping results and copying to Drive =="
    ZIP_PATH="results_${TASK}_backup.zip"
    rm -f "$ZIP_PATH"
    zip -rq "$ZIP_PATH" "$RESULTS_DIR" -x "*/pooled_*"
    cp "$ZIP_PATH" "$DRIVE_BACKUP_DIR/"

    echo "== final sync =="
    rsync -a --exclude 'pooled_*' "$RESULTS_DIR/" "$DRIVE_BACKUP_DIR/"
    echo "== done. Everything is in $DRIVE_BACKUP_DIR =="
else
    echo "== done. Results are local only in $RESULTS_DIR (no Drive mounted) =="
fi
