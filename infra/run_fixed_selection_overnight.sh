#!/usr/bin/env bash
# Runs scripts/rerun_with_fixed_selection.py: re-selects conceptor's/proper's tiered-search
# winners under the corrected avg_tokens-primary ranking (paying real cost only where the fix
# actually requires new evaluation -- conceptor needs Final only, proper needs Tier 2 + Final),
# regenerates a real checkpoint at each corrected winner, then runs the direction-only ablation
# (evals/ablation_direction_only.py) on both.
#
# REQUIRES results/caveman/conceptor_tiered_search.jsonl and proper_tiered_search.jsonl to already
# exist with real tier1 data in them (they do, from the earlier overnight runs).
#
# RUN THIS IN A WAY THAT SURVIVES YOU DISCONNECTING:
#   tmux new -s fixed_selection   # then inside: bash infra/run_fixed_selection_overnight.sh
#   nohup bash infra/run_fixed_selection_overnight.sh > /dev/null 2>&1 &
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/src:${PYTHONPATH:-}"
cd "$REPO_ROOT"

TASK="caveman"
RESULTS_DIR="results/$TASK"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="$RESULTS_DIR/fixed_selection_${TIMESTAMP}.log"
mkdir -p "$RESULTS_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

TIER1_N="${TIER1_N:-20}"
TIER2_N="${TIER2_N:-20}"
FINAL_N="${FINAL_N:-180}"
TOP_K_LAYERS="${TOP_K_LAYERS:-3}"
SKIP_ABLATION="${SKIP_ABLATION:-0}"

echo "################################################################"
echo "# avg_tokens-primary re-selection -- starting at $(date)"
echo "# Log: $LOG_FILE"
echo "################################################################"

echo ""
echo "== pre-flight checks =="
PREFLIGHT_OK=1

if [ ! -f "openai.key" ]; then
    echo "PRE-FLIGHT FAILURE: openai.key not found."
    PREFLIGHT_OK=0
else
    echo "openai.key: found"
fi

if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "PRE-FLIGHT FAILURE: torch.cuda.is_available() is False."
    PREFLIGHT_OK=0
else
    echo "CUDA: available"
fi

for f in "$RESULTS_DIR/conceptor_tiered_search.jsonl" "$RESULTS_DIR/proper_tiered_search.jsonl"; do
    if [ ! -f "$f" ]; then
        echo "PRE-FLIGHT FAILURE: $f not found -- this script re-selects from an existing tiered search, it doesn't start one from scratch."
        PREFLIGHT_OK=0
    fi
done

if [ "$PREFLIGHT_OK" != "1" ]; then
    echo ""
    echo "Aborting BEFORE touching the GPU/API -- fix the above and rerun."
    exit 1
fi
echo "pre-flight checks passed -- proceeding."

ABLATION_FLAG=""
if [ "$SKIP_ABLATION" = "1" ]; then
    ABLATION_FLAG="--skip-ablation"
fi

python3 -u scripts/rerun_with_fixed_selection.py \
    --timestamp "$TIMESTAMP" --tier1-n "$TIER1_N" --tier2-n "$TIER2_N" \
    --final-n "$FINAL_N" --top-k-layers "$TOP_K_LAYERS" $ABLATION_FLAG
STATUS=$?

echo ""
echo "################################################################"
if [ "$STATUS" -eq 0 ]; then
    echo "# DONE at $(date). Completed with no failures."
else
    echo "# FAILED at $(date) (exit $STATUS) -- rerun this exact script, already-truncated files"
    echo "# and already-regenerated checkpoints are left in place, so a rerun should pick up close"
    echo "# to where it stopped, not redo everything from scratch."
fi
echo "# Compare *_tiered_search.jsonl against *.pre_avg_tokens_fix_${TIMESTAMP} to see what changed."
echo "# Full log: $LOG_FILE"
echo "################################################################"
