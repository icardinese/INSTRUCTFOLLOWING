#!/usr/bin/env bash
# Runs the REAL judged-evidence layer/hyperparameter search (evals/layer_hparam_search.py) for
# every trainable variant, then prints a cross-variant comparison. This is the piece
# run_caveman_overnight.sh does NOT do -- that script only ever ran the MSE-based sweep and
# generated/judged whichever single config MSE picked as best. This script is what actually
# answers "does real judged evidence agree with what MSE picked" (see project finding #5 and
# BATCHED_STEERING.md/EXTENSION_CHANGES.md for why that question is the whole point).
#
# REQUIRES each variant's sweep to already be complete -- this reads results/<task>/psr_<variant>
# _sweep.jsonl, it does not run the sweep itself. Run infra/run_caveman_overnight.sh first if you
# haven't already (or confirm those files exist and are complete).
#
# RESUMABLE, same as everything else in this project: evals/layer_hparam_search.py itself skips a
# variant entirely if its Final result already exists, and reuses Tier 1/Tier 2 results from a
# previous interrupted run instead of re-paying the real GPU/OpenAI-API cost to redo them. A crash
# partway through costs at most whatever tier was in flight, not the whole run -- rerun this exact
# script and it picks up where it left off.
#
# RUN THIS IN A WAY THAT SURVIVES YOU DISCONNECTING -- same reason as run_caveman_overnight.sh:
#   tmux new -s layer_search   # then inside: bash infra/run_layer_search_overnight.sh
#   nohup bash infra/run_layer_search_overnight.sh > /dev/null 2>&1 &
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/src:${PYTHONPATH:-}"
cd "$REPO_ROOT"

TASK="caveman"
RESULTS_DIR="results/$TASK"
mkdir -p "$RESULTS_DIR"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="$RESULTS_DIR/layer_search_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

# Overridable, same pattern as the sweep script's env vars -- e.g. TIER1_N=10 for a faster,
# lower-power smoke test before committing to the default budget.
TIER1_N="${TIER1_N:-20}"
TIER2_N="${TIER2_N:-20}"
FINAL_N="${FINAL_N:-180}"
TOP_K_LAYERS="${TOP_K_LAYERS:-3}"
VARIANTS="${VARIANTS:-proper conceptor conceptor_matrix conceptor_selfproj}"

FAILURES=()
START_TS=$(date +%s)

run_step() {
    local desc="$1"; shift
    echo ""
    echo "== [$(date '+%Y-%m-%d %H:%M:%S')] START: $desc =="
    if "$@"; then
        echo "== [$(date '+%Y-%m-%d %H:%M:%S')] OK: $desc =="
    else
        echo "== [$(date '+%Y-%m-%d %H:%M:%S')] FAILED (continuing anyway): $desc =="
        FAILURES+=("$desc")
    fi
}

echo "################################################################"
echo "# Layer/hyperparameter search starting at $(date)"
echo "# Task=$TASK  variants=[$VARIANTS]  tier1_n=$TIER1_N tier2_n=$TIER2_N final_n=$FINAL_N top_k=$TOP_K_LAYERS"
echo "# Log: $LOG_FILE"
echo "################################################################"

echo ""
echo "== pre-flight checks =="
PREFLIGHT_OK=1

if [ ! -f "openai.key" ]; then
    echo "PRE-FLIGHT FAILURE: openai.key not found -- every judged evaluation in every tier needs it."
    PREFLIGHT_OK=0
else
    echo "openai.key: found"
fi

if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "PRE-FLIGHT FAILURE: torch.cuda.is_available() is False -- retraining and batched"
    echo "  generation both need a real GPU."
    PREFLIGHT_OK=0
else
    echo "CUDA: available"
fi

MISSING_SWEEPS=()
for v in $VARIANTS; do
    if [ ! -f "$RESULTS_DIR/psr_${v}_sweep.jsonl" ]; then
        MISSING_SWEEPS+=("$v")
    fi
done
if [ "${#MISSING_SWEEPS[@]}" -gt 0 ]; then
    echo "PRE-FLIGHT FAILURE: missing sweep file(s) for: ${MISSING_SWEEPS[*]}"
    echo "  Run infra/run_caveman_overnight.sh (or that variant's --sweep directly) first --"
    echo "  this script reads an already-completed sweep, it does not run one."
    PREFLIGHT_OK=0
else
    echo "sweep files: found for every requested variant"
fi

if [ "$PREFLIGHT_OK" != "1" ]; then
    echo ""
    echo "Aborting BEFORE touching the GPU/API -- fix the above and rerun."
    exit 1
fi
echo "pre-flight checks passed -- proceeding."

for v in $VARIANTS; do
    run_step "layer/hyperparameter search: $v" \
        python3 -u evals/layer_hparam_search.py --task "$TASK" --variant "$v" \
            --tier1-n "$TIER1_N" --tier2-n "$TIER2_N" --final-n "$FINAL_N" --top-k-layers "$TOP_K_LAYERS"
done

echo ""
echo "== cross-variant comparison =="
python3 -u evals/layer_hparam_search.py --task "$TASK" --summarize

END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))

echo ""
echo "################################################################"
echo "# DONE at $(date). Elapsed: $((ELAPSED / 3600))h $(((ELAPSED % 3600) / 60))m"
if [ "${#FAILURES[@]}" -eq 0 ]; then
    echo "# All variants completed."
else
    echo "# ${#FAILURES[@]} step(s) failed (everything else still ran and is usable):"
    for f in "${FAILURES[@]}"; do
        echo "#   - $f"
    done
    echo "# Rerun this exact script to retry failed variants -- completed ones (and completed"
    echo "# tiers within a failed one) are skipped automatically, not redone."
fi
echo "# Comparison: $RESULTS_DIR/tiered_search_summary.json"
echo "# Full log: $LOG_FILE"
echo "################################################################"
