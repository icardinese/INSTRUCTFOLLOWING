#!/usr/bin/env bash
# Re-runs ONLY what the two conceptor/loss-grid fixes actually invalidated -- NOT a full re-sweep
# of everything. proper/conceptor_matrix/conceptor_selfproj's sweep files still contain valid pure-
# MSE and pure-NLL rows (the loss-grid change just stopped generating the 3 useless blend rows
# going forward -- it didn't invalidate rows already on disk, and best_row_per_layer always picked
# pure MSE anyway, so those three don't need re-running tonight). ONLY conceptor's sweep is
# actually wrong on disk right now, because its direction source changed. Old conceptor
# sweep/tiered-search/probe files are backed up with a timestamp suffix, never deleted.
#
# RUN THIS IN A WAY THAT SURVIVES YOU DISCONNECTING:
#   tmux new -s conceptor_fix   # then inside: bash infra/run_conceptor_fix_overnight.sh
#   nohup bash infra/run_conceptor_fix_overnight.sh > /dev/null 2>&1 &
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/src:${PYTHONPATH:-}"
cd "$REPO_ROOT"

TASK="caveman"
RESULTS_DIR="results/$TASK"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="$RESULTS_DIR/conceptor_fix_${TIMESTAMP}.log"
mkdir -p "$RESULTS_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

TIER1_N="${TIER1_N:-20}"
TIER2_N="${TIER2_N:-20}"
FINAL_N="${FINAL_N:-180}"
TOP_K_LAYERS="${TOP_K_LAYERS:-3}"

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
echo "# Conceptor direction fix -- re-run starting at $(date)"
echo "# Log: $LOG_FILE"
echo "################################################################"

echo ""
echo "== pre-flight checks =="
PREFLIGHT_OK=1

if [ ! -f "openai.key" ]; then
    echo "PRE-FLIGHT FAILURE: openai.key not found -- judged evaluation needs it."
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

if [ "$PREFLIGHT_OK" != "1" ]; then
    echo ""
    echo "Aborting BEFORE touching the GPU/API -- fix the above and rerun."
    exit 1
fi
echo "pre-flight checks passed -- proceeding."

echo ""
echo "== backing up stale conceptor artifacts (never deleting) =="
for f in "$RESULTS_DIR/psr_conceptor_sweep.jsonl" "$RESULTS_DIR/conceptor_tiered_search.jsonl" \
         "$RESULTS_DIR/psr_conceptor_probe.pt" "$RESULTS_DIR/psr_conceptor_train_log.json" \
         "cache/$TASK/pooled_base_layer"* "cache/$TASK/pooled_instr_layer"*; do
    for match in $f; do
        if [ -e "$match" ]; then
            mv "$match" "${match}.pre_direction_fix_${TIMESTAMP}"
            echo "  backed up: $match -> ${match}.pre_direction_fix_${TIMESTAMP}"
        fi
    done
done
# NOTE: response-token pooled activations (cache/.../pooled_base_layer*, pooled_instr_layer*) are
# also moved aside -- they're still valid for C's construction (that part didn't change), but
# moving them forces a clean re-pool alongside the new pool_prompt_last_token cache files rather
# than mixing pre-fix and post-fix cache state in the same run. Costs one extra re-pooling pass,
# buys certainty that nothing stale leaks in.

echo ""
echo "== re-running conceptor sweep with the restored prompt-representation direction =="
run_step "conceptor sweep (fixed direction, 2-point loss grid)" \
    python3 -u src/psr/conceptor/train.py --task "$TASK" --sweep

echo ""
echo "== re-running conceptor's tiered layer/hyperparameter search =="
run_step "conceptor tiered search" \
    python3 -u evals/layer_hparam_search.py --task "$TASK" --variant conceptor \
        --tier1-n "$TIER1_N" --tier2-n "$TIER2_N" --final-n "$FINAL_N" --top-k-layers "$TOP_K_LAYERS"

echo ""
echo "== cross-variant comparison (includes proper/matrix/selfproj from before, unaffected) =="
run_step "cross-variant summary" \
    python3 -u evals/layer_hparam_search.py --task "$TASK" --summarize

END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))

echo ""
echo "################################################################"
echo "# DONE at $(date). Elapsed: $((ELAPSED / 3600))h $(((ELAPSED % 3600) / 60))m"
if [ "${#FAILURES[@]}" -eq 0 ]; then
    echo "# Completed with no failures."
else
    echo "# ${#FAILURES[@]} step(s) failed:"
    for f in "${FAILURES[@]}"; do
        echo "#   - $f"
    done
    echo "# Rerun this exact script to retry -- completed tiers/steps are skipped, not redone."
fi
echo "# Compare $RESULTS_DIR/conceptor_tiered_search.jsonl's new Final avg_tokens against the"
echo "# backed-up ${RESULTS_DIR}/conceptor_tiered_search.jsonl.pre_direction_fix_${TIMESTAMP}"
echo "# to see whether the direction fix actually closed the gap."
echo "################################################################"
