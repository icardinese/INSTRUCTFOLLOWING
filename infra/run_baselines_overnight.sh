#!/usr/bin/env bash
# Runs Const's and Stolfo's tiered searches (evals/layer_hparam_search.py --variant const/stolfo),
# then the reproducible cosine similarity script. Both are training-free -- no gradient descent at
# all -- so this should be noticeably faster than the PSR-variant overnight runs despite Const's
# grid being fairly large (13 layers x 10 coefficients = 130 Tier 1 points).
#
#   tmux new -s baselines   # then inside: bash infra/run_baselines_overnight.sh
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/src:${PYTHONPATH:-}"
cd "$REPO_ROOT"

TASK="caveman"
RESULTS_DIR="results/$TASK"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="$RESULTS_DIR/baselines_${TIMESTAMP}.log"
mkdir -p "$RESULTS_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

TIER1_N="${TIER1_N:-20}"
FINAL_N="${FINAL_N:-180}"

echo "################################################################"
echo "# Const + Stolfo-projection baselines -- starting at $(date)"
echo "# Log: $LOG_FILE"
echo "################################################################"

echo ""
echo "== pre-flight checks =="
PREFLIGHT_OK=1
[ -f "openai.key" ] && echo "openai.key: found" || { echo "PRE-FLIGHT FAILURE: openai.key missing"; PREFLIGHT_OK=0; }
python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null && echo "CUDA: available" || { echo "PRE-FLIGHT FAILURE: no CUDA"; PREFLIGHT_OK=0; }
[ "$PREFLIGHT_OK" = "1" ] || { echo ""; echo "Aborting before touching GPU/API."; exit 1; }
echo "pre-flight checks passed."

FAILURES=()
for VARIANT in const stolfo; do
    echo ""
    echo "== [$(date '+%Y-%m-%d %H:%M:%S')] $VARIANT =="
    if python3 -u evals/layer_hparam_search.py --task "$TASK" --variant "$VARIANT" --tier1-n "$TIER1_N" --final-n "$FINAL_N"; then
        echo "== [$(date '+%Y-%m-%d %H:%M:%S')] OK: $VARIANT =="
    else
        echo "== [$(date '+%Y-%m-%d %H:%M:%S')] FAILED: $VARIANT (continuing) =="
        FAILURES+=("$VARIANT")
    fi
done

echo ""
echo "== cross-variant summary (now includes const/stolfo alongside everything else) =="
python3 -u evals/layer_hparam_search.py --task "$TASK" --summarize

echo ""
echo "== reproducible cosine similarity matrix =="
if python3 -u scripts/compute_cosine_similarities.py --task "$TASK"; then
    echo "OK: cosine matrix"
else
    echo "FAILED: cosine matrix (const/stolfo tiered searches may not have reached Final -- check above)"
    FAILURES+=("cosine_matrix")
fi

echo ""
echo "################################################################"
if [ "${#FAILURES[@]}" -eq 0 ]; then
    echo "# DONE at $(date). Completed with no failures."
else
    echo "# DONE at $(date) with failures: ${FAILURES[*]} -- rerun this script, already-completed"
    echo "# variants are skipped (Final already exists), not redone."
fi
echo "# Results: $RESULTS_DIR/const_tiered_search.jsonl"
echo "#          $RESULTS_DIR/stolfo_tiered_search.jsonl"
echo "#          $RESULTS_DIR/cosine_similarity_matrix.{csv,json}"
echo "# Full log: $LOG_FILE"
echo "################################################################"
