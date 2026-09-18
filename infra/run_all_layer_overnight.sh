#!/usr/bin/env bash
# A-PSR (faithful, ALL layers) + its Multi-Gate ablation, both loss objectives, then judged eval.
#
# 4 training runs total (2 variants x 2 loss configs) -- NO layer sweep, because A-PSR has no layer
# hyperparameter (all layers, definitionally). That makes this far cheaper than proper's 65-point
# sweep. Each variant's sweep() is resumable and checkpoints per loss config, so a crash partway
# only costs the in-flight config.
#
#   tmux new -s all_layer   # then inside: bash infra/run_all_layer_overnight.sh
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/src:${PYTHONPATH:-}"
cd "$REPO_ROOT"

TASK="${TASK:-caveman}"
RESULTS_DIR="results/$TASK"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="$RESULTS_DIR/all_layer_${TIMESTAMP}.log"
mkdir -p "$RESULTS_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

EVAL_N="${EVAL_N:-180}"

echo "################################################################"
echo "# A-PSR (all layers) + Multi-Gate ablation -- starting at $(date)"
echo "# Task: $TASK   Log: $LOG_FILE"
echo "################################################################"

echo ""
echo "== pre-flight checks =="
PREFLIGHT_OK=1
[ -f "openai.key" ] && echo "openai.key: found" || { echo "PRE-FLIGHT FAILURE: openai.key missing"; PREFLIGHT_OK=0; }
python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null && echo "CUDA: available" || { echo "PRE-FLIGHT FAILURE: no CUDA"; PREFLIGHT_OK=0; }
[ "$PREFLIGHT_OK" = "1" ] || { echo ""; echo "Aborting before touching GPU/API."; exit 1; }
echo "pre-flight checks passed."

FAILURES=()
run_step() {
    local desc="$1"; shift
    echo ""
    echo "== [$(date '+%Y-%m-%d %H:%M:%S')] START: $desc =="
    if "$@"; then
        echo "== [$(date '+%Y-%m-%d %H:%M:%S')] OK: $desc =="
    else
        echo "== [$(date '+%Y-%m-%d %H:%M:%S')] FAILED (continuing): $desc =="
        FAILURES+=("$desc")
    fi
}

# --direction-source trained     = A-PSR, gradient-trained directions (faithful H&V)
# --direction-source diff_in_means = Multi-Gate ablation, fixed DiM directions
# Omitting --layers is deliberate and load-bearing: that's what selects ALL layers.
run_step "A-PSR (trained directions), both loss configs" \
    python3 -u src/psr/all_layer/train.py --task "$TASK" --direction-source trained --sweep

run_step "Multi-Gate ablation (diff-in-means directions), both loss configs" \
    python3 -u src/psr/all_layer/train.py --task "$TASK" --direction-source diff_in_means --sweep

run_step "Multi-Gate + CONCEPTOR directions, both loss configs" \
    python3 -u src/psr/all_layer/train.py --task "$TASK" --direction-source conceptor --sweep

run_step "judged evaluation of all 6 configs (alone + combined) vs Base/Prompt" \
    python3 -u scripts/eval_all_layer_variants.py --task "$TASK" --n "$EVAL_N"

run_step "corrected-sign analysis + plots" \
    python3 -u scripts/analyze_all_layer_results.py --task "$TASK"

run_step "cosine similarity matrix + per-layer depth profile" \
    python3 -u scripts/compute_cosine_similarities.py --task "$TASK"

echo ""
echo "################################################################"
if [ "${#FAILURES[@]}" -eq 0 ]; then
    echo "# DONE at $(date). Completed with no failures."
else
    echo "# DONE at $(date) with failures:"
    for f in "${FAILURES[@]}"; do echo "#   - $f"; done
    echo "# Rerun this script -- completed configs are skipped (checkpoints already on disk)."
fi
echo "# Checkpoints: $RESULTS_DIR/psr_all_layer_probe_{mse,nll}.pt"
echo "#              $RESULTS_DIR/psr_multi_gate_probe_{mse,nll}.pt"
echo "#              $RESULTS_DIR/psr_multi_gate_conceptor_probe_{mse,nll}.pt"
echo "# Cosine:      $RESULTS_DIR/cosine_similarity_{matrix.csv,matrix.json,by_layer.csv}"
echo "# Plots:       plots/all_layer_{frontier,correct_bars,tokens_bars}.png"
echo "# Sweep rows:  $RESULTS_DIR/psr_{all_layer,multi_gate}_sweep.jsonl"
echo "# Eval table:  $RESULTS_DIR/all_layer_variants_eval.json"
echo "# Full log:    $LOG_FILE"
echo "################################################################"
