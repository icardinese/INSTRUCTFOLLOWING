#!/usr/bin/env bash
# SG (single-gated, frozen diff-in-means direction): sweep -> tiered layer search -> combined.
# 26 sweep points (13 layers x 2 loss configs), no alpha dimension -- the cheapest trainable
# variant in the project, and the same grid size as S-PSR's.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/src:${PYTHONPATH:-}"
cd "$REPO_ROOT"

TASK="${TASK:-caveman}"
RESULTS_DIR="results/$TASK"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="$RESULTS_DIR/sg_${TIMESTAMP}.log"
mkdir -p "$RESULTS_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "################################################################"
echo "# SG (single-gated, diff-in-means) -- starting at $(date)"
echo "# Log: $LOG_FILE"
echo "################################################################"

echo ""
echo "== pre-flight =="
OK=1
[ -f openai.key ] && echo "openai.key: found" || { echo "FAIL: openai.key missing"; OK=0; }
python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null && echo "CUDA: available" || { echo "FAIL: no CUDA"; OK=0; }
[ "$OK" = 1 ] || { echo "Aborting before touching GPU/API."; exit 1; }

FAILURES=()
run_step() {
  local d="$1"; shift
  echo ""; echo "== [$(date '+%H:%M:%S')] START: $d =="
  if "$@"; then echo "== [$(date '+%H:%M:%S')] OK: $d =="
  else echo "== [$(date '+%H:%M:%S')] FAILED (continuing): $d =="; FAILURES+=("$d"); fi
}

run_step "SG sweep (26 points)" \
  python3 -u src/psr/single_gate/train.py --task "$TASK" --sweep

run_step "SG tiered layer search (n=20 -> n=180 final)" \
  python3 -u evals/layer_hparam_search.py --task "$TASK" --variant sg

run_step "cross-variant summary" \
  python3 -u evals/layer_hparam_search.py --task "$TASK" --summarize

echo ""
echo "################################################################"
if [ "${#FAILURES[@]}" -eq 0 ]; then echo "# DONE at $(date). No failures."
else echo "# DONE with failures:"; for f in "${FAILURES[@]}"; do echo "#   - $f"; done; fi
echo "# Sweep:  $RESULTS_DIR/psr_sg_sweep.jsonl"
echo "# Tiered: $RESULTS_DIR/sg_tiered_search.jsonl"
echo "# Probe:  $RESULTS_DIR/psr_sg_probe.pt"
echo "################################################################"
