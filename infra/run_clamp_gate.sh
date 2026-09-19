#!/usr/bin/env bash
# Gated clamp only: train SG+Clamp and MG+Clamp, then generate and report avg_tokens.
# ZERO OpenAI calls -- training scores nothing, and avg_tokens comes from the local tokenizer.
# Safe to run while rate-limited. Judge later with:
#   python3 scripts/eval_clamp_gate.py --task caveman --judge-saved
#
# Deliberately does NOT include the old run_clamp_investigation.sh's Stage 1/2: const_resp there
# re-searches a 130-point grid (hours of judge calls, and the wrong experiment -- use
# scripts/surface_ablation.py for a config-matched surface test instead), and the +Prompt stage
# needs request budget.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/src:${PYTHONPATH:-}"
cd "$REPO_ROOT"
TASK="${TASK:-caveman}"
RESULTS_DIR="results/$TASK"; mkdir -p "$RESULTS_DIR"
TS="$(date '+%Y%m%d_%H%M%S')"; LOG="$RESULTS_DIR/clamp_gate_${TS}.log"
exec > >(tee -a "$LOG") 2>&1

echo "################################################################"
echo "# Gated clamp (SG+Clamp, MG+Clamp) -- $(date)"
echo "# No API calls in this script. Log: $LOG"
echo "################################################################"

python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null \
  && echo "CUDA: ok" || { echo "FAIL: no CUDA"; exit 1; }
# Teacher-forced responses must already be cached, or training would try to generate them.
for f in "cache/$TASK/teacher_responses_train.json" "cache/$TASK/teacher_responses_dev.json"; do
  [ -f "$f" ] && echo "cache: $(basename "$f")" || echo "NOTE: $f missing -- will be regenerated (GPU time, still no API)"
done

F=()
step() { local d="$1"; shift; echo ""; echo "== [$(date '+%H:%M:%S')] $d =="
  if "$@"; then echo "== OK: $d =="; else echo "== FAILED: $d =="; F+=("$d"); fi; }

# SG+Clamp at the ungated clamp's OWN winning layer, so the gate is compared against the
# baseline it's meant to improve on at matched depth rather than an arbitrary one.
SL=$(python3 - <<'PY'
import json
for c in ("results/caveman/nogate_clamp_tiered_search.jsonl","results/caveman/stolfo_tiered_search.jsonl"):
    try:
        f=[json.loads(l) for l in open(c) if l.strip()]
        f=[r for r in f if r.get("tier")=="final"]
        if f: print(f[0]["layer"]); break
    except FileNotFoundError: pass
else: print(16)
PY
)
echo ""; echo "SG+Clamp layer: $SL (ungated clamp's tiered-search winner)"

step "SG+Clamp sweep (layer $SL, 2 loss configs)" \
  python3 -u src/psr/clamp_gate/train.py --task "$TASK" --layers "$SL" --sweep
step "MG+Clamp sweep (all layers, 2 loss configs)" \
  python3 -u src/psr/clamp_gate/train.py --task "$TASK" --sweep
step "generate + avg_tokens (no judging)" \
  python3 -u scripts/eval_clamp_gate.py --task "$TASK"

echo ""
echo "################################################################"
if [ ${#F[@]} -eq 0 ]; then echo "# DONE $(date). No failures."
else echo "# DONE $(date) with failures:"; for x in "${F[@]}"; do echo "#   - $x"; done; fi
echo "# Checkpoints: $RESULTS_DIR/{sg,mg}_clamp_probe_{mse,nll}.pt"
echo "# Tokens:      $RESULTS_DIR/clamp_gate_eval.json"
echo "# Judge later: python3 scripts/eval_clamp_gate.py --task $TASK --judge-saved"
echo "################################################################"
