#!/usr/bin/env bash
# Settles where Stolfo's advantage comes from, in three stages of increasing cost.
#
# STAGE 1 (no training, ~40 min): the surface 2x2. Const and Stolfo already exist at ALL positions
#   (145.1 and 59.5 tokens); this adds both at PSR's response-only surface. Four cells, one
#   functional-form axis, one surface axis, no gate anywhere -> tells you immediately whether the
#   86-token gap is the clamp or the prompt access.
# STAGE 2 (~1h): the 3 missing +Prompt conditions.
# STAGE 3 (training, ~1.5h): SG+Clamp and MG+Clamp -- does PSR's gate add anything on top of the
#   clamp, and does that survive joint intervention.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/src:${PYTHONPATH:-}"
cd "$REPO_ROOT"
TASK="${TASK:-caveman}"
RESULTS_DIR="results/$TASK"; mkdir -p "$RESULTS_DIR"
TS="$(date '+%Y%m%d_%H%M%S')"; LOG="$RESULTS_DIR/clamp_investigation_${TS}.log"
exec > >(tee -a "$LOG") 2>&1

echo "################################################################"
echo "# Clamp investigation -- $(date)"
echo "# Log: $LOG"
echo "################################################################"
OK=1
[ -f openai.key ] && echo "openai.key: found" || { echo "FAIL: openai.key"; OK=0; }
python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null && echo "CUDA: ok" || { echo "FAIL: CUDA"; OK=0; }
[ "$OK" = 1 ] || { echo "Aborting."; exit 1; }

F=()
step() { local d="$1"; shift; echo ""; echo "== [$(date '+%H:%M:%S')] $d =="
  if "$@"; then echo "== OK: $d =="; else echo "== FAILED: $d =="; F+=("$d"); fi; }

echo ""; echo "########## STAGE 1: surface 2x2 (no training) ##########"
step "Stolfo, response-only surface" \
  python3 -u evals/layer_hparam_search.py --task "$TASK" --variant stolfo_resp
step "Const, response-only surface" \
  python3 -u evals/layer_hparam_search.py --task "$TASK" --variant const_resp

echo ""; echo "########## STAGE 2: missing +Prompt ##########"
step "+Prompt for stolfo, const, sg" \
  python3 -u scripts/run_prompt_plus.py --task "$TASK" --variants stolfo,const,sg

echo ""; echo "########## STAGE 3: gated clamp ##########"
# SG+Clamp at Stolfo's own winning layer -- read from its tiered search so the gate is compared
# against the ungated clamp at the SAME depth rather than an arbitrary one.
SL=$(python3 - <<'PY'
import json,glob
p=None
for c in ("results/caveman/nogate_clamp_tiered_search.jsonl","results/caveman/stolfo_tiered_search.jsonl"):
    try:
        rows=[json.loads(l) for l in open(c) if l.strip()]
        f=[r for r in rows if r.get("tier")=="final"]
        if f: p=f[0]["layer"]; break
    except FileNotFoundError: pass
print(p if p is not None else 16)
PY
)
echo "SG+Clamp will use layer $SL (Stolfo's own tiered-search winner)"
step "SG+Clamp sweep (layer $SL)" \
  python3 -u src/psr/clamp_gate/train.py --task "$TASK" --layers "$SL" --sweep
step "MG+Clamp sweep (all layers)" \
  python3 -u src/psr/clamp_gate/train.py --task "$TASK" --sweep

echo ""
echo "################################################################"
if [ ${#F[@]} -eq 0 ]; then echo "# DONE $(date). No failures."
else echo "# DONE $(date) with failures:"; for x in "${F[@]}"; do echo "#   - $x"; done; fi
echo "# Then: python3 scripts/inventory.py --task $TASK"
echo "################################################################"
