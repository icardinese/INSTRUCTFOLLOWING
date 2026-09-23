#!/usr/bin/env bash
# Tonight's triage run on a fresh Colab A100. Run from a Colab cell:
#
#     from google.colab import drive; drive.mount('/content/drive')
#     !bash tonight.sh
#
# Restores from the Drive backup, symlinks results/ and cache/ so writes persist live, then
# launches a grid sized to actually finish inside a Colab session.
set -uo pipefail

BACKUP="/content/drive/Shareddrives/Eric/triage_backup"
REPO="/content/INSTRUCTFOLLOWING"

# ---- repo ----
if [ -d "$BACKUP/repo/.git" ]; then
    mkdir -p "$REPO" && rsync -a "$BACKUP/repo/" "$REPO/"
    echo "repo: restored from Drive (local patches intact)"
elif [ -d "$REPO/.git" ]; then
    echo "repo: already present at $REPO"
else
    git clone -q https://github.com/icardinese/INSTRUCTFOLLOWING.git "$REPO"
    echo "repo: cloned from GitHub"
fi
cd "$REPO" || exit 1
export PYTHONPATH="$REPO:$REPO/src:${PYTHONPATH:-}"

# ---- deps ----
pip install -q torch transformers accelerate langdetect openai matplotlib pyyaml 2>&1 | tail -1

# ---- restore data (not symlinked: read-only during the run, and small) ----
mkdir -p data
[ -d "$BACKUP/data" ] && rsync -a "$BACKUP/data/" data/

# ---- symlink results/ and cache/ straight into Drive ----
# Writes land in Drive as they happen, so a runtime death costs the in-progress grid point
# instead of everything since the last sync. Replaces the polling loop entirely.
for d in results cache; do
    if [ ! -L "$d" ]; then
        mkdir -p "$BACKUP/$d"
        [ -d "$d" ] && rsync -a "$d/" "$BACKUP/$d/"
        rm -rf "$d"
        ln -s "$BACKUP/$d" "$d"
    fi
    echo "$d -> $(readlink -f $d)"
done

# ---- verify the patches this task depends on actually made it ----
echo ""
echo "=== patch check ==="
python3 - <<'PY'
import sys
sys.path.insert(0, ".")
ok = True
try:
    from adapters.registry import TASK_CHOICES, steering_location_for
    assert "triage" in TASK_CHOICES, "triage not registered"
    assert steering_location_for("triage") == "question_and_answer", "triage not on QR"
    print("  triage registered, surface=QR")
except Exception as e:
    print(f"  FAIL: {e}"); ok = False
try:
    from steering.psr.reference_config import N_EPOCHS_MSE, WEIGHT_DECAY
    assert (N_EPOCHS_MSE, WEIGHT_DECAY) == (15, 1e-6)
    print("  PSR fidelity fixes present (15 epochs, wd=1e-6)")
except Exception as e:
    print(f"  FAIL: {e}"); ok = False
try:
    import importlib
    j = importlib.import_module("evals.triage.judge")
    assert j.RANK_BY == "primary_desc"
    print("  ranking axis = primary_desc (accuracy, not tokens)")
except Exception as e:
    print(f"  FAIL: {e}"); ok = False
sys.exit(0 if ok else 1)
PY
[ $? -ne 0 ] && { echo "Patches missing -- re-apply the fix scripts before running."; exit 1; }

# ---- resume state ----
echo ""
echo "=== already done ==="
python3 - <<'PY'
import json, pathlib
r = pathlib.Path("results/triage")
if r.exists():
    for f in sorted(r.glob("*_sweep.jsonl")):
        n = sum(1 for l in f.open() if l.strip())
        print(f"  {f.name:40s} {n:4d} points")
    for f in sorted(r.glob("*_tiered_search.jsonl")):
        rows = [json.loads(l) for l in f.open() if l.strip()]
        t = {}
        for row in rows: t[row.get("tier","?")] = t.get(row.get("tier","?"),0)+1
        print(f"  {f.name:40s} {t}")
    print(f"  checkpoints: {len(list(r.glob('*.pt')))}")
else:
    print("  none -- cold start")
PY

# ---- run ----
# A100 80GB -> MAX_BATCH_ROWS=11 (the calibrated default).
# Grid trimmed to 4 layers x 2 loss configs so it completes inside one Colab session; the full
# 13-layer grid is ~30-40h and will be cut off mid-run regardless. Refine around the winner on a
# persistent box later. Sweeps append per point, so nothing here is wasted if it does get cut.
echo ""
echo "=== launching ==="
export SWEEP_LAYERS="${SWEEP_LAYERS:-10,14,18,22}"
export MAX_BATCH_ROWS="${MAX_BATCH_ROWS:-11}"
echo "SWEEP_LAYERS=$SWEEP_LAYERS  MAX_BATCH_ROWS=$MAX_BATCH_ROWS"
exec bash infra/run_triage_overnight.sh
