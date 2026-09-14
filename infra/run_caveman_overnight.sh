#!/usr/bin/env bash
# One-shot, unattended, overnight run: every layer + hyperparameter sweep, every trainable PSR
# variant, caveman only, then generate -> judge -> summarize -> bootstrap CIs -> every plot we
# can make. Designed to be started once and left alone for hours.
#
# RUN THIS IN A WAY THAT SURVIVES YOU DISCONNECTING -- a plain `bash infra/run_caveman_overnight.sh`
# in a foreground SSH session dies the moment your laptop sleeps or the connection drops. Use one of:
#   tmux new -s overnight    # then inside: bash infra/run_caveman_overnight.sh
#   nohup bash infra/run_caveman_overnight.sh > /dev/null 2>&1 &
# This script also tees its own output to results/caveman/overnight_<timestamp>.log regardless,
# so you have a real log to read in the morning either way.
#
# Loss configs swept per (layer[, alpha]) point: 5 -- pure MSE, three MSE+NLL blends, and pure
# NLL (mse_weight=0). See src/psr/proper/train.py's DEFAULT_LOSS_CONFIG_GRID. This is ~1040 total
# training runs across the four variants at full default grids (was ~832 before pure-NLL was
# added) -- proper: 13 layers x 5 configs = 65; conceptor/matrix/selfproj: 13 x 5 alpha(or
# percentile) x 5 configs = 325 each.
#
# What this does NOT do, on purpose:
#   - const steering calibration (src/const/sweep.py). Picking its winning (layer, coefficient)
#     needs a human to judge sweep_dev.jsonl -- that was true before this extension and is still
#     true now; automating it would mean silently guessing, not computing. generate.py already
#     gracefully skips the "const" condition if const_steer_config.json doesn't exist, so nothing
#     downstream breaks by skipping this. Run src/const/sweep.py + write const_steer_config.json
#     by hand separately if you want that condition included later.
#   - ifeval. Caveman only, per tonight's ask.
#   - S-PSR/A-PSR sweeps. These are the deliberately-preserved pre-fidelity-fix baselines (see
#     ARCHITECTURE.md) -- S-PSR runs once at layer 14 (empirically near-optimal for PSR-Proper per
#     the project's own earlier full layer sweep, see EXTENSION_CHANGES.md finding #4), A-PSR runs
#     once across its own automatic multi-layer fraction set. Neither is part of "the loss
#     function variations we built" -- they don't use train_gate/forward_with_gate_hook at all.
#
# Every step below continues past a failure rather than aborting the whole night (deliberately NOT
# `set -e` for the long section) -- a single crashed sweep point or a transient API error
# shouldn't cost you the other ~7 hours of results. Failures are collected and printed in the
# final summary, and every *_sweep.jsonl is independently resumable (core/sweep.py), so rerunning
# this exact script tomorrow picks up where anything failed, not from scratch.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/src:${PYTHONPATH:-}"
cd "$REPO_ROOT"

TASK="caveman"
RESULTS_DIR="results/$TASK"
mkdir -p "$RESULTS_DIR"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="$RESULTS_DIR/overnight_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

# Sweep grids -- override any of these via env var if you want to shrink/expand before starting,
# e.g. `SWEEP_LAYERS=10,14,18 bash infra/run_caveman_overnight.sh` for a quick smoke test first.
# Defaults are each variant's own full DEFAULT_* grid (nothing passed through) -- "go all out."
# LOSS_CONFIGS is "mse_weight:nll_weight" pairs, e.g. "1.0:0.0,0.0:1.0" for pure-MSE + pure-NLL
# only -- the default (unset) grid is 5 points: pure MSE, three MSE+NLL blends, and pure NLL. See
# src/psr/proper/train.py's DEFAULT_LOSS_CONFIG_GRID for why these are paired points, not two
# independent grids (mse_weight=0 with a small nll_weight isn't a meaningful configuration).
SWEEP_LAYERS="${SWEEP_LAYERS:-}"
SWEEP_ALPHAS="${SWEEP_ALPHAS:-}"
LOSS_CONFIGS="${LOSS_CONFIGS:-}"

LAYER_ARG=(); [ -n "$SWEEP_LAYERS" ] && LAYER_ARG=(--sweep-layers "$SWEEP_LAYERS")
ALPHA_ARG=(); [ -n "$SWEEP_ALPHAS" ] && ALPHA_ARG=(--sweep-alphas "$SWEEP_ALPHAS")
LOSS_ARG=(); [ -n "$LOSS_CONFIGS" ] && LOSS_ARG=(--loss-configs "$LOSS_CONFIGS")

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
echo "# Overnight caveman run starting at $(date)"
echo "# Log: $LOG_FILE"
echo "################################################################"

# ---- Pre-flight checks: fail FAST, before spending hours of GPU time, not at 3am. ----
echo ""
echo "== pre-flight checks =="
PREFLIGHT_OK=1

if [ ! -f "openai.key" ]; then
    echo "PRE-FLIGHT FAILURE: openai.key not found at repo root -- evals/run_judge.py needs it"
    echo "  for BOTH the correctness/coherence AND the new conciseness judge call. Create it"
    echo "  (a plain text file containing your API key, nothing else) before running this."
    PREFLIGHT_OK=0
else
    echo "openai.key: found"
fi

if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "PRE-FLIGHT FAILURE: torch.cuda.is_available() is False -- every train/generate script"
    echo "  in this pipeline hardcodes device=\"cuda\" and will crash immediately without a GPU."
    PREFLIGHT_OK=0
else
    echo "CUDA: available"
fi

if ! python3 -c "import torch, transformers, openai, matplotlib" 2>/dev/null; then
    echo "PRE-FLIGHT FAILURE: one of torch/transformers/openai/matplotlib failed to import --"
    echo "  run 'pip install -r requirements.txt' first."
    PREFLIGHT_OK=0
else
    echo "core dependencies: importable"
fi

if [ "$PREFLIGHT_OK" != "1" ]; then
    echo ""
    echo "Aborting BEFORE touching the GPU -- fix the above and rerun. Nothing was trained."
    exit 1
fi
echo "pre-flight checks passed -- proceeding."

# ---- Baselines (S-PSR at the empirically-justified layer 14, A-PSR automatic) ----
run_step "S-PSR baseline (layer=14)" \
    python3 -u src/psr/old_baseline/train.py --task "$TASK" --layer 14
run_step "A-PSR baseline (automatic multi-layer)" \
    python3 -u src/psr/old_baseline/train_a_psr.py --task "$TASK"

# ---- Every trainable variant's full layer x hyperparameter sweep ----
run_step "PSR-Proper sweep (layer x loss-config)" \
    python3 -u src/psr/proper/train.py --task "$TASK" --sweep "${LAYER_ARG[@]}" "${LOSS_ARG[@]}"
run_step "Conceptor fixed-vector sweep (layer x alpha x loss-config)" \
    python3 -u src/psr/conceptor/train.py --task "$TASK" --sweep "${LAYER_ARG[@]}" "${ALPHA_ARG[@]}" "${LOSS_ARG[@]}"
run_step "Conceptor/matrix sweep (layer x alpha x loss-config)" \
    python3 -u src/psr/conceptor/matrix/train.py --task "$TASK" --sweep "${LAYER_ARG[@]}" "${ALPHA_ARG[@]}" "${LOSS_ARG[@]}"
run_step "Conceptor/selfproj sweep (layer x adaptive-alpha x loss-config)" \
    python3 -u src/psr/conceptor/selfproj/train.py --task "$TASK" --sweep "${LAYER_ARG[@]}" "${LOSS_ARG[@]}"

# ---- Generation, judging, summary, CIs ----
run_step "generate every available condition" \
    python3 -u src/generate.py --task "$TASK" --split test
run_step "judge every generated response (correctness + coherence + conciseness)" \
    python3 -u evals/run_judge.py --task "$TASK" --split test
run_step "summarize" \
    python3 -u evals/summarize.py --task "$TASK" --split test
run_step "bootstrap 95% CIs" \
    python3 -u evals/bootstrap_analysis.py --task "$TASK" --split test

# ---- Every plot we can make: judged-result plots + every discovered sweep file, every metric,
#      every hyperparameter grouping. See evals/plotting.py's plot_everything(). ----
run_step "plot everything" \
    python3 -u evals/plotting.py everything --task "$TASK" --split test

# ---- Summary ----
END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))
N_PLOTS=$(find "$RESULTS_DIR/plots" -name "*.png" 2>/dev/null | wc -l | tr -d ' ')

echo ""
echo "################################################################"
echo "# DONE at $(date). Elapsed: $((ELAPSED / 3600))h $(((ELAPSED % 3600) / 60))m"
echo "# Plots written: $N_PLOTS  (in $RESULTS_DIR/plots/)"
echo "# Sweep files:   $(find "$RESULTS_DIR" -maxdepth 1 -name '*_sweep.jsonl' | wc -l | tr -d ' ')"
if [ "${#FAILURES[@]}" -eq 0 ]; then
    echo "# All steps succeeded."
else
    echo "# ${#FAILURES[@]} step(s) failed (everything else still ran and is usable):"
    for f in "${FAILURES[@]}"; do
        echo "#   - $f"
    done
fi
echo "# Full log: $LOG_FILE"
echo "################################################################"
