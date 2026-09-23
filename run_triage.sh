#!/usr/bin/env bash
# Triage run. No options, no env vars, no layer flags.
#
# Every trainer below is called with ONLY --task and --sweep. That means each one uses its own
# built-in default grid, which is what it was written to do:
#
#   proper, single_gate, conceptor*  ->  DEFAULT_SWEEP_LAYERS = range(2,27,2), all 13 layers
#   all_layer (A-PSR)                ->  all 28 layers (that is what A-PSR IS)
#   clamp_gate, no --layers          ->  all layers = MG+Clamp
#
# There is exactly ONE place a layer flag appears in this file: the SG+Clamp loop, where a single
# layer is what makes it single-gated. Nowhere else. If you are reading this file looking for the
# thing that broke A-PSR last time, it was a --layers flag being passed to all_layer and
# clamp_gate. It is gone.
#
# Resume: every sweep appends per grid point and skips what is already done. Ctrl-C any time,
# rerun this script, it picks up where it stopped.
#
# Run it:  bash run_triage.sh
set -uo pipefail

cd /content/INSTRUCTFOLLOWING || exit 1
export PYTHONPATH="/content/INSTRUCTFOLLOWING:/content/INSTRUCTFOLLOWING/src:${PYTHONPATH:-}"

mkdir -p results/triage
LOG="results/triage/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

FAILED=()
step() {
    local name="$1"; shift
    echo ""
    echo "=== [$(date '+%H:%M:%S')] $name ==="
    if "$@"; then echo "--- OK: $name"; else echo "--- FAILED: $name"; FAILED+=("$name"); fi
}

echo "log: $LOG"
echo "start: $(date)"

# ---------------------------------------------------------------- sweeps
step "S-PSR (proper)" \
    python3 -u src/psr/proper/train.py --task triage --sweep

step "SG single-gate" \
    python3 -u src/psr/single_gate/train.py --task triage --sweep

# A-PSR: all 28 layers. No --layers. Three direction sources = three separate table rows.
step "A-PSR trained" \
    python3 -u src/psr/all_layer/train.py --task triage --direction-source trained --sweep
step "A-PSR diff_in_means" \
    python3 -u src/psr/all_layer/train.py --task triage --direction-source diff_in_means --sweep
step "A-PSR conceptor" \
    python3 -u src/psr/all_layer/train.py --task triage --direction-source conceptor --sweep

# MG+Clamp: no --layers means all layers.
step "MG+Clamp" \
    python3 -u src/psr/clamp_gate/train.py --task triage --sweep

# SG+Clamp: one layer per run is what makes it SINGLE-gated. This is the only --layers in the file.
for L in 2 4 6 8 10 12 14 16 18 20 22 24 26; do
    step "SG+Clamp layer $L" \
        python3 -u src/psr/clamp_gate/train.py --task triage --sweep --layers "$L"
done

# Fixed-vector conceptor ONLY. Its direction is a diff-in-means vector projected once through
# the conceptor matrix C, giving a rank-1 correction -- this is the SG+Conc / MG+Conc row.
#
# conceptor/matrix/ and conceptor/selfproj/ are deliberately NOT run. They are different methods,
# not variants of this one -- matrix applies C fresh at every position as C @ (mu_instr - h),
# selfproj is C @ h - h with no external target -- and handoff v3 section 5 already records their
# tiered searches as missing from disk at ~5x SG's cost to redo, with "probably skip" as the call.
step "Conceptor (fixed-vector, diff-in-means through C)" \
    python3 -u src/psr/conceptor/train.py --task triage --sweep

# ---------------------------------------------------------------- training-free baselines
for v in const stolfo const_resp stolfo_resp; do
    step "training-free: $v" \
        python3 -u evals/layer_hparam_search.py --task triage --variant "$v" --max-batch-rows 11
done

# ---------------------------------------------------------------- judged layer selection
# This is what actually picks the reported layer. The sweeps above only minimize dev MSE/NLL,
# which is not the metric the paper reports.
for v in proper sg sg_clamp conceptor; do
    step "judged layer search: $v" \
        python3 -u evals/layer_hparam_search.py --task triage --variant "$v" --max-batch-rows 11
done

step "tiered summary" \
    python3 -u evals/layer_hparam_search.py --task triage --summarize

# ---------------------------------------------------------------- generate, judge, report
step "generate" \
    python3 -u src/generate.py --task triage --split test
step "judge (exact match, no API)" \
    python3 -u evals/run_judge.py --task triage --split test
step "summarize" \
    python3 -u evals/summarize.py --task triage --split test
step "bootstrap CIs" \
    python3 -u evals/bootstrap_analysis.py --task triage --split test
step "token CIs" \
    python3 -u scripts/regen_token_cis.py --task triage --max-batch-rows 11
step "inventory" \
    python3 -u scripts/inventory.py --task triage
step "plots" \
    python3 -u evals/plotting.py everything --task triage --split test

# ---------------------------------------------------------------- done
echo ""
echo "================================"
echo "finished: $(date)"
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "all steps OK"
else
    echo "${#FAILED[@]} failed:"
    for f in "${FAILED[@]}"; do echo "  $f"; done
    echo "rerun this script -- completed work is skipped."
fi
echo "log: $LOG"
echo "majority-class baseline is ~64%. Anything near it is predicting 'no', not triaging."
