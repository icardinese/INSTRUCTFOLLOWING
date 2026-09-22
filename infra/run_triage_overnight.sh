#!/usr/bin/env bash
# One-shot, unattended, overnight run for the TRIAGE task: every trainable variant's full
# layer x hyperparameter sweep, every training-free baseline, the tiered judged layer search for
# each, then generate -> judge -> summarize -> bootstrap CIs -> every plot.
#
# RUN THIS IN A WAY THAT SURVIVES YOU DISCONNECTING:
#   tmux new -s triage       # then inside: bash infra/run_triage_overnight.sh
#   nohup bash infra/run_triage_overnight.sh > /dev/null 2>&1 &
# It also tees to results/triage/overnight_<timestamp>.log regardless.
#
# WHAT THIS COVERS THAT run_caveman_overnight.sh DOES NOT. That script predates three things and
# so sweeps only proper/conceptor/conceptor_matrix/conceptor_selfproj:
#   1. single_gate, all_layer and clamp_gate -- i.e. the SG+*, MG+* and SG+Clamp rows that are in
#      the results table. Without these the Gate x Direction grid has a hole in it.
#   2. evals/layer_hparam_search.py -- the tiered judged layer search. A raw --sweep only
#      minimizes dev MSE/NLL; it does NOT pick a layer by the metric the paper reports. Skipping
#      it means the layer winner was never validated against real judged performance.
#   3. The training-free baselines (const, stolfo, and their response-only surface twins), which
#      go through run_training_free_search rather than any --sweep.
# It also used the synchronous evals/run_judge.py path. For triage that distinction is moot --
# see the NO API KEY note below.
#
# NO API KEY IS NEEDED. Triage scoring is exact-match against gold labels written by
# scripts/build_triage_data.py (evals/triage/judge.py, zero API calls, ever). The caveman script's
# openai.key pre-flight check is deliberately absent here -- including it would abort a run that
# has no need of a key. This is also why there is no judge_via_batch.py step: nothing to batch.
#
# RANKING AXIS. evals/triage/judge.py declares RANK_BY="primary_desc", so the tiered search ranks
# survivors by classification accuracy, not avg_tokens. Triage has no length axis; ranking it by
# tokens would select the variant with the shortest reasoning regardless of whether the label is
# right. Do not "fix" this by passing a length-based override.
#
# Every step continues past a failure rather than aborting the night (deliberately NOT `set -e`
# for the long section) -- one crashed sweep point should not cost you the other hours. Failures
# are collected and printed at the end, and every *_sweep.jsonl is independently resumable
# (core/sweep.py), so rerunning this script picks up where anything failed rather than from
# scratch. The tiered searches are resumable too (load_existing_tiered_results).
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/src:${PYTHONPATH:-}"
cd "$REPO_ROOT"

TASK="triage"
RESULTS_DIR="results/$TASK"
mkdir -p "$RESULTS_DIR"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="$RESULTS_DIR/overnight_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

# Sweep grids -- override via env var to shrink for a smoke test, e.g.
#   SWEEP_LAYERS=14 bash infra/run_triage_overnight.sh
# Defaults are each variant's own full DEFAULT_* grid (nothing passed through) -- go all out.
SWEEP_LAYERS="${SWEEP_LAYERS:-}"
SWEEP_ALPHAS="${SWEEP_ALPHAS:-}"
LOSS_CONFIGS="${LOSS_CONFIGS:-}"
# Tier sizes for the judged search. Defaults match evals/layer_hparam_search.py
# (DEFAULT_TIER1_N=20, DEFAULT_TIER2_N=20, DEFAULT_FINAL_N=180).
TIER1_N="${TIER1_N:-}"
TIER2_N="${TIER2_N:-}"
FINAL_N="${FINAL_N:-}"

# BATCH ROW CAP -- the single most important knob for this task, and NOT safe at its default.
#
# evals/layer_hparam_search.py caps rows (candidates x prompts) per generate_with_routed_configs
# call at DEFAULT_MAX_BATCH_ROWS=60. That number was set after a REAL CUDA OOM on
# Qwen2.5-7B-Instruct at 300 rows on an 80GB A100 (see its docstring), with CAVEMAN-length
# prompts. Memory scales with batch_size x seq_len, so what matters is token-rows, not rows:
#
#   caveman safe : 60 rows  x ~469 tok  =  28,140 token-rows
#   caveman OOM  : 300 rows x ~469 tok  = 140,700 token-rows   <- measured failure
#   triage at 60 : 60 rows  x ~2400 tok = 144,000 token-rows   <- 1.02x THE MEASURED OOM
#
# Triage's instructed prompt is ~2,400 tokens against caveman's ~469, so the default would put
# this run essentially exactly at the known failure point. 11 is the equivalent-memory cap
# (28,140 / 2,400). Raise it only after watching nvidia-smi through a full Tier 2 call.
MAX_BATCH_ROWS="${MAX_BATCH_ROWS:-11}"

LAYER_ARG=(); [ -n "$SWEEP_LAYERS" ] && LAYER_ARG=(--sweep-layers "$SWEEP_LAYERS")
LAYERS_ARG=(); [ -n "$SWEEP_LAYERS" ] && LAYERS_ARG=(--layers "$SWEEP_LAYERS")
ALPHA_ARG=(); [ -n "$SWEEP_ALPHAS" ] && ALPHA_ARG=(--sweep-alphas "$SWEEP_ALPHAS")
LOSS_ARG=(); [ -n "$LOSS_CONFIGS" ] && LOSS_ARG=(--loss-configs "$LOSS_CONFIGS")
TIER_ARG=()
[ -n "$TIER1_N" ] && TIER_ARG+=(--tier1-n "$TIER1_N")
[ -n "$TIER2_N" ] && TIER_ARG+=(--tier2-n "$TIER2_N")
[ -n "$FINAL_N" ] && TIER_ARG+=(--final-n "$FINAL_N")
# Applies to every layer_hparam_search invocation (both the training-free searches and the
# tiered judged searches) -- those are the only steps that batch multiple configs per call.
TIER_ARG+=(--max-batch-rows "$MAX_BATCH_ROWS")

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
echo "# Overnight TRIAGE run starting at $(date)"
echo "# Log: $LOG_FILE"
echo "# max_batch_rows=$MAX_BATCH_ROWS (default 11 for triage, NOT the library default of 60 --"
echo "#   60 would sit at 1.02x the OOM measured on an 80GB A100; see the comment in this script)"
echo "################################################################"

# ---- Pre-flight: fail FAST, before spending hours of GPU time, not at 3am. ----
echo ""
echo "== pre-flight checks =="
PREFLIGHT_OK=1

# Gold labels must already exist. Without them evals/triage/judge.py raises on every row, so the
# whole night would train fine and then produce nothing scoreable.
for split in train dev test; do
    if [ ! -f "data/$TASK/${split}.jsonl" ]; then
        echo "PRE-FLIGHT FAILURE: data/$TASK/${split}.jsonl missing. Run:"
        echo "    python3 scripts/build_triage_data.py --aeslc third_party/AESLC/enron_subject_line"
        echo "    python3 scripts/build_triage_data.py --label"
        PREFLIGHT_OK=0
    fi
done
if [ "$PREFLIGHT_OK" = "1" ]; then
    echo "gold labels: found for train/dev/test"
    python3 -c "
import json, collections
for s in ['train','dev','test']:
    rows=[json.loads(l) for l in open(f'data/$TASK/{s}.jsonl')]
    d=collections.Counter(r['gold'] for r in rows)
    top=max(d.values())/len(rows)
    print(f'  {s}: n={len(rows)} dist={dict(d)} majority-baseline={top:.1%}')
"
fi

if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "PRE-FLIGHT FAILURE: torch.cuda.is_available() is False -- every train/generate script"
    echo "  in this pipeline hardcodes device=\"cuda\" and will crash immediately without a GPU."
    PREFLIGHT_OK=0
else
    echo "CUDA: available"
fi

# openai is NOT required for triage (exact-match judging) but IS imported by the shared
# evals/run_judge.py module chain, so check importability without requiring a key.
if ! python3 -c "import torch, transformers, matplotlib" 2>/dev/null; then
    echo "PRE-FLIGHT FAILURE: torch/transformers/matplotlib failed to import --"
    echo "  run 'pip install -r requirements.txt' first."
    PREFLIGHT_OK=0
else
    echo "core dependencies: importable"
fi

# One real training step before committing the night. Triage's instructed prompt is ~2,400 tokens
# against caveman's ~469, and QR steers essentially all of it, so the backward-pass activation
# footprint is several times larger. Finding an OOM here costs a minute; finding it at 3am costs
# the night.
echo "-- smoke test: one triage training point (catches OOM at QR's ~2.4k-token span) --"
if SWEEP_LAYERS=14 LOSS_CONFIGS="1.0:0.0" timeout 1800 python3 -u src/psr/proper/train.py \
        --task "$TASK" --sweep --sweep-layers 14 --loss-configs "1.0:0.0" >/dev/null 2>&1; then
    echo "smoke test: PASSED (one point trained end to end)"
else
    echo "PRE-FLIGHT FAILURE: a single training point failed. Most likely CUDA OOM at triage's"
    echo "  prompt length -- note this smoke test is batch-size 1, so an OOM HERE means a single"
    echo "  2.4k-token backward pass does not fit and lowering the batch cap will NOT help."
    echo "  Instead, shorten the prompts:"
    echo "    python3 scripts/build_triage_data.py --aeslc <path> --max-body-chars 800"
    echo "    python3 scripts/build_triage_data.py --label"
    echo "  or drop a slot from the ablation (adapters/triage_adapter.py SLOT_ABLATIONS)."
    echo "  If instead a LATER phase OOMs during judged eval, that IS the batch cap -- lower it:"
    echo "    MAX_BATCH_ROWS=6 bash infra/run_triage_overnight.sh"
    PREFLIGHT_OK=0
fi

if [ "$PREFLIGHT_OK" != "1" ]; then
    echo ""
    echo "Aborting -- fix the above and rerun. Sweeps are resumable, so nothing already done is lost."
    exit 1
fi
echo "pre-flight checks passed -- proceeding."

# ================================================================
# PHASE 1: every trainable variant's full layer x hyperparameter sweep.
# Writes results/triage/psr_<variant>_sweep.jsonl (dev mse/nll per point). These sweeps are the
# cheap prefilter -- they do NOT pick the reported layer. Phase 3 does that with judged eval.
# ================================================================

run_step "S-PSR / PSR-Proper sweep (layer x loss-config)" \
    python3 -u src/psr/proper/train.py --task "$TASK" --sweep "${LAYER_ARG[@]}" "${LOSS_ARG[@]}"

run_step "SG single-gate sweep (layer x loss-config)" \
    python3 -u src/psr/single_gate/train.py --task "$TASK" --sweep "${LAYER_ARG[@]}" "${LOSS_ARG[@]}"

run_step "SG+Clamp gated-clamp sweep (layer x loss-config)" \
    python3 -u src/psr/clamp_gate/train.py --task "$TASK" --sweep "${LAYERS_ARG[@]}" "${LOSS_ARG[@]}"

# A-PSR / MG: one sweep per direction source. All three are real rows in the Gate x Direction
# grid (MG+GradientTrained, MG+DiM, MG+Conc), so all three get swept.
for src in trained diff_in_means conceptor; do
    run_step "MG all-layer sweep (direction_source=$src)" \
        python3 -u src/psr/all_layer/train.py --task "$TASK" --direction-source "$src" \
            --sweep "${LAYERS_ARG[@]}" "${LOSS_ARG[@]}"
done

run_step "Conceptor fixed-vector sweep (layer x alpha x loss-config)" \
    python3 -u src/psr/conceptor/train.py --task "$TASK" --sweep "${LAYER_ARG[@]}" "${ALPHA_ARG[@]}" "${LOSS_ARG[@]}"
run_step "Conceptor/matrix sweep (layer x alpha x loss-config)" \
    python3 -u src/psr/conceptor/matrix/train.py --task "$TASK" --sweep "${LAYER_ARG[@]}" "${ALPHA_ARG[@]}" "${LOSS_ARG[@]}"
run_step "Conceptor/selfproj sweep (layer x adaptive-alpha x loss-config)" \
    python3 -u src/psr/conceptor/selfproj/train.py --task "$TASK" --sweep "${LAYER_ARG[@]}" "${LOSS_ARG[@]}"

# ================================================================
# PHASE 2: training-free baselines. No gradient descent, so no --sweep; these go through
# run_training_free_search's dedicated 2-tier flow (full grid at Tier 1, winner confirmed at
# Final) rather than the MSE-prefilter machinery, which has no training loss to prefilter with.
# const/const_resp additionally sweep the additive scalar coefficient.
# ================================================================

for variant in const stolfo const_resp stolfo_resp; do
    run_step "training-free search: $variant" \
        python3 -u evals/layer_hparam_search.py --task "$TASK" --variant "$variant" "${TIER_ARG[@]}"
done

# ================================================================
# PHASE 3: the tiered JUDGED layer search per trainable variant. This is what actually selects
# the reported layer, using real triage accuracy (ranked by RANK_BY="primary_desc"), with a
# correctness gate against the grid's best and a soft gate against Prompt-alone.
# Requires Phase 1's psr_<variant>_sweep.jsonl to exist -- hence the ordering.
# ================================================================

for variant in proper sg sg_clamp conceptor conceptor_matrix conceptor_selfproj; do
    run_step "tiered judged layer search: $variant" \
        python3 -u evals/layer_hparam_search.py --task "$TASK" --variant "$variant" "${TIER_ARG[@]}"
done

run_step "tiered search cross-variant summary" \
    python3 -u evals/layer_hparam_search.py --task "$TASK" --summarize

# ================================================================
# PHASE 4: generation, judging, summary, CIs, plots.
# ================================================================

run_step "generate every available condition" \
    python3 -u src/generate.py --task "$TASK" --split test
run_step "judge every generated response (exact match, zero API calls)" \
    python3 -u evals/run_judge.py --task "$TASK" --split test
run_step "summarize" \
    python3 -u evals/summarize.py --task "$TASK" --split test
run_step "bootstrap 95% CIs" \
    python3 -u evals/bootstrap_analysis.py --task "$TASK" --split test
# regen_token_cis.py batches via generate_batched_uniform and has its own --max-batch-rows
# (same library default of 60), so it needs the triage cap too. src/generate.py above does NOT
# batch -- it is a per-item generate_response loop -- so it has no cap to set and no OOM risk
# from batch width, only from a single 2.4k-token prompt.
run_step "per-response token counts + CIs on the SAME responses" \
    python3 -u scripts/regen_token_cis.py --task "$TASK" --max-batch-rows "$MAX_BATCH_ROWS"
run_step "inventory (Gate x Direction grid coverage, degenerate-run flags)" \
    python3 -u scripts/inventory.py --task "$TASK"
run_step "plot everything" \
    python3 -u evals/plotting.py everything --task "$TASK" --split test

# ---- Summary ----
END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))
N_PLOTS=$(find "$RESULTS_DIR/plots" -name "*.png" 2>/dev/null | wc -l | tr -d ' ')

echo ""
echo "################################################################"
echo "# Overnight TRIAGE run finished at $(date)"
echo "# Elapsed: $((ELAPSED / 3600))h $(((ELAPSED % 3600) / 60))m"
echo "# Plots written: $N_PLOTS"
echo "# Log: $LOG_FILE"
if [ ${#FAILURES[@]} -eq 0 ]; then
    echo "# All steps OK."
else
    echo "# ${#FAILURES[@]} step(s) FAILED (the rest still ran):"
    for f in "${FAILURES[@]}"; do echo "#   - $f"; done
    echo "# Sweeps and tiered searches are resumable -- rerunning this script retries only"
    echo "# what did not finish."
fi
echo "################################################################"
echo ""
echo "FIRST THINGS TO CHECK IN THE MORNING:"
echo "  1. Majority-class baseline printed in pre-flight above (~64% for 'no'). Any variant"
echo "     scoring near it is not doing real triage -- it is predicting the majority class."
echo "  2. results/$TASK/tiered_search_summary.json -- the selected layer per variant."
echo "  3. 'parsed' rate in the judged output. A low rate means the Triage: <label> format is"
echo "     not landing and accuracy numbers are measuring format failure, not classification."
echo "  4. Any prompt_floor_fallback=True in the tiered logs -- means nothing beat Prompt-alone."
