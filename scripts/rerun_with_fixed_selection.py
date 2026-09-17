"""One driver for everything discussed: (1) re-select conceptor's/proper's tiered-search winners
under the fixed avg_tokens-primary ranking, paying real GPU/API cost ONLY where the fix actually
requires new evaluation, not by resweeping anything; (2) regenerate a real checkpoint file at each
variant's ACTUAL winning (layer, alpha, loss-config) -- `run_tiered_search`'s retrain functions
train in-memory for evaluation and never persist a .pt, so the checkpoint sweep()'s own MSE-based
selection saved is very likely NOT the tiered search's real winner; (3) run the direction-only
ablation (evals/ablation_direction_only.py) on both resulting checkpoints.

Why conceptor only needs its Final re-run, but proper needs Tier 2 AND Final re-run:
  conceptor's Tier 1 survivors [16, 4, 10] were a real tie at n=20 (not the ascending-layer-index
  bug) -- avg_tokens-primary ranking, re-applied to the SAME already-evaluated Tier 1 candidates,
  is very likely to pick the same survivor set (or a substantively similar one), so Tier 2 is
  reused as-is. But Tier 2's WINNER among those survivors (layer 16's 5 alpha/loss points) DOES
  change under the fix -- alpha=2.0 (85.55 avg_tokens) beats alpha=1.0 (92.15) once avg_tokens
  actually drives the ranking, so only Final needs to re-run.
  proper's Tier 1 survivors [2, 8, 10] were the OTHER failure mode -- a real ascending-layer-index
  artifact discarding layer 18 (105.8 avg_tokens) in favor of ties at 147-150. Re-ranking the same
  Tier 1 data under the fix should surface a materially different survivor set (something in the
  12-18 range), which the existing Tier 2 data (evaluated only at layers 2/8/10) doesn't cover at
  all -- so Tier 2 must be freshly evaluated at whatever the corrected survivors turn out to be,
  and Final after that.
Both are determined by the actual re-selection at runtime, not hardcoded here -- this script reads
whatever `select_constrained_survivors` (now avg_tokens-primary) actually decides, for either
variant, rather than assuming the reasoning above holds exactly.

Nothing is deleted -- every JSONL this script truncates gets backed up with a timestamp suffix
first. Reruns are done via run_tiered_search directly (same resumability/skip-logic as running the
CLI), so a crash partway through this script is safe to just rerun from the top.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for `evals.*`/`src.*` imports

from adapters.registry import get_adapter
from evals.layer_hparam_search import (
    DEFAULT_FINAL_N, DEFAULT_TIER1_N, DEFAULT_TIER2_N, DEFAULT_TOP_K_LAYERS,
    load_existing_tiered_results, run_tiered_search, summarize_all_variants,
)

TASK = "caveman"


def backup_and_truncate(out_path: Path, tiers_to_keep: set[str], timestamp: str) -> None:
    """Copies out_path aside (untouched, full original content), then rewrites out_path with only
    the rows whose "tier" is in tiers_to_keep -- run_tiered_search's own resumability logic reads
    this back and treats missing tiers as "not done yet", so this is what forces exactly the
    tiers that need fresh evaluation to actually get it."""
    if not out_path.exists():
        print(f"  {out_path} doesn't exist -- nothing to truncate, run_tiered_search will do a full fresh run")
        return
    backup_path = out_path.with_suffix(out_path.suffix + f".pre_avg_tokens_fix_{timestamp}")
    backup_path.write_text(out_path.read_text())
    print(f"  backed up: {out_path} -> {backup_path}")
    kept = []
    with out_path.open() as f:
        for line in f:
            row = json.loads(line)
            if row["tier"] in tiers_to_keep:
                kept.append(line)
    with out_path.open("w") as f:
        f.writelines(kept)
    print(f"  rewrote {out_path}: kept tiers {sorted(tiers_to_keep)} ({len(kept)} rows)")


def rerun_variant(adapter, variant: str, tiers_to_keep: set[str], timestamp: str,
                   tier1_n: int, tier2_n: int, final_n: int, top_k_layers: int) -> dict:
    out_path = adapter.RESULTS_DIR / f"{variant}_tiered_search.jsonl"
    print(f"\n== {variant}: truncating to force re-selection under the fixed ranking ==")
    backup_and_truncate(out_path, tiers_to_keep, timestamp)

    print(f"\n== {variant}: running (reuses whatever's left on disk, evaluates only what's missing) ==")
    run_tiered_search(TASK, variant, tier1_n, tier2_n, final_n, top_k_layers)

    existing = load_existing_tiered_results(out_path)
    if "final" not in existing:
        raise RuntimeError(f"{variant}: no Final result after rerun -- check the run's own output above")
    winner = existing["final"][0]
    print(f"{variant} corrected winner: layer={winner['layer']} "
          f"alpha={winner.get('alpha', 'n/a')} mse_weight={winner['mse_weight']} nll_weight={winner['nll_weight']} "
          f"avg_tokens={winner['avg_tokens']:.1f}")
    return winner


def regenerate_checkpoint(adapter, variant: str, winner: dict, out_tag: str, seed: int) -> Path:
    """Runs that variant's train.py main() at the exact (layer, alpha, loss-config) the corrected
    tiered search actually picked, saved under out_tag so it never collides with whatever
    sweep()'s own MSE-based selection already saved under the plain filename."""
    env = os.environ.copy()
    env["FORCE_RERUN"] = "1"
    if variant == "proper":
        script = "src/psr/proper/train.py"
        env["PSR_PROPER_OUT_TAG"] = out_tag
        env["PSR_PROPER_MSE_WEIGHT"] = str(winner["mse_weight"])
        env["PSR_PROPER_NLL_WEIGHT"] = str(winner["nll_weight"])
        expected_path = adapter.RESULTS_DIR / f"psr_proper_probe{out_tag}.pt"
    elif variant == "conceptor":
        script = "src/psr/conceptor/train.py"
        env["PSR_CONCEPTOR_OUT_TAG"] = out_tag
        env["PSR_CONCEPTOR_ALPHA"] = str(winner["alpha"])
        env["PSR_CONCEPTOR_MSE_WEIGHT"] = str(winner["mse_weight"])
        env["PSR_CONCEPTOR_NLL_WEIGHT"] = str(winner["nll_weight"])
        expected_path = adapter.RESULTS_DIR / f"psr_conceptor_probe{out_tag}.pt"
    else:
        raise ValueError(variant)

    print(f"\n== {variant}: regenerating a real checkpoint at the corrected winner -> {expected_path} ==")
    subprocess.run(
        [sys.executable, script, "--task", TASK, "--layer", str(winner["layer"]), "--seed", str(seed)],
        check=True, env=env,
    )
    if not expected_path.exists():
        raise RuntimeError(f"expected {expected_path} after training but it's not there")
    return expected_path


def run_ablation(checkpoint_path: Path) -> None:
    print(f"\n== direction-only ablation on {checkpoint_path.name} ==")
    subprocess.run(
        [sys.executable, "evals/ablation_direction_only.py", "--task", TASK, "--checkpoint", checkpoint_path.name],
        check=True,
    )


def main(timestamp: str, tier1_n: int, tier2_n: int, final_n: int, top_k_layers: int, seed: int, skip_ablation: bool) -> None:
    adapter = get_adapter(TASK)

    conceptor_winner = rerun_variant(adapter, "conceptor", tiers_to_keep={"tier1", "tier2"},
                                      timestamp=timestamp, tier1_n=tier1_n, tier2_n=tier2_n,
                                      final_n=final_n, top_k_layers=top_k_layers)
    proper_winner = rerun_variant(adapter, "proper", tiers_to_keep={"tier1"},
                                   timestamp=timestamp, tier1_n=tier1_n, tier2_n=tier2_n,
                                   final_n=final_n, top_k_layers=top_k_layers)

    print("\n== cross-variant summary (includes matrix/selfproj from before, untouched by this fix) ==")
    summarize_all_variants(TASK)

    conceptor_ckpt = regenerate_checkpoint(adapter, "conceptor", conceptor_winner, "_tiered_winner", seed)
    proper_ckpt = regenerate_checkpoint(adapter, "proper", proper_winner, "_tiered_winner", seed)

    if not skip_ablation:
        run_ablation(proper_ckpt)
        run_ablation(conceptor_ckpt)
    else:
        print("\n--skip-ablation given -- checkpoints regenerated, ablation runs skipped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", required=True, help="passed in from the shell wrapper so backup filenames match the run's log filename")
    parser.add_argument("--tier1-n", type=int, default=DEFAULT_TIER1_N)
    parser.add_argument("--tier2-n", type=int, default=DEFAULT_TIER2_N)
    parser.add_argument("--final-n", type=int, default=DEFAULT_FINAL_N)
    parser.add_argument("--top-k-layers", type=int, default=DEFAULT_TOP_K_LAYERS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-ablation", action="store_true")
    args = parser.parse_args()
    main(args.timestamp, args.tier1_n, args.tier2_n, args.final_n, args.top_k_layers, args.seed, args.skip_ablation)
