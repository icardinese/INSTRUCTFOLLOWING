"""Rename result files to the Gate+Direction convention.

DRY RUN BY DEFAULT. Prints every planned action and does nothing. Add --apply to execute.

COPY BY DEFAULT (--mode copy). Copying means the training/search pipeline keeps finding its old
filenames and nothing breaks, while the new names exist for the paper, the inventory and the
figures. --mode move is the clean end state but REQUIRES the code rename (variant keys in
evals/layer_hparam_search.py, output paths in every src/psr/*/train.py, src/generate.py, and 9
test files -- roughly 250 references across 34 files), so it should wait until there's time to do
that properly and rerun the test suite.

WHY THE OLD NAMES HAVE TO GO. They encode implementation history rather than method identity, and
in three cases they are actively wrong:

  psr_proper_*        "proper" is not a method. It IS Heyman & Vandeputte's S-PSR (SG +
                      gradient-trained direction).
  psr_probe.pt        was labelled "S-PSR" but is the OFFLINE-regression baseline -- 200 epochs,
                      precomputed fixed activation pairs, no answer-only masking, no
                      subsequent-layers loss. It is not H&V's S-PSR and not comparable to anything
                      else here, so it is renamed legacy_offline_* rather than to any grid cell.
  a_psr_probe.pt      same problem one level up: N probes trained INDEPENDENTLY, no layer ever
                      seeing another's correction. Not A-PSR, and not MG either (MG is jointly
                      optimized in one shared forward pass).

The gate-stripped ablations move into the NoGate row: ablation_direction_only_<ckpt>.json takes a
trained checkpoint, discards the gate, and applies its frozen direction at one calibrated constant
coefficient -- mechanically Const's mechanism with a different direction. The new name records the
cell (nogate_<direction>) and that the direction came from a previously trained run.

Also rewrites the condition KEYS inside the all-layer eval JSON when --rewrite-keys is passed,
since renaming that file does not touch the psr_multi_gate_probe_* / psr_all_layer_probe_* stems
stored inside it.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path
from adapters.registry import TASK_CHOICES

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# old stem -> new stem. Explicit rather than pattern-based: several old names map to cells that a
# regex would get wrong (psr_probe -> legacy, not sg_dim), and an explicit table is auditable.
RENAMES = {
    # --- tiered searches (judged layer/hyperparameter selection, alone condition) ---
    "proper_tiered_search.jsonl":             "sg_gradient_trained_tiered_search.jsonl",
    "conceptor_tiered_search.jsonl":          "sg_conc_tiered_search.jsonl",
    "sg_tiered_search.jsonl":                 "sg_dim_tiered_search.jsonl",
    "conceptor_matrix_tiered_search.jsonl":   "sg_conc_matrix_tiered_search.jsonl",
    "conceptor_selfproj_tiered_search.jsonl": "sg_conc_selfproj_tiered_search.jsonl",
    "const_tiered_search.jsonl":              "nogate_dim_tiered_search.jsonl",
    "stolfo_tiered_search.jsonl":             "nogate_clamp_tiered_search.jsonl",
    # --- training sweeps ---
    "psr_proper_sweep.jsonl":                 "sg_gradient_trained_sweep.jsonl",
    "psr_conceptor_sweep.jsonl":              "sg_conc_sweep.jsonl",
    "psr_sg_sweep.jsonl":                     "sg_dim_sweep.jsonl",
    "psr_conceptor_matrix_sweep.jsonl":       "sg_conc_matrix_sweep.jsonl",
    "psr_conceptor_selfproj_sweep.jsonl":     "sg_conc_selfproj_sweep.jsonl",
    "psr_all_layer_sweep.jsonl":              "mg_gradient_trained_sweep.jsonl",
    "psr_multi_gate_sweep.jsonl":             "mg_dim_sweep.jsonl",
    "psr_multi_gate_conceptor_sweep.jsonl":   "mg_conc_sweep.jsonl",
    # --- checkpoints ---
    "psr_proper_probe.pt":                        "sg_gradient_trained_probe.pt",
    "psr_proper_probe_tiered_winner.pt":          "sg_gradient_trained_probe_tiered_winner.pt",
    "psr_conceptor_probe.pt":                     "sg_conc_probe.pt",
    "psr_conceptor_probe_tiered_winner.pt":       "sg_conc_probe_tiered_winner.pt",
    "psr_sg_probe.pt":                            "sg_dim_probe.pt",
    "psr_conceptor_matrix_probe.pt":              "sg_conc_matrix_probe.pt",
    "psr_conceptor_selfproj_probe.pt":            "sg_conc_selfproj_probe.pt",
    "psr_all_layer_probe_mse.pt":                 "mg_gradient_trained_probe_mse.pt",
    "psr_all_layer_probe_nll.pt":                 "mg_gradient_trained_probe_nll.pt",
    "psr_multi_gate_probe_mse.pt":                "mg_dim_probe_mse.pt",
    "psr_multi_gate_probe_nll.pt":                "mg_dim_probe_nll.pt",
    "psr_multi_gate_conceptor_probe_mse.pt":      "mg_conc_probe_mse.pt",
    "psr_multi_gate_conceptor_probe_nll.pt":      "mg_conc_probe_nll.pt",
    # NOT grid cells -- see module docstring. Named legacy_* so they can never be mistaken for
    # SG+DiM or MG+DiM, which are different methods with a different training regime.
    "psr_probe.pt":                               "legacy_offline_dim_probe.pt",
    "a_psr_probe.pt":                             "legacy_offline_multilayer_dim_probe.pt",
    # --- +Prompt condition ---
    "prompt_plus_steer_proper.json":          "sg_gradient_trained_prompt_plus.json",
    "prompt_plus_steer_conceptor.json":       "sg_conc_prompt_plus.json",
    "prompt_plus_steer_sg.json":              "sg_dim_prompt_plus.json",
    # --- gate-stripped ablations -> NoGate row ---
    "ablation_direction_only_psr_proper_probe_tiered_winner.json":    "nogate_gradient_trained_eval.json",
    "ablation_direction_only_psr_conceptor_probe_tiered_winner.json": "nogate_conc_eval.json",
    "ablation_direction_only_psr_sg_probe.json":                      "nogate_dim_from_sg_eval.json",
    # --- multi-condition eval + train logs ---
    "all_layer_variants_eval.json":           "mg_variants_eval.json",
    "psr_proper_train_log.json":              "sg_gradient_trained_train_log.json",
    "psr_conceptor_train_log.json":           "sg_conc_train_log.json",
    "psr_conceptor_train_log_tiered_winner.json": "sg_conc_train_log_tiered_winner.json",
    "psr_train_log.json":                     "legacy_offline_dim_train_log.json",
    "a_psr_train_log.json":                   "legacy_offline_multilayer_dim_train_log.json",
}

# Condition-key stems inside mg_variants_eval.json (keys look like "<stem>_<loss>_<condition>").
KEY_RENAMES = {
    "psr_all_layer_probe":            "mg_gradient_trained",
    "psr_multi_gate_probe":           "mg_dim",
    "psr_multi_gate_conceptor_probe": "mg_conc",
}

# Files that legitimately have no grid cell -- listed so the script can say "known, skipping"
# instead of leaving them in an "unmapped" pile that looks like an oversight.
KNOWN_NON_METHOD = {
    "summary_test.json", "summary_dev.json", "tiered_search_summary.json",
    "cosine_similarity_matrix.json", "cosine_similarity_matrix.csv",
    "cosine_similarity_by_layer.csv", "inventory.csv",
    "generations_test.jsonl", "generations_dev.jsonl",
    "judged_test.jsonl", "judged_dev.jsonl",
    "const_steer_config.json", "const_steer_directions.pt",
}


def rewrite_keys(path: Path, apply: bool) -> int:
    """Renames the condition-key stems inside the all-layer eval JSON. Returns keys changed."""
    with path.open() as f:
        payload = json.load(f)
    results = payload.get("results")
    if not isinstance(results, dict):
        return 0
    new_results, changed = {}, 0
    for key, val in results.items():
        new_key = key
        for old, new in KEY_RENAMES.items():
            if key.startswith(old):
                new_key = new + key[len(old):]
                break
        if new_key != key:
            changed += 1
        new_results[new_key] = val
    if changed and apply:
        payload["results"] = new_results
        with path.open("w") as f:
            json.dump(payload, f, indent=2)
    return changed


def main(task: str, apply: bool, mode: str, do_keys: bool) -> None:
    from adapters.registry import get_adapter
    results_dir = get_adapter(task).RESULTS_DIR
    present = {f.name for f in results_dir.iterdir() if f.is_file()}

    planned, collisions, missing = [], [], []
    for old, new in sorted(RENAMES.items()):
        if old not in present:
            missing.append(old)
            continue
        if new in present:
            collisions.append((old, new))
            continue
        planned.append((old, new))

    unmapped = sorted(
        n for n in present
        if n not in RENAMES and n not in RENAMES.values()
        and n not in KNOWN_NON_METHOD and not n.endswith(".log")
        and not n.endswith(".pre_direction_fix") and ".pre_" not in n
    )

    verb = "COPY" if mode == "copy" else "MOVE"
    header = f"{'APPLYING' if apply else 'DRY RUN (no changes)'} -- {verb} mode"
    print(f"\n{'=' * 76}\n{header}\n{results_dir}\n{'=' * 76}")

    print(f"\nPLANNED ({len(planned)}):")
    for old, new in planned:
        print(f"  {old}\n    -> {new}")
    if not planned:
        print("  (nothing to do)")

    if collisions:
        print(f"\nSKIPPED -- destination already exists ({len(collisions)}):")
        for old, new in collisions:
            print(f"  {old} -> {new}  [{new} present; delete it first if you want to redo this]")

    if missing:
        print(f"\nNOT PRESENT ({len(missing)}) -- expected, these methods just haven't been run:")
        print("  " + ", ".join(missing))

    if unmapped:
        print(f"\nUNMAPPED ({len(unmapped)}) -- present but not in the rename table.")
        print("  Add a line to RENAMES if any of these is a method result:")
        for n in unmapped:
            print(f"    {n}")

    key_target = results_dir / ("mg_variants_eval.json" if (results_dir / "mg_variants_eval.json").exists()
                                 else "all_layer_variants_eval.json")
    if do_keys and key_target.exists():
        n_keys = rewrite_keys(key_target, apply=False)
        print(f"\nKEY REWRITE in {key_target.name}: {n_keys} condition keys would change")
        print("  " + ", ".join(f"{o} -> {n}" for o, n in KEY_RENAMES.items()))

    if not apply:
        print(f"\n{'=' * 76}\nNothing was changed. Re-run with --apply to execute.\n{'=' * 76}")
        return

    undo_lines = ["#!/usr/bin/env bash", "# Undo the naming migration. Generated automatically.",
                   "set -euo pipefail", f"cd {results_dir}"]
    for old, new in planned:
        src, dst = results_dir / old, results_dir / new
        if mode == "copy":
            shutil.copy2(src, dst)
            undo_lines.append(f"rm -f -- '{new}'")
        else:
            shutil.move(str(src), str(dst))
            undo_lines.append(f"mv -- '{new}' '{old}'")
        print(f"  {verb.lower()}: {old} -> {new}")

    if do_keys:
        target = results_dir / "mg_variants_eval.json"
        if not target.exists():
            target = results_dir / "all_layer_variants_eval.json"
        if target.exists():
            n_keys = rewrite_keys(target, apply=True)
            print(f"  rewrote {n_keys} condition keys in {target.name}")
            undo_lines.append(f"# NOTE: condition keys inside {target.name} were rewritten in place;")
            undo_lines.append(f"# this undo script does NOT revert them (rerun the eval, or edit by hand).")

    undo = results_dir / "undo_naming_migration.sh"
    undo.write_text("\n".join(undo_lines) + "\n")
    undo.chmod(0o755)
    print(f"\nwrote undo script: {undo}")
    if mode == "copy":
        print("COPY mode: old filenames are still present, so the training/search pipeline is unaffected.")
    else:
        print("MOVE mode: the pipeline WILL break until the code rename is done "
              "(variant keys, train.py output paths, tests).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="caveman", choices=TASK_CHOICES)
    ap.add_argument("--apply", action="store_true", help="actually do it (default is dry run)")
    ap.add_argument("--mode", default="copy", choices=["copy", "move"],
                     help="copy (safe, old names remain) or move (clean, needs the code rename)")
    ap.add_argument("--rewrite-keys", action="store_true",
                     help="also rename condition-key stems inside the all-layer eval JSON")
    a = ap.parse_args()
    main(a.task, a.apply, a.mode, a.rewrite_keys)
