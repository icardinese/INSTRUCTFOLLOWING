"""Disk store for trained gates: every (variant, layer, alpha, loss-config) is trained AT MOST ONCE.

WHY THIS EXISTS. Sweeps kept every point's weights in memory (checkpoints_by_point) but only ever
wrote ONE to disk -- the best-by-final_mse probe. Everything downstream that needed another
point's gate had no choice but to retrain it from scratch: the judged layer search retrained each
Tier 1 candidate, then retrained the same configs again for Tier 2, again for Final, and
scripts/regen_token_cis.py retrained the winner once more. A winning config was trained five
times. Each gate is ~30KB; storage was never the constraint.

HOW IT'S USED
  - Sweeps call save() for every point they train.
  - evals/layer_hparam_search.py wraps every RETRAIN_FNS entry with stored(): load if present,
    otherwise train once, save, and return. Tier 2, Final, regen_token_cis, and any rerun after a
    crash then load instead of retraining.

STALENESS GUARD. A stored gate is only valid for the exact training setup that produced it. The
path therefore includes a fingerprint of everything that changes the trained weights: the
reference hyperparameters (lr, weight decay, epoch budgets, coeff-bias flag, loss-balance
package, shuffle seed), the seed, any PSR_* environment overrides, and the training data itself
(item ids plus the exact teacher responses and their `finished` flags). Change any of them and
the fingerprint changes, so an old gate can never be silently reused under a new setup. Bump
STORE_VERSION whenever the training MATH changes in a way none of those inputs capture.

WRITES ARE ATOMIC. Files land on a Drive FUSE mount, where a killed process mid-write would
otherwise leave a truncated file. Each save writes to a temp name and then os.replace()s it, and
an unreadable file is treated as a miss (retrain + overwrite) rather than trusted.
"""
import hashlib
import json
import os
from pathlib import Path

import torch

STORE_VERSION = 1
STORE_DIRNAME = "gate_store"

# Only what each variant's inference hook actually needs, plus a little provenance. Keeps the
# store small and means a stored file never drags along a 3584x3584 conceptor matrix it doesn't use.
KEEP_FIELDS = {
    "proper": ("weight", "bias", "coeff_bias", "direction"),
    "sg": ("weight", "bias", "coeff_bias", "direction"),
    "single_gate": ("weight", "bias", "coeff_bias", "direction"),
    "conceptor": ("weight", "bias", "coeff_bias", "direction"),
    "sg_clamp": ("gates", "directions", "targets"),
    "conceptor_matrix": ("weight", "bias", "coeff_bias", "conceptor", "mu_instr", "delta_scale"),
    "conceptor_selfproj": ("skipped", "weight", "bias", "coeff_bias", "conceptor", "delta_scale_tensor"),
}
PROVENANCE_FIELDS = ("final_mse", "final_nll", "completed_epochs")


def _to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(v) for v in obj)
    return obj


def fingerprint(seed: int, train_items: list, train_responses: dict, dev_items: list) -> str:
    """Hash of everything that determines the trained weights. See module docstring."""
    from core.generation_cache import normalize_cache_entry
    from steering.psr import reference_config as rc

    cfg = {
        "store_version": STORE_VERSION,
        "lr": rc.LR,
        "weight_decay": rc.WEIGHT_DECAY,
        "epochs_mse": rc.N_EPOCHS_MSE,
        "epochs_ll": rc.N_EPOCHS_LL,
        "use_coeff_bias": rc.USE_COEFF_BIAS,
        "default_loss_balance": rc.DEFAULT_LOSS_BALANCE,
        "data_shuffle_seed": rc.DATA_SHUFFLE_SEED,
        "seed": seed,
        "env": {k: v for k, v in sorted(os.environ.items()) if k.startswith("PSR_")},
        "train": [
            [str(it["id"]), normalize_cache_entry(train_responses[str(it["id"])])]
            for it in train_items if str(it["id"]) in train_responses
        ],
        # dev only matters if normalize_psi is on (it sets the MSE normalizer), but it is cheap
        # to include and removes the need to reason about which package is active.
        "dev": [str(it["id"]) for it in dev_items],
    }
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def point_path(results_dir: Path, variant: str, row: dict, fp: str) -> Path:
    # int()/float() canonicalize, so a JSON-loaded 1.0 and an in-memory 1 name the same file.
    parts = [f"L{int(row['layer'])}"]
    if row.get("alpha") is not None:
        parts.append(f"a{float(row['alpha'])}")
    parts += [f"m{float(row['mse_weight'])}", f"n{float(row['nll_weight'])}"]
    return Path(results_dir) / STORE_DIRNAME / variant / fp / ("_".join(parts) + ".pt")


def save(path: Path, variant: str, result: dict) -> None:
    keep = KEEP_FIELDS.get(variant)
    if keep is None:
        raise KeyError(f"gate_store has no KEEP_FIELDS entry for variant {variant!r}")
    payload = {k: result[k] for k in keep if k in result}
    payload.update({k: result[k] for k in PROVENANCE_FIELDS if k in result})
    payload["_variant"] = variant
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(_to_cpu(payload), tmp)
    os.replace(tmp, path)


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None  # truncated/corrupt -- caller retrains and overwrites
