"""A-PSR: Heyman & Vandeputte's all-layer Prompt Steering Replacement, plus its Multi-Gate
ablation, on ONE shared code path.

Both variants are this same script, selected by --direction-source:

  trained        A-PSR, faithful. N gates + N gradient-trained directions at every layer, jointly
                 optimized in one shared forward pass, one backward, one optimizer.
  diff_in_means  "Multi-Gate" ablation. Identical in every respect EXCEPT the directions are fixed
                 diff-in-means vectors (requires_grad=False, absent from the optimizer). Isolates
                 exactly one variable: does A-PSR's advantage come from gradient-training the
                 directions, or from having a trained per-token gate at every layer? This is the
                 multi-layer analogue of the single-layer proper-vs-fixed-direction comparison.

Deliberately ONE script with a flag rather than two near-duplicate files: the two variants must
stay byte-identical in their training loop, layer set, epochs, LR, regularization, masking and
loss, or the ablation proves nothing. A flag makes that structural; two files make it a promise
someone has to keep by hand every time the code changes (and this codebase is about to be run
across multiple models, where that drift would be invisible).

ALL LAYERS, faithfully. Nokia's A-PSR config is `layers=list(range(num_layers))` -- literally every
layer (28 for Qwen2.5-Coder-7B), not a subset and not a fractional grid. That is the default here.
--layers exists only for cheaper debugging runs; using it makes the result NOT A-PSR, and the
chosen layer set is recorded in both the sweep row and the checkpoint so that can never be
silently misreported later.

NO LAYER SWEEP EXISTS FOR THIS METHOD. A-PSR has no layer hyperparameter -- "which layer" is
answered by "all of them" definitionally. So sweep() here is a 2-point grid over loss configs only
(pure MSE = H&V's _MSE, pure NLL = their _LL), not the (layer x hyperparameter) grid the
single-layer variants need. This makes A-PSR dramatically CHEAPER than proper's 65-point sweep,
not more expensive.

MSE IS OVER ALL LAYERS. subsequent_layers_mse is called with layer_idx=0, which covers
hidden_states indices 1..n_layers -- i.e. every layer's output. That matches Nokia's
LossSpecification(layers_to_imitate="all") and is the right choice regardless of --layers, since
the earliest intervention is at or after layer 0 in every case.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from adapters.registry import get_adapter
from core.generation_cache import load_or_compute_responses
from core.model_common import generate_response, load_model, num_layers, generate_response_with_meta
from core.reproducibility import set_seed
from core.sweep import run_grid_sweep
from steering.psr.conceptor.direction import compute_conceptor_from_correlation, project_direction
from steering.psr.data import accumulate_bipolar_correlation_all_layers, load_or_pool_prompt_last_token_all_layers
from steering.psr.gate import init_gate_state
from steering.psr.multi_gate import forward_with_multi_gate_hook
from steering.psr.proper.direction import init_direction
from steering.psr.reference_config import (
    DEFAULT_LOSS_BALANCE,
    N_EPOCHS_MSE,
    WEIGHT_DECAY as REF_WEIGHT_DECAY,
    epochs_for,
    loss_balance,
)
from steering.psr.training_loop import train_gate
from adapters.registry import TASK_CHOICES

N_EPOCHS = N_EPOCHS_MSE  # reference: 15 for MSE, 7 for LL -- see epochs_for()
LR = float(os.environ.get("PSR_ALL_LAYER_LR", 1e-3))
WEIGHT_DECAY = REF_WEIGHT_DECAY  # 1e-6; the reference's dataclass default of 1e-4 is never used
LOSS_BALANCE = os.environ.get("PSR_ALL_LAYER_LOSS_BALANCE", DEFAULT_LOSS_BALANCE)
REG_COEFF = float(os.environ.get("PSR_ALL_LAYER_REG_COEFF", loss_balance(LOSS_BALANCE)["reg_coeff"]))
NORMALIZE_PSI = loss_balance(LOSS_BALANCE)["normalize_psi"]
MSE_WEIGHT = float(os.environ.get("PSR_ALL_LAYER_MSE_WEIGHT", 1.0))
NLL_WEIGHT = float(os.environ.get("PSR_ALL_LAYER_NLL_WEIGHT", 0.0))

DIRECTION_SOURCES = ("trained", "diff_in_means", "conceptor")
ALPHA = float(os.environ.get("PSR_ALL_LAYER_ALPHA", 4.0))

# Same two endpoints as every other variant post-2026-09-17 (see src/psr/proper/train.py for the
# full reasoning): H&V train MSE and LL as mutually-exclusive alternatives, never blended.
DEFAULT_LOSS_CONFIG_GRID = [
    {"mse_weight": 1.0, "nll_weight": 0.0},   # pure MSE (H&V's "_MSE" variant)
    {"mse_weight": 0.0, "nll_weight": 1.0},   # pure NLL (H&V's "_LL" variant)
]


def build_directions(
    direction_source: str, layer_indices: list[int], hidden_size: int, device: str,
    model=None, tokenizer=None, train_items=None, n_layers=None, cache_dir=None,
    train_responses=None, alpha: float = ALPHA,
) -> tuple[dict[int, torch.Tensor], list[torch.Tensor]]:
    """Returns (directions, trainable_params). trainable_params is [] for every non-'trained'
    source -- that emptiness IS the ablation, and it's what keeps those directions out of the
    optimizer without the caller needing a second flag.

    The two closed-form sources both use the PROMPT's last-token diff-in-means as their starting
    vector (matching Const, Stolfo and the fixed-vector conceptor in this codebase as of
    2026-09-17); 'conceptor' then additionally reshapes it through that layer's own conceptor
    matrix. So the three sources isolate a clean progression at fixed gate machinery:
    gradient-trained direction -> raw DiM -> conceptor-projected DiM.
    """
    if direction_source == "trained":
        directions = {l: init_direction(hidden_size, device) for l in layer_indices}
        return directions, list(directions.values())

    if direction_source in ("diff_in_means", "conceptor"):
        base_by_layer, instr_by_layer = load_or_pool_prompt_last_token_all_layers(
            model, tokenizer, train_items, n_layers, cache_dir
        )
        raw = {}
        for l in layer_indices:
            diff = instr_by_layer[l].to(device).mean(0) - base_by_layer[l].to(device).mean(0)
            # Unit-normalized so the gate's learned coefficient is the ONLY thing setting
            # magnitude -- otherwise each layer's raw diff norm silently rescales its own gate,
            # and per-layer magnitudes differ a lot across depth.
            raw[l] = diff / diff.norm()

        if direction_source == "diff_in_means":
            return {l: raw[l].detach().requires_grad_(False) for l in layer_indices}, []

        # conceptor: one C per layer, built from that layer's own bipolar response-token
        # correlation matrix. Accumulated in a single pass (see the data.py function's docstring
        # for why R rather than raw pooled activations at this scale), then each C is used once to
        # reshape that layer's DiM vector and immediately discarded -- only the resulting (d,)
        # direction is retained, so peak memory is one C at a time, not n_layers of them.
        corr = accumulate_bipolar_correlation_all_layers(
            model, tokenizer, train_items, n_layers, train_responses, device
        )
        directions = {}
        for l in layer_indices:
            c = compute_conceptor_from_correlation(corr[l], alpha)
            directions[l] = project_direction(c, raw[l]).detach().requires_grad_(False)
            del c
        del corr
        torch.cuda.empty_cache()
        return directions, []

    raise ValueError(f"unknown direction_source {direction_source!r}, expected one of {DIRECTION_SOURCES}")


def train_one_config(
    model, tokenizer, direction_source: str, layer_indices: list[int], seed: int,
    n_layers: int, hidden_size: int, device: str,
    train_items: list[dict], dev_items: list[dict], train_responses: dict, dev_responses: dict,
    cache_dir, lr: float = LR, weight_decay: float = WEIGHT_DECAY, reg_coeff: float = REG_COEFF,
    n_epochs: int | None = None, mse_weight: float = MSE_WEIGHT, nll_weight: float = NLL_WEIGHT,
    normalize_psi: bool = NORMALIZE_PSI,
    alpha: float = ALPHA, on_epoch_end=None,
) -> dict:
    """Trains N gates (+ N directions, if direction_source == "trained") jointly. Mirrors
    src/psr/proper/train.py's train_one_config contract: scalar dev metrics AND the trained tensors
    in one dict, callers pick out what they need."""
    # Reference epoch budget is objective-dependent (15 MSE / 7 LL). Training both endpoints
    # for the same number of epochs confounds 'which objective wins' with 'which converged'.
    if n_epochs is None:
        n_epochs = epochs_for(mse_weight, nll_weight)
    set_seed(seed)
    gates = {l: init_gate_state(hidden_size, device) for l in layer_indices}
    directions, direction_params = build_directions(
        direction_source, layer_indices, hidden_size, device,
        model=model, tokenizer=tokenizer, train_items=train_items,
        n_layers=n_layers, cache_dir=cache_dir, train_responses=train_responses, alpha=alpha,
    )

    # ONE optimizer over every layer's parameters -> one joint update step. This, plus one
    # backward in train_gate, is what makes the interventions genuinely simultaneous rather than
    # N independent single-layer trainings (the flaw in the old mislabeled "A-PSR").
    gate_params = [p for l in layer_indices for p in gates[l].parameters()]
    optimizer = torch.optim.AdamW(gate_params + direction_params, lr=lr, weight_decay=weight_decay)

    forward_fn = lambda pair: forward_with_multi_gate_hook(
        model, gates, directions, layer_indices, pair["full_base"], pair["n_resp"]
    )

    wrapped_on_epoch_end = (lambda epoch, metrics: on_epoch_end(epoch, gates, directions, metrics)) if on_epoch_end else None
    # layer_idx=0 -> MSE over ALL layers (see module docstring).
    baseline_metrics, final_metrics = train_gate(
        model, tokenizer, forward_fn, optimizer, 0, n_layers,
        train_items, dev_items, train_responses, dev_responses, n_epochs, reg_coeff,
        mse_weight=mse_weight, nll_weight=nll_weight, on_epoch_end=wrapped_on_epoch_end,
        normalize_psi=normalize_psi,
    )
    return {
        "completed_epochs": n_epochs,
        "baseline_mse": baseline_metrics["mse"], "baseline_nll": baseline_metrics["nll"],
        "final_mse": final_metrics["mse"], "final_nll": final_metrics["nll"],
        "gates": {l: {"weight": gates[l].weight.detach().cpu(), "bias": gates[l].bias.detach().cpu(),
                       "coeff_bias": gates[l].coeff_bias.detach().cpu()} for l in layer_indices},
        "directions": {l: directions[l].detach().cpu() for l in layer_indices},
    }


def _save_checkpoint(out_path, result: dict, direction_source: str, layer_indices: list[int],
                      task: str, mse_weight: float, nll_weight: float, alpha: float | None = None) -> None:
    torch.save({
        "gates": result["gates"], "directions": result["directions"],
        "layer_indices": layer_indices, "direction_source": direction_source,
        "task": task, "mse_weight": mse_weight, "nll_weight": nll_weight,
        # alpha only means anything for direction_source="conceptor"; stored as None otherwise so
        # the key is always present and a reader never has to guess whether it was applicable.
        "alpha": alpha if direction_source == "conceptor" else None,
        "completed_epochs": result["completed_epochs"],
    }, out_path)


def _setup(task: str, device: str):
    adapter = get_adapter(task)
    adapter.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    adapter.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model(device, model_name=adapter.MODEL_NAME)
    for p in model.parameters():
        p.requires_grad_(False)
    n_layers = num_layers(model)
    train_items = adapter.to_items(tokenizer, adapter.load_rows("train"))
    dev_items = adapter.to_items(tokenizer, adapter.load_rows("dev"))
    train_responses = load_or_compute_responses(model, tokenizer, train_items, adapter.CACHE_DIR / "teacher_responses_train.json", generate_response_with_meta)
    dev_responses = load_or_compute_responses(model, tokenizer, dev_items, adapter.CACHE_DIR / "teacher_responses_dev.json", generate_response_with_meta)
    return adapter, model, tokenizer, n_layers, train_items, dev_items, train_responses, dev_responses


def _variant_tag(direction_source: str) -> str:
    """File-naming stem. Kept as a function so the two variants' artifact names are derived in
    exactly one place rather than formatted ad hoc at each call site."""
    return {
        "trained": "psr_all_layer",
        "diff_in_means": "psr_multi_gate",
        "conceptor": "psr_multi_gate_conceptor",
    }[direction_source]


def main(task: str, direction_source: str, layers: list[int] | None, seed: int = 42, device: str = "cuda") -> None:
    out_tag = os.environ.get("PSR_ALL_LAYER_OUT_TAG", "")
    adapter, model, tokenizer, n_layers, train_items, dev_items, train_responses, dev_responses = _setup(task, device)
    out_path = adapter.RESULTS_DIR / f"{_variant_tag(direction_source)}_probe{out_tag}.pt"
    if out_path.exists() and os.environ.get("FORCE_RERUN", "0") != "1":
        print(f">>> {out_path} already exists, skipping")
        return

    layer_indices = layers if layers is not None else list(range(n_layers))
    alpha_note = f", alpha={ALPHA}" if direction_source == "conceptor" else ""
    print(f"{_variant_tag(direction_source)}: {len(layer_indices)} layers, direction_source={direction_source}, "
          f"mse_weight={MSE_WEIGHT}, nll_weight={NLL_WEIGHT}{alpha_note}")
    result = train_one_config(
        model, tokenizer, direction_source, layer_indices, seed, n_layers, model.config.hidden_size,
        device, train_items, dev_items, train_responses, dev_responses, adapter.CACHE_DIR,
        mse_weight=MSE_WEIGHT, nll_weight=NLL_WEIGHT,
    )
    _save_checkpoint(out_path, result, direction_source, layer_indices, task, MSE_WEIGHT, NLL_WEIGHT, ALPHA)
    print(f"task={task} layers={len(layer_indices)} baseline_dev_mse={result['baseline_mse']:.4f} "
          f"final_dev_mse={result['final_mse']:.4f}")
    print(f"wrote {out_path}")


def sweep(task: str, direction_source: str, layers: list[int] | None = None,
          loss_config_grid: list[dict] | None = None, seed: int = 42, device: str = "cuda") -> None:
    """2-point grid over loss configs only -- A-PSR has no layer hyperparameter (see module
    docstring). Resumable via core.sweep.run_grid_sweep, same as every other variant.

    Saves a SEPARATE checkpoint per loss config (..._probe_mse.pt, ..._probe_nll.pt) rather than
    one "winner". run_grid_sweep selects best by lowest final_mse, which would pick pure MSE every
    single time by construction -- a pure-NLL run optimizes a different objective and lands at a
    much higher MSE (real measurement on the single-layer variant: 25.5 vs 5.7), so "lowest
    final_mse" is not a meaningful comparison ACROSS objectives, only within one. Both objectives
    are real experiments H&V report separately (_MSE and _LL), and which one actually produces
    better generations is a question for judged evaluation, not training loss. So: keep both.
    """
    loss_config_grid = loss_config_grid if loss_config_grid is not None else DEFAULT_LOSS_CONFIG_GRID
    adapter, model, tokenizer, n_layers, train_items, dev_items, train_responses, dev_responses = _setup(task, device)
    layer_indices = layers if layers is not None else list(range(n_layers))
    tag = _variant_tag(direction_source)
    print(f"{tag} sweep: {len(loss_config_grid)} loss configs x {len(layer_indices)} layers "
          f"(all layers, no layer sweep -- A-PSR has no layer hyperparameter)")

    def _config_suffix(mse_weight: float, nll_weight: float) -> str:
        if mse_weight > 0 and nll_weight == 0:
            return "_mse"
        if mse_weight == 0 and nll_weight > 0:
            return "_nll"
        return f"_mse{mse_weight}_nll{nll_weight}"

    def train_fn(point: dict) -> dict:
        result = train_one_config(
            model, tokenizer, direction_source, layer_indices, seed, n_layers,
            model.config.hidden_size, device, train_items, dev_items, train_responses,
            dev_responses, adapter.CACHE_DIR,
            mse_weight=point["mse_weight"], nll_weight=point["nll_weight"],
        )
        # Checkpoint immediately, inside train_fn, so a resumed sweep never has to retrain a
        # completed point just to recover its tensors (the failure mode src/psr/proper/train.py's
        # sweep() has to work around with a retrain-the-winner branch).
        probe_path = adapter.RESULTS_DIR / f"{tag}_probe{_config_suffix(point['mse_weight'], point['nll_weight'])}.pt"
        _save_checkpoint(probe_path, result, direction_source, layer_indices, task,
                          point["mse_weight"], point["nll_weight"], ALPHA)
        print(f"{tag} mse_weight={point['mse_weight']} nll_weight={point['nll_weight']} "
              f"final_mse={result['final_mse']:.4f} final_nll={result['final_nll']:.4f} -> {probe_path.name}")
        return {k: v for k, v in result.items() if k not in ("gates", "directions")}

    grid = [{"n_layers_hooked": len(layer_indices), **cfg} for cfg in loss_config_grid]
    out_path = adapter.RESULTS_DIR / f"{tag}_sweep.jsonl"
    results, _best = run_grid_sweep(grid, train_fn, out_path, key_fields=["mse_weight", "nll_weight"])
    print(f"\nwrote {len(results)} sweep rows to {out_path}")
    print(f"checkpoints saved per loss config -- compare them by JUDGED evaluation "
          f"(scripts/eval_all_layer_variants.py), not by final_mse across different objectives")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=TASK_CHOICES)
    parser.add_argument("--direction-source", required=True, choices=DIRECTION_SOURCES,
                         help="'trained' = A-PSR (faithful); 'diff_in_means' = Multi-Gate ablation")
    parser.add_argument("--layers", type=str, default=None,
                         help="comma-separated layer indices. DEFAULT (omit) = ALL layers, which is "
                              "what A-PSR actually is; setting this makes the run NOT faithful A-PSR")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sweep", action="store_true", help="sweep the 2 loss configs instead of one run")
    parser.add_argument("--loss-configs", type=str, default=None,
                         help='JSON list, e.g. \'[{"mse_weight":1.0,"nll_weight":0.0}]\'')
    args = parser.parse_args()

    layers = [int(x) for x in args.layers.split(",")] if args.layers else None
    loss_configs = json.loads(args.loss_configs) if args.loss_configs else None
    if args.sweep:
        sweep(args.task, args.direction_source, layers, loss_configs, args.seed)
    else:
        main(args.task, args.direction_source, layers, args.seed)
