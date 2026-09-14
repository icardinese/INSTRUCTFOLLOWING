"""Fixed-vector conceptor PSR: direction comes from projecting a diff-in-means vector through the
closed-form conceptor matrix, not gradient-trained. Reuses steering.psr.gate.forward_with_gate_hook
directly -- mechanically identical to proper/train.py's injection, differing only in where
`direction` comes from (see steering/psr/conceptor/direction.py).
"""
import argparse
import json
import os

import torch

from adapters.registry import get_adapter
from core.reproducibility import set_seed
from core.generation_cache import load_or_compute_responses
from core.model_common import generate_response, load_model, num_layers
from steering.psr.conceptor.direction import compute_conceptor, project_direction
from steering.psr.data import load_or_pool_separate_poles
from steering.psr.gate import forward_with_gate_hook, init_gate_state
from steering.psr.training_loop import train_gate

N_EPOCHS = 3
LR = float(os.environ.get("PSR_CONCEPTOR_LR", 1e-3))
WEIGHT_DECAY = 1e-4
REG_COEFF = float(os.environ.get("PSR_CONCEPTOR_REG_COEFF", 0.1))
ALPHA = float(os.environ.get("PSR_CONCEPTOR_ALPHA", 4.0))
OUT_TAG = os.environ.get("PSR_CONCEPTOR_OUT_TAG", "")  # e.g. "_alpha8" for sweep runs


def main(task: str, layer_idx: int | None, seed: int = 42) -> None:
    set_seed(seed)
    adapter = get_adapter(task)
    out_path = adapter.RESULTS_DIR / f"psr_conceptor_probe{OUT_TAG}.pt"
    if out_path.exists() and os.environ.get("FORCE_RERUN", "0") != "1":
        print(f">>> {out_path} already exists, skipping")
        return

    device = "cuda"
    adapter.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    adapter.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model(device, model_name=adapter.MODEL_NAME)
    for p in model.parameters():
        p.requires_grad_(False)
    n_layers = num_layers(model)

    train_items = adapter.to_items(tokenizer, adapter.load_rows("train"))
    dev_items = adapter.to_items(tokenizer, adapter.load_rows("dev"))

    if layer_idx is None:
        with (adapter.RESULTS_DIR / "const_steer_config.json").open() as f:
            layer_idx = json.load(f)["layer"]

    train_responses = load_or_compute_responses(model, tokenizer, train_items, adapter.CACHE_DIR / "teacher_responses_train.json", generate_response)
    dev_responses = load_or_compute_responses(model, tokenizer, dev_items, adapter.CACHE_DIR / "teacher_responses_dev.json", generate_response)

    print(f"loading/pooling activations at layer {layer_idx} for the conceptor and diff-in-means direction")
    base_pool, instr_pool = load_or_pool_separate_poles(model, tokenizer, train_items, layer_idx, train_responses, adapter.CACHE_DIR)
    diff_mean_direction = instr_pool.mean(0) - base_pool.mean(0)
    diff_mean_direction = diff_mean_direction / diff_mean_direction.norm()

    conceptor = compute_conceptor(torch.cat([base_pool, instr_pool], dim=0), alpha=ALPHA)
    direction = project_direction(conceptor, diff_mean_direction)

    hidden_size = base_pool.shape[1]
    gate = init_gate_state(hidden_size, device)
    optimizer = torch.optim.Adam(gate.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    forward_fn = lambda pair: forward_with_gate_hook(model, gate, direction, layer_idx, pair["full_base"], pair["n_resp"])

    def save_checkpoint(epoch: int | None, dev_mse: float) -> None:
        torch.save({
            "weight": gate.weight.detach().cpu(), "bias": gate.bias.detach().cpu(),
            "coeff_bias": gate.coeff_bias.detach().cpu(),
            "conceptor": conceptor.detach().cpu(), "direction": direction.detach().cpu(),
            "layer": layer_idx, "alpha": ALPHA, "task": task,
            "completed_epochs": N_EPOCHS if epoch is None else epoch,
        }, out_path)
        with (adapter.RESULTS_DIR / f"psr_conceptor_train_log{OUT_TAG}.json").open("w") as f:
            json.dump({
                "task": task, "layer": layer_idx, "alpha": ALPHA, "seed": seed, "lr": LR,
                "weight_decay": WEIGHT_DECAY, "reg_coeff": REG_COEFF,
                "completed_epochs": N_EPOCHS if epoch is None else epoch,
                "dev_mse": dev_mse,
            }, f, indent=2)

    def on_epoch_end(epoch: int, dev_mse: float) -> None:
        print(f"epoch={epoch} dev_mse={dev_mse:.4f} -- checkpointing")
        save_checkpoint(epoch, dev_mse)

    baseline_mse, final_mse = train_gate(
        model, tokenizer, forward_fn, optimizer, layer_idx, n_layers,
        train_items, dev_items, train_responses, dev_responses, N_EPOCHS, REG_COEFF,
        on_epoch_end=on_epoch_end,
    )
    print(f"task={task} layer={layer_idx} alpha={ALPHA} baseline_dev_mse={baseline_mse:.4f} final_dev_mse={final_mse:.4f}")
    save_checkpoint(None, final_mse)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["caveman", "ifeval"])
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.task, args.layer, args.seed)
