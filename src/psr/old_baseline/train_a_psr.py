"""Original A-PSR baseline: multi-layer offline regression, probes trained JOINTLY across every
candidate layer (from core.model_common.layer_indices_from_fractions) -- same offline, no-masking,
injection-layer-only design as the S-PSR baseline (src/psr/old_baseline/train.py), just applied at
several layers at once instead of one.
"""
import argparse
import json
import os

import torch
import torch.nn.functional as F

from adapters.registry import get_adapter
from core.generation_cache import load_or_compute_responses, response_text
from core.model_common import generate_response, layer_indices_from_fractions, load_model, generate_response_with_meta
from core.reproducibility import set_seed
from steering.psr.old_baseline import MultiLayerPSRProbe
from adapters.registry import TASK_CHOICES

N_EPOCHS = 200
LR = 1e-3


@torch.no_grad()
def collect_fixed_pairs_all_layers(model, tokenizer, items: list[dict], layer_indices: list[int], responses: dict):
    """One forward pass per row per pole gives every layer's activations at once -- same
    efficiency trick as sweep_layers.py's collect_all_layers_pooled, applied to offline pairs."""
    xs = {l: [] for l in layer_indices}
    ys = {l: [] for l in layer_indices}
    for item in items:
        teacher_response = response_text(responses[str(item["id"])])
        resp_ids = tokenizer(teacher_response, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
        if resp_ids.shape[1] == 0:
            continue
        base_ids = tokenizer(item["base_prompt"], return_tensors="pt")["input_ids"].to(model.device)
        instr_ids = tokenizer(item["terse_prompt"], return_tensors="pt")["input_ids"].to(model.device)
        full_base = torch.cat([base_ids, resp_ids], dim=1)
        full_instr = torch.cat([instr_ids, resp_ids], dim=1)
        n_resp = resp_ids.shape[1]
        out_base = model(input_ids=full_base, output_hidden_states=True)
        out_instr = model(input_ids=full_instr, output_hidden_states=True)
        for l in layer_indices:
            xs[l].append(out_base.hidden_states[l + 1][0, -n_resp:, :].float())
            ys[l].append(out_instr.hidden_states[l + 1][0, -n_resp:, :].float())
    return {l: torch.cat(xs[l], dim=0) for l in layer_indices}, {l: torch.cat(ys[l], dim=0) for l in layer_indices}


def main(task: str, seed: int = 42) -> None:
    set_seed(seed)
    adapter = get_adapter(task)
    out_path = adapter.RESULTS_DIR / "a_psr_probe.pt"
    if out_path.exists() and os.environ.get("FORCE_RERUN", "0") != "1":
        print(f">>> {out_path} already exists, skipping")
        return

    device = "cuda"
    adapter.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    adapter.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model(device, model_name=adapter.MODEL_NAME)
    for p in model.parameters():
        p.requires_grad_(False)

    layer_indices = layer_indices_from_fractions(model)
    print(f"candidate layers: {layer_indices}")

    train_items = adapter.to_items(tokenizer, adapter.load_rows("train"))
    dev_items = adapter.to_items(tokenizer, adapter.load_rows("dev"))

    train_responses = load_or_compute_responses(model, tokenizer, train_items, adapter.CACHE_DIR / "teacher_responses_train.json", generate_response_with_meta)
    dev_responses = load_or_compute_responses(model, tokenizer, dev_items, adapter.CACHE_DIR / "teacher_responses_dev.json", generate_response_with_meta)

    print("collecting fixed (x, y) pairs at every candidate layer (offline, one pass per row)")
    x_train, y_train = collect_fixed_pairs_all_layers(model, tokenizer, train_items, layer_indices, train_responses)
    x_dev, y_dev = collect_fixed_pairs_all_layers(model, tokenizer, dev_items, layer_indices, dev_responses)

    directions = {}
    for l in layer_indices:
        d = y_train[l].mean(0) - x_train[l].mean(0)
        directions[l] = d / d.norm()

    hidden_size = x_train[layer_indices[0]].shape[1]
    probe = MultiLayerPSRProbe(hidden_size, layer_indices).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=LR)

    def eval_dev() -> float:
        with torch.no_grad():
            total = 0.0
            for l in layer_indices:
                lam = probe(l, x_dev[l])
                pred = x_dev[l] + lam * directions[l]
                total += F.mse_loss(pred, y_dev[l]).item()
            return total / len(layer_indices)

    baseline_mse = eval_dev()
    print(f"baseline_dev_mse={baseline_mse:.4f}")

    for epoch in range(N_EPOCHS):
        loss = torch.tensor(0.0, device=device)
        for l in layer_indices:
            lam = probe(l, x_train[l])
            pred = x_train[l] + lam * directions[l]
            loss = loss + F.mse_loss(pred, y_train[l])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f"epoch={epoch + 1} train_loss={loss.item():.4f}")

    final_mse = eval_dev()
    print(f"final_dev_mse={final_mse:.4f} (baseline was {baseline_mse:.4f})")

    torch.save({
        "layer_indices": layer_indices,
        "probes": {
            str(l): {
                "weight": probe.probes[str(l)].linear.weight.T.detach().cpu(),
                "bias": probe.probes[str(l)].linear.bias.detach().cpu(),
            } for l in layer_indices
        },
        "directions": {l: directions[l].detach().cpu() for l in layer_indices},
        "task": task,
    }, out_path)
    with (adapter.RESULTS_DIR / "a_psr_train_log.json").open("w") as f:
        json.dump({
            "task": task, "layer_indices": layer_indices, "seed": seed, "lr": LR,
            "baseline_dev_mse": baseline_mse, "final_dev_mse": final_mse,
        }, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=TASK_CHOICES)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.task, args.seed)
