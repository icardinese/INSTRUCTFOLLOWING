"""Original S-PSR baseline: OFFLINE regression against precomputed, FIXED activation pairs -- no
live gradient-tracked forward pass, no answer_only masking, no subsequent-layers loss. This is the
pre-fidelity-fix design (matches the actual old steering_psr.py exactly), kept deliberately as a
baseline comparison point against psr/proper and psr/conceptor, not something to "fix" here.

Checkpoint note: PSRProbe's formula (correction = relu(linear(h)) * direction) is mathematically
the SAME as steering/psr/gate.py's GateState formula with coeff_bias fixed at 0 -- so this saves
in GateState-compatible shape, meaning src/generate.py's existing load_gate_condition/
make_inference_hook already supports this checkpoint with zero changes needed there.
"""
import argparse
import json
import os

import torch
import torch.nn.functional as F

from adapters.registry import get_adapter
from core.reproducibility import set_seed
from core.generation_cache import load_or_compute_responses, response_text
from core.model_common import generate_response, load_model, generate_response_with_meta
from steering.psr.old_baseline import PSRProbe
from adapters.registry import TASK_CHOICES

N_EPOCHS = 200
LR = 1e-3


@torch.no_grad()
def collect_fixed_pairs(model, tokenizer, items: list[dict], layer_idx: int, responses: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """x = base-prompt response-token activations, y = instructed-prompt response-token
    activations, AT THE INJECTION LAYER ONLY -- exactly the fidelity gap the later variants fixed;
    kept as-is here since replicating the original baseline faithfully is the whole point."""
    xs, ys = [], []
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
        xs.append(out_base.hidden_states[layer_idx + 1][0, -n_resp:, :].float())
        ys.append(out_instr.hidden_states[layer_idx + 1][0, -n_resp:, :].float())
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def main(task: str, layer_idx: int | None, seed: int = 42) -> None:
    set_seed(seed)
    adapter = get_adapter(task)
    out_path = adapter.RESULTS_DIR / "psr_probe.pt"
    if out_path.exists() and os.environ.get("FORCE_RERUN", "0") != "1":
        print(f">>> {out_path} already exists, skipping")
        return

    device = "cuda"
    adapter.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    adapter.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model(device, model_name=adapter.MODEL_NAME)
    for p in model.parameters():
        p.requires_grad_(False)

    train_items = adapter.to_items(tokenizer, adapter.load_rows("train"))
    dev_items = adapter.to_items(tokenizer, adapter.load_rows("dev"))

    if layer_idx is None:
        with (adapter.RESULTS_DIR / "const_steer_config.json").open() as f:
            layer_idx = json.load(f)["layer"]

    train_responses = load_or_compute_responses(model, tokenizer, train_items, adapter.CACHE_DIR / "teacher_responses_train.json", generate_response_with_meta)
    dev_responses = load_or_compute_responses(model, tokenizer, dev_items, adapter.CACHE_DIR / "teacher_responses_dev.json", generate_response_with_meta)

    print(f"collecting fixed (x, y) pairs at layer {layer_idx} (offline, injection-layer only)")
    x_train, y_train = collect_fixed_pairs(model, tokenizer, train_items, layer_idx, train_responses)
    x_dev, y_dev = collect_fixed_pairs(model, tokenizer, dev_items, layer_idx, dev_responses)
    print(f"n_train_pairs={x_train.shape[0]}, n_dev_pairs={x_dev.shape[0]}")

    diff_mean_direction = y_train.mean(0) - x_train.mean(0)
    diff_mean_direction = diff_mean_direction / diff_mean_direction.norm()

    hidden_size = x_train.shape[1]
    probe = PSRProbe(hidden_size).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=LR)

    def eval_dev() -> float:
        with torch.no_grad():
            lam = probe(x_dev)
            pred = x_dev + lam * diff_mean_direction
            return F.mse_loss(pred, y_dev).item()

    baseline_mse = eval_dev()
    print(f"baseline_dev_mse={baseline_mse:.4f}")

    for epoch in range(N_EPOCHS):
        lam = probe(x_train)
        pred = x_train + lam * diff_mean_direction
        loss = F.mse_loss(pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f"epoch={epoch + 1} train_loss={loss.item():.4f}")

    final_mse = eval_dev()
    print(f"final_dev_mse={final_mse:.4f} (baseline was {baseline_mse:.4f})")

    torch.save({
        "weight": probe.linear.weight.T.detach().cpu(),  # (1,d) -> (d,1), GateState's expected shape
        "bias": probe.linear.bias.detach().cpu(),
        "coeff_bias": torch.zeros(1),  # makes this a GateState-compatible checkpoint, see module docstring
        "direction": diff_mean_direction.detach().cpu(),
        "layer": layer_idx, "task": task,
    }, out_path)
    with (adapter.RESULTS_DIR / "psr_train_log.json").open("w") as f:
        json.dump({
            "task": task, "layer": layer_idx, "seed": seed, "lr": LR,
            "baseline_dev_mse": baseline_mse,
            "final_dev_mse": final_mse, "n_train_pairs": x_train.shape[0],
        }, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=TASK_CHOICES)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.task, args.layer, args.seed)
