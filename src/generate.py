"""Generates every available condition's responses for a task+split. Every condition, however
different its hook mechanism (single-layer coefficient, single-layer gate, multi-layer gate),
reduces to a context_fn(model) -> context manager -- the main loop below never needs to know which
kind it's dealing with. Gracefully skips a condition if its checkpoint doesn't exist yet, so this
file never needs editing as new methods get trained, only checkpoints need to show up.

Resumable at the row level, same discipline as every other long-running script in this project.
"""
import argparse
import json

import torch

from adapters.registry import get_adapter
from core.model_common import generate_response, load_model, token_count
from steering.const.hooks import make_const_hook
from steering.hooks import multi_steering_hook, steering_hook
from steering.psr.conceptor.matrix.logic import make_inference_hook as matrix_inference_hook
from steering.psr.conceptor.selfproj.logic import make_inference_hook as selfproj_inference_hook
from steering.psr.gate import GateState, make_inference_hook
from steering.psr.old_baseline import MultiLayerPSRProbe, make_multi_psr_hooks


def load_const_condition(adapter, device):
    config_path = adapter.RESULTS_DIR / "const_steer_config.json"
    directions_path = adapter.RESULTS_DIR / "const_steer_directions.pt"
    if not config_path.exists() or not directions_path.exists():
        return None
    with config_path.open() as f:
        config = json.load(f)
    directions = torch.load(directions_path, map_location=device)
    direction, layer_idx, coeff = directions[config["layer"]].to(device), config["layer"], config["coeff"]

    def context_fn(model):
        return steering_hook(model, layer_idx, make_const_hook(direction, coeff))
    return context_fn


def load_gate_condition(adapter, filename: str, device):
    """proper, conceptor(fixed-vector), and the old S-PSR baseline are all inference-identical: a
    gate plus a fixed direction vector via the same make_inference_hook -- only their TRAINING
    differed. This one loader covers all three."""
    path = adapter.RESULTS_DIR / filename
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location=device)
    gate = GateState(weight=ckpt["weight"].to(device), bias=ckpt["bias"].to(device), coeff_bias=ckpt["coeff_bias"].to(device))
    direction, layer_idx = ckpt["direction"].to(device), ckpt["layer"]

    def context_fn(model):
        return steering_hook(model, layer_idx, make_inference_hook(gate, direction))
    return context_fn


def load_matrix_condition(adapter, device):
    path = adapter.RESULTS_DIR / "psr_conceptor_matrix_probe.pt"
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location=device)
    gate = GateState(weight=ckpt["weight"].to(device), bias=ckpt["bias"].to(device), coeff_bias=ckpt["coeff_bias"].to(device))
    conceptor, mu_instr, delta_scale = ckpt["conceptor"].to(device), ckpt["mu_instr"].to(device), ckpt["delta_scale"].to(device)
    layer_idx = ckpt["layer"]

    def context_fn(model):
        return steering_hook(model, layer_idx, matrix_inference_hook(gate, conceptor, mu_instr, delta_scale))
    return context_fn


def load_selfproj_condition(adapter, device):
    path = adapter.RESULTS_DIR / "psr_conceptor_selfproj_probe.pt"
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location=device)
    gate = GateState(weight=ckpt["weight"].to(device), bias=ckpt["bias"].to(device), coeff_bias=ckpt["coeff_bias"].to(device))
    conceptor, delta_scale = ckpt["conceptor"].to(device), ckpt["delta_scale"].to(device)
    layer_idx = ckpt["layer"]

    def context_fn(model):
        return steering_hook(model, layer_idx, selfproj_inference_hook(gate, conceptor, delta_scale))
    return context_fn


def load_a_psr_condition(adapter, device):
    path = adapter.RESULTS_DIR / "a_psr_probe.pt"
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location=device)
    layer_indices = ckpt["layer_indices"]
    hidden_size = ckpt["probes"][str(layer_indices[0])]["weight"].shape[0]
    probe = MultiLayerPSRProbe(hidden_size, layer_indices).to(device)
    for l in layer_indices:
        probe.probes[str(l)].linear.weight.data = ckpt["probes"][str(l)]["weight"].T.to(device)
        probe.probes[str(l)].linear.bias.data = ckpt["probes"][str(l)]["bias"].to(device)
    directions = {l: ckpt["directions"][l].to(device) for l in layer_indices}

    def context_fn(model):
        hooks = make_multi_psr_hooks(directions, probe)
        return multi_steering_hook(model, hooks)
    return context_fn


CONDITION_LOADERS = [
    ("const", lambda adapter, device: load_const_condition(adapter, device)),
    ("psr_proper", lambda adapter, device: load_gate_condition(adapter, "psr_proper_probe.pt", device)),
    ("psr_conceptor", lambda adapter, device: load_gate_condition(adapter, "psr_conceptor_probe.pt", device)),
    ("psr", lambda adapter, device: load_gate_condition(adapter, "psr_probe.pt", device)),
    ("psr_conceptor_matrix", lambda adapter, device: load_matrix_condition(adapter, device)),
    ("psr_conceptor_selfproj", lambda adapter, device: load_selfproj_condition(adapter, device)),
    ("a_psr", lambda adapter, device: load_a_psr_condition(adapter, device)),
]


def already_done_ids(out_path) -> set:
    if not out_path.exists():
        return set()
    done = set()
    with out_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    done.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def main(task: str, split: str) -> None:
    adapter = get_adapter(task)
    device = "cuda"
    model, tokenizer = load_model(device, model_name=adapter.MODEL_NAME)

    conditions = {}
    for name, loader in CONDITION_LOADERS:
        context_fn = loader(adapter, device)
        if context_fn:
            conditions[name] = context_fn
    print(f"available steered conditions: {list(conditions.keys())} "
          f"(missing checkpoints are skipped, not errors -- rerun once they exist)")

    rows = adapter.load_rows(split)
    items = adapter.to_items(tokenizer, rows)

    out_path = adapter.RESULTS_DIR / f"generations_{split}.jsonl"
    done = already_done_ids(out_path)
    if done:
        print(f">>> resuming: {len(done)}/{len(items)} rows already done in {out_path}")

    n_processed = 0
    for item in items:
        if item["id"] in done:
            continue

        out_row = {"id": item["id"]}

        base_resp = generate_response(model, tokenizer, item["base_prompt"])
        prompt_resp = generate_response(model, tokenizer, item["terse_prompt"])
        out_row["base_response"] = base_resp
        out_row["base_tokens"] = token_count(tokenizer, base_resp)
        out_row["prompt_response"] = prompt_resp
        out_row["prompt_tokens"] = token_count(tokenizer, prompt_resp)

        for name, context_fn in conditions.items():
            with context_fn(model):
                alone_resp = generate_response(model, tokenizer, item["base_prompt"])
            with context_fn(model):
                combined_resp = generate_response(model, tokenizer, item["terse_prompt"])

            out_row[f"{name}_response"] = alone_resp
            out_row[f"{name}_tokens"] = token_count(tokenizer, alone_resp)
            out_row[f"prompt_{name}_response"] = combined_resp
            out_row[f"prompt_{name}_tokens"] = token_count(tokenizer, combined_resp)

        with out_path.open("a") as f:
            f.write(json.dumps(out_row) + "\n")
        n_processed += 1
        if n_processed % 10 == 0:
            print(f"{len(done) + n_processed}/{len(items)} done this session")

    print(f"done. {out_path} now has {len(done) + n_processed}/{len(items)} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["caveman", "ifeval"])
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    main(args.task, args.split)
