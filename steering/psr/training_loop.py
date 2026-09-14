"""The PSR training loop -- byte-for-byte identical across every variant (proper, conceptor,
conceptor/matrix, conceptor/selfproj). What differs between them is only WHICH correction mechanism
`forward_fn` calls (steering.psr.gate.forward_with_gate_hook vs. conceptor/matrix/logic's vs.
conceptor/selfproj/logic's), and each variant's train.py builds that closure over whatever direction/
conceptor/mu_instr/delta_scale it needs. This file never imports any of those -- it only ever calls
whatever `forward_fn` it's handed.
"""
import torch

from steering.psr.data import build_training_pair
from steering.psr.gate import collect_target_hidden_states, regularization_loss, subsequent_layers_mse


@torch.no_grad()
def eval_dev_mse(model, tokenizer, forward_fn, layer_idx: int, n_layers: int, dev_items: list[dict], dev_responses: dict) -> float:
    losses = []
    for item in dev_items:
        pair = build_training_pair(model, tokenizer, item, dev_responses)
        if pair is None:
            continue
        target = collect_target_hidden_states(model, pair["full_instr"])
        pred, _ = forward_fn(pair)
        losses.append(subsequent_layers_mse(pred, target, layer_idx, pair["n_resp"], n_layers).item())
    return sum(losses) / len(losses)


def train_gate(
    model, tokenizer, forward_fn, optimizer, layer_idx: int, n_layers: int,
    train_items: list[dict], dev_items: list[dict], train_responses: dict, dev_responses: dict,
    n_epochs: int, reg_coeff: float, on_epoch_end=None,
) -> tuple[float, float]:
    """Returns (baseline_mse, final_mse). on_epoch_end(epoch, dev_mse), if given, runs after every
    epoch -- checkpoint FORMAT differs by variant (conceptor saves the conceptor matrix too, proper
    doesn't), so checkpointing stays each variant's own job, not baked into this shared loop."""
    baseline_mse = eval_dev_mse(model, tokenizer, forward_fn, layer_idx, n_layers, dev_items, dev_responses)

    for epoch in range(n_epochs):
        for item in train_items:
            pair = build_training_pair(model, tokenizer, item, train_responses)
            if pair is None:
                continue
            target = collect_target_hidden_states(model, pair["full_instr"])
            pred, fit = forward_fn(pair)
            loss = subsequent_layers_mse(pred, target, layer_idx, pair["n_resp"], n_layers) + regularization_loss(fit, reg_coeff)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        dev_mse = eval_dev_mse(model, tokenizer, forward_fn, layer_idx, n_layers, dev_items, dev_responses)
        if on_epoch_end:
            on_epoch_end(epoch, dev_mse)

    final_mse = eval_dev_mse(model, tokenizer, forward_fn, layer_idx, n_layers, dev_items, dev_responses)
    return baseline_mse, final_mse
