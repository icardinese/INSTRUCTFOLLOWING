"""The PSR training loop -- byte-for-byte identical across every variant (proper, conceptor,
conceptor/matrix, conceptor/selfproj). What differs between them is only WHICH correction mechanism
`forward_fn` calls (steering.psr.gate.forward_with_gate_hook vs. conceptor/matrix/logic's vs.
conceptor/selfproj/logic's), and each variant's train.py builds that closure over whatever direction/
conceptor/mu_instr/delta_scale it needs. This file never imports any of those -- it only ever calls
whatever `forward_fn` it's handed (which must now return (hidden_states, logits, fit_vals) -- see
steering/psr/gate.py's forward_with_gate_hook docstring).

Loss = subsequent-layers MSE + dead-gate regularization + nll_weight * auxiliary NLL (Eq. 4, see
steering/psr/nll.py). nll_weight defaults to 0.0 so every existing call site and every existing
result reproduces unchanged unless a variant's train.py explicitly opts in via its own env-var
hyperparameter -- matching this project's established "env-var-overridable constant, not a config
layer" convention (see ARCHITECTURE.md). Dev NLL is always computed and reported alongside dev MSE
(even at nll_weight=0.0) purely as a diagnostic -- cheap (one extra cross_entropy per item) and
directly useful for the "did adding this loss actually help" question the whole feature exists to
answer.
"""
import torch

from steering.psr.data import build_training_pair
from steering.psr.gate import collect_target_hidden_states, regularization_loss, subsequent_layers_mse
from steering.psr.nll import response_nll


@torch.no_grad()
def eval_dev_metrics(model, tokenizer, forward_fn, layer_idx: int, n_layers: int, dev_items: list[dict], dev_responses: dict) -> dict:
    """Returns {"mse": ..., "nll": ...}, each the mean over dev items (NLL per-item uses
    reduction="sum" over response tokens, same as the training loss below, then averaged across
    items -- i.e. "mean total NLL per response", not per-token, matching how mse is already
    averaged as "mean total subsequent-layers MSE per response")."""
    mse_losses, nll_losses = [], []
    for item in dev_items:
        pair = build_training_pair(model, tokenizer, item, dev_responses)
        if pair is None:
            continue
        target = collect_target_hidden_states(model, pair["full_instr"])
        pred, logits, _ = forward_fn(pair)
        mse_losses.append(subsequent_layers_mse(pred, target, layer_idx, pair["n_resp"], n_layers).item())
        nll_losses.append(response_nll(logits, pair["full_base"], pair["n_resp"]).item())
    return {"mse": sum(mse_losses) / len(mse_losses), "nll": sum(nll_losses) / len(nll_losses)}


def train_gate(
    model, tokenizer, forward_fn, optimizer, layer_idx: int, n_layers: int,
    train_items: list[dict], dev_items: list[dict], train_responses: dict, dev_responses: dict,
    n_epochs: int, reg_coeff: float, nll_weight: float = 0.0, on_epoch_end=None,
) -> tuple[dict, dict]:
    """Returns (baseline_metrics, final_metrics), each a {"mse": float, "nll": float} dict.
    on_epoch_end(epoch, dev_metrics), if given, runs after every epoch with that same dict shape --
    checkpoint FORMAT differs by variant (conceptor saves the conceptor matrix too, proper doesn't),
    so checkpointing stays each variant's own job, not baked into this shared loop."""
    baseline_metrics = eval_dev_metrics(model, tokenizer, forward_fn, layer_idx, n_layers, dev_items, dev_responses)

    for epoch in range(n_epochs):
        for item in train_items:
            pair = build_training_pair(model, tokenizer, item, train_responses)
            if pair is None:
                continue
            target = collect_target_hidden_states(model, pair["full_instr"])
            pred, logits, fit = forward_fn(pair)
            loss = (
                subsequent_layers_mse(pred, target, layer_idx, pair["n_resp"], n_layers)
                + regularization_loss(fit, reg_coeff)
                + nll_weight * response_nll(logits, pair["full_base"], pair["n_resp"])
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        dev_metrics = eval_dev_metrics(model, tokenizer, forward_fn, layer_idx, n_layers, dev_items, dev_responses)
        if on_epoch_end:
            on_epoch_end(epoch, dev_metrics)

    final_metrics = eval_dev_metrics(model, tokenizer, forward_fn, layer_idx, n_layers, dev_items, dev_responses)
    return baseline_metrics, final_metrics
