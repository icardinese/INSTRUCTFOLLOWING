"""The PSR training loop -- byte-for-byte identical across every variant (proper, conceptor,
conceptor/matrix, conceptor/selfproj). What differs between them is only WHICH correction mechanism
`forward_fn` calls (steering.psr.gate.forward_with_gate_hook vs. conceptor/matrix/logic's vs.
conceptor/selfproj/logic's), and each variant's train.py builds that closure over whatever direction/
conceptor/mu_instr/delta_scale it needs. This file never imports any of those -- it only ever calls
whatever `forward_fn` it's handed (which must now return (hidden_states, logits, fit_vals) -- see
steering/psr/gate.py's forward_with_gate_hook docstring).

Loss = mse_weight * subsequent-layers MSE + dead-gate regularization + nll_weight * auxiliary NLL
(Eq. 4, see steering/psr/nll.py). Both weights are independent -- mse_weight=1.0, nll_weight=0.0
is pure MSE (the default, so every existing call site and every existing result reproduces
unchanged); mse_weight=0.0, nll_weight=1.0 is pure NLL, with NO MSE term in the gradient at all.
This is the real distinction Heyman & Vandeputte draw in their paper: MSE and loglikelihood are
trained as two SEPARATE, mutually-exclusive objectives (their `_MSE` vs `_LL` variants), not
blended into one combined loss -- see Section 3.5 of the actual paper ("Loglikelihood (LL). As an
ALTERNATIVE to MSE..."). Earlier language in this project's own draft ("we incorporate an
auxiliary log-likelihood objective... integrating this into the total loss") describes a
DIFFERENT, additive design that is this project's own proposed extension, not a replication of
H&V's ablation -- mse_weight=1.0 with a nonzero nll_weight is that additive blend; mse_weight=0.0
is the faithful, mutually-exclusive replication of H&V's `_LL` variant. Both are supported, as
different points in the same (mse_weight, nll_weight) space, rather than picking one design.
Regularization is independent of both weights -- it guards against the gate dying (always-zero
output), which is orthogonal to which representation-matching objective is active.

Dev MSE and dev NLL are ALWAYS both computed and reported (regardless of either weight) purely as
diagnostics -- cheap, and directly useful for the "what does pure-NLL training do to the MSE
metric, and vice versa" question a real MSE-vs-LL ablation needs to answer.
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
    n_epochs: int, reg_coeff: float, mse_weight: float = 1.0, nll_weight: float = 0.0, on_epoch_end=None,
) -> tuple[dict, dict]:
    """Returns (baseline_metrics, final_metrics), each a {"mse": float, "nll": float} dict.
    on_epoch_end(epoch, dev_metrics), if given, runs after every epoch with that same dict shape --
    checkpoint FORMAT differs by variant (conceptor saves the conceptor matrix too, proper doesn't),
    so checkpointing stays each variant's own job, not baked into this shared loop.

    mse_weight=0.0 means the MSE term contributes NO gradient at all (not just a small one) --
    subsequent_layers_mse is still COMPUTED (dev metrics always report it), just multiplied by
    zero before being added to the backprop loss, which is exactly "pure NLL training" rather than
    "mostly NLL training with a token gesture at MSE."."""
    baseline_metrics = eval_dev_metrics(model, tokenizer, forward_fn, layer_idx, n_layers, dev_items, dev_responses)

    for epoch in range(n_epochs):
        for item in train_items:
            pair = build_training_pair(model, tokenizer, item, train_responses)
            if pair is None:
                continue
            target = collect_target_hidden_states(model, pair["full_instr"])
            pred, logits, fit = forward_fn(pair)
            loss = (
                mse_weight * subsequent_layers_mse(pred, target, layer_idx, pair["n_resp"], n_layers)
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
