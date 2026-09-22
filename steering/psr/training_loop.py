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
import random

import torch
from transformers import get_scheduler

from steering.psr.data import build_training_pair
from steering.psr.gate import collect_target_hidden_states, regularization_loss, subsequent_layers_mse
from steering.psr.nll import response_nll
from steering.psr.reference_config import DATA_SHUFFLE_SEED


@torch.no_grad()
def eval_dev_metrics(model, tokenizer, forward_fn, layer_idx: int, n_layers: int, dev_items: list[dict], dev_responses: dict) -> dict:
    """Returns {"mse": ..., "nll": ..., "nll_sum": ...}, each the mean over dev items.

    "nll" is now MEAN-reduced (per-token), matching what the training loss optimizes as of
    2026-09-20 -- see steering/psr/nll.py for why the reference's mean reduction beats the
    paper's sum notation. "nll_sum" preserves the old sum-reduced number so rows logged in
    psr_proper_sweep.jsonl before that change remain comparable; it is a diagnostic only and is
    never optimized."""
    mse_losses, nll_losses, nll_sum_losses = [], [], []
    for item in dev_items:
        pair = build_training_pair(model, tokenizer, item, dev_responses)
        if pair is None:
            continue
        target = collect_target_hidden_states(model, pair["full_instr"])
        pred, logits, _ = forward_fn(pair)
        mse_losses.append(subsequent_layers_mse(pred, target, layer_idx, pair["n_resp"], n_layers).item())
        nll_losses.append(response_nll(logits, pair["full_base"], pair["n_resp"], reduction="mean").item())
        nll_sum_losses.append(response_nll(logits, pair["full_base"], pair["n_resp"], reduction="sum").item())
    n = len(mse_losses)
    return {
        "mse": sum(mse_losses) / n,
        "nll": sum(nll_losses) / n,
        "nll_sum": sum(nll_sum_losses) / n,
    }


def train_gate(
    model, tokenizer, forward_fn, optimizer, layer_idx: int, n_layers: int,
    train_items: list[dict], dev_items: list[dict], train_responses: dict, dev_responses: dict,
    n_epochs: int, reg_coeff: float, mse_weight: float = 1.0, nll_weight: float = 0.0,
    on_epoch_end=None, normalize_psi: bool = False, data_shuffle_seed: int = DATA_SHUFFLE_SEED,
    use_lr_schedule: bool = True,
) -> tuple[dict, dict]:
    """Returns (baseline_metrics, final_metrics), each a {"mse": float, "nll": float} dict.
    on_epoch_end(epoch, dev_metrics), if given, runs after every epoch with that same dict shape --
    checkpoint FORMAT differs by variant (conceptor saves the conceptor matrix too, proper doesn't),
    so checkpointing stays each variant's own job, not baked into this shared loop.

    mse_weight=0.0 means the MSE term contributes NO gradient at all (not just a small one) --
    subsequent_layers_mse is still COMPUTED (dev metrics always report it), just multiplied by
    zero before being added to the backprop loss, which is exactly "pure NLL training" rather than
    "mostly NLL training with a token gesture at MSE.".

    normalize_psi divides the MSE term by its own pre-training value, reproducing the reference's
    LossSpecification(normalize_psi_loss=True). It is COUPLED to reg_coeff and should not be set
    independently -- see steering/psr/reference_config.LOSS_BALANCE_PACKAGES for why the mixed
    settings are incoherent. Callers should pass both from reference_config.loss_balance().

    Data is shuffled each epoch from a seeded RNG, matching the reference's
    DataLoader(shuffle=True, generator=manual_seed(123)). With batch size 1 and a handful of
    epochs, a fixed item order lets the last items seen disproportionately determine the final
    weights.

    LR follows a linear decay to zero with no warmup over the full run, matching the reference's
    get_scheduler("linear", num_warmup_steps=0, num_training_steps=total_steps)."""
    baseline_metrics = eval_dev_metrics(model, tokenizer, forward_fn, layer_idx, n_layers, dev_items, dev_responses)

    # Normalizer is fixed once, BEFORE training, from the same dev-set MSE the reference uses as
    # its average_psi_before_training. Recomputing it per epoch would make the loss a moving
    # target and silently flatten any real improvement.
    psi_norm = 1.0
    if normalize_psi:
        psi_norm = baseline_metrics["mse"] if baseline_metrics["mse"] > 0 else 1.0

    total_steps = max(1, n_epochs * len(train_items))
    lr_scheduler = (
        get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        if use_lr_schedule else None
    )
    rng = random.Random(data_shuffle_seed)

    for epoch in range(n_epochs):
        epoch_items = list(train_items)
        rng.shuffle(epoch_items)
        for item in epoch_items:
            pair = build_training_pair(model, tokenizer, item, train_responses)
            if pair is None:
                continue
            target = collect_target_hidden_states(model, pair["full_instr"])
            pred, logits, fit = forward_fn(pair)
            loss = (
                mse_weight * (subsequent_layers_mse(pred, target, layer_idx, pair["n_resp"], n_layers) / psi_norm)
                + regularization_loss(fit, reg_coeff)
                + nll_weight * response_nll(logits, pair["full_base"], pair["n_resp"])
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

        dev_metrics = eval_dev_metrics(model, tokenizer, forward_fn, layer_idx, n_layers, dev_items, dev_responses)
        if on_epoch_end:
            on_epoch_end(epoch, dev_metrics)

    final_metrics = eval_dev_metrics(model, tokenizer, forward_fn, layer_idx, n_layers, dev_items, dev_responses)
    return baseline_metrics, final_metrics
