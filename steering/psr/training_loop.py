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
import time

import torch
from transformers import get_scheduler

from steering.psr.data import build_training_pair
from steering.psr.gate import collect_target_hidden_states, regularization_loss, subsequent_layers_mse
from steering.psr.nll import response_nll
from steering.psr.reference_config import DATA_SHUFFLE_SEED


@torch.no_grad()
def build_target_cache(model, tokenizer, items: list[dict], responses: dict,
                       layer_idx: int, n_layers: int) -> list:
    """Precompute, ONCE per item, everything a training/eval step needs that does not depend on
    the trainable parameters: the tokenized pair and the target hidden states.

    WHY. The target is the frozen model's hidden states on the INSTRUCTED sequence. The model is
    frozen (requires_grad False) and in eval() mode, and the input never changes -- so the target
    is a constant per item. Previously it was recomputed on every training step of every epoch,
    and again on every dev eval: 15 epochs x 180 items plus 16 evals x 60 items of full forward
    passes over the ~1.4k-token instructed prompt, including a 152k-vocab LM head whose output was
    discarded. Measured against the triage config that is ~67% of all compute per grid point.
    Caching it is the single largest speedup available and changes no numbers: the cached value
    is produced by the identical call (collect_target_hidden_states) on the identical input.

    WHAT IS KEPT. Only the slice subsequent_layers_mse actually reads -- layers layer_idx+1 ..
    n_layers, positions -(n_resp+1): -- stored as (1, n_resp+1, d) so the existing
    `hidden_target[idx][0, -(n_resp+1):, :]` indexing returns it unchanged. Unused layer indices
    are None. Each slice is .clone()d: a bare slice is a VIEW that pins the entire (29-layer x
    full-sequence) storage, which across 240 items would be ~70GB rather than ~6GB.

    ORDER. The returned list is ALIGNED with `items`, with None where build_training_pair returned
    None. Keeping the None placeholders means shuffling this list with the same RNG yields exactly
    the same permutation as shuffling `items` did, so the item order each epoch is unchanged.
    """
    cache = []
    with torch.no_grad():
        for item in items:
            pair = build_training_pair(model, tokenizer, item, responses)
            if pair is None:
                cache.append(None)
                continue
            full = collect_target_hidden_states(model, pair["full_instr"])
            span = pair["n_resp"] + 1
            target = [None] * (n_layers + 1)
            for idx in range(layer_idx + 1, n_layers + 1):
                target[idx] = full[idx][:, -span:, :].clone()
            del full
            cache.append((pair, target))
    return cache


@torch.no_grad()
def eval_dev_metrics(model, tokenizer, forward_fn, layer_idx: int, n_layers: int, dev_items: list[dict],
                     dev_responses: dict, cache: list | None = None) -> dict:
    """Returns {"mse": ..., "nll": ..., "nll_sum": ...}, each the mean over dev items.

    "nll" is now MEAN-reduced (per-token), matching what the training loss optimizes as of
    2026-09-20 -- see steering/psr/nll.py for why the reference's mean reduction beats the
    paper's sum notation. "nll_sum" preserves the old sum-reduced number so rows logged in
    psr_proper_sweep.jsonl before that change remain comparable; it is a diagnostic only and is
    never optimized."""
    # @torch.no_grad(): this function never calls backward(). Without it, the gate parameters'
    # requires_grad made every dev forward build a full autograd graph and keep every activation
    # alive -- memory and time spent on a graph that was then thrown away.
    if cache is None:
        cache = build_target_cache(model, tokenizer, dev_items, dev_responses, layer_idx, n_layers)
    mse_losses, nll_losses, nll_sum_losses = [], [], []
    for entry in cache:
        if entry is None:
            continue
        pair, target = entry
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
    # Targets are constant per item (frozen model, eval mode, fixed input), so compute them ONCE
    # here instead of on every step of every epoch. See build_target_cache for the full rationale;
    # this is ~67% of per-grid-point compute on triage and changes no numbers.
    _t0 = time.time()
    train_cache = build_target_cache(model, tokenizer, train_items, train_responses, layer_idx, n_layers)
    dev_cache = build_target_cache(model, tokenizer, dev_items, dev_responses, layer_idx, n_layers)
    print(f"  target cache: {sum(e is not None for e in train_cache)} train + "
          f"{sum(e is not None for e in dev_cache)} dev items in {time.time() - _t0:.1f}s", flush=True)

    baseline_metrics = eval_dev_metrics(model, tokenizer, forward_fn, layer_idx, n_layers, dev_items,
                                        dev_responses, cache=dev_cache)

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
        # Shuffling the item-ALIGNED cache (None placeholders kept) consumes the RNG identically
        # to shuffling train_items, so each epoch visits items in exactly the same order as before.
        epoch_entries = list(train_cache)
        rng.shuffle(epoch_entries)
        for entry in epoch_entries:
            pair = None if entry is None else entry[0]
            if pair is None:
                continue
            target = entry[1]
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

        dev_metrics = eval_dev_metrics(model, tokenizer, forward_fn, layer_idx, n_layers, dev_items, dev_responses, cache=dev_cache)
        if on_epoch_end:
            on_epoch_end(epoch, dev_metrics)

    final_metrics = eval_dev_metrics(model, tokenizer, forward_fn, layer_idx, n_layers, dev_items, dev_responses, cache=dev_cache)
    return baseline_metrics, final_metrics
