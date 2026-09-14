"""Auxiliary log-likelihood objective for gate-based PSR training (Heyman & Vandeputte 2026, Eq. 4):

    L_NLL = - sum_{t=1}^{T} log P_M(y_t | y_{<t}, x_{1:T}; h'_x)

Motivation (from the same source): pure MSE alignment between the corrected hidden state and the
instructed-prompt target can harshly penalize a correction that is semantically valid but doesn't
land at exactly the same point in activation space -- e.g. a paraphrase of the target continuation.
L_NLL doesn't care about matching a specific hidden-state target; it only asks whether the model,
CONDITIONED ON THE CORRECTION (h'_x), still assigns high likelihood to the actual response tokens
(y_t) it was trained to produce. That makes it a softer, complementary signal to the hard MSE term,
not a replacement for it -- see steering/psr/training_loop.py's combined loss.

This lives in steering/psr/ (method-shared, task-agnostic) rather than any single variant's folder
because every gate-based PSR variant (proper, conceptor fixed-vector, conceptor/matrix,
conceptor/selfproj) computes it the exact same way once it has (logits, input_ids, n_resp) --
compare to steering/psr/gate.py's subsequent_layers_mse, which this is a direct sibling of.
"""
import torch
import torch.nn.functional as F


def response_nll(logits: torch.Tensor, input_ids: torch.Tensor, n_resp: int, reduction: str = "sum") -> torch.Tensor:
    """logits: (1, seq_len, vocab) from the SAME corrected forward pass forward_with_gate_hook
    already ran (h'_x is baked into these logits, not recomputed separately -- so this is free
    besides one extra cross_entropy call). input_ids: (1, seq_len), the same sequence that was fed
    in (prompt + response tokens). n_resp: length of the response span at the END of input_ids.

    Standard teacher-forced next-token cross-entropy, restricted to the response span: the token at
    position p (one of the last n_resp positions) is "predicted" by logits at position p-1, so the
    relevant slice is logits[:, -(n_resp+1):-1, :] against targets input_ids[:, -n_resp:].

    reduction="sum" matches the literal equation (a sum over t, not a mean) -- deliberately, since
    fidelity to the paper's actual formula is the point of this fix (see finding #1 in the project
    handoff: the original port's fidelity gaps were exactly this kind of "close enough" substitution).
    Note this DOES make the raw loss value scale with n_resp (longer responses contribute a larger
    sum) -- pass reduction="mean" instead if you want a length-normalized variant for comparing
    across items of very different response lengths; "sum" is what Eq. (4) actually specifies.
    """
    if n_resp < 1:
        raise ValueError(f"response_nll needs n_resp >= 1, got {n_resp}")
    seq_len = input_ids.shape[1]
    if n_resp + 1 > seq_len:
        raise ValueError(f"n_resp={n_resp} leaves no context token before the response span (seq_len={seq_len})")

    shift_logits = logits[:, -(n_resp + 1):-1, :].float()
    shift_targets = input_ids[:, -n_resp:]
    vocab = shift_logits.shape[-1]
    return F.cross_entropy(shift_logits.reshape(-1, vocab), shift_targets.reshape(-1), reduction=reduction)
