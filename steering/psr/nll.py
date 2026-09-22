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


def response_nll(logits: torch.Tensor, input_ids: torch.Tensor, n_resp: int, reduction: str = "mean") -> torch.Tensor:
    """logits: (1, seq_len, vocab) from the SAME corrected forward pass forward_with_gate_hook
    already ran (h'_x is baked into these logits, not recomputed separately -- so this is free
    besides one extra cross_entropy call). input_ids: (1, seq_len), the same sequence that was fed
    in (prompt + response tokens). n_resp: length of the response span at the END of input_ids.

    Standard teacher-forced next-token cross-entropy over the STEERED SPAN, which is the last
    prompt token plus the n_resp response tokens -- NOT the response tokens alone. The reference
    builds its labels as `where(token_positions >= last_input_token_position, input_ids, -100)`
    (steering_base.py collate_fn), and last_input_token_position is the index of the FINAL PROMPT
    token (tokenization_utils.compute_last_input_token_index returns len(prompt_tokens) - 1).
    So their supervised span is n_resp + 1 targets, and it lines up exactly with the span
    steering/psr/gate.py::answer_only_mask steers. Keeping the loss span and the steering span
    identical is the point; they were off by one from each other before 2026-09-20.

    A target at position p is predicted by logits at position p-1, so the slice is
    logits[:, -(n_resp+2):-1, :] against targets input_ids[:, -(n_resp+1):].

    reduction DEFAULTS TO "mean", not "sum", even though Eq. (4) is written as a sum over t.
    This is a deliberate choice of the reference IMPLEMENTATION over the paper's notation, for
    two reasons:

    1. The reference computes this loss as HuggingFace's causal-LM loss
       (`model_outputs.clm_outputs.loss` in steering_base.py::forward_pass_steered_model, with
       labels masked to -100 outside the response span). That loss is a MEAN over non-ignored
       tokens. Their lr=1e-3 was tuned against a mean-reduced loss; pairing that same lr with a
       sum reduction multiplies the effective learning rate by roughly the response length
       (~60x here). Adopting their lr but not their reduction is LESS faithful than adopting
       both, not more.
    2. Under "sum", each training example is weighted by however long its teacher response
       happened to be. In a study whose dependent variable IS output length, that lets response
       length silently reweight the training signal.

    "sum" remains available for diagnostics, and eval_dev_metrics reports both so that
    previously-logged sweep rows stay comparable.
    """
    if n_resp < 1:
        raise ValueError(f"response_nll needs n_resp >= 1, got {n_resp}")
    seq_len = input_ids.shape[1]
    if n_resp + 2 > seq_len:
        raise ValueError(
            f"n_resp={n_resp} leaves no context token before the steered span "
            f"(seq_len={seq_len}); the span is n_resp+1 tokens and needs one more for context"
        )

    shift_logits = logits[:, -(n_resp + 2):-1, :].float()
    shift_targets = input_ids[:, -(n_resp + 1):]
    vocab = shift_logits.shape[-1]
    return F.cross_entropy(shift_logits.reshape(-1, vocab), shift_targets.reshape(-1), reduction=reduction)
