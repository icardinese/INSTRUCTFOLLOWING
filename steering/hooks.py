"""Hook attachment mechanics shared by every steering method. Contains zero method-specific math --
just the plumbing for registering/removing forward hooks on one or several decoder layers. Every
steering/<method>/ folder imports from here; nothing here imports from any of them.
"""
from contextlib import contextmanager
from typing import Callable

import torch
from transformers import PreTrainedModel

from core.model_common import get_decoder_layers


def unwrap_hidden(output):
    """A decoder layer's forward can return EITHER a plain hidden_states tensor OR a tuple whose
    first element is hidden_states, depending on transformers version and whether cache/attention
    outputs are requested -- confirmed as a REAL, version-dependent difference (not a hypothetical
    one): a hook written assuming "always a tuple" silently mis-indexes a plain tensor (`output[0]`
    on a tensor selects along the batch dimension instead of raising) and then wraps the result
    BACK into a tuple, so the next layer receives a tuple where it expects a tensor --
    `AttributeError: 'tuple' object has no attribute 'dtype'` deep inside the model's own forward,
    on a transformers version that returns a plain tensor here. Returns (hidden_states, rest),
    where `rest` is None if the original output was a plain tensor (so rewrap_hidden knows to
    return a plain tensor back, not a 1-tuple) or the remaining tuple elements otherwise."""
    if isinstance(output, tuple):
        return output[0], output[1:]
    return output, None


def rewrap_hidden(new_hidden: torch.Tensor, rest):
    """Inverse of unwrap_hidden -- reconstructs whatever shape the original output had (plain
    tensor if rest is None, else a tuple with new_hidden in the first slot)."""
    if rest is None:
        return new_hidden
    return (new_hidden,) + tuple(rest)


@contextmanager
def steering_hook(model: PreTrainedModel, layer_idx: int, hook_fn: Callable[[torch.Tensor], torch.Tensor]):
    """Registers a forward hook on decoder layer `layer_idx` that rewrites its output hidden states."""
    layer = get_decoder_layers(model)[layer_idx]

    def wrapped(module, inputs, output):
        hidden, rest = unwrap_hidden(output)
        return rewrap_hidden(hook_fn(hidden), rest)

    handle = layer.register_forward_hook(wrapped)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def multi_steering_hook(model: PreTrainedModel, hooks: dict[int, Callable[[torch.Tensor], torch.Tensor]]):
    """Registers a forward hook on several decoder layers at once (A-PSR / A-Const). Each layer's hook
    sees whatever activation actually arrives there at inference time -- which already reflects any
    correction applied by earlier layers, since hooks are chained through the real forward pass. This
    is where the "iteratively apply the intervention at all layers" behavior from the PSR paper actually
    happens; nothing extra needs to be done here to make that occur, it falls out of hooking multiple
    layers in the same forward pass."""
    decoder_layers = get_decoder_layers(model)
    handles = []
    for layer_idx, hook_fn in hooks.items():
        layer = decoder_layers[layer_idx]

        def _make_wrapped(fn):
            # Factory function is required here: without it, every hook in the loop would close over
            # the same loop variable `hook_fn` by reference (Python's late-binding closures), so all
            # layers would end up running whichever hook_fn was assigned LAST in the loop.
            def wrapped(module, inputs, output):
                hidden, rest = unwrap_hidden(output)
                return rewrap_hidden(fn(hidden), rest)

            return wrapped

        handles.append(layer.register_forward_hook(_make_wrapped(hook_fn)))
    try:
        yield
    finally:
        for h in handles:
            h.remove()
