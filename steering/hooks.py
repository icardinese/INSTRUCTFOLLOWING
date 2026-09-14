"""Hook attachment mechanics shared by every steering method. Contains zero method-specific math --
just the plumbing for registering/removing forward hooks on one or several decoder layers. Every
steering/<method>/ folder imports from here; nothing here imports from any of them.
"""
from contextlib import contextmanager
from typing import Callable

import torch
from transformers import PreTrainedModel

from core.model_common import get_decoder_layers


@contextmanager
def steering_hook(model: PreTrainedModel, layer_idx: int, hook_fn: Callable[[torch.Tensor], torch.Tensor]):
    """Registers a forward hook on decoder layer `layer_idx` that rewrites its output hidden states."""
    layer = get_decoder_layers(model)[layer_idx]

    def wrapped(module, inputs, output):
        hidden = hook_fn(output[0])
        return (hidden,) + tuple(output[1:])

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
                hidden = fn(output[0])
                return (hidden,) + tuple(output[1:])

            return wrapped

        handles.append(layer.register_forward_hook(_make_wrapped(hook_fn)))
    try:
        yield
    finally:
        for h in handles:
            h.remove()
