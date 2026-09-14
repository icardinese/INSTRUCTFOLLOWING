"""Tests for steering/hooks.py's unwrap_hidden/rewrap_hidden -- these exist specifically because a
real transformers version returns a decoder layer's output as a plain tensor instead of the
classic (hidden_states, ...) tuple, and a hook that assumed "always a tuple" crashed several
layers downstream with a confusing AttributeError. Covers plain tensor, tuple, list, and an
HF-ModelOutput-shaped object (indexable via [0]/[1:] but NOT a literal tuple instance) --
the last one matters because a naive `isinstance(output, tuple)` check would wrongly treat a
ModelOutput as "plain tensor" and return the whole container as if it WERE hidden_states, a
second, subtler path to the same crash.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from steering.hooks import rewrap_hidden, unwrap_hidden


class _FakeModelOutput:
    """Minimal stand-in for an HF ModelOutput: NOT a tuple instance, but supports [0]/[1:]
    indexing, same as the real thing."""

    def __init__(self, *values):
        self._values = values

    def __getitem__(self, item):
        return self._values[item]


def test_unwrap_plain_tensor():
    hidden = torch.randn(1, 5, 8)
    unwrapped, rest = unwrap_hidden(hidden)
    assert unwrapped is hidden
    assert rest is None


def test_unwrap_tuple():
    hidden = torch.randn(1, 5, 8)
    extra = torch.randn(2)
    unwrapped, rest = unwrap_hidden((hidden, extra))
    assert unwrapped is hidden
    assert rest == (extra,)


def test_unwrap_list():
    hidden = torch.randn(1, 5, 8)
    extra = "some_cache_object"
    unwrapped, rest = unwrap_hidden([hidden, extra])
    assert unwrapped is hidden
    assert rest == (extra,)


def test_unwrap_model_output_shaped_object_not_a_literal_tuple():
    """The real regression case: a container that supports sequence indexing but ISN'T an
    instance of `tuple` -- isinstance(output, tuple) would be False here, which is exactly why
    unwrap_hidden checks for torch.Tensor instead (see its docstring)."""
    hidden = torch.randn(1, 5, 8)
    extra = torch.randn(2)
    fake_output = _FakeModelOutput(hidden, extra)
    assert not isinstance(fake_output, tuple)

    unwrapped, rest = unwrap_hidden(fake_output)
    assert unwrapped is hidden
    assert rest == (extra,)


def test_rewrap_roundtrip_plain_tensor():
    hidden = torch.randn(1, 5, 8)
    new_hidden = hidden + 1
    _, rest = unwrap_hidden(hidden)
    rewrapped = rewrap_hidden(new_hidden, rest)
    assert isinstance(rewrapped, torch.Tensor)
    assert torch.equal(rewrapped, new_hidden)


def test_rewrap_roundtrip_tuple():
    hidden = torch.randn(1, 5, 8)
    extra = torch.randn(2)
    new_hidden = hidden + 1
    _, rest = unwrap_hidden((hidden, extra))
    rewrapped = rewrap_hidden(new_hidden, rest)
    assert isinstance(rewrapped, tuple)
    assert torch.equal(rewrapped[0], new_hidden)
    assert torch.equal(rewrapped[1], extra)


def test_rewrap_roundtrip_model_output_shaped_object():
    hidden = torch.randn(1, 5, 8)
    extra = torch.randn(2)
    new_hidden = hidden + 1
    _, rest = unwrap_hidden(_FakeModelOutput(hidden, extra))
    rewrapped = rewrap_hidden(new_hidden, rest)
    assert isinstance(rewrapped, tuple)  # rewrap always normalizes back to a plain tuple
    assert torch.equal(rewrapped[0], new_hidden)
    assert torch.equal(rewrapped[1], extra)
