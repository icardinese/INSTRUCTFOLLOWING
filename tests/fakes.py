"""Shared fake model/tokenizer for tests. The attention-mask-respecting mixing in TinyLayer matters:
an earlier version of this without masking silently made every batched-generation test meaningless
(padding tokens corrupted real ones without the test noticing) -- this version was specifically
built to catch that class of bug.

TinyLayer's `return_tuple` flag exists for the same reason, added after a REAL incident: every
hook in this project (steering/hooks.py, every forward_with_gate_hook) assumed a decoder layer's
forward always returns a `(hidden_states, ...)` tuple. TinyLayer used to only ever return a tuple
too, so nothing in this test suite could ever have caught the real transformers version where a
decoder layer returns hidden_states as a PLAIN TENSOR instead -- the hook's `output[0]` then
silently mis-indexes along the batch dimension instead of raising, corrupting the forward pass
several layers downstream (`AttributeError: 'tuple' object has no attribute 'dtype'`, observed in
a real overnight run). Tests that care about this (see test_steering_psr_gate.py's
tuple-vs-plain-tensor tests) build a model with `return_tuple=False` specifically to exercise the
path that used to be untestable.
"""
import types

import torch


class TinyLayer(torch.nn.Module):
    def __init__(self, d, return_tuple: bool = True):
        super().__init__()
        self.lin = torch.nn.Linear(d, d)
        self.return_tuple = return_tuple

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.ones(x.shape[0], x.shape[1], 1)
        masked_x = x * mask
        cumsum = torch.cumsum(masked_x, dim=1)
        count = torch.cumsum(mask, dim=1).clamp(min=1)
        cum_mean = cumsum / count
        out = self.lin(x + 0.3 * cum_mean) + x
        return (out,) if self.return_tuple else out


class TinyModel(torch.nn.Module):
    def __init__(self, d=16, n=3, vocab=200, layer_returns_tuple: bool = True):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, d)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([TinyLayer(d, return_tuple=layer_returns_tuple) for _ in range(n)])
        self.device = "cpu"
        self.config = type("Cfg", (), {"hidden_size": d})()

    def forward(self, input_ids, attention_mask=None, output_hidden_states=True):
        h = self.embed(input_ids)
        mask = attention_mask.unsqueeze(-1).float() if attention_mask is not None else None
        hs = [h]
        for layer in self.model.layers:
            out = layer(h, mask)
            h = out[0] if isinstance(out, tuple) else out
            hs.append(h)
        out = types.SimpleNamespace()
        out.hidden_states = tuple(hs)
        out.logits = h @ self.embed.weight.T
        return out

    def generate(self, input_ids, attention_mask=None, max_new_tokens=10, **kw):
        cur = input_ids
        mask = attention_mask if attention_mask is not None else torch.ones_like(input_ids)
        for _ in range(max_new_tokens):
            logits = self.forward(cur, attention_mask=mask).logits
            nxt = logits[:, -1, :].argmax(-1, keepdim=True)
            cur = torch.cat([cur, nxt], dim=1)
            mask = torch.cat([mask, torch.ones_like(nxt)], dim=1)
        return cur


class BatchEncodingLike(dict):
    def to(self, device):
        return self

    def __getattr__(self, k):
        return self[k]


class TinyTokenizer:
    eos_token_id = 0
    padding_side = "right"

    def __call__(self, texts, return_tensors=None, add_special_tokens=True, padding=False):
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        all_ids = [[max(1, hash(c) % 199 + 1) for c in t[:200]] or [1] for t in texts]
        if padding:
            max_len = max(len(ids) for ids in all_ids)
            padded, masks = [], []
            for ids in all_ids:
                pad_len = max_len - len(ids)
                if self.padding_side == "left":
                    padded.append([0] * pad_len + ids)
                    masks.append([0] * pad_len + [1] * len(ids))
                else:
                    padded.append(ids + [0] * pad_len)
                    masks.append([1] * len(ids) + [0] * pad_len)
            return BatchEncodingLike(input_ids=torch.tensor(padded), attention_mask=torch.tensor(masks))
        ids = all_ids[0]
        return BatchEncodingLike(input_ids=torch.tensor([ids]))

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i.item() if hasattr(i, "item") else i) for i in ids if (i.item() if hasattr(i, "item") else i) != 0)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return messages[0]["content"]


def make_fake_model_and_tokenizer(d: int = 16, n_layers: int = 3, layer_returns_tuple: bool = True):
    return TinyModel(d=d, n=n_layers, layer_returns_tuple=layer_returns_tuple), TinyTokenizer()
