"""Shared fake model/tokenizer for tests. The attention-mask-respecting mixing in TinyLayer matters:
an earlier version of this without masking silently made every batched-generation test meaningless
(padding tokens corrupted real ones without the test noticing) -- this version was specifically
built to catch that class of bug.
"""
import types

import torch


class TinyLayer(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = torch.nn.Linear(d, d)

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.ones(x.shape[0], x.shape[1], 1)
        masked_x = x * mask
        cumsum = torch.cumsum(masked_x, dim=1)
        count = torch.cumsum(mask, dim=1).clamp(min=1)
        cum_mean = cumsum / count
        return (self.lin(x + 0.3 * cum_mean) + x,)


class TinyModel(torch.nn.Module):
    def __init__(self, d=16, n=3, vocab=200):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, d)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([TinyLayer(d) for _ in range(n)])
        self.device = "cpu"
        self.config = type("Cfg", (), {"hidden_size": d})()

    def forward(self, input_ids, attention_mask=None, output_hidden_states=True):
        h = self.embed(input_ids)
        mask = attention_mask.unsqueeze(-1).float() if attention_mask is not None else None
        hs = [h]
        for layer in self.model.layers:
            (h,) = layer(h, mask)
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


def make_fake_model_and_tokenizer(d: int = 16, n_layers: int = 3):
    return TinyModel(d=d, n=n_layers), TinyTokenizer()
