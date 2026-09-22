"""Caches teacher-forced generations keyed by item id. Nothing here is PSR-specific -- any method
needing "generate once, reuse across epochs/configs" reuses this exactly as-is.

CACHE FORMAT (changed 2026-09-20): values are now {"text": str, "finished": bool} rather than a
bare string. `finished` records whether generation stopped on its own or hit the max_new_tokens
cap, which steering/psr/data.py needs in order to decide whether appending an end-of-turn token to
the teacher-forced response is legitimate (see generate_response_with_meta's docstring).

OLD CACHES STILL LOAD. A bare-string value is read as {"text": <str>, "finished": None}, where
None means "unknown". Callers must treat unknown as not-finished, since assuming a truncated
response terminated naturally is the failure mode that actually corrupts training. Delete the
cache file and regenerate to get real flags -- generation is batched and cheap; it was never the
bottleneck.
"""
import json


def normalize_cache_entry(value) -> dict:
    """Accepts either the new dict form or a legacy bare string. Returns the dict form.

    `finished` is None (not False) for legacy entries, so that "we never recorded this" stays
    distinguishable from "we recorded that it was truncated" in any future diagnostic. Both are
    treated as not-safe-to-append downstream."""
    if isinstance(value, str):
        return {"text": value, "finished": None}
    return {"text": value["text"], "finished": value.get("finished")}


def response_text(value) -> str:
    """The generated text, from either cache format. Use this anywhere the old code did a bare
    `responses[str(item_id)]` and only wanted the string."""
    return normalize_cache_entry(value)["text"]


def precompute_responses(model, tokenizer, items: list[dict], generate_response_with_meta) -> dict:
    """items: [{"id": ..., "terse_prompt": ...}, ...]. The generation function is passed in rather
    than imported, so this file has zero dependency on any specific model-loading module."""
    return {
        str(item["id"]): generate_response_with_meta(model, tokenizer, item["terse_prompt"])
        for item in items
    }


def load_or_compute_responses(model, tokenizer, items: list[dict], cache_path, generate_response_with_meta) -> dict:
    if cache_path.exists():
        with cache_path.open() as f:
            raw = json.load(f)
        return {k: normalize_cache_entry(v) for k, v in raw.items()}
    responses = precompute_responses(model, tokenizer, items, generate_response_with_meta)
    with cache_path.open("w") as f:
        json.dump(responses, f)
    return responses
