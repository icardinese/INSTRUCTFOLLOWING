"""Caches teacher-forced generations keyed by item id. Nothing here is PSR-specific -- any method
needing "generate once, reuse across epochs/configs" reuses this exactly as-is.
"""
import json


def precompute_responses(model, tokenizer, items: list[dict], generate_response) -> dict:
    """items: [{"id": ..., "terse_prompt": ...}, ...]. generate_response is passed in rather than
    imported, so this file has zero dependency on any specific model-loading module."""
    return {str(item["id"]): generate_response(model, tokenizer, item["terse_prompt"]) for item in items}


def load_or_compute_responses(model, tokenizer, items: list[dict], cache_path, generate_response) -> dict:
    if cache_path.exists():
        with cache_path.open() as f:
            return json.load(f)
    responses = precompute_responses(model, tokenizer, items, generate_response)
    with cache_path.open("w") as f:
        json.dump(responses, f)
    return responses
