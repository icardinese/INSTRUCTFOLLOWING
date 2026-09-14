"""Builds the specific tensor shape PSR-style contrastive training needs: a base-prompt sequence
and an instructed-prompt sequence sharing the same teacher-forced response tokens. This is PSR-
specific structure, not generic data loading -- a method that isn't doing base-vs-instructed
contrastive training (task_matrix, projection) has no reason to import this.
"""
import torch


def build_training_pair(model, tokenizer, item: dict, responses: dict) -> dict | None:
    """item: {"id", "base_prompt", "terse_prompt"}. Returns None for empty-response items (skip,
    don't pad -- batch size is always 1 here, so there's no batching to preserve by padding)."""
    teacher_response = responses[str(item["id"])]
    resp_ids = tokenizer(teacher_response, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
    if resp_ids.shape[1] == 0:
        return None

    base_ids = tokenizer(item["base_prompt"], return_tensors="pt")["input_ids"].to(model.device)
    instr_ids = tokenizer(item["terse_prompt"], return_tensors="pt")["input_ids"].to(model.device)
    return {
        "full_base": torch.cat([base_ids, resp_ids], dim=1),
        "full_instr": torch.cat([instr_ids, resp_ids], dim=1),
        "n_resp": resp_ids.shape[1],
    }


@torch.no_grad()
def pool_separate_poles(model, tokenizer, items: list[dict], layer_idx: int, responses: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Base-prompt and instructed-prompt response-token activations, kept SEPARATE (not
    concatenated) -- needed for computing a diff-in-means direction (instr_pool.mean(0) -
    base_pool.mean(0)), which pool_bipolar_activations's concatenated version can't give you."""
    base_list, instr_list = [], []
    for item in items:
        pair = build_training_pair(model, tokenizer, item, responses)
        if pair is None:
            continue
        n_resp = pair["n_resp"]
        out_base = model(input_ids=pair["full_base"], output_hidden_states=True)
        out_instr = model(input_ids=pair["full_instr"], output_hidden_states=True)
        base_list.append(out_base.hidden_states[layer_idx + 1][0, -n_resp:, :].float().cpu())
        instr_list.append(out_instr.hidden_states[layer_idx + 1][0, -n_resp:, :].float().cpu())
    return torch.cat(base_list, dim=0), torch.cat(instr_list, dim=0)


def load_or_pool_separate_poles(model, tokenizer, items: list[dict], layer_idx: int, responses: dict, cache_dir) -> tuple[torch.Tensor, torch.Tensor]:
    """Disk-cached wrapper around pool_separate_poles. Pooling doesn't depend on alpha -- only the
    closed-form conceptor step built on top of it does -- so an alpha sweep, or running proper/
    conceptor/matrix/selfproj back to back at the same layer, should pool ONCE and reuse this, not
    redundantly re-pool (a real forward-pass cost) on every run."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    base_path = cache_dir / f"pooled_base_layer{layer_idx}.pt"
    instr_path = cache_dir / f"pooled_instr_layer{layer_idx}.pt"
    if base_path.exists() and instr_path.exists():
        return torch.load(base_path), torch.load(instr_path)
    base_pool, instr_pool = pool_separate_poles(model, tokenizer, items, layer_idx, responses)
    torch.save(base_pool, base_path)
    torch.save(instr_pool, instr_path)
    return base_pool, instr_pool


def pool_bipolar_activations(model, tokenizer, items: list[dict], layer_idx: int, responses: dict) -> torch.Tensor:
    """Both poles concatenated together -- this is the raw material compute_conceptor's correlation
    matrix gets built from. "Bipolar" (both poles pooled together) matches Triantafyllopoulos et
    al.'s training choice. Built on pool_separate_poles rather than duplicating the forward-pass
    logic -- if you also need the separate pools (e.g. for a diff-in-means direction), call that
    directly instead of concatenating this result back apart."""
    base_pool, instr_pool = pool_separate_poles(model, tokenizer, items, layer_idx, responses)
    return torch.cat([base_pool, instr_pool], dim=0)
