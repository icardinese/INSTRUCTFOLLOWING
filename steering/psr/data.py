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


@torch.no_grad()
def pool_prompt_last_token(model, tokenizer, items: list[dict], layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Base-prompt and instructed-prompt activations at the PROMPT's own last token -- no teacher-
    forced response involved at all. This is the ORIGINAL (pre-refactor) contrastive signal the
    project's Const/PSR-Conceptor methods were built on (see caveman-steer's steering_const.py
    ``hidden_at_last_token_all_layers``): "how does the model's representation of the instruction
    differ, right at the point where generation is about to begin" -- as opposed to
    pool_separate_poles' response-token-pooled signal, which mixes many heterogeneous positions
    across a whole generated response together. Kept as a SEPARATE function (not a mode flag on
    pool_separate_poles) since the two pool fundamentally different things: one row per item here
    (one prompt, one last-token vector), versus one row per RESPONSE TOKEN there.

    `responses` is deliberately not a parameter -- unlike every other pooling function in this
    module, this one never touches the teacher-forced response at all, by design."""
    base_list, instr_list = [], []
    for item in items:
        base_ids = tokenizer(item["base_prompt"], return_tensors="pt")["input_ids"].to(model.device)
        instr_ids = tokenizer(item["terse_prompt"], return_tensors="pt")["input_ids"].to(model.device)
        out_base = model(input_ids=base_ids, output_hidden_states=True)
        out_instr = model(input_ids=instr_ids, output_hidden_states=True)
        base_list.append(out_base.hidden_states[layer_idx + 1][0, -1, :].float().cpu())
        instr_list.append(out_instr.hidden_states[layer_idx + 1][0, -1, :].float().cpu())
    return torch.stack(base_list, dim=0), torch.stack(instr_list, dim=0)


def load_or_pool_prompt_last_token(model, tokenizer, items: list[dict], layer_idx: int, cache_dir) -> tuple[torch.Tensor, torch.Tensor]:
    """Disk-cached wrapper around pool_prompt_last_token, same discipline as
    load_or_pool_separate_poles -- this doesn't depend on alpha or loss-config either, so an alpha
    x loss-config grid at one layer should pool once and reuse, not re-run a prompt-only forward
    pass per grid point."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    base_path = cache_dir / f"pooled_prompt_last_token_base_layer{layer_idx}.pt"
    instr_path = cache_dir / f"pooled_prompt_last_token_instr_layer{layer_idx}.pt"
    if base_path.exists() and instr_path.exists():
        return torch.load(base_path), torch.load(instr_path)
    base_pool, instr_pool = pool_prompt_last_token(model, tokenizer, items, layer_idx)
    torch.save(base_pool, base_path)
    torch.save(instr_pool, instr_path)
    return base_pool, instr_pool


@torch.no_grad()
def pool_prompt_last_token_all_layers(model, tokenizer, items: list[dict], n_layers: int) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Same extraction as pool_prompt_last_token, but returns EVERY layer from a single forward
    pass per prompt instead of one layer per call. Returns ({layer: (N,d)}, {layer: (N,d)}) for
    (base, instructed).

    This exists specifically for A-PSR / Multi-Gate, which need a direction at all n_layers layers:
    calling pool_prompt_last_token in a loop would re-run 2 forward passes per item PER LAYER
    (2 * n_items * 28 passes for Qwen2.5-7B), when out.hidden_states from ONE pass already contains
    every layer. Same numbers, ~28x less compute -- worth the separate function rather than a
    caching-layer workaround.
    """
    base_by_layer = {l: [] for l in range(n_layers)}
    instr_by_layer = {l: [] for l in range(n_layers)}
    for item in items:
        base_ids = tokenizer(item["base_prompt"], return_tensors="pt")["input_ids"].to(model.device)
        instr_ids = tokenizer(item["terse_prompt"], return_tensors="pt")["input_ids"].to(model.device)
        out_base = model(input_ids=base_ids, output_hidden_states=True)
        out_instr = model(input_ids=instr_ids, output_hidden_states=True)
        for l in range(n_layers):
            # hidden_states[0] is the embedding output; hidden_states[i+1] is layer i's output --
            # same indexing convention steering/psr/gate.py's subsequent_layers_mse relies on.
            base_by_layer[l].append(out_base.hidden_states[l + 1][0, -1, :].float().cpu())
            instr_by_layer[l].append(out_instr.hidden_states[l + 1][0, -1, :].float().cpu())
    return (
        {l: torch.stack(v, dim=0) for l, v in base_by_layer.items()},
        {l: torch.stack(v, dim=0) for l, v in instr_by_layer.items()},
    )


def load_or_pool_prompt_last_token_all_layers(model, tokenizer, items: list[dict], n_layers: int, cache_dir) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Disk-cached wrapper, same discipline as load_or_pool_prompt_last_token. One file per pole
    holding all layers -- A-PSR always wants every layer at once, so splitting per-layer would just
    mean 56 tiny files and 56 torch.load calls for no benefit."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    base_path = cache_dir / f"pooled_prompt_last_token_base_alllayers{n_layers}.pt"
    instr_path = cache_dir / f"pooled_prompt_last_token_instr_alllayers{n_layers}.pt"
    if base_path.exists() and instr_path.exists():
        return torch.load(base_path), torch.load(instr_path)
    base_by_layer, instr_by_layer = pool_prompt_last_token_all_layers(model, tokenizer, items, n_layers)
    torch.save(base_by_layer, base_path)
    torch.save(instr_by_layer, instr_path)
    return base_by_layer, instr_by_layer


@torch.no_grad()
def accumulate_bipolar_correlation_all_layers(model, tokenizer, items: list[dict], n_layers: int, responses: dict, device: str) -> dict[int, torch.Tensor]:
    """Per-layer bipolar correlation matrices R_l = E[h h^T] over BOTH poles' response-token
    activations, for every layer, in ONE pass over the data. Returns {layer: (d, d)}.

    This is the all-layer counterpart to pool_bipolar_activations -> compute_conceptor, and it
    accumulates R directly rather than returning pooled activations for a reason: the bipolar pool
    is ~10.9k response-token rows, so materializing it for all n_layers layers would be roughly
    8.8GB at d=3584, while the correlation matrices are n_layers * d^2 * 4B ~= 1.4GB and are all
    that compute_conceptor_from_correlation actually needs. Same math, bounded memory.

    R is accumulated on `device` (GPU) in float32: each item contributes h.T @ h summed over its
    response tokens, and the running sums stay resident rather than round-tripping to CPU per item.
    "Bipolar" means both poles land in the SAME R (matching pool_bipolar_activations' concatenation
    and Triantafyllopoulos et al.'s training choice), so the normalizer counts both poles' rows.
    """
    sums = {l: torch.zeros(model.config.hidden_size, model.config.hidden_size, device=device, dtype=torch.float32)
            for l in range(n_layers)}
    total_rows = 0
    for item in items:
        pair = build_training_pair(model, tokenizer, item, responses)
        if pair is None:
            continue
        n_resp = pair["n_resp"]
        out_base = model(input_ids=pair["full_base"], output_hidden_states=True)
        out_instr = model(input_ids=pair["full_instr"], output_hidden_states=True)
        for l in range(n_layers):
            hb = out_base.hidden_states[l + 1][0, -n_resp:, :].float()
            hi = out_instr.hidden_states[l + 1][0, -n_resp:, :].float()
            sums[l] += hb.T @ hb
            sums[l] += hi.T @ hi
        total_rows += 2 * n_resp
    if total_rows == 0:
        raise RuntimeError("no usable training pairs -- every build_training_pair returned None")
    return {l: s / total_rows for l, s in sums.items()}


def pool_bipolar_activations(model, tokenizer, items: list[dict], layer_idx: int, responses: dict) -> torch.Tensor:
    """Both poles concatenated together -- this is the raw material compute_conceptor's correlation
    matrix gets built from. "Bipolar" (both poles pooled together) matches Triantafyllopoulos et
    al.'s training choice. Built on pool_separate_poles rather than duplicating the forward-pass
    logic -- if you also need the separate pools (e.g. for a diff-in-means direction), call that
    directly instead of concatenating this result back apart."""
    base_pool, instr_pool = pool_separate_poles(model, tokenizer, items, layer_idx, responses)
    return torch.cat([base_pool, instr_pool], dim=0)
