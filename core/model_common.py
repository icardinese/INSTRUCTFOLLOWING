"""Model loading, activation extraction, and generation -- 100% task- and method-agnostic. Nothing
here knows about caveman, IFEval, or any specific steering technique. Raw HuggingFace `transformers`
only, no TransformerLens: confirmed via the actual Stolfo/Microsoft repo's own model_utils.py that
their `hf_model=True` bypass (skipping HookedTransformer entirely) is already a first-class supported
path, used in their own format_evaluation.yaml specifically for speed on the no-steering baseline.
The models this needs to support (Phi-3, Gemma2, Mistral, Llama, Qwen2) all follow the same modern
HF convention (XxxForCausalLM -> XxxModel -> .layers) -- see get_decoder_layers() below for what
happens the day a model finally breaks that assumption.
"""
from typing import Callable

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"  # overridable per-call; this default exists
# only so every EXISTING call site (load_model(device)) keeps working unchanged after this refactor.
DEFAULT_LAYER_FRACTIONS = [0.25, 0.5, 0.65, 0.8]
MAX_NEW_TOKENS = 150


def load_model(
    device: str = "cuda",
    model_name: str = DEFAULT_MODEL_NAME,
    dtype: torch.dtype = torch.bfloat16,
    trust_remote_code: bool = True,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """`device` stays the first positional argument (not `model_name`) specifically so every
    existing call site in this codebase (`load_model(device)`, `load_model("cuda")`) keeps working
    with zero changes -- model_name only needs to be passed explicitly by NEW callers targeting a
    non-default model (e.g. IFEval's Phi-3/Gemma2/Mistral roster).

    MODEL_NAME_OVERRIDE env var takes priority over the `model_name` argument if set -- lets you
    switch models for a single run (e.g. `MODEL_NAME_OVERRIDE=google/gemma-2-9b-it python3 ...`)
    without editing any script or creating a near-duplicate adapter per model. The adapter pattern
    is one-per-task, not one-per-model; this is the actual place model choice should be overridden."""
    model_name = os.environ.get("MODEL_NAME_OVERRIDE", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device, trust_remote_code=trust_remote_code
    )
    model.eval()
    return model, tokenizer


def get_decoder_layers(model: PreTrainedModel) -> torch.nn.ModuleList:
    """The list of transformer decoder layers -- for register_forward_hook, len(), or indexing.
    Assumes the now-standard XxxForCausalLM -> XxxModel -> .layers convention that Llama, Mistral,
    Gemma2, Phi-3, and Qwen2 all follow. Raises a clear, named error instead of a bare AttributeError
    three functions away if a genuinely different architecture ever breaks that assumption -- a new
    model family should surface as one explicit line added here, never a guess."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise AttributeError(
        f"{type(model).__name__} doesn't expose the standard model.model.layers path this codebase "
        f"assumes (Llama/Mistral/Gemma2/Phi-3/Qwen2 all do). Add this architecture's real decoder-"
        f"layer attribute path here explicitly."
    )


def num_layers(model: PreTrainedModel) -> int:
    return len(get_decoder_layers(model))


def layer_indices_from_fractions(model: PreTrainedModel, fractions: list[float] = DEFAULT_LAYER_FRACTIONS) -> list[int]:
    n = num_layers(model)
    return sorted({int(frac * n) for frac in fractions})


@torch.no_grad()
def extract_hidden_states(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt_text: str,
    layer_indices: list[int] | None = None,
    num_final_tokens: int = 1,
) -> dict[int, torch.Tensor]:
    """Residual-stream activations for the final `num_final_tokens` prompt tokens, for `layer_indices`
    (every layer if None), from a single forward pass. Merges what used to be two separate functions:
    the original hidden_at_last_token_all_layers (specific layers, always exactly the last token) and
    a TransformerLens-based extract_representation (every layer, last N tokens via run_with_cache).

    Backward-compat note: with the defaults (num_final_tokens=1), each returned tensor has shape (d,)
    -- identical to the original hidden_at_last_token_all_layers's return shape, so every existing
    caller (steering_const.py) needs zero changes. Only requesting num_final_tokens > 1 changes the
    per-layer shape to (num_final_tokens, d)."""
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    out = model(**inputs, output_hidden_states=True)
    n_layers = len(out.hidden_states) - 1  # hidden_states[0] is the embedding output
    indices = layer_indices if layer_indices is not None else list(range(n_layers))

    result = {}
    for layer_idx in indices:
        # hidden_states[i] is the output of layer (i-1)
        h = out.hidden_states[layer_idx + 1][0, -num_final_tokens:, :].float().cpu()
        result[layer_idx] = h[0] if num_final_tokens == 1 else h
    return result


@torch.no_grad()
def generate_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt_text: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    extra_stop_token_ids: list[int] | None = None,
) -> str:
    """extra_stop_token_ids covers models whose chat-template turn-end token isn't set as the
    tokenizer's default eos_token_id (Phi-3's is a documented example of this) -- HF's own
    model.generate() already accepts a list for eos_token_id, no TransformerLens needed for it."""
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    stop_ids = [tokenizer.eos_token_id] + (extra_stop_token_ids or [])
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=stop_ids,
    )
    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


@torch.no_grad()
def generate_response_with_meta(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt_text: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    extra_stop_token_ids: list[int] | None = None,
) -> dict:
    """generate_response, plus whether the generation STOPPED ON ITS OWN rather than hitting the
    max_new_tokens cap.

    Needed because steering/psr/data.py appends the assistant turn-end token to the teacher-forced
    response (the reference builds its training sequence via apply_chat_template with the
    assistant message included, which emits that token; decoding with skip_special_tokens=True
    strips it). Appending it to a response that was TRUNCATED at the cap would teach the gate that
    an arbitrary 150-token cutoff is a valid place to stop -- which, on a task whose entire
    dependent variable is output length, is exactly the wrong lesson. So the flag is recorded here
    and honored there.

    Returns {"text": str, "finished": bool}."""
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    stop_ids = [tokenizer.eos_token_id] + (extra_stop_token_ids or [])
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=stop_ids,
    )
    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    # Terminated naturally iff a stop token was actually emitted. Checking for the token is more
    # reliable than comparing len(new_tokens) to max_new_tokens, which would misreport a response
    # that happens to stop at exactly the cap.
    finished = bool(len(new_tokens) > 0 and new_tokens[-1].item() in set(stop_ids))
    return {
        "text": tokenizer.decode(new_tokens, skip_special_tokens=True).strip(),
        "finished": finished,
    }


def token_count(tokenizer: PreTrainedTokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])
