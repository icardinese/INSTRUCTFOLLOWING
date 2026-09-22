"""Intervention SURFACE as a first-class, switchable config rather than a hardcoded mask.

Nokia's reference exposes this as `steering_location` on SteeredModelConfig and uses both values:
"answer_only" for IFEval and AxBench, "question_and_answer" for the persona-vectors eval. This
module reproduces both, plus the tokenizer bookkeeping needed to locate the system/user boundary.

WHY THIS IS A FLAG AND NOT A CONSTANT: finding F2 established that the intervention surface
interacts with the functional form -- moving Stolfo's clamp from all-positions to response-only
cost 47 tokens, while the same move cost the additive form nothing. The generalization is that an
intervention can only substitute for prompt content if it reaches the positions where that
content had its effect. Which surface is correct is therefore a property OF THE TASK, not of the
codebase:

  - caveman:  the instruction shapes generation STYLE      -> acts on response tokens -> R
  - ifeval:   the instruction shapes output FORMAT          -> acts on response tokens -> R
  - triage:   the instruction shapes how the input is READ  -> acts on prompt tokens   -> QR

A task whose removed instruction governs interpretation of the input cannot be served by R at all:
the input was already encoded, unsteered, before the first steered position. Adapters declare
their own default via STEERING_LOCATION and every call site takes an override.

POSITION CONVENTIONS, both inherited from the reference (see constant_steering.py::
compute_steering_mask and tokenization_utils.compute_last_input_token_index):
  - "answer_only"         := pos >= last_input_token_index  (the FINAL PROMPT token, inclusive)
  - "question_and_answer" := pos >= last_system_token_index (the FINAL SYSTEM token, inclusive)
Both are `>=` against the last token of the preceding span, so both include one boundary token.
"""
import torch

ANSWER_ONLY = "answer_only"
QUESTION_AND_ANSWER = "question_and_answer"
STEERING_LOCATIONS = (ANSWER_ONLY, QUESTION_AND_ANSWER)

# Paper-facing shorthand. Used in prose and figure labels only, never as a config value.
LOCATION_ALIASES = {ANSWER_ONLY: "R", QUESTION_AND_ANSWER: "QR"}


def validate_location(location: str) -> str:
    if location not in STEERING_LOCATIONS:
        raise ValueError(
            f"unknown steering_location {location!r}; expected one of {list(STEERING_LOCATIONS)}. "
            f"Paper shorthand R/QR maps to {LOCATION_ALIASES}."
        )
    return location


def last_system_token_index(tokenizer, system: str, user: str, min_prefix_ratio: float = 0.9) -> int:
    """Index of the final token of the system turn within the full chat-templated prompt.

    Found by taking the longest common token PREFIX of (system turn alone) and (system + user),
    rather than assuming the system turn tokenizes to an exact prefix of the full prompt. The
    exact-prefix assumption is nearly true but not reliably so: a BPE merge can produce a single
    token spanning the end of the system turn and the start of the next turn's header, which
    shortens the common prefix by one. Requiring exact equality would then raise on a perfectly
    healthy tokenizer.

    When the boundary token is merged, this returns an index one position EARLIER, i.e. the mask
    starts one token sooner. That direction is deliberate: QR is `>= last_system_token_index` and
    so includes the boundary token anyway, and over-steering by one position is harmless whereas
    under-steering would leave a prompt token the intervention was supposed to reach.

    Raises only when the common prefix is far shorter than the system turn, which means the
    template genuinely reorders or interleaves turns and no prefix-based boundary exists. Better
    to fail loudly than to silently misplace the QR mask by an unknown amount -- an error that
    would be effectively invisible in results.
    """
    sys_only = tokenizer.apply_chat_template(
        [{"role": "system", "content": system}], tokenize=False, add_generation_prompt=False
    )
    full = tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )
    sys_ids = tokenizer(sys_only, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]

    lcp = 0
    for a, b in zip(sys_ids, full_ids):
        if a != b:
            break
        lcp += 1

    if not sys_ids or lcp < min_prefix_ratio * len(sys_ids):
        raise ValueError(
            f"the system turn shares only {lcp} of {len(sys_ids)} leading tokens with the full "
            f"prompt, so this chat template does not place the system turn at the front and the "
            f"system/user boundary cannot be located by prefix matching. QR steering needs that "
            f"boundary; use answer_only for this model, or add a template-specific finder."
        )
    return lcp - 1


def steering_mask(
    seq_len: int,
    n_resp: int,
    device,
    location: str = ANSWER_ONLY,
    last_sys_idx: int | None = None,
) -> torch.Tensor:
    """(1, seq_len, 1) bool mask over a single unpadded train sequence (prompt + response).

    answer_only         -> last n_resp + 1 positions (final prompt token + response)
    question_and_answer -> everything from the final system token onward
    """
    validate_location(location)
    positions = torch.arange(seq_len, device=device)
    if location == ANSWER_ONLY:
        start = seq_len - n_resp - 1
    else:
        if last_sys_idx is None:
            raise ValueError(
                "question_and_answer steering needs last_sys_idx (the index of the final system "
                "token). Adapters that support QR must emit it; see adapters/triage_adapter.py."
            )
        start = last_sys_idx
    return (positions >= start).view(1, seq_len, 1)


def prefill_tail_length(
    prompt_len: int, location: str = ANSWER_ONLY, last_sys_idx: int | None = None
) -> int:
    """How many trailing PREFILL positions the inference hook should steer.

    Expressed as a count FROM THE RIGHT on purpose. Generation batches are left-padded
    (steering/batch_routing.py::left_pad_batch), which right-aligns every row, so a
    count-from-the-right is identical across rows regardless of how much padding each one
    carries. An absolute index would need per-row padding offsets and would silently steer pad
    tokens the moment batch composition changed.

    answer_only         -> 1 (the final prompt token)
    question_and_answer -> prompt_len - last_sys_idx
    """
    validate_location(location)
    if location == ANSWER_ONLY:
        return 1
    if last_sys_idx is None:
        raise ValueError("question_and_answer steering needs last_sys_idx")
    return max(1, prompt_len - last_sys_idx)
