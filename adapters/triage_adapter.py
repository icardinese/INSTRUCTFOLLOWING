"""Triage task: 3-way email classification (no / email / notify), adapted from
langchain-ai/executive-ai-assistant (eaia/main/triage.py).

WHAT THE INSTRUCTION IS HERE. For caveman the instruction is an appended suffix; for IFEval it is
a natural-language clause already embedded in the prompt. For triage the instruction is the set of
TEMPLATE SLOTS that tell the model how to classify -- {background}, {triage_no}, {triage_email},
{triage_notify}, {fewshotexamples}. Strip them and the scaffold still asks for a no/email/notify
label, but nothing defines what those labels mean for this person. So the base/instructed contrast
is slot-stripped vs. slot-filled, exactly parallel to caveman's base vs. base+suffix.

THIS IS NOT A LENGTH TASK. Caveman's instruction happens to be about brevity, so there its
compliance metric is output length. Here the instruction governs a classification decision, and
compliance is whether the stripped prompt still classifies the way the filled prompt did. The
~1.4k tokens of slot content is the INDEPENDENT variable (how much instruction steering must
carry), not a savings target.

NO AGENT MACHINERY. Upstream triage.py runs inside LangGraph with a store-backed few-shot
retriever and OpenAI structured output (`with_structured_output(RespondTo)` plus a forced
tool_choice). None of that is the task. What remains after removing it is one string formatted
from slots, one greedy completion, and one label parsed out -- pure prompt in, text out, which is
what activation steering needs and what makes this an instruction-following experiment rather than
an agent experiment. The two upstream pieces that had to be replaced:
  - few-shot retrieval -> a STATIC block from triage_config.yaml. Better here anyway: a
    retrieval-dependent slot would vary per item and confound the base/instructed contrast.
  - structured output -> the model is asked to reason and then emit a final `Triage: <label>`
    line, parsed by evals/triage/judge.py. Keeping the reasoning is deliberate; see N_RESP below.

WHY THE REASONING FIELD IS KEPT. Upstream's RespondTo schema is `logic` then `response`, so this
is faithful. It also matters mechanically: a bare label makes the response span 1 token, so the R
mask covers 2 positions and the MSE objective matches hidden states over 2 positions instead of
caveman's ~59 -- roughly 20x less training signal per item. Reasoning-then-label keeps the span in
the same order of magnitude as caveman.

STEERING SURFACE IS QR, NOT R. The slots govern how the EMAIL IS INTERPRETED, so their effect
lands on prompt tokens. An intervention confined to the response span cannot substitute for them:
by the time the first steered position is reached, the email has already been encoded unsteered.
This is finding F2 generalized -- the surface has to reach where the removed instruction acted.
See steering/psr/spans.py. Declared here as STEERING_LOCATION and overridable per run, so the
R-vs-QR comparison is a flag rather than a fork.

THE SYSTEM/USER SPLIT IS LOAD-BEARING. Scaffold and slots go in the system turn; the email goes in
the user turn. Two consequences, both necessary:
  1. Nokia's QR mask is `pos >= last_system_token_position`, so this split makes QR select exactly
     {email, response} with no custom masking.
  2. All removed slot content lives inside the system span, so the base and instructed sequences
     are IDENTICAL IN LENGTH from the email onward. The MSE objective can therefore align from the
     end with no segment-alignment machinery -- which is the only reason this is a small change
     rather than a rewrite of the loss path.
"""
import json
from pathlib import Path

import yaml
from transformers import PreTrainedTokenizer

from steering.psr.spans import QUESTION_AND_ANSWER, last_system_token_index

DATA_DIR = Path("data/triage")
RESULTS_DIR = Path("results/triage")
CACHE_DIR = Path("cache/triage")
# Held constant with caveman on purpose. A code model doing email triage is out of distribution,
# and that is a real limitation to footnote -- but changing models between tasks would make the
# gate-ladder and surface findings incomparable across tasks, which costs more than it buys.
# Override per-run with MODEL_NAME_OVERRIDE (see core/model_common.load_model).
MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"

STEERING_LOCATION = QUESTION_AND_ANSWER
LABELS = ("no", "email", "notify")

CONFIG_PATH = Path(__file__).parent / "triage_config.yaml"

# Verbatim from eaia/main/triage.py::triage_prompt, restructured only by splitting at the point
# where the persona/rules end and the specific email begins. Slot names are unchanged.
SYSTEM_TEMPLATE = """You are {full_name}'s executive assistant. You are a top-notch executive assistant who cares about {name} performing as well as possible.

{background}

{name} gets lots of emails. Your job is to categorize the below email to see whether is it worth responding to.

Emails that are not worth responding to:
{triage_no}

Emails that are worth responding to:
{triage_email}

There are also other things that {name} should know about, but don't require an email response. For these, you should notify {name} (using the `notify` response). Examples include:
{triage_notify}

For emails not worth responding to, respond `no`. For something where {name} should respond over email, respond `email`. If it's important to notify {name}, but no email is required, respond `notify`.

If unsure, opt to `notify` {name} - you will learn from this in the future.

{fewshotexamples}

First give brief reasoning, then end your reply with a final line of exactly this form:
Triage: <no|email|notify>"""

# Slot-free scaffold. Keeps the task definition (the three label names, the output format) so the
# stripped prompt is still a well-posed request, and removes only the content that DEFINES the
# labels. A stripped prompt that didn't name the labels would be testing whether steering can
# invent an output format, which is a different and less interesting question.
SYSTEM_TEMPLATE_STRIPPED = """You are {full_name}'s executive assistant. You are a top-notch executive assistant who cares about {name} performing as well as possible.

{name} gets lots of emails. Your job is to categorize the below email to see whether is it worth responding to.

For emails not worth responding to, respond `no`. For something where {name} should respond over email, respond `email`. If it's important to notify {name}, but no email is required, respond `notify`.

First give brief reasoning, then end your reply with a final line of exactly this form:
Triage: <no|email|notify>"""

USER_TEMPLATE = """Please determine how to handle the below email thread:

From: {author}
To: {to}
Subject: {subject}

{email_thread}"""

# Upstream's few-shot rendering, verbatim from eaia/main/fewshot.py.
_FEWSHOT_ITEM = """Email Subject: {subject}
Email From: {from_email}
Email To: {to_email}
Email Content: 
```
{content}
```
> Triage Result: {result}"""

# Which slots each ablation level removes. Enables the sweep the paper actually wants -- accuracy
# retained as a function of how much instruction was taken out -- instead of a single data point.
# "none" is the fully instructed prompt; "all" is the fully stripped one.
SLOT_ABLATIONS = {
    "none": (),
    "fewshot": ("fewshotexamples",),
    "rules": ("triage_no", "triage_email", "triage_notify"),
    "rules_fewshot": ("triage_no", "triage_email", "triage_notify", "fewshotexamples"),
    "all": ("background", "triage_no", "triage_email", "triage_notify", "fewshotexamples"),
}
DEFAULT_ABLATION = "all"


def load_config() -> dict:
    with CONFIG_PATH.open() as f:
        return yaml.safe_load(f)


def render_fewshot(cfg: dict) -> str:
    items = [
        _FEWSHOT_ITEM.format(
            subject=eg["subject"],
            from_email=eg["from_email"],
            to_email=eg["to_email"],
            content=eg["content"][:400],
            result=eg["result"],
        )
        for eg in cfg.get("fewshot_examples", [])
    ]
    if not items:
        return ""
    return "\n\n------------\n\n".join(["Here are some previous examples:"] + items)


def build_system(cfg: dict, removed: tuple[str, ...]) -> str:
    """The instructed system turn with `removed` slots blanked, or the stripped scaffold when
    every slot is removed. Blanking rather than deleting the surrounding prose keeps partial
    ablations grammatical -- an empty rules section still reads as a section with nothing in it,
    which is the honest rendering of "this instruction was withheld"."""
    if set(removed) >= set(SLOT_ABLATIONS["all"]):
        return SYSTEM_TEMPLATE_STRIPPED.format(full_name=cfg["full_name"], name=cfg["name"])
    values = {
        "full_name": cfg["full_name"],
        "name": cfg["name"],
        "background": cfg["background"],
        "triage_no": cfg["triage_no"],
        "triage_email": cfg["triage_email"],
        "triage_notify": cfg["triage_notify"],
        "fewshotexamples": render_fewshot(cfg),
    }
    for slot in removed:
        values[slot] = ""
    return SYSTEM_TEMPLATE.format(**values)


def build_user(row: dict) -> str:
    return USER_TEMPLATE.format(
        author=row.get("author", "unknown@enron.com"),
        to=row.get("to", "unknown@enron.com"),
        subject=row.get("subject", ""),
        email_thread=row["email_body"],
    )


def build_prompt(tokenizer: PreTrainedTokenizer, row: dict, instructed: bool,
                 ablation: str = DEFAULT_ABLATION, cfg: dict | None = None) -> str:
    cfg = cfg or load_config()
    removed = () if instructed else SLOT_ABLATIONS[ablation]
    messages = [
        {"role": "system", "content": build_system(cfg, removed)},
        {"role": "user", "content": build_user(row)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_rows(split: str) -> list[dict]:
    """Each row: {"id", "subject", "email_body", "author", "to", "gold"}. `gold` is written by
    scripts/build_triage_data.py as the fully-instructed model's own label -- see that script for
    why self-labelling is the right ground truth for a substitution experiment."""
    rows = []
    with (DATA_DIR / f"{split}.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def to_items(tokenizer: PreTrainedTokenizer, rows: list[dict],
             ablation: str = DEFAULT_ABLATION) -> list[dict]:
    """base_prompt = slot-stripped (uninstructed), terse_prompt = slot-filled (instructed).

    "terse_prompt" is a caveman-era name for "the instructed side" and is kept because it is the
    key the whole training path already reads (steering/psr/data.py, evals/layer_hparam_search.py).
    Nothing here is terse; renaming it is a separate mechanical change across every adapter.

    Also emits last_sys_idx per item, which QR steering needs to locate the system/user boundary.
    Both prompts carry their own index because the instructed system turn is ~1.4k tokens longer.
    """
    cfg = load_config()
    items = []
    for row in rows:
        user = build_user(row)
        base_sys = build_system(cfg, SLOT_ABLATIONS[ablation])
        instr_sys = build_system(cfg, ())
        items.append({
            "id": row["id"],
            "base_prompt": build_prompt(tokenizer, row, instructed=False, ablation=ablation, cfg=cfg),
            "terse_prompt": build_prompt(tokenizer, row, instructed=True, ablation=ablation, cfg=cfg),
            "base_last_sys_idx": last_system_token_index(tokenizer, base_sys, user),
            "terse_last_sys_idx": last_system_token_index(tokenizer, instr_sys, user),
            "gold": row.get("gold"),
        })
    return items
