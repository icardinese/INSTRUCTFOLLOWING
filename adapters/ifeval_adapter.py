"""IFEval task: instruction-following compliance. Unlike caveman (where CAVEMAN_SUFFIX is actively
appended to build an instructed prompt from an uninstructed one), IFEval's instruction is already
embedded in "prompt" as natural language -- "prompt_without_instruction" is a pre-stripped base
version, both already present in the data (an augmentation on top of Google's raw IFEval format,
not derived here). See third_party/microsoft_llm_steer_instruct/ifeval_scripts/evaluation_main.py
for the exact downstream consumer of instruction_id_list/kwargs (in evals/ifeval/judge.py, not here
-- this file only builds generation-ready prompts, judging needs the raw row untouched).
"""
import json
from pathlib import Path

from transformers import PreTrainedTokenizer

DATA_DIR = Path("data/ifeval")
RESULTS_DIR = Path("results/ifeval")
CACHE_DIR = Path("cache/ifeval")
# IFEval work spans several models (Phi-3, Gemma2, Mistral in this project's prior work) -- this
# is just the default; override per-run with MODEL_NAME_OVERRIDE (see core/model_common.load_model)
# rather than creating a near-duplicate adapter per model.
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"


def _wrap_chat_template(tokenizer: PreTrainedTokenizer, text: str) -> str:
    messages = [{"role": "user", "content": text}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_rows(split: str) -> list[dict]:
    """Each row: {"id" or "key", "prompt", "prompt_without_instruction", "instruction_id_list",
    "kwargs"} -- kwargs stays whatever json.loads gives back (list of dicts with str/int/None
    values), unmodified, since evals/ifeval/judge.py passes it straight into Google's InputExample
    without any translation layer."""
    rows = []
    with (DATA_DIR / f"{split}.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def to_items(tokenizer: PreTrainedTokenizer, rows: list[dict]) -> list[dict]:
    return [
        {
            "id": row.get("id", row.get("key")),
            "base_prompt": _wrap_chat_template(tokenizer, row["prompt_without_instruction"]),
            "terse_prompt": _wrap_chat_template(tokenizer, row["prompt"]),
        }
        for row in rows
    ]
