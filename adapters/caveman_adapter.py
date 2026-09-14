"""Caveman task: explain a Python function, terse-caveman-style instruction on the steered side.
The only file in this codebase that knows about "code" fields or the caveman prompt template.
"""
import json
from pathlib import Path

from transformers import PreTrainedTokenizer

DATA_DIR = Path("data/caveman")
RESULTS_DIR = Path("results/caveman")
CACHE_DIR = Path("cache/caveman")
MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"

BASE_INSTRUCTION = "Explain what the following Python function does.\n\n```python\n{code}\n```"
# Verbatim from caveman's "full" mode (skills/caveman/SKILL.md), word for word, except Auto-Clarity
# (destructive-op handling), Boundaries (about commits/PRs), and the language-preservation paragraph —
# none of those scenarios exist in a single-shot, English-only code-explanation call.
CAVEMAN_SUFFIX = (
    "\n\nRespond terse like smart caveman. All technical substance stay. Only fluff die.\n\n"
    'ACTIVE EVERY RESPONSE. No revert after many turns. No filler drift. Still active if unsure. '
    'Off only: "stop caveman" / "normal mode".\n\n'
    "Drop: articles (a/an/the), filler (just/really/basically/actually/simply), pleasantries "
    "(sure/certainly/of course/happy to), hedging. Fragments OK. Short synonyms (big not extensive, "
    'fix not "implement a solution for"). No tool-call narration, no decorative tables/emoji, no '
    "dumping long raw error logs unless asked — quote shortest decisive line. Standard well-known "
    "tech acronyms OK (DB/API/HTTP); never invent new abbreviations (cfg/impl/req/res/fn) — tokenizer "
    "split them same as full word: zero token saved, reader still decode. Full word cheaper AND "
    "clearer. No causal arrows (→) either — own token, save nothing. Technical terms exact. Code "
    "blocks unchanged. Errors quoted exact.\n\n"
    'No self-reference. Never name or announce the style. No "caveman mode on", "me caveman think", '
    'no third-person caveman tags. Output caveman-only — never normal answer plus "Caveman:" recap.\n\n'
    "Pattern: `[thing] [action] [reason]. [next step].`"
)


def build_prompt(tokenizer: PreTrainedTokenizer, code: str, terse: bool) -> str:
    user_content = BASE_INSTRUCTION.format(code=code)
    if terse:
        user_content += CAVEMAN_SUFFIX
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_rows(split: str) -> list[dict]:
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
            "id": row["id"],
            "base_prompt": build_prompt(tokenizer, row["code"], terse=False),
            "terse_prompt": build_prompt(tokenizer, row["code"], terse=True),
        }
        for row in rows
    ]
