"""Builds data/triage/{train,dev,test}.jsonl from the Annotated Enron Subject Line Corpus.

  git clone https://github.com/ryanzhumich/AESLC.git third_party/AESLC
  python scripts/build_triage_data.py --aeslc third_party/AESLC/enron_subject_line
  python scripts/build_triage_data.py --label          # second pass, needs the GPU

WHY AESLC. Discussed at length in the handoff; the short version is that it is the same
population as raw Enron (senior-management mailboxes, 135 owners in the inbox subset) but already
cleaned and already split train/dev/test, which maps one-to-one onto the adapter's
load_rows(split) contract. Raw Enron would mean parsing ~500k RFC-822 maildir files, stripping
quoted replies and signatures, reconstructing threads and deduping -- days of off-thesis work.

THE COST OF AESLC, stated plainly: it drops the From/To headers, keeping only body and subject.
Sender identity is therefore unavailable. Two consequences, both handled:
  - adapters/triage_config.yaml's rules are written to key off SUBJECT AND BODY CONTENT only, so
    no rule is unevaluable.
  - {author} and {to} are filled from the filename, which encodes the mailbox owner and folder
    (allen-p_inbox_20 -> owner allen-p, folder inbox). The owner is the RECIPIENT. The sender is
    genuinely unknown and is stubbed. Because the stub is IDENTICAL in the base and instructed
    prompts, it contributes nothing to the contrast and cannot bias the substitution measurement
    -- it only limits realism, not validity.

INBOX ONLY. 6,374 of 14,436 train files come from `sent` folders: mail the owner WROTE. Triage is
a decision about received mail, so sent items are dropped. Remaining: 8,062 / 1,183 / 1,163.

GOLD LABELS ARE THE FULLY-INSTRUCTED MODEL'S OWN OUTPUT, not human annotation. This is the right
ground truth here and not a shortcut: the question is whether steering can carry what the prompt
slots carried, so the reference behaviour IS the filled-prompt behaviour. Human labels would
measure something else (whether the persona rules are good), and no public corpus carries
no/email/notify labels for an Enron executive anyway. Consequence to keep in view: accuracy is
measured against the model's own filled-prompt decisions, so the ceiling is self-consistency, and
items the filled prompt cannot label are dropped rather than guessed.
"""
import argparse
import json
import random
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

sys.path.insert(0, str(REPO_ROOT))

from adapters.triage_adapter import DATA_DIR, MODEL_NAME, build_prompt, load_config  # noqa: E402
from evals.triage.judge import parse_label  # noqa: E402

SPLITS = ("train", "dev", "test")
# Matches caveman's n=180 per split so cross-task comparisons aren't confounded by sample size.
DEFAULT_PER_SPLIT = {"train": 180, "dev": 60, "test": 180}
_FNAME = re.compile(r"^(?P<owner>[a-z0-9\-]+)_(?P<folder>[a-z0-9]+)_(?P<idx>\d+)\.subject$")


def parse_aeslc_file(path: Path) -> dict | None:
    """AESLC files are the body, then '@subject', then one or more annotated subject lines."""
    text = path.read_text(encoding="utf-8", errors="replace")
    if "@subject" not in text:
        return None
    body, _, rest = text.partition("@subject")
    body = body.strip()
    subject = next((ln.strip() for ln in rest.strip().splitlines() if ln.strip()), "")
    if not body or not subject:
        return None
    m = _FNAME.match(path.name)
    if not m or m.group("folder") != "inbox":
        return None
    owner = m.group("owner")
    return {
        "id": path.stem,
        "subject": subject,
        "email_body": body,
        # Genuinely unknown -- see module docstring. Constant across base and instructed prompts.
        "author": "unknown.sender@external.com",
        "to": f"{owner}@enron.com",
    }


def build_raw(aeslc_dir: Path, per_split: dict, seed: int, max_body_chars: int) -> dict:
    out = {}
    rng = random.Random(seed)
    for split in SPLITS:
        files = sorted((aeslc_dir / split).glob("*.subject"))
        rows = [r for r in (parse_aeslc_file(f) for f in files) if r is not None]
        # Very long bodies blow up the instructed prompt (already ~1.6k tokens of slots) and the
        # backward-pass activation memory with it. p90 is ~1,415 chars, so this trims a thin tail.
        rows = [r for r in rows if len(r["email_body"]) <= max_body_chars]
        rng.shuffle(rows)
        out[split] = rows[: per_split[split]]
        print(f"{split}: {len(files)} files -> {len(rows)} usable inbox -> {len(out[split])} kept")
    return out


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def label_split(model, tokenizer, rows: list[dict], cfg: dict, max_new_tokens: int) -> list[dict]:
    """Run the FULLY INSTRUCTED prompt and keep its label as gold. Unparseable -> dropped."""
    from core.model_common import generate_response

    labelled, dropped = [], 0
    for row in rows:
        prompt = build_prompt(tokenizer, row, instructed=True, cfg=cfg)
        response = generate_response(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
        label = parse_label(response)
        if label is None:
            dropped += 1
            continue
        labelled.append({**row, "gold": label, "gold_response": response})
    print(f"  labelled {len(labelled)}, dropped {dropped} unparseable")
    return labelled


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aeslc", type=Path, help="path to AESLC/enron_subject_line")
    ap.add_argument("--label", action="store_true", help="second pass: generate gold labels")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-body-chars", type=int, default=2000)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    if args.aeslc:
        raw = build_raw(args.aeslc, DEFAULT_PER_SPLIT, args.seed, args.max_body_chars)
        for split, rows in raw.items():
            write_jsonl(rows, DATA_DIR / f"{split}.raw.jsonl")
        print(f"\nwrote *.raw.jsonl to {DATA_DIR}. Now rerun with --label to add gold labels.")
        return

    if args.label:
        from core.model_common import load_model

        cfg = load_config()
        model, tokenizer = load_model("cuda", model_name=MODEL_NAME)
        for split in SPLITS:
            raw_path = DATA_DIR / f"{split}.raw.jsonl"
            if not raw_path.exists():
                raise FileNotFoundError(f"{raw_path} missing; run with --aeslc first")
            rows = [json.loads(l) for l in raw_path.open() if l.strip()]
            print(f"{split}: labelling {len(rows)}")
            write_jsonl(label_split(model, tokenizer, rows, cfg, args.max_new_tokens),
                        DATA_DIR / f"{split}.jsonl")
        print(f"\nwrote {DATA_DIR}/{{train,dev,test}}.jsonl with gold labels.")
        return

    ap.error("pass --aeslc to build raw rows, or --label to add gold labels")


if __name__ == "__main__":
    main()
