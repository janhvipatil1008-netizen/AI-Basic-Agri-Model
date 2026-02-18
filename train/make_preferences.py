"""
Create preference-pair data for alignment training.

Inputs:
- data/train_data.jsonl (required): instruction, context, answer
- data/baseline_outputs.jsonl (optional): extra questions

Output:
- data/preferences.jsonl with keys: prompt, chosen, rejected

Design notes:
- prompt is always built from SYSTEM + CONTEXT + USER + "ASSISTANT:\\n"
- chosen comes from train_data answer (ideal reference)
- rejected is created deterministically by corrupting chosen
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


SYSTEM_PROMPT = (
    "You are AgriSathi, a beginner-safe agriculture assistant.\n"
    "Use headings in order:\n"
    "Problem summary:\n"
    "Action steps or Follow-up questions:\n"
    "Safety disclaimer:\n"
    "When to consult expert:\n"
    "Never provide exact chemical dosage or mixing instructions."
)

HEADING_LINES = [
    "Problem summary:",
    "Action steps:",
    "Follow-up questions:",
    "Safety disclaimer:",
    "When to consult expert:",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build preference pairs from AgriSathi data")
    parser.add_argument("--train_file", type=str, default="data/train_data.jsonl")
    parser.add_argument("--baseline_file", type=str, default="", help="Optional baseline JSONL.")
    parser.add_argument("--output_file", type=str, default="data/preferences.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on output rows.")
    return parser.parse_args()


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_prompt(instruction: str, context: Dict) -> str:
    context_json = json.dumps(context if isinstance(context, dict) else {}, ensure_ascii=False)
    return (
        f"SYSTEM:{SYSTEM_PROMPT}\n"
        f"CONTEXT:{context_json}\n"
        f"USER:{instruction.strip()}\n"
        "ASSISTANT:\n"
    )


def remove_headings(chosen: str) -> str:
    """Corruption 1: strip required headings to create poorly formatted output."""
    out_lines = []
    for line in chosen.splitlines():
        if any(line.strip().startswith(h) for h in HEADING_LINES):
            continue
        out_lines.append(line)
    text = "\n".join(out_lines).strip()
    if not text:
        text = "General advice: do good farming practices and monitor regularly."
    return text


def reorder_sections(chosen: str) -> str:
    """Corruption 2: reorder sections so structure quality degrades."""
    sections: Dict[str, List[str]] = {h: [] for h in HEADING_LINES}
    current = None
    for raw in chosen.splitlines():
        line = raw.rstrip()
        hit = None
        for h in HEADING_LINES:
            if line.strip().startswith(h):
                hit = h
                break
        if hit is not None:
            current = hit
            continue
        if current is not None:
            sections[current].append(line)

    # Deterministic wrong order.
    wrong_order = [
        "Safety disclaimer:",
        "Problem summary:",
        "When to consult expert:",
        "Action steps:",
        "Follow-up questions:",
    ]
    parts = []
    for h in wrong_order:
        if sections[h]:
            parts.append(h)
            parts.extend(sections[h])
    if not parts:
        return "Safety disclaimer:\nDo good farming practices."
    return "\n".join(parts).strip()


def vague_or_unsafe(chosen: str) -> str:
    """Corruption 3: make answer vague and inject unsafe dosage pattern."""
    return (
        "Problem summary:\n"
        "The issue is noted but details are general.\n\n"
        "Action steps:\n"
        "1) Do good farming practices.\n"
        "2) Keep field clean.\n"
        "3) Spray 2 ml per litre for quick control.\n\n"
        "Safety disclaimer:\n"
        "Basic caution only.\n\n"
        "When to consult expert:\n"
        "If still unsure."
    )


def corrupt_answer(chosen: str, index: int) -> Tuple[str, str]:
    """
    Deterministic corruption selector.
    index % 3 decides which corruption is used.
    """
    mode = index % 3
    if mode == 0:
        return remove_headings(chosen), "removed_headings"
    if mode == 1:
        return reorder_sections(chosen), "reordered_sections"
    return vague_or_unsafe(chosen), "vague_or_unsafe"


def stable_pick_answer(answers: List[str], instruction: str) -> str:
    """
    Deterministic fallback picker for optional baseline-only instructions.
    Always picks from train_data answers (required by spec).
    """
    score = sum(ord(c) for c in instruction)
    return answers[score % len(answers)]


def main() -> None:
    args = parse_args()
    train_path = Path(args.train_file)
    baseline_path = Path(args.baseline_file) if args.baseline_file else None
    output_path = Path(args.output_file)

    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")

    train_rows = read_jsonl(train_path)
    baseline_rows = []
    if baseline_path and baseline_path.exists():
        baseline_rows = read_jsonl(baseline_path)

    # Filter valid train examples first.
    valid_train = []
    for row in train_rows:
        instruction = str(row.get("instruction", "")).strip()
        answer = str(row.get("answer", "")).strip()
        context = row.get("context", {})
        if instruction and answer:
            valid_train.append(
                {
                    "instruction": instruction,
                    "context": context if isinstance(context, dict) else {},
                    "answer": answer,
                }
            )

    if not valid_train:
        raise RuntimeError("No valid train examples found.")

    all_train_answers = [x["answer"] for x in valid_train]
    seen_prompts = set()
    pref_rows = []
    transform_counts = {"removed_headings": 0, "reordered_sections": 0, "vague_or_unsafe": 0}

    # 1) Primary pairs from train_data (chosen from train answer directly).
    for i, item in enumerate(valid_train):
        prompt = build_prompt(item["instruction"], item["context"])
        if prompt in seen_prompts:
            continue
        seen_prompts.add(prompt)

        chosen = item["answer"]
        rejected, mode = corrupt_answer(chosen, i)
        transform_counts[mode] += 1
        pref_rows.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

    # 2) Optional extra prompts from baseline questions.
    # Chosen is still sourced from train_data answers (deterministic mapping).
    for j, row in enumerate(baseline_rows):
        question = str(row.get("question", "")).strip()
        if not question:
            continue
        prompt = build_prompt(question, {})
        if prompt in seen_prompts:
            continue
        seen_prompts.add(prompt)

        chosen = stable_pick_answer(all_train_answers, question)
        rejected, mode = corrupt_answer(chosen, len(pref_rows) + j)
        transform_counts[mode] += 1
        pref_rows.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

    if args.limit is not None and args.limit >= 0:
        pref_rows = pref_rows[: args.limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in pref_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("Preference data summary")
    print(f"train rows read: {len(train_rows)}")
    print(f"valid train rows: {len(valid_train)}")
    print(f"baseline rows read: {len(baseline_rows)}")
    print(f"written preference rows: {len(pref_rows)}")
    print(f"transform removed_headings: {transform_counts['removed_headings']}")
    print(f"transform reordered_sections: {transform_counts['reordered_sections']}")
    print(f"transform vague_or_unsafe: {transform_counts['vague_or_unsafe']}")
    print(f"output file: {output_path}")


if __name__ == "__main__":
    main()
