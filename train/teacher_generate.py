"""
Generate teacher answers for AgriSathi training data.

Reads:
- data/train_data.jsonl (expects keys: instruction, context)

Writes:
- data/teacher_answers.jsonl with keys:
  instruction, context, teacher_answer, teacher_model, timestamp

Example (PowerShell):
python train/teacher_generate.py
python train/teacher_generate.py --teacher_model google/flan-t5-small --task text2text-generation --limit 100
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import torch
from transformers import pipeline


# Prompt rules for consistent teacher outputs.
SYSTEM_PROMPT = (
    "You are AgriSathi Teacher, producing high-quality safe agricultural answers.\n"
    "Follow this strictly:\n"
    "- Use plain text only.\n"
    "- Use these headings in this order.\n"
    "- If key details are missing, use Follow-up questions instead of Action steps.\n"
    "- Never provide exact chemical dosage or mixing instructions.\n"
    "- Always mention label-check and consulting local agronomist.\n\n"
    "Required format:\n"
    "Problem summary:\n"
    "Action steps: OR Follow-up questions:\n"
    "Safety disclaimer:\n"
    "When to consult expert:"
)

REQUIRED_HEADINGS = [
    "Problem summary:",
    "Safety disclaimer:",
    "When to consult expert:",
]


def parse_args() -> argparse.Namespace:
    """Parse CLI options."""
    parser = argparse.ArgumentParser(description="Generate teacher answers from train_data.jsonl")
    parser.add_argument("--input_file", type=str, default="data/train_data.jsonl")
    parser.add_argument("--output_file", type=str, default="data/teacher_answers.jsonl")
    parser.add_argument("--teacher_model", type=str, default="google/flan-t5-small")
    parser.add_argument(
        "--task",
        type=str,
        choices=["text-generation", "text2text-generation"],
        default="text2text-generation",
        help="Pipeline task used for teacher generation.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional max rows to process.")
    parser.add_argument("--max_new_tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.3)
    return parser.parse_args()


def read_jsonl(path: Path) -> List[dict]:
    """Read JSONL into a list of dicts."""
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def build_prompt(instruction: str, context: Dict) -> str:
    """Create one generation prompt from input row."""
    context_json = json.dumps(context if isinstance(context, dict) else {}, ensure_ascii=False)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"CONTEXT:\n{context_json}\n\n"
        f"USER QUESTION:\n{instruction}\n\n"
        "ASSISTANT ANSWER:\n"
    )


def is_valid_agrisathi_format(text: str) -> bool:
    """Basic format validation for required headings and main section choice."""
    has_common = all(h in text for h in REQUIRED_HEADINGS)
    has_choice = ("Action steps:" in text) or ("Follow-up questions:" in text)
    return has_common and has_choice


def generate_teacher_answer(
    generator,
    prompt: str,
    task: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    """Run generation and return only the answer text."""
    if task == "text-generation":
        out = generator(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            num_return_sequences=1,
            eos_token_id=generator.tokenizer.eos_token_id,
            pad_token_id=generator.tokenizer.eos_token_id,
        )[0]["generated_text"]
        if out.startswith(prompt):
            return out[len(prompt) :].strip()
        return out.strip()

    # text2text-generation path
    out = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
        num_return_sequences=1,
    )[0]["generated_text"]
    return out.strip()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Auto-pick GPU if available.
    use_gpu = torch.cuda.is_available()
    device = 0 if use_gpu else -1

    print(f"Teacher model: {args.teacher_model}")
    print(f"Task: {args.task}")
    print(f"Device: {'GPU' if use_gpu else 'CPU'}")

    generator = pipeline(
        task=args.task,
        model=args.teacher_model,
        device=device,
    )

    rows = read_jsonl(input_path)
    if args.limit is not None and args.limit >= 0:
        rows = rows[: args.limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0

    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            instruction = str(row.get("instruction", "")).strip()
            context = row.get("context", {})
            if not instruction:
                continue

            prompt = build_prompt(instruction, context)
            teacher_answer = generate_teacher_answer(
                generator=generator,
                prompt=prompt,
                task=args.task,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )

            # Retry once with lower temperature if format is invalid.
            if not is_valid_agrisathi_format(teacher_answer):
                teacher_answer = generate_teacher_answer(
                    generator=generator,
                    prompt=prompt,
                    task=args.task,
                    max_new_tokens=args.max_new_tokens,
                    temperature=0.2,
                )

            out_record = {
                "instruction": instruction,
                "context": context if isinstance(context, dict) else {},
                "teacher_answer": teacher_answer,
                "teacher_model": args.teacher_model,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            f.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Saved {kept} teacher answers to: {output_path}")


if __name__ == "__main__":
    main()
