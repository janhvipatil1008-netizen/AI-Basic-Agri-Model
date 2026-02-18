"""
Compare base model vs LoRA-adapted model on eval set and write Markdown report.

Inputs:
- data/eval_data.jsonl  (JSONL rows with instruction, context, answer)

Behavior:
- Generate output with base model (no adapter)
- Generate output with same base model + LoRA adapter from lora_adapter/
- Save comparison report to data/eval_report.md

Example:
python train/eval_compare.py
python train/eval_compare.py --model_name distilgpt2 --adapter_dir lora_adapter
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


# Keep prompt style aligned with training format for fair comparison.
SYSTEM_RULES = (
    "You are AgriSathi, a beginner-safe agriculture assistant.\n"
    "Use sections in order:\n"
    "Problem summary:\n"
    "Action steps or Follow-up questions:\n"
    "Safety disclaimer:\n"
    "When to consult expert:\n"
    "Never provide exact chemical dosage or mixing instructions."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare base vs LoRA outputs on eval data")
    parser.add_argument("--eval_file", type=str, default="data/eval_data.jsonl")
    parser.add_argument("--model_name", type=str, default="distilgpt2")
    parser.add_argument("--adapter_dir", type=str, default="lora_adapter")
    parser.add_argument("--report_path", type=str, default="data/eval_report.md")
    parser.add_argument("--max_new_tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on eval rows.")
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
        f"SYSTEM: {SYSTEM_RULES}\n"
        f"CONTEXT: {context_json}\n"
        f"USER: {instruction}\n"
        "ASSISTANT:\n"
    )


def generate_text(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if full_text.startswith(prompt):
        return full_text[len(prompt) :].strip()
    return full_text.strip()


def markdown_code_block(text: str) -> str:
    # Escape accidental triple-backticks to keep Markdown valid.
    safe = text.replace("```", "'''")
    return f"```\n{safe}\n```"


def main() -> None:
    args = parse_args()
    eval_path = Path(args.eval_file)
    report_path = Path(args.report_path)

    if not eval_path.exists():
        raise FileNotFoundError(f"Eval file not found: {eval_path}")

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    print(f"Loading model on: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Base model (no adapter).
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    base_model.eval()

    # Same base model + LoRA adapter.
    lora_backbone = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    lora_model = PeftModel.from_pretrained(lora_backbone, args.adapter_dir).to(device)
    lora_model.eval()

    rows = read_jsonl(eval_path)
    if args.limit is not None and args.limit >= 0:
        rows = rows[: args.limit]

    lines: List[str] = []
    lines.append("# Eval Comparison Report")
    lines.append("")
    lines.append(f"- Eval file: `{args.eval_file}`")
    lines.append(f"- Base model: `{args.model_name}`")
    lines.append(f"- Adapter: `{args.adapter_dir}`")
    lines.append(f"- Samples compared: {len(rows)}")
    lines.append(
        f"- Generation settings: `max_new_tokens={args.max_new_tokens}`, `temperature={args.temperature}`, `top_p={args.top_p}`"
    )
    lines.append("")

    for idx, row in enumerate(rows, start=1):
        instruction = str(row.get("instruction", "")).strip()
        context = row.get("context", {})
        reference = row.get("answer")

        prompt = build_prompt(instruction, context)
        base_output = generate_text(
            base_model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        lora_output = generate_text(
            lora_model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        lines.append(f"## Example {idx}")
        lines.append("")
        lines.append(f"**Question**: {instruction}")
        lines.append("")
        lines.append("**Base Output**")
        lines.append(markdown_code_block(base_output))
        lines.append("")
        lines.append("**LoRA Output**")
        lines.append(markdown_code_block(lora_output))
        lines.append("")
        if isinstance(reference, str) and reference.strip():
            lines.append("**Reference Answer**")
            lines.append(markdown_code_block(reference.strip()))
            lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
