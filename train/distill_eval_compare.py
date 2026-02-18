"""
Compare base student vs distilled student outputs on eval data.

Inputs:
- data/eval_data.jsonl
- base student model name
- distilled adapter path OR full distilled model path

Output:
- data/distill_report.md

Examples (PowerShell):
python train/distill_eval_compare.py --base_model distilgpt2 --distilled_path student_distilled_response_adapter --distilled_type adapter
python train/distill_eval_compare.py --base_model distilgpt2 --distilled_path student_distilled_rkld_full --distilled_type full
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


SYSTEM_RULES = (
    "You are AgriSathi, a beginner-safe agriculture assistant.\n"
    "Use headings in order:\n"
    "Problem summary:\n"
    "Action steps or Follow-up questions:\n"
    "Safety disclaimer:\n"
    "When to consult expert:\n"
    "Never provide exact dosage or mixing instructions."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare base vs distilled student outputs")
    parser.add_argument("--eval_file", type=str, default="data/eval_data.jsonl")
    parser.add_argument("--base_model", type=str, default="distilgpt2")
    parser.add_argument("--distilled_path", type=str, required=True)
    parser.add_argument(
        "--distilled_type",
        type=str,
        choices=["adapter", "full"],
        default="adapter",
        help="adapter = LoRA adapter path, full = full fine-tuned model path",
    )
    parser.add_argument("--report_path", type=str, default="data/distill_report.md")
    parser.add_argument("--max_new_tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--limit", type=int, default=None)
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
        f"SYSTEM:{SYSTEM_RULES}\n"
        f"CONTEXT:{context_json}\n"
        f"USER:{instruction}\n"
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
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    if text.startswith(prompt):
        return text[len(prompt) :].strip()
    return text.strip()


def md_block(text: str) -> str:
    safe = str(text).replace("```", "'''")
    return f"```\n{safe}\n```"


def main() -> None:
    args = parse_args()
    eval_path = Path(args.eval_file)
    report_path = Path(args.report_path)

    if not eval_path.exists():
        raise FileNotFoundError(f"Eval file not found: {eval_path}")
    if not Path(args.distilled_path).exists():
        raise FileNotFoundError(f"Distilled path not found: {args.distilled_path}")

    if args.distilled_type == "adapter" and not PEFT_AVAILABLE:
        raise ImportError("peft is required for --distilled_type adapter")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Base student model
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model).to(device)
    base_model.eval()

    # Distilled student model
    if args.distilled_type == "adapter":
        distilled_backbone = AutoModelForCausalLM.from_pretrained(args.base_model).to(device)
        distilled_model = PeftModel.from_pretrained(distilled_backbone, args.distilled_path).to(device)
    else:
        distilled_model = AutoModelForCausalLM.from_pretrained(args.distilled_path).to(device)
    distilled_model.eval()

    rows = read_jsonl(eval_path)
    if args.limit is not None and args.limit >= 0:
        rows = rows[: args.limit]

    lines: List[str] = []
    lines.append("# Distillation Eval Report")
    lines.append("")
    lines.append(f"- Eval file: `{args.eval_file}`")
    lines.append(f"- Base model: `{args.base_model}`")
    lines.append(f"- Distilled path: `{args.distilled_path}`")
    lines.append(f"- Distilled type: `{args.distilled_type}`")
    lines.append(f"- Samples: {len(rows)}")
    lines.append(
        f"- Generation: max_new_tokens={args.max_new_tokens}, temperature={args.temperature}, top_p={args.top_p}"
    )
    lines.append("")

    for i, row in enumerate(rows, start=1):
        question = str(row.get("instruction", "")).strip()
        context = row.get("context", {})
        if not question:
            continue

        prompt = build_prompt(question, context)
        base_output = generate_text(
            model=base_model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        distilled_output = generate_text(
            model=distilled_model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        lines.append(f"## Example {i}")
        lines.append("")
        lines.append(f"**Question**: {question}")
        lines.append("")
        lines.append("**Base Output**")
        lines.append(md_block(base_output))
        lines.append("")
        lines.append("**Distilled Output**")
        lines.append(md_block(distilled_output))
        lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved report to: {report_path}")


if __name__ == "__main__":
    main()
