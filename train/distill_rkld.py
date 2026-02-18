"""
Reverse-KL distillation for causal LMs (teacher -> student).

Inputs:
- data/teacher_answers.jsonl
- local teacher model name (logits required)
- local student model name

Behavior:
- Teacher is frozen
- Student is trainable (optional LoRA)
- Loss: KL(student || teacher) on assistant answer tokens only

Output:
- student_distilled_rkld_adapter/ (LoRA) or full model folder

Example (PowerShell):
python train/distill_rkld.py --teacher_model distilgpt2 --student_model distilgpt2
python train/distill_rkld.py --teacher_model distilgpt2 --student_model distilgpt2 --use_lora true
python train/distill_rkld.py --teacher_model distilgpt2 --student_model distilgpt2 --temperature 2.0 --epochs 1
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

try:
    from peft import LoraConfig, TaskType, get_peft_model

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


SYSTEM_RULES = (
    "You are AgriSathi, a beginner-safe agriculture assistant.\n"
    "Use sections in order:\n"
    "Problem summary:\n"
    "Action steps or Follow-up questions:\n"
    "Safety disclaimer:\n"
    "When to consult expert:\n"
    "Never provide exact chemical dosage or mixing recipes.\n"
    "Always advise checking labels and consulting a local agronomist."
)


def str2bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reverse-KL distillation for student LM")
    parser.add_argument("--train_file", type=str, default="data/teacher_answers.jsonl")
    parser.add_argument("--teacher_model", type=str, required=True)
    parser.add_argument("--student_model", type=str, default="distilgpt2")
    parser.add_argument("--output_dir", type=str, default="student_distilled_rkld_adapter")
    parser.add_argument("--use_lora", type=str, default="true", help="true/false")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_steps", type=int, default=-1, help="Optional hard cap, -1 disables.")
    return parser.parse_args()


def pick_lora_targets(model: torch.nn.Module) -> List[str]:
    candidates = ["c_attn", "c_proj", "q_proj", "k_proj", "v_proj", "o_proj"]
    found = set()
    for name, _mod in model.named_modules():
        for suffix in candidates:
            if name.endswith(suffix):
                found.add(suffix)
    return sorted(found) if found else ["c_attn", "c_proj"]


def build_prompt(instruction: str, context: Dict) -> str:
    context_json = json.dumps(context if isinstance(context, dict) else {}, ensure_ascii=False)
    return (
        f"SYSTEM:{SYSTEM_RULES}\n"
        f"CONTEXT:{context_json}\n"
        f"USER:{instruction}\n"
        "ASSISTANT:\n"
    )


@dataclass
class ExampleFeatures:
    input_ids: List[int]
    attention_mask: List[int]
    assistant_mask: List[int]


def prepare_features(example: Dict, tokenizer: AutoTokenizer, max_length: int) -> ExampleFeatures:
    instruction = str(example.get("instruction", "")).strip()
    context = example.get("context", {})
    teacher_answer = str(example.get("teacher_answer", "")).strip()

    prompt = build_prompt(instruction, context)
    full_text = prompt + teacher_answer

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_enc = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    input_ids = full_enc["input_ids"]
    attention_mask = full_enc["attention_mask"]

    # assistant_mask marks tokens belonging to teacher answer span.
    prompt_len = min(len(prompt_ids), len(input_ids))
    assistant_mask = [0] * prompt_len + [1] * (len(input_ids) - prompt_len)
    return ExampleFeatures(
        input_ids=input_ids,
        attention_mask=attention_mask,
        assistant_mask=assistant_mask,
    )


def collate_batch(features: List[ExampleFeatures], pad_token_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(len(x.input_ids) for x in features)
    input_ids, attn, assistant = [], [], []
    for f in features:
        pad_len = max_len - len(f.input_ids)
        input_ids.append(f.input_ids + [pad_token_id] * pad_len)
        attn.append(f.attention_mask + [0] * pad_len)
        assistant.append(f.assistant_mask + [0] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
        "assistant_mask": torch.tensor(assistant, dtype=torch.float32),
    }


def compute_reverse_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    attention_mask: torch.Tensor,
    assistant_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """
    Reverse KL: KL(student || teacher), masked to assistant tokens only.
    """
    # Causal shift: token t is predicted from logits at t-1.
    s = student_logits[:, :-1, :] / temperature
    t = teacher_logits[:, :-1, :] / temperature

    # Valid loss positions correspond to target token positions [1:].
    valid = attention_mask[:, 1:].float() * assistant_mask[:, 1:]

    s_log_probs = F.log_softmax(s, dim=-1)
    t_log_probs = F.log_softmax(t, dim=-1)
    s_probs = s_log_probs.exp()

    # KL(student||teacher) = sum p_s * (log p_s - log p_t)
    token_kl = torch.sum(s_probs * (s_log_probs - t_log_probs), dim=-1)
    token_kl = token_kl * valid

    denom = valid.sum().clamp(min=1.0)
    loss = token_kl.sum() / denom
    return loss * (temperature ** 2)


def main() -> None:
    args = parse_args()
    use_lora = str2bool(args.use_lora)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if use_lora and not PEFT_AVAILABLE:
        raise ImportError("peft is required for LoRA mode. Install peft or use --use_lora false.")

    print(f"Device: {device}")
    print(f"Teacher model: {args.teacher_model}")
    print(f"Student model: {args.student_model}")
    print(f"Use LoRA: {use_lora}")

    tokenizer = AutoTokenizer.from_pretrained(args.student_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    teacher = AutoModelForCausalLM.from_pretrained(args.teacher_model).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = AutoModelForCausalLM.from_pretrained(args.student_model).to(device)
    student.config.pad_token_id = tokenizer.pad_token_id

    if use_lora:
        targets = pick_lora_targets(student)
        print(f"LoRA targets: {targets}")
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=targets,
        )
        student = get_peft_model(student, lora_cfg)
        student.print_trainable_parameters()

    raw_ds = load_dataset("json", data_files=args.train_file, split="train")
    processed = [
        prepare_features(row, tokenizer=tokenizer, max_length=args.max_length)
        for row in raw_ds
        if str(row.get("instruction", "")).strip() and str(row.get("teacher_answer", "")).strip()
    ]
    if not processed:
        raise RuntimeError("No valid training rows found in teacher_answers.jsonl")

    loader = DataLoader(
        processed,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_batch(x, pad_token_id=tokenizer.pad_token_id),
    )

    trainable_params = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    total_update_steps_per_epoch = max(1, len(loader) // args.gradient_accumulation_steps)
    total_steps = total_update_steps_per_epoch * args.epochs
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(1, total_steps),
    )

    student.train()
    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        running_loss = 0.0
        for step, batch in enumerate(loader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            assistant_mask = batch["assistant_mask"].to(device)

            with torch.no_grad():
                teacher_logits = teacher(input_ids=input_ids, attention_mask=attention_mask).logits

            student_logits = student(input_ids=input_ids, attention_mask=attention_mask).logits
            loss = compute_reverse_kl_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                attention_mask=attention_mask,
                assistant_mask=assistant_mask,
                temperature=args.temperature,
            )

            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            running_loss += loss.item()

            if step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % 10 == 0:
                    print(f"epoch={epoch+1} step={global_step} loss={running_loss:.4f}")
                    running_loss = 0.0

                if args.max_steps > 0 and global_step >= args.max_steps:
                    break

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    student.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved student artifacts to: {args.output_dir}")


if __name__ == "__main__":
    main()
