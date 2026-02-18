"""
Response distillation training script for AgriSathi.

Goal:
- Train a small student model on teacher responses.

Input:
- data/teacher_answers.jsonl

Output:
- LoRA adapter folder (default): student_distilled_response_adapter/
  OR
- Full fine-tuned model folder (if --use_lora false)

Example:
python train/distill_response_sft.py
python train/distill_response_sft.py --epochs 2 --use_lora true
python train/distill_response_sft.py --use_lora false --output_dir student_distilled_full
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

try:
    from peft import LoraConfig, TaskType, get_peft_model

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


# Shared rules included in each example so the student learns stable format + safety behavior.
SYSTEM_RULES = (
    "You are AgriSathi, a beginner-safe agriculture assistant.\n"
    "Always follow format with headings:\n"
    "Problem summary:\n"
    "Action steps or Follow-up questions:\n"
    "Safety disclaimer:\n"
    "When to consult expert:\n"
    "Never provide exact chemical dosage or mixing recipes.\n"
    "Always advise checking labels and consulting a local agronomist."
)


def str2bool(value: str) -> bool:
    """Parse common string forms of booleans for argparse."""
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Response distillation SFT for student model")
    parser.add_argument("--train_file", type=str, default="data/teacher_answers.jsonl")
    parser.add_argument("--model_name", type=str, default="distilgpt2")
    parser.add_argument("--output_dir", type=str, default="student_distilled_response_adapter")
    parser.add_argument(
        "--use_lora",
        type=str,
        default="true",
        help="Use LoRA adapter training (true/false).",
    )
    parser.add_argument("--epochs", type=float, default=1.0, help="Default 1 epoch.")
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    return parser.parse_args()


def build_training_text(example: Dict) -> Dict[str, str]:
    """
    Build one distillation training sample:
    SYSTEM:<rules>
    CONTEXT:<json>
    USER:<instruction>
    ASSISTANT:
    <teacher_answer>
    """
    context_obj = example.get("context", {})
    if not isinstance(context_obj, dict):
        context_obj = {}

    instruction = str(example.get("instruction", "")).strip()
    teacher_answer = str(example.get("teacher_answer", "")).strip()
    context_json = json.dumps(context_obj, ensure_ascii=False)

    text = (
        f"SYSTEM:{SYSTEM_RULES}\n"
        f"CONTEXT:{context_json}\n"
        f"USER:{instruction}\n"
        "ASSISTANT:\n"
        f"{teacher_answer}"
    )
    return {"text": text}


def pick_lora_targets(model: torch.nn.Module) -> List[str]:
    """
    Choose target modules that commonly work for GPT-style models.
    DistilGPT2/GPT2: c_attn, c_proj
    LLaMA-like: q_proj, k_proj, v_proj, o_proj
    """
    candidates = ["c_attn", "c_proj", "q_proj", "k_proj", "v_proj", "o_proj"]
    found = set()
    for name, _mod in model.named_modules():
        for suffix in candidates:
            if name.endswith(suffix):
                found.add(suffix)
    return sorted(found) if found else ["c_attn", "c_proj"]


def main() -> None:
    args = parse_args()
    use_lora = str2bool(args.use_lora)

    if use_lora and not PEFT_AVAILABLE:
        raise ImportError("peft is required for LoRA training. Install it or use --use_lora false.")

    use_cuda = torch.cuda.is_available()
    print(f"Loading student model: {args.model_name}")
    print(f"Device: {'CUDA' if use_cuda else 'CPU'}")
    print(f"LoRA enabled: {use_lora}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    # DistilGPT2 has no pad token; reuse EOS token for stable batch padding.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    if use_lora:
        target_modules = pick_lora_targets(model)
        print(f"LoRA target modules: {target_modules}")
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    dataset = load_dataset("json", data_files=args.train_file, split="train")
    dataset = dataset.map(build_training_text, remove_columns=dataset.column_names)

    def tokenize_batch(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        tokens = tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized = dataset.map(tokenize_batch, batched=True, remove_columns=["text"])

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        fp16=use_cuda,
        bf16=False,
        dataloader_pin_memory=use_cuda,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )
    trainer.train()

    # Save behavior:
    # - LoRA mode: adapter files in output_dir
    # - Full mode: full fine-tuned model in output_dir
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved artifacts to: {args.output_dir}")


if __name__ == "__main__":
    main()
