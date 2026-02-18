"""
LoRA SFT training on augmented AgriSathi data.

Input:
- data/train_data_augmented.jsonl
  (expects keys: instruction, context, answer)

Output:
- augmented_lora_adapter/ (LoRA adapter via save_pretrained)

Example:
python train/finetune_on_augmented.py
python train/finetune_on_augmented.py --model_name distilgpt2 --epochs 2 --output_dir augmented_lora_adapter
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)


SYSTEM_RULES = (
    "You are AgriSathi, a beginner-safe agriculture assistant.\n"
    "Use sections in order:\n"
    "Problem summary:\n"
    "Action steps or Follow-up questions:\n"
    "Safety disclaimer:\n"
    "When to consult expert:\n"
    "Never provide exact chemical dosage or mixing instructions.\n"
    "Always advise label-check and consulting a local agronomist."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA SFT on augmented AgriSathi data")
    parser.add_argument("--train_file", type=str, default="data/train_data_augmented.jsonl")
    parser.add_argument("--model_name", type=str, default="distilgpt2")
    parser.add_argument("--output_dir", type=str, default="augmented_lora_adapter")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    return parser.parse_args()


def build_training_text(example: Dict) -> Dict[str, str]:
    context_obj = example.get("context", {})
    if not isinstance(context_obj, dict):
        context_obj = {}

    instruction = str(example.get("instruction", "")).strip()
    answer = str(example.get("answer", "")).strip()
    context_json = json.dumps(context_obj, ensure_ascii=False)

    text = (
        f"SYSTEM:{SYSTEM_RULES}\n"
        f"CONTEXT:{context_json}\n"
        f"USER:{instruction}\n"
        "ASSISTANT:\n"
        f"{answer}"
    )
    return {"text": text}


def pick_lora_targets(model: torch.nn.Module) -> List[str]:
    candidates = ["c_attn", "c_proj", "q_proj", "k_proj", "v_proj", "o_proj"]
    found = set()
    for name, _mod in model.named_modules():
        for suffix in candidates:
            if name.endswith(suffix):
                found.add(suffix)
    return sorted(found) if found else ["c_attn", "c_proj"]


def main() -> None:
    args = parse_args()
    use_cuda = torch.cuda.is_available()

    print(f"Model: {args.model_name}")
    print(f"Device: {'CUDA' if use_cuda else 'CPU'}")
    print(f"Train file: {args.train_file}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    targets = pick_lora_targets(model)
    print(f"LoRA targets: {targets}")
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=targets,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    ds = load_dataset("json", data_files=args.train_file, split="train")
    ds = ds.map(build_training_text, remove_columns=ds.column_names)

    def tokenize_batch(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        tok = tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )
        tok["labels"] = tok["input_ids"].copy()
        return tok

    tokenized = ds.map(tokenize_batch, batched=True, remove_columns=["text"])

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

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter to: {args.output_dir}")


if __name__ == "__main__":
    main()
