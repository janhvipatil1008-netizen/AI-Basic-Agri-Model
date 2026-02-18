"""
Beginner-friendly LoRA SFT script for AgriSathi-style data.

Input:
- data/train_data.jsonl
  (JSONL rows with keys: instruction, context (object), answer)

Output:
- lora_adapter/
  (PEFT LoRA adapter files saved via save_pretrained)

Why LoRA:
- Full fine-tuning updates all model weights (large memory + compute cost).
- LoRA trains only small adapter matrices added to selected layers.
- This is much cheaper and is practical on smaller hardware.

Example usage:
1) CPU-safe default (distilgpt2):
   python train/lora_sft.py

2) CUDA machine, different base model:
   python train/lora_sft.py --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0
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


# System rules are injected into every training sample so the assistant learns
# output format and safety behavior consistently.
SYSTEM_RULES = (
    "You are AgriSathi, a beginner-safe agriculture assistant.\n"
    "Use exactly these sections in order:\n"
    "Problem summary:\n"
    "Action steps: OR Follow-up questions:\n"
    "Safety disclaimer:\n"
    "When to consult expert:\n"
    "Never provide exact chemical dosage or mixing recipes.\n"
    "Always recommend checking product label and consulting a local agronomist."
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for data path, model choice, and training settings."""
    parser = argparse.ArgumentParser(description="LoRA SFT for AgriSathi JSONL data")
    parser.add_argument("--train_file", type=str, default="data/train_data.jsonl")
    parser.add_argument("--output_dir", type=str, default="lora_adapter")
    parser.add_argument(
        "--model_name",
        type=str,
        default="distilgpt2",
        help="Base causal LM (distilgpt2 is CPU-safe).",
    )
    parser.add_argument("--epochs", type=float, default=2.0, help="Recommended: 1 to 3")
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    return parser.parse_args()


def build_training_text(example: Dict) -> Dict[str, str]:
    """
    Create one plain training string from one dataset row.

    Target format:
    SYSTEM: <rules>
    CONTEXT: <json>
    USER: <instruction>
    ASSISTANT:
    <answer>
    """
    context_obj = example.get("context", {})
    if not isinstance(context_obj, dict):
        context_obj = {}

    context_json = json.dumps(context_obj, ensure_ascii=False)
    instruction = str(example.get("instruction", "")).strip()
    answer = str(example.get("answer", "")).strip()

    text = (
        f"SYSTEM: {SYSTEM_RULES}\n"
        f"CONTEXT: {context_json}\n"
        f"USER: {instruction}\n"
        "ASSISTANT:\n"
        f"{answer}"
    )
    return {"text": text}


def pick_lora_targets(model: torch.nn.Module) -> List[str]:
    """
    Pick LoRA target modules that work across common GPT-style architectures.

    - distilgpt2/gpt2 often use: c_attn, c_proj
    - many LLaMA-style models use: q_proj, k_proj, v_proj, o_proj
    """
    candidate_suffixes = [
        "c_attn",
        "c_proj",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ]

    found = set()
    for name, _module in model.named_modules():
        for suffix in candidate_suffixes:
            if name.endswith(suffix):
                found.add(suffix)

    if found:
        return sorted(found)

    # Safe fallback for unusual GPT-style naming.
    return ["c_attn", "c_proj"]


def main() -> None:
    args = parse_args()
    use_cuda = torch.cuda.is_available()

    print(f"Loading base model: {args.model_name}")
    print(f"Device: {'CUDA' if use_cuda else 'CPU'}")

    # Base model = original pretrained LM.
    # Adapter = small LoRA weights learned during SFT (saved separately).
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    # GPT-like tokenizers may not define a pad token.
    # Reuse EOS token to enable padding in Trainer batches.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    target_modules = pick_lora_targets(model)
    print(f"LoRA target modules: {target_modules}")

    # LoRA adapter configuration:
    # - r: rank of adapter matrices
    # - lora_alpha: scaling factor
    # - lora_dropout: regularization
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"Loading dataset: {args.train_file}")
    raw_ds = load_dataset("json", data_files=args.train_file, split="train")
    text_ds = raw_ds.map(build_training_text, remove_columns=raw_ds.column_names)

    def tokenize_batch(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        # Truncation keeps memory usage stable on small machines.
        tokens = tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )
        # Causal LM training uses input_ids as labels.
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized_ds = text_ds.map(tokenize_batch, batched=True, remove_columns=["text"])

    # Keep training settings conservative for beginner hardware.
    training_args = TrainingArguments(
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
        args=training_args,
        train_dataset=tokenized_ds,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    trainer.train()

    # Save adapter only (not full base model). This is the LoRA output artifact.
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter to: {args.output_dir}")


if __name__ == "__main__":
    main()
