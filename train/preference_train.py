"""
Simple pairwise preference training (RLHF/DPO-style intuition) with LoRA.

Inputs:
- base model name (default: distilgpt2)
- data/preferences.jsonl with keys: prompt, chosen, rejected

Output:
- preference_adapter/ (LoRA adapter + tokenizer)

Core objective per pair:
loss = -log(sigmoid(beta * (logp_chosen - logp_rejected)))

Why this resembles RLHF/DPO:
- We do not need a separate reward model here.
- We directly train the policy/model to prefer "chosen" over "rejected"
  by increasing chosen log-prob relative to rejected log-prob.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup


@dataclass
class PrefExample:
    prompt: str
    chosen: str
    rejected: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pairwise preference training with LoRA")
    parser.add_argument("--model_name", type=str, default="distilgpt2")
    parser.add_argument("--prefs_file", type=str, default="data/preferences.jsonl")
    parser.add_argument("--output_dir", type=str, default="preference_adapter")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_steps", type=int, default=-1, help="Optional hard cap; -1 disables.")
    return parser.parse_args()


def read_preferences(path: Path) -> List[PrefExample]:
    if not path.exists():
        raise FileNotFoundError(f"Preferences file not found: {path}")
    rows: List[PrefExample] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompt = str(obj.get("prompt", "")).strip()
            chosen = str(obj.get("chosen", "")).strip()
            rejected = str(obj.get("rejected", "")).strip()
            if prompt and chosen and rejected:
                rows.append(PrefExample(prompt=prompt, chosen=chosen, rejected=rejected))
    if not rows:
        raise RuntimeError("No valid preference rows found.")
    return rows


def pick_lora_targets(model: torch.nn.Module) -> List[str]:
    # Common GPT-style module names.
    candidates = ["c_attn", "c_proj", "q_proj", "k_proj", "v_proj", "o_proj"]
    found = set()
    for name, _ in model.named_modules():
        for suffix in candidates:
            if name.endswith(suffix):
                found.add(suffix)
    return sorted(found) if found else ["c_attn", "c_proj"]


def build_sequence_tensors(
    tokenizer: AutoTokenizer,
    prompt: str,
    continuation: str,
    max_length: int,
) -> dict:
    """
    Build one full sequence = prompt + continuation and a continuation mask.
    We score only continuation tokens, not prompt tokens.
    """
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(
        prompt + continuation,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )["input_ids"]

    prompt_len = min(len(prompt_ids), len(full_ids))
    cont_mask = [0] * prompt_len + [1] * (len(full_ids) - prompt_len)
    attn = [1] * len(full_ids)
    return {"input_ids": full_ids, "attention_mask": attn, "cont_mask": cont_mask}


def collate_pref_batch(batch: List[dict], pad_token_id: int) -> dict:
    """
    Batch collator for chosen/rejected tensors.
    Pads chosen and rejected streams separately.
    """
    def pad_side(items: List[dict]) -> dict:
        max_len = max(len(x["input_ids"]) for x in items)
        ids, attn, mask = [], [], []
        for x in items:
            p = max_len - len(x["input_ids"])
            ids.append(x["input_ids"] + [pad_token_id] * p)
            attn.append(x["attention_mask"] + [0] * p)
            mask.append(x["cont_mask"] + [0] * p)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "cont_mask": torch.tensor(mask, dtype=torch.float32),
        }

    chosen_items = [x["chosen"] for x in batch]
    rejected_items = [x["rejected"] for x in batch]
    return {"chosen": pad_side(chosen_items), "rejected": pad_side(rejected_items)}


def sequence_logprob(model: torch.nn.Module, batch_side: dict) -> torch.Tensor:
    """
    Compute summed token log-prob over continuation tokens only.
    Returns tensor of shape [batch].
    """
    out = model(
        input_ids=batch_side["input_ids"],
        attention_mask=batch_side["attention_mask"],
    )
    logits = out.logits  # [B, T, V]

    # Causal shift: logits at t predict token at t+1
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)               # [B, T-1, V]
    target_ids = batch_side["input_ids"][:, 1:]                        # [B, T-1]
    token_lp = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)  # [B, T-1]

    # Continuation mask aligned to targets [1:].
    cont = batch_side["cont_mask"][:, 1:] * batch_side["attention_mask"][:, 1:].float()
    return (token_lp * cont).sum(dim=-1)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Base model: {args.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(args.model_name)
    base_model.config.pad_token_id = tokenizer.pad_token_id

    # LoRA makes training much cheaper: we train small adapter matrices only.
    targets = pick_lora_targets(base_model)
    print(f"LoRA targets: {targets}")
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=targets,
    )
    model = get_peft_model(base_model, lora_cfg).to(device)
    model.print_trainable_parameters()

    pref_rows = read_preferences(Path(args.prefs_file))
    processed = []
    for row in pref_rows:
        chosen = build_sequence_tensors(tokenizer, row.prompt, row.chosen, args.max_length)
        rejected = build_sequence_tensors(tokenizer, row.prompt, row.rejected, args.max_length)
        processed.append({"chosen": chosen, "rejected": rejected})

    loader = DataLoader(
        processed,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_pref_batch(b, tokenizer.pad_token_id),
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay
    )

    updates_per_epoch = max(1, len(loader) // args.grad_accum_steps)
    total_steps = updates_per_epoch * args.epochs
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=max(1, total_steps)
    )

    model.train()
    global_step = 0
    running = 0.0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        for step, batch in enumerate(loader, start=1):
            chosen = {k: v.to(device) for k, v in batch["chosen"].items()}
            rejected = {k: v.to(device) for k, v in batch["rejected"].items()}

            logp_chosen = sequence_logprob(model, chosen)
            logp_rejected = sequence_logprob(model, rejected)

            # Pairwise preference objective:
            # maximize (logp_chosen - logp_rejected), scaled by beta.
            # Equivalent minimization form below is smooth and stable.
            margin = args.beta * (logp_chosen - logp_rejected)
            loss = -F.logsigmoid(margin).mean()

            loss = loss / args.grad_accum_steps
            loss.backward()
            running += loss.item()

            if step % args.grad_accum_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % 10 == 0:
                    print(f"epoch={epoch+1} step={global_step} loss={running:.4f}")
                    running = 0.0

                if args.max_steps > 0 and global_step >= args.max_steps:
                    break

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved preference adapter to: {args.output_dir}")


if __name__ == "__main__":
    main()
