"""
Build train/eval JSONL datasets from baseline chatbot logs.

Input:
- data/baseline_outputs.jsonl

Outputs:
- data/train_data.jsonl
- data/eval_data.jsonl

Each output row format:
{
  "instruction": "...",
  "context": {"crop": "...", "stage": "...", "location": "...", "symptoms": "..."},
  "answer": "..."
}

Usage examples (PowerShell):
1) Default settings (20% eval split):
   python data/make_dataset_from_logs.py

2) Cap dataset to 300 examples:
   python data/make_dataset_from_logs.py --limit 300

3) Custom split ratio (10% eval):
   python data/make_dataset_from_logs.py --eval-ratio 0.1

4) Custom paths:
   python data/make_dataset_from_logs.py --input data/baseline_outputs.jsonl --train-output data/train_data.jsonl --eval-output data/eval_data.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple


# Required structure for valid AgriSathi answers.
REQUIRED_HEADINGS = [
    "Problem summary:",
    "Safety disclaimer:",
    "When to consult expert:",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Create train/eval dataset from baseline logs.")
    parser.add_argument(
        "--input",
        default="data/baseline_outputs.jsonl",
        help="Input JSONL log file path.",
    )
    parser.add_argument(
        "--train-output",
        default="data/train_data.jsonl",
        help="Output JSONL path for train split.",
    )
    parser.add_argument(
        "--eval-output",
        default="data/eval_data.jsonl",
        help="Output JSONL path for eval split.",
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.2,
        help="Fraction of examples to place in eval split (default: 0.2).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap for number of kept examples before splitting.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for shuffle/split.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> List[dict]:
    """Read a JSONL file into a list of dict records."""
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip malformed lines, but continue processing the rest.
                print(f"Warning: skipped malformed JSON at line {line_num}")
    return records


def is_valid_agrisathi_answer(answer: str) -> bool:
    """
    Check required format:
    - Must have common headings.
    - Must include either "Action steps:" or "Follow-up questions:".
    """
    if not isinstance(answer, str):
        return False

    has_common = all(h in answer for h in REQUIRED_HEADINGS)
    has_action_or_followup = ("Action steps:" in answer) or ("Follow-up questions:" in answer)
    return has_common and has_action_or_followup


def has_unsafe_exact_dosage(answer: str) -> bool:
    """
    Detect exact dosage-style patterns such as:
    - "2 ml per litre"
    - "1.5 ml/l"
    - "3 grams per litre"
    - "5 g/l"
    """
    if not isinstance(answer, str):
        return True

    patterns = [
        r"\b\d+(?:\.\d+)?\s*ml\s*(?:/|per)\s*(?:l|litre|liter)\b",
        r"\b\d+(?:\.\d+)?\s*(?:g|gram|grams)\s*(?:/|per)\s*(?:l|litre|liter)\b",
    ]
    text = answer.lower()
    return any(re.search(p, text) for p in patterns)


def normalize_instruction(text: str) -> str:
    """Normalize instruction text for duplicate detection."""
    return re.sub(r"\s+", " ", text.strip().lower())


def extract_context(question: str) -> Dict[str, str]:
    """
    Naive keyword-based context extractor.
    Detects:
    - crop names
    - crop stage words
    - Maharashtra + district names
    - symptom keywords

    If nothing is detected, returns an empty object.
    """
    context: Dict[str, str] = {}
    q = question.lower().strip()

    # Crop detection
    crop_keywords = {
        "cotton": "cotton",
        "soybean": "soybean",
        "tomato": "tomato",
        "brinjal": "brinjal",
        "wheat": "wheat",
        "chili": "chili",
        "chilli": "chili",
    }
    for key, normalized in crop_keywords.items():
        if re.search(rf"\b{re.escape(key)}\b", q):
            context["crop"] = normalized
            break

    # Stage detection
    stage_keywords = ["sowing", "vegetative", "flowering", "fruiting"]
    for stage in stage_keywords:
        if re.search(rf"\b{re.escape(stage)}\b", q):
            context["stage"] = stage
            break

    # Location detection: Maharashtra + common district names.
    district_keywords = [
        "pune",
        "nashik",
        "nagpur",
        "aurangabad",
        "jalgaon",
        "solapur",
        "satara",
        "kolhapur",
        "ahmednagar",
        "sangli",
        "amravati",
        "akola",
        "yavatmal",
        "latur",
        "beed",
        "dhule",
        "nandurbar",
        "osmanabad",
        "wardha",
        "buldhana",
        "parbhani",
        "hingoli",
        "nanded",
        "washim",
        "gondia",
        "bhandara",
        "ratnagiri",
        "sindhudurg",
        "raigad",
        "palghar",
        "thane",
        "mumbai",
    ]
    found_locations: List[str] = []
    if re.search(r"\bmaharashtra\b", q):
        found_locations.append("Maharashtra")
    for district in district_keywords:
        if re.search(rf"\b{re.escape(district)}\b", q):
            found_locations.append(district.title())
    if found_locations:
        # Keep unique order.
        unique_locations = list(dict.fromkeys(found_locations))
        context["location"] = ", ".join(unique_locations)

    # Symptom detection
    symptom_keywords = ["curling", "yellowing", "spots", "sticky", "powdery", "holes"]
    found_symptoms = [s for s in symptom_keywords if re.search(rf"\b{re.escape(s)}\b", q)]
    if found_symptoms:
        context["symptoms"] = ", ".join(found_symptoms)

    return context


def convert_records(raw_records: List[dict]) -> Tuple[List[dict], int]:
    """
    Convert raw logs to dataset rows.
    Returns:
    - kept_rows
    - removed_count (invalid format, unsafe dosage, duplicates, missing required fields)
    """
    rows: List[dict] = []
    seen_instructions = set()
    removed_count = 0

    for rec in raw_records:
        question = rec.get("question")
        answer = rec.get("answer")
        if not question or not answer:
            removed_count += 1
            continue

        if not is_valid_agrisathi_answer(answer):
            removed_count += 1
            continue

        if has_unsafe_exact_dosage(answer):
            removed_count += 1
            continue

        norm_instruction = normalize_instruction(question)
        if norm_instruction in seen_instructions:
            removed_count += 1
            continue
        seen_instructions.add(norm_instruction)

        rows.append(
            {
                "instruction": question.strip(),
                "context": extract_context(question),
                "answer": answer.strip(),
            }
        )

    return rows, removed_count


def split_rows(rows: List[dict], eval_ratio: float, seed: int) -> Tuple[List[dict], List[dict]]:
    """Shuffle and split rows into train/eval sets."""
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)

    eval_count = int(len(shuffled) * eval_ratio)
    eval_rows = shuffled[:eval_count]
    train_rows = shuffled[eval_count:]
    return train_rows, eval_rows


def write_jsonl(rows: List[dict], path: Path) -> None:
    """Write list of dict rows to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()

    if not (0.0 <= args.eval_ratio < 1.0):
        raise ValueError("--eval-ratio must be in [0.0, 1.0).")

    input_path = Path(args.input)
    train_path = Path(args.train_output)
    eval_path = Path(args.eval_output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    raw_records = read_jsonl(input_path)
    total_read = len(raw_records)

    kept_rows, removed_count = convert_records(raw_records)

    # Optional global cap after filtering and deduplication.
    if args.limit is not None and args.limit >= 0:
        kept_rows = kept_rows[: args.limit]

    kept_final = len(kept_rows)
    removed_final = total_read - kept_final

    train_rows, eval_rows = split_rows(kept_rows, args.eval_ratio, args.seed)
    write_jsonl(train_rows, train_path)
    write_jsonl(eval_rows, eval_path)

    print("Dataset build summary")
    print(f"total read: {total_read}")
    print(f"kept: {kept_final}")
    print(f"removed: {removed_final}")
    print(f"train count: {len(train_rows)}")
    print(f"eval count: {len(eval_rows)}")


if __name__ == "__main__":
    main()
