"""
Beginner-friendly CLI chatbot with optional RAG + optional LoRA adapter.

Features:
- Hugging Face text-generation pipeline
- CPU by default, GPU if available
- Guardrails via SYSTEM_PROMPT
- Optional LoRA adapter (--adapter_path)
- Optional RAG retrieval from FAISS (--rag on/off)
- Saves interactions to data/baseline_outputs.jsonl
"""

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Dict, List

import faiss
import numpy as np
import torch
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# You can override this model from your terminal if needed:
#   $env:MODEL_NAME = "distilgpt2"
MODEL_NAME = os.getenv("MODEL_NAME", "distilgpt2")

# Guardrails and stable response format are enforced through this prompt.
SYSTEM_PROMPT = (
    "You are AgriSathi, a beginner-safe agriculture assistant.\n"
    "Output plain text only.\n\n"
    "Core safety rules:\n"
    "- Never provide exact chemical dosage, concentration, or tank-mix instructions.\n"
    "- Never provide step-by-step pesticide mixing procedures.\n"
    "- If asked for chemical specifics, provide only general safety-first guidance.\n"
    "- Always include: check the product label and consult a local agronomist.\n\n"
    "Decide first whether key details are present.\n"
    "Required details: crop name, crop stage, and location/symptoms.\n\n"
    "If ALL required details are present, use EXACTLY this format and order:\n"
    "Problem summary:\n"
    "Action steps:\n"
    "Safety disclaimer:\n"
    "When to consult expert:\n\n"
    "If ANY required detail is missing, do NOT output Action steps.\n"
    "Use EXACTLY this format and order instead:\n"
    "Problem summary:\n"
    "Follow-up questions:\n"
    "1) ...\n"
    "2) ...\n"
    "3) ... (optional, max 3 total)\n"
    "Safety disclaimer:\n"
    "When to consult expert:\n\n"
    "Formatting constraints:\n"
    "- Do not add extra headings, markdown, or text outside the required sections.\n"
    "- Keep content concise, practical, and easy for beginners."
)

MAX_NEW_TOKENS = 220
TEMPERATURE = 0.3
RETRY_TEMPERATURE = 0.2
OUTPUT_PATH = os.path.join("data", "baseline_outputs.jsonl")

COMMON_REQUIRED_HEADINGS = [
    "Problem summary:",
    "Safety disclaimer:",
    "When to consult expert:",
]

AUTO10_QUESTIONS = [
    "My tomato leaves are turning yellow from the bottom. What should I do first?",
    "There are tiny white insects under my chilli leaves. How can I control them safely?",
    "Rice seedlings look weak after transplanting. How can I help recovery?",
    "My wheat field has patchy growth. What basic checks should I do?",
    "How often should I irrigate brinjal plants during hot summer days?",
    "Some onion plants are rotting near the base. What are possible reasons and steps?",
    "A cow in my farm is eating less feed than usual. What immediate actions are safe?",
    "What are practical steps to improve soil health before the next cropping season?",
    "How can I prepare a low-cost pest monitoring plan for a small vegetable farm?",
    "My drip irrigation lines clog frequently. How can I troubleshoot this?",
]


class RagRetriever:
    """Load FAISS + metadata and retrieve top-k relevant chunks per question."""

    def __init__(
        self,
        index_path: str,
        meta_path: str,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"RAG index not found: {index_path}")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"RAG metadata not found: {meta_path}")

        self.index = faiss.read_index(index_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.embedder = SentenceTransformer(embedding_model)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, str]]:
        vec = self.embedder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        k = min(top_k, len(self.meta))
        scores, indices = self.index.search(vec, k)
        results: List[Dict[str, str]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            item = self.meta[idx]
            results.append(
                {
                    "title": item.get("title", f"chunk-{idx}"),
                    "text": item.get("text", ""),
                    "source_path": item.get("source_path", ""),
                    "score": float(score),
                }
            )
        return results


def build_generator(model_name: str, adapter_path: str = ""):
    """Create text-generation pipeline and optionally load LoRA adapter."""
    use_gpu = torch.cuda.is_available()
    device = 0 if use_gpu else -1

    print(f"Loading model: {model_name}")
    print(f"Running on: {'GPU' if use_gpu else 'CPU'}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    if adapter_path:
        print(f"Loading LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        model = base_model

    return pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
    )


def build_prompt(user_question: str, retrieved_chunks: List[Dict[str, str]] | None = None) -> str:
    """Compose prompt with optional retrieved context (open-book style)."""
    rag_context = "No external context provided."
    if retrieved_chunks:
        lines = []
        for i, ch in enumerate(retrieved_chunks, start=1):
            lines.append(f"[Source {i}] {ch['title']}\n{ch['text']}")
        rag_context = "\n\n".join(lines)

    return (
        f"{SYSTEM_PROMPT}\n\n"
        "Trusted context (use this as primary evidence; do not invent facts beyond it):\n"
        f"{rag_context}\n\n"
        f"User question:\n{user_question}\n\n"
        "Answer:\n"
    )


def generate_answer(generator, prompt_used: str, temperature: float = TEMPERATURE) -> str:
    eos_id = generator.tokenizer.eos_token_id
    result = generator(
        prompt_used,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
        num_return_sequences=1,
        eos_token_id=eos_id,
        pad_token_id=eos_id,
    )[0]["generated_text"]

    if result.startswith(prompt_used):
        return result[len(prompt_used) :].strip()
    return result.strip()


def is_valid_output(answer: str) -> bool:
    has_common = all(h in answer for h in COMMON_REQUIRED_HEADINGS)
    has_action_or_followup = ("Action steps:" in answer) or ("Follow-up questions:" in answer)
    return has_common and has_action_or_followup


def append_jsonl(record: dict, output_path: str = OUTPUT_PATH) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_one_question(generator, question: str, retriever: RagRetriever | None, top_k: int):
    retrieved_chunks = retriever.retrieve(question, top_k=top_k) if retriever else []
    prompt_used = build_prompt(question, retrieved_chunks=retrieved_chunks)

    answer = generate_answer(generator, prompt_used, temperature=TEMPERATURE)
    if not is_valid_output(answer):
        answer = generate_answer(generator, prompt_used, temperature=RETRY_TEMPERATURE)

    source_titles = [c["title"] for c in retrieved_chunks]
    return answer, prompt_used, source_titles


def run_auto10(generator, retriever: RagRetriever | None, top_k: int) -> None:
    print("\nRunning --auto10 with pre-defined farmer-like questions...\n")
    for idx, question in enumerate(AUTO10_QUESTIONS, start=1):
        answer, prompt_used, source_titles = run_one_question(generator, question, retriever, top_k)

        print(f"[{idx}/10] Question: {question}")
        print(answer + "\n")
        if source_titles:
            print("Sources used:")
            for title in source_titles:
                print(f"- {title}")
            print("")

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_name": MODEL_NAME,
            "question": question,
            "prompt_used": prompt_used,
            "answer": answer,
            "sources": source_titles,
        }
        append_jsonl(record)


def run_interactive(generator, retriever: RagRetriever | None, top_k: int) -> None:
    print("\nAgriSathi CLI ready. Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("AgriSathi: Goodbye!")
            break

        if not question:
            print("AgriSathi: Please enter a question.")
            continue

        answer, prompt_used, source_titles = run_one_question(generator, question, retriever, top_k)
        print("\n" + answer + "\n")

        if source_titles:
            print("Sources used:")
            for title in source_titles:
                print(f"- {title}")
            print("")

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_name": MODEL_NAME,
            "question": question,
            "prompt_used": prompt_used,
            "answer": answer,
            "sources": source_titles,
        }
        append_jsonl(record)


def parse_args():
    parser = argparse.ArgumentParser(description="AgriSathi CLI with optional RAG and LoRA")
    parser.add_argument(
        "--model_name",
        type=str,
        default=MODEL_NAME,
        help="Base model name (default from MODEL_NAME env var or distilgpt2).",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="",
        help="Optional LoRA adapter path. Leave empty to use base model only.",
    )
    parser.add_argument(
        "--rag",
        type=str,
        choices=["on", "off"],
        default="off",
        help="Toggle retrieval-augmented generation.",
    )
    parser.add_argument("--rag_index_path", type=str, default="rag_index.faiss")
    parser.add_argument("--rag_meta_path", type=str, default="rag_index_meta.json")
    parser.add_argument(
        "--rag_embedding_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--rag_top_k", type=int, default=3)
    parser.add_argument(
        "--auto10",
        action="store_true",
        help="Run 10 pre-defined farmer-like questions and save outputs to JSONL.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    generator = build_generator(model_name=args.model_name, adapter_path=args.adapter_path)

    retriever = None
    if args.rag == "on":
        print("RAG mode: ON")
        retriever = RagRetriever(
            index_path=args.rag_index_path,
            meta_path=args.rag_meta_path,
            embedding_model=args.rag_embedding_model,
        )
    else:
        print("RAG mode: OFF")

    if args.auto10:
        run_auto10(generator, retriever=retriever, top_k=args.rag_top_k)
    else:
        run_interactive(generator, retriever=retriever, top_k=args.rag_top_k)


if __name__ == "__main__":
    main()
