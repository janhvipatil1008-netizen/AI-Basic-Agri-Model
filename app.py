"""
AgriSathi mini product demo (single file).

Modes:
1) Streamlit UI (default if Streamlit is installed and --cli is not used)
2) CLI fallback (if Streamlit is missing OR --cli flag is provided)

Run:
- python app.py
- python app.py --cli
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from rag.retrieve import retrieve_top_k

try:
    import streamlit as st

    STREAMLIT_AVAILABLE = True
except Exception:
    st = None  # type: ignore
    STREAMLIT_AVAILABLE = False


SYSTEM_PROMPT = (
    "You are AgriSathi, a beginner-safe agriculture assistant.\n"
    "Output plain text only.\n\n"
    "Core safety rules:\n"
    "- Never provide exact chemical dosage, concentration, or tank-mix instructions.\n"
    "- Never provide step-by-step pesticide mixing procedures.\n"
    "- If asked for chemical specifics, provide only general safety-first guidance.\n"
    "- Always include: check the product label and consult a local agronomist.\n\n"
    "Required answer format:\n"
    "Problem summary:\n"
    "Action steps or Follow-up questions:\n"
    "Safety disclaimer:\n"
    "When to consult expert:\n\n"
    "If key details are missing (crop/stage/location/symptoms), ask 2-3 follow-up questions first."
)

DEFAULT_MODEL = os.getenv("MODEL_NAME", "distilgpt2")
LOG_PATH = Path("data/product_logs.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AgriSathi demo app")
    parser.add_argument("--cli", action="store_true", help="Force CLI mode.")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--adapter_path", type=str, default="")
    parser.add_argument("--rag", action="store_true", help="Enable RAG in CLI mode.")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--crop", type=str, default="tomato")
    parser.add_argument("--stage", type=str, default="vegetative")
    parser.add_argument("--location", type=str, default="")
    parser.add_argument("--symptoms", type=str, default="")
    return parser.parse_args()


def _is_running_inside_streamlit() -> bool:
    if not STREAMLIT_AVAILABLE:
        return False
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def _auto_launch_streamlit_if_needed() -> None:
    """
    Supports `python app.py` startup:
    if script is not already in Streamlit runtime, launch streamlit run.
    """
    if _is_running_inside_streamlit():
        return
    cmd = [sys.executable, "-m", "streamlit", "run", __file__]
    raise SystemExit(subprocess.call(cmd))


def load_generator(model_name: str, adapter_path: str = ""):
    """Load tokenizer + base model + optional adapter, then build generation pipeline."""
    use_gpu = torch.cuda.is_available()
    device = 0 if use_gpu else -1

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    if adapter_path.strip():
        model = PeftModel.from_pretrained(base_model, adapter_path.strip())
    else:
        model = base_model

    return pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
    )


def build_context_json(crop: str, stage: str, location: str, symptoms: str) -> Dict:
    ctx: Dict[str, str] = {"crop": crop, "stage": stage}
    if location.strip():
        ctx["location"] = location.strip()
    if symptoms.strip():
        ctx["symptoms"] = symptoms.strip()
    return ctx


def build_prompt(user_question: str, context: Dict, rag_chunks: List[Dict]) -> str:
    ctx_json = json.dumps(context, ensure_ascii=False)
    if rag_chunks:
        source_blocks = []
        for i, ch in enumerate(rag_chunks, start=1):
            source_blocks.append(f"[Source {i}] {ch.get('title', 'untitled')}\n{ch.get('text', '')}")
        rag_text = "\n\n".join(source_blocks)
    else:
        rag_text = "No retrieved context."

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"CONTEXT:\n{ctx_json}\n\n"
        "RETRIEVED SOURCES (use as evidence when available):\n"
        f"{rag_text}\n\n"
        f"USER QUESTION:\n{user_question.strip()}\n\n"
        "ASSISTANT:\n"
    )


def generate_answer(gen, prompt: str) -> str:
    eos_id = gen.tokenizer.eos_token_id
    out = gen(
        prompt,
        max_new_tokens=220,
        do_sample=True,
        temperature=0.3,
        top_p=0.9,
        num_return_sequences=1,
        eos_token_id=eos_id,
        pad_token_id=eos_id,
    )[0]["generated_text"]
    if out.startswith(prompt):
        return out[len(prompt) :].strip()
    return out.strip()


def append_log(record: Dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_cli(args: argparse.Namespace) -> None:
    """
    CLI fallback mode.
    Toggles use defaults: RAG off unless --rag is passed; top_k default 3.
    """
    print("AgriSathi CLI mode")
    print("Type 'exit' to quit.")
    print("Safety: always check product labels and consult local agronomist.\n")

    gen = load_generator(model_name=args.model_name, adapter_path=args.adapter_path)
    context = build_context_json(
        crop=args.crop,
        stage=args.stage,
        location=args.location,
        symptoms=args.symptoms,
    )

    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("AgriSathi: Goodbye!")
            break
        if not question:
            print("AgriSathi: Please enter a question.")
            continue

        rag_chunks: List[Dict] = []
        if args.rag:
            rag_chunks = retrieve_top_k(
                query=question,
                top_k=max(1, min(6, int(args.top_k))),
                index_path="rag_index.faiss",
                meta_path="rag_index_meta.json",
            )

        prompt = build_prompt(question, context=context, rag_chunks=rag_chunks)
        answer = generate_answer(gen, prompt)

        print("\n" + answer + "\n")
        if rag_chunks:
            print("Sources used:")
            for i, ch in enumerate(rag_chunks, start=1):
                print(f"{i}. {ch.get('title', 'untitled')} ({ch.get('source_path', '')})")
            print("")

        log_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": "cli",
            "model_name": args.model_name,
            "adapter_path": args.adapter_path,
            "question": question,
            "context": context,
            "rag_enabled": bool(args.rag),
            "top_k": int(args.top_k),
            "sources": [c.get("title", "") for c in rag_chunks],
            "prompt_used": prompt,
            "answer": answer,
        }
        append_log(log_record)


def run_streamlit_ui() -> None:
    """Streamlit UI mode."""
    assert st is not None  # for type checkers

    st.set_page_config(page_title="AgriSathi Demo", page_icon=":seedling:", layout="wide")
    st.title("AgriSathi - Mini Product Demo")
    st.warning(
        "Safety first: AgriSathi gives informational guidance only. "
        "For chemical use, always check product labels and consult a local agronomist."
    )

    with st.sidebar:
        st.header("Settings")
        language = st.selectbox("Language", ["English", "Marathi (Coming soon)"], index=0)
        if language != "English":
            st.info("Marathi support is coming soon. Switching to English for now.")

        crop = st.selectbox("Crop", ["cotton", "soybean", "tomato", "brinjal", "wheat", "chili"])
        stage = st.selectbox("Stage", ["sowing", "vegetative", "flowering", "fruiting"])
        location = st.text_input("Location (optional)", placeholder="e.g., Nashik, Maharashtra")
        symptoms = st.text_area("Symptoms (optional)", placeholder="Describe symptoms you observe...")

        uploaded_img = st.file_uploader("Attach photo", type=["jpg", "jpeg", "png"])
        if uploaded_img is not None:
            st.info("Coming soon: disease detection")

        rag_toggle = st.toggle("RAG", value=False)
        top_k = st.slider("top_k sources", min_value=1, max_value=6, value=3, step=1)
        adapter_path = st.text_input("LoRA adapter path (optional)", value="")
        model_name = st.text_input("Base model", value=DEFAULT_MODEL)

    st.subheader("Ask AgriSathi")
    user_question = st.text_area("Question", placeholder="Ask your farming question...")
    run_btn = st.button("Get advice", type="primary")
    if not run_btn:
        return
    if not user_question.strip():
        st.error("Please enter a question first.")
        return

    with st.spinner("Loading model..."):
        try:
            gen = load_generator(model_name=model_name.strip(), adapter_path=adapter_path.strip())
        except Exception as e:
            st.error(f"Model loading failed: {e}")
            return

    context = build_context_json(crop=crop, stage=stage, location=location, symptoms=symptoms)
    rag_chunks: List[Dict] = []
    if rag_toggle:
        with st.spinner("Retrieving trusted sources..."):
            rag_chunks = retrieve_top_k(
                query=user_question,
                top_k=top_k,
                index_path="rag_index.faiss",
                meta_path="rag_index_meta.json",
            )

    prompt = build_prompt(user_question=user_question, context=context, rag_chunks=rag_chunks)
    with st.spinner("Generating advice..."):
        answer = generate_answer(gen, prompt)

    st.markdown("### Advice")
    st.text(answer)
    st.markdown("### Sources used")
    if rag_chunks:
        for i, ch in enumerate(rag_chunks, start=1):
            st.markdown(f"{i}. **{ch.get('title', 'untitled')}**  \n`{ch.get('source_path', '')}`")
    else:
        st.caption("No sources used.")

    log_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "streamlit",
        "model_name": model_name.strip(),
        "adapter_path": adapter_path.strip(),
        "question": user_question.strip(),
        "context": context,
        "rag_enabled": bool(rag_toggle),
        "top_k": int(top_k),
        "sources": [c.get("title", "") for c in rag_chunks],
        "prompt_used": prompt,
        "answer": answer,
    }
    append_log(log_record)
    st.success("Interaction saved to data/product_logs.jsonl")


if __name__ == "__main__":
    args = parse_args()

    # CLI fallback conditions:
    # 1) user explicitly asked for CLI via --cli
    # 2) Streamlit package is not available
    if args.cli or not STREAMLIT_AVAILABLE:
        run_cli(args)
    else:
        # If launched with plain python, auto-start Streamlit.
        _auto_launch_streamlit_if_needed()
        run_streamlit_ui()

