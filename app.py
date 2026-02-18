"""
AgriSathi Mini MVP - Streamlit app (Hugging Face Spaces compatible).

Notes for Spaces:
- Streamlit SDK runs this file directly.
- No subprocess auto-launch logic is used.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import streamlit as st
import streamlit.components.v1 as components
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from rag.retrieve import retrieve_top_k


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
RAG_INDEX_PATH = Path("rag/rag_index.faiss")
RAG_CHUNKS_PATH = Path("rag/rag_chunks.jsonl")


@st.cache_resource(show_spinner=False)
def load_generator(model_name: str, adapter_path: str = ""):
    """Load tokenizer + base model + optional LoRA adapter."""
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


def render_copy_answer_button(answer: str) -> None:
    """Copy-to-clipboard button using browser navigator API."""
    safe_answer_js = json.dumps(answer)
    components.html(
        f"""
        <div>
          <button id="copyAnswerBtn" style="padding:8px 12px; border-radius:8px; border:1px solid #ccc; cursor:pointer;">
            Copy answer
          </button>
          <span id="copyStatus" style="margin-left:10px; font-size:13px;"></span>
        </div>
        <script>
          const btn = document.getElementById("copyAnswerBtn");
          const status = document.getElementById("copyStatus");
          btn.onclick = async () => {{
            try {{
              await navigator.clipboard.writeText({safe_answer_js});
              status.textContent = "Copied to clipboard.";
              status.style.color = "green";
            }} catch (e) {{
              status.textContent = "Copy failed.";
              status.style.color = "red";
            }}
          }};
        </script>
        """,
        height=55,
    )


def rag_assets_ready() -> bool:
    return RAG_INDEX_PATH.exists() and RAG_CHUNKS_PATH.exists()


def main() -> None:
    st.set_page_config(page_title="AgriSathi Demo", page_icon=":seedling:", layout="wide")
    st.title("AgriSathi - Mini Product Demo")
    st.warning(
        "Safety first: AgriSathi gives informational guidance only. "
        "For chemical use, always check product labels and consult a local agronomist."
    )
    if rag_assets_ready():
        st.caption("RAG index: FOUND ✅")
    else:
        st.caption("RAG index: MISSING ❌")

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
    english_question = st.text_area("Question (English)", placeholder="Ask your farming question...")
    marathi_question = st.text_area("Question (Marathi)", placeholder="Write Marathi question (optional)...")
    use_marathi_question = st.toggle("Use Marathi question", value=False)
    if use_marathi_question:
        st.info("Marathi answering coming soon. We will still generate an English answer.")

    run_btn = st.button("Get advice", type="primary")
    if not run_btn:
        return

    selected_question = marathi_question.strip() if use_marathi_question else english_question.strip()
    if not selected_question:
        st.error("Please enter a question in the selected input.")
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
        if not rag_assets_ready():
            st.warning("RAG is ON but index files are missing (`rag/rag_index.faiss`, `rag/rag_chunks.jsonl`).")
            return
        with st.spinner("Retrieving trusted sources..."):
            # Retrieve from required Spaces paths.
            rag_chunks = retrieve_top_k(
                query=selected_question,
                top_k=top_k,
                index_path=str(RAG_INDEX_PATH),
                meta_path=str(RAG_CHUNKS_PATH),
            )

    prompt = build_prompt(user_question=selected_question, context=context, rag_chunks=rag_chunks)
    with st.spinner("Generating advice..."):
        answer = generate_answer(gen, prompt)

    st.markdown("### Advice")
    st.text(answer)
    render_copy_answer_button(answer)

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
        "question": selected_question,
        "english_question": english_question.strip(),
        "marathi_question": marathi_question.strip(),
        "used_marathi_question": bool(use_marathi_question),
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
    main()
