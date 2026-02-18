# AgriSathi Mini AI Model

RAG-powered agriculture advisory assistant using Hugging Face, FAISS, and Streamlit.

## Live Demo
- Hugging Face Space: https://huggingface.co/spaces/Janhvi1008/agrisathi-mini

## Description
AgriSathi Mini is a practical AI assistant for farmers and agri-learners. It combines retrieval-augmented generation (RAG) with structured safety-first responses to reduce hallucinations and improve grounded advice quality.

## Features
- RAG pipeline with FAISS vector search over trusted local documents
- ICAR advisory ingestion (PDF/TXT/MD) for domain-grounded responses
- Marathi input support (English answer generation currently)
- Streamlit-based UI with copy-answer flow and source display
- Hugging Face Spaces deployment support (Docker)

## Folder Structure
```text
.
|-- app.py                          # Streamlit entrypoint (HF Spaces)
|-- requirements.txt                # Runtime dependencies
|-- Dockerfile                      # Docker launch config for Spaces
|-- LICENSE                         # MIT license
|-- DEPLOY.md                       # Deployment notes
|-- data/
|   |-- docs/                       # Source docs for RAG ingestion
|   |-- train_data.jsonl            # Training dataset (optional)
|   |-- eval_data.jsonl             # Eval dataset (optional)
|   |-- teacher_answers.jsonl       # Teacher outputs (optional)
|   |-- product_logs.jsonl          # Runtime logs (ignored)
|-- rag/
|   |-- build_index.py              # Builds FAISS index from data/docs/
|   |-- retrieve.py                 # Top-k chunk retrieval utility
|   |-- rag_index.faiss             # Vector index
|   |-- rag_chunks.jsonl            # Chunk metadata
|-- train/
|   |-- lora_sft.py                 # LoRA SFT scripts
|   |-- distill_*.py                # Distillation scripts
|   |-- preference_*.py             # Preference training/eval scripts
```

## How It Works (RAG)
1. Put trusted documents in `data/docs/`.
2. Run index build:
   - `python rag/build_index.py`
3. The app retrieves top-k relevant chunks from FAISS.
4. Retrieved chunks are injected into the prompt.
5. Model returns structured advisory output with safety sections.

## Local Run
Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```
