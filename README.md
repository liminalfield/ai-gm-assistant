# RPG Rules Assistant

This project is an AI-enhanced rules assistant for tabletop RPGs. It enables players and game masters to semantically search through RPG PDFs and ask AI-powered questions about the rules. The assistant uses ChromaDB for document retrieval and a quantized LLaMA-2 model (run locally) for natural language understanding.

---

## 🚀 Features

- 🔍 **Semantic Search**: Query embedded RPG rulebooks using ChromaDB and SentenceTransformers.
- 🤖 **AI-Powered Responses**: Get contextual answers grounded in actual rulebooks using a local LLaMA-2 model.
- 📄 **PDF Ingestion**: Converts, chunks, and embeds PDFs using Markdown conversion and sentence embeddings.
- 🧠 **Local LLM Integration**: Hugging Face Transformers with 4-bit quantized LLaMA-2.
- 🌐 **FastAPI Backend**: Clean, modular Python API for integration and use.
- 🧪 **Makefile Convenience**: Includes Makefile shortcuts for local development and processing.

---

## 🧱 Tech Stack

| Component         | Tech                                        |
|------------------|---------------------------------------------|
| API Framework     | FastAPI                                     |
| Embedding Model   | `sentence-transformers/all-MiniLM-L6-v2`    |
| Vector DB         | ChromaDB (persistent local storage)         |
| LLM               | LLaMA-2-13B-chat (4-bit, Hugging Face)      |
| Markdown Parsing  | `markitdown` (Microsoft PDF converter)      |

---

## 🗂 Folder Structure

```
project-root/
├── ai_gm_assistant/       # Core logic for PDF processing, retrieval, and LLM interface
│   ├── hf_integration.py  # Loads and queries LLaMA-2 model
│   ├── main.py            # FastAPI entrypoint
│   ├── retriever.py       # ChromaDB and embedding logic
│   └── process_pdfs.py    # PDF ingestion, markdown conversion, chunking
├── data/
│   ├── pdfs/              # Place your RPG PDFs here
│   └── rpg_sources_db/    # ChromaDB persistent storage
├── env.clean              # Sample environment variables (rename to `.env`)
├── Makefile               # Developer shortcuts
├── requirements.txt
└── README.md
```

---

## 🔑 Setup Instructions

### 1. Environment Variables

Copy `env.clean` to `.env` before running anything:

```bash
cp env.clean .env
```

Ensure the `.env` includes values like:

- `CHROMA_INDEX_PATH=./data/rpg_sources_db`
- `EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2`
- `RETRIEVAL_RESULTS=8`
- `ALLOWED_ORIGINS=http://localhost:5173`

### 2. Hugging Face Authentication

If you're using gated models (like `meta-llama/Llama-2-13b-chat-hf`), you **must be authenticated with Hugging Face**. There are two options:

- Login via CLI:
  ```bash
  huggingface-cli login
  ```
- Or set the environment variable in your `.env`:
  ```env
  HF_TOKEN=your_huggingface_token
  ```

The code expects access to Hugging Face via `use_auth_token=True`.

---


## 🧪 API Endpoints

| Method | Endpoint         | Description                         |
|--------|------------------|-------------------------------------|
| GET    | `/search`        | Returns semantic matches from ChromaDB |
| GET    | `/ai-search`     | Retrieves + answers with context-grounded LLM |
| GET    | `/ai-direct`     | Sends query to the LLM with no external context |

---

## 💬 Example Queries

- "How does weather affect overland travel?"
- "What are the rules for grappling?"
- "Can you flank an enemy in difficult terrain?"

---

## 📜 License

Apache 2.0

---

## 🙌 Credits

Built by [Su Soloway](https://www.linkedin.com/in/...) and [Oluf Andrews](https://www.linkedin.com/in/...) using FastAPI, ChromaDB, and Hugging Face Transformers.