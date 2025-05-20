# AI GM Assistant

## Project Overview

**AI GM Assistant** is an AI-enhanced rules assistant for tabletop role-playing games. It allows players and game masters to **semantically search** their RPG rulebook PDFs and ask natural-language questions about the rules, getting answers that are grounded in the actual rule text. The assistant solves the problem of flipping through rulebooks by using AI to quickly retrieve relevant rules and explain them. It uses a local vector database (ChromaDB) to index the rulebooks and a quantized LLaMA-2 13B chat model to generate answers, so everything runs locally without requiring an internet connection. In short, AI GM Assistant can serve as your personal rules reference librarian, fetching exact rules and providing clear answers during gameplay.

**Key Features:**
- **Semantic Search** – Find relevant rules by meaning, not just keywords.
- **Contextual AI Answers** – Get AI-generated answers that are grounded in the rulebooks.
- **PDF Ingestion Pipeline** – Easily add your RPG PDFs.
- **Web Interface** – A React-based frontend for a user-friendly Q&A experience.
- **Fully Local LLM** – Runs a quantized LLaMA-2 model (4-bit).
- **FastAPI Backend** – Lightweight server with clean REST endpoints.

## How It Works

This project uses Retrieval-Augmented Generation (RAG) to combine semantic document search with LLM-based answer generation. PDF rulebooks are ingested and embedded into a ChromaDB vector store. When a user asks a question, relevant text chunks are retrieved and passed to an LLM to generate a grounded answer.

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js v18+
- Hugging Face account (for LLaMA-2 download)

### Backend Setup

```bash
git clone https://github.com/liminalfield/ai-gm-assistant.git
cd ai-gm-assistant
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt
cp env.clean .env
huggingface-cli login
python ai_gm_assistant/process_pdfs.py
make run
```

### Frontend Setup

```bash
cd webapp
npm install
npm run dev
```

Navigate to `http://localhost:5173` in your browser.

## Usage

- **Basic Search**: View matching rulebook passages.
- **AI Search**: AI answers based on retrieved rule text.
- **Direct AI**: AI answers from model knowledge only.

## API Endpoints

| Method | Endpoint     | Description                        |
|--------|--------------|------------------------------------|
| GET    | /search      | Semantic document search           |
| GET    | /ai-search   | RAG: Retrieval-Augmented Answering |
| GET    | /ai-direct   | Direct LLM query                   |

## Tech Stack

- **Frontend**: React + TypeScript + Vite
- **Backend**: FastAPI + Uvicorn
- **LLM**: LLaMA-2 13B Chat (4-bit, HF Transformers)
- **Vector DB**: ChromaDB
- **Embeddings**: all-MiniLM-L6-v2
- **PDF Processing**: markitdown + LangChain

## License

Apache 2.0. Contributions must be made under the same license.

---

Enjoy building with AI GM Assistant!