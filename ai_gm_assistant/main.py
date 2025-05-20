import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from ai_gm_assistant.retriever import Retriever
from ai_gm_assistant.llm_integration import LLMIntegrationHF as LLMIntegration

load_dotenv()
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
NUM_RESULTS = int(os.getenv("RETRIEVAL_RESULTS", 8))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

retriever = Retriever()
llm = LLMIntegration()

@app.get("/")
def read_root() -> dict:
    return {"message": "FastAPI is running!"}

@app.get("/search")
def search(query: str = Query(..., min_length=1, max_length=500)) -> dict:
    """
    Search endpoint that retrieves relevant documents from ChromaDB based on the query.

    This endpoint performs a semantic search using the query string and returns matching
    documents from the ChromaDB vector database without any AI processing.

    Args:
        query: The search query string

    Returns:
        A dictionary with "response" key containing a list of document objects,
        each with text content, source filename, and chunk index.
    """
    try:
        search_results = retriever.search(query)

        documents = search_results["documents"]
        metadatas = search_results.get("metadatas", [{}] * len(documents))

        logging.info(f"DOCUMENTS: {documents}")
        logging.info(f"METADATAS: {metadatas}")

        combined = [
            {
                "text": doc,
                "source": os.path.basename(meta.get("source", "unknown.pdf")),
                "chunk_index": meta.get("chunk_index", 0)
            }
            for doc, meta in zip(documents, metadatas)
        ]

        return {"response": combined}
    except Exception as e:
        logging.error(f"Error during search: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.get("/ai-search")
def ai_search(query: str) -> dict:
    """
    AI-enhanced search endpoint that combines ChromaDB retrieval with LLM processing.

    This endpoint:
    1. Retrieves relevant documents from ChromaDB using semantic search
    2. Formats documents with their source metadata
    3. Passes both the query and retrieved context to a Llama-2 model
    4. Returns an AI-generated response that extracts and presents RPG rules from the context

    Args:
        query: The search query string

    Returns:
        A dictionary with "response" key containing the AI-generated answer
    """
    try:
        search_results = retriever.search(query, n_results=NUM_RESULTS)
        docs = search_results["documents"]
        metas = search_results.get("metadatas", [{}] * len(docs))
        lines = []
        for doc, meta in zip(docs, metas):
            filename = meta.get("source", "unknown.pdf").split("/")[-1]
            idx = meta.get("chunk_index", 0)
            lines.append(f"{filename} [chunk {idx}]: {doc}")
        context = "\n\n".join(lines)
        ai_response = llm.generate_response(query, context)
        return {"response": ai_response}
    except Exception as e:
        logging.error(f"AI Search failed: {e}")
        raise HTTPException(status_code=500, detail="AI search error")


@app.get("/ai-direct")
def ai_direct(query: str) -> dict:
    """
    Direct AI query endpoint that bypasses the retrieval step.

    This endpoint sends the user query directly to the Llama-2 model without retrieving
    any context from ChromaDB. It provides a general RPG rules assistant response
    based solely on the model's pre-trained knowledge.

    Args:
        query: The question to ask the AI directly

    Returns:
        A dictionary with "response" key containing the AI-generated answer
    """
    try:
        ai_response = llm.generate_direct(query)
        return {"response": ai_response}
    except Exception as e:
        logging.error(f"AI Direct failed: {e}")
        raise HTTPException(status_code=500, detail="AI direct error")