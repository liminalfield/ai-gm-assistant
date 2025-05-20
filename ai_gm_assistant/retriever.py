import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv


load_dotenv()


logger = logging.getLogger(__name__)

DEFAULT_N_RESULTS = int(os.getenv("DEFAULT_N_RESULTS", 5))
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rpg_sources")

base_dir = Path(__file__).parent.absolute()
CHROMA_DB_PATH = os.getenv("CHROMA_INDEX_PATH", str(base_dir / "../data/rpg_sources_db"))


class Retriever:
    def __init__(
            self,
            db_path: Optional[str] = None,
            collection_name: str = COLLECTION_NAME,
            embedding_model_name: str = EMBEDDING_MODEL_NAME
    ) -> None:
        """
        Initialize the retriever with ChromaDB and embedding model.

        Args:
            db_path: Path to ChromaDB database. If None, uses default path
            collection_name: Name of the ChromaDB collection to use
            embedding_model_name: Name of the sentence transformer model for embeddings
        """
        try:
            if db_path is None:
                db_path = CHROMA_DB_PATH

            logger.info(f"Initializing ChromaDB with path: {db_path}")
            self.client = chromadb.PersistentClient(path=db_path)
            self.collection = self.client.get_or_create_collection(collection_name)

            logger.info(f"Loading embedding model: {embedding_model_name}")
            self.embedding_model = SentenceTransformer(embedding_model_name)

            logger.info("Retriever initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Retriever: {e}", exc_info=True)
            raise

    def search(self, query: str, n_results: int = DEFAULT_N_RESULTS) -> Dict[str, List[Any]]:
        """
        Search for documents related to the query.

        Args:
            query: The search query string
            n_results: Number of results to return

        Returns:
            Dictionary containing documents and their metadata
        """
        if not query.strip():
            logger.warning("Empty query received")
            return {"documents": [], "metadatas": []}

        try:
            logger.info(f"Searching for: '{query}' with {n_results} results")
            query_embedding = self.embedding_model.encode(query)

            search_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas"]
            )

            # Handle case where no results found
            if not search_results["documents"] or len(search_results["documents"][0]) == 0:
                logger.info("No search results found")
                return {"documents": [], "metadatas": []}

            logger.info(f"Found {len(search_results['documents'][0])} results")
            return {
                "documents": search_results["documents"][0],
                "metadatas": search_results["metadatas"][0]
            }
        except Exception as e:
            logger.error(f"Error during search: {e}", exc_info=True)
            return {"documents": [], "metadatas": []}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    retriever = Retriever()
    test_query = "How do critical hits work?"
    results = retriever.search(test_query)
    logger.info(f"Found {len(results['documents'])} documents")