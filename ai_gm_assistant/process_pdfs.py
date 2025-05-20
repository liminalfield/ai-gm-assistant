import os
import json
import hashlib
import logging
from typing import Dict, List, Optional
from pathlib import Path
from markitdown import MarkItDown
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
DEFAULT_N_RESULTS = int(os.getenv("DEFAULT_N_RESULTS", 5))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rpg_sources")

base_dir = Path(__file__).parent.absolute()
CHROMA_DB_PATH = os.getenv("CHROMA_INDEX_PATH", str(base_dir / "../data/rpg_sources_db"))


logger = logging.getLogger(__name__)


class PDFProcessor:
    def __init__(
            self,
            base_dir: Optional[str] = None,
            chunk_size: int = CHUNK_SIZE,
            chunk_overlap: int = CHUNK_OVERLAP,
            embedding_model_name: str = EMBEDDING_MODEL_NAME,


    ) -> None:
        """
        Initialize PDF processor with ChromaDB and embedding model.

        Args:
            base_dir: Base directory for data storage. Defaults to script location
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between chunks for context continuity
            embedding_model_name: Name of the sentence transformer model
        """
        self.base_dir = Path(base_dir or os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = self.base_dir.parent / "data"

        # Setup paths
        self.chromadb_path = Path(CHROMA_DB_PATH).resolve()
        self.hash_file_path = self.data_dir / "processed_files.json"
        self.pdf_store = self.data_dir / "pdfs"

        # Initialize components
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.collection_name = "rpg_sources"

        self._setup_storage()
        self._initialize_components(embedding_model_name)

    def _setup_storage(self) -> None:
        """Create necessary directories and files."""
        self.pdf_store.mkdir(parents=True, exist_ok=True)
        if not self.hash_file_path.exists():
            self.hash_file_path.write_text("{}")

    def _initialize_components(self, embedding_model_name: str) -> None:
        """Initialize ChromaDB, embedding model, and markdown converter."""
        try:
            self.client = chromadb.PersistentClient(path=str(self.chromadb_path))
            self.collection = self.client.get_or_create_collection(self.collection_name)
            self.embedding_model = SentenceTransformer(embedding_model_name)
            self.markdown_converter = MarkItDown()
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            logger.info("Components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise

    def process_pdfs(self) -> None:
        """Process all PDFs in the store directory."""
        processed_files = self._load_processed_files()

        for pdf_file in self.pdf_store.glob("*.pdf"):
            file_hash = self._compute_file_hash(pdf_file)

            if processed_files.get(str(pdf_file)) != file_hash:
                logger.info(f"Processing: {pdf_file.name}")
                self._process_single_pdf(pdf_file)
                processed_files[str(pdf_file)] = file_hash
            else:
                logger.info(f"Skipping unchanged file: {pdf_file.name}")

        self._save_processed_files(processed_files)

    def _process_single_pdf(self, pdf_path: Path) -> None:
        """Process a single PDF file and store its chunks in ChromaDB."""
        try:
            # Convert PDF to markdown and chunk it
            result = self.markdown_converter.convert(str(pdf_path))
            chunks = self._chunk_text(result.text_content)

            # Prepare data for ChromaDB
            embeddings = self.embedding_model.encode(chunks)
            ids = [f"{pdf_path.name}_chunk_{i}" for i in range(len(chunks))]
            metadatas = [{"source": pdf_path.name, "chunk_index": i} for i in range(len(chunks))]

            # Store in ChromaDB
            self.collection.add(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas
            )
            logger.info(f"Stored {len(chunks)} chunks from {pdf_path.name}")
        except Exception as e:
            logger.error(f"Failed to process {pdf_path.name}: {e}")
            raise

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks using the text splitter."""
        chunks = self.text_splitter.split_text(text)
        return [chunk.strip() for chunk in chunks if chunk.strip()]

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute MD5 hash of a file."""
        return hashlib.md5(file_path.read_bytes()).hexdigest()

    def _load_processed_files(self) -> Dict[str, str]:
        """Load the processed files tracking data."""
        return json.loads(self.hash_file_path.read_text())

    def _save_processed_files(self, data: Dict[str, str]) -> None:
        """Save the processed files tracking data."""
        self.hash_file_path.write_text(json.dumps(data))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    processor = PDFProcessor()
    processor.process_pdfs()