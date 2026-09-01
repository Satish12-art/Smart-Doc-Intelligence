import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
INDEX_DIR = DATA_DIR / "indices"

# Create directories if they don't exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# API Keys (Loaded from environment variables)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY", "")

# Vector Database (Qdrant) Configuration
# By default, use local file storage for Qdrant so Docker is not strictly required.
# If QDRANT_HOST is set (e.g., 'localhost'), it will connect to that container.
QDRANT_HOST = os.environ.get("QDRANT_HOST", "")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "smart_documents")

# Fallback: Qdrant local path if no host provided
QDRANT_LOCAL_PATH = str(DATA_DIR / "qdrant_local")

# RAG Hyperparameters
CHUNK_SIZE = 500  # Characters (approx 100-150 words)
CHUNK_OVERLAP = 100  # Characters
MAX_TOP_K = 20  # Max results retrieved from each index before hybrid fusion
FINAL_TOP_K = 5  # Final number of chunks passed to the LLM
RRF_CONSTANT = 60  # Constant for Reciprocal Rank Fusion

# Embedding Configuration
EMBEDDING_MODEL = "models/text-embedding-004"
LLM_MODEL = "gemini-3.6-flash"  # Default Gemini Model
