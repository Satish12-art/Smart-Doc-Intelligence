import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from rank_bm25 import BM25Okapi

from backend.config import (
    GEMINI_API_KEY, QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION, QDRANT_LOCAL_PATH,
    INDEX_DIR, EMBEDDING_MODEL, MAX_TOP_K, FINAL_TOP_K, RRF_CONSTANT
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchEngine:
    def __init__(self):
        # Configure Gemini
        self.local_embedder = None  # Always initialize for fallback use
        if GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
            self.use_gemini = True
            self.vector_dim = 768  # text-embedding-004 dimension
            logger.info("Configured search engine to use Google Gemini embeddings.")
        else:
            self.use_gemini = False
            self.vector_dim = 384  # sentence-transformers MiniLM dimension
            logger.warning("GEMINI_API_KEY not found. Using local sentence-transformers fallback.")

        # Initialize Qdrant Client
        if QDRANT_HOST:
            logger.info(f"Connecting to Qdrant server at {QDRANT_HOST}:{QDRANT_PORT}")
            self.qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        else:
            logger.info(f"Connecting to local Qdrant instance at {QDRANT_LOCAL_PATH}")
            self.qdrant_client = QdrantClient(path=QDRANT_LOCAL_PATH)

        # Initialize collection
        self._init_qdrant_collection()

        # BM25 setup
        self.bm25_data_path = INDEX_DIR / "bm25_index.json"
        self.chunks_cache: List[Dict[str, Any]] = []
        self.bm25_index: BM25Okapi = None
        self._load_bm25_index()

    def _init_qdrant_collection(self):
        try:
            collections = self.qdrant_client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if QDRANT_COLLECTION in collection_names:
                # Check if existing collection has a matching vector dimension
                collection_info = self.qdrant_client.get_collection(QDRANT_COLLECTION)
                existing_dim = collection_info.config.params.vectors.size
                if existing_dim != self.vector_dim:
                    logger.warning(
                        f"Qdrant collection '{QDRANT_COLLECTION}' exists with dimension {existing_dim}, "
                        f"but current embedding provider requires {self.vector_dim}. "
                        f"Recreating collection (existing data will be lost)."
                    )
                    self.qdrant_client.delete_collection(QDRANT_COLLECTION)
                    self.qdrant_client.create_collection(
                        collection_name=QDRANT_COLLECTION,
                        vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE)
                    )
                    # Also clear the BM25 cache to stay in sync
                    bm25_path = INDEX_DIR / "bm25_index.json"
                    if bm25_path.exists():
                        bm25_path.unlink()
                        logger.info("Cleared BM25 index to stay in sync with recreated Qdrant collection.")
                else:
                    logger.info(f"Qdrant collection '{QDRANT_COLLECTION}' exists with matching dimension ({existing_dim}).")
            else:
                logger.info(f"Creating Qdrant collection: {QDRANT_COLLECTION} (dim={self.vector_dim})")
                self.qdrant_client.create_collection(
                    collection_name=QDRANT_COLLECTION,
                    vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE)
                )
        except Exception as e:
            logger.error(f"Error initializing Qdrant collection: {e}")

    def _get_local_embedder(self):
        if self.local_embedder is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading local sentence-transformers model (all-MiniLM-L6-v2)...")
            self.local_embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return self.local_embedder

    def get_embedding(self, text: str, is_query: bool = False) -> List[float]:
        """
        Generates embedding using Gemini or falls back to local SentenceTransformer.
        """
        if self.use_gemini:
            try:
                task_type = "retrieval_query" if is_query else "retrieval_document"
                response = genai.embed_content(
                    model=EMBEDDING_MODEL,
                    content=text,
                    task_type=task_type
                )
                return response["embedding"]
            except Exception as e:
                logger.error(f"Gemini embedding error: {e}. Switching to local embeddings permanently for this session.")
                # Permanently switch to local mode for this session
                self.use_gemini = False
                self.vector_dim = 384
                self._init_qdrant_collection()  # Will detect mismatch and recreate
                self._load_bm25_index()
                return self._get_local_embedding_fallback(text)
        else:
            return self._get_local_embedding_fallback(text)

    def _get_local_embedding_fallback(self, text: str) -> List[float]:
        embedder = self._get_local_embedder()
        emb = embedder.encode(text)
        return emb.tolist()

    # --- BM25 Keyword Search Methods ---

    def _tokenize(self, text: str) -> List[str]:
        # Simple tokenization: lowercase and extract words
        return re.findall(r'\w+', text.lower())

    def _load_bm25_index(self):
        """Loads cached chunks and initializes BM25 index."""
        if self.bm25_data_path.exists():
            try:
                with open(self.bm25_data_path, "r", encoding="utf-8") as f:
                    self.chunks_cache = json.load(f)
                
                if self.chunks_cache:
                    corpus = [self._tokenize(chunk["text"]) for chunk in self.chunks_cache]
                    self.bm25_index = BM25Okapi(corpus)
                    logger.info(f"Initialized BM25 with {len(self.chunks_cache)} chunks.")
                else:
                    self.bm25_index = None
            except Exception as e:
                logger.error(f"Error loading BM25 index: {e}")
                self.bm25_index = None
        else:
            self.chunks_cache = []
            self.bm25_index = None

    def _save_bm25_index(self):
        """Saves current chunks cache to disk."""
        try:
            with open(self.bm25_data_path, "w", encoding="utf-8") as f:
                json.dump(self.chunks_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving BM25 index: {e}")

    # --- Indexing and Deletion ---

    def add_documents(self, chunks: List[Dict[str, Any]]):
        """
        Adds multiple chunks to both Qdrant Vector DB and local BM25.
        """
        if not chunks:
            return

        logger.info(f"Indexing {len(chunks)} chunks...")
        
        # 1. Add to Vector DB (Qdrant)
        points = []
        for idx, chunk in enumerate(chunks):
            vector = self.get_embedding(chunk["text"], is_query=False)
            points.append(PointStruct(
                id=chunk["id"],
                vector=vector,
                payload={
                    "text": chunk["text"],
                    "source": chunk["metadata"]["source"],
                    "page": chunk["metadata"]["page"],
                    "chunk_idx": chunk["metadata"]["chunk_idx"]
                }
            ))
            
        self.qdrant_client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=points
        )
        logger.info("Successfully uploaded vectors to Qdrant.")

        # 2. Add to BM25 Cache
        # Filter out existing chunk IDs just in case
        existing_ids = {c["id"] for c in self.chunks_cache}
        new_chunks = [c for c in chunks if c["id"] not in existing_ids]
        self.chunks_cache.extend(new_chunks)
        self._save_bm25_index()
        self._load_bm25_index() # Re-initialize model
        logger.info("Successfully updated BM25 index.")

    def delete_document(self, filename: str):
        """
        Deletes all chunks associated with a specific file from Qdrant and BM25.
        """
        # 1. Delete from Qdrant
        self.qdrant_client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="source",
                        match=MatchValue(value=filename)
                    )
                ]
            )
        )
        logger.info(f"Deleted vectors for '{filename}' from Qdrant.")

        # 2. Delete from BM25 Cache
        self.chunks_cache = [c for c in self.chunks_cache if c["metadata"]["source"] != filename]
        self._save_bm25_index()
        self._load_bm25_index()
        logger.info(f"Deleted chunks for '{filename}' from BM25 index.")

    def get_all_indexed_documents(self) -> List[str]:
        """Returns unique filenames of all indexed documents."""
        sources = {c["metadata"]["source"] for c in self.chunks_cache}
        return list(sources)

    # --- Search Implementations ---

    def vector_search(self, query: str, top_k: int = MAX_TOP_K) -> List[Dict[str, Any]]:
        """Performs vector-based semantic search in Qdrant."""
        query_vector = self.get_embedding(query, is_query=True)
        response = self.qdrant_client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vector,
            limit=top_k
        )
        
        formatted_results = []
        for point in response.points:
            formatted_results.append({
                "id": point.id,
                "text": point.payload["text"],
                "metadata": {
                    "source": point.payload["source"],
                    "page": point.payload["page"],
                    "chunk_idx": point.payload["chunk_idx"]
                },
                "score": point.score
            })
        return formatted_results

    def bm25_search(self, query: str, top_k: int = MAX_TOP_K) -> List[Dict[str, Any]]:
        """Performs keyword search using BM25Okapi."""
        if not self.bm25_index or not self.chunks_cache:
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25_index.get_scores(tokenized_query)
        
        # Argsort scores descending
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = scores[idx]
            if score > 0:  # Only return matching documents
                chunk = self.chunks_cache[idx]
                results.append({
                    "id": chunk["id"],
                    "text": chunk["text"],
                    "metadata": chunk["metadata"],
                    "score": float(score)
                })
        return results

    def hybrid_search(self, query: str, top_k: int = FINAL_TOP_K) -> List[Dict[str, Any]]:
        """
        Combines Semantic (Vector) Search & Keyword (BM25) Search using Reciprocal Rank Fusion (RRF).
        RRF combines ranks: rrf_score = sum(1 / (k + rank))
        """
        vector_res = self.vector_search(query, top_k=MAX_TOP_K)
        bm25_res = self.bm25_search(query, top_k=MAX_TOP_K)

        if not vector_res and not bm25_res:
            return []

        # RRF Scoring Map
        rrf_scores = {}
        chunks_map = {}

        # Process Semantic Results
        for rank, res in enumerate(vector_res):
            chunk_id = res["id"]
            chunks_map[chunk_id] = res
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (RRF_CONSTANT + rank + 1))

        # Process BM25 Results
        for rank, res in enumerate(bm25_res):
            chunk_id = res["id"]
            chunks_map[chunk_id] = res
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (RRF_CONSTANT + rank + 1))

        # Sort based on RRF scores
        sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        hybrid_results = []
        for chunk_id in sorted_chunk_ids[:top_k]:
            original_chunk = chunks_map[chunk_id]
            # Add RRF score to metadata
            original_chunk["rrf_score"] = rrf_scores[chunk_id]
            hybrid_results.append(original_chunk)

        return hybrid_results
