import logging
from typing import List, Dict, Any
import cohere

from backend.config import COHERE_API_KEY, FINAL_TOP_K

logger = logging.getLogger(__name__)

class Reranker:
    def __init__(self):
        self.cohere_client = None
        self.local_reranker = None
        self.use_cohere = False
        self.use_local = False

        if COHERE_API_KEY:
            try:
                self.cohere_client = cohere.Client(COHERE_API_KEY)
                self.use_cohere = True
                logger.info("Configured Cohere Rerank API.")
            except Exception as e:
                logger.error(f"Failed to initialize Cohere Client: {e}")

        # Local cross-encoder fallback if Cohere is not configured
        if not self.use_cohere:
            logger.info("Cohere Rerank API key not found. Reranking will default to RRF or local model.")

    def _get_local_reranker(self):
        if not self.use_cohere and not self.use_local and self.local_reranker is None:
            try:
                from sentence_transformers import CrossEncoder
                logger.info("Loading local Cross-Encoder reranking model (cross-encoder/ms-marco-MiniLM-L-6-v2)...")
                # Using a very lightweight cross-encoder model (about 80MB)
                self.local_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
                self.use_local = True
            except Exception as e:
                logger.warning(f"Could not load local Cross-Encoder (sentence-transformers missing or error): {e}. "
                               f"Reranking will be bypassed and will rely on RRF order.")
                self.use_local = False
        return self.local_reranker

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = FINAL_TOP_K) -> List[Dict[str, Any]]:
        """
        Reranks a list of retrieved chunks relative to the query.
        Falls back gracefully if Cohere API key is not configured.
        """
        if not documents:
            return []

        # If Cohere API Key is available, use it
        if self.use_cohere and self.cohere_client:
            try:
                texts = [doc["text"] for doc in documents]
                response = self.cohere_client.rerank(
                    model="rerank-english-v3.0",
                    query=query,
                    documents=texts,
                    top_n=top_k
                )
                
                reranked_docs = []
                for result in response.results:
                    idx = result.index
                    doc = documents[idx].copy()
                    doc["rerank_score"] = float(result.relevance_score)
                    reranked_docs.append(doc)
                
                logger.info("Reranking completed using Cohere Rerank.")
                return reranked_docs
            except Exception as e:
                logger.error(f"Cohere Rerank failed: {e}. Falling back to default order / local reranker.")

        # Attempt Local Cross-Encoder Reranking
        local_model = self._get_local_reranker()
        if self.use_local and local_model:
            try:
                pairs = [[query, doc["text"]] for doc in documents]
                scores = local_model.predict(pairs)
                
                # Zip scores with documents and sort
                scored_docs = []
                for score, doc in zip(scores, documents):
                    doc_copy = doc.copy()
                    doc_copy["rerank_score"] = float(score)
                    scored_docs.append(doc_copy)
                
                # Sort descending
                scored_docs = sorted(scored_docs, key=lambda x: x["rerank_score"], reverse=True)
                logger.info("Reranking completed using local Cross-Encoder.")
                return scored_docs[:top_k]
            except Exception as e:
                logger.error(f"Local Cross-Encoder reranking failed: {e}. Using RRF order.")

        # Bypass Reranking (return first top_k elements based on original order)
        logger.info("Bypassing reranker; using original RRF order.")
        return documents[:top_k]
