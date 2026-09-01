import logging
import google.generativeai as genai
from typing import List, Dict, Any

from backend.config import GEMINI_API_KEY, LLM_MODEL

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        self.configured = False
        if GEMINI_API_KEY:
            try:
                genai.configure(api_key=GEMINI_API_KEY)
                self.model = genai.GenerativeModel(LLM_MODEL)
                self.configured = True
                logger.info(f"Initialized Gemini LLM service with model: {LLM_MODEL}")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini LLM model: {e}")
        else:
            logger.warning("GEMINI_API_KEY not found. LLM answers will use a demonstration mockup.")

    def generate_grounded_answer(self, query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generates an answer grounded ONLY in the retrieved document chunks.
        Forces the model to cite sources and page numbers.
        """
        if not chunks:
            return {
                "answer": "No documents or relevant chunks found in the database. Please upload documents first.",
                "sources": []
            }

        # Format context for the prompt
        context_parts = []
        sources = []
        
        for idx, chunk in enumerate(chunks):
            source_info = f"Source {idx + 1}: {chunk['metadata']['source']} (Page {chunk['metadata']['page']})"
            context_parts.append(f"{source_info}\nContent: {chunk['text']}")
            
            sources.append({
                "source": chunk['metadata']['source'],
                "page": chunk['metadata']['page'],
                "text_snippet": chunk['text'][:150] + "..." if len(chunk['text']) > 150 else chunk['text']
            })

        context_text = "\n\n---\n\n".join(context_parts)

        # Core Prompt enforcing grounding and citation rules
        system_instruction = (
            "You are an AI-powered smart document assistant. Your task is to answer the user's question using ONLY the provided Context.\n"
            "Strict Guidelines:\n"
            "1. Answer the question truthfully and concisely based solely on the Context.\n"
            "2. For every fact or statement you make, you MUST cite the source document name and page number at the end of the sentence or clause using brackets, e.g. [HR Policy.pdf, Page 3].\n"
            "3. If the Context does not contain the answer to the question, state exactly: 'I could not find this information in the uploaded documents.' and do not extrapolate.\n"
            "4. Do not use any external knowledge outside of the provided Context."
        )

        prompt = (
            f"{system_instruction}\n\n"
            f"Context:\n"
            f"{context_text}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )

        if self.configured:
            try:
                # Generate content using Gemini
                response = self.model.generate_content(prompt)
                answer = response.text.strip()
                return {
                    "answer": answer,
                    "sources": sources
                }
            except Exception as e:
                logger.error(f"Gemini generation error: {e}")
                return {
                    "answer": f"Error generating answer with Gemini: {e}",
                    "sources": sources
                }
        else:
            # Fallback mockup response demonstrating what the LLM would output
            mock_answer = (
                "⚠️ **[DEMO MODE: GEMINI API KEY MISSING]**\n\n"
                "Here is the retrieved context from your documents. To generate live grounded answers, please set your GEMINI_API_KEY environment variable.\n\n"
                f"**Question:** {query}\n\n"
                f"**Top Retrieved Context:**\n"
            )
            for idx, chunk in enumerate(chunks[:2]):
                mock_answer += f"- From *{chunk['metadata']['source']}* (Page {chunk['metadata']['page']}): \"{chunk['text'][:120]}...\"\n"
            
            return {
                "answer": mock_answer,
                "sources": sources
            }
