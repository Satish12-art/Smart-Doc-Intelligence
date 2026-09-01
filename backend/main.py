import shutil
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.config import UPLOAD_DIR
from backend.services.document_processor import DocumentProcessor
from backend.services.search_engine import SearchEngine
from backend.services.reranker import Reranker
from backend.services.llm_service import LLMService

app = FastAPI(title="Smart Document Search System", version="1.0.0")

# Enable CORS for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Services
search_engine = SearchEngine()
reranker = Reranker()
llm_service = LLMService()

class QueryRequest(BaseModel):
    query: str
    search_type: str = "hybrid"  # "vector", "bm25", "hybrid"
    use_reranking: bool = True
    top_k: int = 5

@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Upload files, extract text, chunk them, and add to the Search Engine index.
    """
    uploaded_files_summary = []
    
    for file in files:
        file_path = UPLOAD_DIR / file.filename
        
        # Save file to local uploads directory
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save file {file.filename}: {e}")
            
        try:
            # Process and chunk document
            chunks = DocumentProcessor.process_document(str(file_path), file.filename)
            
            # Index chunks in Search Engine (Qdrant & BM25)
            search_engine.add_documents(chunks)
            
            uploaded_files_summary.append({
                "filename": file.filename,
                "chunks_count": len(chunks),
                "status": "Indexed"
            })
        except Exception as e:
            # Clean up saved file on indexing failure
            if file_path.exists():
                file_path.unlink()
            raise HTTPException(status_code=400, detail=f"Failed to process and index {file.filename}: {e}")
            
    return {"message": "Files successfully uploaded and indexed", "details": uploaded_files_summary}

@app.post("/query")
async def query_documents(request: QueryRequest):
    """
    Search and generate response from indexed documents based on query.
    """
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Step 1: Retrieval
    if request.search_type == "vector":
        retrieved_chunks = search_engine.vector_search(query)
    elif request.search_type == "bm25":
        retrieved_chunks = search_engine.bm25_search(query)
    else:  # hybrid
        retrieved_chunks = search_engine.hybrid_search(query)

    if not retrieved_chunks:
        return {
            "answer": "I could not find any relevant information in the uploaded documents.",
            "sources": []
        }

    # Step 2: Reranking
    if request.use_reranking:
        final_chunks = reranker.rerank(query, retrieved_chunks, top_k=request.top_k)
    else:
        final_chunks = retrieved_chunks[:request.top_k]

    # Step 3: LLM Answer Generation
    result = llm_service.generate_grounded_answer(query, final_chunks)
    return result

@app.get("/documents")
async def list_documents():
    """
    List all uploaded and indexed documents.
    """
    documents = search_engine.get_all_indexed_documents()
    return {"documents": documents}

@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    """
    Delete a document and its indices.
    """
    # Delete from indexing
    try:
        search_engine.delete_document(filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete index for {filename}: {e}")

    # Delete local file
    file_path = UPLOAD_DIR / filename
    if file_path.exists():
        file_path.unlink()
        
    return {"message": f"Document '{filename}' successfully deleted."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
