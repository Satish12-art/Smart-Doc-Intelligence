import html
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import requests
import pandas as pd
import streamlit as st

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Sync Streamlit Secrets to environment variables for seamless cloud deployment
try:
    if hasattr(st, "secrets"):
        for k, v in st.secrets.items():
            if isinstance(v, str) and k not in os.environ:
                os.environ[k] = v
except Exception:
    pass

# Page Configuration
st.set_page_config(
    page_title="Smart Doc Intelligence | Hybrid RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configurations
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

# Engine Bridge: Supports both Remote FastAPI Backend and Direct In-Process Engine
@st.cache_resource
def get_inprocess_engine():
    """Initializes and caches backend services for zero-server in-process execution."""
    try:
        from backend.config import UPLOAD_DIR
        from backend.services.document_processor import DocumentProcessor
        from backend.services.search_engine import SearchEngine
        from backend.services.reranker import Reranker
        from backend.services.llm_service import LLMService
        return {
            "search_engine": SearchEngine(),
            "reranker": Reranker(),
            "llm_service": LLMService(),
            "processor": DocumentProcessor,
            "upload_dir": UPLOAD_DIR
        }
    except Exception as e:
        st.error(f"Failed to initialize direct engine: {e}")
        return None

def check_remote_api() -> bool:
    try:
        r = requests.get(f"{BACKEND_URL}/documents", timeout=1.0)
        return r.status_code == 200
    except Exception:
        return False

# Determine active engine mode
has_remote_api = check_remote_api()
engine = None if has_remote_api else get_inprocess_engine()
is_connected = has_remote_api or (engine is not None)
engine_mode = "FastAPI Backend" if has_remote_api else "Direct Cloud Engine"

# Unified Service Functions
def list_documents() -> List[str]:
    if has_remote_api:
        try:
            r = requests.get(f"{BACKEND_URL}/documents", timeout=3.0)
            if r.status_code == 200:
                return r.json().get("documents", [])
        except Exception:
            return []
    elif engine:
        return engine["search_engine"].get_all_indexed_documents()
    return []

def delete_document_service(doc_name: str) -> bool:
    if has_remote_api:
        try:
            r = requests.delete(f"{BACKEND_URL}/documents/{doc_name}")
            return r.status_code == 200
        except Exception:
            return False
    elif engine:
        try:
            engine["search_engine"].delete_document(doc_name)
            target = engine["upload_dir"] / doc_name
            if target.exists():
                target.unlink()
            return True
        except Exception:
            return False
    return False

def upload_and_index_files(uploaded_files) -> tuple[bool, str]:
    if has_remote_api:
        files_payload = [("files", (f.name, f.getvalue(), f.type)) for f in uploaded_files]
        try:
            res = requests.post(f"{BACKEND_URL}/upload", files=files_payload)
            if res.status_code == 200:
                return True, "Documents successfully processed & indexed!"
            return False, res.json().get("detail", "Upload failed")
        except Exception as e:
            return False, str(e)
    elif engine:
        try:
            for f in uploaded_files:
                file_path = engine["upload_dir"] / f.name
                with open(file_path, "wb") as buffer:
                    buffer.write(f.getvalue())
                chunks = engine["processor"].process_document(str(file_path), f.name)
                engine["search_engine"].add_documents(chunks)
            return True, "Documents successfully processed & indexed!"
        except Exception as e:
            return False, f"Failed to index files: {e}"
    return False, "No engine available"

def execute_search_query(query: str, search_type: str, use_reranking: bool, top_k: int) -> dict:
    if has_remote_api:
        payload = {
            "query": query,
            "search_type": search_type,
            "use_reranking": use_reranking,
            "top_k": top_k
        }
        res = requests.post(f"{BACKEND_URL}/query", json=payload)
        if res.status_code == 200:
            return res.json()
        return {"answer": f"Error: {res.text}", "sources": []}
    elif engine:
        # 1. Retrieval
        if search_type == "vector":
            retrieved = engine["search_engine"].vector_search(query)
        elif search_type == "bm25":
            retrieved = engine["search_engine"].bm25_search(query)
        else:
            retrieved = engine["search_engine"].hybrid_search(query)

        if not retrieved:
            return {
                "answer": "I could not find any relevant information in the uploaded documents.",
                "sources": []
            }

        # 2. Re-ranking
        if use_reranking:
            final_chunks = engine["reranker"].rerank(query, retrieved, top_k=top_k)
        else:
            final_chunks = retrieved[:top_k]

        # 3. LLM Synthesis
        return engine["llm_service"].generate_grounded_answer(query, final_chunks)

    return {"answer": "Search engine is offline.", "sources": []}

# Custom Premium Styling (Dark Theme & Neon Glassmorphism)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 800;
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        margin-bottom: 0.2rem;
    }
    
    .subtitle {
        color: #a0aec0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    .stat-card {
        background: rgba(25, 30, 45, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.25rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        margin-bottom: 1rem;
    }
    
    .stat-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #00f2fe;
        margin-bottom: 0.2rem;
    }
    
    .stat-label {
        font-size: 0.85rem;
        color: #a0aec0;
    }

    .answer-card {
        background: linear-gradient(135deg, rgba(20, 30, 48, 0.8) 0%, rgba(36, 59, 85, 0.8) 100%);
        border: 1.5px solid rgba(0, 242, 254, 0.3);
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 10px 40px 0 rgba(0, 242, 254, 0.1);
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    .answer-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.4rem;
        font-weight: 700;
        color: #00f2fe;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .source-tag {
        display: inline-block;
        background: rgba(0, 242, 254, 0.1);
        border: 1px solid rgba(0, 242, 254, 0.3);
        color: #00f2fe;
        border-radius: 8px;
        padding: 0.25rem 0.6rem;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    .source-snippet {
        background: rgba(255, 255, 255, 0.03);
        border-left: 3px solid #805ad5;
        padding: 0.75rem 1rem;
        margin-top: 0.5rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.9rem;
        color: #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# Layout: Main Panel Title
st.markdown('<div class="main-title">🧠 Smart Doc Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-Powered Hybrid Document Search & RAG System</div>', unsafe_allow_html=True)

# Main Dash Stats
indexed_docs = list_documents() if is_connected else []

col1, col2, col3 = st.columns(3)
with col1:
    status_label = f"🟢 Connected" if is_connected else "🔴 Offline"
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-val">{status_label}</div>
        <div class="stat-label">Mode: {engine_mode}</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-val">{len(indexed_docs)}</div>
        <div class="stat-label">Total Indexed Documents</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-val">Hybrid RAG</div>
        <div class="stat-label">Vector + BM25 + Rerank</div>
    </div>
    """, unsafe_allow_html=True)

# SIDEBAR: Upload & Configuration
with st.sidebar:
    st.image("https://img.icons8.com/nolan/96/brain.png", width=80)
    st.markdown("### 📤 Control Center")
    st.write("Upload company policies, notes, PDFs, Word documents or spreadsheets here.")
    
    # File Uploader
    uploaded_files = st.file_uploader(
        "Select Files", 
        type=["pdf", "docx", "csv", "txt", "md"], 
        accept_multiple_files=True
    )
    
    if st.button("🚀 Upload & Index", use_container_width=True):
        if not uploaded_files:
            st.warning("Please select at least one file first.")
        else:
            with st.spinner("Processing documents..."):
                ok, msg = upload_and_index_files(uploaded_files)
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(f"Failed to process: {msg}")
                    
    st.markdown("---")
    st.markdown("### ⚙️ Search Configuration")
    
    search_type = st.radio(
        "Search Strategy",
        options=["hybrid", "vector", "bm25"],
        format_func=lambda x: {
            "hybrid": "🔗 Hybrid Search (Semantic + BM25)",
            "vector": "⚡ Vector Search (Semantic)",
            "bm25": "🔍 BM25 Search (Keyword)"
        }[x]
    )
    
    use_reranking = st.toggle("🛡️ Enable Re-ranking", value=True)
    top_k = st.slider("Top Chunks (Context size)", min_value=1, max_value=10, value=5)
    
    st.markdown("---")
    st.markdown("### 📚 Indexed Documents")
    if not indexed_docs:
        st.info("No documents uploaded yet.")
    else:
        for doc in indexed_docs:
            doc_col1, doc_col2 = st.columns([0.85, 0.15])
            with doc_col1:
                st.text(f"📄 {doc[:23]}...")
            with doc_col2:
                if st.button("🗑️", key=f"del_{doc}"):
                    with st.spinner(f"Deleting {doc}..."):
                        if delete_document_service(doc):
                            st.toast(f"Deleted {doc}")
                            st.rerun()
                        else:
                            st.error("Failed to delete document.")

# MAIN PANEL: Chat / Search Interface
st.write("---")
st.markdown("### 💬 Ask anything to your documents")

query = st.text_input(
    "Enter your question here...",
    placeholder="e.g., Mujhe kitni paid leaves milti hain? Or remote work guidelines kya hain?",
    label_visibility="collapsed"
)

if query:
    if not is_connected:
        st.error("Engine is currently unavailable.")
    elif not indexed_docs:
        st.warning("No documents indexed yet. Please upload documents in the sidebar first.")
    else:
        with st.spinner("Searching and synthesizing answer..."):
            try:
                data = execute_search_query(query, search_type, use_reranking, top_k)
                answer = data.get("answer", "")
                sources = data.get("sources", [])
                
                # Display Answer (sanitize to prevent XSS)
                safe_answer = html.escape(answer)
                st.markdown(f"""
                <div class="answer-card">
                    <div class="answer-header">🤖 Answer</div>
                    <div style="font-size: 1.1rem; line-height: 1.6; color: #f7fafc; white-space: pre-wrap;">
                        {safe_answer}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Display Source Citations
                if sources:
                    st.markdown("#### 📚 Grounding Source Snippets")
                    seen_snippets = set()
                    unique_sources = []
                    for src in sources:
                        if src.get('text_snippet') and src['text_snippet'] not in seen_snippets:
                            seen_snippets.add(src['text_snippet'])
                            unique_sources.append(src)
                            
                    for idx, src in enumerate(unique_sources):
                        safe_source = html.escape(str(src.get('source', 'Unknown')))
                        safe_page = html.escape(str(src.get('page', '1')))
                        safe_snippet = html.escape(str(src.get('text_snippet', '')))
                        st.markdown(f"""
                        <div class="stat-card" style="background: rgba(255, 255, 255, 0.02); margin-bottom: 0.75rem;">
                            <span class="source-tag">📄 {safe_source}</span>
                            <span class="source-tag" style="background: rgba(128, 90, 213, 0.1); border-color: rgba(128, 90, 213, 0.3); color: #b7791f;">Page {safe_page}</span>
                            <div class="source-snippet">
                                "{safe_snippet}"
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Search failed: {e}")
