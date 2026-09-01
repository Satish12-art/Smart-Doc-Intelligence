import html
import streamlit as st
import requests
import os
import pandas as pd
from typing import List, Dict, Any

# Page Configuration
st.set_page_config(
    page_title="Smart Doc Intelligence | Hybrid RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configurations
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

# Custom Premium Styling (Dark Theme & Neon Glassmorphism)
st.markdown("""
<style>
    /* Main Background & Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Title styling */
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

    /* Cards & Glassmorphic Blocks */
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
        font-size: 2rem;
        font-weight: 700;
        color: #00f2fe;
        margin-bottom: 0.2rem;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #a0aec0;
    }

    /* Answer box */
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
    
    /* Sources style */
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
    
    /* Sidebar adjustments */
    .sidebar .sidebar-content {
        background-image: linear-gradient(#141e30, #243b55);
    }
</style>
""", unsafe_allow_html=True)

# Helper function to query backend
def query_backend_health():
    try:
        # Simple test to check if backend is reachable
        res = requests.get(f"{BACKEND_URL}/documents")
        return res.status_code == 200
    except:
        return False

def get_indexed_documents() -> List[str]:
    try:
        res = requests.get(f"{BACKEND_URL}/documents")
        if res.status_code == 200:
            return res.json().get("documents", [])
    except Exception as e:
        st.error(f"Error fetching documents: {e}")
    return []

# Layout: Main Panel Title
st.markdown('<div class="main-title">🧠 Smart Doc Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-Powered Hybrid Document Search & RAG System</div>', unsafe_allow_html=True)

# Main Dash Stats
api_healthy = query_backend_health()
indexed_docs = get_indexed_documents() if api_healthy else []

col1, col2, col3 = st.columns(3)
with col1:
    status_color = "🟢 Connected" if api_healthy else "🔴 Offline"
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-val">{status_color}</div>
        <div class="stat-label">FastAPI Backend Status</div>
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
                files_payload = []
                for f in uploaded_files:
                    files_payload.append(
                        ("files", (f.name, f.getvalue(), f.type))
                    )
                
                try:
                    res = requests.post(f"{BACKEND_URL}/upload", files=files_payload)
                    if res.status_code == 200:
                        st.success("Documents successfully processed & indexed!")
                        # Force refresh
                        st.rerun()
                    else:
                        st.error(f"Failed to process: {res.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Upload failed: {e}")
                    
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
                        try:
                            res = requests.delete(f"{BACKEND_URL}/documents/{doc}")
                            if res.status_code == 200:
                                st.toast(f"Deleted {doc}")
                                st.rerun()
                            else:
                                st.error("Failed to delete document.")
                        except Exception as e:
                            st.error(f"Error deleting doc: {e}")

# MAIN PANEL: Chat / Search Interface
st.write("---")
st.markdown("### 💬 Ask anything to your documents")

query = st.text_input(
    "Enter your question here...",
    placeholder="e.g., Mujhe kitni paid leaves milti hain? Or remote work guidelines kya hain?",
    label_visibility="collapsed"
)

if query:
    if not api_healthy:
        st.error("Cannot execute search: Backend server is unreachable. Please verify that FastAPI backend is running.")
    elif not indexed_docs:
        st.warning("No documents indexed yet. Please upload documents in the sidebar first.")
    else:
        with st.spinner("Searching and synthesizing answer..."):
            try:
                payload = {
                    "query": query,
                    "search_type": search_type,
                    "use_reranking": use_reranking,
                    "top_k": top_k
                }
                res = requests.post(f"{BACKEND_URL}/query", json=payload)
                
                if res.status_code == 200:
                    data = res.json()
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
                        # Remove duplicate snippets for cleaner listing
                        seen_snippets = set()
                        unique_sources = []
                        for src in sources:
                            if src['text_snippet'] not in seen_snippets:
                                seen_snippets.add(src['text_snippet'])
                                unique_sources.append(src)
                                
                        for idx, src in enumerate(unique_sources):
                            safe_source = html.escape(str(src['source']))
                            safe_page = html.escape(str(src['page']))
                            safe_snippet = html.escape(str(src['text_snippet']))
                            st.markdown(f"""
                            <div class="stat-card" style="background: rgba(255, 255, 255, 0.02); margin-bottom: 0.75rem;">
                                <span class="source-tag">📄 {safe_source}</span>
                                <span class="source-tag" style="background: rgba(128, 90, 213, 0.1); border-color: rgba(128, 90, 213, 0.3); color: #b7791f;">Page {safe_page}</span>
                                <div class="source-snippet">
                                    "{safe_snippet}"
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.error(f"Error querying: {res.json().get('detail', 'Unknown error')}")
            except Exception as e:
                st.error(f"Search failed: {e}")
