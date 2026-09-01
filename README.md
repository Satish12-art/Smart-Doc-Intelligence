# 🧠 Smart Doc Intelligence — Enterprise Hybrid RAG System

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-DC2626.svg?logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![Google Gemini](https://img.shields.io/badge/LLM-Gemini_Flash-8E75B2.svg?logo=google&logoColor=white)](https://ai.google.dev/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An enterprise-grade **Hybrid Retrieval-Augmented Generation (RAG)** and **Document Intelligence Platform**. It enables semantic query matching, exact keyword retrieval, rank fusion, cross-encoder re-ranking, and grounded multi-format document Q&A with strict citations.

---

## 📑 Table of Contents

- [Key Features](#-key-features)
- [Architecture & Data Pipeline](#-architecture--data-pipeline)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Variables](#environment-variables)
  - [Running the Application](#running-the-application)
- [API Reference](#-api-reference)
- [Search & Retrieval Mechanics](#-search--retrieval-mechanics)
- [License](#-license)

---

## 🌟 Key Features

* **Multi-Format Parsing**: Built-in support for `.pdf` (PyMuPDF with page-level tracking), `.docx` (python-docx), `.csv` (Pandas tabular parsing), `.txt`, and `.md`.
* **Hybrid Search (Dense + Sparse)**: Combines dense semantic vector embeddings (Gemini `text-embedding-004` / local `all-MiniLM-L6-v2`) with sparse keyword matching (**BM25**).
* **Reciprocal Rank Fusion (RRF)**: Merges heterogeneous scoring spaces from vector and keyword search pipelines into balanced, rank-ordered candidates.
* **Precision Re-ranking**: Uses **Cohere Rerank API** with an automatic offline fallback to local **Cross-Encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`).
* **Source-Grounded Answers with Citations**: Powered by Google Gemini with strict prompt constraints that prevent hallucinations and mandate grounded bracket citations `[Doc: filename, Page: X]`.
* **Dual Database Operating Modes**:
  * **Zero-Setup Local Mode**: Runs Qdrant embedded directly in Python storage without external containers.
  * **Docker Mode**: Ready for high-concurrency production deployments via `docker-compose`.
* **Modern Dashboard**: Glassmorphic dark UI built with Streamlit, real-time index metrics, chunk inspection, and source segment previewers.

---

## 🏗️ Architecture & Data Pipeline

```
               📄 Upload Documents (PDF / DOCX / TXT / CSV)
                                   │
                                   ▼
               Document Ingestion & Page-level Chunking
                                   │
                 ┌─────────────────┴─────────────────┐
                 ▼                                   ▼
        Vector Embeddings                       BM25 Index
     (Gemini / local MiniLM)               (Tokenized Keywords)
                 │                                   │
                 ▼                                   ▼
          Qdrant Vector DB                     BM25 Disk Cache
                 │                                   │
     ┌───────────┴───────────────────────────────────┴───────────┐
     │                       USER QUERY                          │
     └───────────┬───────────────────────────────────┬───────────┘
                 │                                   │
                 ▼                                   ▼
           Vector Search                        BM25 Search
       (Semantic Similarity)                 (Exact Keyword Match)
                 │                                   │
                 └─────────────────┬─────────────────┘
                                   ▼
                      Reciprocal Rank Fusion (RRF)
                                   │
                                   ▼
                   Cross-Encoder / Cohere Re-ranker
                                   │
                                   ▼
                         Context-Grounded Prompt
                                   │
                                   ▼
                     LLM Answer Generation (Gemini)
                                   │
                                   ▼
                  🤖 Verified Response + 📚 Citations
```

---

## 🛠️ Tech Stack

| Component | Technology | Description |
|---|---|---|
| **Backend Framework** | [FastAPI](https://fastapi.tiangolo.com/) | High-performance asynchronous REST API |
| **Frontend UI** | [Streamlit](https://streamlit.io/) | Interactive dashboard with custom glassmorphism styles |
| **Vector Database** | [Qdrant](https://qdrant.tech/) | Vector similarity search engine (local persistent or Docker) |
| **Sparse Search** | [Rank-BM25](https://github.com/dorianbrown/rank_bm25) | BM25 Okapi keyword ranking algorithm |
| **Embeddings & LLM** | [Google Gemini](https://ai.google.dev/) | `text-embedding-004` & `gemini-1.5-flash` / `gemini-2.5-flash` |
| **Offline Embeddings** | [Sentence-Transformers](https://www.sbert.net/) | `all-MiniLM-L6-v2` for zero-API-key local execution |
| **Re-ranking** | [Cohere](https://cohere.com/rerank) / Cross-Encoder | Contextual passage re-ranking |
| **Document Parsers** | PyMuPDF, python-docx, Pandas | Structured text extraction with page metadata |

---

## 📁 Project Structure

```bash
smart-document-search/
├── backend/
│   ├── config.py                 # System hyperparameters, paths & env configs
│   ├── main.py                   # FastAPI application & REST route definitions
│   ├── requirements.txt          # Backend dependencies
│   ├── data/
│   │   ├── uploads/              # Raw uploaded source files
│   │   ├── indices/              # BM25 pickle indices & metadata
│   │   └── qdrant_local/         # Local embedded Qdrant vector storage
│   └── services/
│       ├── document_processor.py # Parsing & page-aware chunking pipeline
│       ├── search_engine.py      # Qdrant client, BM25, & RRF Hybrid logic
│       ├── reranker.py           # Cohere & Cross-Encoder re-ranking
│       └── llm_service.py        # Gemini prompting, grounding & citation parsing
├── frontend/
│   ├── app.py                    # Streamlit UI dashboard & interactions
│   └── requirements.txt          # Frontend dependencies
├── docker-compose.yml            # Docker container configuration for Qdrant
├── .gitignore                    # Ignored artifacts, virtualenvs & indices
└── README.md                     # Documentation
```

---

## 🚀 Getting Started

### Prerequisites

* **Python 3.10+** installed on your system.
* *(Optional)* **Docker** if you prefer running Qdrant via container.

---

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Satish12-art/Smart-Doc-Intelligence.git
   cd Smart-Doc-Intelligence
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r backend/requirements.txt
   pip install -r frontend/requirements.txt
   ```

---

### Environment Variables

Set your API keys (Optional — the application includes offline fallback models for local execution):

```bash
# Gemini API Key (Embeddings & LLM Generation)
export GEMINI_API_KEY="your-gemini-api-key"

# Cohere API Key (Precision Re-ranking)
export COHERE_API_KEY="your-cohere-api-key"

# Optional: Set Qdrant Host if running via Docker
# export QDRANT_HOST="localhost"
```

---

### Running the Application

Open two terminal windows (with `venv` activated in both):

#### 1. Start the FastAPI Backend
```bash
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
* API Documentation (Swagger UI): `http://localhost:8000/docs`

#### 2. Start the Streamlit Frontend
```bash
streamlit run frontend/app.py --server.port 8501
```
* Access the web interface at: `http://localhost:8501`

---

## 🔌 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/upload` | `POST` | Upload multi-format documents, chunk text, and index into vector & BM25 stores. |
| `/query` | `POST` | Execute hybrid search, re-ranking, and grounded LLM answer generation. |
| `/documents` | `GET` | List all indexed documents and chunk statistics. |
| `/documents/{filename}` | `DELETE` | Remove a document and purge its corresponding vectors and BM25 index. |

### Example Query Request Payload:
```json
{
  "query": "What are the key policy requirements for annual leave?",
  "search_type": "hybrid",
  "use_reranking": true,
  "top_k": 5
}
```

---

## 🔍 Search & Retrieval Mechanics

1. **Reciprocal Rank Fusion (RRF)**:
   Scores from dense vector similarity and sparse BM25 are fused using the formula:
   $$\text{RRF Score}(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
   where $k = 60$ and $r_m(d)$ represents the document's ordinal rank in retrieval method $m$.

2. **Two-Stage Re-ranking**:
   The top candidate chunks retrieved by RRF are passed through a cross-encoder / Cohere re-ranker to score semantic relevance against the query at fine token granularity.

3. **Grounded Generation**:
   The top re-ranked chunks are formatted with document and page metadata into a grounded context window, instructing the LLM to strictly cite evidence in square brackets.

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
