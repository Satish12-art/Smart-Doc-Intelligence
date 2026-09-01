import fitz  # PyMuPDF
import docx
import pandas as pd
import uuid
from typing import List, Dict, Any
from backend.config import CHUNK_SIZE, CHUNK_OVERLAP

class DocumentProcessor:
    @staticmethod
    def extract_text_from_pdf(file_path: str) -> List[Dict[str, Any]]:
        """
        Extracts text page-by-page from a PDF file.
        Returns a list of dicts: [{"text": str, "page": int}]
        """
        pages = []
        doc = fitz.open(file_path)
        for page_idx, page in enumerate(doc):
            text = page.get_text("text").strip()
            if text:
                pages.append({
                    "text": text,
                    "page": page_idx + 1  # 1-indexed page
                })
        doc.close()
        return pages

    @staticmethod
    def extract_text_from_docx(file_path: str) -> List[Dict[str, Any]]:
        """
        Extracts text from a Word document (.docx).
        Word docs don't have natural 'pages' in the XML easily, so we extract paragraphs 
        and group them into logical page-like segments or treat the document as a single flow.
        """
        doc = docx.Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                full_text.append(text)
        
        # Join paragraphs with double newlines
        content = "\n\n".join(full_text)
        return [{"text": content, "page": 1}]

    @staticmethod
    def extract_text_from_csv(file_path: str) -> List[Dict[str, Any]]:
        """
        Extracts text from a CSV file by converting rows to structured textual formats.
        """
        df = pd.read_csv(file_path)
        rows_text = []
        for idx, row in df.iterrows():
            row_str = ", ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
            rows_text.append(f"Row {idx + 1}: {row_str}")
        
        content = "\n".join(rows_text)
        return [{"text": content, "page": 1}]

    @staticmethod
    def extract_text_from_txt(file_path: str) -> List[Dict[str, Any]]:
        """
        Extracts text from a plain text file.
        """
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read().strip()
        return [{"text": content, "page": 1}]

    @classmethod
    def process_document(cls, file_path: str, filename: str) -> List[Dict[str, Any]]:
        """
        Extracts and chunks the document based on its file extension.
        Returns a list of chunks:
        [{
            "id": str,
            "text": str,
            "metadata": {
                "source": str,
                "page": int,
                "chunk_idx": int
            }
        }]
        """
        ext = filename.split(".")[-1].lower()
        pages = []

        if ext == "pdf":
            pages = cls.extract_text_from_pdf(file_path)
        elif ext == "docx":
            pages = cls.extract_text_from_docx(file_path)
        elif ext in ["csv", "tsv"]:
            pages = cls.extract_text_from_csv(file_path)
        elif ext in ["txt", "md"]:
            pages = cls.extract_text_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        chunks = []
        for page_data in pages:
            text = page_data["text"]
            page_num = page_data["page"]
            
            # Sub-chunking the page content if it exceeds CHUNK_SIZE
            start = 0
            chunk_idx = 0
            while start < len(text):
                end = start + CHUNK_SIZE
                chunk_text = text[start:end].strip()
                
                if chunk_text:
                    chunks.append({
                        "id": str(uuid.uuid4()),
                        "text": chunk_text,
                        "metadata": {
                            "source": filename,
                            "page": page_num,
                            "chunk_idx": chunk_idx
                        }
                    })
                    chunk_idx += 1
                
                start += (CHUNK_SIZE - CHUNK_OVERLAP)
                
        return chunks
