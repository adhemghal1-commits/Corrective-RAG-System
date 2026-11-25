"""
Document Processing Pipeline
=============================
Handles: upload → extract text → clean → split into chunks.
Supports PDF, TXT, DOCX, and Markdown files.
"""

import re
import unicodedata
from pathlib import Path
from typing import List, Dict, Any

from config import (
    DOCUMENTS_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SUPPORTED_EXTENSIONS,
)


# ──────────────────────────────────────────────
# Text Extraction
# ──────────────────────────────────────────────

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""
    from PyPDF2 import PdfReader

    reader = PdfReader(file_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n\n".join(pages)


def extract_text_from_txt(file_path: str) -> str:
    """Extract text from a plain text file."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def extract_text_from_docx(file_path: str) -> str:
    """Extract text from a DOCX file."""
    import docx

    doc = docx.Document(file_path)
    return "\n\n".join([para.text for para in doc.paragraphs if para.text.strip()])


def extract_text(file_path: str) -> str:
    """Route to the appropriate extractor based on file extension."""
    ext = Path(file_path).suffix.lower()
    extractors = {
        ".pdf": extract_text_from_pdf,
        ".txt": extract_text_from_txt,
        ".md": extract_text_from_txt,
        ".docx": extract_text_from_docx,
    }
    extractor = extractors.get(ext)
    if extractor is None:
        raise ValueError(f"Unsupported file type: {ext}")
    return extractor(file_path)


# ──────────────────────────────────────────────
# Text Cleaning
# ──────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Clean extracted text:
    - Normalize unicode
    - Remove excessive whitespace
    - Remove non-printable characters
    - Collapse multiple newlines
    """
    # Normalize unicode characters
    text = unicodedata.normalize("NFKD", text)

    # Remove non-printable characters (keep newlines and tabs)
    text = re.sub(r'[^\x20-\x7E\n\t\r]', ' ', text)

    # Collapse multiple spaces into one
    text = re.sub(r'[ \t]+', ' ', text)

    # Collapse multiple blank lines into at most two newlines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Strip leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    return text.strip()


# ──────────────────────────────────────────────
# Text Splitting (Chunking)
# ──────────────────────────────────────────────

def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Split text into overlapping chunks.
    Tries to split on paragraph/sentence boundaries first.
    """
    if not text:
        return []

    # Try splitting on paragraphs first
    paragraphs = re.split(r'\n\n+', text)

    chunks: List[str] = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # If adding this paragraph exceeds chunk size,
        # save current chunk and start a new one
        if len(current_chunk) + len(para) + 1 > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())

                # Keep overlap from the end of the current chunk
                if chunk_overlap > 0:
                    overlap_text = current_chunk[-chunk_overlap:]
                    current_chunk = overlap_text + " " + para
                else:
                    current_chunk = para
            else:
                # Paragraph itself is bigger than chunk_size → split by sentences
                sentences = re.split(r'(?<=[.!?])\s+', para)
                for sent in sentences:
                    if len(current_chunk) + len(sent) + 1 > chunk_size:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                            if chunk_overlap > 0:
                                overlap_text = current_chunk[-chunk_overlap:]
                                current_chunk = overlap_text + " " + sent
                            else:
                                current_chunk = sent
                        else:
                            # Single sentence is too long → hard split
                            for i in range(0, len(sent), chunk_size - chunk_overlap):
                                chunks.append(sent[i:i + chunk_size].strip())
                            current_chunk = ""
                    else:
                        current_chunk = (current_chunk + " " + sent).strip()
        else:
            current_chunk = (current_chunk + "\n\n" + para).strip()

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # Filter out very small chunks (noise)
    chunks = [c for c in chunks if len(c) > 30]

    return chunks


# ──────────────────────────────────────────────
# Full Processing Pipeline
# ──────────────────────────────────────────────

def process_document(file_path: str) -> List[Dict[str, Any]]:
    """
    Full pipeline for a single document:
    extract → clean → split → return list of chunk dicts.
    """
    path = Path(file_path)
    raw_text = extract_text(str(path))
    cleaned_text = clean_text(raw_text)
    chunks = split_text(cleaned_text)

    return [
        {
            "text": chunk,
            "metadata": {
                "source": path.name,
                "source_path": str(path),
                "chunk_index": i,
                "total_chunks": len(chunks),
            },
        }
        for i, chunk in enumerate(chunks)
    ]


def process_all_documents(directory: str = None) -> List[Dict[str, Any]]:
    """
    Process all supported documents in a directory.
    Returns a flat list of all chunk dicts.
    """
    if directory is None:
        directory = str(DOCUMENTS_DIR)

    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    all_chunks = []
    for file_path in sorted(dir_path.iterdir()):
        if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            print(f"  Processing: {file_path.name}")
            chunks = process_document(str(file_path))
            all_chunks.extend(chunks)
            print(f"    → {len(chunks)} chunks extracted")

    print(f"\nTotal: {len(all_chunks)} chunks from {directory}")
    return all_chunks


def process_uploaded_file(uploaded_file) -> List[Dict[str, Any]]:
    """
    Process a Streamlit UploadedFile object.
    Saves to a temp location, processes, then returns chunks.
    """
    import tempfile

    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    chunks = process_document(tmp_path)

    # Update metadata to show original filename
    for chunk in chunks:
        chunk["metadata"]["source"] = uploaded_file.name

    # Clean up temp file
    Path(tmp_path).unlink(missing_ok=True)

    return chunks

