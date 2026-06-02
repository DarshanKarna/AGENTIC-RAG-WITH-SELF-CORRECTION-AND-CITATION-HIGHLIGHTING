"""
ingest.py - Unified Data Ingestion Pipeline
===========================================
Ingests either local PDFs or the rag-mini-bioasq Hugging Face dataset into ChromaDB.

Usage:
    python ingest.py --source pdf
    python ingest.py --source hf
"""
import os
import re
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import fitz  # PyMuPDF
import chromadb
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import tiktoken

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = Path("data")
RAW_PDF_DIR = DATA_DIR / "raw_pdfs"
CHROMA_DB_DIR_PDF = DATA_DIR / "chroma_db"
CHROMA_DB_DIR_HF = DATA_DIR / "chroma_db_hf"

MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MAX_WORKERS = os.cpu_count() or 4
ENCODE_BATCH = 32
CHROMA_BATCH = 5000

# PDF Specific Constants
PDF_COLLECTION_NAME = "document_chunks"

# HF Specific Constants
HF_DATASET = "rag-datasets/rag-mini-bioasq"
HF_CONFIG = "text-corpus"
HF_SPLIT = "passages"
HF_COLLECTION_NAME = "bioasq_chunks"

# --- PDF Helper Functions ---
def clean_text(text: str) -> str:
    if not text: return ""
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\bPage \d+ of \d+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(?i)^\s*page \d+\s*$', '', text)
    return text.strip()

def process_single_pdf(pdf_path: Path) -> List[Dict[str, Any]]:
    pages_data = []
    try:
        doc = fitz.open(str(pdf_path))
        doc_id = pdf_path.stem
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            cleaned_text = clean_text(page.get_text("text"))
            if cleaned_text:
                pages_data.append({
                    "text": cleaned_text,
                    "metadata": {"document_id": doc_id, "page_number": page_num + 1}
                })
        doc.close()
    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {e}")
    return pages_data

def extract_text_parallel(pdf_files: List[Path]) -> List[Dict[str, Any]]:
    all_pages_data = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_pdf = {executor.submit(process_single_pdf, pdf): pdf for pdf in pdf_files}
        for future in as_completed(future_to_pdf):
            try:
                all_pages_data.extend(future.result())
                logger.info(f"Successfully processed: {future_to_pdf[future].name}")
            except Exception as e:
                logger.error(f"Failed to process {future_to_pdf[future].name}: {e}")
    return all_pages_data

# --- HF Helper Functions ---
def download_dataset():
    logger.info(f"Downloading dataset: {HF_DATASET} (config={HF_CONFIG}, split={HF_SPLIT}) ...")
    ds = load_dataset(HF_DATASET, HF_CONFIG, split=HF_SPLIT)
    logger.info(f"Loaded {len(ds)} passages from Hugging Face.")
    return ds

def extract_passages_hf(ds) -> List[Dict[str, Any]]:
    passages = []
    skipped = 0
    for row in ds:
        text = row.get("passage")
        doc_id = row.get("id")
        if not text or not isinstance(text, str) or text.strip().lower() == "nan":
            skipped += 1
            continue
        passages.append({
            "text": text.strip(),
            "metadata": {"dataset_id": int(doc_id)}
        })
    logger.info(f"Extracted {len(passages)} valid passages (skipped {skipped} invalid rows).")
    return passages

# --- Shared Functions ---
def chunk_text(passages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    try:
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception as e:
        logger.warning(f"Failed to load tiktoken encoding: {e}. Falling back to basic word splitting.")
        enc = None

    chunks = []
    chunk_index = 0
    for item in passages_data:
        text = item["text"]
        metadata = item["metadata"].copy()
        
        if enc:
            tokens = enc.encode(text)
            start = 0
            while start < len(tokens):
                end = min(start + CHUNK_SIZE, len(tokens))
                chunk_text_str = enc.decode(tokens[start:end])
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_index"] = chunk_index
                chunks.append({"text": chunk_text_str, "metadata": chunk_metadata})
                chunk_index += 1
                start += CHUNK_SIZE - CHUNK_OVERLAP
        else:
            words = text.split()
            start = 0
            while start < len(words):
                end = min(start + CHUNK_SIZE, len(words))
                chunk_text_str = " ".join(words[start:end])
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_index"] = chunk_index
                chunks.append({"text": chunk_text_str, "metadata": chunk_metadata})
                chunk_index += 1
                start += CHUNK_SIZE - CHUNK_OVERLAP
    logger.info(f"Created {len(chunks)} chunks.")
    return chunks

def embed_chunks(chunks: List[Dict[str, Any]]) -> list:
    logger.info(f"Loading embedding model ({MODEL_NAME})...")
    model = SentenceTransformer(MODEL_NAME)
    texts = [chunk["text"] for chunk in chunks]
    logger.info("Generating embeddings...")
    return model.encode(texts, batch_size=ENCODE_BATCH, show_progress_bar=True).tolist()

def store_in_chromadb(chunks: List[Dict[str, Any]], embeddings: list, db_dir: Path, collection_name: str, source_type: str):
    db_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Storing in ChromaDB at {db_dir}...")
    client = chromadb.PersistentClient(path=str(db_dir))
    collection = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
    
    if source_type == "pdf":
        ids = [f"{c['metadata']['document_id']}_p{c['metadata']['page_number']}_c{c['metadata']['chunk_index']}" for c in chunks]
    else:
        ids = [f"hf_{c['metadata']['dataset_id']}_c{c['metadata']['chunk_index']}" for c in chunks]
        
    metadatas = [c["metadata"] for c in chunks]
    texts = [c["text"] for c in chunks]
    
    for i in range(0, len(ids), CHROMA_BATCH):
        collection.add(
            ids=ids[i:i+CHROMA_BATCH],
            embeddings=embeddings[i:i+CHROMA_BATCH],
            metadatas=metadatas[i:i+CHROMA_BATCH],
            documents=texts[i:i+CHROMA_BATCH]
        )
    logger.info(f"Successfully stored {len(ids)} chunks in ChromaDB '{collection_name}'.")

def ingest_pdf():
    RAW_PDF_DIR.mkdir(parents=True, exist_ok=True)
    pdf_files = list(RAW_PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDFs found in {RAW_PDF_DIR}.")
        return
    logger.info(f"Starting PDF ingestion for {len(pdf_files)} files...")
    
    pages_data = extract_text_parallel(pdf_files)
    if not pages_data:
        logger.warning("No text extracted.")
        return
        
    chunks = chunk_text(pages_data)
    embeddings = embed_chunks(chunks)
    store_in_chromadb(chunks, embeddings, CHROMA_DB_DIR_PDF, PDF_COLLECTION_NAME, "pdf")

def ingest_hf():
    logger.info("Starting Hugging Face dataset ingestion...")
    ds = download_dataset()
    passages = extract_passages_hf(ds)
    if not passages:
        logger.warning("No valid passages found.")
        return
        
    chunks = chunk_text(passages)
    embeddings = embed_chunks(chunks)
    store_in_chromadb(chunks, embeddings, CHROMA_DB_DIR_HF, HF_COLLECTION_NAME, "hf")

def main():
    parser = argparse.ArgumentParser(description="Unified Data Ingestion Pipeline")
    parser.add_argument("--source", type=str, choices=["pdf", "hf"], required=True,
                        help="Data source to ingest: 'pdf' for local PDFs, 'hf' for HuggingFace dataset.")
    args = parser.parse_args()

    if args.source == "pdf":
        ingest_pdf()
    elif args.source == "hf":
        ingest_hf()

if __name__ == "__main__":
    main()
