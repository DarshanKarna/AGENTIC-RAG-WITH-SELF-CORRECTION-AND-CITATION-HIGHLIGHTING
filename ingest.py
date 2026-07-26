"""
ingest.py - OKF Bundle Ingestion Pipeline
==========================================
Ingests OKF v0.2 concept documents (markdown + YAML frontmatter) from
okf_bundle/ into a persistent ChromaDB vector store.

Fallback: use --legacy-pdf to ingest raw PDFs from data/ directly.

Usage:
    python ingest.py                # Ingest from okf_bundle/
    python ingest.py --legacy-pdf   # Ingest raw PDFs from data/
"""
import argparse
import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any

import fitz  # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer
import tiktoken
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = Path("data")
OKF_BUNDLE_DIR = Path("okf_bundle")
CHROMA_DB_DIR = DATA_DIR / "chroma_db"
COLLECTION_NAME = "document_chunks"

MODEL_NAME = "BAAI/bge-m3"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MAX_WORKERS = os.cpu_count() or 4
ENCODE_BATCH = 32
CHROMA_BATCH = 5000


# ---------------------------------------------------------------------------
# OKF Concept Parsing
# ---------------------------------------------------------------------------
def parse_okf_concept(md_path: Path) -> Dict[str, Any]:
    """
    Parse an OKF concept .md file into its frontmatter and body text.
    Returns {"metadata": dict, "text": str} or empty dict on failure.
    """
    try:
        content = md_path.read_text(encoding="utf-8")
        parts = content.split("---", 2)
        if len(parts) < 3:
            logger.warning(f"Skipping {md_path.name}: invalid frontmatter format")
            return {}

        fm = yaml.safe_load(parts[1])
        if not fm or not isinstance(fm, dict):
            logger.warning(f"Skipping {md_path.name}: empty or invalid frontmatter")
            return {}

        body = parts[2].strip()
        return {"metadata": fm, "text": body}
    except Exception as e:
        logger.error(f"Failed to parse {md_path}: {e}")
        return {}


def process_single_concept(md_path: Path) -> List[Dict[str, Any]]:
    """
    Process a single OKF concept file into page-level passages
    suitable for chunking. Returns list of {"text": str, "metadata": dict}.
    """
    parsed = parse_okf_concept(md_path)
    if not parsed:
        return []

    fm = parsed["metadata"]
    body = parsed["text"]

    # Skip failed extractions
    if fm.get("extraction_method") == "failed":
        logger.info(f"Skipping {md_path.name}: extraction_method=failed")
        return []

    # Skip empty bodies
    clean_body = re.sub(r'<!--.*?-->', '', body, flags=re.DOTALL).strip()
    clean_body = re.sub(r'^#+\s*Page\s+\d+\s*$', '', clean_body, flags=re.MULTILINE).strip()
    if len(clean_body) < 50:
        logger.info(f"Skipping {md_path.name}: body too short ({len(clean_body)} chars)")
        return []

    # Derive document ID from concept path
    concept_id = md_path.stem
    doc_type = fm.get("type", "unknown")
    language = fm.get("language", "en")
    resource = fm.get("resource", "")
    tags = fm.get("tags", [])

    # Split body on ## Page N headers to get per-page passages
    page_pattern = re.compile(r'^##\s+Page\s+(\d+)\s*$', re.MULTILINE)
    page_splits = page_pattern.split(body)

    passages: List[Dict[str, Any]] = []

    if len(page_splits) > 1:
        # Format: [pre_text, page_num_1, page_text_1, page_num_2, page_text_2, ...]
        for j in range(1, len(page_splits), 2):
            page_num = int(page_splits[j])
            page_text = page_splits[j + 1].strip() if j + 1 < len(page_splits) else ""
            if page_text:
                passages.append({
                    "text": page_text,
                    "metadata": {
                        "document_id": concept_id,
                        "document_type": doc_type,
                        "language": language,
                        "page_number": page_num,
                        "source_pdf": resource,
                    }
                })
    else:
        # No page headers — treat entire body as a single passage
        passages.append({
            "text": clean_body,
            "metadata": {
                "document_id": concept_id,
                "document_type": doc_type,
                "language": language,
                "page_number": 1,
                "source_pdf": resource,
            }
        })

    return passages


# ---------------------------------------------------------------------------
# Legacy PDF Processing (retained for --legacy-pdf mode)
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    if not text: return ""
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\bPage \d+ of \d+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(?i)^\s*page \d+\s*$', '', text)
    return text.strip()


def _process_single_pdf_legacy(pdf_path: Path) -> List[Dict[str, Any]]:
    """Legacy: extract text directly from a PDF file."""
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


# ---------------------------------------------------------------------------
# Shared: chunking, embedding, storage
# ---------------------------------------------------------------------------
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


def store_in_chromadb(chunks: List[Dict[str, Any]], embeddings: list, db_dir: Path, collection_name: str):
    db_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Storing in ChromaDB at {db_dir}...")
    client = chromadb.PersistentClient(path=str(db_dir))
    collection = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    ids = [f"{c['metadata']['document_id']}_p{c['metadata']['page_number']}_c{c['metadata']['chunk_index']}" for c in chunks]
    # ChromaDB requires all metadata values to be str, int, float, or bool
    metadatas = []
    for c in chunks:
        m = {}
        for k, v in c["metadata"].items():
            if isinstance(v, list):
                m[k] = ", ".join(str(x) for x in v)
            else:
                m[k] = v
        metadatas.append(m)
    texts = [c["text"] for c in chunks]

    for i in range(0, len(ids), CHROMA_BATCH):
        collection.add(
            ids=ids[i:i+CHROMA_BATCH],
            embeddings=embeddings[i:i+CHROMA_BATCH],
            metadatas=metadatas[i:i+CHROMA_BATCH],
            documents=texts[i:i+CHROMA_BATCH]
        )
    logger.info(f"Successfully stored {len(ids)} chunks in ChromaDB '{collection_name}'.")


# ---------------------------------------------------------------------------
# Ingestion modes
# ---------------------------------------------------------------------------
def ingest_okf():
    """Ingest from OKF bundle (default mode)."""
    if not OKF_BUNDLE_DIR.exists():
        logger.error(
            f"OKF bundle not found at {OKF_BUNDLE_DIR}/. "
            "Run 'python convert_to_okf.py' first, or use '--legacy-pdf' for raw PDF ingestion."
        )
        return

    # Collect all .md concept files (exclude index.md, log.md)
    md_files = [
        f for f in sorted(OKF_BUNDLE_DIR.rglob("*.md"))
        if f.name not in ("index.md", "log.md")
    ]

    if not md_files:
        logger.warning(f"No concept files found in {OKF_BUNDLE_DIR}/.")
        return

    logger.info(f"Starting OKF ingestion for {len(md_files)} concept files...")

    # Process all concepts
    all_passages: List[Dict[str, Any]] = []
    processed = 0
    skipped = 0

    for md_path in md_files:
        passages = process_single_concept(md_path)
        if passages:
            all_passages.extend(passages)
            processed += 1
        else:
            skipped += 1

    logger.info(f"Processed {processed} concepts, skipped {skipped} (failed/empty).")

    if not all_passages:
        logger.warning("No text extracted from OKF concepts.")
        return

    chunks = chunk_text(all_passages)
    embeddings = embed_chunks(chunks)
    store_in_chromadb(chunks, embeddings, CHROMA_DB_DIR, COLLECTION_NAME)


def ingest_legacy_pdf():
    """Legacy mode: ingest raw PDFs from data/ (--legacy-pdf)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    pdf_files = [p for p in DATA_DIR.rglob("*.pdf") if "chroma_db" not in str(p) and "legacy_data" not in str(p)]

    if not pdf_files:
        logger.warning(f"No PDFs found in {DATA_DIR}/.")
        return

    logger.info(f"Starting legacy PDF ingestion for {len(pdf_files)} files...")

    all_pages_data: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_pdf = {executor.submit(_process_single_pdf_legacy, pdf): pdf for pdf in pdf_files}
        for future in as_completed(future_to_pdf):
            try:
                all_pages_data.extend(future.result())
                logger.info(f"Successfully processed: {future_to_pdf[future].name}")
            except Exception as e:
                logger.error(f"Failed to process {future_to_pdf[future].name}: {e}")

    if not all_pages_data:
        logger.warning("No text extracted.")
        return

    chunks = chunk_text(all_pages_data)
    embeddings = embed_chunks(chunks)
    store_in_chromadb(chunks, embeddings, CHROMA_DB_DIR, COLLECTION_NAME)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB vector store.")
    parser.add_argument(
        "--legacy-pdf",
        action="store_true",
        help="Ingest raw PDFs from data/ instead of OKF bundle.",
    )
    args = parser.parse_args()

    if args.legacy_pdf:
        logger.info("Running in legacy PDF mode...")
        ingest_legacy_pdf()
    else:
        logger.info("Running in OKF mode (default)...")
        ingest_okf()


if __name__ == "__main__":
    main()
