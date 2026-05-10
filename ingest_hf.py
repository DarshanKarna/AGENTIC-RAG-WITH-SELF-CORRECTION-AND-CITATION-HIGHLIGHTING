"""
ingest_hf.py - Hugging Face Dataset Ingestion Pipeline
=======================================================

Downloads the rag-datasets/rag-mini-bioasq text corpus from Hugging Face,
chunks passages into 500-token windows with 50-token overlap, encodes them
with sentence-transformers/all-MiniLM-L6-v2, and stores the embeddings in
a local ChromaDB vector store for downstream RAG retrieval.

Usage:
    python ingest_hf.py
"""

import chromadb
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import tiktoken
from typing import List, Dict, Any
import logging
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Dataset
HF_DATASET = "rag-datasets/rag-mini-bioasq"
HF_CONFIG = "text-corpus"
HF_SPLIT = "passages"

# Chunking
CHUNK_SIZE = 500      # tokens
CHUNK_OVERLAP = 50    # tokens

# Embedding
MODEL_NAME = "all-MiniLM-L6-v2"

# Storage
DATA_DIR = Path("data")
CHROMA_DB_DIR = DATA_DIR / "chroma_db_hf"
COLLECTION_NAME = "bioasq_chunks"

# Batch sizes
ENCODE_BATCH = 32     # sentence-transformers batch size
CHROMA_BATCH = 5_000  # ChromaDB insertion batch size


# ---------------------------------------------------------------------------
# Step 1 — Download dataset from Hugging Face
# ---------------------------------------------------------------------------
def download_dataset():
    """
    Loads the rag-mini-bioasq text-corpus split from Hugging Face.
    Returns the dataset object (~40k passages, ~24 MB download).
    """
    logger.info(
        f"Downloading dataset: {HF_DATASET} (config={HF_CONFIG}, split={HF_SPLIT}) ..."
    )
    ds = load_dataset(HF_DATASET, HF_CONFIG, split=HF_SPLIT)
    logger.info(f"Loaded {len(ds)} passages from Hugging Face.")
    return ds


# ---------------------------------------------------------------------------
# Step 2 — Extract raw text and filter invalid rows
# ---------------------------------------------------------------------------
def extract_passages(ds) -> List[Dict[str, Any]]:
    """
    Iterates through the dataset and extracts valid text passages.
    Skips rows where the `passage` field is None, empty, or 'nan'.
    """
    passages = []
    skipped = 0

    for row in ds:
        text = row.get("passage")
        doc_id = row.get("id")

        # Filter out NaN / empty entries (known issue in this dataset)
        if not text or not isinstance(text, str) or text.strip().lower() == "nan":
            skipped += 1
            continue

        passages.append({
            "text": text.strip(),
            "metadata": {"dataset_id": int(doc_id)},
        })

    logger.info(
        f"Extracted {len(passages)} valid passages (skipped {skipped} invalid rows)."
    )
    return passages


# ---------------------------------------------------------------------------
# Step 3 — Chunk passages into 500-token windows with 50-token overlap
# ---------------------------------------------------------------------------
def chunk_passages(
    passages: List[Dict[str, Any]],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """
    Splits each passage into chunks of `chunk_size` tokens with `overlap`
    tokens of overlap using tiktoken's cl100k_base encoding.
    Falls back to simple word splitting if tiktoken is unavailable.
    """
    # Load tokenizer
    try:
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception as e:
        logger.warning(
            f"Failed to load tiktoken encoding: {e}. Falling back to word splitting."
        )
        enc = None

    chunks: List[Dict[str, Any]] = []
    global_chunk_idx = 0

    for passage in passages:
        text = passage["text"]
        base_meta = passage["metadata"].copy()

        if enc:
            tokens = enc.encode(text)
            start = 0
            while start < len(tokens):
                end = min(start + chunk_size, len(tokens))
                chunk_text = enc.decode(tokens[start:end])

                chunk_meta = base_meta.copy()
                chunk_meta["chunk_index"] = global_chunk_idx

                chunks.append({"text": chunk_text, "metadata": chunk_meta})
                global_chunk_idx += 1
                start += chunk_size - overlap
        else:
            # Fallback: approximate tokens as whitespace-delimited words
            words = text.split()
            start = 0
            while start < len(words):
                end = min(start + chunk_size, len(words))
                chunk_text = " ".join(words[start:end])

                chunk_meta = base_meta.copy()
                chunk_meta["chunk_index"] = global_chunk_idx

                chunks.append({"text": chunk_text, "metadata": chunk_meta})
                global_chunk_idx += 1
                start += chunk_size - overlap

    logger.info(
        f"Created {len(chunks)} chunks "
        f"(size={chunk_size} tokens, overlap={overlap} tokens)."
    )
    return chunks


# ---------------------------------------------------------------------------
# Step 4 — Embed chunks with sentence-transformers/all-MiniLM-L6-v2
# ---------------------------------------------------------------------------
def embed_chunks(chunks: List[Dict[str, Any]]) -> list:
    """
    Encodes all chunk texts using the all-MiniLM-L6-v2 model.
    Returns a list of embedding vectors.
    """
    logger.info(f"Loading embedding model: {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    texts = [c["text"] for c in chunks]
    logger.info(f"Generating embeddings for {len(texts)} chunks ...")
    embeddings = model.encode(
        texts, batch_size=ENCODE_BATCH, show_progress_bar=True
    ).tolist()

    logger.info("Embeddings generated successfully.")
    return embeddings


# ---------------------------------------------------------------------------
# Step 5 — Store in ChromaDB
# ---------------------------------------------------------------------------
def store_in_chromadb(
    chunks: List[Dict[str, Any]],
    embeddings: list,
) -> None:
    """
    Persists embeddings, metadata, and raw text into a local ChromaDB
    collection using cosine similarity.
    """
    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Initialising ChromaDB at {CHROMA_DB_DIR} ...")

    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Build parallel lists for ChromaDB
    ids = [
        f"hf_{c['metadata']['dataset_id']}_c{c['metadata']['chunk_index']}"
        for c in chunks
    ]
    metadatas = [c["metadata"] for c in chunks]
    documents = [c["text"] for c in chunks]

    # Insert in batches to stay within SQLite limits
    total = len(ids)
    for i in range(0, total, CHROMA_BATCH):
        end = min(i + CHROMA_BATCH, total)
        collection.add(
            ids=ids[i:end],
            embeddings=embeddings[i:end],
            metadatas=metadatas[i:end],
            documents=documents[i:end],
        )
        logger.info(f"  Inserted batch {i // CHROMA_BATCH + 1} ({end}/{total})")

    logger.info(
        f"ChromaDB collection '{COLLECTION_NAME}' now contains "
        f"{collection.count()} chunks."
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    logger.info("=" * 60)
    logger.info("  Hugging Face RAG Ingestion Pipeline")
    logger.info("=" * 60)

    # 1. Download
    ds = download_dataset()

    # 2. Extract
    passages = extract_passages(ds)
    if not passages:
        logger.warning("No valid passages found. Exiting.")
        return

    # 3. Chunk
    chunks = chunk_passages(passages)

    # 4. Embed
    embeddings = embed_chunks(chunks)

    # 5. Store
    store_in_chromadb(chunks, embeddings)

    logger.info("=" * 60)
    logger.info("  Pipeline completed successfully! Database is ready.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
