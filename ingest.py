import os
import re
import fitz  # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer
import tiktoken
from typing import List, Dict, Any
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = Path("data")
RAW_PDF_DIR = DATA_DIR / "raw_pdfs"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"
COLLECTION_NAME = "document_chunks"
MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MAX_WORKERS = os.cpu_count() or 4

def clean_text(text: str) -> str:
    """
    Cleans extracted text by removing extra spaces, fixing hyphenation,
    and attempting to remove common header/footer patterns.
    """
    if not text:
        return ""
    
    # Fix hyphenation across newlines (e.g., "trans-\nformer" -> "transformer")
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    
    # Attempt to remove basic header/footer artifacts (like page numbers)
    text = re.sub(r'\bPage \d+ of \d+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(?i)^\s*page \d+\s*$', '', text)
    
    return text.strip()

def process_single_pdf(pdf_path: Path) -> List[Dict[str, Any]]:
    """
    Extracts and cleans text from a single PDF. Designed to be run in parallel.
    """
    pages_data = []
    try:
        doc = fitz.open(pdf_path)
        doc_id = pdf_path.stem
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text")
            cleaned_text = clean_text(text)
            
            if cleaned_text:
                pages_data.append({
                    "text": cleaned_text,
                    "metadata": {
                        "document_id": doc_id,
                        "page_number": page_num + 1
                    }
                })
        doc.close()
    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {e}")
        
    return pages_data

def extract_text_parallel(pdf_files: List[Path]) -> List[Dict[str, Any]]:
    """
    Uses ThreadPoolExecutor to process multiple PDFs concurrently.
    """
    all_pages_data = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_pdf = {executor.submit(process_single_pdf, pdf): pdf for pdf in pdf_files}
        for future in as_completed(future_to_pdf):
            pdf_path = future_to_pdf[future]
            try:
                pages_data = future.result()
                all_pages_data.extend(pages_data)
                logger.info(f"Successfully processed: {pdf_path.name}")
            except Exception as e:
                logger.error(f"Failed to process {pdf_path.name}: {e}")
                
    return all_pages_data

def chunk_text(pages_data: List[Dict[str, Any]], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Dict[str, Any]]:
    """
    Splits text into chunks of `chunk_size` tokens with `overlap` tokens overlap.
    Maintains metadata across chunks.
    """
    # Use standard cl100k_base tokenizer for accurate token counting
    try:
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception as e:
        logger.warning(f"Failed to load tiktoken encoding: {e}. Falling back to basic word splitting.")
        enc = None

    chunks = []
    chunk_index = 0
    
    for page in pages_data:
        text = page["text"]
        metadata = page["metadata"].copy()
        
        if enc:
            tokens = enc.encode(text)
            start = 0
            while start < len(tokens):
                end = min(start + chunk_size, len(tokens))
                chunk_tokens = tokens[start:end]
                chunk_text = enc.decode(chunk_tokens)
                
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_index"] = chunk_index
                
                chunks.append({
                    "text": chunk_text,
                    "metadata": chunk_metadata
                })
                
                chunk_index += 1
                start += chunk_size - overlap
        else:
            # Fallback: simple word-based splitting (approximate tokens)
            words = text.split()
            start = 0
            while start < len(words):
                end = min(start + chunk_size, len(words))
                chunk_text = " ".join(words[start:end])
                
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_index"] = chunk_index
                
                chunks.append({
                    "text": chunk_text,
                    "metadata": chunk_metadata
                })
                
                chunk_index += 1
                start += chunk_size - overlap
                
    return chunks

def ingest_documents():
    """
    Main pipeline to extract, clean, chunk, embed, and store documents.
    """
    # 1. Setup & Directory Management
    RAW_PDF_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
    
    pdf_files = list(RAW_PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDFs found in {RAW_PDF_DIR}. Please add some and run again.")
        return
        
    logger.info(f"Starting ingestion pipeline for {len(pdf_files)} PDFs...")
    
    # 2. Extraction (Parallel)
    logger.info("Step 1: Extracting text from PDFs...")
    all_pages_data = extract_text_parallel(pdf_files)
    logger.info(f"Extracted {len(all_pages_data)} total pages.")
    
    if not all_pages_data:
        logger.warning("No text extracted. Exiting.")
        return
        
    # 3. Chunking
    logger.info("Step 2: Semantic Chunking...")
    all_chunks = chunk_text(all_pages_data)
    logger.info(f"Created {len(all_chunks)} chunks (Size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}).")
    
    # 4. Embedding
    logger.info(f"Step 3: Loading embedding model ({MODEL_NAME})...")
    model = SentenceTransformer(MODEL_NAME)
    
    logger.info("Generating embeddings...")
    texts = [chunk["text"] for chunk in all_chunks]
    
    # SentenceTransformer handles batching automatically via `batch_size` parameter
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True).tolist()
    
    # 5. Vector Store Setup & Insertion
    logger.info("Step 4: Storing in ChromaDB...")
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    
    # Initialize collection with cosine similarity suitable for sentence-transformers
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    
    # Generate unique IDs for each chunk
    ids = [
        f"{chunk['metadata']['document_id']}_p{chunk['metadata']['page_number']}_c{chunk['metadata']['chunk_index']}" 
        for chunk in all_chunks
    ]
    metadatas = [chunk["metadata"] for chunk in all_chunks]
    
    # Insert in batches to prevent SQLite limits in Chroma
    BATCH_SIZE = 5000
    for i in range(0, len(ids), BATCH_SIZE):
        collection.add(
            ids=ids[i:i+BATCH_SIZE],
            embeddings=embeddings[i:i+BATCH_SIZE],
            metadatas=metadatas[i:i+BATCH_SIZE],
            documents=texts[i:i+BATCH_SIZE]
        )
        
    logger.info("Ingestion pipeline completed successfully! Database is ready for queries.")

if __name__ == "__main__":
    ingest_documents()
