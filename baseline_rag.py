"""
baseline_rag.py - Naive RAG Baseline (Ablation Comparison)
===========================================================

Connects to the active Nepali legal corpus ChromaDB vector store (in data/),
retrieves the top-k most relevant chunks for a Nepali legal query, and
generates an unverified single-pass answer using local Qwen3 via Ollama.

This script serves as the "naive RAG baseline" for ablation studies against
self_correcting_rag.py. It intentionally has:
  - NO relevance grading / query reformulation
  - NO NLI-based critic / hallucination verification
  - NO sentence-level citation tracking

Usage:
    python baseline_rag.py
"""

import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHROMA_DB_DIR = os.path.join("data", "chroma_db")
COLLECTION_NAME = "document_chunks"
EMBEDDING_MODEL = "BAAI/bge-m3"
LLM_MODEL = "qwen3"
TOP_K = 5
QUERY = "नेपालको संविधान अनुसार नागरिकका प्रमुख मौलिक हकहरू के के हुन्?"

SYSTEM_INSTRUCTION = (
    "You are a helpful legal assistant for Nepali legal queries. "
    "Answer the question strictly based on the provided context passages. "
    "Do not introduce any external information or facts."
)


# ---------------------------------------------------------------------------
# 2. Initialize ChromaDB retriever via LangChain
# ---------------------------------------------------------------------------
def init_retriever():
    """
    Wraps the existing ChromaDB store with the same embedding model
    used during ingestion (BAAI/bge-m3) and returns a retriever
    configured for top-k=5 cosine similarity search.
    """
    embedding_fn = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vectorstore = Chroma(
        persist_directory=CHROMA_DB_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )
    return retriever


# ---------------------------------------------------------------------------
# 3. Initialize the Local Ollama LLM (Qwen3)
# ---------------------------------------------------------------------------
def init_llm():
    """Creates a ChatOllama instance with local Qwen3."""
    llm = ChatOllama(
        model=LLM_MODEL,
        temperature=0,  # deterministic for reproducibility
        base_url="http://localhost:11434"
    )
    return llm


# ---------------------------------------------------------------------------
# 4. Build the RAG chain (LCEL)
# ---------------------------------------------------------------------------
def format_docs(docs):
    """Concatenates retrieved document contents into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(retriever, llm):
    """
    Constructs a LangChain LCEL retrieval chain:
      query → retriever → format docs into context → prompt → LLM → answer
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_INSTRUCTION),
        ("human",
          "Context:\n{context}\n\n"
          "Question: {question}"),
    ])

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# ---------------------------------------------------------------------------
# 5. Pretty-print results
# ---------------------------------------------------------------------------
def print_retrieved_chunks(docs):
    """Formats and prints the retrieved chunks with metadata."""
    print("\n" + "=" * 70)
    print(f"  RETRIEVED CHUNKS (top {TOP_K})")
    print("=" * 70)

    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        doc_id = meta.get("document_id", "N/A")
        page_num = meta.get("page_number", "N/A")
        chunk_idx = meta.get("chunk_index", "N/A")
        print(f"\n--- Chunk {i} ---")
        print(f"  Document ID: {doc_id}")
        print(f"  Page Number: {page_num}")
        print(f"  Chunk Index: {chunk_idx}")
        # Show first 300 characters of text
        snippet = doc.page_content[:300].replace("\n", " ")
        print(f"  Text       : {snippet}...")

    print("\n" + "-" * 70)


def print_answer(answer: str):
    """Formats and prints the LLM's generated answer."""
    print("\n" + "=" * 70)
    print("  LLM GENERATED ANSWER")
    print("=" * 70)
    print(f"\n{answer}\n")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"\nQuery: \"{QUERY}\"\n")

    # 2. Retriever
    print("Initializing retriever (loading embedding model)...")
    retriever = init_retriever()

    # 3. Retrieve chunks (separately for display)
    print("Retrieving relevant chunks...")
    retrieved_docs = retriever.invoke(QUERY)
    print_retrieved_chunks(retrieved_docs)

    # 4. LLM
    print(f"\nInitializing LLM ({LLM_MODEL} via local Ollama)...")
    llm = init_llm()

    # 5. Build chain & generate answer
    print("Building RAG chain and generating answer...")
    rag_chain = build_rag_chain(retriever, llm)
    answer = rag_chain.invoke(QUERY)

    # 6. Output
    print_answer(answer)


if __name__ == "__main__":
    main()
