"""
baseline_rag.py - Naive RAG Retrieval & Generation
====================================================

Connects to the local ChromaDB vector store (populated by ingest_hf.py),
retrieves the top-k most relevant chunks for a hardcoded query, and
generates a grounded answer using Google Gemini via LangChain.

Usage:
    python baseline_rag.py
"""

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHROMA_DB_DIR = os.path.join("data", "chroma_db_hf")
COLLECTION_NAME = "bioasq_chunks"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gemini-2.0-flash"
TOP_K = 5
QUERY = "What is the function of the BRCA1 gene?"

SYSTEM_INSTRUCTION = (
    "Answer strictly based on the provided context. "
    "Do not introduce any information not in the context passages. "
    "Output your answer in standalone sentences, each stating one claim."
)


# ---------------------------------------------------------------------------
# 1. Load environment variables
# ---------------------------------------------------------------------------
def load_env():
    """Load the GOOGLE_API_KEY from the .env file."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY not found. Please set it in your .env file."
        )
    return api_key


# ---------------------------------------------------------------------------
# 2. Initialize ChromaDB retriever via LangChain
# ---------------------------------------------------------------------------
def init_retriever():
    """
    Wraps the existing ChromaDB store with the same embedding model
    used during ingestion (all-MiniLM-L6-v2) and returns a retriever
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
# 3. Initialize the Groq LLM
# ---------------------------------------------------------------------------
def init_llm(api_key: str):
    """Creates a ChatGoogleGenerativeAI instance with Gemini."""
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=api_key,
        temperature=0,  # deterministic for reproducibility
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
        chunk_id = f"hf_{meta.get('dataset_id', '?')}_c{meta.get('chunk_index', '?')}"
        print(f"\n--- Chunk {i} ---")
        print(f"  ID        : {chunk_id}")
        print(f"  dataset_id: {meta.get('dataset_id', 'N/A')}")
        print(f"  chunk_idx : {meta.get('chunk_index', 'N/A')}")
        # Show first 300 characters of text
        snippet = doc.page_content[:300].replace("\n", " ")
        print(f"  Text      : {snippet}...")

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

    # 1. Environment
    api_key = load_env()

    # 2. Retriever
    print("Initializing retriever (loading embedding model)...")
    retriever = init_retriever()

    # 3. Retrieve chunks (separately for display)
    print("Retrieving relevant chunks...")
    retrieved_docs = retriever.invoke(QUERY)
    print_retrieved_chunks(retrieved_docs)

    # 4. LLM
    print(f"\nInitializing LLM ({LLM_MODEL} via Google Gemini)...")
    llm = init_llm(api_key)

    # 5. Build chain & generate answer
    print("Building RAG chain and generating answer...")
    rag_chain = build_rag_chain(retriever, llm)
    answer = rag_chain.invoke(QUERY)

    # 6. Output
    print_answer(answer)


if __name__ == "__main__":
    main()
