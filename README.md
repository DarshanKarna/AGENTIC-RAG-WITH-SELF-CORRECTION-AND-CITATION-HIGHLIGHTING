# Agentic RAG with Self-Correction and Citation Highlighting

A robust Retrieval-Augmented Generation (RAG) system built with LangGraph, ChromaDB, and Groq (Llama 3). This project demonstrates an agentic pipeline with sentence-level hallucination detection, dynamic self-correction, and query reformulation.

Developed as a B.Tech AI 4th-Semester Project by **Darshan Karna**.

## Features

- **Agentic LangGraph Pipeline**: A state machine that intelligently routes queries, grades documents, and triggers self-correction loops.
- **NLI-Based Hallucination Critic**: Uses a local DeBERTa Cross-Encoder (`nli-deberta-base`) to verify every generated sentence against retrieved context.
- **Self-Correction & Fallback**: Automatically reformulates queries if retrieved documents are irrelevant, and regenerates answers if hallucinations are detected.
- **Unified Evaluation Suite**: Integrates **RAGAS** metrics to compare Naive RAG vs. Self-Correcting RAG using Faithfulness and Answer Relevancy scores.
- **Flexible Data Ingestion**: Supports parsing local PDFs and downloading remote HuggingFace datasets (e.g., `rag-datasets/rag-mini-bioasq`) into a local ChromaDB vector store.
- **Citation Highlighting**: Tracks which chunks of source documents support each sentence, enabling transparent and trustworthy citations.

## System Architecture

1. **Retrieval**: Uses `all-MiniLM-L6-v2` SentenceTransformers to query `ChromaDB`.
2. **Relevance Grading**: An LLM critic checks if retrieved documents answer the question. If not, the query is reformulated.
3. **Draft Generation**: Llama 3 generates an initial draft answer.
4. **NLI Verification**: The DeBERTa critic scores the entailment of each sentence. Unsubstantiated claims are flagged as hallucinations.
5. **Regeneration**: If hallucinations exist, the draft is sent back to the generator with specific instructions to remove or fix the flagged claims.

## Setup & Installation

### Prerequisites

- Python 3.10+
- [Groq API Key](https://console.groq.com/) for Llama 3 access.

### Installation

1. **Clone the repository**

   ```bash
   git clone <your-repo-url>
   cd <your-repo-directory>
   ```

2. **Create a virtual environment and install dependencies**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**
   Create a `.env` file in the root directory and add your API keys:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

## Usage

### 1. Data Ingestion

You can ingest data from local PDFs or HuggingFace datasets into the vector database.

```bash
# Ingest local PDFs (place PDFs in data/raw_pdfs/)
python ingest.py --source pdf

# Ingest HuggingFace dataset (rag-datasets/rag-mini-bioasq)
python ingest.py --source hf
```

### 2. Running the Agentic Pipeline

Run the main script to test the LangGraph self-correcting RAG pipeline directly:

```bash
python self_correcting_rag.py
```

### 3. Running Evaluations

Evaluate the improvements of the self-correcting pipeline over a baseline naive RAG using RAGAS:

```bash
python evaluate.py
```

## Project Structure

- `self_correcting_rag.py`: The core LangGraph state machine and LLM interaction logic.
- `ingest.py`: Unified ingestion pipeline supporting `--source pdf` and `--source hf`.
- `evaluate.py`: RAGAS evaluation suite to calculate the "Correction Delta".
- `baseline_rag.py`: A simple Naive RAG implementation for comparison.
- `api.py`: FastAPI backend to serve the application.
- `frontend/`: Frontend application code.
- `data/`: Contains the raw documents and ChromaDB vector stores.

---

**Author**: Darshan Karna
