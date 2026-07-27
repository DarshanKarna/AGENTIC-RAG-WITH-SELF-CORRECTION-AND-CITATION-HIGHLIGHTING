# Agentic RAG with Self-Correction and Citation Highlighting
### A Bilingual (English + Nepali) Legal Advisor System

A 4th-semester AI project at Kathmandu University, implementing an agentic
Retrieval-Augmented Generation (RAG) pipeline for Nepali legal and
regulatory question-answering, with self-correction via NLI-based
hallucination checking and sentence-level citation tracking.

Developed by **Darshan Karna**.

---

## Overview

Generic RAG pipelines answer questions by retrieving text and generating
a response — with no guarantee the response is actually supported by
what was retrieved. This project adds an agentic self-correction loop
on top of retrieval: every generated answer is checked for entailment
against its source context before being returned, and low-confidence
or unsupported answers trigger query reformulation and a retry.

The system is built and evaluated on a real, high-stakes domain — Nepali
statutes, NRB circulars, court verdicts, and regulatory directives —
where citation accuracy and hallucination avoidance genuinely matter,
rather than a generic benchmark dataset.

---

## Corpus

| Metric | Value |
|---|---|
| Source PDFs | 1,073 |
| Categories | 19 (statutes, legislative bills, case law, circulars, regulatory directives, fiscal policy, annual reports) |
| OKF concept files | 1,073 (1,021 direct-extracted, 52 OCR-recovered, 0 failed) |
| Vector store chunks | 80,767 |
| Languages | English + Nepali (Devanagari), including 358 bilingual Act pairs |

Source categories include the Companies Act, BAFIA, Income Tax Act,
NRB Act and circulars, SEBON capital market regulations, Supreme and
High Court verdicts, parliamentary bills, fiscal policy directives,
and energy/procurement regulations.

Raw PDFs are converted into an [Open Knowledge Format (OKF)](https://github.com/GoogleCloudPlatform/knowledge-catalog)
bundle — one Markdown file per document with YAML frontmatter (`type`,
`language`, `title_en`/`title_ne`, source provenance, bilingual
cross-links) — before being chunked and embedded. Scanned/image-based
PDFs (mostly court verdicts) are recovered via an EasyOCR fallback
tier before falling back to a failed-extraction stub.

---

## Architecture

```
User Query
    │
    ▼
Embedding (BAAI/bge-m3, multilingual) → ChromaDB retrieval
    │
    ▼
Relevance Grading (mDeBERTa-v3 NLI) ──fail──> Query Rewriter ──┐
    │ pass                                                      │
    ▼                                                           │
Answer Generation (Qwen3, via Ollama)                           │
    │                                                           │
    ▼                                                           │
Hallucination / Faithfulness Check (mDeBERTa-v3 NLI) ──fail────┘
    │ pass
    ▼
Final Answer + Sentence-Level Citations
```

The graph is implemented with LangGraph in `self_correcting_rag.py`.
A non-agentic `baseline_rag.py` (single-pass retrieval + generation,
no grading, no verification, no citation tracking) is kept as the
naive-RAG comparison point for ablation studies.

### Model stack

| Component | Model | Notes |
|---|---|---|
| Embeddings | `BAAI/bge-m3` | Multilingual dense retrieval (1024-dim), replacing an earlier English-only MiniLM model |
| NLI critic | `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` | Used for both relevance grading and hallucination/faithfulness checking |
| Generator | `Qwen3` (via Ollama) | Chosen for demonstrated Nepali-language performance over comparable-sized alternatives |
| OCR (scanned PDFs) | EasyOCR | GPU-accelerated where available; falls back to Tesseract if EasyOCR fails to initialize |
| Vector store | ChromaDB | Cosine-similarity HNSW index, `data/chroma_db` |

---

## Repository structure

```
data/                  Source PDF corpus, organized by category
okf_bundle/             Generated OKF markdown bundle (gitignored)
legacy_data/             Archived early-stage BioASQ/RAGTruth artifacts
convert_to_okf.py         PDF → OKF markdown converter (with OCR fallback)
ingest.py                 OKF bundle → chunked, embedded ChromaDB store
self_correcting_rag.py    Agentic self-correction RAG pipeline (LangGraph)
baseline_rag.py            Naive RAG baseline for ablation comparison
evaluate.py                 RAGAS-based evaluation suite
api.py                        FastAPI backend
frontend/                      React + Vite web UI
documentation/                 Project proposal, methodology, planning docs
```

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- Node.js & npm (for the React frontend)
- [Ollama](https://ollama.com/) running locally with the `qwen3` model pulled

### Backend Setup

```bash
pip install -r requirements.txt
ollama pull qwen3
```

Ingest the corpus (requires the OKF bundle to already be generated via
`convert_to_okf.py`):

```bash
python convert_to_okf.py       # PDF → OKF markdown (one-time / incremental)
python ingest.py                # OKF bundle → ChromaDB
```

Run the pipeline endpoints:

```bash
python api.py
```

Or run the pipeline standalone:

```bash
python self_correcting_rag.py    # agentic, self-correcting
python baseline_rag.py            # naive baseline, for comparison
```

### Known hardware constraint

Running the full stack (Qwen3 generation + mDeBERTa NLI + bge-m3
embeddings) concurrently can exceed available VRAM/RAM on
lower-spec GPUs (tested failure case: 6GB VRAM / 16GB system RAM).
`convert_to_okf.py`'s OCR step includes a concurrency guard that
blocks running alongside `ingest.py` for the same reason. If you hit
an out-of-memory error during generation, consider reducing Ollama's
context window (`num_ctx`) before switching to a smaller model.

### Frontend Setup

1. **Navigate to the Frontend Directory**
   ```bash
   cd frontend
   ```

2. **Install Packages**
   ```bash
   npm install
   ```

3. **Start the Dev Server**
   ```bash
   npm run dev
   ```
   Open the provided local URL (typically `http://localhost:5173`) in your browser.

---

## API Endpoints

The FastAPI server exposes the following endpoints:

### 1. POST `/api/chat`
Submit a question to run through the self-correcting agentic pipeline.

* **Request Body:**
  ```json
  {
    "question": "What is the penalty for tax evasion under the Income Tax Act?"
  }
  ```

* **Response Body:**
  ```json
  {
    "baseline": {
      "answer": "Baseline answer...",
      "hallucinated_sentences": ["unverified sentence..."]
    },
    "corrected": {
      "answer": "Corrected and verified answer...",
      "citations": [
        {
          "sentence": "Verified sentence...",
          "citation": "source_document.pdf (Page 2)"
        }
      ]
    }
  }
  ```

### 2. POST `/api/upload`
Dynamically upload a PDF document. Chunks and indexes the document pages into ChromaDB vector store on the fly.

* **Request Multi-part Form:**
  - `file`: PDF file

---

## Evaluation

`evaluate.py` runs a RAGAS-based benchmark (faithfulness, answer
relevance, context precision/recall) comparing the self-correcting
pipeline against the naive `baseline_rag.py` baseline. Evaluation is
intended to run against a hand-built adversarial question set
(straightforward lookups, cross-references, superseded-provision
traps, out-of-corpus questions, and bilingual pairs) rather than a
generic benchmark, to directly test the claims this project makes
about hallucination resistance and citation accuracy.

---

## Project status

This project began as a generic RAG demo evaluated on a public
biomedical QA benchmark, and was refocused onto a real, high-stakes,
bilingual legal domain — with an expanded, verified corpus, a
multilingual model stack, and OCR recovery for scanned court
documents. See `documentation/` for the original proposal and
methodology, and the project roadmap for planned next steps
(fine-tuning, procedural/advisor question support, and formal
ablation studies).

---

**Author**: Darshan Karna  
**License**: MIT
