"""
evaluate.py - Unified RAG Evaluation Suite
=====================================================
Evaluates the Baseline RAG vs Self-Correcting RAG using RAGAS metrics 
(Faithfulness and Answer Relevance) as defined in the project scope.

Usage:
    python evaluate.py
"""

import os
import sys

# Ensure UTF-8 standard output on Windows console
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

# Set up LangChain groq for RAGAS evaluation
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# Import our LangGraph application to get baseline and corrected drafts
from self_correcting_rag import app

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TEST_QUERIES = [
    "What is the function of the BRCA1 gene?",
    "What is the role of p53 protein in cell cycle control?",
    "Which gene mutations lead to cystic fibrosis?",
    "Explain the mechanism of quantum entanglement in quantum computing.",
    "Who won the FIFA World Cup in 2022?"
]

# Configure Groq LLM and HuggingFace Embeddings to be used by RAGAS natively
EVAL_LLM = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
EVAL_EMBEDDINGS = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def calculate_correction_delta(baseline_faithfulness: float, corrected_faithfulness: float) -> float:
    """
    Calculates the 'Correction Delta' (the difference in faithfulness between a 
    baseline naive RAG answer and the final self-corrected answer).
    """
    return corrected_faithfulness - baseline_faithfulness

def run_evaluation():
    print("======================================================================")
    print("              STARTING UNIFIED RAGAS EVALUATION SUITE                 ")
    print("======================================================================\n")

    baseline_data = {"question": [], "answer": [], "contexts": []}
    corrected_data = {"question": [], "answer": [], "contexts": []}

    for idx, query in enumerate(TEST_QUERIES, 1):
        print(f"--- Processing Query {idx}/{len(TEST_QUERIES)}: \"{query}\" ---")
        
        initial_state = {
            "question": query,
            "original_question": query,
            "documents": [],
            "draft_answer": "",
            "baseline_draft": "",
            "baseline_hallucinated": [],
            "drafts": [],
            "flagged_sentences": [],
            "documents_relevant": "no",
            "hallucination_retries": 0,
            "retrieval_retries": 0,
            "verified_citations": []
        }
        
        # Invoke the LangGraph pipeline
        final_output = app.invoke(initial_state)
        
        # Extract contexts
        contexts = [doc.page_content for doc in final_output.get("documents", [])]
        if not contexts:
            contexts = [""] # RAGAS requires non-empty lists to process
            
        # Extract answers
        baseline_ans = final_output.get("baseline_draft", final_output["draft_answer"])
        corrected_ans = final_output["draft_answer"]
        
        baseline_data["question"].append(query)
        baseline_data["answer"].append(baseline_ans)
        baseline_data["contexts"].append(contexts)
        
        corrected_data["question"].append(query)
        corrected_data["answer"].append(corrected_ans)
        corrected_data["contexts"].append(contexts)

    print("\n[1/2] Evaluating Baseline Responses with RAGAS...")
    baseline_dataset = Dataset.from_dict(baseline_data)
    baseline_results = evaluate(
        baseline_dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=EVAL_LLM,
        embeddings=EVAL_EMBEDDINGS
    )
    
    print("\n[2/2] Evaluating Corrected Responses with RAGAS...")
    corrected_dataset = Dataset.from_dict(corrected_data)
    corrected_results = evaluate(
        corrected_dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=EVAL_LLM,
        embeddings=EVAL_EMBEDDINGS
    )
    
    # Extract average scores safely
    b_faith = baseline_results.get("faithfulness", 0.0)
    c_faith = corrected_results.get("faithfulness", 0.0)
    b_relev = baseline_results.get("answer_relevancy", 0.0)
    c_relev = corrected_results.get("answer_relevancy", 0.0)
    
    delta = calculate_correction_delta(b_faith, c_faith)

    print("\n" + "="*70)
    print("                      OVERALL RAG EVALUATION REPORT                    ")
    print("="*70)
    
    print("\n1. Faithfulness (Groundedness)")
    print(f"   - Baseline Naive RAG   : {b_faith:.4f}")
    print(f"   - Self-Corrected RAG   : {c_faith:.4f}")
    print(f"   -> Correction Delta    : {delta:+.4f}")
    
    print("\n2. Answer Relevance")
    print(f"   - Baseline Naive RAG   : {b_relev:.4f}")
    print(f"   - Self-Corrected RAG   : {c_relev:.4f}")
    
    print("\n" + "="*70)
    print("Evaluation completed successfully.")
    print("="*70)

if __name__ == "__main__":
    run_evaluation()
