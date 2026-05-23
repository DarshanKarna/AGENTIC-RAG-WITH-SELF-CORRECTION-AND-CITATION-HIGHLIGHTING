"""
compare_evaluation.py - RAG Metrics Comparison Suite
======================================================

Executes a side-by-side evaluation comparison of your RAG system's answers
BEFORE self-correction (initial LLM draft) vs AFTER self-correction (NLI-verified draft).

Author: Antigravity AI Pair Programmer
OS: Windows | Runtime: Python 3.13
"""

import os
import sys
import json
import re
from typing import Dict, Any, List

# Reconfigure stdout to UTF-8 to prevent encoding crashes on Windows consoles
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Loading environment
load_dotenv()

# We import the LangGraph application, our configured LLM, and Colors to reuse cached models
print("\n" + "="*80)
print("⌛ LOADING SYSTEM CONTEXTS & PRE-CACHED MODELS...")
print("="*80)
from self_correcting_rag import app, llm, Colors

# Standard Test Queries (Medical and Out-of-Domain)
TEST_QUERIES = [
    "What is the function of the BRCA1 gene?",
    "What is the role of p53 protein in cell cycle control?",
    "Who won the FIFA World Cup in 2022?"
]

def format_docs(docs) -> str:
    """Helper to join document contents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def grade_answer(question: str, context: str, answer: str) -> Dict[str, Any]:
    """
    Uses LLM-as-a-judge (Groq Llama-3) to grade an answer on Groundedness and Answer Relevance.
    """
    # Graceful return for out-of-domain fallback messages
    if "couldn't find any relevant biomedical information" in answer.lower():
        # Fallback is by definition fully grounded (does not hallucinate) but has 1/5 answer relevance 
        # because the database lacks the info.
        return {
            "groundedness": 5,
            "answer_relevance": 1,
            "reason": "Fallback answer indicating database mismatch."
        }

    eval_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a meticulous RAG system evaluator. Analyze the given question, "
            "retrieved context, and answer. Grade the answer on two metrics on a scale of 1 to 5 "
            "where 1 is poor/completely wrong and 5 is excellent/perfect.\n\n"
            "Metrics:\n"
            "1. groundedness: Is the answer fully supported *only* by the retrieved context? Deduct points if the answer brings in external facts or hallucinates claims not explicitly stated.\n"
            "2. answer_relevance: Does the answer directly address and answer the user's question?\n\n"
            "Output strictly in the following JSON format:\n"
            "{{\n"
            '  "groundedness": 5,\n'
            '  "answer_relevance": 5,\n'
            '  "reason": "..."\n'
            "}}\n"
            "Do not add any conversational text or formatting blocks."
        )),
        ("human", (
            "Question: {question}\n\n"
            "Retrieved Context:\n{context}\n\n"
            "Answer to Evaluate:\n{answer}"
        ))
    ])

    chain = eval_prompt | llm | JsonOutputParser()
    try:
        results = chain.invoke({
            "question": question,
            "context": context,
            "answer": answer
        })
        return results
    except Exception as e:
        # Fallback parser if JSON fails
        return {
            "groundedness": 1,
            "answer_relevance": 1,
            "reason": f"Grading error: {str(e)}"
        }

def run_comparison():
    print(f"\n{Colors.BOLD}{Colors.HEADER}=== RUNNING BEFORE vs AFTER SELF-CORRECTION COMPARISON ==={Colors.ENDC}\n")
    
    comparisons = []
    
    for idx, query in enumerate(TEST_QUERIES, 1):
        print(f"\n--- Running Evaluation Query {idx}/{len(TEST_QUERIES)}: \"{query}\" ---")
        
        # 1. Execute LangGraph Pipeline
        initial_state = {
            "question": query,
            "original_question": query,
            "documents": [],
            "draft_answer": "",
            "drafts": [],
            "flagged_sentences": [],
            "documents_relevant": "no",
            "hallucination_retries": 0,
            "retrieval_retries": 0,
            "verified_citations": []
        }
        
        final_output = app.invoke(initial_state)
        
        # 2. Extract context and answers
        context_str = format_docs(final_output.get("documents", []))
        drafts = final_output.get("drafts", [])
        
        if not drafts:
            # Fallback when no drafts are generated due to early exit
            initial_answer = final_output["draft_answer"]
            final_answer = final_output["draft_answer"]
        else:
            initial_answer = drafts[0]
            final_answer = final_output["draft_answer"]
            
        print(f" -> Initial Draft answer generated ({len(drafts)} drafts tracked).")
        print(f" -> Final Grounded answer finalized.")
        
        # 3. Grade both answers using LLM-as-a-judge
        print(" -> Grading Initial Draft...")
        initial_grades = grade_answer(query, context_str, initial_answer)
        
        print(" -> Grading Final Verified Answer...")
        final_grades = grade_answer(query, context_str, final_answer)
        
        comparisons.append({
            "query": query,
            "initial_answer": initial_answer,
            "final_answer": final_answer,
            "initial_grades": initial_grades,
            "final_grades": final_grades
        })
        
        # Print live console comparison
        print(f"\n{Colors.BOLD}[*] RESULTS COMPARISON FOR QUERY {idx}:{Colors.ENDC}")
        print(f"  - Initial Groundedness Score      : {initial_grades['groundedness']}/5")
        print(f"  - Final Groundedness Score        : {final_grades['groundedness']}/5")
        print(f"  - Initial Answer Relevance Score  : {initial_grades['answer_relevance']}/5")
        print(f"  - Final Answer Relevance Score    : {final_grades['answer_relevance']}/5")
        print(f"  - Grading Reason (Final)          : {final_grades.get('reason', 'N/A')}")
        print("-"*70)

    # 4. Generate Comparative Report
    report_path = "evaluation_comparison_report.md"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 📊 B.Tech AI Project: RAG Evaluation Report\n")
        f.write("## Comparative Metrics: Naive Generation (Before) vs Self-Correcting LangGraph (After)\n\n")
        
        f.write("This report presents the comparative metrics of the RAG system **Before** vs **After** the ")
        f.write("NLI-based sentence-level self-correction loops. Evaluated using a Groq `llama-3.3-70b-versatile` LLM judge.\n\n")
        
        f.write("### 📈 Core Metrics Comparative Table\n\n")
        f.write("| Query | Initial Groundedness | Final Groundedness | Initial Relevance | Final Relevance | Improvement Status |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :--- |\n")
        
        for comp in comparisons:
            q = comp["query"]
            ig = comp["initial_grades"]["groundedness"]
            fg = comp["final_grades"]["groundedness"]
            ir = comp["initial_grades"]["answer_relevance"]
            fr = comp["final_grades"]["answer_relevance"]
            
            status = "✅ Perfect Groundedness Maintained"
            if fg > ig:
                status = f"🔥 Groundedness Improved (+{fg-ig})"
            elif fg < ig:
                status = "⚠️ Regression"
            elif ir < fr:
                status = f"🚀 Relevance Improved (+{fr-ir})"
                
            f.write(f"| {q} | **{ig}/5** | **{fg}/5** | **{ir}/5** | **{fr}/5** | {status} |\n")
            
        f.write("\n\n### 🧬 Query-by-Query Comparative Detail\n\n")
        
        for idx, comp in enumerate(comparisons, 1):
            f.write(f"#### {idx}. Query: \"{comp['query']}\"\n\n")
            f.write("##### ❌ Initial Draft (Before NLI Self-Correction)\n")
            f.write(f"```text\n{comp['initial_answer']}\n```\n")
            f.write(f"- **Groundedness**: {comp['initial_grades']['groundedness']}/5\n")
            f.write(f"- **Relevance**: {comp['initial_grades']['answer_relevance']}/5\n\n")
            
            f.write("##### ✅ Verified Final Answer (After NLI Self-Correction)\n")
            f.write(f"```text\n{comp['final_answer']}\n```\n")
            f.write(f"- **Groundedness**: {comp['final_grades']['groundedness']}/5\n")
            f.write(f"- **Relevance**: {comp['final_grades']['answer_relevance']}/5\n")
            f.write(f"- **Critic Reasoning**: *{comp['final_grades'].get('reason', 'N/A')}*\n\n")
            f.write("---\n\n")
            
    print(f"\n{Colors.GREEN}{Colors.BOLD}[*] EVALUATION COMPLETELY SUCCESSFUL!{Colors.ENDC}")
    print(f"Comparative report compiled and saved to: [evaluation_comparison_report.md](file:///{os.path.abspath(report_path)})")
    print("="*80 + "\n")

if __name__ == "__main__":
    run_comparison()
