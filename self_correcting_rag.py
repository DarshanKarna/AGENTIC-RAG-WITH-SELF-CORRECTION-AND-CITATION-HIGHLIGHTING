"""
self_correcting_rag.py - Self-Correcting Agentic RAG System
============================================================

A B.Tech AI 4th-Semester Project RAG pipeline built with LangGraph, ChromaDB,
and Groq (Llama 3). Performs:
  1. Vector DB retrieval (ChromaDB + SentenceTransformers).
  2. Semantic relevance grading (Llama 3 via ChatGroq).
  3. Agentic fallback query reformulation (Llama 3 via ChatGroq).
  4. Sentence-level NLI-based hallucination critic (nli-deberta-base cross-encoder).
  5. Dynamic answer regeneration logic based on critic feedback.

Author: Antigravity AI Pair Programmer
OS: Windows | Runtime: Python 3.13
"""

import os
import sys
import json
import re
import warnings
from typing import TypedDict, List, Dict, Any

# Ensure UTF-8 standard output on Windows console to prevent UnicodeEncodeError
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# Suppress Hugging Face symlink warnings on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import nltk
from dotenv import load_dotenv

# LangChain & LangGraph Imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

# NLTK sentence tokenizer downloader
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception:
    pass

# Loading global environment
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise EnvironmentError("GROQ_API_KEY not found. Please verify your .env file.")

# Global Configuration Constants
CHROMA_DB_DIR = os.path.join("data", "chroma_db_hf")
COLLECTION_NAME = "bioasq_chunks"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.3-70b-versatile"
NLI_MODEL_NAME = "cross-encoder/nli-deberta-base"

# =====================================================================
# 🎨 COLOR-CODED LOGGING
# =====================================================================
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def log_agent(agent_name: str, message: str, color: str = Colors.CYAN):
    """Prints beautiful, styled, color-coded console logs for live demonstration."""
    print(f"\n{color}{Colors.BOLD}[{agent_name.upper()}]{Colors.ENDC}{color} {message}{Colors.ENDC}")

# =====================================================================
# 🧠 MODELS & DATABASE INITIALIZATION (SINGLETON PATTERN FOR PERFORMANCE)
# =====================================================================
print(f"{Colors.BOLD}{Colors.HEADER}=== INITIALIZING SYSTEM MODELS ==={Colors.ENDC}")

print("[1/3] Initializing Embeddings & local ChromaDB...")
embedding_fn = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = Chroma(
    persist_directory=CHROMA_DB_DIR,
    collection_name=COLLECTION_NAME,
    embedding_function=embedding_fn,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

print("[2/3] Connecting to Groq Cloud LLM...")
llm = ChatGroq(model=LLM_MODEL, api_key=GROQ_API_KEY, temperature=0)

print("[3/3] Loading local NLI DeBERTa Cross-Encoder (The Critic)...")
from sentence_transformers import CrossEncoder
nli_model = CrossEncoder(NLI_MODEL_NAME)

print(f"{Colors.GREEN}{Colors.BOLD}System fully initialized and ready!{Colors.ENDC}\n")

# =====================================================================
# 📊 STATE DEFINITION
# =====================================================================
class GraphState(TypedDict):
    question: str
    original_question: str
    documents: List[Any]
    draft_answer: str
    flagged_sentences: List[str]
    documents_relevant: str  # "yes" | "no"
    hallucination_retries: int
    retrieval_retries: int

# =====================================================================
# 🛠️ HELPER PARSERS & CALCULATORS
# =====================================================================
def parse_grader_response(response_text: str) -> str:
    """Robustly extracts 'yes' or 'no' from LLM document grader outputs."""
    try:
        # Attempt to parse as raw JSON
        clean_text = response_text.strip()
        # Remove markdown JSON code blocks if present
        clean_text = re.sub(r"^```json\s*", "", clean_text)
        clean_text = re.sub(r"\s*```$", "", clean_text)
        
        data = json.loads(clean_text)
        if "binary_score" in data:
            return data["binary_score"].lower().strip()
    except Exception:
        pass
    
    # Regex fallback if LLM included conversational text
    match = re.search(r'"binary_score"\s*:\s*"([^"]+)"', response_text)
    if match:
        return match.group(1).lower().strip()
        
    if "yes" in response_text.lower():
        return "yes"
    return "no"

def compute_entailment_scores(sentence: str, documents: List[Any]) -> float:
    """
    Computes the maximum NLI entailment probability score for a sentence
    against each of the retrieved document chunks.
    """
    if not documents:
        return 0.0
        
    # Cross-encoder pairs: (Context Chunk, Answer Sentence)
    pairs = [(doc.page_content, sentence) for doc in documents]
    
    # Get raw logits from DeBERTa
    # Shape: (num_pairs, 3)
    logits = nli_model.predict(pairs)
    logits = np.atleast_2d(logits)
    
    # Compute Softmax: Label mapping [0: contradiction, 1: entailment, 2: neutral]
    e_x = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = e_x / e_x.sum(axis=-1, keepdims=True)
    
    # Extract entailment probabilities (index 1)
    entailment_probs = probs[:, 1]
    
    # Return highest entailment score among all retrieved passages
    return float(np.max(entailment_probs))

# =====================================================================
# 🕸️ LANGGRAPH NODES
# =====================================================================

def retrieval_node(state: GraphState) -> Dict[str, Any]:
    """Node 1: Retrieves the top k=5 documents from ChromaDB."""
    query = state["question"]
    log_agent("RETRIEVER", f"Retrieving contexts for query: '{query}'...", Colors.BLUE)
    
    docs = retriever.invoke(query)
    log_agent("RETRIEVER", f"Retrieved {len(docs)} chunks from ChromaDB.", Colors.BLUE)
    
    return {"documents": docs}

def document_grader_node(state: GraphState) -> Dict[str, Any]:
    """Node 2: Evaluates semantic relevance of retrieved chunks."""
    question = state["question"]
    docs = state["documents"]
    
    log_agent("DOCUMENT GRADER", "Grading document relevance...", Colors.CYAN)
    
    if not docs:
        log_agent("DOCUMENT GRADER", "No documents retrieved. Relevance set to 'no'.", Colors.RED)
        return {"documents_relevant": "no"}
        
    context = "\n\n".join([f"--- Chunk {i+1} ---\n{d.page_content}" for i, d in enumerate(docs)])
    
    grader_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert document relevance grader. Your task is to check if a set of retrieved documents "
            "contain information that is relevant to answering the user's question.\n\n"
            "Analyze the question and the documents. Output a strict JSON object with a single key 'binary_score' "
            "which must be either 'yes' (if relevant) or 'no' (if irrelevant).\n\n"
            "Strict JSON Output Format:\n"
            "{{\n"
            '  "binary_score": "yes"\n'
            "}}\n"
            "Do not include any other text, explanation, or conversational filler."
        )),
        ("human", "Question: {question}\n\nRetrieved Documents:\n{context}")
    ])
    
    chain = grader_prompt | llm
    response = chain.invoke({"question": question, "context": context})
    
    score = parse_grader_response(response.content)
    
    if score == "yes":
        log_agent("DOCUMENT GRADER", "RELEVANT - Chunks contain information about the query.", Colors.GREEN)
    else:
        log_agent("DOCUMENT GRADER", "IRRELEVANT - Chunks do NOT contain information about the query.", Colors.RED)
        
    return {"documents_relevant": score}

def query_reformulator_node(state: GraphState) -> Dict[str, Any]:
    """Node 3: Reformulates queries into optimized biological search queries."""
    old_query = state["question"]
    original_query = state["original_question"]
    
    log_agent("QUERY REFORMULATOR", f"Irrelevant contexts found. Initiating query reformulation...", Colors.YELLOW)
    
    reformulate_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert query reformulation assistant. The user asked a question, but our initial search in a "
            "biomedical database yielded irrelevant results.\n\n"
            "Your task is to rewrite the original question into a better, highly specific search query "
            "focused strictly on biological or medical terms to retrieve relevant scientific publications.\n\n"
            "Provide ONLY the rewritten search query. Do not add any conversational text or explanation."
        )),
        ("human", "Original Question: {question}")
    ])
    
    chain = reformulate_prompt | llm
    response = chain.invoke({"question": original_query})
    new_query = response.content.strip().replace('"', '')
    
    log_agent("QUERY REFORMULATOR", f"Old Query: '{old_query}' -> Reformulated Query: '{new_query}'", Colors.YELLOW)
    
    return {
        "question": new_query,
        "retrieval_retries": state["retrieval_retries"] + 1
    }

def generation_node(state: GraphState) -> Dict[str, Any]:
    """Node 4: Generates the answer based on context, augmenting prompts if retrying."""
    question = state["original_question"]
    docs = state["documents"]
    flagged = state["flagged_sentences"]
    hallucination_retries = state["hallucination_retries"]
    retrieval_retries = state["retrieval_retries"]
    
    # Edge case: No relevant documents found after retries
    if not docs or state["documents_relevant"] == "no":
        log_agent("GENERATOR", "No relevant context available. Returning graceful fallback.", Colors.YELLOW)
        fallback = "I'm sorry, but I couldn't find any relevant biomedical information in the local database to answer your question."
        return {"draft_answer": fallback}
        
    context = "\n\n".join([doc.page_content for doc in docs])
    
    if flagged:
        log_agent("GENERATOR", f"Hallucination flagged by Critic! Regenerating answer (Retry {hallucination_retries}/3)...", Colors.YELLOW)
        log_agent("GENERATOR", f"Claims to fix/remove: {flagged}", Colors.RED)
        
        rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an expert medical answering assistant. "
                "We drafted an answer, but our NLI verification system flagged some of the sentences as hallucinations "
                "(claims not grounded in or supported by the context).\n\n"
                "Your task is to rewrite the draft answer. Specifically, you must rewrite or completely remove the flagged sentences "
                "to ensure that every single sentence in your final output is 100% grounded in and supported by the provided context.\n\n"
                "Retrieved Context:\n{context}\n\n"
                "Previous Draft Answer:\n{draft_answer}\n\n"
                "Flagged Sentences (Hallucinations to fix or remove):\n{flagged_sentences}\n\n"
                "Formatting Constraints:\n"
                "- Your revised answer MUST consist of standalone, short sentences.\n"
                "- Each sentence must state exactly one factual claim supported by the context.\n"
                "- Do not use lists or bullet points."
            )),
            ("human", "Question: {question}")
        ])
        chain = rewrite_prompt | llm
        response = chain.invoke({
            "context": context,
            "draft_answer": state["draft_answer"],
            "flagged_sentences": "\n".join([f"- {s}" for s in flagged]),
            "question": question
        })
    else:
        log_agent("GENERATOR", "Drafting first response based on retrieved contexts...", Colors.GREEN)
        
        generation_prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an expert medical answering assistant. "
                "Answer the user's question strictly based on the provided context passages. "
                "Do not introduce any external information or facts.\n\n"
                "Formatting Constraints:\n"
                "- Your answer MUST consist of standalone, short sentences.\n"
                "- Each sentence must state exactly one factual claim.\n"
                "- Do not use lists or bullet points."
            )),
            ("human", "Context:\n{context}\n\nQuestion: {question}")
        ])
        chain = generation_prompt | llm
        response = chain.invoke({"context": context, "question": question})
        
    answer = response.content.strip()
    log_agent("GENERATOR", f"Draft Generated:\n\"{answer}\"", Colors.GREEN)
    
    return {"draft_answer": answer}

def nli_critic_node(state: GraphState) -> Dict[str, Any]:
    """Node 5: Verification Agent (NLI Critic). Performs sentence-level entailment checks."""
    answer = state["draft_answer"]
    docs = state["documents"]
    
    log_agent("NLI CRITIC", "Initiating sentence-level validation...", Colors.CYAN)
    
    # Graceful bypass for fallbacks
    if "couldn't find any relevant biomedical information" in answer.lower():
        log_agent("NLI CRITIC", "Fallback response detected. Skipping validation.", Colors.GREEN)
        return {"flagged_sentences": []}
        
    # Tokenize answer into sentences
    sentences = nltk.tokenize.sent_tokenize(answer)
    flagged = []
    
    for idx, sentence in enumerate(sentences, 1):
        # Calculate maximum entailment score against retrieved passages
        max_entailment = compute_entailment_scores(sentence, docs)
        
        if max_entailment >= 0.80:
            log_agent("NLI CRITIC", f"Sentence {idx} passed NLI check (Max Entailment: {max_entailment:.4f})", Colors.GREEN)
        else:
            log_agent("NLI CRITIC", f"Sentence {idx} FAILED NLI check (Max Entailment: {max_entailment:.4f} < 0.80)!", Colors.RED)
            print(f"   -> Flagged: \"{sentence}\"")
            flagged.append(sentence)
            
    # Update retries if we are about to loop back
    hallucination_retries = state["hallucination_retries"]
    if flagged:
        next_retries = hallucination_retries + 1
    else:
        next_retries = hallucination_retries
        
    return {
        "flagged_sentences": flagged,
        "hallucination_retries": next_retries
    }

# =====================================================================
# 🕸️ LANGGRAPH ROUTING EDGES
# =====================================================================

def decide_after_grading(state: GraphState) -> str:
    """Routes state based on document grader decision and retrieval retries."""
    if state["documents_relevant"] == "yes":
        return "generate"
        
    # If irrelevant, check retries
    if state["retrieval_retries"] < 1:
        return "query_reformulate"
    else:
        log_agent("DOCUMENT GRADER", "Max retrieval retries reached. Bypassing to generate fallback.", Colors.RED)
        return "generate"

def decide_after_nli(state: GraphState) -> str:
    """Routes state based on critic validation and hallucination limits."""
    flagged = state["flagged_sentences"]
    hallucination_retries = state["hallucination_retries"]
    
    if not flagged:
        log_agent("NLI CRITIC", "All sentences verified. Synthesizing final answer.", Colors.GREEN)
        return "end"
        
    # Hallucinations flagged. Can we retry?
    if hallucination_retries < 3:
        log_agent("NLI CRITIC", f"Self-correction loop triggered. Sending back to Generator.", Colors.YELLOW)
        return "generate"
    else:
        log_agent("NLI CRITIC", "Max hallucination retries (3) reached. Exiting with best available draft.", Colors.RED)
        return "end"

# =====================================================================
# 🕸️ LANGGRAPH STATE MACHINE BUILDER
# =====================================================================
workflow = StateGraph(GraphState)

# Add Nodes
workflow.add_node("retrieve", retrieval_node)
workflow.add_node("grade_documents", document_grader_node)
workflow.add_node("query_reformulate", query_reformulator_node)
workflow.add_node("generate", generation_node)
workflow.add_node("nli_critic", nli_critic_node)

# Add Standard Edges
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_edge("query_reformulate", "retrieve")
workflow.add_edge("generate", "nli_critic")

# Add Conditional Edges
workflow.add_conditional_edges(
    "grade_documents",
    decide_after_grading,
    {
        "generate": "generate",
        "query_reformulate": "query_reformulate"
    }
)

workflow.add_conditional_edges(
    "nli_critic",
    decide_after_nli,
    {
        "generate": "generate",
        "end": END
    }
)

# Compile LangGraph App
app = workflow.compile()

# =====================================================================
# 🚀 INTERACTIVE SYSTEM EXECUTION RUNNER
# =====================================================================
def run_pipeline(question: str) -> str:
    """Runs the self-correcting agentic pipeline on the given question."""
    print("\n" + "="*80)
    print(f"[*] INITIATING SELF-CORRECTING AGENTIC RAG FOR: \"{question}\"")
    print("="*80)
    
    initial_state = {
        "question": question,
        "original_question": question,
        "documents": [],
        "draft_answer": "",
        "flagged_sentences": [],
        "documents_relevant": "no",
        "hallucination_retries": 0,
        "retrieval_retries": 0
    }
    
    final_output = app.invoke(initial_state)
    
    print("\n" + "="*80)
    print(f"{Colors.GREEN}{Colors.BOLD}[*] FINAL SYSTEM ANSWER:{Colors.ENDC}")
    print(f"\"{final_output['draft_answer']}\"")
    print("="*80 + "\n")
    
    return final_output['draft_answer']

def main():
    print(f"{Colors.BOLD}{Colors.HEADER}========================================================================")
    print("           B.TECH AI 4TH-SEM PROJECT: SELF-CORRECTING AGENTIC RAG       ")
    print(f"========================================================================{Colors.ENDC}\n")
    
    # 1. Standard In-Domain Query
    run_pipeline("What is the function of the BRCA1 gene?")
    
    # 2. Out-of-Domain / Adversarial Query (Triggers query reformulation and fallback)
    run_pipeline("Who won the FIFA World Cup in 2022?")

if __name__ == "__main__":
    main()
