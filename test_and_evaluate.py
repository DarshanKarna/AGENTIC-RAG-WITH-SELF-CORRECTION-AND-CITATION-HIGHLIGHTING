"""
test_and_evaluate.py - Baseline RAG Evaluation Suite
=====================================================

Performs structured evaluation on the current baseline RAG system.
Tests 5 distinct queries (in-domain and out-of-domain) and evaluates them using
LLM-as-a-judge (Groq llama3-8b-8192) on three metrics:
  1. Context Relevance
  2. Groundedness (Faithfulness)
  3. Answer Relevance

Usage:
    python test_and_evaluate.py
"""

import os
import json
import logging
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Setup Logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Config
CHROMA_DB_DIR = os.path.join("data", "chroma_db_hf")
COLLECTION_NAME = "bioasq_chunks"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.3-70b-versatile"

TEST_QUERIES = [
    # In-Domain (Biomedical)
    "What is the function of the BRCA1 gene?",
    "What is the role of p53 protein in cell cycle control?",
    "Which gene mutations lead to cystic fibrosis?",
    
    # Out-of-Domain / Adversarial (Should trigger failures in Naive RAG)
    "Explain the mechanism of quantum entanglement in quantum computing.",
    "Who won the FIFA World Cup in 2022?"
]

SYSTEM_INSTRUCTION = (
    "Answer strictly based on the provided context. "
    "Do not introduce any information not in the context passages. "
    "Output your answer in standalone sentences, each stating one claim."
)

def load_env():
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not found. Please check your .env file.")
    return api_key

def evaluate_generation(llm, question, context, answer):
    """
    Evaluates RAG performance using LLM-as-a-judge.
    Returns scores (1-5) and reasoning for:
      - Context Relevance
      - Groundedness (Faithfulness)
      - Answer Relevance
    """
    eval_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a meticulous RAG system evaluator. Analyze the given question, "
            "retrieved context, and generated answer. Grade them on three metrics on a scale of 1 to 5 "
            "where 1 is poor/completely wrong and 5 is excellent/perfect.\n\n"
            "Metrics definition:\n"
            "1. context_relevance: Does the retrieved context contain information relevant to answering the question?\n"
            "2. groundedness: Is the generated answer fully grounded in and supported *only* by the retrieved context? Deduct points if the answer brings in external facts or hallucinates.\n"
            "3. answer_relevance: Does the generated answer directly address and answer the user's question?\n\n"
            "Output your evaluation strictly in the following JSON format:\n"
            "{{\n"
            '  "context_relevance": {{\n'
            '    "score": 5,\n'
            '    "reason": "..."\n'
            '  }},\n'
            '  "groundedness": {{\n'
            '    "score": 5,\n'
            '    "reason": "..."\n'
            '  }},\n'
            '  "answer_relevance": {{\n'
            '    "score": 5,\n'
            '    "reason": "..."\n'
            '  }}\n'
            "}}\n"
            "Do not add any other conversational text or surrounding formatting."
        )),
        ("human", (
            "Question: {question}\n\n"
            "Retrieved Context:\n{context}\n\n"
            "Generated Answer:\n{answer}"
        ))
    ])

    eval_chain = eval_prompt | llm | JsonOutputParser()
    try:
        results = eval_chain.invoke({
            "question": question,
            "context": context,
            "answer": answer
        })
        return results
    except Exception as e:
        logger.error(f"Error running evaluator LLM: {e}")
        return {
            "context_relevance": {"score": 1, "reason": f"Evaluation error: {str(e)}"},
            "groundedness": {"score": 1, "reason": f"Evaluation error: {str(e)}"},
            "answer_relevance": {"score": 1, "reason": f"Evaluation error: {str(e)}"}
        }

def main():
    print("======================================================================")
    print("                  STARTING BASELINE RAG EVALUATION                    ")
    print("======================================================================\n")

    api_key = load_env()

    # 1. Initialize Retriever
    print("[1/3] Loading Embedding Model & ChromaDB...")
    embedding_fn = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # 2. Initialize LLMs
    print("[2/3] Connecting to Groq LLM...")
    llm = ChatGroq(model=LLM_MODEL, api_key=api_key, temperature=0)

    # Core generation chain
    gen_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_INSTRUCTION),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | gen_prompt
        | llm
        | StrOutputParser()
    )

    # 3. Run Benchmark
    print("[3/3] Running Benchmark Queries and Grading...")
    
    results_summary = []
    
    for idx, query in enumerate(TEST_QUERIES, 1):
        print(f"\n--- Query {idx}: \"{query}\" ---")
        
        # A. Retrieve
        retrieved_docs = retriever.invoke(query)
        context_str = format_docs(retrieved_docs)
        print(f" -> Retrieved {len(retrieved_docs)} chunks from ChromaDB.")
        
        # B. Generate
        answer = rag_chain.invoke(query)
        print(f" -> Answer Generated (length: {len(answer)} chars).")
        
        # C. Evaluate
        grade = evaluate_generation(llm, query, context_str, answer)
        
        results_summary.append({
            "query": query,
            "context": context_str,
            "answer": answer,
            "eval": grade
        })
        
        # Pretty print results
        print(f"\n[Generated Answer]:\n{answer}")
        print("\n[Evaluation Grades]:")
        for metric, details in grade.items():
            print(f"  - {metric.replace('_', ' ').title()}: {details['score']}/5")
            print(f"    Reason: {details['reason']}")
        print("-" * 70)

    # 4. Print Overall Report
    print("\n" + "=" * 70)
    print("                      OVERALL RAG EVALUATION REPORT                    ")
    print("=" * 70)
    
    avg_context_relevance = sum(r['eval']['context_relevance']['score'] for r in results_summary) / len(results_summary)
    avg_groundedness = sum(r['eval']['groundedness']['score'] for r in results_summary) / len(results_summary)
    avg_answer_relevance = sum(r['eval']['answer_relevance']['score'] for r in results_summary) / len(results_summary)
    
    print(f"\n  Average Context Relevance Score   : {avg_context_relevance:.2f} / 5")
    print(f"  Average Groundedness Score        : {avg_groundedness:.2f} / 5")
    print(f"  Average Answer Relevance Score    : {avg_answer_relevance:.2f} / 5")
    
    print("\nKey Vulnerability Analysis:")
    # We analyze where things failed (e.g. low scores for out-of-domain)
    for r in results_summary:
        cr = r['eval']['context_relevance']['score']
        gr = r['eval']['groundedness']['score']
        ar = r['eval']['answer_relevance']['score']
        
        if cr <= 2:
            print(f"\n[!] Failure in Retriever (Query: \"{r['query']}\"):")
            print(f"    Context Relevance was only {cr}/5 because the query is out-of-domain.")
            print(f"    Answer Relevance was graded {ar}/5.")
            print(f"    Resulting Answer: \"{r['answer'].strip()[:100]}...\"")
            
    print("\n" + "=" * 70)
    print("Evaluation completed successfully.")
    print("=" * 70)

if __name__ == "__main__":
    main()
