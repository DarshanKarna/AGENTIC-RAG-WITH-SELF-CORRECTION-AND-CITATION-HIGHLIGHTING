import os
import sys
import json
import time
import argparse
import random
import re
import requests
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
import gc
import logging
from ingest import process_single_concept, chunk_text

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3:latest"
NLI_MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
TRAIN_OUTPUT = os.path.join("data", "train.jsonl")
OKF_DIR = Path("okf_bundle")
NLI_THRESHOLD = 0.85
TARGET_SAMPLES = 16000

# ---------------------------------------------------------------------------
# Sampling Logic
# ---------------------------------------------------------------------------
def perform_stratified_sampling(all_chunks, target_samples, floor=200):
    filtered_chunks = []
    for c in all_chunks:
        dt = c['metadata'].get('document_type', 'unknown')
        # Skip statute/act chunks since external dataset covers them
        if dt in ['statute', 'act']:
            continue
        filtered_chunks.append(c)

    groups = defaultdict(list)
    for c in filtered_chunks:
        dt = c['metadata'].get('document_type', 'unknown')
        lang = c['metadata'].get('language', 'unknown')
        groups[(dt, lang)].append(c)

    sampled = []
    remaining_budget = target_samples
    
    # 1. Floor allocation
    for key, chunks in groups.items():
        allocation = min(floor, len(chunks))
        if allocation > 0:
            sampled.extend(random.sample(chunks, allocation))
            remaining_budget -= allocation
            
    if remaining_budget <= 0:
        return sampled[:target_samples]
        
    # 2. Proportional allocation for the rest
    remaining_chunks = {k: [c for c in v if c not in sampled] for k, v in groups.items()}
    total_remaining = sum(len(v) for v in remaining_chunks.values())
    
    if total_remaining == 0:
        return sampled
        
    for key, chunks in remaining_chunks.items():
        if not chunks: continue
        proportion = len(chunks) / total_remaining
        allocation = min(len(chunks), int(remaining_budget * proportion))
        if allocation > 0:
            sampled.extend(random.sample(chunks, allocation))
            
    random.shuffle(sampled)
    return sampled

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
def get_prompt(text_chunk, doc_type, language):
    if language == 'ne':
        system_prompt = "You are a legal data extractor. Generate a question and answer based STRICTLY on the provided text chunk. Do not introduce outside knowledge. If the text chunk lacks meaningful information (e.g., a table of contents), output nothing."
        
        if doc_type in ['statute', 'act']:
            user_prompt = f"यस खण्डको आधारमा एउटा प्रश्न र उत्तर तयार गर्नुहोस्। प्रश्न कानुनको दफा वा सजायसँग सम्बन्धित हुनुपर्छ।\n\nChunk:\n{text_chunk}"
        elif doc_type in ['case_law', 'verdict']:
            user_prompt = f"यस अदालतको फैसलाको आधारमा एउटा प्रश्न र उत्तर तयार गर्नुहोस्।\n\nChunk:\n{text_chunk}"
        elif doc_type in ['circular', 'directive']:
            user_prompt = f"यस परिपत्रको निर्देशनमा आधारित एउटा प्रश्न र उत्तर तयार गर्नुहोस्।\n\nChunk:\n{text_chunk}"
        else:
            user_prompt = f"यस पाठको आधारमा एउटा प्रश्न र उत्तर तयार गर्नुहोस्।\n\nChunk:\n{text_chunk}"
            
    else:
        system_prompt = "You are a legal data extractor. Generate a question and answer based STRICTLY on the provided text chunk. Do not introduce outside knowledge. If the text chunk lacks meaningful information (e.g., a table of contents), output nothing."
        
        if doc_type in ['statute', 'act']:
            user_prompt = f"Generate a question and answer based on this legal text chunk. Focus on sections, rules, or penalties.\n\nChunk:\n{text_chunk}"
        elif doc_type in ['case_law', 'verdict']:
            user_prompt = f"Generate a question and answer based on this court verdict chunk. Focus on the ruling or precedent.\n\nChunk:\n{text_chunk}"
        elif doc_type in ['circular', 'directive']:
            user_prompt = f"Generate a question and answer based on this regulatory circular.\n\nChunk:\n{text_chunk}"
        else:
            user_prompt = f"Generate a question and answer based on this text.\n\nChunk:\n{text_chunk}"
            
    return system_prompt, user_prompt

def get_sentences(text):
    sentences = re.split(r'(?<=[.!?।])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

def generate_qa_pair(chunk_text, doc_type, language):
    sys_prompt, user_prompt = get_prompt(chunk_text, doc_type, language)
    
    prompt = f"{sys_prompt}\n\n{user_prompt}\n\nPlease format your response strictly as JSON:\n{{\"question\": \"...\", \"answer\": \"...\"}}"
    
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1}
        }, timeout=120)
        response.raise_for_status()
        result = response.json().get("response", "")
        if not result.strip(): return None, None
        data = json.loads(result)
        return data.get("question"), data.get("answer")
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return None, None

def unload_ollama():
    try:
        requests.post(OLLAMA_URL, json={'model': MODEL_NAME, 'keep_alive': 0})
        time.sleep(2) # Give it a moment to free VRAM
    except:
        pass

# ---------------------------------------------------------------------------
# Main Script
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=int, help="Run a small pilot with N samples")
    parser.add_argument("--resume", action="store_true", help="Resume from existing train.jsonl")
    parser.add_argument("--target-n", type=int, default=TARGET_SAMPLES, help="Target number of chunks to sample")
    args = parser.parse_args()

    processed_chunks = set()
    seen_qa = set()
    if args.resume and os.path.exists(TRAIN_OUTPUT):
        with open(TRAIN_OUTPUT, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_chunks.add(f"{data['source_act']}_{data['chunk_index']}")
                    seen_qa.add((data.get('instruction', ''), data.get('output', '')))
                except:
                    pass
        logger.info(f"Resuming: found {len(processed_chunks)} already processed chunks and {len(seen_qa)} existing QA pairs.")

    logger.info("Parsing OKF Bundle...")
    all_chunks = []
    md_files = list(OKF_DIR.rglob("*.md"))
    
    for fpath in tqdm(md_files, desc="Chunking files"):
        passages = process_single_concept(fpath)
        chunks = chunk_text(passages)
        for c in chunks:
            c['chunk_id'] = f"{c['metadata'].get('document_id', 'unknown')}_{c['metadata'].get('chunk_index', 0)}"
            
            
            all_chunks.append(c)
        
    logger.info(f"Total chunks extracted (after filtering excluded dirs): {len(all_chunks)}")
    
    if args.resume:
        all_chunks = [c for c in all_chunks if c['chunk_id'] not in processed_chunks]
        logger.info(f"Chunks remaining to process: {len(all_chunks)}")

    sampled_chunks = perform_stratified_sampling(all_chunks, args.target_n)
    if args.pilot:
        sampled_chunks = random.sample(sampled_chunks, min(args.pilot, len(sampled_chunks)))
        logger.info(f"PILOT RUN: Selected {len(sampled_chunks)} chunks.")
    else:
        logger.info(f"Selected {len(sampled_chunks)} chunks for full run.")

    os.makedirs(os.path.dirname(TRAIN_OUTPUT), exist_ok=True)
    
    batch_size = 50
    start_time = time.time()
    
    logger.info("Loading mDeBERTa for NLI scoring...")
    from sentence_transformers import CrossEncoder
    nli_model = CrossEncoder(NLI_MODEL_NAME, device="cpu")
    
    stats = {
        'total_attempted': 0,
        'low_content_skipped': 0,
        'nli_rejected': 0,
        'nli_rejected_by_type': defaultdict(int),
        'nli_rejected_by_lang': defaultdict(int),
        'successful': 0
    }
    
    mode = "a" if args.resume else "w"
    tracking_f = open("generation_report.csv", mode, encoding="utf-8")
    if not args.resume:
        tracking_f.write("batch,chunk_id,sentences_produced,flagged,reason\n")

    for i in tqdm(range(0, len(sampled_chunks), batch_size), desc="Processing Batches"):
        batch = sampled_chunks[i:i+batch_size]
        batch_idx = (i // batch_size) + 1
        
        logger.info(f"Generating QA pairs for batch {batch_idx}...")
        batch_results = []
        for chunk in tqdm(batch, desc="Qwen3 Generation", leave=False):
            dt = chunk['metadata'].get('document_type', 'unknown')
            lang = chunk['metadata'].get('language', 'unknown')
            chunk_id = chunk.get('chunk_id', 'unknown')
            
            q, a = generate_qa_pair(chunk['text'], dt, lang)
            if not q or not a:
                stats['low_content_skipped'] += 1
                tracking_f.write(f"{batch_idx},{chunk_id},0,True,Generation Failed/Empty\n")
                continue
            
            # 15% chance to generate a refusal example
            if random.random() < 0.15:
                wrong_chunk = random.choice(all_chunks)
                # Ensure the wrong chunk is from a completely different document
                while wrong_chunk['metadata'].get('document_id') == chunk['metadata'].get('document_id'):
                    wrong_chunk = random.choice(all_chunks)
                
                # Overwrite answer with a polite refusal
                if lang == 'ne':
                    a = "माफ गर्नुहोला, प्रदान गरिएको सन्दर्भमा यस प्रश्नको उत्तर समावेश छैन।"
                else:
                    a = "The provided context does not contain information to answer this question. Please refer to the appropriate document."
                
                batch_results.append({
                    "chunk": wrong_chunk,
                    "question": q,
                    "answer": a,
                    "example_type": "refusal"
                })
            else:
                batch_results.append({
                    "chunk": chunk,
                    "question": q,
                    "answer": a,
                    "example_type": "rag_grounded"
                })
                
            stats['total_attempted'] += 1

        if not batch_results:
            continue
            
        with open(TRAIN_OUTPUT, "a", encoding="utf-8") as out_f:
            for idx, r in enumerate(batch_results):
                dt = r['chunk']['metadata'].get('document_type', 'unknown')
                lang = r['chunk']['metadata'].get('language', 'unknown')
                chunk_id = r['chunk'].get('chunk_id', 'unknown')
                
                # Deduplication check
                qa_tuple = (r.get('question', ''), r.get('answer', ''))
                if qa_tuple in seen_qa:
                    logger.info(f"[REJECTED] Duplicate QA pair generated.")
                    stats['nli_rejected'] += 1 # Or track in a new counter
                    tracking_f.write(f"{batch_idx},{chunk_id},0,True,Duplicate QA Pair\n")
                    continue
                
                sentences = get_sentences(r['answer'])
                num_sentences = len(sentences)
                
                if r.get('example_type') == 'refusal':
                    # Lightweight quality check: Reject if refusal hallucinated concrete citations/claims
                    ans = r['answer'].lower()
                    if re.search(r'\d+|section|article|दफा|धारा|नियम', ans):
                        logger.info(f"[REJECTED REFUSAL] Hallucination detected: {ans}")
                        stats['nli_rejected'] += 1
                        stats['nli_rejected_by_type'][dt] += 1
                        stats['nli_rejected_by_lang'][lang] += 1
                        tracking_f.write(f"{batch_idx},{chunk_id},{num_sentences},True,Refusal Hallucination\n")
                        continue
                    prob = 1.0 # Force pass for clean refusals
                else:
                    if not sentences:
                        prob = 0.0
                    else:
                        pairs_for_model = [(r['chunk']['text'], s) for s in sentences]
                        logits = nli_model.predict(pairs_for_model)
                        logits = np.atleast_2d(logits)
                        
                        e_x = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
                        probs = e_x / e_x.sum(axis=-1, keepdims=True)
                        entailment_probs = probs[:, 0]
                        # Score is the minimum entailment across all generated sentences
                        prob = float(np.min(entailment_probs))
                    
                if prob < NLI_THRESHOLD:
                    stats['nli_rejected'] += 1
                    stats['nli_rejected_by_type'][dt] += 1
                    stats['nli_rejected_by_lang'][lang] += 1
                    tracking_f.write(f"{batch_idx},{chunk_id},{num_sentences},True,NLI Score {prob:.3f}\n")
                    
                    if args.pilot and idx < 3: 
                        logger.info(f"[REJECTED] Score: {prob:.3f} | Q: {r['question']} | A: {r['answer']}")
                else:
                    stats['successful'] += 1
                    tracking_f.write(f"{batch_idx},{chunk_id},{num_sentences},False,Accepted\n")
                    record = {
                        "instruction": r["question"],
                        "input": r["chunk"]["text"],
                        "output": r["answer"],
                        "language": lang,
                        "source_act": r["chunk"]["metadata"].get("document_id"),
                        "section": r["chunk"]["metadata"].get("page_number", ""),
                        "doc_type": dt,
                        "chunk_index": r["chunk"]["metadata"].get("chunk_index"),
                        "example_type": r.get("example_type", "rag_grounded"),
                        "nli_score": prob if r.get('example_type') != 'refusal' else None
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    seen_qa.add(qa_tuple)
                    
                    if args.pilot and idx < 3:
                        logger.info(f"[ACCEPTED] Score: {prob:.3f} | Q: {r['question']} | A: {r['answer']}")
                        
    tracking_f.close()

    elapsed = time.time() - start_time
    logger.info("="*50)
    logger.info("RUN COMPLETED")
    logger.info(f"Total Time: {elapsed/60:.2f} mins")
    logger.info(f"Total Attempted: {stats['total_attempted']}")
    logger.info(f"Low-Content Skipped: {stats['low_content_skipped']}")
    logger.info(f"NLI Rejected: {stats['nli_rejected']} ({(stats['nli_rejected']/stats['total_attempted']*100 if stats['total_attempted'] else 0):.1f}%)")
    logger.info(f"Successfully Generated: {stats['successful']}")
    logger.info("NLI Rejections by Type:")
    for k, v in stats['nli_rejected_by_type'].items(): logger.info(f"  {k}: {v}")
    logger.info("NLI Rejections by Lang:")
    for k, v in stats['nli_rejected_by_lang'].items(): logger.info(f"  {k}: {v}")
    logger.info("="*50)

if __name__ == "__main__":
    main()
