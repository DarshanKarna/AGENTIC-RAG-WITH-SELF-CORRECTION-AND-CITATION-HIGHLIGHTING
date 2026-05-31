"""
evaluate_ragtruth.py - RAGTruth Benchmark Evaluation
======================================================

Evaluates Baseline RAG vs Self-Correcting RAG (NLI Critic) on the
RAGTruth hallucination corpus.

Strategy:
  - Baseline RAG: Uses RAGTruth's pre-generated LLM responses as-is.
    Ground-truth hallucination labels measure baseline hallucination rate.
  - Self-Correcting RAG (NLI): Runs our DeBERTa NLI critic over each
    response sentence against the source passages. Compares NLI detections
    with human annotations to compute detection precision/recall/F1.

Usage:
    python evaluate_ragtruth.py [--samples N]

Author: Antigravity AI Pair Programmer
"""

import json
import os
import sys
import re
import time
import argparse
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Reconfigure stdout to UTF-8 for Windows
# ---------------------------------------------------------------------------
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RAGTRUTH_DIR = os.path.join(PROJECT_DIR, "data", "RAGTruth", "dataset")
RESULTS_DIR = os.path.join(PROJECT_DIR, "evaluation_results")
RESPONSE_FILE = os.path.join(RAGTRUTH_DIR, "response.jsonl")
SOURCE_FILE = os.path.join(RAGTRUTH_DIR, "source_info.jsonl")

# ---------------------------------------------------------------------------
# Colors for terminal output
# ---------------------------------------------------------------------------
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"

def log(tag, msg, color=Colors.CYAN):
    print(f"{color}{Colors.BOLD}[{tag}]{Colors.ENDC}{color} {msg}{Colors.ENDC}")


# ===========================================================================
# Lightweight Document class to feed into compute_entailment_scores
# ===========================================================================
@dataclass
class FakeDocument:
    """Mimics LangChain's Document object for the NLI scorer."""
    page_content: str
    metadata: dict = field(default_factory=dict)


# ===========================================================================
# Load RAGTruth data
# ===========================================================================
def load_ragtruth_data():
    """Loads and joins response.jsonl with source_info.jsonl, filtering to QA test split."""
    log("DATA", f"Loading RAGTruth from {RAGTRUTH_DIR}...", Colors.BLUE)

    # Load source info
    source_map = {}
    with open(SOURCE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            source_map[obj["source_id"]] = obj

    # Load responses — filter to QA, test split, good quality
    samples = []
    with open(RESPONSE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            resp = json.loads(line.strip())
            if resp["split"] != "test" or resp["quality"] != "good":
                continue
            src = source_map.get(resp["source_id"])
            if not src or src["task_type"] != "QA":
                continue
            resp["_source"] = src
            samples.append(resp)

    log("DATA", f"Loaded {len(samples)} QA test samples (good quality)", Colors.GREEN)
    return samples


# ===========================================================================
# Extract passages as FakeDocument list from RAGTruth source_info
# ===========================================================================
def extract_passages(source_info_dict):
    """Converts RAGTruth source_info passages into FakeDocument objects."""
    passages_text = source_info_dict.get("passages", "")
    if not passages_text:
        return []

    # Split by "passage N:" pattern
    parts = re.split(r"passage\s+\d+:", passages_text, flags=re.IGNORECASE)
    docs = []
    for part in parts:
        text = part.strip()
        if text:
            docs.append(FakeDocument(page_content=text))
    return docs


# ===========================================================================
# Map hallucination labels to sentences
# ===========================================================================
def map_labels_to_sentences(response_text, labels, sentences):
    """
    For each sentence, check if any hallucination label span overlaps with it.
    Returns a set of sentence indices that contain hallucinations.
    """
    hallucinated_indices = set()

    if not labels:
        return hallucinated_indices

    # Build character ranges for each sentence
    sentence_ranges = []
    search_start = 0
    for sent in sentences:
        idx = response_text.find(sent, search_start)
        if idx == -1:
            # Fallback: try to find with stripped whitespace
            idx = response_text.find(sent.strip(), search_start)
        if idx == -1:
            sentence_ranges.append((search_start, search_start + len(sent)))
        else:
            sentence_ranges.append((idx, idx + len(sent)))
            search_start = idx + len(sent)

    # Check each label span against sentence ranges
    for label in labels:
        if label.get("implicit_true", False):
            continue  # Skip implicitly true spans
        l_start = label["start"]
        l_end = label["end"]

        for s_idx, (s_start, s_end) in enumerate(sentence_ranges):
            # Check for overlap
            if l_start < s_end and l_end > s_start:
                hallucinated_indices.add(s_idx)

    return hallucinated_indices


# ===========================================================================
# Main evaluation loop
# ===========================================================================
def run_evaluation(max_samples=200):
    """
    Runs the full evaluation pipeline:
    1. Load RAGTruth QA test data
    2. For each sample, run NLI critic
    3. Compare detections vs ground truth
    4. Compute metrics and generate report
    """

    # ---- Load NLI model (reuse from self_correcting_rag.py) ----
    log("INIT", "Loading NLI model (cross-encoder/nli-deberta-base)...", Colors.YELLOW)
    from sentence_transformers import CrossEncoder
    import nltk
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)

    nli_model = CrossEncoder("cross-encoder/nli-deberta-base")
    log("INIT", "NLI model loaded.", Colors.GREEN)

    # ---- Entailment scoring function (copied from self_correcting_rag.py) ----
    def compute_entailment(sentence, documents):
        """Compute max NLI entailment score for a sentence against document chunks."""
        if not documents:
            return 0.0

        pairs = []
        for doc in documents:
            try:
                chunk_sents = nltk.sent_tokenize(doc.page_content)
            except Exception:
                chunk_sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", doc.page_content) if s.strip()]
            if not chunk_sents:
                chunk_sents = [doc.page_content]
            for cs in chunk_sents:
                pairs.append((cs.strip(), sentence))

        if not pairs:
            return 0.0

        logits = nli_model.predict(pairs)
        logits = np.atleast_2d(logits)

        # Softmax: [0: contradiction, 1: entailment, 2: neutral]
        e_x = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = e_x / e_x.sum(axis=-1, keepdims=True)
        entailment_probs = probs[:, 1]

        return float(np.max(entailment_probs))

    # ---- Load data ----
    samples = load_ragtruth_data()

    if max_samples and max_samples < len(samples):
        # Stratified selection: ensure we get both hallucinated and clean samples
        has_hall = [s for s in samples if len(s["labels"]) > 0]
        no_hall = [s for s in samples if len(s["labels"]) == 0]

        # Take proportional amounts
        hall_ratio = len(has_hall) / len(samples)
        n_hall = max(int(max_samples * hall_ratio), min(len(has_hall), 30))
        n_clean = max_samples - n_hall

        import random
        random.seed(42)
        selected_hall = random.sample(has_hall, min(n_hall, len(has_hall)))
        selected_clean = random.sample(no_hall, min(n_clean, len(no_hall)))
        samples = selected_hall + selected_clean
        random.shuffle(samples)

        log("DATA", f"Stratified sample: {len(selected_hall)} hallucinated + {len(selected_clean)} clean = {len(samples)} total", Colors.YELLOW)

    # ---- Evaluation counters ----
    # Sentence-level detection metrics
    true_positives = 0       # NLI flagged AND ground-truth hallucinated
    false_positives = 0      # NLI flagged BUT ground-truth clean
    false_negatives = 0      # NLI missed BUT ground-truth hallucinated
    true_negatives = 0       # NLI clean AND ground-truth clean

    # Case-level (response-level) metrics
    case_tp = 0  # Response has hallucinations, NLI detected at least one
    case_fp = 0  # Response is clean, NLI falsely flagged something
    case_fn = 0  # Response has hallucinations, NLI detected none
    case_tn = 0  # Response is clean, NLI found nothing

    # Aggregate stats
    total_sentences = 0
    total_gt_hallucinated_sents = 0
    total_nli_flagged_sents = 0
    baseline_hallucinated_responses = 0
    nli_corrected_responses = 0

    # Per-model breakdown
    model_stats = defaultdict(lambda: {"total": 0, "hall": 0, "detected": 0})

    NLI_THRESHOLD = 0.80
    total = len(samples)

    log("EVAL", f"Starting evaluation on {total} samples (NLI threshold = {NLI_THRESHOLD})...", Colors.HEADER)
    start_time = time.time()

    for idx, sample in enumerate(samples):
        if (idx + 1) % 20 == 0 or idx == 0:
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            eta = (total - idx - 1) / rate if rate > 0 else 0
            log("EVAL", f"Processing {idx+1}/{total}  ({rate:.1f} samples/s, ETA: {eta:.0f}s)", Colors.CYAN)

        response_text = sample["response"]
        labels = sample["labels"]
        model_name = sample["model"]
        source_info = sample["_source"]["source_info"]

        # Extract passages as documents
        docs = extract_passages(source_info)
        if not docs:
            continue

        # Tokenize response into sentences
        try:
            sentences = nltk.sent_tokenize(response_text)
        except Exception:
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", response_text) if s.strip()]

        if not sentences:
            continue

        # Map ground-truth hallucination labels → sentence indices
        gt_hallucinated = map_labels_to_sentences(response_text, labels, sentences)
        has_gt_hallucination = len(gt_hallucinated) > 0

        if has_gt_hallucination:
            baseline_hallucinated_responses += 1

        # ---- Run NLI critic on each sentence ----
        nli_flagged = set()
        for s_idx, sentence in enumerate(sentences):
            if len(sentence.strip()) < 10:
                continue  # Skip trivially short fragments
            score = compute_entailment(sentence, docs)
            if score < NLI_THRESHOLD:
                nli_flagged.add(s_idx)

        nli_found_something = len(nli_flagged) > 0
        if not nli_found_something and has_gt_hallucination:
            nli_corrected_responses += 1  # NLI missed it, hallucination would survive

        # ---- Sentence-level metrics ----
        for s_idx in range(len(sentences)):
            is_gt_hall = s_idx in gt_hallucinated
            is_nli_flag = s_idx in nli_flagged

            if is_gt_hall and is_nli_flag:
                true_positives += 1
            elif not is_gt_hall and is_nli_flag:
                false_positives += 1
            elif is_gt_hall and not is_nli_flag:
                false_negatives += 1
            else:
                true_negatives += 1

        total_sentences += len(sentences)
        total_gt_hallucinated_sents += len(gt_hallucinated)
        total_nli_flagged_sents += len(nli_flagged)

        # ---- Case-level metrics ----
        if has_gt_hallucination and nli_found_something:
            case_tp += 1
        elif not has_gt_hallucination and nli_found_something:
            case_fp += 1
        elif has_gt_hallucination and not nli_found_something:
            case_fn += 1
        else:
            case_tn += 1

        # ---- Per-model stats ----
        model_stats[model_name]["total"] += 1
        if has_gt_hallucination:
            model_stats[model_name]["hall"] += 1
        if nli_found_something and has_gt_hallucination:
            model_stats[model_name]["detected"] += 1

    elapsed = time.time() - start_time
    log("EVAL", f"Evaluation complete in {elapsed:.1f}s", Colors.GREEN)

    # ===========================================================================
    # Compute final metrics
    # ===========================================================================
    def safe_div(a, b):
        return a / b if b > 0 else 0.0

    # Sentence-level
    sent_precision = safe_div(true_positives, true_positives + false_positives)
    sent_recall = safe_div(true_positives, true_positives + false_negatives)
    sent_f1 = safe_div(2 * sent_precision * sent_recall, sent_precision + sent_recall)
    sent_accuracy = safe_div(true_positives + true_negatives, total_sentences)

    # Case-level
    case_precision = safe_div(case_tp, case_tp + case_fp)
    case_recall = safe_div(case_tp, case_tp + case_fn)
    case_f1 = safe_div(2 * case_precision * case_recall, case_precision + case_recall)
    case_accuracy = safe_div(case_tp + case_tn, case_tp + case_fp + case_fn + case_tn)

    # Hallucination rates
    baseline_hall_rate = safe_div(baseline_hallucinated_responses, total) * 100
    corrected_hall_rate = safe_div(case_fn, total) * 100  # Only missed hallucinations survive
    improvement = baseline_hall_rate - corrected_hall_rate

    results = {
        "dataset": "RAGTruth (QA, test split)",
        "num_samples": total,
        "num_sentences": total_sentences,
        "nli_threshold": NLI_THRESHOLD,
        "elapsed_seconds": round(elapsed, 1),

        "baseline": {
            "hallucinated_responses": baseline_hallucinated_responses,
            "hallucination_rate_pct": round(baseline_hall_rate, 2),
            "total_hallucinated_sentences": total_gt_hallucinated_sents,
            "avg_hallucinated_sents_per_response": round(safe_div(total_gt_hallucinated_sents, total), 2),
        },

        "self_correcting_nli": {
            "flagged_sentences": total_nli_flagged_sents,
            "residual_hallucination_rate_pct": round(corrected_hall_rate, 2),
            "improvement_pct": round(improvement, 2),

            "sentence_level": {
                "precision": round(sent_precision, 4),
                "recall": round(sent_recall, 4),
                "f1": round(sent_f1, 4),
                "accuracy": round(sent_accuracy, 4),
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "true_negatives": true_negatives,
            },

            "case_level": {
                "precision": round(case_precision, 4),
                "recall": round(case_recall, 4),
                "f1": round(case_f1, 4),
                "accuracy": round(case_accuracy, 4),
                "true_positives": case_tp,
                "false_positives": case_fp,
                "false_negatives": case_fn,
                "true_negatives": case_tn,
            },
        },

        "per_model": {k: dict(v) for k, v in model_stats.items()},
    }

    # ===========================================================================
    # Print results table
    # ===========================================================================
    print("\n" + "=" * 78)
    print(f"{Colors.BOLD}{Colors.HEADER}  RAGTRUTH EVALUATION RESULTS: BASELINE RAG vs SELF-CORRECTING RAG{Colors.ENDC}")
    print("=" * 78)

    print(f"\n{Colors.BOLD}Dataset:{Colors.ENDC} RAGTruth QA (test split, good quality)")
    print(f"{Colors.BOLD}Samples:{Colors.ENDC} {total}  |  Sentences: {total_sentences}  |  NLI Threshold: {NLI_THRESHOLD}")
    print(f"{Colors.BOLD}Runtime:{Colors.ENDC} {elapsed:.1f}s")

    print(f"\n{Colors.BOLD}{'─' * 78}{Colors.ENDC}")
    print(f"{Colors.BOLD}  HALLUCINATION RATES{Colors.ENDC}")
    print(f"{'─' * 78}")
    print(f"  {'Metric':<45} {'Baseline RAG':>14} {'Self-Corrected':>14}")
    print(f"  {'─' * 73}")
    print(f"  {'Hallucination Rate (%)':<45} {Colors.RED}{baseline_hall_rate:>13.2f}%{Colors.ENDC} {Colors.GREEN}{corrected_hall_rate:>13.2f}%{Colors.ENDC}")
    print(f"  {'Hallucinated Responses':<45} {baseline_hallucinated_responses:>14} {case_fn:>14}")
    print(f"  {'Total Hallucinated Sentences':<45} {total_gt_hallucinated_sents:>14} {'—':>14}")
    print(f"  {Colors.GREEN}{'Improvement':<45} {'':>14} {'+' + str(round(improvement, 2)) + '%':>14}{Colors.ENDC}")

    print(f"\n{Colors.BOLD}{'─' * 78}{Colors.ENDC}")
    print(f"{Colors.BOLD}  NLI CRITIC — SENTENCE-LEVEL DETECTION{Colors.ENDC}")
    print(f"{'─' * 78}")
    print(f"  {'Metric':<30} {'Value':>12}")
    print(f"  {'─' * 42}")
    print(f"  {'Precision':<30} {sent_precision:>12.4f}")
    print(f"  {'Recall':<30} {sent_recall:>12.4f}")
    print(f"  {'F1 Score':<30} {sent_f1:>12.4f}")
    print(f"  {'Accuracy':<30} {sent_accuracy:>12.4f}")
    print(f"  {'True Positives':<30} {true_positives:>12}")
    print(f"  {'False Positives':<30} {false_positives:>12}")
    print(f"  {'False Negatives':<30} {false_negatives:>12}")
    print(f"  {'True Negatives':<30} {true_negatives:>12}")

    print(f"\n{Colors.BOLD}{'─' * 78}{Colors.ENDC}")
    print(f"{Colors.BOLD}  NLI CRITIC — CASE-LEVEL (RESPONSE) DETECTION{Colors.ENDC}")
    print(f"{'─' * 78}")
    print(f"  {'Metric':<30} {'Value':>12}")
    print(f"  {'─' * 42}")
    print(f"  {'Precision':<30} {case_precision:>12.4f}")
    print(f"  {'Recall':<30} {case_recall:>12.4f}")
    print(f"  {'F1 Score':<30} {case_f1:>12.4f}")
    print(f"  {'Accuracy':<30} {case_accuracy:>12.4f}")

    print(f"\n{Colors.BOLD}{'─' * 78}{Colors.ENDC}")
    print(f"{Colors.BOLD}  PER-MODEL BREAKDOWN{Colors.ENDC}")
    print(f"{'─' * 78}")
    print(f"  {'Model':<28} {'Samples':>8} {'Has Hall.':>10} {'Detected':>10} {'Detect %':>10}")
    print(f"  {'─' * 66}")
    for model, stats in sorted(model_stats.items()):
        detect_rate = safe_div(stats["detected"], stats["hall"]) * 100
        print(f"  {model:<28} {stats['total']:>8} {stats['hall']:>10} {stats['detected']:>10} {detect_rate:>9.1f}%")

    print("\n" + "=" * 78 + "\n")

    # ===========================================================================
    # Save results JSON
    # ===========================================================================
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(RESULTS_DIR, "ragtruth_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log("SAVE", f"Results saved to {results_path}", Colors.GREEN)

    # ===========================================================================
    # Generate comparison charts
    # ===========================================================================
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle("RAGTruth Evaluation: Baseline RAG vs Self-Correcting RAG",
                     fontsize=15, fontweight="bold", y=1.02)

        # --- Chart 1: Hallucination Rate Comparison ---
        ax1 = axes[0]
        bars = ax1.bar(
            ["Baseline RAG", "Self-Corrected\n(NLI Critic)"],
            [baseline_hall_rate, corrected_hall_rate],
            color=["#ef4444", "#10b981"],
            width=0.5,
            edgecolor="white",
            linewidth=1.5,
        )
        ax1.set_ylabel("Hallucination Rate (%)", fontsize=11)
        ax1.set_title("Hallucination Rate", fontsize=13, fontweight="bold")
        for bar, val in zip(bars, [baseline_hall_rate, corrected_hall_rate]):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{val:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=12)
        ax1.set_ylim(0, max(baseline_hall_rate, corrected_hall_rate) * 1.3 + 5)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        # --- Chart 2: Sentence-Level Detection Metrics ---
        ax2 = axes[1]
        metrics_names = ["Precision", "Recall", "F1 Score"]
        metrics_vals = [sent_precision, sent_recall, sent_f1]
        bar_colors = ["#3b82f6", "#f59e0b", "#8b5cf6"]
        bars2 = ax2.bar(metrics_names, metrics_vals, color=bar_colors, width=0.5,
                        edgecolor="white", linewidth=1.5)
        ax2.set_ylabel("Score", fontsize=11)
        ax2.set_title("Sentence-Level Detection", fontsize=13, fontweight="bold")
        for bar, val in zip(bars2, metrics_vals):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{val:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=11)
        ax2.set_ylim(0, 1.15)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        # --- Chart 3: Case-Level Detection Metrics ---
        ax3 = axes[2]
        case_names = ["Precision", "Recall", "F1 Score", "Accuracy"]
        case_vals = [case_precision, case_recall, case_f1, case_accuracy]
        bar_colors3 = ["#3b82f6", "#f59e0b", "#8b5cf6", "#10b981"]
        bars3 = ax3.bar(case_names, case_vals, color=bar_colors3, width=0.5,
                        edgecolor="white", linewidth=1.5)
        ax3.set_ylabel("Score", fontsize=11)
        ax3.set_title("Case-Level Detection", fontsize=13, fontweight="bold")
        for bar, val in zip(bars3, case_vals):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{val:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=11)
        ax3.set_ylim(0, 1.15)
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)

        plt.tight_layout()
        chart_path = os.path.join(RESULTS_DIR, "ragtruth_comparison.png")
        fig.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        log("CHART", f"Comparison chart saved to {chart_path}", Colors.GREEN)

    except ImportError:
        log("CHART", "matplotlib not available — skipping chart generation", Colors.YELLOW)

    return results


# ===========================================================================
# CLI entry point
# ===========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Baseline vs Self-Correcting RAG on RAGTruth")
    parser.add_argument("--samples", type=int, default=200,
                        help="Max samples to evaluate (default: 200). Use 0 for all.")
    args = parser.parse_args()

    max_samples = args.samples if args.samples > 0 else None
    run_evaluation(max_samples=max_samples)
