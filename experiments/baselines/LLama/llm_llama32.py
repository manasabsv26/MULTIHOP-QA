"""
LLaMA Full-Context QA Baseline for MultiHopRAG
-----------------------------------------------
Loads all gold evidence documents for each query and passes them as full
context to a LLaMA model (via DeepInfra or Together.AI).

Run from experiments/baselines/LLama/ directory.

Usage:
  python llm_llama32.py --mode togetherai_free
  python llm_llama32.py --mode deepinfra
  python llm_llama32.py --mode stats_only
"""

import os
import json
import pickle
import argparse
from difflib import SequenceMatcher

import evaluate as hf_evaluate
import nltk
import requests
from openai import OpenAI

nltk.download("wordnet", quiet=True)

# ── Argument parsing ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="LLaMA full-context QA for MultiHopRAG.")
parser.add_argument("--queries", type=str,
                    default="../../multiHopRag-dataset/MultiHopRAG.json",
                    help="Path to MultiHopRAG.json.")
parser.add_argument("--corpus", type=str,
                    default="../../multiHopRag-dataset/corpus.json",
                    help="Path to corpus.json.")
parser.add_argument("--output", type=str, default="results.json",
                    help="Output file for predictions.")
parser.add_argument("--checkpoint", type=str, default="intermediate_results.pkl",
                    help="Pickle file for intermediate checkpointing.")
parser.add_argument("--mode", type=str,
                    choices=["togetherai_free", "deepinfra", "stats_only"],
                    default="togetherai_free",
                    help="Which LLM backend to use.")
parser.add_argument("--start", type=int, default=0, help="Start query index.")
parser.add_argument("--end",   type=int, default=None, help="End query index (exclusive).")
args = parser.parse_args()

# ── API keys (set via environment variables or fill directly) ──────────────────
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY", "")
TOGETHER_API_KEY  = os.getenv("TOGETHER_API_KEY", "")

# ── Load data ──────────────────────────────────────────────────────────────────
with open(args.queries, "r", encoding="utf-8") as f:
    queries = json.load(f)

with open(args.corpus, "r", encoding="utf-8") as f:
    corpus = json.load(f)

print(f"Loaded {len(queries)} queries and {len(corpus)} corpus documents.")

# ── Corpus statistics ──────────────────────────────────────────────────────────
total_words = 0
doc_lengths = []
for doc in corpus:
    word_count = len(doc["body"].split())
    total_words += word_count
    doc_lengths.append((doc["title"], word_count))

doc_lengths.sort(key=lambda x: x[1], reverse=True)
print(f"Total word count: {total_words}")
print(f"Average words per document: {total_words / len(corpus):.2f}")
print("Top 10 documents by word length:")
for i, (title, word_count) in enumerate(doc_lengths[:10], 1):
    print(f"  {i}. {title} — {word_count} words")

# ── Query answer statistics ────────────────────────────────────────────────────
answer_lengths = [len(q["answer"].split()) for q in queries]
print(f"\nMaximum answer length: {max(answer_lengths)} words")
print(f"Average answer length: {sum(answer_lengths) / len(answer_lengths):.2f} words")

# Build lookup maps
title_to_doc = {doc["title"]: doc["body"] for doc in corpus}
query_to_evidence_list = {q["query"]: q["evidence_list"] for q in queries}

if args.mode == "stats_only":
    print("Stats-only mode complete.")
    exit(0)

# ── OpenAI-compatible clients ──────────────────────────────────────────────────
deepinfra_client = OpenAI(
    api_key=DEEPINFRA_API_KEY,
    base_url="https://api.deepinfra.com/v1/openai",
)

stream = False


# ── LLM query helpers ──────────────────────────────────────────────────────────

def query_llama_deepinfra(prompt):
    response = deepinfra_client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-70B-Instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for multi-hop QA. Provide specific answer without details"},
            {"role": "user",   "content": prompt},
        ],
        stream=stream,
    )
    prompt_tokens     = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens
    total_tokens      = response.usage.total_tokens
    print(f"Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}")
    return response.choices[0].message.content


def query_llama_togetherai_freemodel(prompt):
    together_client = OpenAI(
        api_key=TOGETHER_API_KEY,
        base_url="https://api.together.xyz/v1",
    )
    response = together_client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant for multi-hop QA. "
                    "Answer very concisely without providing details. "
                    "Use information from the context to answer the question. "
                    "If the answer requires combining facts from different documents, do so. "
                    "If context is insufficient to answer the question, answer 'Insufficient Information'."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        stream=stream,
    )
    prompt_tokens     = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens
    total_tokens      = response.usage.total_tokens
    print(f"Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}")
    return response.choices[0].message.content


def query_llama_togetherai(prompt):
    together_client = OpenAI(
        api_key=TOGETHER_API_KEY,
        base_url="https://api.together.xyz/v1",
    )
    response = together_client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for multi-hop QA. Provide specific answer without details"},
            {"role": "user",   "content": prompt},
        ],
        stream=stream,
    )
    prompt_tokens     = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens
    total_tokens      = response.usage.total_tokens
    print(f"Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}")
    return response.choices[0].message.content


def build_prompt_full_corpus(query_obj):
    """Build a prompt with all gold-evidence document bodies as context."""
    evidence_list   = query_obj.get("evidence_list", [])
    evidence_titles = [ev["title"] for ev in evidence_list]
    context_docs = [
        f"[Doc {i+1}] {title_to_doc[title]}"
        for i, title in enumerate(evidence_titles)
        if title in title_to_doc
    ]
    context = "\n\n".join(context_docs)
    return f"Question: {query_obj['query']}\n\nContext:\n{context}\n\nAnswer:"


# ── Evaluation metric helpers ──────────────────────────────────────────────────
rouge  = hf_evaluate.load("rouge")
meteor = hf_evaluate.load("meteor")


def compute_mrr(preds, refs):
    ranks = [1 if pred.strip().lower() == ref.strip().lower() else 0
             for pred, ref in zip(preds, refs)]
    return sum(1 / (i + 1) for i, r in enumerate(ranks) if r) / max(1, sum(ranks))


def fuzzy_contains(text, substring, threshold=0.8):
    return SequenceMatcher(None, text.lower(), substring.lower()).ratio() >= threshold


# ── Select LLM function ────────────────────────────────────────────────────────
query_fn = query_llama_togetherai_freemodel if args.mode == "togetherai_free" else query_llama_deepinfra

# ── Main inference loop ────────────────────────────────────────────────────────
end_idx = args.end if args.end is not None else len(queries)
results = []
prompts = []
greater_than_8k = []

for i in range(args.start, end_idx):
    print(f"Query: {i}")
    query_obj = queries[i]
    gold      = query_obj["answer"]
    prompt    = build_prompt_full_corpus(query_obj)
    prompts.append(prompt)

    try:
        pred = query_fn(prompt)
        results.append({"query": query_obj["query"], "gold": gold, "pred": pred})
    except Exception as e:
        print("API Error:", e)
        greater_than_8k.append(i)
        pred = ""

    print({"query": query_obj["query"], "gold": gold, "pred": pred})

# ── Checkpoint save ────────────────────────────────────────────────────────────
with open(args.checkpoint, "wb") as f:
    pickle.dump({"results": results, "prompts": prompts}, f)
print(f"Checkpoint saved to {args.checkpoint}")

print(f"Length of results: {len(results)}")
print(f"Length of 'greater_than_8k' queries: {len(greater_than_8k)}")

# ── Retry failed queries ───────────────────────────────────────────────────────
for idx in greater_than_8k:
    print(f"Retrying query: {idx}")
    query_obj = queries[idx]
    gold      = query_obj["answer"]
    prompt    = build_prompt_full_corpus(query_obj)
    prompts.append(prompt)
    try:
        pred = query_fn(prompt)
        results.append({"query": query_obj["query"], "gold": gold, "pred": pred})
    except Exception as e:
        print("API Error:", e)
        pred = ""
    print({"query": query_obj["query"], "gold": gold, "pred": pred})

# ── Save final results ─────────────────────────────────────────────────────────
with open(args.output, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"Results saved to {args.output}")

with open("greater_than_8k.txt", "w") as f:
    for num in greater_than_8k:
        f.write(f"{num}\n")

# ── Deduplicate results (if merging multiple checkpoint files) ─────────────────
# To merge multiple checkpoint files, load each and extend results_new, e.g.:
#   results_new = []
#   for filename in ["results100.json", "results500-346.json", ...]:
#       with open(filename, "r", encoding="utf-8") as f:
#           results_new.extend(json.load(f))

results_new = results  # use current run results if not merging

seen = set()
results_non_dup = []
for ex in results_new:
    query = ex["query"]
    if query not in seen:
        seen.add(query)
        results_non_dup.append(ex)

results_new = results_non_dup
print(f"Unique query results: {len(results_new)}")

# ── Evaluation ─────────────────────────────────────────────────────────────────
if results_new:
    preds = [r["pred"]  for r in results_new]
    refs  = [r["gold"]  for r in results_new]

    meteor_result = meteor.compute(predictions=preds, references=refs)
    rouge_result  = rouge.compute(predictions=preds, references=refs)
    mrr_score     = compute_mrr(preds, refs)

    print("\nEvaluation Results")
    print("METEOR:", round(meteor_result["meteor"], 4))
    print("ROUGE:",  {k: round(v, 4) for k, v in rouge_result.items()})
    print("MRR:",    round(mrr_score, 4))

    # Context recall
    results_new_contexts = []
    for ex in results_new:
        query          = ex["query"]
        evidence_list  = query_to_evidence_list.get(query, [])
        evidence_titles = [ev["title"] for ev in evidence_list]
        context_docs = [
            f"[Doc {i+1}] {title_to_doc[title]}"
            for i, title in enumerate(evidence_titles)
            if title in title_to_doc
        ]
        results_new_contexts.append({
            "query":   query,
            "gold":    ex["gold"],
            "pred":    ex["pred"],
            "context": "\n\n".join(context_docs),
        })

    context_recall_count = sum(
        1 for ex in results_new_contexts
        if ex["gold"] in ex["context"] or fuzzy_contains(ex["context"], ex["gold"])
    )
    context_recall = context_recall_count / len(results_new)
    print(f"Context Recall: {context_recall:.4f}")

    # BERTScore (optional – requires bert_score package)
    try:
        from bert_score import score as bert_score
        candidates = [ex["pred"] for ex in results_new]
        references = [ex["gold"] for ex in results_new]
        P, R, F1 = bert_score(
            candidates, references, lang="en",
            model_type="microsoft/deberta-xlarge-mnli", verbose=True,
        )
        print(f"BERTScore Precision: {P.mean().item():.4f}")
        print(f"BERTScore Recall:    {R.mean().item():.4f}")
        print(f"BERTScore F1:        {F1.mean().item():.4f}")
    except ImportError:
        print("bert_score not installed; skipping BERTScore.")
