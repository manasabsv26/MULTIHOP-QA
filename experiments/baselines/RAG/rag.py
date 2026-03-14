"""
RAG Pipeline for MultiHopRAG
-------------------------------
Stages:
  1. Index  – loads corpus.json, splits into nodes, uploads embeddings to Qdrant.
  2. Retrieve – for every query in MultiHopRAG.json, retrieves top-k chunks and saves results.
  3. QA (Kindo / GPT-4o) – runs an LLM over retrieved chunks and saves answers.

Run from experiments/baselines/RAG/ directory.

Usage examples:
  # Index + retrieve + save retrieval results
  python rag.py --mode index_retrieve

  # Skip indexing (already done), just retrieve
  python rag.py --mode retrieve

  # Run QA with Kindo (llama3-70b-8192) over existing retrieval output
  python rag.py --mode qa_kindo

  # Run QA with GPT-4o over existing retrieval output
  python rag.py --mode qa_gpt4o
"""

import os
import json
import time
import argparse
from io import StringIO

from dotenv import load_dotenv

load_dotenv()

# ── Imports ────────────────────────────────────────────────────────────────────
from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core.node_parser import SentenceSplitter
import qdrant_client

from kindo_api_methods import KindoAPI

# ── Argument parsing ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="RAG pipeline for MultiHopRAG.")
parser.add_argument("--corpus",      type=str, default="corpus.json",
                    help="Path to corpus.json (relative to cwd).")
parser.add_argument("--queries",     type=str, default="MultiHopRAG.json",
                    help="Path to MultiHopRAG.json (relative to cwd).")
parser.add_argument("--retrieval_output", type=str, default="multihop_qa_256_output.json",
                    help="Output file for retrieval results.")
parser.add_argument("--qa_output",   type=str, default="multihop_qa_256_final_output.json",
                    help="Output file for Kindo QA results.")
parser.add_argument("--gpt4o_output", type=str, default="multihop_qa_256_gpt4o_output.json",
                    help="Output file for GPT-4o QA results.")
parser.add_argument("--chunk_size",  type=int, default=256,
                    help="Chunk size for node parser.")
parser.add_argument("--collection",  type=str, default="bge-large-256-embedds",
                    help="Qdrant collection name.")
parser.add_argument("--mode", type=str,
                    choices=["index_retrieve", "retrieve", "qa_kindo", "qa_gpt4o"],
                    default="retrieve",
                    help="Pipeline stage to run.")
args = parser.parse_args()

# ── Clients ────────────────────────────────────────────────────────────────────
embed_model = FastEmbedEmbedding(model_name="BAAI/bge-large-en-v1.5")

quadrant_client = qdrant_client.QdrantClient(
    url=os.getenv("QUADRANT_DB_URL"),
    api_key=os.getenv("QUADRANT_API_KEY"),
)

vector_store = QdrantVectorStore(
    client=quadrant_client,
    collection_name=args.collection,
    embedding_dim=1024,  # dimension for bge-large-en-v1.5
)

# ── Helper functions ───────────────────────────────────────────────────────────

def load_corpus(json_path):
    """Read corpus.json and convert to LlamaIndex Document objects."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    documents = []
    for item in data:
        documents.append(Document(
            text=item["body"],
            metadata={
                "title":        item["title"],
                "author":       item.get("author", ""),
                "source":       item.get("source", ""),
                "published_at": item.get("published_at", ""),
                "category":     item.get("category", ""),
                "url":          item.get("url", ""),
            },
        ))
    return documents


def search_similar(query, limit=10):
    """Retrieve top-k similar document chunks from Qdrant for a given query."""
    query_embedding = embed_model.get_text_embedding(query)
    search_result = quadrant_client.search(
        collection_name=args.collection,
        query_vector=query_embedding,
        limit=limit,
    )
    data_list = []
    for point in search_result:
        node_content = json.loads(point.payload.get("_node_content", "{}"))
        data_list.append({
            "text": (
                f"[Excerpt from document]\n"
                f"title: {point.payload.get('title')}\n"
                f"published_at: {point.payload.get('published_at')}\n"
                f"source: {point.payload.get('source')}\n"
                f"Excerpt:\n-----\n{node_content.get('text', '')}"
            ),
            "score": point.score,
        })
    return data_list


def get_relevant_docs(json_path):
    """For each query in MultiHopRAG.json, retrieve similar docs and return records."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    documents = []
    for item in data:
        documents.append({
            "query":          item["query"],
            "question_type":  item["question_type"],
            "retrieval_list": search_similar(item["query"]),
            "gold_list":      item["evidence_list"],
        })
    return documents


# ── Stage: Index ───────────────────────────────────────────────────────────────
if args.mode == "index_retrieve":
    print("Loading corpus and creating nodes …")
    documents = load_corpus(args.corpus)
    node_parser = SentenceSplitter(chunk_size=args.chunk_size, chunk_overlap=25)
    nodes = node_parser.get_nodes_from_documents(documents)
    print(f"  {len(nodes)} nodes created from {len(documents)} documents.")

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )
    print("Indexing complete.")

# ── Stage: Retrieve ────────────────────────────────────────────────────────────
if args.mode in ("index_retrieve", "retrieve"):
    print(f"Retrieving docs for queries in {args.queries} …")
    similar_docs = get_relevant_docs(args.queries)

    with open(args.retrieval_output, "w", encoding="utf-8") as out_file:
        json.dump(similar_docs, out_file, indent=6)
    print(f"Retrieval results saved to {args.retrieval_output}")

# ── Stage: QA with Kindo (llama3-70b-8192) ────────────────────────────────────
if args.mode == "qa_kindo":
    kindo_api = KindoAPI(api_key=os.getenv("KINDO_API_KEY"))

    with open(args.retrieval_output, encoding="utf-8") as f:
        qa_docs = json.load(f)

    PREFIX = (
        "Below is a question followed by some context from different sources. "
        "Please answer the question based on the context. "
        "The answer to the question is a word or entity. "
        "If the provided information is insufficient to answer the question, "
        "respond 'Insufficient Information'. Answer directly without explanation."
    )

    for i in range(len(qa_docs)):
        query     = qa_docs[i]["query"]
        ret_texts = [doc["text"] for doc in qa_docs[i]["retrieval_list"]]
        response  = kindo_api.call_kindo_api(
            model="groq/llama3-70b-8192",
            messages=[{"role": "user", "content": f"{PREFIX}:{query}:\n{chr(10).join(ret_texts)}"}],
            max_tokens=200,
        ).json()["choices"][0]["message"]["content"]
        qa_docs[i]["answer"] = response
        print(i, response)
        if i % 10 == 0:
            time.sleep(2)

    with open(args.qa_output, "w", encoding="utf-8") as out_file:
        json.dump(qa_docs, out_file, indent=6)
    print(f"Kindo QA results saved to {args.qa_output}")

# ── Stage: QA with GPT-4o (via Kindo) ─────────────────────────────────────────
if args.mode == "qa_gpt4o":
    kindo_api = KindoAPI(api_key=os.getenv("KINDO_API_KEY"))

    with open(args.retrieval_output, encoding="utf-8") as f:
        gpt4o_docs = json.load(f)

    PREFIX = (
        "Below is a question followed by some context from different sources. "
        "Please answer the question based on the context. "
        "The answer to the question is a word or entity. "
        "If the provided information is insufficient to answer the question, "
        "respond 'Insufficient Information'. Answer directly without explanation."
    )

    total_tokens = 0
    for i in range(len(gpt4o_docs)):
        query     = gpt4o_docs[i]["query"]
        ret_texts = [doc["text"] for doc in gpt4o_docs[i]["retrieval_list"]]
        total_tokens += len("".join(ret_texts).split())
        response = kindo_api.call_kindo_api(
            model="azure/gpt-4o",
            messages=[{"role": "user", "content": f"{PREFIX}:{query}:\n{chr(10).join(ret_texts)}"}],
            max_tokens=200,
        ).json()["choices"][0]["message"]["content"]
        gpt4o_docs[i]["answer"] = response
        print(i, response)
        if i % 5 == 0:
            print(f"  Approx token count so far: {total_tokens}")
            total_tokens = 0
            print("  Sleeping 60s to respect rate limits …")
            time.sleep(60)

    with open(args.gpt4o_output, "w", encoding="utf-8") as out_file:
        json.dump(gpt4o_docs, out_file, indent=6)
    print(f"GPT-4o QA results saved to {args.gpt4o_output}")
