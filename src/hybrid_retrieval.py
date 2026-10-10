from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# 1. Project paths

BASE_DIR = Path(__file__).resolve().parent.parent

DOCUMENTS_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "rag_documents.csv"
)

INDEX_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "clinical_knowledge.index"
)


# 2. Load documents

print("Loading documents...")

df = pd.read_csv(DOCUMENTS_FILE)

print(f"Loaded {len(df)} documents.")


# 3. Load FAISS index

print("Loading FAISS index...")

faiss_index = faiss.read_index(
    str(INDEX_FILE)
)

print(
    f"Loaded FAISS index with "
    f"{faiss_index.ntotal} vectors."
)


# 4. Load embedding model

print("Loading embedding model...")

MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)

print("Model loaded.")


# 5. Build BM25 index

print("Building BM25 index...")

documents = df["rag_text"].fillna("").tolist()

tokenized_documents = [
    document.lower().split()
    for document in documents
]

bm25 = BM25Okapi(tokenized_documents)

print("BM25 index ready.")


# 6. Dense retrieval

def retrieve_faiss(query, top_k=10):

    query_embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embedding = query_embedding.reshape(1, -1)

    scores, indices = faiss_index.search(
        query_embedding,
        top_k
    )

    return indices[0].tolist()


# 7. BM25 retrieval

def retrieve_bm25(query, top_k=10):

    tokenized_query = query.lower().split()

    scores = bm25.get_scores(tokenized_query)

    top_indices = scores.argsort()[::-1][:top_k]

    return top_indices.tolist()


# 8. Reciprocal Rank Fusion

def reciprocal_rank_fusion(
    result_lists,
    k=60
):

    fused_scores = {}

    for results in result_lists:

        for rank, document_index in enumerate(
            results,
            start=1
        ):

            if document_index not in fused_scores:
                fused_scores[document_index] = 0

            fused_scores[document_index] += (
                1 / (k + rank)
            )

    ranked_documents = sorted(
        fused_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked_documents


# 9. Hybrid retrieval

def retrieve_hybrid(query, top_k=5):

    faiss_results = retrieve_faiss(
        query,
        top_k=10
    )

    bm25_results = retrieve_bm25(
        query,
        top_k=10
    )

    fused_results = reciprocal_rank_fusion(
        [
            faiss_results,
            bm25_results
        ]
    )

    results = []

    for rank, (index, score) in enumerate(
        fused_results[:top_k],
        start=1
    ):

        row = df.iloc[index]

        results.append({
            "rank": rank,
            "score": float(score),
            "document_id": row["document_id"],
            "qid": row["qid"],
            "rag_text": row["rag_text"],
            "source": row["source"],
            "url": row["url"],
        })

    return results


# 10. Test hybrid retrieval

if __name__ == "__main__":

    query = "What are the symptoms of diabetes?"

    results = retrieve_hybrid(
        query,
        top_k=5
    )

    print("\nQuery:")
    print(query)

    print("\nTop hybrid results:\n")

    for result in results:

        print("=" * 80)

        print(f"Rank: {result['rank']}")
        print(f"RRF Score: {result['score']:.6f}")
        print(f"Document ID: {result['document_id']}")
        print(f"QID: {result['qid']}")
        print(f"Source: {result['source']}")

        print("\nRAG Text:")
        print(result["rag_text"][:500])

        print()