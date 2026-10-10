from pathlib import Path

import faiss
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder


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

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("Embedding model loaded.")


# 5. Build BM25 index

print("Building BM25 index...")

documents = df["rag_text"].fillna("").tolist()

tokenized_documents = [
    document.lower().split()
    for document in documents
]

bm25 = BM25Okapi(tokenized_documents)

print("BM25 index ready.")


# 6. Load CrossEncoder reranker

print("Loading reranker model...")

RERANKER_MODEL = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

reranker = CrossEncoder(
    RERANKER_MODEL
)

print("Reranker model loaded.")


# 7. FAISS retrieval

def retrieve_faiss(query, top_k=10):

    query_embedding = embedding_model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embedding = query_embedding.reshape(
        1,
        -1
    )

    scores, indices = faiss_index.search(
        query_embedding,
        top_k
    )

    return indices[0].tolist()


# 8. BM25 retrieval

def retrieve_bm25(query, top_k=10):

    tokenized_query = query.lower().split()

    scores = bm25.get_scores(
        tokenized_query
    )

    top_indices = scores.argsort()[::-1][:top_k]

    return top_indices.tolist()


# 9. Reciprocal Rank Fusion

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


# 10. Rerank candidates

def rerank_candidates(
    query,
    candidate_indices,
    top_k=5
):

    candidate_documents = [
        df.iloc[index]["rag_text"]
        for index in candidate_indices
    ]

    pairs = [
        (query, document)
        for document in candidate_documents
    ]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(candidate_indices, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked[:top_k]


# 11. Complete retrieval pipeline

def retrieve(
    query,
    candidate_k=10,
    top_k=5
):

    faiss_results = retrieve_faiss(
        query,
        top_k=candidate_k
    )

    bm25_results = retrieve_bm25(
        query,
        top_k=candidate_k
    )

    fused_results = reciprocal_rank_fusion(
        [
            faiss_results,
            bm25_results
        ]
    )

    candidate_indices = [
        index
        for index, score in fused_results[:candidate_k]
    ]

    reranked_results = rerank_candidates(
        query,
        candidate_indices,
        top_k=top_k
    )

    results = []

    for rank, (index, score) in enumerate(
        reranked_results,
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


# 12. Test the complete pipeline

if __name__ == "__main__":

    query = "What are the symptoms of diabetes?"

    results = retrieve(
        query,
        candidate_k=10,
        top_k=5
    )

    print("\nQuery:")
    print(query)

    print("\nFinal reranked results:\n")

    for result in results:

        print("=" * 80)

        print(f"Rank: {result['rank']}")
        print(
            f"Reranker Score: "
            f"{result['score']:.4f}"
        )
        print(
            f"Document ID: "
            f"{result['document_id']}"
        )
        print(f"QID: {result['qid']}")
        print(f"Source: {result['source']}")

        print("\nRAG Text:")
        print(result["rag_text"][:500])

        print() 