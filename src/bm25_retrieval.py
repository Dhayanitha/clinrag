from pathlib import Path

import pandas as pd
from rank_bm25 import BM25Okapi


# 1. Project paths

BASE_DIR = Path(__file__).resolve().parent.parent

DOCUMENTS_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "rag_documents.csv"
)


# 2. Load documents

print("Loading documents...")

df = pd.read_csv(DOCUMENTS_FILE)

print(f"Loaded {len(df)} documents.")


# 3. Prepare documents for BM25

print("Preparing documents for BM25...")

documents = df["rag_text"].fillna("").tolist()

tokenized_documents = [
    document.lower().split()
    for document in documents
]


# 4. Build BM25 index

print("Building BM25 index...")

bm25 = BM25Okapi(tokenized_documents)

print("BM25 index ready.")


# 5. Create BM25 retrieval function

def retrieve_bm25(query, top_k=5):

    tokenized_query = query.lower().split()

    scores = bm25.get_scores(tokenized_query)

    top_indices = scores.argsort()[::-1][:top_k]

    results = []

    for rank, index in enumerate(top_indices, start=1):

        row = df.iloc[index]

        results.append({
            "rank": rank,
            "score": float(scores[index]),
            "document_id": row["document_id"],
            "qid": row["qid"],
            "rag_text": row["rag_text"],
            "source": row["source"],
            "url": row["url"],
        })

    return results


# 6. Test BM25 retrieval

if __name__ == "__main__":

    query = "What are the symptoms of diabetes?"

    results = retrieve_bm25(
        query,
        top_k=5
    )

    print("\nQuery:")
    print(query)

    print("\nTop BM25 results:\n")

    for result in results:

        print(f"Rank: {result['rank']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Document ID: {result['document_id']}")
        print(f"QID: {result['qid']}")
        print(f"Source: {result['source']}")

        print("\nRAG Text:")
        print(result["rag_text"][:500])

        print()