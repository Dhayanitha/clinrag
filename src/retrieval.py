from pathlib import Path
import numpy as np
import pandas as pd
import faiss
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

index = faiss.read_index(
    str(INDEX_FILE)
)

print(
    f"Loaded FAISS index with {index.ntotal} vectors."
)

# 4. Load embedding model

print("Loading embedding model...")

MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)

print("Model loaded.")

# 5. Retrieval function

def retrieve(query, top_k=5):

    # Convert query into an embedding
    query_embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # FAISS expects a 2D array
    query_embedding = query_embedding.reshape(1, -1)

    # Search the FAISS index
    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for rank, (score, index_id) in enumerate(
        zip(scores[0], indices[0]),
        start=1
    ):

        row = df.iloc[index_id]

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

# 6. Test retrieval

if __name__ == "__main__":

    query = "What are the symptoms of diabetes?"

    results = retrieve(
        query,
        top_k=5
    )

    print("\nQuery:")
    print(query)

    print("\nTop results:\n")

    for result in results:

        print("=" * 80)

        print(f"Rank: {result['rank']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Document ID: {result['document_id']}")
        print(f"QID: {result['qid']}")
        print(f"Source: {result['source']}")
        print(f"URL: {result['url']}")

        print("\nRAG Text:")
        print(result["rag_text"][:500])