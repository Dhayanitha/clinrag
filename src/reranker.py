from sentence_transformers import CrossEncoder


# 1. Load reranker model

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

print("Loading reranker model...")

reranker = CrossEncoder(MODEL_NAME)

print("Reranker model loaded.")


# 2. Create reranking function

def rerank(query, documents, top_k=5):

    pairs = [
        (query, document)
        for document in documents
    ]

    scores = reranker.predict(pairs)

    ranked_results = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked_results[:top_k]


# 3. Test reranker

if __name__ == "__main__":

    query = "What are the symptoms of diabetes?"

    documents = [
        "Diabetes can cause increased thirst, frequent urination, fatigue, and blurred vision.",
        "Brittle diabetes is characterized by unstable blood glucose levels.",
        "Asthma can cause coughing, wheezing, and difficulty breathing.",
        "Type 2 diabetes may cause increased thirst, frequent urination, and unexplained weight loss.",
    ]

    results = rerank(
        query,
        documents,
        top_k=4
    )

    print("\nQuery:")
    print(query)

    print("\nReranked results:\n")

    for rank, (document, score) in enumerate(
        results,
        start=1
    ):

        print(f"Rank: {rank}")
        print(f"Score: {score:.4f}")
        print(f"Document: {document}")
        print()