from src.reranked_retrieval import retrieve
from src.generation import generate_answer


# 1. Run the complete RAG pipeline

def answer_question(query: str) -> dict:
    # Retrieve and rerank relevant documents
    documents = retrieve(query, top_k=5)

    if not documents:
        return {
            "question": query,
            "answer": "No relevant evidence was retrieved.",
            "evidence": [],
            "sources": []
        }

    # Generate an answer using the retrieved evidence
    answer = generate_answer(query, documents)

    # 2. Build the evidence list
    evidence = []

    for document in documents:
        evidence.append({
            "rank": document["rank"],
            "score": float(document["score"]),
            "qid": document["qid"],
            "source": document["source"],
            "url": document["url"],
            "text": document["rag_text"]
        })

    # 3. Collect unique source URLs
    sources = []
    seen_urls = set()

    for document in documents:
        url = document.get("url")

        if url and url not in seen_urls:
            seen_urls.add(url)

            sources.append({
                "source": document.get("source"),
                "url": url
            })

    # 4. Return the complete result
    return {
        "question": query,
        "answer": answer,
        "evidence": evidence,
        "sources": sources
    }


# 5. Test the pipeline

if __name__ == "__main__":
    query = "What are the symptoms of diabetes?"

    result = answer_question(query)

    print("\nQuestion:")
    print(result["question"])

    print("\nGenerated answer:")
    print(result["answer"])

    print("\nRetrieved evidence:")

    for item in result["evidence"]:
        print(f"\nRank: {item['rank']}")
        print(f"Score: {item['score']:.4f}")
        print(f"Question ID: {item['qid']}")
        print(f"Source: {item['source']}")
        print(f"Evidence: {item['text'][:300]}...")

    print("\nSources:")

    for source in result["sources"]:
        print(f"- {source['source']}: {source['url']}")
