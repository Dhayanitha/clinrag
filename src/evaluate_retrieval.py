from pathlib import Path

import pandas as pd

from retrieval import retrieve as dense_retrieve
from bm25_retrieval import retrieve_bm25
from hybrid_retrieval import retrieve_hybrid
from reranked_retrieval import retrieve as reranked_retrieve


# 1. Project paths

BASE_DIR = Path(__file__).resolve().parent.parent

EVALUATION_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "evaluation_queries.csv"
)


# 2. Load evaluation queries

print("Loading evaluation dataset...")

evaluation_df = pd.read_csv(
    EVALUATION_FILE,
    dtype=str
)

print(
    f"Loaded {len(evaluation_df)} "
    f"evaluation queries."
)


# 3. Check whether any relevant QID was retrieved

def is_relevant(results, relevant_qids):

    retrieved_qids = [
        result["qid"]
        for result in results
    ]

    return any(
        qid in retrieved_qids
        for qid in relevant_qids
    )


# 4. Calculate Reciprocal Rank

def reciprocal_rank(results, relevant_qids):

    for rank, result in enumerate(
        results,
        start=1
    ):

        if result["qid"] in relevant_qids:
            return 1 / rank

    return 0.0


# 5. Evaluate one retrieval method

def evaluate_method(
    method_name,
    retrieval_function,
    top_k=5
):

    recall_scores = []
    reciprocal_ranks = []

    print(
        f"\nEvaluating: {method_name}"
    )

    for _, row in evaluation_df.iterrows():

        query = row["query"]

        relevant_qids = (
            row["relevant_qids"]
            .split("|")
        )

        results = retrieval_function(
            query,
            top_k=top_k
        )

        recall = int(
            is_relevant(
                results,
                relevant_qids
            )
        )

        rr = reciprocal_rank(
            results,
            relevant_qids
        )

        recall_scores.append(recall)
        reciprocal_ranks.append(rr)

        print(
            f"\nQuery: {query}"
        )

        print(
            f"Relevant QIDs: "
            f"{relevant_qids}"
        )

        print(
            "Retrieved QIDs:"
        )

        for result in results:

            print(
                f"  Rank {result['rank']}: "
                f"{result['qid']}"
            )

        print(
            f"Recall@{top_k}: {recall}"
        )

        print(
            f"Reciprocal Rank: "
            f"{rr:.4f}"
        )

    recall_at_k = (
        sum(recall_scores)
        / len(recall_scores)
    )

    mrr = (
        sum(reciprocal_ranks)
        / len(reciprocal_ranks)
    )

    return {
        "method": method_name,
        f"Recall@{top_k}": recall_at_k,
        "MRR": mrr
    }


# 6. Evaluate all retrieval methods

if __name__ == "__main__":

    results = []

    results.append(
        evaluate_method(
            "FAISS",
            dense_retrieve,
            top_k=5
        )
    )

    results.append(
        evaluate_method(
            "BM25",
            retrieve_bm25,
            top_k=5
        )
    )

    results.append(
        evaluate_method(
            "Hybrid RRF",
            retrieve_hybrid,
            top_k=5
        )
    )

    results.append(
        evaluate_method(
            "Hybrid + CrossEncoder",
            reranked_retrieve,
            top_k=5
        )
    )


# 7. Display comparison

    comparison_df = pd.DataFrame(
        results
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "RETRIEVAL EVALUATION RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        comparison_df.to_string(
            index=False
        )
    )