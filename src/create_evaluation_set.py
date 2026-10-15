from pathlib import Path
import re
import pandas as pd


# 1. Project paths

BASE_DIR = Path(__file__).resolve().parent.parent

DOCUMENTS_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "rag_documents.csv"
)

EVALUATION_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "evaluation_queries_expanded.csv"
)


# 2. Load dataset

print("Loading dataset...")

df = pd.read_csv(
    DOCUMENTS_FILE,
    dtype=str
).fillna("")

print(f"Loaded {len(df)} documents.")


# 3. Keep the existing evaluation queries

existing_queries = [
    {
        "query": "What are the symptoms of diabetes?",
        "relevant_qids": "0000015-15|0000015-4|0000082-3|0000022-4",
    },
    {
        "query": "What causes asthma?",
        "relevant_qids": "0000010-2",
    },
    {
        "query": "What are the symptoms of breast cancer?",
        "relevant_qids": "0000027_3-3",
    },
    {
        "query": "What is type 2 diabetes?",
        "relevant_qids": "0000022-1",
    },
    {
        "query": "What are the symptoms of diabetic neuropathy?",
        "relevant_qids": "0000034-3",
    },
]

evaluation_rows = list(existing_queries)

existing_question_texts = {
    item["query"].strip().lower()
    for item in existing_queries
}


# 4. Select useful question types

question_types = [
    "symptoms",
    "causes",
    "treatment",
    "exams and tests",
    "prevention",
    "complications",
    "inheritance",
    "frequency",
    "outlook",
    "susceptibility",
]


# 5. Sample real questions from the dataset

for question_type in question_types:

    subset = df[
        df["question_type"]
        .str.strip()
        .str.lower()
        .eq(question_type)
    ].copy()

    # Extract the question from rag_text.
    subset["question"] = subset["rag_text"].str.extract(
        r"Question:\s*(.*?)\s*\n\nAnswer:",
        flags=re.DOTALL,
        expand=False,
    )

    subset = subset.dropna(subset=["question"])

    subset = subset[
        subset["question"].str.strip().ne("")
        & subset["qid"].str.strip().ne("")
    ]

    subset = subset[
        ~subset["question"]
        .str.strip()
        .str.lower()
        .isin(existing_question_texts)
    ]

    examples = subset.sample(
        n=min(3, len(subset)),
        random_state=42,
    )

    for _, row in examples.iterrows():

        question = row["question"].strip()

        evaluation_rows.append({
            "query": question,
            "relevant_qids": row["qid"].strip(),
        })

        existing_question_texts.add(
            question.lower()
        )


# 6. Save expanded evaluation set

evaluation_df = pd.DataFrame(
    evaluation_rows
).drop_duplicates(subset=["query"])

evaluation_df.to_csv(
    EVALUATION_FILE,
    index=False,
)

print(
    f"\nSaved {len(evaluation_df)} queries to:"
)

print(EVALUATION_FILE)

print("\nNew evaluation queries:")

print(
    evaluation_df.iloc[len(existing_queries):]["query"]
    .to_string(index=False)
)

