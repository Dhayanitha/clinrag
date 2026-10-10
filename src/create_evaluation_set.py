from pathlib import Path

import pandas as pd


# 1. Project paths

BASE_DIR = Path(__file__).resolve().parent.parent

DOCUMENTS_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "rag_documents.csv"
)


# 2. Load dataset

print("Loading dataset...")

df = pd.read_csv(
    DOCUMENTS_FILE,
    dtype=str
)

print(
    f"Loaded {len(df)} documents."
)


# 3. Select useful question types

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


# 4. Display examples

for question_type in question_types:

    subset = df[
        df["question_type"]
        .str.lower()
        .eq(question_type)
    ]

    if len(subset) == 0:
        continue

    print("\n")
    print("=" * 70)
    print(
        f"QUESTION TYPE: {question_type.upper()}"
    )
    print("=" * 70)

    examples = subset.sample(
        min(3, len(subset)),
        random_state=42
    )

    for _, row in examples.iterrows():

        print("\nQuestion:")
        print(row["question"])

        print("\nQID:")
        print(row["qid"])

        print("\nDocument ID:")
        print(row["document_id"])

        print("\nTopic:")
        print(row["focus"])

        print()