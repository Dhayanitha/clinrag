from pathlib import Path
import re
import html
import pandas as pd


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "clinical_knowledge.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "rag_documents.csv"
)

# 2. Text cleaning

def clean_text(text):
    """
    Clean a text value before using it in the RAG pipeline.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Convert HTML entities such as &nbsp; and &amp;
    text = html.unescape(text)

    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove unnecessary spaces before punctuation
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)

    return text.strip()

# 3. Create text for embedding

def create_rag_text(row):
    """
    Combine useful fields into one searchable text.
    """

    parts = []

    if row["focus_clean"]:
        parts.append(
            f"Topic: {row['focus_clean']}"
        )

    if row["synonyms_clean"]:
        parts.append(
            f"Related terms: {row['synonyms_clean']}"
        )

    parts.append(
        f"Question: {row['question_clean']}"
    )

    parts.append(
        f"Answer: {row['answer_clean']}"
    )

    return "\n\n".join(parts)

# 4. Main cleaning pipeline

def clean_dataset():

    print("Loading dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Input records: {len(df)}")

    # Clean text columns
    df["question_clean"] = (
        df["question"].apply(clean_text)
    )

    df["answer_clean"] = (
        df["answer"].apply(clean_text)
    )

    df["focus_clean"] = (
        df["focus"].apply(clean_text)
    )

    df["synonyms_clean"] = (
        df["synonyms"].apply(clean_text)
    )

    # Create RAG text
    df["rag_text"] = df.apply(
        create_rag_text,
        axis=1
    )

    # Keep only the fields needed by the RAG system
    rag_df = df[
        [
            "document_id",
            "qid",
            "rag_text",
            "source",
            "url",
            "focus",
            "question_type",
            "umls_cui",
            "semantic_group",
        ]
    ].copy()

    # Remove records where RAG text is empty
    rag_df = rag_df[
        rag_df["rag_text"].str.strip() != ""
    ]

    # Save
    rag_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    print(f"Output records: {len(rag_df)}")
    print(f"Saved to: {OUTPUT_FILE}")

    return rag_df

# 5. Run the pipeline

if __name__ == "__main__":
    clean_dataset()