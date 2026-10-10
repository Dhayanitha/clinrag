from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# 1. Project paths

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "rag_documents.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "embeddings.npy"
)

# 2. Load dataset

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Loaded {len(df)} documents.")

# 3. Get text that will be embedded

texts = df["rag_text"].tolist()

print(f"Texts ready for embedding: {len(texts)}")

# 4. Load embedding model

MODEL_NAME = "all-MiniLM-L6-v2"

print(f"Loading model: {MODEL_NAME}")

model = SentenceTransformer(MODEL_NAME)

print("Model loaded.")


# 5. Generate embeddings

print("Generating embeddings...")

embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

# 6. Inspect embeddings

print("\nEmbedding generation complete.")

print("Shape:", embeddings.shape)
print("Data type:", embeddings.dtype)

# 7. Save embeddings

np.save(
    OUTPUT_FILE,
    embeddings
)

print("\nEmbeddings saved to:")
print(OUTPUT_FILE)