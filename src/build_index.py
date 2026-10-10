from pathlib import Path

import numpy as np
import faiss

# 1. Find project directory

BASE_DIR = Path(__file__).resolve().parent.parent

# 2. Define input and output files

EMBEDDINGS_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "embeddings.npy"
)

INDEX_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "clinical_knowledge.index"
)


# 3. Load embeddings

print("Loading embeddings...")

embeddings = np.load(EMBEDDINGS_FILE)

print("Embedding shape:", embeddings.shape)

# 4. Get embedding dimension

dimension = embeddings.shape[1]

print("Embedding dimension:", dimension)


# 5. Create FAISS index

index = faiss.IndexFlatIP(dimension)


# 6. Add embeddings to the index

index.add(embeddings)

print("Number of vectors in index:", index.ntotal)


# 7. Save the index

faiss.write_index(index, str(INDEX_FILE))

print("\nFAISS index saved to:")
print(INDEX_FILE)