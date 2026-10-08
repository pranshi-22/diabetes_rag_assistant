from pathlib import Path
import json

import faiss
from sentence_transformers import SentenceTransformer


# ==========================================
# 1. Define paths
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VECTOR_STORE_FOLDER = (
    PROJECT_ROOT
    / "data"
    / "knowledge_base"
    / "vector_store"
)

INDEX_PATH = VECTOR_STORE_FOLDER / "diabetes_faiss.index"
METADATA_PATH = VECTOR_STORE_FOLDER / "metadata.json"


# ==========================================
# 2. Load vector store
# ==========================================

print("=" * 60)
print("LOADING DIABETES VECTOR STORE")
print("=" * 60)

index = faiss.read_index(str(INDEX_PATH))

with METADATA_PATH.open("r", encoding="utf-8") as f:
    metadata = json.load(f)

print(f"Vectors loaded: {index.ntotal}")
print(f"Metadata records: {len(metadata)}")


# ==========================================
# 3. Load embedding model
# ==========================================

print("\nLoading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded successfully!")


# ==========================================
# 4. Define test query
# ==========================================

query = "What are the symptoms of diabetes?"

print("\n" + "=" * 60)
print(f"QUERY: {query}")
print("=" * 60)


# ==========================================
# 5. Convert query into embedding
# ==========================================

query_embedding = model.encode(
    [query],
    convert_to_numpy=True
)

query_embedding = query_embedding.astype("float32")

faiss.normalize_L2(query_embedding)


# ==========================================
# 6. Search FAISS
# ==========================================

top_k = 5

scores, indices = index.search(
    query_embedding,
    top_k
)


# ==========================================
# 7. Display retrieved results
# ==========================================

print("\nTOP RETRIEVED RESULTS")
print("=" * 60)

for rank, (score, idx) in enumerate(
    zip(scores[0], indices[0]),
    start=1
):

    document = metadata[idx]

    print(f"\n--- RESULT {rank} ---")
    print(f"Similarity score: {score:.4f}")
    print(f"Source: {document['source']}")
    print(f"Chunk ID: {document['chunk_id']}")
    print("\nRetrieved text:")
    print(document["text"][:800])


print("\n" + "=" * 60)
print("RETRIEVAL TEST COMPLETED")
print("=" * 60)