from pathlib import Path
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder


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
# 2. Load FAISS index and metadata
# ==========================================

print("=" * 60)
print("LOADING VECTOR STORE")
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

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded successfully!")


# ==========================================
# 4. Load Cross-Encoder reranker
# ==========================================

print("\nLoading Cross-Encoder reranker...")

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

print("Cross-Encoder loaded successfully!")


# ==========================================
# 5. Define query
# ==========================================

query = "What are the symptoms of diabetes?"

print("\n" + "=" * 60)
print(f"QUERY: {query}")
print("=" * 60)


# ==========================================
# 6. First-stage FAISS retrieval
# ==========================================

query_embedding = embedding_model.encode(
    [query],
    convert_to_numpy=True
)

query_embedding = query_embedding.astype("float32")

faiss.normalize_L2(query_embedding)

# Retrieve more candidates than we finally need
candidate_k = min(10, index.ntotal)

scores, indices = index.search(
    query_embedding,
    candidate_k
)

candidates = []

for score, idx in zip(scores[0], indices[0]):

    candidates.append({
        "faiss_score": float(score),
        "metadata_index": int(idx),
        "source": metadata[idx]["source"],
        "chunk_id": metadata[idx]["chunk_id"],
        "text": metadata[idx]["text"]
    })


# ==========================================
# 7. Cross-Encoder reranking
# ==========================================

print("\nReranking retrieved candidates...")

pairs = [
    [query, candidate["text"]]
    for candidate in candidates
]

rerank_scores = reranker.predict(pairs)

for candidate, rerank_score in zip(
    candidates,
    rerank_scores
):
    candidate["rerank_score"] = float(rerank_score)


# Sort by Cross-Encoder score
reranked_candidates = sorted(
    candidates,
    key=lambda x: x["rerank_score"],
    reverse=True
)


# ==========================================
# 8. Display results
# ==========================================

print("\n" + "=" * 60)
print("RERANKED RESULTS")
print("=" * 60)

top_k = min(5, len(reranked_candidates))

for rank, candidate in enumerate(
    reranked_candidates[:top_k],
    start=1
):

    print(f"\n--- RESULT {rank} ---")

    print(
        f"FAISS score: "
        f"{candidate['faiss_score']:.4f}"
    )

    print(
        f"Reranker score: "
        f"{candidate['rerank_score']:.4f}"
    )

    print(f"Source: {candidate['source']}")
    print(f"Chunk ID: {candidate['chunk_id']}")

    print("\nRetrieved text:")

    print(candidate["text"][:800])


print("\n" + "=" * 60)
print("RERANKING TEST COMPLETED")
print("=" * 60)