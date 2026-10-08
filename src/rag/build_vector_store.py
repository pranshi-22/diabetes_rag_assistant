from pathlib import Path
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ==========================================
# 1. Define folders
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHUNKS_FOLDER = PROJECT_ROOT / "data" / "knowledge_base" / "chunks"
VECTOR_STORE_FOLDER = PROJECT_ROOT / "data" / "knowledge_base" / "vector_store"

VECTOR_STORE_FOLDER.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. Load chunks from text files
# ==========================================

def load_chunks():

    documents = []

    chunk_files = sorted(CHUNKS_FOLDER.glob("*_chunks.txt"))

    if not chunk_files:
        print("No chunk files found!")
        return documents

    for chunk_file in chunk_files:

        print(f"Loading: {chunk_file.name}")

        text = chunk_file.read_text(encoding="utf-8")

        # Split using our chunk markers
        raw_chunks = text.split("--- CHUNK ")

        for raw_chunk in raw_chunks:

            raw_chunk = raw_chunk.strip()

            if not raw_chunk:
                continue

            # Remove chunk number from beginning
            parts = raw_chunk.split("---", 1)

            if len(parts) == 2:
                chunk_number = parts[0].strip()
                chunk_text = parts[1].strip()
            else:
                chunk_number = "unknown"
                chunk_text = raw_chunk

            if chunk_text:

                documents.append({
                    "source": chunk_file.stem,
                    "chunk_id": chunk_number,
                    "text": chunk_text
                })

    return documents


# ==========================================
# 3. Build FAISS vector store
# ==========================================

def main():

    print("=" * 60)
    print("BUILDING DIABETES VECTOR STORE")
    print("=" * 60)

    # Load documents
    documents = load_chunks()

    if not documents:
        print("No documents available.")
        return

    print(f"\nTotal chunks loaded: {len(documents)}")

    # ======================================
    # Load embedding model
    # ======================================

    print("\nLoading embedding model...")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Embedding model loaded successfully!")

    # ======================================
    # Create embeddings
    # ======================================

    texts = [doc["text"] for doc in documents]

    print("\nCreating embeddings...")

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    embeddings = embeddings.astype("float32")

    print(f"Embedding shape: {embeddings.shape}")

    # ======================================
    # Normalize embeddings
    # ======================================

    faiss.normalize_L2(embeddings)

    # ======================================
    # Create FAISS index
    # ======================================

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    print(f"FAISS index created!")
    print(f"Number of vectors: {index.ntotal}")
    print(f"Vector dimension: {dimension}")

    # ======================================
    # Save FAISS index
    # ======================================

    index_path = VECTOR_STORE_FOLDER / "diabetes_faiss.index"

    faiss.write_index(index, str(index_path))

    # ======================================
    # Save metadata
    # ======================================

    metadata_path = VECTOR_STORE_FOLDER / "metadata.json"

    with metadata_path.open("w", encoding="utf-8") as f:

        json.dump(
            documents,
            f,
            indent=2,
            ensure_ascii=False
        )

    # ======================================
    # Final output
    # ======================================

    print("\n" + "=" * 60)
    print("VECTOR STORE CREATED SUCCESSFULLY")
    print("=" * 60)

    print(f"FAISS index saved to:")
    print(index_path)

    print(f"\nMetadata saved to:")
    print(metadata_path)

    print(f"\nTotal vectors: {index.ntotal}")
    print(f"Vector dimension: {dimension}")


# ==========================================
# 4. Run
# ==========================================

if __name__ == "__main__":
    main()