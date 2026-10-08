import os
import sys
import json
import re
import gc

import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

VECTOR_DIR = os.path.join(
    BASE_DIR,
    "data",
    "knowledge_base",
    "vector_store"
)

QUESTIONS_FILE = os.path.join(
    BASE_DIR,
    "evaluation",
    "evaluation_questions.json"
)

INDEX_FILE = os.path.join(
    VECTOR_DIR,
    "diabetes_faiss.index"
)

METADATA_FILE = os.path.join(
    VECTOR_DIR,
    "metadata.json"
)


# ============================================================
# MODELS
# ============================================================

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

RERANKER_MODEL = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

TOP_K = 8
RERANK_TOP_K = 5


# ============================================================
# LOAD DATA
# ============================================================

with open(
    QUESTIONS_FILE,
    "r",
    encoding="utf-8"
) as f:

    questions = json.load(f)


index = faiss.read_index(
    INDEX_FILE
)

with open(
    METADATA_FILE,
    "r",
    encoding="utf-8"
) as f:

    metadata = json.load(f)


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(query):

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    embedding = np.asarray(
        embedding,
        dtype="float32"
    )

    scores, indices = index.search(
        embedding,
        TOP_K
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx < 0:
            continue

        item = metadata[idx].copy()

        item["retrieval_score"] = float(
            score
        )

        results.append(item)

    del model
    gc.collect()

    return results


# ============================================================
# RERANKING
# ============================================================

def rerank(query, documents):

    if not documents:
        return []

    model = CrossEncoder(
        RERANKER_MODEL
    )

    pairs = [
        (
            query,
            document["text"]
        )
        for document in documents
    ]

    scores = model.predict(
        pairs
    )

    for document, score in zip(
        documents,
        scores
    ):

        document["rerank_score"] = float(
            score
        )

    documents.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    result = documents[
        :RERANK_TOP_K
    ]

    del model
    gc.collect()

    return result


# ============================================================
# TEXT QUALITY
# ============================================================

def clean_text(text):

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def keyword_overlap(query, text):

    query_words = set(
        re.findall(
            r"\b[a-zA-Z]{4,}\b",
            query.lower()
        )
    )

    text_words = set(
        re.findall(
            r"\b[a-zA-Z]{4,}\b",
            text.lower()
        )
    )

    if not query_words:
        return 0

    return len(
        query_words.intersection(
            text_words
        )
    ) / len(query_words)


# ============================================================
# EVALUATE RETRIEVAL
# ============================================================

def evaluate_question(question):

    query = question["question"]

    retrieved = retrieve(
        query
    )

    reranked = rerank(
        query,
        retrieved
    )

    if not reranked:

        return {
            "id": question["id"],
            "question": query,
            "category": question["category"],
            "retrieval_count": 0,
            "average_overlap": 0,
            "top_overlap": 0
        }

    overlaps = []

    for document in reranked:

        overlap = keyword_overlap(
            query,
            document["text"]
        )

        overlaps.append(
            overlap
        )

    return {
        "id": question["id"],
        "question": query,
        "category": question["category"],
        "retrieval_count": len(reranked),
        "average_overlap": round(
            float(np.mean(overlaps)),
            4
        ),
        "top_overlap": round(
            float(max(overlaps)),
            4
        ),
        "top_source": reranked[0]["source"],
        "top_chunk": reranked[0]["chunk_id"],
        "top_score": round(
            reranked[0]["rerank_score"],
            4
        )
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("DIABETES RAG RETRIEVAL EVALUATION")
    print("=" * 70)

    results = []

    for i, question in enumerate(
        questions,
        start=1
    ):

        print(
            f"\n[{i}/{len(questions)}] "
            f"{question['question']}"
        )

        result = evaluate_question(
            question
        )

        results.append(result)

        print(
            f"Top overlap: "
            f"{result['top_overlap']}"
        )

        print(
            f"Top reranker score: "
            f"{result.get('top_score', 0)}"
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    average_overlap = np.mean(
        [
            r["average_overlap"]
            for r in results
        ]
    )

    average_top_overlap = np.mean(
        [
            r["top_overlap"]
            for r in results
        ]
    )

    print("\n")
    print("=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    print(
        f"Questions evaluated: {len(results)}"
    )

    print(
        f"Average keyword overlap: "
        f"{average_overlap:.4f}"
    )

    print(
        f"Average top-document overlap: "
        f"{average_top_overlap:.4f}"
    )

    output_file = os.path.join(
        BASE_DIR,
        "evaluation",
        "retrieval_results.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    print(
        f"\nResults saved to:"
    )

    print(
        output_file
    )


if __name__ == "__main__":
    main()