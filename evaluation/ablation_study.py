import os
import json
import re
import gc

import numpy as np
import pandas as pd
import faiss

from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
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

RETRIEVE_TOP_K = 8
RERANK_TOP_K = 5


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("DIABETES RAG ABLATION STUDY")
print("=" * 70)

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

def semantic_retrieve(query):

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
        RETRIEVE_TOP_K
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
# RERANK
# ============================================================

def rerank(
    query,
    documents
):

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
# TEXT
# ============================================================

def get_words(text):

    return set(
        re.findall(
            r"\b[a-zA-Z]{4,}\b",
            text.lower()
        )
    )


def keyword_overlap(
    query,
    text
):

    query_words = get_words(
        query
    )

    text_words = get_words(
        text
    )

    if not query_words:
        return 0.0

    return len(
        query_words.intersection(
            text_words
        )
    ) / len(query_words)


# ============================================================
# MRR
# ============================================================

def reciprocal_rank(
    query,
    documents,
    relevance_threshold=0.50
):

    for rank, document in enumerate(
        documents,
        start=1
    ):

        overlap = keyword_overlap(
            query,
            document["text"]
        )

        if overlap >= relevance_threshold:

            return 1.0 / rank

    return 0.0


# ============================================================
# EVALUATE ONE QUESTION
# ============================================================

def evaluate_question(
    question,
    use_reranker=False
):

    query = question["question"]

    retrieved = semantic_retrieve(
        query
    )

    if not retrieved:

        return {
            "question": query,
            "category": question["category"],
            "top1_overlap": 0.0,
            "average_overlap": 0.0,
            "mrr": 0.0
        }

    # --------------------------------------------------------
    # Save original retrieval order
    # --------------------------------------------------------

    if use_reranker:

        documents = rerank(
            query,
            retrieved
        )

    else:

        documents = retrieved[
            :RERANK_TOP_K
        ]

    overlaps = [
        keyword_overlap(
            query,
            document["text"]
        )
        for document in documents
    ]

    top1_overlap = overlaps[0]

    average_overlap = np.mean(
        overlaps
    )

    mrr = reciprocal_rank(
        query,
        documents
    )

    return {
        "question": query,
        "category": question["category"],
        "top1_overlap": float(
            top1_overlap
        ),
        "average_overlap": float(
            average_overlap
        ),
        "mrr": float(
            mrr
        )
    }


# ============================================================
# RUN EXPERIMENT
# ============================================================

def run_experiment(
    name,
    use_reranker
):

    print("\n")
    print("-" * 70)
    print(name)
    print("-" * 70)

    results = []

    for i, question in enumerate(
        questions,
        start=1
    ):

        print(
            f"[{i}/{len(questions)}] "
            f"{question['question']}"
        )

        result = evaluate_question(
            question,
            use_reranker
        )

        results.append(
            result
        )

    avg_top1 = np.mean(
        [
            r["top1_overlap"]
            for r in results
        ]
    )

    avg_overlap = np.mean(
        [
            r["average_overlap"]
            for r in results
        ]
    )

    avg_mrr = np.mean(
        [
            r["mrr"]
            for r in results
        ]
    )

    print(
        f"\nAverage Top-1 Overlap: "
        f"{avg_top1:.4f}"
    )

    print(
        f"Average Document Overlap: "
        f"{avg_overlap:.4f}"
    )

    print(
        f"MRR: "
        f"{avg_mrr:.4f}"
    )

    return {
        "experiment": name,
        "average_top1_overlap": float(
            avg_top1
        ),
        "average_document_overlap": float(
            avg_overlap
        ),
        "mrr": float(
            avg_mrr
        ),
        "question_count": len(results)
    }


# ============================================================
# MAIN
# ============================================================

def main():

    baseline = run_experiment(
        "A: Semantic Retrieval Only",
        False
    )

    reranked = run_experiment(
        "B: Semantic Retrieval + Cross-Encoder Reranker",
        True
    )

    # --------------------------------------------------------
    # Improvements
    # --------------------------------------------------------

    baseline_mrr = baseline["mrr"]
    reranked_mrr = reranked["mrr"]

    if baseline_mrr > 0:

        mrr_improvement = (
            (
                reranked_mrr
                - baseline_mrr
            )
            / baseline_mrr
        ) * 100

    else:

        mrr_improvement = 0.0

    baseline_overlap = (
        baseline["average_document_overlap"]
    )

    reranked_overlap = (
        reranked["average_document_overlap"]
    )

    if baseline_overlap > 0:

        overlap_improvement = (
            (
                reranked_overlap
                - baseline_overlap
            )
            / baseline_overlap
        ) * 100

    else:

        overlap_improvement = 0.0

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("ABLATION STUDY SUMMARY")
    print("=" * 70)

    print(
        f"{'Metric':<35}"
        f"{'Retrieval':>15}"
        f"{'Reranked':>15}"
    )

    print("-" * 70)

    print(
        f"{'Top-1 Overlap':<35}"
        f"{baseline['average_top1_overlap']:>15.4f}"
        f"{reranked['average_top1_overlap']:>15.4f}"
    )

    print(
        f"{'Average Document Overlap':<35}"
        f"{baseline_overlap:>15.4f}"
        f"{reranked_overlap:>15.4f}"
    )

    print(
        f"{'MRR':<35}"
        f"{baseline_mrr:>15.4f}"
        f"{reranked_mrr:>15.4f}"
    )

    print("-" * 70)

    print(
        f"Document-overlap improvement: "
        f"{overlap_improvement:.2f}%"
    )

    print(
        f"MRR improvement: "
        f"{mrr_improvement:.2f}%"
    )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    output = {
        "baseline": baseline,
        "reranked": reranked,
        "document_overlap_improvement_percent":
            float(overlap_improvement),
        "mrr_improvement_percent":
            float(mrr_improvement)
    }

    json_file = os.path.join(
        BASE_DIR,
        "evaluation",
        "ablation_results.json"
    )

    with open(
        json_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    dataframe = pd.DataFrame(
        [
            baseline,
            reranked
        ]
    )

    csv_file = os.path.join(
        BASE_DIR,
        "evaluation",
        "ablation_results.csv"
    )

    dataframe.to_csv(
        csv_file,
        index=False
    )

    print("\nResults saved:")
    print(json_file)
    print(csv_file)


if __name__ == "__main__":
    main()