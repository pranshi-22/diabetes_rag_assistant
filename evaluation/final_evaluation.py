import os
import json
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

EVALUATION_DIR = os.path.join(
    BASE_DIR,
    "evaluation"
)

MODEL_RESULTS_FILE = os.path.join(
    EVALUATION_DIR,
    "model_comparison.csv"
)

ABLATION_FILE = os.path.join(
    EVALUATION_DIR,
    "ablation_results.json"
)

RETRIEVAL_FILE = os.path.join(
    EVALUATION_DIR,
    "retrieval_results.json"
)


# ============================================================
# LOAD RESULTS
# ============================================================

print("=" * 70)
print("FINAL DIABETES AI ASSISTANT EVALUATION")
print("=" * 70)


# ============================================================
# MODEL RESULTS
# ============================================================

model_results = None

if os.path.exists(
    MODEL_RESULTS_FILE
):

    model_results = pd.read_csv(
        MODEL_RESULTS_FILE
    )

    print("\n")
    print("=" * 70)
    print("MACHINE LEARNING RESULTS")
    print("=" * 70)

    print(
        model_results.to_string(
            index=False
        )
    )

else:

    print(
        "\nModel comparison file not found:"
    )

    print(
        MODEL_RESULTS_FILE
    )


# ============================================================
# ABLATION RESULTS
# ============================================================

ablation_results = None

if os.path.exists(
    ABLATION_FILE
):

    with open(
        ABLATION_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        ablation_results = json.load(f)

    print("\n")
    print("=" * 70)
    print("RAG ABLATION RESULTS")
    print("=" * 70)

    baseline = ablation_results[
        "baseline"
    ]

    reranked = ablation_results[
        "reranked"
    ]

    print(
        f"Semantic Retrieval:"
    )

    print(
        f"  Top-1 overlap: "
        f"{baseline['average_top1_overlap']:.4f}"
    )

    print(
        f"  Document overlap: "
        f"{baseline['average_document_overlap']:.4f}"
    )

    print(
        f"  MRR: "
        f"{baseline['mrr']:.4f}"
    )

    print(
        "\nSemantic Retrieval + Reranker:"
    )

    print(
        f"  Top-1 overlap: "
        f"{reranked['average_top1_overlap']:.4f}"
    )

    print(
        f"  Document overlap: "
        f"{reranked['average_document_overlap']:.4f}"
    )

    print(
        f"  MRR: "
        f"{reranked['mrr']:.4f}"
    )

    print(
        "\nImprovements:"
    )

    print(
        f"  Document overlap: "
        f"{ablation_results['document_overlap_improvement_percent']:.2f}%"
    )

    print(
        f"  MRR: "
        f"{ablation_results['mrr_improvement_percent']:.2f}%"
    )

else:

    print(
        "\nAblation results file not found:"
    )

    print(
        ABLATION_FILE
    )


# ============================================================
# RETRIEVAL RESULTS
# ============================================================

retrieval_results = None

if os.path.exists(
    RETRIEVAL_FILE
):

    with open(
        RETRIEVAL_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        retrieval_results = json.load(f)

    print("\n")
    print("=" * 70)
    print("RETRIEVAL EVALUATION")
    print("=" * 70)

    print(
        f"Questions evaluated: "
        f"{len(retrieval_results)}"
    )

    average_overlap = sum(
        result["average_overlap"]
        for result in retrieval_results
    ) / len(retrieval_results)

    average_top_overlap = sum(
        result["top_overlap"]
        for result in retrieval_results
    ) / len(retrieval_results)

    print(
        f"Average keyword overlap: "
        f"{average_overlap:.4f}"
    )

    print(
        f"Average top-document overlap: "
        f"{average_top_overlap:.4f}"
    )

else:

    print(
        "\nRetrieval results file not found:"
    )

    print(
        RETRIEVAL_FILE
    )


# ============================================================
# FINAL PROJECT SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("FINAL PROJECT SUMMARY")
print("=" * 70)

print(
    """
Project:
Retrieval-Augmented Conversational Assistant for Diabetes

Phase 1:
- XGBoost-based diabetes risk assessment
- SHAP-based model explainability
- Input validation
- Risk context generation

Phase 2:
- Trusted diabetes knowledge base
- PDF document ingestion
- Semantic retrieval using FAISS
- Cross-encoder reranking
- Evidence-grounded responses
- Safety-aware query handling
- Risk-aware conversational context
- Streamlit prototype

Evaluation:
- Machine learning classification metrics
- Retrieval overlap
- Mean Reciprocal Rank (MRR)
- Reranker ablation study
"""
)


# ============================================================
# FINAL JSON REPORT
# ============================================================

final_report = {
    "project_title":
        "Retrieval-Augmented Conversational Assistant for Diabetes",

    "phase_1": {
        "components": [
            "XGBoost",
            "SHAP",
            "Input Validation",
            "Risk Assessment"
        ]
    },

    "phase_2": {
        "components": [
            "Trusted Knowledge Base",
            "FAISS Semantic Retrieval",
            "Cross-Encoder Reranking",
            "Safety Layer",
            "Risk Context",
            "Grounded Response Generation",
            "Streamlit Interface"
        ]
    }
}


if model_results is not None:

    final_report[
        "machine_learning_results"
    ] = model_results.to_dict(
        orient="records"
    )


if ablation_results is not None:

    final_report[
        "rag_ablation_results"
    ] = ablation_results


if retrieval_results is not None:

    final_report[
        "retrieval_evaluation"
    ] = {
        "question_count":
            len(retrieval_results),
        "average_keyword_overlap":
            average_overlap,
        "average_top_document_overlap":
            average_top_overlap
    }


OUTPUT_FILE = os.path.join(
    EVALUATION_DIR,
    "final_evaluation_report.json"
)


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        final_report,
        f,
        indent=4
    )


print("\n")
print("=" * 70)
print("FINAL REPORT SAVED")
print("=" * 70)

print(
    OUTPUT_FILE
)

print("\nEvaluation pipeline complete.")