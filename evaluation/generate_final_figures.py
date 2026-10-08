import os
import json
import pandas as pd
import matplotlib.pyplot as plt


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


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

FIGURES_DIR = os.path.join(
    EVALUATION_DIR,
    "figures"
)

os.makedirs(
    FIGURES_DIR,
    exist_ok=True
)


print("=" * 70)
print("GENERATING FINAL RESEARCH FIGURES")
print("=" * 70)


# ============================================================
# 1. MACHINE LEARNING MODEL COMPARISON
# ============================================================

model_file = os.path.join(
    EVALUATION_DIR,
    "model_comparison.csv"
)

model_df = pd.read_csv(
    model_file
)

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC-AUC"
]

plt.figure(
    figsize=(10, 6)
)

x = range(
    len(model_df)
)

width = 0.15

for i, metric in enumerate(metrics):

    values = model_df[
        metric
    ].values

    positions = [
        value + (
            i - 2
        ) * width
        for value in x
    ]

    plt.bar(
        positions,
        values,
        width=width,
        label=metric
    )

plt.xticks(
    list(x),
    model_df["Model"],
    rotation=15
)

plt.ylabel(
    "Score"
)

plt.xlabel(
    "Machine Learning Model"
)

plt.title(
    "Comparison of Diabetes Risk Prediction Models"
)

plt.ylim(
    0,
    1.05
)

plt.legend()

plt.tight_layout()

model_output = os.path.join(
    FIGURES_DIR,
    "model_comparison.png"
)

plt.savefig(
    model_output,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"[1/3] Saved: {model_output}"
)


# ============================================================
# 2. RETRIEVAL VS RERANKING
# ============================================================

ablation_file = os.path.join(
    EVALUATION_DIR,
    "ablation_results.json"
)

with open(
    ablation_file,
    "r",
    encoding="utf-8"
) as f:

    ablation = json.load(f)


baseline = ablation[
    "baseline"
]

reranked = ablation[
    "reranked"
]


retrieval_metrics = [
    "Top-1 Overlap",
    "Document Overlap",
    "MRR"
]

baseline_values = [
    baseline["average_top1_overlap"],
    baseline["average_document_overlap"],
    baseline["mrr"]
]

reranked_values = [
    reranked["average_top1_overlap"],
    reranked["average_document_overlap"],
    reranked["mrr"]
]


plt.figure(
    figsize=(9, 6)
)

x = range(
    len(retrieval_metrics)
)

width = 0.35

plt.bar(
    [
        value - width / 2
        for value in x
    ],
    baseline_values,
    width=width,
    label="Semantic Retrieval"
)

plt.bar(
    [
        value + width / 2
        for value in x
    ],
    reranked_values,
    width=width,
    label="Retrieval + Reranker"
)

plt.xticks(
    list(x),
    retrieval_metrics
)

plt.ylabel(
    "Score"
)

plt.xlabel(
    "Retrieval Metric"
)

plt.title(
    "Effect of Cross-Encoder Reranking"
)

plt.ylim(
    0,
    1.05
)

plt.legend()

plt.tight_layout()

retrieval_output = os.path.join(
    FIGURES_DIR,
    "retrieval_reranker_comparison.png"
)

plt.savefig(
    retrieval_output,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"[2/3] Saved: {retrieval_output}"
)


# ============================================================
# 3. ABLATION IMPROVEMENT
# ============================================================

improvement_metrics = [
    "Document Overlap",
    "MRR"
]

improvement_values = [
    ablation[
        "document_overlap_improvement_percent"
    ],
    ablation[
        "mrr_improvement_percent"
    ]
]


plt.figure(
    figsize=(8, 6)
)

bars = plt.bar(
    improvement_metrics,
    improvement_values
)

plt.ylabel(
    "Improvement (%)"
)

plt.xlabel(
    "Metric"
)

plt.title(
    "Improvement from Cross-Encoder Reranking"
)

plt.axhline(
    0,
    linewidth=1
)

for bar, value in zip(
    bars,
    improvement_values
):

    plt.text(
        bar.get_x()
        + bar.get_width() / 2,
        bar.get_height()
        + 0.1,
        f"{value:.2f}%",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

improvement_output = os.path.join(
    FIGURES_DIR,
    "reranker_improvement.png"
)

plt.savefig(
    improvement_output,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"[3/3] Saved: {improvement_output}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("FIGURE GENERATION COMPLETE")
print("=" * 70)

print(
    f"""
Figures created in:

{FIGURES_DIR}

Files:

1. model_comparison.png
2. retrieval_reranker_comparison.png
3. reranker_improvement.png

Existing SHAP figure:

4. shap_summary.png
"""
)

print(
    "All figures use the actual experimental results."
)