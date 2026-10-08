import os
import gc
import json
import re
import builtins
import requests
import numpy as np
import pandas as pd
import faiss
import shap

from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder

# Add project root / prediction folder to Python path
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

from src.prediction.risk_predictor import (
    predict_diabetes_risk,
    load_model,
    create_input_dataframe,
    preprocess_for_model,
    FEATURE_COLUMNS,
)

from src.prediction.risk_context import (
    save_risk_context,
    load_risk_context,
    clear_risk_context,
)

from src.rag.safety import check_safety


# ============================================================
# PATHS
# ============================================================

VECTOR_STORE_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "knowledge_base",
    "vector_store",
)

FAISS_INDEX_PATH = os.path.join(
    VECTOR_STORE_DIR,
    "diabetes_faiss.index",
)

METADATA_PATH = os.path.join(
    VECTOR_STORE_DIR,
    "metadata.json",
)


# ============================================================
# MODELS
# ============================================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

OLLAMA_MODEL = "qwen3:4b"

OLLAMA_URL = "http://localhost:11434/api/generate"


# ============================================================
# SETTINGS
# ============================================================

RETRIEVE_TOP_K = 8
RERANK_TOP_K = 5

MAX_GENERATION_TOKENS = 160


# ============================================================
# LOAD VECTOR STORE
# ============================================================

def load_vector_store():

    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(
            f"FAISS index not found:\n{FAISS_INDEX_PATH}"
        )

    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(
            f"Metadata file not found:\n{METADATA_PATH}"
        )

    index = faiss.read_index(FAISS_INDEX_PATH)

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        metadata = json.load(f)

    return index, metadata


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(query, top_k=RETRIEVE_TOP_K):

    index, metadata = load_vector_store()

    print("\n[1/4] Retrieving trusted evidence...")

    model = SentenceTransformer(EMBEDDING_MODEL)

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    scores, indices = index.search(
        np.asarray(query_embedding),
        top_k
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx < 0 or idx >= len(metadata):
            continue

        item = metadata[idx].copy()

        item["retrieval_score"] = float(score)

        results.append(item)

    del model
    gc.collect()

    return results


# ============================================================
# RERANKING
# ============================================================

def rerank(query, documents, top_k=RERANK_TOP_K):

    if not documents:
        return []

    print("[2/4] Reranking evidence...")

    reranker = CrossEncoder(RERANKER_MODEL)

    pairs = [
        (
            query,
            document["text"]
        )
        for document in documents
    ]

    scores = reranker.predict(pairs)

    for document, score in zip(
        documents,
        scores
    ):
        document["rerank_score"] = float(score)

    documents.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    results = documents[:top_k]

    del reranker
    gc.collect()

    return results


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(text):

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# BUILD EVIDENCE
# ============================================================

def build_evidence(documents):

    evidence_parts = []

    for i, document in enumerate(documents, start=1):

        source = document.get(
            "source",
            "Unknown source"
        )

        chunk_id = document.get(
            "chunk_id",
            "Unknown"
        )

        text = clean_text(
            document.get(
                "text",
                ""
            )
        )

        evidence_parts.append(
            f"[SOURCE {i}]\n"
            f"Document: {source}\n"
            f"Chunk: {chunk_id}\n"
            f"Content: {text}"
        )

    return "\n\n".join(evidence_parts)


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(query, evidence):

    print("[3/4] Generating grounded answer...")

    prompt = f"""
You are a diabetes education assistant.

Answer the user's question using ONLY the evidence provided below.

Rules:
1. Do not invent medical facts.
2. Do not diagnose the user.
3. Do not prescribe medication or dosage.
4. If the evidence does not contain enough information, say:
   "I don't have enough information in my trusted sources to answer that reliably."
5. Keep the answer concise and easy to understand.
6. Do not mention internal reasoning.
7. Do not repeat the evidence unnecessarily.

USER QUESTION:
{query}

TRUSTED EVIDENCE:
{evidence}

ANSWER:
"""

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "temperature": 0.2,
        "num_predict": MAX_GENERATION_TOKENS,
        "think": False,
        "keep_alive": 0,
    }

    try:

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=15,
        )

        response.raise_for_status()

        data = response.json()

        answer = data.get(
            "response",
            ""
        ).strip()

        return answer

    except Exception as e:

        print(
            f"LLM generation unavailable: {e}"
        )

        return ""


# ============================================================
# ANSWER VALIDATION
# ============================================================

def is_good_answer(answer):

    if not answer:
        return False

    lowered = answer.lower()

    bad_patterns = [
        "i need to answer",
        "let me review",
        "hmm,",
        "the user is asking",
        "we are given",
        "i should",
        "i need to",
        "reasoning",
        "chain of thought",
    ]

    for pattern in bad_patterns:

        if pattern in lowered:
            return False

    if len(answer.strip()) < 20:
        return False

    return True


# ============================================================
# EXTRACTIVE FALLBACK
# ============================================================

def extractive_answer(query, documents):
    """
    Grounded fallback answer.

    Selects the most relevant complete sentences from retrieved
    evidence while filtering PDF metadata, headers, URLs,
    table-of-contents fragments, and other noisy text.
    """

    if not documents:
        return (
            "I don't have enough information in my trusted "
            "sources to answer that reliably."
        )

    query_words = set(
        re.findall(
            r"\b[a-zA-Z]{4,}\b",
            query.lower()
        )
    )

    # --------------------------------------------------------
    # Words/fragments that usually indicate PDF noise
    # --------------------------------------------------------

    noise_patterns = [
        "return to overview page",
        "page ",
        "https://",
        "www.",
        "table of contents",
        "contents",
        "chapter",
        "references",
        "copyright",
        "all content",
        "prepared using",
        "figure ",
        "table ",
        "---",
    ]

    candidates = []

    # --------------------------------------------------------
    # Extract sentences from retrieved documents
    # --------------------------------------------------------

    for document in documents:

        text = clean_text(
            document.get(
                "text",
                ""
            )
        )

        if not text:
            continue

        # Remove common PDF page artifacts
        text = re.sub(
            r"\b\d+\s*/\s*\d+\b",
            " ",
            text
        )

        text = re.sub(
            r"\bPage\s+\d+\b",
            " ",
            text,
            flags=re.IGNORECASE
        )

        text = re.sub(
            r"https?://\S+",
            " ",
            text
        )

        text = clean_text(text)

        sentences = re.split(
            r"(?<=[.!?])\s+",
            text
        )

        for sentence in sentences:

            sentence = clean_text(
                sentence
            )

            # ------------------------------------------------
            # Basic quality filtering
            # ------------------------------------------------

            if len(sentence) < 50:
                continue

            if len(sentence) > 500:
                continue

            lowered = sentence.lower()

            # Skip PDF noise
            if any(
                pattern in lowered
                for pattern in noise_patterns
            ):
                continue

            # Skip sentences that are mostly metadata
            if lowered.count("|") >= 2:
                continue

            # Skip obvious headings
            if sentence.endswith(":"):
                continue

            # ------------------------------------------------
            # Query overlap
            # ------------------------------------------------

            sentence_words = set(
                re.findall(
                    r"\b[a-zA-Z]{4,}\b",
                    lowered
                )
            )

            overlap = len(
                query_words.intersection(
                    sentence_words
                )
            )

            # ------------------------------------------------
            # Prefer informative sentences
            # ------------------------------------------------

            informative_terms = [
                "diabetes",
                "blood glucose",
                "blood sugar",
                "symptom",
                "type 1",
                "type 2",
                "gestational",
                "insulin",
                "health",
                "risk",
                "treatment",
                "prevent",
            ]

            information_score = sum(
                1
                for term in informative_terms
                if term in lowered
            )

            # Penalize suspiciously short fragments
            if len(sentence.split()) < 8:
                continue

            score = (
                overlap * 3
                + information_score * 2
            )

            candidates.append(
                (
                    score,
                    len(sentence),
                    sentence
                )
            )

    # --------------------------------------------------------
    # No usable sentence
    # --------------------------------------------------------

    if not candidates:

        return (
            "I don't have enough information in my trusted "
            "sources to answer that reliably."
        )

    # --------------------------------------------------------
    # Rank candidates
    # --------------------------------------------------------

    candidates.sort(
        key=lambda x: (
            x[0],
            -abs(x[1] - 180)
        ),
        reverse=True
    )

    # --------------------------------------------------------
    # Select non-duplicate sentences
    # --------------------------------------------------------

    selected = []

    seen = set()

    for score, length, sentence in candidates:

        normalized = re.sub(
            r"[^a-z0-9 ]",
            "",
            sentence.lower()
        )

        if normalized in seen:
            continue

        seen.add(normalized)

        selected.append(sentence)

        # Keep fallback concise
        if len(selected) >= 2:
            break

    # --------------------------------------------------------
    # Return grounded response
    # --------------------------------------------------------

    return " ".join(selected)


# ============================================================
# DISPLAY SOURCES
# ============================================================

def display_sources(documents):

    print("\nSOURCES USED")

    for i, document in enumerate(
        documents,
        start=1
    ):

        source = document.get(
            "source",
            "Unknown"
        )

        chunk_id = document.get(
            "chunk_id",
            "Unknown"
        )

        print(
            f"{i}. {source} "
            f"(Chunk {chunk_id})"
        )


# ============================================================
# RAG QUESTION
# ============================================================

def answer_question(query):

    safety_result = check_safety(query)

    if not safety_result["safe"]:

        print("\nSAFETY RESPONSE")
        print("-" * 60)

        print(
            safety_result["message"]
        )

        return

    try:

        # ----------------------------------------------------
        # Retrieve trusted evidence
        # ----------------------------------------------------

        retrieved = retrieve(
            query,
            RETRIEVE_TOP_K
        )

        if not retrieved:

            print(
                "\nI couldn't find relevant "
                "information in the trusted sources."
            )

            return

        # ----------------------------------------------------
        # Rerank evidence
        # ----------------------------------------------------

        reranked = rerank(
            query,
            retrieved,
            RERANK_TOP_K
        )

        # ----------------------------------------------------
        # Build evidence
        # ----------------------------------------------------

        evidence = build_evidence(
            reranked
        )

        # ----------------------------------------------------
        # Load latest risk context
        # ----------------------------------------------------

        risk_context = load_risk_context()

        # ----------------------------------------------------
        # Add risk context if available
        # ----------------------------------------------------

        if risk_context:

            risk_information = f"""
MODEL-BASED USER RISK CONTEXT

Risk probability:
{risk_context.get("risk_percentage", "Unknown")}%

Risk level:
{risk_context.get("risk_level", "Unknown")}

Important:
This is a machine-learning risk assessment,
NOT a medical diagnosis.

Top SHAP contributing factors:
"""

            for factor in risk_context.get(
                "shap_factors",
                []
            ):

                risk_information += (
                    f"\n- {factor['feature']}: "
                    f"{factor['shap_value']:.4f}"
                )

            evidence = (
                risk_information
                + "\n\nTRUSTED DIABETES EVIDENCE\n"
                + evidence
            )

        # ----------------------------------------------------
        # Generate answer
        # ----------------------------------------------------

        answer = generate_answer(
            query,
            evidence
        )

        # ----------------------------------------------------
        # Validate generated answer
        # ----------------------------------------------------

        if is_good_answer(answer):

            print("\nFINAL ANSWER")
            print("-" * 60)

            print(answer)

            # Show risk context when available
            if risk_context:

                print("\nRISK CONTEXT")
                print("-" * 60)

                print(
                    f"Model risk: "
                    f"{risk_context.get('risk_percentage')}%"
                )

                print(
                    f"Risk level: "
                    f"{risk_context.get('risk_level')}"
                )

                print(
                    "\nThis risk estimate is "
                    "not a medical diagnosis."
                )

            display_sources(
                reranked
            )

            print(
                "\nAnswer mode: "
                "LLM, evidence-grounded"
            )

        else:

            # ------------------------------------------------
            # Extractive fallback
            # ------------------------------------------------

            print(
                "\n[4/4] Using grounded "
                "extractive fallback..."
            )

            answer = extractive_answer(
                query,
                reranked
            )

            print("\nFINAL ANSWER")
            print("-" * 60)

            print(answer)

            if risk_context:

                print("\nRISK CONTEXT")
                print("-" * 60)

                print(
                    f"Model risk: "
                    f"{risk_context.get('risk_percentage')}%"
                )

                print(
                    f"Risk level: "
                    f"{risk_context.get('risk_level')}"
                )

                print(
                    "\nThis risk estimate is "
                    "not a medical diagnosis."
                )

            display_sources(
                reranked
            )

            print(
                "\nAnswer mode: "
                "Extractive, evidence-grounded fallback"
            )

    except Exception as e:

        print("\nRAG ERROR")
        print("-" * 60)

        print(
            str(e)
        )


# ============================================================
# SHAP EXPLANATION
# ============================================================

def explain_risk_with_shap(
    gender,
    age,
    hypertension,
    heart_disease,
    smoking_history,
    bmi,
    HbA1c_level,
    blood_glucose_level,
):
    """
    Generate a SHAP explanation for the XGBoost model.

    The saved model is a scikit-learn Pipeline containing
    preprocessing followed by the XGBoost estimator.

    SHAP TreeExplainer is therefore applied only to the
    underlying XGBoost estimator.
    """

    print("\nGenerating SHAP explanation...")

    # --------------------------------------------------------
    # Load the trained pipeline
    # --------------------------------------------------------

    model_pipeline = load_model()

    # --------------------------------------------------------
    # Create the user's input
    # --------------------------------------------------------

    input_df = create_input_dataframe(
        gender=gender,
        age=age,
        hypertension=hypertension,
        heart_disease=heart_disease,
        smoking_history=smoking_history,
        bmi=bmi,
        HbA1c_level=HbA1c_level,
        blood_glucose_level=blood_glucose_level,
    )

    # --------------------------------------------------------
    # Extract preprocessing + XGBoost estimator
    # --------------------------------------------------------

    if hasattr(model_pipeline, "steps"):

        # Everything except the final estimator
        preprocessor = model_pipeline[:-1]

        # Final XGBoost estimator
        xgb_model = model_pipeline.steps[-1][1]

        # Transform user's input exactly as the pipeline does
        processed_input = preprocessor.transform(
            input_df
        )

    else:

        # Fallback in case the saved object is not a Pipeline
        xgb_model = model_pipeline

        processed_input = preprocess_for_model(
            input_df
        )

    # --------------------------------------------------------
    # Convert sparse output if necessary
    # --------------------------------------------------------

    if hasattr(
        processed_input,
        "toarray"
    ):
        processed_input = processed_input.toarray()

    # --------------------------------------------------------
    # Get feature names
    # --------------------------------------------------------

    try:

        feature_names = list(
            preprocessor.get_feature_names_out()
        )

    except Exception:

        feature_names = [
            f"Feature_{i}"
            for i in range(
                processed_input.shape[1]
            )
        ]

    # --------------------------------------------------------
    # SHAP TreeExplainer
    # --------------------------------------------------------

    explainer = shap.TreeExplainer(
        xgb_model
    )

    shap_values = explainer.shap_values(
        processed_input
    )

    # --------------------------------------------------------
    # Handle SHAP output formats
    # --------------------------------------------------------

    if isinstance(
        shap_values,
        list
    ):

        values = np.asarray(
            shap_values[-1]
        )[0]

    else:

        values = np.asarray(
            shap_values
        )

        if values.ndim == 3:

            values = values[0, :, -1]

        elif values.ndim == 2:

            values = values[0]

        else:

            values = values.flatten()

    # --------------------------------------------------------
    # Match feature names and SHAP values
    # --------------------------------------------------------

    if len(feature_names) != len(values):

        feature_names = [
            f"Feature_{i}"
            for i in range(
                len(values)
            )
        ]

    contributions = list(
        zip(
            feature_names,
            values
        )
    )

    # Sort by absolute contribution
    contributions.sort(
        key=lambda x: abs(x[1]),
        reverse=True
    )

    # Top 5 factors
    top_factors = contributions[:5]

    # --------------------------------------------------------
    # Clean memory
    # --------------------------------------------------------

    del explainer
    del model_pipeline
    del processed_input

    gc.collect()

    return top_factors


# ============================================================
# RISK ASSESSMENT
# ============================================================

def run_risk_assessment():

    print("\n")
    print("=" * 60)
    print("DIABETES RISK ASSESSMENT")
    print("=" * 60)

    print(
        "\nThis assessment estimates model-based diabetes risk."
    )

    print(
        "It is NOT a medical diagnosis."
    )

    print("\nEnter the following information.")

    # --------------------------------------------------------
    # Gender
    # --------------------------------------------------------

    gender = input(
        "\nGender "
        "(Female/Male/Other): "
    ).strip()

    if gender not in [
        "Female",
        "Male",
        "Other"
    ]:

        gender = "Other"

    # --------------------------------------------------------
    # Age
    # --------------------------------------------------------

    while True:

        try:

            age = float(
                input("Age: ")
            )

            if age <= 0:

                print(
                    "Please enter a valid age."
                )
                continue

            break

        except ValueError:

            print(
                "Please enter a number."
            )

    # --------------------------------------------------------
    # Hypertension
    # --------------------------------------------------------

    hypertension_input = input(
        "Hypertension? (yes/no): "
    ).strip().lower()

    hypertension = (
        1
        if hypertension_input in [
            "yes",
            "y",
            "1"
        ]
        else 0
    )

    # --------------------------------------------------------
    # Heart disease
    # --------------------------------------------------------

    heart_input = input(
        "Heart disease? (yes/no): "
    ).strip().lower()

    heart_disease = (
        1
        if heart_input in [
            "yes",
            "y",
            "1"
        ]
        else 0
    )

    # --------------------------------------------------------
    # Smoking history
    # --------------------------------------------------------

    smoking_history = input(
        "Smoking history "
        "(never/current/former/ever/not current/No Info): "
    ).strip()

    allowed_smoking = [
        "never",
        "current",
        "former",
        "ever",
        "not current",
        "No Info"
    ]

    if smoking_history not in allowed_smoking:

        smoking_history = "No Info"

    # --------------------------------------------------------
    # BMI
    # --------------------------------------------------------

    while True:

        try:

            bmi = float(
                input("BMI: ")
            )

            if bmi <= 0:

                print(
                    "Please enter a valid BMI."
                )
                continue

            break

        except ValueError:

            print(
                "Please enter a number."
            )

    # --------------------------------------------------------
    # HbA1c
    # --------------------------------------------------------

    while True:

        try:

            HbA1c_level = float(
                input("HbA1c level: ")
            )

            if HbA1c_level <= 0:

                print(
                    "Please enter a valid HbA1c value."
                )
                continue

            break

        except ValueError:

            print(
                "Please enter a number."
            )

    # --------------------------------------------------------
    # Blood glucose
    # --------------------------------------------------------

    while True:

        try:

            blood_glucose_level = float(
                input("Blood glucose level: ")
            )

            if blood_glucose_level <= 0:

                print(
                    "Please enter a valid blood glucose value."
                )
                continue

            break

        except ValueError:

            print(
                "Please enter a number."
            )

            continue

    # ========================================================
    # XGBOOST
    # ========================================================

    print("\nRunning XGBoost risk model...")

    try:

        result = predict_diabetes_risk(
            gender=gender,
            age=age,
            hypertension=hypertension,
            heart_disease=heart_disease,
            smoking_history=smoking_history,
            bmi=bmi,
            hba1c_level=HbA1c_level,
            blood_glucose_level=blood_glucose_level
    )

    except ValueError as e:

        print("\n")
        print("=" * 60)
        print("INPUT VALIDATION ERROR")
        print("=" * 60)

        print(str(e))

        print("=" * 60)
        print("Please enter valid values and try /risk again.")
        print("=" * 60)

        return

    # ========================================================
    # SHAP
    # ========================================================

    top_factors = explain_risk_with_shap(
        gender=gender,
        age=age,
        hypertension=hypertension,
        heart_disease=heart_disease,
        smoking_history=smoking_history,
        bmi=bmi,
        HbA1c_level=HbA1c_level,
        blood_glucose_level=blood_glucose_level,
    )

    save_risk_context(
        risk_result=result,
        shap_factors=top_factors
    )

    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    print("\n")
    print("=" * 60)
    print("RISK ASSESSMENT RESULT")
    print("=" * 60)

    print(
        f"\nRisk probability: "
        f"{result['risk_probability']}"
    )

    print(
        f"Risk percentage: "
        f"{result['risk_percentage']}%"
    )

    print(
        f"Risk level: "
        f"{result['risk_level']}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "This is a machine-learning risk assessment "
        "and not a medical diagnosis."
    )

    # ========================================================
    # SHAP FACTORS
    # ========================================================

    print("\n")
    print("=" * 60)
    print("TOP CONTRIBUTING FACTORS")
    print("=" * 60)

    feature_labels = {
        "categorical__gender_Female":
            "Gender: Female",

        "categorical__gender_Male":
            "Gender: Male",

        "categorical__gender_Other":
            "Gender: Other",

        "categorical__smoking_history_No Info":
            "Smoking history: No Info",

        "categorical__smoking_history_current":
            "Smoking history: Current",

        "categorical__smoking_history_ever":
            "Smoking history: Ever",

        "categorical__smoking_history_former":
            "Smoking history: Former",

        "categorical__smoking_history_never":
            "Smoking history: Never",

        "categorical__smoking_history_not current":
            "Smoking history: Not current",

        "numerical__age":
            "Age",

        "numerical__hypertension":
            "Hypertension",

        "numerical__heart_disease":
            "Heart disease",

        "numerical__bmi":
            "BMI",

        "numerical__HbA1c_level":
            "HbA1c level",

        "numerical__blood_glucose_level":
            "Blood glucose level",
    }

    for feature, value in top_factors:

        label = feature_labels.get(
            feature,
            feature
        )

        direction = (
            "increased"
            if value > 0
            else "decreased"
        )

        print(
            f"- {label}: "
            f"{direction} model risk "
            f"(SHAP = {value:.4f})"
        )

    print(
        "\nSHAP explains which input features "
        "contributed most to this model prediction."
    )

    print("=" * 60)

    return result


# ============================================================
# HELP
# ============================================================

def print_help():

    print("\nCOMMANDS")
    print("-" * 60)

    print(
        "/risk"
        "   Run diabetes risk assessment + SHAP explanation"
    )

    print(
        "/help"
        "   Show available commands"
    )

    print(
        "exit"
        "   Quit the assistant"
    )

    print(
        "\nFor normal questions, simply type your "
        "diabetes-related question."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("DIABETES RAG ASSISTANT")
    print("=" * 60)

    print(
        "Type '/risk' for risk assessment + SHAP."
    )

    print(
        "Type '/help' for commands."
    )

    print(
        "Type 'exit' to quit."
    )

    print("=" * 60)

    while True:

        try:

            user_query = input(
                "\nAsk a diabetes question: "
            ).strip()

        except (
            KeyboardInterrupt,
            EOFError
        ):

            print(
                "\nGoodbye!"
            )

            break

        if not user_query:
            continue

        lowered = user_query.lower()

        # ----------------------------------------------------
        # Exit
        # ----------------------------------------------------

        if lowered in [
            "exit",
            "quit",
            "bye"
        ]:

            print(
                "\nGoodbye!"
            )

            break

        # ----------------------------------------------------
        # Help
        # ----------------------------------------------------

        if lowered == "/reset":

            clear_risk_context()

            print(
                "\nRisk context cleared."
            )

            continue

        if lowered == "/help":

            print_help(
               "/reset"
               "  Clear the latest risk assessment" 
            )

            continue

        # ----------------------------------------------------
        # Risk assessment
        # ----------------------------------------------------

        if lowered in [
            "/risk",
            "risk assessment",
            "diabetes risk assessment"
        ]:

            try:

                run_risk_assessment()

            except Exception as e:

                print(
                    "\nRisk assessment error:"
                )

                print(
                    str(e)
                )

            continue

        # ----------------------------------------------------
        # Normal RAG question
        # ----------------------------------------------------

        answer_question(
            user_query
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()