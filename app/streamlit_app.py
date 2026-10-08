import os
import sys

import streamlit as st
import pandas as pd
import numpy as np


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.prediction.risk_predictor import (
    predict_diabetes_risk
)

from src.rag.rag_answer import (
    retrieve,
    rerank,
    generate_answer,
    is_good_answer,
    extractive_answer,
    clean_text,
    explain_risk_with_shap
)

from src.rag.safety import (
    check_safety
)

from src.prediction.risk_context import (
    save_risk_context,
    load_risk_context,
    clear_risk_context
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Diabetes AI Assistant",
    page_icon="🩺",
    layout="wide"
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "risk_result" not in st.session_state:
    st.session_state["risk_result"] = None

if "shap_factors" not in st.session_state:
    st.session_state["shap_factors"] = []


# ============================================================
# HEADER
# ============================================================

st.title("🩺 Diabetes AI Assistant")

st.markdown(
    """
### Retrieval-Augmented Conversational Assistant for Diabetes

This research prototype combines:

- **Machine Learning-based diabetes risk assessment**
- **SHAP-based explainability**
- **Retrieval-Augmented Generation (RAG)**
- **Cross-encoder evidence reranking**
- **Safety-aware conversational responses**
- **Trusted diabetes knowledge sources**
"""
)

st.warning(
    "This system provides educational information and "
    "model-based risk assessment. It is NOT a medical "
    "diagnosis and should not replace professional medical care."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("About the System")

    st.markdown(
        """
        **Phase 1 — Diabetes Intelligence**

        XGBoost + SHAP for model-based risk assessment.

        **Phase 2 — Conversational Assistant**

        FAISS retrieval + cross-encoder reranking +
        evidence-grounded responses.

        **Knowledge Sources**

        - CDC
        - NIDDK
        """
    )

    st.divider()

    if st.button(
        "Reset Risk Context",
        use_container_width=True
    ):

        clear_risk_context()

        st.session_state["risk_result"] = None
        st.session_state["shap_factors"] = []

        st.success(
            "Risk context cleared."
        )


# ============================================================
# TABS
# ============================================================

chat_tab, risk_tab, evaluation_tab = st.tabs(
    [
        "💬 Diabetes Assistant",
        "📊 Risk Assessment",
        "📈 Evaluation"
    ]
)


# ============================================================
# CHAT TAB
# ============================================================

with chat_tab:

    st.subheader(
        "Ask a Diabetes Question"
    )

    # --------------------------------------------------------
    # Previous conversation
    # --------------------------------------------------------

    for message in st.session_state["messages"]:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

            if message.get("sources"):

                with st.expander(
                    "📚 Sources used"
                ):

                    for source in message["sources"]:

                        st.write(
                            f"- {source}"
                        )

    # --------------------------------------------------------
    # User question
    # --------------------------------------------------------

    user_question = st.chat_input(
        "Example: What are common symptoms of diabetes?"
    )

    if user_question:

        st.session_state["messages"].append(
            {
                "role": "user",
                "content": user_question
            }
        )

        with st.chat_message(
            "user"
        ):

            st.markdown(
                user_question
            )

        # ----------------------------------------------------
        # SAFETY CHECK
        # ----------------------------------------------------

        safety_result = check_safety(
            user_question
        )

        if not safety_result["safe"]:

            response = safety_result["message"]

            st.session_state["messages"].append(
                {
                    "role": "assistant",
                    "content": response
                }
            )

            with st.chat_message(
                "assistant"
            ):

                st.warning(
                    response
                )

        else:

            with st.chat_message(
                "assistant"
            ):

                # ------------------------------------------------
                # RETRIEVAL
                # ------------------------------------------------

                with st.spinner(
                    "Retrieving trusted evidence..."
                ):

                    documents = retrieve(
                        user_question
                    )

                # ------------------------------------------------
                # RERANKING
                # ------------------------------------------------

                with st.spinner(
                    "Reranking evidence..."
                ):

                    ranked_documents = rerank(
                        user_question,
                        documents
                    )

                # ------------------------------------------------
                # BUILD EVIDENCE
                # ------------------------------------------------

                evidence_parts = []

                for document in ranked_documents:

                    text = clean_text(
                        document.get(
                            "text",
                            ""
                        )
                    )

                    if text:

                        evidence_parts.append(
                            text
                        )

                evidence = "\n\n".join(
                    evidence_parts
                )

                # ------------------------------------------------
                # LOAD RISK CONTEXT
                # ------------------------------------------------

                risk_context = load_risk_context()

                if risk_context:

                    evidence += (
                        "\n\nMODEL RISK CONTEXT:\n"
                        f"Risk level: "
                        f"{risk_context.get('risk_level', 'Unknown')}\n"
                        f"Risk percentage: "
                        f"{risk_context.get('risk_percentage', 0):.2f}%"
                    )

                # ------------------------------------------------
                # GENERATE ANSWER
                # ------------------------------------------------

                with st.spinner(
                    "Generating grounded answer..."
                ):

                    answer = generate_answer(
                        user_question,
                        evidence
                    )

                # ------------------------------------------------
                # FALLBACK
                # ------------------------------------------------

                if not is_good_answer(
                    answer
                ):

                    answer = extractive_answer(
                        user_question,
                        ranked_documents
                    )

                    answer_mode = (
                        "Evidence-grounded extractive response"
                    )

                else:

                    answer_mode = (
                        "LLM-generated grounded response"
                    )

                # ------------------------------------------------
                # DISPLAY ANSWER
                # ------------------------------------------------

                st.markdown(
                    answer
                )

                st.caption(
                    f"Answer mode: {answer_mode}"
                )

                # ------------------------------------------------
                # DISPLAY RISK CONTEXT
                # ------------------------------------------------

                if risk_context:

                    st.info(
                        f"Current model risk context: "
                        f"{risk_context.get('risk_percentage', 0):.2f}% "
                        f"({risk_context.get('risk_level', 'Unknown')}). "
                        "This is not a diagnosis."
                    )

                # ------------------------------------------------
                # SOURCES
                # ------------------------------------------------

                source_names = []

                for document in ranked_documents:

                    source = (
                        f"{document.get('source', 'Unknown source')} "
                        f"(Chunk {document.get('chunk_id', 'Unknown')})"
                    )

                    source_names.append(
                        source
                    )

                if source_names:

                    with st.expander(
                        "📚 Sources used"
                    ):

                        for source in source_names:

                            st.write(
                                f"- {source}"
                            )

                # ------------------------------------------------
                # SAVE MESSAGE
                # ------------------------------------------------

                st.session_state["messages"].append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": source_names
                    }
                )


# ============================================================
# RISK ASSESSMENT TAB
# ============================================================

with risk_tab:

    st.subheader(
        "Diabetes Risk Assessment"
    )

    st.info(
        "This module estimates model-based diabetes risk. "
        "It does not diagnose diabetes."
    )

    # --------------------------------------------------------
    # INPUTS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Gender",
            [
                "Female",
                "Male",
                "Other"
            ]
        )

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=120,
            value=45,
            step=1
        )

        hypertension = st.selectbox(
            "Hypertension",
            [
                "No",
                "Yes"
            ]
        )

        heart_disease = st.selectbox(
            "Heart disease",
            [
                "No",
                "Yes"
            ]
        )

    with col2:

        smoking_history = st.selectbox(
            "Smoking history",
            [
                "never",
                "current",
                "former",
                "ever",
                "not current",
                "No Info"
            ]
        )

        bmi = st.number_input(
            "BMI",
            min_value=10.0,
            max_value=80.0,
            value=25.0,
            step=0.1
        )

        hba1c_level = st.number_input(
            "HbA1c level",
            min_value=3.0,
            max_value=20.0,
            value=5.8,
            step=0.1
        )

        blood_glucose_level = st.number_input(
            "Blood glucose level",
            min_value=40.0,
            max_value=600.0,
            value=120.0,
            step=1.0
        )

    # --------------------------------------------------------
    # RUN RISK ASSESSMENT
    # --------------------------------------------------------

    if st.button(
        "🔍 Assess Diabetes Risk",
        type="primary",
        use_container_width=True
    ):

        try:

            hypertension_value = (
                1
                if hypertension == "Yes"
                else 0
            )

            heart_disease_value = (
                1
                if heart_disease == "Yes"
                else 0
            )

            # ------------------------------------------------
            # XGBoost prediction
            # ------------------------------------------------

            with st.spinner(
                "Running XGBoost risk assessment..."
            ):

                result = predict_diabetes_risk(
                    gender=gender,
                    age=age,
                    hypertension=hypertension_value,
                    heart_disease=heart_disease_value,
                    smoking_history=smoking_history,
                    bmi=bmi,
                    hba1c_level=hba1c_level,
                    blood_glucose_level=blood_glucose_level
                )

            # ------------------------------------------------
            # SHAP explanation
            # ------------------------------------------------

            with st.spinner(
                "Generating SHAP explanation..."
            ):

                shap_factors = explain_risk_with_shap(
                    gender,
                    age,
                    hypertension_value,
                    heart_disease_value,
                    smoking_history,
                    bmi,
                    hba1c_level,
                    blood_glucose_level
                )

            # ------------------------------------------------
            # Save results
            # ------------------------------------------------

            st.session_state["risk_result"] = result

            st.session_state["shap_factors"] = (
                shap_factors
            )

            save_risk_context(
                result,
                shap_factors
            )

            st.success(
                "Risk assessment completed successfully."
            )

        except ValueError as e:

            st.error(
                str(e)
            )

        except Exception as e:

            st.error(
                f"Unable to complete assessment: {e}"
            )

    # --------------------------------------------------------
    # DISPLAY RISK RESULT
    # --------------------------------------------------------

    if st.session_state["risk_result"]:

        result = st.session_state[
            "risk_result"
        ]

        st.divider()

        st.subheader(
            "Risk Assessment Result"
        )

        metric1, metric2, metric3 = st.columns(3)

        with metric1:

            st.metric(
                "Risk Probability",
                f"{result['risk_probability']:.4f}"
            )

        with metric2:

            st.metric(
                "Risk Percentage",
                f"{result['risk_percentage']:.2f}%"
            )

        with metric3:

            st.metric(
                "Risk Level",
                result["risk_level"]
            )

        st.warning(
            "This machine-learning risk estimate is "
            "not a medical diagnosis."
        )

        # ----------------------------------------------------
        # SHAP SECTION
        # ----------------------------------------------------

        st.subheader(
            "SHAP Explanation"
        )

        shap_factors = st.session_state.get(
            "shap_factors",
            []
        )

        if shap_factors:

            shap_rows = []

            for factor in shap_factors:

                feature = "Unknown"
                shap_value = 0.0
                direction = "Unknown"

                # --------------------------------------------
                # Tuple / list format
                # --------------------------------------------

                if isinstance(
                    factor,
                    (tuple, list)
                ):

                    if len(factor) >= 2:

                        feature = str(
                            factor[0]
                        )

                        try:

                            shap_value = float(
                                factor[1]
                            )

                        except (
                            ValueError,
                            TypeError
                        ):

                            shap_value = 0.0

                        if shap_value > 0:

                            direction = (
                                "Increased model risk"
                            )

                        elif shap_value < 0:

                            direction = (
                                "Decreased model risk"
                            )

                        else:

                            direction = (
                                "No significant contribution"
                            )

                # --------------------------------------------
                # Dictionary format
                # --------------------------------------------

                elif isinstance(
                    factor,
                    dict
                ):

                    feature = str(
                        factor.get(
                            "feature",
                            "Unknown"
                        )
                    )

                    try:

                        shap_value = float(
                            factor.get(
                                "shap_value",
                                0.0
                            )
                        )

                    except (
                        ValueError,
                        TypeError
                    ):

                        shap_value = 0.0

                    direction = str(
                        factor.get(
                            "direction",
                            "Unknown"
                        )
                    )

                # --------------------------------------------
                # Ignore invalid format
                # --------------------------------------------

                else:

                    continue

                shap_rows.append(
                    {
                        "Feature": feature,
                        "Contribution": direction,
                        "SHAP Value": round(
                            shap_value,
                            4
                        )
                    }
                )

            # ------------------------------------------------
            # Display table
            # ------------------------------------------------

            if shap_rows:

                dataframe = pd.DataFrame(
                    shap_rows
                )

                st.dataframe(
                    dataframe,
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.info(
                    "No SHAP explanation was available."
                )

        else:

            st.info(
                "Run the risk assessment to generate "
                "the SHAP explanation."
            )


# ============================================================
# EVALUATION TAB
# ============================================================

with evaluation_tab:

    st.subheader(
        "System Evaluation"
    )

    st.markdown(
        """
        The retrieval system was evaluated using
        12 diabetes-domain questions.
        """
    )

    evaluation_file = os.path.join(
        BASE_DIR,
        "evaluation",
        "ablation_results.csv"
    )

    if os.path.exists(
        evaluation_file
    ):

        dataframe = pd.read_csv(
            evaluation_file
        )

        st.dataframe(
            dataframe,
            use_container_width=True,
            hide_index=True
        )

        st.subheader(
            "Ablation Results"
        )

        if len(dataframe) >= 2:

            baseline = dataframe.iloc[0]
            reranked = dataframe.iloc[1]

            metric1, metric2, metric3 = st.columns(3)

            with metric1:

                st.metric(
                    "Top-1 Overlap",
                    f"{reranked['average_top1_overlap']:.4f}",
                    f"{(
                        reranked['average_top1_overlap']
                        - baseline['average_top1_overlap']
                    ):.4f}"
                )

            with metric2:

                st.metric(
                    "Document Overlap",
                    f"{reranked['average_document_overlap']:.4f}",
                    f"{(
                        reranked['average_document_overlap']
                        - baseline['average_document_overlap']
                    ):.4f}"
                )

            with metric3:

                st.metric(
                    "MRR",
                    f"{reranked['mrr']:.4f}",
                    f"{(
                        reranked['mrr']
                        - baseline['mrr']
                    ):.4f}"
                )

            st.success(
                "Cross-encoder reranking improved "
                "retrieval ranking compared with "
                "semantic retrieval alone."
            )

    else:

        st.info(
            "Run the ablation study to generate "
            "evaluation results."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Research prototype — Retrieval-Augmented "
    "Conversational Assistant for Diabetes"
)