# src/rag/safety.py

import re


# ============================================================
# SAFETY CATEGORIES
# ============================================================

EMERGENCY_PATTERNS = [
    r"\bunconscious\b",
    r"\bcan't breathe\b",
    r"\bcannot breathe\b",
    r"\bnot breathing\b",
    r"\bseizure\b",
    r"\bpassed out\b",
    r"\bfainted\b",
    r"\bunresponsive\b",
    r"\bsevere chest pain\b",
    r"\bsevere difficulty breathing\b",
]


DIAGNOSIS_PATTERNS = [
    r"\bdo i have diabetes\b",
    r"\bam i diabetic\b",
    r"\bdo i have type 1 diabetes\b",
    r"\bdo i have type 2 diabetes\b",
    r"\bcan you diagnose me\b",
    r"\bdiagnose me\b",
    r"\bcan you tell if i have diabetes\b",
    r"\bdo my symptoms mean( that)? i have diabetes\b",
]


MEDICATION_PATTERNS = [
    r"\bwhat medicine should i take\b",
    r"\bwhich medicine should i take\b",
    r"\bwhat medication should i take\b",
    r"\bwhich medication should i take\b",
    r"\bwhat dose should i take\b",
    r"\bwhat dosage should i take\b",
    r"\bhow much insulin should i take\b",
    r"\bhow much insulin do i need\b",
    r"\bshould i increase my insulin\b",
    r"\bshould i decrease my insulin\b",
    r"\bcan i increase my insulin\b",
    r"\bcan i decrease my insulin\b",
    r"\bshould i change my medication\b",
    r"\bcan i change my medication\b",
    r"\bwhat insulin dose should i take\b",
    r"\bshould i increase my diabetes medication\b",
    r"\bcan you prescribe medicine\b",
]


# ============================================================
# SAFETY CHECK
# ============================================================

def check_safety(query):
    """
    Classifies a user query before RAG generation.

    Returns:
        {
            "safe": True/False,
            "category": "...",
            "message": "..."
        }
    """

    query_lower = query.lower().strip()

    # --------------------------------------------------------
    # Emergency
    # --------------------------------------------------------

    for pattern in EMERGENCY_PATTERNS:

        if re.search(pattern, query_lower):

            return {
                "safe": False,
                "category": "emergency",
                "message": (
                    "This may describe a medical emergency. "
                    "Please seek immediate medical attention or "
                    "contact your local emergency service. "
                    "This assistant cannot safely assess or manage "
                    "an emergency."
                )
            }

    # --------------------------------------------------------
    # Diagnosis
    # --------------------------------------------------------

    for pattern in DIAGNOSIS_PATTERNS:

        if re.search(pattern, query_lower):

            return {
                "safe": False,
                "category": "diagnosis",
                "message": (
                    "I can provide general information about diabetes "
                    "and help explain diabetes risk factors, symptoms, "
                    "and screening information, but I cannot diagnose "
                    "you. A healthcare professional can determine "
                    "whether diabetes is present using appropriate "
                    "clinical evaluation and testing."
                )
            }

    # --------------------------------------------------------
    # Medication / dosage
    # --------------------------------------------------------

    for pattern in MEDICATION_PATTERNS:

        if re.search(pattern, query_lower):

            return {
                "safe": False,
                "category": "medication",
                "message": (
                    "I can provide general educational information "
                    "about diabetes medicines, including medication "
                    "classes, common uses, precautions, and possible "
                    "side effects. I cannot recommend a personalized "
                    "medicine, treatment change, or dosage. Please "
                    "consult a qualified healthcare professional for "
                    "individual treatment decisions."
                )
            }

    # --------------------------------------------------------
    # Normal educational query
    # --------------------------------------------------------

    return {
        "safe": True,
        "category": "educational",
        "message": ""
    }