import json
import os


RISK_CONTEXT_PATH = os.path.join(
    os.path.dirname(__file__),
    "latest_risk_context.json"
)


def save_risk_context(
    risk_result,
    shap_factors
):
    """
    Save the latest risk assessment so the conversational
    assistant can use it as context.

    This is model-generated risk information and must not
    be interpreted as a medical diagnosis.
    """

    context = {
        "risk_probability": risk_result.get(
            "risk_probability"
        ),

        "risk_percentage": risk_result.get(
            "risk_percentage"
        ),

        "risk_level": risk_result.get(
            "risk_level"
        ),

        "shap_factors": [
            {
                "feature": feature,
                "shap_value": float(value)
            }
            for feature, value in shap_factors
        ],

        "disclaimer": (
            "This is a machine-learning risk assessment "
            "and not a medical diagnosis."
        )
    }

    with open(
        RISK_CONTEXT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            context,
            f,
            indent=4
        )


def load_risk_context():

    if not os.path.exists(
        RISK_CONTEXT_PATH
    ):
        return None

    try:

        with open(
            RISK_CONTEXT_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except Exception:

        return None


def clear_risk_context():

    if os.path.exists(
        RISK_CONTEXT_PATH
    ):

        os.remove(
            RISK_CONTEXT_PATH
        )