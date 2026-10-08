import os
import sys
import gc
import joblib
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT, "models", "xgboost.pkl"
)

DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "diabetes_prediction_dataset.csv"
)


# ============================================================
# FEATURES
# ============================================================

FEATURE_COLUMNS = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "smoking_history",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
]

CATEGORICAL_FEATURES = [
    "gender",
    "smoking_history",
]

NUMERICAL_FEATURES = [
    "age",
    "hypertension",
    "heart_disease",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
]


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    """Load the trained XGBoost model."""

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"XGBoost model not found:\n{MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    return model


# ============================================================
# CREATE INPUT DATAFRAME
# ============================================================

def create_input_dataframe(
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
    Create a one-row DataFrame using the same feature names
    as the training dataset.
    """

    data = {
        "gender": [gender],
        "age": [float(age)],
        "hypertension": [int(hypertension)],
        "heart_disease": [int(heart_disease)],
        "smoking_history": [smoking_history],
        "bmi": [float(bmi)],
        "HbA1c_level": [float(HbA1c_level)],
        "blood_glucose_level": [float(blood_glucose_level)],
    }

    return pd.DataFrame(data, columns=FEATURE_COLUMNS)


# ============================================================
# PREPROCESSOR
# ============================================================

def create_preprocessor():
    """
    Create the same categorical/numerical preprocessing
    structure used during model training.
    """

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "numerical",
                "passthrough",
                NUMERICAL_FEATURES,
            ),
        ]
    )

    return preprocessor


def preprocess_for_model(input_df):
    """
    Prepare input for models that were saved without the
    preprocessing pipeline.

    The preprocessor is fitted on the original training
    dataset so the feature representation matches training.
    """

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Training dataset not found:\n{DATASET_PATH}"
        )

    training_df = pd.read_csv(DATASET_PATH)

    X_train = training_df[FEATURE_COLUMNS]

    preprocessor = create_preprocessor()

    X_train_processed = preprocessor.fit_transform(X_train)

    X_input_processed = preprocessor.transform(input_df)

    feature_names = preprocessor.get_feature_names_out()

    X_train_processed = pd.DataFrame(
        X_train_processed,
        columns=feature_names
    )

    X_input_processed = pd.DataFrame(
        X_input_processed,
        columns=feature_names
    )

    del training_df
    del X_train
    del X_train_processed
    gc.collect()

    return X_input_processed

def validate_patient_input(
    gender,
    age,
    hypertension,
    heart_disease,
    smoking_history,
    bmi,
    hba1c_level,
    blood_glucose_level
):
    """
    Validate patient inputs before risk prediction.

    These checks are input-quality checks only. They do not
    determine whether a person has diabetes.
    """

    valid_genders = {
        "female",
        "male",
        "other"
    }

    valid_smoking = {
        "never",
        "current",
        "former",
        "ever",
        "not current",
        "no info"
    }

    errors = []

    if gender.strip().lower() not in valid_genders:
        errors.append(
            "Gender must be Female, Male, or Other."
        )

    if not (18 <= float(age) <= 120):
        errors.append(
            "Age must be between 18 and 120."
        )

    if int(hypertension) not in (0, 1):
        errors.append(
            "Hypertension must be 0 or 1."
        )

    if int(heart_disease) not in (0, 1):
        errors.append(
            "Heart disease must be 0 or 1."
        )

    if smoking_history.strip().lower() not in valid_smoking:
        errors.append(
            "Invalid smoking history category."
        )

    if not (10 <= float(bmi) <= 80):
        errors.append(
            "BMI must be between 10 and 80."
        )

    if not (3 <= float(hba1c_level) <= 20):
        errors.append(
            "HbA1c level must be between 3 and 20."
        )

    if not (40 <= float(blood_glucose_level) <= 600):
        errors.append(
            "Blood glucose level must be between 40 and 600."
        )

    return errors


# ============================================================
# PREDICTION
# ============================================================

def validate_patient_input(
    gender,
    age,
    hypertension,
    heart_disease,
    smoking_history,
    bmi,
    hba1c_level,
    blood_glucose_level
):
    """
    Validate patient inputs before risk prediction.

    These checks verify that the entered values are within
    reasonable input ranges. They do not diagnose diabetes.
    """

    valid_genders = {
        "female",
        "male",
        "other"
    }

    valid_smoking = {
        "never",
        "current",
        "former",
        "ever",
        "not current",
        "no info"
    }

    errors = []

    # Gender
    if str(gender).strip().lower() not in valid_genders:
        errors.append(
            "Gender must be Female, Male, or Other."
        )

    # Age
    try:
        age_value = float(age)

        if not (18 <= age_value <= 120):
            errors.append(
                "Age must be between 18 and 120."
            )

    except (ValueError, TypeError):
        errors.append(
            "Age must be a valid number."
        )

    # Hypertension
    try:
        hypertension_value = int(hypertension)

        if hypertension_value not in (0, 1):
            errors.append(
                "Hypertension must be 0 or 1."
            )

    except (ValueError, TypeError):
        errors.append(
            "Hypertension must be 0 or 1."
        )

    # Heart disease
    try:
        heart_disease_value = int(heart_disease)

        if heart_disease_value not in (0, 1):
            errors.append(
                "Heart disease must be 0 or 1."
            )

    except (ValueError, TypeError):
        errors.append(
            "Heart disease must be 0 or 1."
        )

    # Smoking history
    if (
        str(smoking_history).strip().lower()
        not in valid_smoking
    ):
        errors.append(
            "Smoking history must be one of: "
            "never, current, former, ever, "
            "not current, No Info."
        )

    # BMI
    try:
        bmi_value = float(bmi)

        if not (10 <= bmi_value <= 80):
            errors.append(
                "BMI must be between 10 and 80."
            )

    except (ValueError, TypeError):
        errors.append(
            "BMI must be a valid number."
        )

    # HbA1c
    try:
        hba1c_value = float(hba1c_level)

        if not (3 <= hba1c_value <= 20):
            errors.append(
                "HbA1c level must be between 3 and 20."
            )

    except (ValueError, TypeError):
        errors.append(
            "HbA1c level must be a valid number."
        )

    # Blood glucose
    try:
        glucose_value = float(
            blood_glucose_level
        )

        if not (40 <= glucose_value <= 600):
            errors.append(
                "Blood glucose level must be between 40 and 600."
            )

    except (ValueError, TypeError):
        errors.append(
            "Blood glucose level must be a valid number."
        )

    return errors


def predict_diabetes_risk(
    gender,
    age,
    hypertension,
    heart_disease,
    smoking_history,
    bmi,
    hba1c_level,
    blood_glucose_level
):
    """
    Predict diabetes risk using the trained XGBoost pipeline.

    Returns:
        dict containing prediction, probability, and risk level.
    """

    # ========================================================
    # STEP 1: VALIDATE INPUT
    # ========================================================

    errors = validate_patient_input(
        gender=gender,
        age=age,
        hypertension=hypertension,
        heart_disease=heart_disease,
        smoking_history=smoking_history,
        bmi=bmi,
        hba1c_level=hba1c_level,
        blood_glucose_level=blood_glucose_level
    )

    if errors:

        raise ValueError(
            "Invalid patient input:\n- "
            + "\n- ".join(errors)
        )

    # ========================================================
    # STEP 2: LOAD TRAINED MODEL
    # ========================================================

    model = load_model()

    # ========================================================
    # STEP 3: CREATE INPUT DATAFRAME
    # ========================================================

    input_data = pd.DataFrame(
        [
            {
                "gender": str(gender).strip(),
                "age": float(age),
                "hypertension": int(hypertension),
                "heart_disease": int(heart_disease),
                "smoking_history": str(
                    smoking_history
                ).strip(),
                "bmi": float(bmi),
                "HbA1c_level": float(hba1c_level),
                "blood_glucose_level": float(
                    blood_glucose_level
                )
            }
        ]
    )

    # ========================================================
    # STEP 4: PREDICTION
    # ========================================================

    prediction = int(
        model.predict(input_data)[0]
    )

    probability = float(
        model.predict_proba(input_data)[0][1]
    )

    # ========================================================
    # STEP 5: RISK LEVEL
    # ========================================================

    if probability < 0.30:

        risk_level = "Low"

    elif probability < 0.60:

        risk_level = "Moderate"

    else:

        risk_level = "High"

    # ========================================================
    # STEP 6: RETURN RESULT
    # ========================================================

    return {
        "prediction": prediction,
        "risk_probability": probability,
        "risk_percentage": probability * 100,
        "risk_level": risk_level
    }


# ============================================================
# TEST
# ============================================================

def main():
    print("=" * 60)
    print("DIABETES RISK PREDICTOR")
    print("=" * 60)

    print("\nLoading XGBoost model...")

    try:
        result = predict_diabetes_risk(
            gender="Female",
            age=45,
            hypertension=0,
            heart_disease=0,
            smoking_history="never",
            bmi=27.5,
            HbA1c_level=5.8,
            blood_glucose_level=120,
        )

        print("\nPrediction successful!")
        print("-" * 60)

        print(
            f"Prediction: "
            f"{result['prediction']}"
        )

        print(
            f"Risk probability: "
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
            f"\n{result['message']}"
        )

        print("-" * 60)

    except Exception as e:
        print("\nERROR:")
        print(str(e))

        print("\nFull error details:")
        import traceback
        traceback.print_exc()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()