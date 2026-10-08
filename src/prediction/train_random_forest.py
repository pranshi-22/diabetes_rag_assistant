import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)


# 1. Load dataset
df = pd.read_csv("data/raw/diabetes_prediction_dataset.csv")


# 2. Separate features and target
X = df.drop("diabetes", axis=1)
y = df["diabetes"]


# 3. Define columns
categorical_features = [
    "gender",
    "smoking_history"
]

numerical_features = [
    column for column in X.columns
    if column not in categorical_features
]


# 4. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)


# 5. Random Forest
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced"
            )
        )
    ]
)


# 6. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# 7. Train
print("Training Random Forest model...")

model.fit(X_train, y_train)

print("Training completed!")

# Save trained model
os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    "models/random_forest.pkl"
)

print("Random Forest model saved successfully!")


# 8. Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


# 9. Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)


# 10. Results
print("\n===== RANDOM FOREST RESULTS =====")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")


print("\n===== CLASSIFICATION REPORT =====")

print(classification_report(y_test, y_pred))