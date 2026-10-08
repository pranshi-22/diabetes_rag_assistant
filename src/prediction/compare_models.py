import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# 1. Load dataset
df = pd.read_csv("data/raw/diabetes_prediction_dataset.csv")


# 2. Separate features and target
X = df.drop("diabetes", axis=1)
y = df["diabetes"]


# 3. Split data exactly as before
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# 4. Load saved models
logistic_model = joblib.load("models/logistic_regression.pkl")
random_forest_model = joblib.load("models/random_forest.pkl")
xgboost_model = joblib.load("models/xgboost.pkl")


# 5. Store models
models = {
    "Logistic Regression": logistic_model,
    "Random Forest": random_forest_model,
    "XGBoost": xgboost_model
}


# 6. Evaluate every model
results = []

for name, model in models.items():

    print(f"\nEvaluating {name}...")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    })


# 7. Create comparison table
results_df = pd.DataFrame(results)


# 8. Display results
print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# 9. Find best model using F1 Score
best_model = results_df.loc[
    results_df["F1 Score"].idxmax()
]

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print(f"Model: {best_model['Model']}")
print(f"F1 Score: {best_model['F1 Score']:.4f}")
print(f"ROC-AUC: {best_model['ROC-AUC']:.4f}")