import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt


# ============================================================
# 1. Load dataset
# ============================================================

df = pd.read_csv("data/raw/diabetes_prediction_dataset.csv")


# ============================================================
# 2. Separate features and target
# ============================================================

X = df.drop("diabetes", axis=1)
y = df["diabetes"]


# ============================================================
# 3. Load our SAVED XGBoost pipeline
# ============================================================

pipeline = joblib.load("models/xgboost.pkl")

print("XGBoost model loaded successfully!")


# ============================================================
# 4. Get preprocessing and XGBoost model
# ============================================================

preprocessor = pipeline.named_steps["preprocessor"]
xgb_model = pipeline.named_steps["model"]


# ============================================================
# 5. Transform the data using the SAME preprocessing
# ============================================================

X_processed = preprocessor.transform(X)

feature_names = preprocessor.get_feature_names_out()


# ============================================================
# 6. Prepare data for SHAP
# ============================================================


print("Data preprocessing completed!")
print(f"Processed features: {X_processed.shape[1]}")


# ============================================================
# 7. Create SHAP Tree Explainer
# ============================================================

explainer = shap.TreeExplainer(xgb_model)

print("SHAP explainer created successfully!")


# ============================================================
# 8. Use a sample for SHAP analysis
# ============================================================

sample_size = min(1000, len(X_processed))

X_sample = X_processed[:sample_size]


# ============================================================
# 9. Calculate SHAP values
# ============================================================

shap_values = explainer.shap_values(X_sample)

print("SHAP values calculated successfully!")


# ============================================================
# 10. Global Feature Importance
# ============================================================

print("\n===== TOP FEATURES =====")

importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": abs(shap_values).mean(axis=0)
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print(importance.head(10).to_string(index=False))


# ============================================================
# 11. Create SHAP Summary Plot
# ============================================================

plt.figure()

shap.summary_plot(
    shap_values,
    X_sample,
    feature_names=feature_names,
    show=False
)

plt.tight_layout()

plt.savefig(
    "evaluation/shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nSHAP summary plot saved successfully!")


# ============================================================
# 12. Explain ONE individual prediction
# ============================================================

single_patient = X_processed[0:1]

single_shap = explainer.shap_values(single_patient)


print("\n===== INDIVIDUAL PREDICTION =====")

prediction = pipeline.predict(X.iloc[[0]])[0]
probability = pipeline.predict_proba(X.iloc[[0]])[0][1]

print(f"Predicted class: {prediction}")
print(f"Risk probability: {probability:.4f}")


# ============================================================
# 13. Display important factors for this prediction
# ============================================================

individual_importance = pd.DataFrame({
    "Feature": feature_names,
    "SHAP_Value": single_shap[0]
})

individual_importance["Absolute_SHAP"] = abs(
    individual_importance["SHAP_Value"]
)

individual_importance = individual_importance.sort_values(
    by="Absolute_SHAP",
    ascending=False
)

print("\nTop factors influencing this prediction:")

print(
    individual_importance[
        ["Feature", "SHAP_Value"]
    ].head(10).to_string(index=False)
)


print("\n===== SHAP ANALYSIS COMPLETED =====")