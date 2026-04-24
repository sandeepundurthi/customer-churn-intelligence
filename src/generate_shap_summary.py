import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

# Load model
model = joblib.load("models/churn_model.pkl")

preprocessor = model.named_steps["preprocessor"]
xgb_model = model.named_steps["model"]

# Load RAW data
df = pd.read_csv("data/telco_churn.csv")

# Clean data (same as training)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Feature engineering (same as training)
df["AvgMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)

def tenure_group(t):
    if t <= 12:
        return "0-1 year"
    elif t <= 24:
        return "1-2 years"
    elif t <= 48:
        return "2-4 years"
    else:
        return "4-6 years"

df["TenureGroup"] = df["tenure"].apply(tenure_group)

# Split features
X = df.drop(["Churn", "customerID"], axis=1)

# Transform using pipeline
X_transformed = preprocessor.transform(X)
feature_names = preprocessor.get_feature_names_out()

X_df = pd.DataFrame(X_transformed, columns=feature_names)

# SHAP
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_df)

# Plot
plt.figure()
shap.summary_plot(shap_values, X_df, show=False)

# Save
plt.savefig("reports/shap_summary.png", bbox_inches="tight")
plt.close()

print("✅ SHAP summary saved to reports/shap_summary.png")
