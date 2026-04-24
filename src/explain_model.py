import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from data_preprocessing import preprocess_data


DATA_PATH = "data/telco_churn.csv"
MODEL_PATH = "models/churn_model.pkl"


def generate_shap_explanations():
    df = preprocess_data(DATA_PATH)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    pipeline = joblib.load(MODEL_PATH)

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    X_test_transformed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    X_test_transformed_df = pd.DataFrame(
        X_test_transformed,
        columns=feature_names
    )

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_transformed_df)

    plt.figure()
    shap.summary_plot(
        shap_values,
        X_test_transformed_df,
        show=False,
        max_display=15
    )

    plt.tight_layout()
    plt.savefig("reports/shap_summary.png", dpi=300, bbox_inches="tight")
    print("SHAP summary saved to reports/shap_summary.png")


if __name__ == "__main__":
    generate_shap_explanations()
