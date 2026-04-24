import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

from data_preprocessing import preprocess_data


DATA_PATH = "data/telco_churn.csv"
MODEL_PATH = "models/churn_model.pkl"


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

model = joblib.load(MODEL_PATH)

y_proba = model.predict_proba(X_test)[:, 1]

thresholds = [0.3, 0.35, 0.4, 0.45, 0.5]

print("\nThreshold Comparison")
print("-" * 70)

for threshold in thresholds:
    y_pred = (y_proba >= threshold).astype(int)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Threshold: {threshold}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-score:  {f1:.3f}")
    print(confusion_matrix(y_test, y_pred))
    print("-" * 70)
