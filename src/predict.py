import joblib
import pandas as pd


MODEL_PATH = "models/churn_model.pkl"


def load_model():
    return joblib.load(MODEL_PATH)


def predict_churn(input_data):
    model = load_model()

    input_df = pd.DataFrame([input_data])

    input_df["TotalCharges"] = pd.to_numeric(input_df["TotalCharges"], errors="coerce")
    input_df["AvgMonthlySpend"] = input_df["TotalCharges"] / (input_df["tenure"] + 1)

    if input_df["tenure"].iloc[0] <= 12:
        input_df["TenureGroup"] = "0-1 year"
    elif input_df["tenure"].iloc[0] <= 24:
        input_df["TenureGroup"] = "1-2 years"
    elif input_df["tenure"].iloc[0] <= 48:
        input_df["TenureGroup"] = "2-4 years"
    else:
        input_df["TenureGroup"] = "4-6 years"

    proba = model.predict_proba(input_df)[0][1]

    threshold = 0.30
    prediction = 1 if proba >= threshold else 0
    probability = model.predict_proba(input_df)[0][1]

    return prediction, probability


if __name__ == "__main__":
    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 90.5,
        "TotalCharges": 450.0
    }

    pred, prob = predict_churn(sample_customer)

    print("Prediction:", "Churn" if pred == 1 else "No Churn")
    print("Churn Probability:", round(prob, 3))
