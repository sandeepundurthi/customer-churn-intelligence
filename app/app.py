import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt


MODEL_PATH = "models/churn_model.pkl"
THRESHOLD = 0.30

st.set_page_config(
    page_title="Customer Churn Intelligence",
    layout="wide"
)

st.title("Customer Churn Intelligence Platform")
st.write("Predict customer churn risk and explain the main reasons behind each prediction.")

model = joblib.load(MODEL_PATH)

preprocessor = model.named_steps["preprocessor"]
xgb_model = model.named_steps["model"]

st.sidebar.header("Customer Information")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
Partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
Dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
tenure = st.sidebar.slider("Tenure Months", 0, 72, 12)

PhoneService = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
InternetService = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

OnlineSecurity = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])

StreamingTV = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

Contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

MonthlyCharges = st.sidebar.number_input("Monthly Charges", min_value=0.0, value=70.0)
TotalCharges = st.sidebar.number_input("Total Charges", min_value=0.0, value=1000.0)

AvgMonthlySpend = TotalCharges / (tenure + 1)

if tenure <= 12:
    TenureGroup = "0-1 year"
elif tenure <= 24:
    TenureGroup = "1-2 years"
elif tenure <= 48:
    TenureGroup = "2-4 years"
else:
    TenureGroup = "4-6 years"

input_data = pd.DataFrame([{
    "gender": gender,
    "SeniorCitizen": SeniorCitizen,
    "Partner": Partner,
    "Dependents": Dependents,
    "tenure": tenure,
    "PhoneService": PhoneService,
    "MultipleLines": MultipleLines,
    "InternetService": InternetService,
    "OnlineSecurity": OnlineSecurity,
    "OnlineBackup": OnlineBackup,
    "DeviceProtection": DeviceProtection,
    "TechSupport": TechSupport,
    "StreamingTV": StreamingTV,
    "StreamingMovies": StreamingMovies,
    "Contract": Contract,
    "PaperlessBilling": PaperlessBilling,
    "PaymentMethod": PaymentMethod,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges,
    "AvgMonthlySpend": AvgMonthlySpend,
    "TenureGroup": TenureGroup
}])

st.subheader("Customer Input")
st.dataframe(input_data)

if st.button("Predict Churn Risk"):
    probability = model.predict_proba(input_data)[0][1]
    prediction = 1 if probability >= THRESHOLD else 0

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Churn Probability", f"{probability * 100:.2f}%")

    with col2:
        st.metric("Decision Threshold", f"{THRESHOLD:.2f}")

    with col3:
        if prediction == 1:
            st.error("High Risk: Customer may churn")
        else:
            st.success("Low Risk: Customer likely to stay")

    st.subheader("Business Recommendation")

    if probability >= 0.70:
        st.write("Offer immediate retention discount, premium support, or contract upgrade incentive.")
    elif probability >= 0.40:
        st.write("Send personalized engagement campaign and monitor customer activity.")
    elif probability >= THRESHOLD:
        st.write("Customer is at moderate risk. Consider light-touch retention outreach.")
    else:
        st.write("No urgent action needed. Continue normal engagement.")

    st.subheader("Why This Prediction?")

    transformed_input = preprocessor.transform(input_data)
    feature_names = preprocessor.get_feature_names_out()

    transformed_df = pd.DataFrame(
        transformed_input,
        columns=feature_names
    )

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(transformed_df)

    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": shap_values[0],
        "Impact": ["Increases Churn Risk" if value > 0 else "Reduces Churn Risk" for value in shap_values[0]]
    })

    shap_df["Absolute Impact"] = shap_df["SHAP Value"].abs()
    shap_df = shap_df.sort_values("Absolute Impact", ascending=False).head(10)

    st.write("Top factors influencing this customer's churn prediction:")
    st.dataframe(shap_df[["Feature", "SHAP Value", "Impact"]])

    st.subheader("SHAP Explanation Plot")

    plt.figure()
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=transformed_df.iloc[0],
            feature_names=feature_names
        ),
        max_display=10,
        show=False
    )

    st.pyplot(plt.gcf())
