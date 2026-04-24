import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt


MODEL_PATH = "models/churn_model.pkl"
GLOBAL_SHAP_PATH = "reports/shap_summary.png"
THRESHOLD = 0.30


st.set_page_config(page_title="Customer Churn Intelligence", layout="wide")


# ===================== LOAD MODEL =====================
@st.cache_resource
def load_churn_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)

# ===================== HELPERS =====================
def get_risk_label(prob):
    if prob >= 0.70:
        return "High Risk", "error"
    elif prob >= 0.40:
        return "Medium Risk", "warning"
    elif prob >= THRESHOLD:
        return "Low-Medium Risk", "info"
    else:
        return "Low Risk", "success"


def clean_feature_name(name):
    name = name.replace("cat__", "").replace("num__", "")

    replacements = {
        "Internetservice": "Internet Service",
        "Onlinesecurity": "Online Security",
        "Onlinebackup": "Online Backup",
        "Deviceprotection": "Device Protection",
        "Techsupport": "Tech Support",
        "Streamingtv": "Streaming TV",
        "Streamingmovies": "Streaming Movies",
        "Paymentmethod": "Payment Method",
        "Paperlessbilling": "Paperless Billing",
        "Monthlycharges": "Monthly Charges",
        "Totalcharges": "Total Charges",
        "Avgmonthlyspend": "Avg Monthly Spend",
        "Seniorcitizen": "Senior Citizen",
    }

    name = name.replace("_", " ")

    for key, value in replacements.items():
        name = name.replace(key, value)

    return name


def highlight_impact(val):
    return "color: red" if val == "Increases Churn Risk" else "color: green"


# ===================== LOAD =====================
model = load_churn_model()
preprocessor = model.named_steps["preprocessor"]
xgb_model = model.named_steps["model"]
explainer = get_explainer(xgb_model)


# ===================== UI =====================
st.title("AI-Powered Customer Churn Intelligence Platform")

st.write(
    "Predict customer churn risk, explain predictions, and recommend targeted retention strategies."
)

st.subheader("Why This Matters")
st.write(
    """
    This tool helps businesses identify customers at risk of leaving before they churn.
    By focusing on high-risk users, companies can improve retention and reduce revenue loss.
    """
)


# ===================== SIDEBAR =====================
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
        "Credit card (automatic)",
    ],
)

MonthlyCharges = st.sidebar.number_input("Monthly Charges", min_value=0.0, value=70.0)
TotalCharges = st.sidebar.number_input("Total Charges", min_value=0.0, value=1000.0)


# ===================== FEATURE ENGINEERING =====================
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
    "TenureGroup": TenureGroup,
}])


st.subheader("Customer Input")
st.dataframe(input_data, use_container_width=True)


# ===================== PREDICTION =====================
if st.button("Predict Churn Risk"):

    probability = model.predict_proba(input_data)[0][1]
    prediction = 1 if probability >= THRESHOLD else 0

    label, level = get_risk_label(probability)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Churn Probability", f"{probability * 100:.2f}%")

    with col2:
        st.metric("Threshold", f"{THRESHOLD}")

    with col3:
        if level == "error":
            st.error(label)
        elif level == "warning":
            st.warning(label)
        elif level == "info":
            st.info(label)
        else:
            st.success(label)

    st.subheader("Churn Risk Score")
    st.progress(float(probability))


    # ===================== BUSINESS RECOMMENDATION =====================
    st.subheader("Business Recommendation")

    if probability >= 0.70:
        st.write("Offer retention discount or contract upgrade.")
    elif probability >= 0.40:
        st.write("Send personalized engagement campaign.")
    elif probability >= THRESHOLD:
        st.write("Light retention outreach recommended.")
    else:
        st.write("Customer shows strong retention signals. No immediate intervention required.")


    # ===================== SHAP =====================
    st.subheader("Why This Prediction?")

    transformed_input = preprocessor.transform(input_data)
    feature_names = preprocessor.get_feature_names_out()

    transformed_df = pd.DataFrame(transformed_input, columns=feature_names)

    shap_values = explainer.shap_values(transformed_df)

    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": shap_values[0],
        "Impact": [
            "Increases Churn Risk" if v > 0 else "Reduces Churn Risk"
            for v in shap_values[0]
        ]
    })

    shap_df["Absolute Impact"] = shap_df["SHAP Value"].abs()
    shap_df = shap_df.sort_values("Absolute Impact", ascending=False).head(10)
    shap_df["Feature"] = shap_df["Feature"].apply(clean_feature_name)

    def color_impact(val):
        if val == "Increases Churn Risk":
            return "🔴 Increases Churn Risk"
        else:
            return "🟢 Reduces Churn Risk"

  display_df = shap_df[["Feature", "SHAP Value", "Impact"]].copy()
  display_df["Impact"] = display_df["Impact"].apply(color_impact)

  st.dataframe(display_df, use_container_width=True)

    # ===================== WATERFALL =====================
    st.subheader("Customer-Level Explanation (Why this prediction?)")

    fig = plt.figure()
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=transformed_df.iloc[0],
            feature_names=[clean_feature_name(f) for f in feature_names],
        ),
        show=False
    )
    st.pyplot(fig)
    plt.close(fig)


    # ===================== GLOBAL SHAP =====================
    st.subheader("Global Churn Drivers")

    if os.path.exists(GLOBAL_SHAP_PATH):
        st.image(GLOBAL_SHAP_PATH, use_container_width=True)
    else:
        st.info("Global SHAP image not found.")
