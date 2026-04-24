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


st.set_page_config(
    page_title="Customer Churn Intelligence",
    layout="wide"
)


@st.cache_resource
def load_churn_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def get_explainer(_xgb_model):
    return shap.TreeExplainer(_xgb_model)


def get_risk_label(probability):
    if probability >= 0.70:
        return "High Risk", "error"
    elif probability >= 0.40:
        return "Medium Risk", "warning"
    elif probability >= THRESHOLD:
        return "Low-Medium Risk", "info"
    else:
        return "Low Risk", "success"


def clean_feature_name(name):
    return (
        name.replace("cat__", "")
        .replace("num__", "")
        .replace("_", " ")
        .title()
    )


model = load_churn_model()
preprocessor = model.named_steps["preprocessor"]
xgb_model = model.named_steps["model"]
explainer = get_explainer(xgb_model)


st.title("AI-Powered Customer Churn Intelligence Platform")
st.write(
    "Predict customer churn risk, explain the main reasons behind each prediction, "
    "and recommend targeted retention actions."
)

st.subheader("Why This Matters")
st.write(
    """
    This tool helps businesses identify customers at risk of leaving before they churn.
    By focusing on high-risk users, teams can improve retention, reduce revenue loss,
    and personalize customer outreach.
    """
)


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

AvgMonthlySpend = TotalCharges / (tenure + 1)

if tenure <= 12:
    TenureGroup = "0-1 year"
elif tenure <= 24:
    TenureGroup = "1-2 years"
elif tenure <= 48:
    TenureGroup = "2-4 years"
else:
    TenureGroup = "4-6 years"


input_data = pd.DataFrame(
    [
        {
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
        }
    ]
)


st.subheader("Customer Input")
st.dataframe(input_data, use_container_width=True)


if st.button("Predict Churn Risk"):
    probability = model.predict_proba(input_data)[0][1]
    prediction = 1 if probability >= THRESHOLD else 0

    risk_label, risk_level = get_risk_label(probability)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Churn Probability", f"{probability * 100:.2f}%")

    with col2:
        st.metric("Decision Threshold", f"{THRESHOLD:.2f}")

    with col3:
        if risk_level == "error":
            st.error(f"{risk_label}: Customer likely to churn")
        elif risk_level == "warning":
            st.warning(f"{risk_label}: Monitor closely")
        elif risk_level == "info":
            st.info(f"{risk_label}: Some churn signals present")
        else:
            st.success(f"{risk_label}: Customer likely to stay")

    st.subheader("Business Recommendation")

    if probability >= 0.70:
        st.write(
            "Offer an immediate retention discount, premium support, or contract upgrade incentive."
        )
    elif probability >= 0.40:
        st.write(
            "Send a personalized engagement campaign and monitor customer activity closely."
        )
    elif probability >= THRESHOLD:
        st.write(
            "Customer shows some churn signals. Consider light-touch retention outreach."
        )
    else:
        st.write("No urgent action needed. Continue normal engagement.")

    st.subheader("Why This Prediction?")

    transformed_input = preprocessor.transform(input_data)
    feature_names = preprocessor.get_feature_names_out()

    transformed_df = pd.DataFrame(
        transformed_input,
        columns=feature_names,
    )

    shap_values = explainer.shap_values(transformed_df)

    shap_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "SHAP Value": shap_values[0],
            "Impact": [
                "Increases Churn Risk" if value > 0 else "Reduces Churn Risk"
                for value in shap_values[0]
            ],
        }
    )

    shap_df["Absolute Impact"] = shap_df["SHAP Value"].abs()
    shap_df = shap_df.sort_values("Absolute Impact", ascending=False).head(10)
    shap_df["Feature"] = shap_df["Feature"].apply(clean_feature_name)

    st.write("Top factors influencing this customer's churn prediction:")
    st.dataframe(
        shap_df[["Feature", "SHAP Value", "Impact"]],
        use_container_width=True,
    )

    st.subheader("SHAP Explanation Plot")

    fig = plt.figure()
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=transformed_df.iloc[0],
            feature_names=[clean_feature_name(name) for name in feature_names],
        ),
        max_display=10,
        show=False,
    )

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Global Churn Drivers")

    if os.path.exists(GLOBAL_SHAP_PATH):
        st.image(
            GLOBAL_SHAP_PATH,
            caption="Top factors influencing churn across all customers",
            use_container_width=True,
        )
    else:
        st.info(
            "Global SHAP summary image not found. Add reports/shap_summary.png to display overall churn drivers."
        )
