import pandas as pd
import numpy as np


def load_data(path):
    df = pd.read_csv(path)
    return df


def clean_data(df):
    df = df.copy()

    # Remove customerID because it is not useful for prediction
    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Fill missing TotalCharges with median
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Convert target column
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


def feature_engineering(df):
    df = df.copy()

    # Create useful business features
    df["AvgMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)

    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0-1 year", "1-2 years", "2-4 years", "4-6 years"]
    )

    return df


def preprocess_data(path):
    df = load_data(path)
    df = clean_data(df)
    df = feature_engineering(df)
    return df


if __name__ == "__main__":
    df = preprocess_data("data/telco_churn.csv")
    print(df.head())
    print(df.info())
