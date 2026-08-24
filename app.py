"""
Credit Card Fraud Detection — Streamlit Dashboard
Run with: streamlit run app.py
Place creditcard.csv in the same folder as this file.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, auc
)
from xgboost import XGBClassifier

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

# ---------------------------------------------------------
# Data + model loading (cached so it only runs once)
# ---------------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("creditcard.csv")
    return df

@st.cache_resource
def train_models(df):
    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[["Time", "Amount"]] = scaler.fit_transform(X_train[["Time", "Amount"]])
    X_test_scaled[["Time", "Amount"]] = scaler.transform(X_test[["Time", "Amount"]])

    # Logistic Regression baseline
    lr = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
    lr.fit(X_train_scaled, y_train)

    # XGBoost
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42, eval_metric="logloss")
    xgb.fit(X_train_scaled, y_train)

    return {
        "scaler": scaler,
        "lr": lr,
        "xgb": xgb,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "X_test_scaled": X_test_scaled,
    }

with st.spinner("Loading data and training models (first run only, cached after)..."):
    df = load_data()
    models = train_models(df)

lr_model = models["lr"]
xgb_model = models["xgb"]
scaler = models["scaler"]
X_test = models["X_test"]
y_test = models["y_test"]
X_test_scaled = models["X_test_scaled"]

# ---------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------

st.sidebar.title("Fraud Detection Dashboard")
page = st.sidebar.radio("Go to", ["Overview", "Model Comparison", "Try a Transaction"])

# ---------------------------------------------------------
# Page 1: Overview
# ---------------------------------------------------------

if page == "Overview":
    st.title("Credit Card Fraud Detection")
    st.markdown("Dataset overview and class imbalance, from the Kaggle Credit Card Fraud dataset.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total transactions", f"{len(df):,}")
    fraud_count = int(df["Class"].sum())
    col2.metric("Fraud cases", f"{fraud_count:,}")
    col3.metric("Fraud rate", f"{fraud_count / len(df) * 100:.3f}%")

    st.subheader("Amount distribution: Fraud vs Normal")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(df[df["Class"] == 0]["Amount"], bins=50, color="steelblue")
    axes[0].set_title("Normal")
    axes[0].set_xlabel("Amount")
    axes[1].hist(df[df["Class"] == 1]["Amount"], bins=50, color="crimson")
    axes[1].set_title("Fraud")
    axes[1].set_xlabel("Amount")
    st.pyplot(fig)

# ---------------------------------------------------------
# Page 2: Model comparison
# ---------------------------------------------------------

elif page == "Model Comparison":
    st.title("Model Comparison")

    y_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]
    y_proba_xgb = xgb_model.predict_proba(X_test_scaled)[:, 1]

    prec_lr, rec_lr, thresh_lr = precision_recall_curve(y_test, y_proba_lr)
    prec_xgb, rec_xgb, thresh_xgb = precision_recall_curve(y_test, y_proba_xgb)
    auc_lr = auc(rec_lr, prec_lr)
    auc_xgb = auc(rec_xgb, prec_xgb)

    col1, col2 = st.columns(2)
    col1.metric("Logistic Regression PR-AUC", f"{auc_lr:.3f}")
    col2.metric("XGBoost PR-AUC", f"{auc_xgb:.3f}", delta=f"{auc_xgb - auc_lr:+.3f}")

    st.subheader("Precision-Recall Curve")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rec_lr, prec_lr, label=f"Logistic Regression (AUC={auc_lr:.3f})")
    ax.plot(rec_xgb, prec_xgb, label=f"XGBoost (AUC={auc_xgb:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("Threshold Explorer (XGBoost)")
    threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)
    y_pred_at_thresh = (y_proba_xgb >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred_at_thresh)
    tn, fp, fn, tp = cm.ravel()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("True Positives (fraud caught)", tp)
    col2.metric("False Negatives (fraud missed)", fn)
    col3.metric("False Positives (false alarms)", fp)
    col4.metric("True Negatives (correctly cleared)", tn)

    st.text("Classification Report at this threshold:")
    st.code(classification_report(y_test, y_pred_at_thresh, target_names=["Normal", "Fraud"]))

# ---------------------------------------------------------
# Page 3: Try a transaction
# ---------------------------------------------------------

elif page == "Try a Transaction":
    st.title("Try a Transaction")
    st.markdown(
        "Pick a real transaction from the test set (or a random one) and see what "
        "the model predicts. V1-V28 are PCA-anonymized features, so this uses "
        "existing rows rather than manual input."
    )

    mode = st.radio("Pick a transaction", ["Random normal", "Random fraud", "By row index"])

    if mode == "Random normal":
        row = X_test[y_test == 0].sample(1, random_state=None)
    elif mode == "Random fraud":
        row = X_test[y_test == 1].sample(1, random_state=None)
    else:
        idx = st.number_input("Row index (from test set)", min_value=0, max_value=len(X_test) - 1, value=0)
        row = X_test.iloc[[idx]]

    actual_label = y_test.loc[row.index[0]]

    row_scaled = row.copy()
    row_scaled[["Time", "Amount"]] = scaler.transform(row[["Time", "Amount"]])

    proba_xgb = xgb_model.predict_proba(row_scaled)[0, 1]
    proba_lr = lr_model.predict_proba(row_scaled)[0, 1]

    st.subheader("Transaction details")
    st.dataframe(row[["Time", "Amount"]])

    col1, col2, col3 = st.columns(3)
    col1.metric("Actual label", "Fraud" if actual_label == 1 else "Normal")
    col2.metric("XGBoost fraud probability", f"{proba_xgb:.3f}")
    col3.metric("Logistic Reg. fraud probability", f"{proba_lr:.3f}")

    if proba_xgb >= 0.5:
        st.error(f"⚠️ Flagged as FRAUD (probability {proba_xgb:.1%})")
    else:
        st.success(f"✅ Flagged as Normal (fraud probability {proba_xgb:.1%})")
