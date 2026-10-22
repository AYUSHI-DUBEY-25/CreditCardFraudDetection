import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection System")
st.caption("AI-powered transaction analysis using Machine Learning")

# ---------------- LOAD MODELS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "data", "creditcard.csv")

@st.cache_resource
def load_models():
    lr = joblib.load(os.path.join(MODEL_DIR, "logistic_regression.pkl"))
    rf = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    features = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
    return lr, rf, scaler, features

try:
    lr_model, rf_model, scaler, feature_columns = load_models()
except Exception as e:
    st.error(f"Could not load saved models. Check your models folder. Details: {e}")
    st.stop()

# ---------------- SIDEBAR ----------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a page",
    ["Dashboard Overview", "Fraud Prediction", "Model Performance"]
)

# ---------------- LOAD DATASET ----------------
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

df = None
if os.path.exists(DATA_PATH):
    try:
        df = load_data(DATA_PATH)
    except Exception:
        st.warning("Dataset could not be loaded.")
else:
    st.warning(
        "Dataset file not found at data/creditcard.csv. "
        "Dataset overview and charts will be unavailable until you add it."
    )

# ---------------- DASHBOARD OVERVIEW ----------------
if page == "Dashboard Overview":
    st.header("Dataset Overview")

    if df is not None:
        total = len(df)
        fraud_count = int((df["Class"] == 1).sum())
        legit_count = int((df["Class"] == 0).sum())
        fraud_percent = fraud_count / total * 100 if total else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Transactions", f"{total:,}")
        c2.metric("Fraudulent Transactions", f"{fraud_count:,}")
        c3.metric("Legitimate Transactions", f"{legit_count:,}")
        c4.metric("Fraud Percentage", f"{fraud_percent:.3f}%")

        st.subheader("Transaction Class Distribution")
        counts = df["Class"].value_counts().reindex([0, 1], fill_value=0)
        st.bar_chart(counts.rename(index={
            0: "Legitimate",
            1: "Fraud"
        }))

        st.subheader("Transaction Amount Distribution")
        fig, ax = plt.subplots(figsize=(9, 4))
        for label, name in [(0, "Legitimate"), (1, "Fraud")]:
            amounts = df.loc[df["Class"] == label, "Amount"]
            ax.hist(
                amounts,
                bins=50,
                alpha=0.6,
                label=name
            )
        ax.set_xlim(0, df["Amount"].quantile(0.99))
        ax.set_xlabel("Transaction Amount")
        ax.set_ylabel("Number of Transactions")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
    else:
        st.info("Add creditcard.csv to your data folder to view dataset analytics.")

# ---------------- FRAUD PREDICTION ----------------
elif page == "Fraud Prediction":
    st.header("Check a Transaction for Fraud")
    st.info(
        "Enter all transaction features. V1–V28 are anonymized dataset features; "
        "use a real row from the dataset for a meaningful demonstration."
    )

    if df is None:
        st.warning("Place creditcard.csv inside the data folder to use this demo.")
    else:
        selected_row = st.number_input(
            "Dataset row number",
            min_value=0,
            max_value=len(df) - 1,
            value=0,
            step=1
        )

        if st.button("Analyze Transaction", type="primary"):
            row = df.iloc[int(selected_row)]
            actual_class = int(row["Class"])
            input_df = pd.DataFrame(
                [row[feature_columns].to_dict()],
                columns=feature_columns
            )

            # Apply the same scaling used during training
            input_df[["Amount", "Time"]] = scaler.transform(
                input_df[["Amount", "Time"]]
            )

            fraud_probability = rf_model.predict_proba(input_df)[0, 1]
            prediction = int(rf_model.predict(input_df)[0])

            st.subheader("Prediction Result")
            if prediction == 1:
                st.error("⚠️ Model flagged this transaction as potentially fraudulent.")
            else:
                st.success("✅ Model classified this transaction as legitimate.")

            st.metric("Predicted Fraud Probability", f"{fraud_probability:.2%}")
            st.write(
                "Dataset's actual label:",
                "Fraud" if actual_class == 1 else "Legitimate"
            )
            st.caption(
                "This is an educational model demonstration, not a production "
                "financial security system."
            )

# ---------------- MODEL PERFORMANCE ----------------
elif page == "Model Performance":
    st.header("Model Performance Comparison")

    results = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest"],
        "Accuracy": [0.9752, 0.9995],
        "Precision": [0.0562, 0.9452],
        "Recall": [0.8737, 0.7263],
        "F1-Score": [0.1057, 0.8214],
        "PR-AUC": [0.6719, 0.8012]
    })

    st.dataframe(
        results.style.format({
            "Accuracy": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1-Score": "{:.4f}",
            "PR-AUC": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("F1-Score Comparison")
    st.bar_chart(results.set_index("Model")[["F1-Score"]])

    st.subheader("Precision–Recall Comparison")
    st.bar_chart(results.set_index("Model")[["Precision", "Recall"]])

    st.write(
        "**Observation:** Random Forest has a substantially higher F1-score "
        "and PR-AUC in this experiment. Logistic Regression has higher recall, "
        "so it catches more fraud cases at its current classification threshold."
    )
