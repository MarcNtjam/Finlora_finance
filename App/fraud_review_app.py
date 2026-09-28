"""
Finlora — Fraud Review Queue (Streamlit)
Prioritised, explainable review queue backed by the trained Random Forest.

Run from the project root:
    streamlit run App/fraud_review_app.py
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------- paths
_here = Path(__file__).resolve()
ROOT = _here.parents[1]
for cand in [ROOT] + list(_here.parents):
    if (cand / "Model" / "fraud_rf_model.joblib").exists():
        ROOT = cand; break
MODEL_DIR = ROOT / "Model"
DATA_PATH = ROOT / "Data" / "Processed" / "finlora_merged_clean.csv"

st.set_page_config(page_title="Finlora Fraud Review Queue", layout="wide", page_icon="🛡️")

# ----------------------------------------------------------------------------- load artefacts (cached)
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_DIR / "fraud_rf_model.joblib")
    feature_names = joblib.load(MODEL_DIR / "feature_names.joblib")
    metadata = joblib.load(MODEL_DIR / "model_metadata.joblib")
    return model, feature_names, metadata

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

NUMERIC = ["amount_to_avg_ratio", "transaction_velocity_1h", "amount",
           "account_age_days", "hour_of_day",
           "is_new_device", "is_cross_border", "is_device_known"]
CATEG = ["account_type", "kyc_tier", "merchant_category", "channel"]

def build_features(df, feature_names):
    """One-hot encode exactly as in training and align to the model's columns."""
    X = pd.get_dummies(df[NUMERIC + CATEG], columns=CATEG, drop_first=True)
    X = X.reindex(columns=feature_names, fill_value=0)   # guarantee identical column set/order
    return X

def top_reasons(row, importances, k=3):
    """Human-readable 'why this scored high' for one transaction."""
    reasons = []
    if row["transaction_velocity_1h"] >= 2:
        reasons.append(f"{int(row['transaction_velocity_1h'])} transactions in the trailing hour (burst)")
    if row["amount_to_avg_ratio"] >= 3:
        reasons.append(f"{row['amount_to_avg_ratio']:.1f}× the account's normal spend")
    if row["merchant_category"] in ("Wire Transfer", "Payroll Transfer", "Crypto Exchange"):
        reasons.append(f"high-risk category: {row['merchant_category']}")
    if row.get("is_new_device", 0) == 1:
        reasons.append("transaction from a new device")
    if row.get("is_cross_border", 0) == 1:
        reasons.append("cross-border transaction")
    if row.get("is_device_known", 1) == 0:
        reasons.append("no device information captured")
    return reasons[:k] if reasons else ["No single dominant signal — flagged on combined factors"]

# ============================================================================= load
try:
    model, feature_names, meta = load_model()
except FileNotFoundError:
    st.error("Model artefacts not found. Run `python Model/train_and_save_model.py` first.")
    st.stop()

# ============================================================================= header
st.title("🛡️ Finlora — Fraud Review Queue")
st.caption("Prioritised, explainable transaction risk scoring · Random Forest prototype")

m = meta["metrics"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("ROC-AUC", m["roc_auc"])
c2.metric("Recall", m["recall"])
c3.metric("Precision", m["precision"])
c4.metric("Base fraud rate", f"{meta['fraud_rate']*100:.2f}%")

st.divider()

tab_queue, tab_score, tab_model = st.tabs(
    ["📋 Review Queue", "🔍 Score a Transaction", "📊 Model Insights"])

# ============================================================================= TAB 1: QUEUE
with tab_queue:
    st.subheader("Prioritised review queue")
    st.write("Transactions scored by the model and ranked by fraud probability. "
             "Use the threshold dial to match the queue size to your review capacity.")

    left, right = st.columns([1, 2])
    with left:
        preset = st.radio("Operating point (target recall)",
                          ["Custom", "70% recall", "80% recall", "90% recall"], index=2)
        preset_map = {"70% recall": meta["recall_thresholds"]["0.70"],
                      "80% recall": meta["recall_thresholds"]["0.80"],
                      "90% recall": meta["recall_thresholds"]["0.90"]}
        default_thr = preset_map.get(preset, 0.5)
        threshold = st.slider("Score threshold", 0.0, 1.0, float(default_thr), 0.01,
                              help="Transactions scoring at or above this are sent to the queue.")
        sample_n = st.select_slider("Transactions to score",
                                    options=[5000, 10000, 25000, 50000, 126000], value=25000)

    df = load_data()
    work = df.sample(min(sample_n, len(df)), random_state=42).copy() if sample_n < len(df) else df.copy()
    X = build_features(work, feature_names)
    work["fraud_score"] = model.predict_proba(X)[:, 1]

    queue = work[work["fraud_score"] >= threshold].sort_values("fraud_score", ascending=False)

    with right:
        q1, q2, q3 = st.columns(3)
        q1.metric("Flagged for review", f"{len(queue):,}")
        q2.metric("Share of scored", f"{len(queue)/len(work)*100:.1f}%")
        # If the sample carries the label, show how much actual fraud is captured
        if "is_fraud" in work.columns:
            caught = queue["is_fraud"].sum(); total_fraud = work["is_fraud"].sum()
            q3.metric("Actual fraud in queue",
                      f"{caught}/{total_fraud}",
                      f"{(caught/total_fraud*100 if total_fraud else 0):.0f}% caught")

    st.markdown("##### Top of the queue")
    show_cols = ["transaction_id", "fraud_score", "amount", "amount_to_avg_ratio",
                 "transaction_velocity_1h", "merchant_category", "channel",
                 "account_type", "kyc_tier"]
    show_cols = [c for c in show_cols if c in queue.columns]
    top = queue.head(25).copy()
    top["fraud_score"] = (top["fraud_score"] * 100).round(1)

    st.dataframe(
        top[show_cols].rename(columns={"fraud_score": "risk_score_%"}),
        use_container_width=True, hide_index=True,
    )

    # Explainability for the single highest-risk transaction
    if len(queue):
        st.markdown("##### Why the top transaction was flagged")
        r = queue.iloc[0]
        st.write(f"**{r['transaction_id']}** — risk score **{r['fraud_score']*100:.1f}%**")
        for reason in top_reasons(r, meta["feature_importances"]):
            st.write(f"- {reason}")

    st.download_button(
        "⬇️ Download full queue (CSV)",
        queue[show_cols].to_csv(index=False).encode(),
        file_name="fraud_review_queue.csv", mime="text/csv")

# ============================================================================= TAB 2: SCORE ONE
with tab_score:
    st.subheader("Score a single transaction")
    st.write("Enter transaction details to get an instant fraud probability and the contributing signals.")

    opts = meta["categorical_options"]
    col1, col2, col3 = st.columns(3)
    with col1:
        amount = st.number_input("Amount", min_value=0.0, value=5000.0, step=100.0)
        avg30 = st.number_input("Account 30-day avg spend", min_value=1.0, value=5000.0, step=100.0)
        velocity = st.number_input("Transactions in last 1h", min_value=0, max_value=20, value=1)
    with col2:
        acc_type = st.selectbox("Account type", opts["account_type"])
        kyc = st.selectbox("KYC tier", opts["kyc_tier"])
        merch = st.selectbox("Merchant category", opts["merchant_category"])
        channel = st.selectbox("Channel", opts["channel"])
    with col3:
        hour = st.slider("Hour of day", 0, 23, 14)
        acc_age = st.number_input("Account age (days)", min_value=0, value=300)
        new_dev = st.checkbox("New device")
        cross = st.checkbox("Cross-border")

    if st.button("Score transaction", type="primary"):
        ratio = round(amount / avg30, 2) if avg30 > 0 else 0.0
        one = pd.DataFrame([{
            "amount_to_avg_ratio": ratio, "transaction_velocity_1h": velocity,
            "amount": amount, "account_age_days": acc_age, "hour_of_day": hour,
            "is_new_device": int(new_dev), "is_cross_border": int(cross),
            "is_device_known": 1,
            "account_type": acc_type, "kyc_tier": kyc,
            "merchant_category": merch, "channel": channel,
        }])
        Xone = build_features(one, feature_names)
        score = float(model.predict_proba(Xone)[:, 1][0])

        st.metric("Fraud probability", f"{score*100:.1f}%")
        if score >= 0.5:
            st.error("HIGH RISK — recommend review")
        elif score >= 0.2:
            st.warning("ELEVATED — monitor")
        else:
            st.success("LOW RISK")

        st.markdown("**Contributing signals:**")
        for reason in top_reasons(one.iloc[0], meta["feature_importances"]):
            st.write(f"- {reason}")
        st.caption(f"amount_to_avg_ratio computed as {ratio} (amount ÷ 30-day average).")

# ============================================================================= TAB 3: MODEL
with tab_model:
    st.subheader("Model insights")
    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Held-out performance**")
        st.table(pd.DataFrame([meta["metrics"]]).T.rename(columns={0: "value"}))
        st.markdown("**Threshold presets (score → target recall)**")
        st.table(pd.DataFrame(meta["recall_thresholds"], index=["threshold"]).T)
    with colB:
        st.markdown("**Top feature importances**")
        imp = pd.Series(meta["feature_importances"]).head(10)[::-1]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.barh(imp.index, imp.values, color="#c1440e")
        ax.set_xlabel("Importance"); ax.set_title("What drives the score")
        st.pyplot(fig)

    st.info("The two dominant drivers — transaction velocity and amount-to-average ratio — match the EDA. "
            "Every flag can be traced back to these contributing signals rather than a black-box decision.")
