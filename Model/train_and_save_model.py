"""
Train the Random Forest fraud model and persist all artefacts the Streamlit app needs.
Run once from the project root:  python Model/train_and_save_model.py
Saves to the Model/ directory:
  - fraud_rf_model.joblib      the trained Random Forest
  - feature_names.joblib       exact column order the model expects
  - model_metadata.joblib      thresholds, metrics, feature options for the app
"""
import joblib, json
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, precision_recall_curve)

RANDOM_STATE = 42
# Find project root by walking up until we see Data/Processed
_here = Path(__file__).resolve()
ROOT = _here.parents[1]
for cand in [ROOT] + list(_here.parents):
    if (cand / "Data" / "Processed" / "finlora_merged_clean.csv").exists():
        ROOT = cand; break
DATA = ROOT / "Data" / "Processed" / "finlora_merged_clean.csv"
MODEL_DIR = ROOT / "Model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

numeric_features = ["amount_to_avg_ratio", "transaction_velocity_1h", "amount",
                    "account_age_days", "hour_of_day",
                    "is_new_device", "is_cross_border", "is_device_known"]
categorical_features = ["account_type", "kyc_tier", "merchant_category", "channel"]

df = pd.read_csv(DATA)
X = pd.get_dummies(df[numeric_features + categorical_features],
                   columns=categorical_features, drop_first=True)
y = df["is_fraud"]
feature_names = X.columns.tolist()

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                          stratify=y, random_state=RANDOM_STATE)
rf = RandomForestClassifier(n_estimators=300, max_depth=10,
                            class_weight="balanced_subsample",
                            random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_tr, y_tr)

proba = rf.predict_proba(X_te)[:, 1]
pred = rf.predict(X_te)
metrics = {"precision": round(precision_score(y_te, pred), 3),
           "recall": round(recall_score(y_te, pred), 3),
           "f1": round(f1_score(y_te, pred), 3),
           "roc_auc": round(roc_auc_score(y_te, proba), 3)}

# thresholds that achieve 70/80/90% recall on the held-out set
prec, rec, thr = precision_recall_curve(y_te, proba)
def thr_for_recall(t):
    idx = np.where(rec >= t)[0]
    if not len(idx): return 0.5
    best = idx[np.argmax(prec[idx])]
    return float(thr[best]) if best < len(thr) else 1.0
recall_thresholds = {"0.70": round(thr_for_recall(0.70), 4),
                     "0.80": round(thr_for_recall(0.80), 4),
                     "0.90": round(thr_for_recall(0.90), 4)}

metadata = {
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "categorical_options": {c: sorted(df[c].unique().tolist()) for c in categorical_features},
    "metrics": metrics,
    "recall_thresholds": recall_thresholds,
    "fraud_rate": round(df["is_fraud"].mean(), 4),
    "n_transactions": int(len(df)),
    "feature_importances": pd.Series(rf.feature_importances_, index=feature_names)
                             .sort_values(ascending=False).round(4).to_dict(),
}

joblib.dump(rf, MODEL_DIR / "fraud_rf_model.joblib")
joblib.dump(feature_names, MODEL_DIR / "feature_names.joblib")
joblib.dump(metadata, MODEL_DIR / "model_metadata.joblib")

print("Saved model artefacts to", MODEL_DIR)
print("Metrics:", metrics)
print("Recall thresholds:", recall_thresholds)
