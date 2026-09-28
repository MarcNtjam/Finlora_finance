# Finlora Fraud Detection — Deployment

Streamlit review-queue app backed by the trained Random Forest model.

## One-time setup

From the project root (`D:\Amdari\Project_finlora`), in your VS Code terminal:

```powershell
# 1. (recommended) activate your virtual environment first, then:
pip install -r requirements.txt

# 2. train and save the model artefacts into Model\
python Model\train_and_save_model.py
```

That writes three files into `Model\`:
- `fraud_rf_model.joblib`   — the trained Random Forest
- `feature_names.joblib`    — the exact feature column order
- `model_metadata.joblib`   — metrics, thresholds, feature options

## Run the app

```powershell
streamlit run App\fraud_review_app.py
```

It opens in your browser at http://localhost:8501.

## What the app does

- **Review Queue** — scores transactions, ranks them by fraud probability, and lets you set the
  score threshold via 70/80/90%-recall presets to match review capacity. Shows how much actual
  fraud the current queue captures, and lets you download the queue as CSV.
- **Score a Transaction** — enter details for one transaction and get an instant probability plus
  the contributing signals (explainability).
- **Model Insights** — held-out metrics, threshold presets, and feature importances.

## Directory layout

```
Project_finlora\
├── Data\
│   ├── Raw\           finlora_accounts.csv, finlora_transactions.csv
│   └── Processed\     finlora_merged_clean.csv
├── Model\             train_and_save_model.py + saved .joblib artefacts
├── Notebook\          01_fraud_detection.ipynb
└── App\               fraud_review_app.py, README.md
```
