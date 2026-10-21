"""
evaluate.py
-----------
Stand-alone evaluation script.
Loads the trained model and re-runs full evaluation on the held-out
test set. Also prints a formatted classification report and writes
a summary JSON to outputs/.

Usage
-----
  python evaluate.py
"""

import sys
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    accuracy_score,
)
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).parent.resolve()
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
DATA_DIR = BASE_DIR / "data"

sys.path.insert(0, str(BASE_DIR))
from preprocessing import preprocess_series  # noqa: E402

MODEL_PATH = MODELS_DIR / "best_model_pipeline.joblib"
META_PATH = MODELS_DIR / "model_metadata.json"


def load_test_data():
    csv_path = DATA_DIR / "sms_spam.csv"
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "v1" in df.columns and "v2" in df.columns:
        df = df.rename(columns={"v1": "label", "v2": "message"})
    df = df[["label", "message"]].dropna()
    df["label"] = df["label"].str.strip().str.lower()
    df = df[df["label"].isin(["spam", "ham"])].copy()
    df["label_enc"] = (df["label"] == "spam").astype(int)
    df["clean_message"] = preprocess_series(df["message"])
    df = df[df["clean_message"].str.strip().str.len() > 0].copy()

    _, X_test, _, y_test = train_test_split(
        df["clean_message"], df["label_enc"],
        test_size=0.20, random_state=42, stratify=df["label_enc"],
    )
    return X_test, y_test


def run_evaluation():
    print("\n" + "=" * 60)
    print("  SPAM CLASSIFIER — EVALUATION REPORT")
    print("=" * 60)

    if not MODEL_PATH.exists():
        print(f"\n[ERROR] Model not found at {MODEL_PATH}")
        print("  Run `python train.py` first.\n")
        sys.exit(1)

    pipeline = joblib.load(MODEL_PATH)
    model_name = "Best Model"
    if META_PATH.exists():
        with open(META_PATH) as fp:
            meta = json.load(fp)
        model_name = meta.get("best_model", "Best Model")

    print(f"\n  Model loaded : {model_name}")
    print("  Loading test data...")
    X_test, y_test = load_test_data()
    print(f"  Test samples : {len(X_test):,}")

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", pos_label=1
    )
    roc_auc = roc_auc_score(y_test, y_prob)

    # Per-class metrics
    prec_all, rec_all, f1_all, sup_all = precision_recall_fscore_support(
        y_test, y_pred, labels=[0, 1]
    )

    # -----------------------------------------------------------------------
    # Console output
    # -----------------------------------------------------------------------

    print(f"\n{'─'*60}")
    print(f"  {'Metric':<20} {'Score':>10}")
    print(f"{'─'*60}")
    print(f"  {'Accuracy':<20} {acc:>10.4f}")
    print(f"  {'Precision (spam)':<20} {prec:>10.4f}")
    print(f"  {'Recall (spam)':<20} {rec:>10.4f}")
    print(f"  {'F1-Score (spam)':<20} {f1:>10.4f}")
    print(f"  {'ROC-AUC':<20} {roc_auc:>10.4f}")
    print(f"{'─'*60}")

    print("\n  Full Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

    # -----------------------------------------------------------------------
    # Confusion Matrix (annotated)
    # -----------------------------------------------------------------------

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n  Confusion Matrix:")
    print(f"    True Negatives  (Ham   -> Ham  ) : {tn:>4}")
    print(f"    False Positives (Ham   -> Spam ) : {fp:>4}  <- false alarms")
    print(f"    False Negatives (Spam  -> Ham  ) : {fn:>4}  <- missed spam")
    print(f"    True Positives  (Spam  -> Spam ) : {tp:>4}")

    # Save evaluation summary
    summary = {
        "model": model_name,
        "test_samples": int(len(X_test)),
        "metrics": {
            "accuracy": round(acc, 4),
            "precision_spam": round(prec, 4),
            "recall_spam": round(rec, 4),
            "f1_spam": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
        },
        "per_class": {
            "ham":  {"precision": round(prec_all[0], 4), "recall": round(rec_all[0], 4),
                     "f1": round(f1_all[0], 4), "support": int(sup_all[0])},
            "spam": {"precision": round(prec_all[1], 4), "recall": round(rec_all[1], 4),
                     "f1": round(f1_all[1], 4), "support": int(sup_all[1])},
        },
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    out_path = OUTPUTS_DIR / "evaluation_summary.json"
    with open(out_path, "w") as fp_:
        json.dump(summary, fp_, indent=2)
    print(f"\n  Evaluation summary saved to: {out_path}")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    run_evaluation()
