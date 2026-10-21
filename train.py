"""
train.py
--------
Full training pipeline for the Spam Message Classifier.

Workflow
--------
1. Load & explore dataset
2. Preprocess text
3. Train/test split (stratified)
4. Feature extraction with TF-IDF
5. Train Multinomial Naive Bayes + Logistic Regression
6. Evaluate both models (Precision, Recall, F1, Confusion Matrix)
7. Save best model pipeline via joblib
8. Generate all evaluation plots to outputs/
"""

import os
import sys
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    accuracy_score,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Resolve paths relative to *this file* so the script is portable
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

for _d in (DATA_DIR, MODELS_DIR, OUTPUTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Add base_dir to path so imports work
sys.path.insert(0, str(BASE_DIR))
from preprocessing import preprocess_series  # noqa: E402

# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

PALETTE = {"ham": "#2ecc71", "spam": "#e74c3c"}
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "DejaVu Sans",
    }
)


def _save_fig(fig: plt.Figure, name: str) -> None:
    path = OUTPUTS_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {path}")


# ---------------------------------------------------------------------------
# 1. Data Loading
# ---------------------------------------------------------------------------


def load_data() -> pd.DataFrame:
    csv_path = DATA_DIR / "sms_spam.csv"
    if not csv_path.exists():
        print("[INFO] Dataset not found. Running downloader...")
        from data.download_dataset import download_sms_spam_dataset
        download_sms_spam_dataset(save_dir=str(DATA_DIR))

    df = pd.read_csv(csv_path)

    # Normalise column names (handle different CSV structures)
    df.columns = [c.strip().lower() for c in df.columns]
    if "v1" in df.columns and "v2" in df.columns:
        df = df.rename(columns={"v1": "label", "v2": "message"})
    df = df[["label", "message"]].dropna()
    df["label"] = df["label"].str.strip().str.lower()
    df = df[df["label"].isin(["spam", "ham"])].copy()
    df["label_enc"] = (df["label"] == "spam").astype(int)  # 1=spam, 0=ham
    return df


# ---------------------------------------------------------------------------
# 2. EDA Visualisations
# ---------------------------------------------------------------------------


def plot_class_distribution(df: pd.DataFrame) -> None:
    counts = df["label"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(counts.index, counts.values,
                  color=[PALETTE[l] for l in counts.index], edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{val:,}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title("Class Distribution", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    total = len(df)
    ax.set_ylim(0, counts.max() * 1.15)
    pct = counts / total * 100
    ax.set_xticklabels([f"{l}\n({pct[l]:.1f}%)" for l in counts.index], fontsize=11)
    _save_fig(fig, "01_class_distribution.png")


def plot_message_length(df: pd.DataFrame) -> None:
    df = df.copy()
    df["msg_len"] = df["message"].str.len()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, label in zip(axes, ["ham", "spam"]):
        data = df.loc[df["label"] == label, "msg_len"]
        ax.hist(data, bins=40, color=PALETTE[label], edgecolor="white", alpha=0.85)
        ax.axvline(data.mean(), color="black", linestyle="--", linewidth=1.5,
                   label=f"Mean: {data.mean():.0f}")
        ax.set_title(f"Message Length — {label.upper()}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Characters")
        ax.set_ylabel("Frequency")
        ax.legend()
    fig.suptitle("Message Length Distribution by Class", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save_fig(fig, "02_message_length_distribution.png")


# ---------------------------------------------------------------------------
# 3. Preprocessing
# ---------------------------------------------------------------------------


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[STEP] Preprocessing messages...")
    df = df.copy()
    df["clean_message"] = preprocess_series(df["message"])
    # Drop rows where preprocessing leaves an empty string
    df = df[df["clean_message"].str.strip().str.len() > 0].copy()
    return df


# ---------------------------------------------------------------------------
# 4. Train / Test Split
# ---------------------------------------------------------------------------


def split_data(df: pd.DataFrame, test_size: float = 0.20, random_state: int = 42):
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_message"],
        df["label_enc"],
        test_size=test_size,
        random_state=random_state,
        stratify=df["label_enc"],
    )
    print(f"\n[STEP] Train/test split (80/20 stratified):")
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"  Train spam ratio: {y_train.mean():.2%}")
    print(f"  Test  spam ratio: {y_test.mean():.2%}")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 5. Model Building
# ---------------------------------------------------------------------------


def build_pipelines() -> dict:
    """
    Build two sklearn Pipelines:
      - Multinomial Naive Bayes  (TF-IDF + MNB)
      - Logistic Regression      (TF-IDF + LR)
    """
    tfidf_params = dict(
        max_features=10_000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )

    pipelines = {
        "Multinomial Naive Bayes": Pipeline(
            [
                ("tfidf", TfidfVectorizer(**tfidf_params)),
                ("clf", MultinomialNB(alpha=0.1)),
            ]
        ),
        "Logistic Regression": Pipeline(
            [
                ("tfidf", TfidfVectorizer(**tfidf_params)),
                ("clf", LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    solver="lbfgs",
                    class_weight="balanced",
                    random_state=42,
                )),
            ]
        ),
    }
    return pipelines


# ---------------------------------------------------------------------------
# 6. Training & Evaluation
# ---------------------------------------------------------------------------


def evaluate_model(name: str, pipeline: Pipeline, X_train, X_test, y_train, y_test) -> dict:
    print(f"\n{'='*60}")
    print(f"  Model: {name}")
    print("=" * 60)

    # Fit
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", pos_label=1
    )
    roc_auc = roc_auc_score(y_test, y_prob)

    # Cross-validation F1
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1 = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1").mean()

    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}  (spam class)")
    print(f"  Recall    : {rec:.4f}  (spam class)")
    print(f"  F1-Score  : {f1:.4f}  (spam class)")
    print(f"  ROC-AUC   : {roc_auc:.4f}")
    print(f"  CV-F1 (5k): {cv_f1:.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

    return {
        "name": name,
        "pipeline": pipeline,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "cv_f1": cv_f1,
    }


# ---------------------------------------------------------------------------
# 7. Visualisations
# ---------------------------------------------------------------------------


def plot_confusion_matrix(result: dict, y_test) -> None:
    cm = confusion_matrix(y_test, result["y_pred"])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Ham", "Spam"],
        yticklabels=["Ham", "Spam"],
        ax=ax,
        linewidths=0.5,
        cbar=False,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    safe_name = result["name"].replace(" ", "_").lower()
    ax.set_title(f"Confusion Matrix\n{result['name']}", fontsize=13, fontweight="bold")

    # Annotate TP / TN / FP / FN
    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.75, labels[i][j],
                    ha="center", va="center", fontsize=9, color="grey")

    plt.tight_layout()
    _save_fig(fig, f"03_confusion_matrix_{safe_name}.png")


def plot_roc_curve(results: list, y_test) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#3498db", "#e67e22"]
    for result, color in zip(results, colors):
        fpr, tpr, _ = roc_curve(y_test, result["y_prob"])
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{result['name']}  (AUC = {result['roc_auc']:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    _save_fig(fig, "04_roc_curve_comparison.png")


def plot_metric_comparison(results: list) -> None:
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    x = np.arange(len(metrics))
    width = 0.35
    colors = ["#3498db", "#e67e22"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (result, color) in enumerate(zip(results, colors)):
        vals = [result[m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=result["name"],
                      color=color, alpha=0.88, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison — Evaluation Metrics", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=10)
    plt.tight_layout()
    _save_fig(fig, "05_metric_comparison.png")


def plot_top_features(pipeline: Pipeline, n: int = 20) -> None:
    """Plot top-N spam and ham features from Logistic Regression."""
    if not hasattr(pipeline.named_steps["clf"], "coef_"):
        return  # Only works with LR
    feature_names = np.array(pipeline.named_steps["tfidf"].get_feature_names_out())
    coefs = pipeline.named_steps["clf"].coef_[0]
    top_spam = np.argsort(coefs)[-n:][::-1]
    top_ham = np.argsort(coefs)[:n]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, indices, label, color in zip(
        axes,
        [top_spam, top_ham],
        ["SPAM", "HAM"],
        [PALETTE["spam"], PALETTE["ham"]],
    ):
        ax.barh(feature_names[indices][::-1], np.abs(coefs[indices])[::-1],
                color=color, alpha=0.85, edgecolor="white")
        ax.set_title(f"Top {n} {label} Features", fontsize=13, fontweight="bold")
        ax.set_xlabel("|Coefficient|", fontsize=11)
    plt.suptitle("Logistic Regression — Most Informative Features", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save_fig(fig, "06_top_features_lr.png")


# ---------------------------------------------------------------------------
# 8. Save best model
# ---------------------------------------------------------------------------


def save_best_model(results: list) -> dict:
    """Save the model with the highest F1-Score."""
    best = max(results, key=lambda r: r["f1"])
    model_path = MODELS_DIR / "best_model_pipeline.joblib"
    joblib.dump(best["pipeline"], model_path)
    print(f"\n[SAVED] Best model ({best['name']}) -> {model_path}")

    meta = {
        "best_model": best["name"],
        "metrics": {
            "accuracy": round(best["accuracy"], 4),
            "precision": round(best["precision"], 4),
            "recall": round(best["recall"], 4),
            "f1": round(best["f1"], 4),
            "roc_auc": round(best["roc_auc"], 4),
            "cv_f1_5fold": round(best["cv_f1"], 4),
        },
        "all_models": [
            {
                "name": r["name"],
                "metrics": {
                    "accuracy": round(r["accuracy"], 4),
                    "precision": round(r["precision"], 4),
                    "recall": round(r["recall"], 4),
                    "f1": round(r["f1"], 4),
                    "roc_auc": round(r["roc_auc"], 4),
                },
            }
            for r in results
        ],
    }
    meta_path = MODELS_DIR / "model_metadata.json"
    with open(meta_path, "w") as fp:
        json.dump(meta, fp, indent=2)
    print(f"[SAVED] Model metadata -> {meta_path}")
    return best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("\n" + "=" * 60)
    print("  SPAM MESSAGE CLASSIFIER — TRAINING PIPELINE")
    print("=" * 60)

    # 1. Load data
    print("\n[STEP] Loading dataset...")
    df = load_data()
    print(f"  Total messages : {len(df):,}")
    print(f"  Spam           : {(df['label']=='spam').sum():,}")
    print(f"  Ham            : {(df['label']=='ham').sum():,}")

    # 2. EDA plots
    print("\n[STEP] Generating EDA visualizations...")
    plot_class_distribution(df)
    plot_message_length(df)

    # 3. Preprocess
    df = preprocess_data(df)

    # 4. Split
    X_train, X_test, y_train, y_test = split_data(df)

    # 5. Build and evaluate models
    print("\n[STEP] Training and evaluating models...")
    pipelines = build_pipelines()
    results = []
    for name, pipeline in pipelines.items():
        result = evaluate_model(name, pipeline, X_train, X_test, y_train, y_test)
        results.append(result)
        plot_confusion_matrix(result, y_test)

    # 6. Comparative plots
    print("\n[STEP] Generating comparison plots...")
    plot_roc_curve(results, y_test)
    plot_metric_comparison(results)

    # Plot top features for LR
    lr_result = next(r for r in results if "Logistic" in r["name"])
    plot_top_features(lr_result["pipeline"])

    # 7. Save best model
    best = save_best_model(results)

    print("\n" + "=" * 60)
    print(f"  TRAINING COMPLETE")
    print(f"  Best Model  : {best['name']}")
    print(f"  Precision   : {best['precision']:.4f}")
    print(f"  Recall      : {best['recall']:.4f}")
    print(f"  F1-Score    : {best['f1']:.4f}")
    print(f"  ROC-AUC     : {best['roc_auc']:.4f}")
    print("=" * 60)
    print(f"\nAll outputs saved to: {OUTPUTS_DIR}")
    print(f"Model saved to      : {MODELS_DIR}")

    return results, best


if __name__ == "__main__":
    main()
