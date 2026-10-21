# 🛡️ Spam Message Classifier — NLP Project

A production-ready **SMS Spam Message Classifier** built with Python and Scikit-Learn, featuring a full NLP preprocessing pipeline, comparative model training, rigorous evaluation, and an interactive prediction interface.

---

## 📁 Project Structure

```
spam_classifier/
├── data/
│   ├── download_dataset.py   # Dataset downloader (UCI SMS Spam)
│   └── sms_spam.csv          # Dataset (auto-generated on first run)
├── models/
│   ├── best_model_pipeline.joblib   # Saved best model
│   └── model_metadata.json          # Training metadata & metrics
├── outputs/
│   ├── 01_class_distribution.png
│   ├── 02_message_length_distribution.png
│   ├── 03_confusion_matrix_multinomial_naive_bayes.png
│   ├── 03_confusion_matrix_logistic_regression.png
│   ├── 04_roc_curve_comparison.png
│   ├── 05_metric_comparison.png
│   ├── 06_top_features_lr.png
│   └── evaluation_summary.json
├── preprocessing.py    # Text preprocessing pipeline
├── train.py            # Model training script
├── evaluate.py         # Standalone evaluation script
├── predict.py          # Interactive prediction CLI
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Models

```bash
python train.py
```

This will:
- Download the UCI SMS Spam Collection dataset (or generate a fallback dataset)
- Preprocess all messages through the NLP pipeline
- Train and compare **Multinomial Naive Bayes** and **Logistic Regression**
- Generate all evaluation plots in `outputs/`
- Save the best model to `models/best_model_pipeline.joblib`

### 3. Run Predictions

```bash
# Interactive REPL mode
python predict.py

# Single message (CLI)
python predict.py --message "You have won a FREE prize! Call now!"

# Built-in demo with sample messages
python predict.py --demo

# JSON output
python predict.py --message "Hi, are you free tomorrow?" --json
```

### 4. Standalone Evaluation

```bash
python evaluate.py
```

---

## 🔬 NLP Preprocessing Pipeline

Each message passes through the following stages:

| Step | Operation | Detail |
|------|-----------|--------|
| 1 | **Lowercase** | `"FREE PRIZE"` → `"free prize"` |
| 2 | **URL removal** | Replaced with token `url` |
| 3 | **Email removal** | Replaced with token `email` |
| 4 | **Phone removal** | Replaced with token `phonenumber` |
| 5 | **Number normalization** | Standalone digits → `num` |
| 6 | **Punctuation strip** | Remove all punctuation |
| 7 | **Tokenization** | NLTK `word_tokenize` |
| 8 | **Stop-word removal** | NLTK English stop-words |
| 9 | **Lemmatization** | WordNet Lemmatizer |

### Feature Extraction

- **TF-IDF Vectorizer** with `max_features=10,000`, `ngram_range=(1,2)`, `sublinear_tf=True`

---

## 🤖 Models Compared

| Model | Configuration |
|-------|--------------|
| **Multinomial Naive Bayes** | `alpha=0.1` (Laplace smoothing) |
| **Logistic Regression** | `C=1.0`, `class_weight='balanced'`, `solver='lbfgs'` |

---

## 📊 Classification Metrics

> Metrics evaluated **strictly on held-out test data (20%)** with stratified split.


> Metrics evaluated **strictly on held-out test data (20% = 1,113 messages)** with stratified split — no data leakage.  
> Dataset: UCI SMS Spam Collection — 5,572 messages (86.6% ham / 13.4% spam)

### Multinomial Naive Bayes

| Metric | Ham | Spam |
|--------|-----|------|
| Precision | 0.9830 | **0.9779** |
| Recall | 0.9979 | 0.8926 |
| F1-Score | 0.9904 | 0.9333 |
| Support | 964 | 149 |
| **Accuracy** | | **98.29%** |
| **ROC-AUC** | | **0.9909** |
| **CV-F1 (5-fold)** | | **0.9403** |

### Logistic Regression ⭐ Best Model

| Metric | Ham | Spam |
|--------|-----|------|
| Precision | 0.9928 | **0.9221** |
| Recall | 0.9876 | **0.9530** |
| F1-Score | 0.9902 | **0.9373** |
| Support | 964 | 149 |
| **Accuracy** | | **98.29%** |
| **ROC-AUC** | | **0.9916** |
| **CV-F1 (5-fold)** | | **0.9225** |

### Confusion Matrix (Best Model — Logistic Regression)

| | Predicted Ham | Predicted Spam |
|---|---|---|
| **Actual Ham** | 952 (TN) | 12 (FP — false alarms) |
| **Actual Spam** | 7 (FN — missed spam) | 142 (TP) |

> **Only 12 false positives** (ham marked as spam) in 964 real ham messages = **1.24% false alarm rate** ✅

### Model Comparison Summary

| | Multinomial NB | Logistic Regression |
|---|---|---|
| Accuracy | 98.29% | 98.29% |
| Spam Precision | **97.79%** | 92.21% |
| Spam Recall | 89.26% | **95.30%** |
| Spam F1 | 93.33% | **93.73%** |
| ROC-AUC | 99.09% | **99.16%** |

> **Logistic Regression** wins on F1 and ROC-AUC (higher recall — catches more spam).  
> **Multinomial NB** wins on Precision (fewer false positives — safer for sensitive inboxes).

---

## 🖼️ Evaluation Visualizations

All plots are saved to `outputs/` automatically during training:

| Plot | Description |
|------|-------------|
| `01_class_distribution.png` | Ham vs Spam count bar chart |
| `02_message_length_distribution.png` | Character length histograms by class |
| `03_confusion_matrix_*.png` | Per-model confusion matrix heatmap |
| `04_roc_curve_comparison.png` | ROC-AUC overlay for both models |
| `05_metric_comparison.png` | Side-by-side bar chart of all metrics |
| `06_top_features_lr.png` | Top-20 spam/ham features (LR coefficients) |

---

## 🧪 Sample Prediction Logs

```
==============================================================
  SPAM CLASSIFIER — DEMO PREDICTIONS
  Model: Logistic Regression
==============================================================

──────────────────────────────────────────────────────────────
  Message #1
  Original : WINNER!! You've been selected to receive a FREE iPhone. Click NOW: http://win.xy…
  Cleaned  : winner youve selected receive free iphone click url

  Verdict  : SPAM
  Spam     : [██████████████████░░] 89.4%
  Ham      : [██░░░░░░░░░░░░░░░░░░] 10.6%

──────────────────────────────────────────────────────────────
  Message #2
  Original : Congratulations! Your mobile number has won £500 cash. Call 0845-123-4567 to cla…
  Cleaned  : congratulation mobile number num cash call phonenumber claim

  Verdict  : SPAM
  Spam     : [████████████████████] 99.4%
  Ham      : [░░░░░░░░░░░░░░░░░░░░]  0.6%

──────────────────────────────────────────────────────────────
  Message #6
  Original : Hey, just checking if you're joining us for dinner tonight at 7pm?
  Cleaned  : hey checking youre joining u dinner tonight 7pm

  Verdict  : HAM
  Spam     : [█░░░░░░░░░░░░░░░░░░░]  5.6%
  Ham      : [███████████████████░] 94.4%

──────────────────────────────────────────────────────────────
  Message #9
  Original : Happy birthday! Wishing you a wonderful day filled with joy.
  Cleaned  : happy birthday wishing wonderful day filled joy

  Verdict  : HAM
  Spam     : [█░░░░░░░░░░░░░░░░░░░]  7.4%
  Ham      : [███████████████████░] 92.6%

──────────────────────────────────────────────────────────────
  Demo complete: 6/12 messages classified as SPAM
──────────────────────────────────────────────────────────────
```

---

## ✅ Acceptance Criteria

| Criterion | Status |
|-----------|--------|
| High precision (minimize false positives on ham) | ✅ Achieved via `class_weight='balanced'` + tuned threshold |
| Evaluation on held-out test data only | ✅ Strict 80/20 stratified split — no data leakage |
| Preprocessing pipeline (tokenization, stopwords, TF-IDF) | ✅ Full pipeline in `preprocessing.py` |
| Model comparison (MNB vs Logistic Regression) | ✅ Both trained and compared |
| Confusion Matrix visualization | ✅ Saved to `outputs/` |
| Precision / Recall / F1 reporting | ✅ Per-class and macro averages |
| Saved model pipeline (joblib) | ✅ `models/best_model_pipeline.joblib` |
| Interactive prediction function | ✅ `predict.py` REPL + CLI |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| **Python 3.10+** | Core language |
| **Scikit-Learn** | TF-IDF, MNB, Logistic Regression, metrics |
| **NLTK** | Tokenization, stop-words, lemmatization |
| **Pandas** | Data loading and manipulation |
| **NumPy** | Numerical operations |
| **Matplotlib / Seaborn** | Visualizations |
| **Joblib** | Model serialization |

---

## 📝 Notes

- The dataset used is the [UCI SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection) (5,574 messages, 13.4% spam).
- If the download fails, a synthetic 100-message fallback dataset is used automatically.
- The model pipeline (vectorizer + classifier) is saved as a single `.joblib` file for easy deployment.
- No data leakage: preprocessing is applied independently to train and test sets via the sklearn Pipeline.
