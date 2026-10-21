"""
Spam Message Classifier — Flask Web App
Vercel-compatible entrypoint (app.py)
"""

import os
import json
import joblib
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string

# ── preprocessing import (same pipeline as CLI) ──────────────────────────────
from preprocessing import preprocess_text

# ── Flask setup ───────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── Model loading (lazy, cached) ──────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "best_model_pipeline.joblib"
META_PATH  = BASE_DIR / "models" / "model_metadata.json"

_pipeline = None
_metadata = {}

def get_pipeline():
    global _pipeline, _metadata
    if _pipeline is None:
        _pipeline = joblib.load(MODEL_PATH)
        if META_PATH.exists():
            with open(META_PATH) as f:
                _metadata = json.load(f)
    return _pipeline

# ── HTML template ─────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Spam Message Classifier</title>
<style>
  :root {
    --bg: #0f172a; --card: #1e293b; --accent: #6366f1;
    --spam: #ef4444; --ham: #22c55e; --text: #f1f5f9;
    --muted: #94a3b8; --border: #334155;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: var(--bg); color: var(--text);
    min-height: 100vh; display: flex; flex-direction: column;
    align-items: center; padding: 2rem 1rem;
  }
  header { text-align: center; margin-bottom: 2rem; }
  header h1 { font-size: 2rem; font-weight: 700; letter-spacing: -0.5px; }
  header h1 span { color: var(--accent); }
  header p { color: var(--muted); margin-top: .4rem; font-size: .95rem; }
  .card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 1rem; padding: 2rem; width: 100%; max-width: 680px;
    box-shadow: 0 20px 60px rgba(0,0,0,.4);
  }
  textarea {
    width: 100%; height: 130px; background: var(--bg);
    border: 1px solid var(--border); border-radius: .6rem;
    color: var(--text); font-size: 1rem; padding: .85rem 1rem;
    resize: vertical; outline: none; transition: border-color .2s;
  }
  textarea:focus { border-color: var(--accent); }
  textarea::placeholder { color: var(--muted); }
  .btn {
    margin-top: 1rem; width: 100%; padding: .85rem;
    background: var(--accent); color: #fff; border: none;
    border-radius: .6rem; font-size: 1rem; font-weight: 600;
    cursor: pointer; transition: opacity .2s;
  }
  .btn:hover { opacity: .85; }
  .btn:disabled { opacity: .5; cursor: not-allowed; }
  #result { margin-top: 1.5rem; display: none; }
  .verdict {
    display: flex; align-items: center; gap: .75rem;
    font-size: 1.5rem; font-weight: 700; margin-bottom: 1rem;
  }
  .badge {
    padding: .25rem .75rem; border-radius: 999px;
    font-size: .85rem; font-weight: 700; letter-spacing: .5px;
  }
  .badge-spam { background: rgba(239,68,68,.15); color: var(--spam); border: 1px solid var(--spam); }
  .badge-ham  { background: rgba(34,197,94,.15);  color: var(--ham);  border: 1px solid var(--ham);  }
  .bar-wrap { margin-bottom: .75rem; }
  .bar-label {
    display: flex; justify-content: space-between;
    font-size: .85rem; color: var(--muted); margin-bottom: .3rem;
  }
  .bar-track {
    height: 10px; background: var(--bg); border-radius: 999px; overflow: hidden;
  }
  .bar-fill {
    height: 100%; border-radius: 999px; transition: width .6s ease;
  }
  .fill-spam { background: var(--spam); }
  .fill-ham  { background: var(--ham); }
  .cleaned {
    margin-top: 1rem; padding: .75rem 1rem;
    background: var(--bg); border-radius: .5rem;
    font-size: .82rem; color: var(--muted); word-break: break-word;
  }
  .cleaned strong { color: var(--text); }
  .samples { margin-top: 1.5rem; }
  .samples h3 { font-size: .85rem; color: var(--muted); margin-bottom: .6rem; text-transform: uppercase; letter-spacing: .5px; }
  .chip-row { display: flex; flex-wrap: wrap; gap: .5rem; }
  .chip {
    padding: .35rem .8rem; border-radius: .4rem; font-size: .8rem;
    cursor: pointer; border: 1px solid var(--border);
    background: var(--bg); color: var(--text); transition: border-color .2s;
  }
  .chip:hover { border-color: var(--accent); }
  .chip.spam-chip { border-color: rgba(239,68,68,.4); }
  .chip.ham-chip  { border-color: rgba(34,197,94,.4); }
  .metrics {
    margin-top: 2rem; border-top: 1px solid var(--border); padding-top: 1.5rem;
    display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 1rem;
  }
  .metric { text-align: center; }
  .metric .val { font-size: 1.4rem; font-weight: 700; color: var(--accent); }
  .metric .lbl { font-size: .75rem; color: var(--muted); margin-top: .2rem; }
  footer { margin-top: 2rem; color: var(--muted); font-size: .8rem; text-align: center; }
  footer a { color: var(--accent); text-decoration: none; }
</style>
</head>
<body>
<header>
  <h1>📩 Spam <span>Message</span> Classifier</h1>
  <p>Logistic Regression + TF-IDF · UCI SMS Spam Collection · 98.29% accuracy</p>
</header>

<div class="card">
  <textarea id="msg" placeholder="Type or paste a message to classify…"></textarea>
  <button class="btn" id="btn" onclick="classify()">Classify Message</button>

  <div id="result">
    <div class="verdict">
      <span id="verdict-icon"></span>
      <span id="verdict-text"></span>
      <span id="verdict-badge" class="badge"></span>
    </div>
    <div class="bar-wrap">
      <div class="bar-label"><span>🚫 Spam</span><span id="spam-pct"></span></div>
      <div class="bar-track"><div id="spam-bar" class="bar-fill fill-spam" style="width:0%"></div></div>
    </div>
    <div class="bar-wrap">
      <div class="bar-label"><span>✅ Ham</span><span id="ham-pct"></span></div>
      <div class="bar-track"><div id="ham-bar" class="bar-fill fill-ham" style="width:0%"></div></div>
    </div>
    <div class="cleaned">
      <strong>Preprocessed tokens:</strong> <span id="cleaned-text"></span>
    </div>
  </div>

  <div class="samples">
    <h3>Try a sample</h3>
    <div class="chip-row">
      <div class="chip spam-chip" onclick="use(this)">WINNER!! You've won a FREE iPhone. Click NOW!</div>
      <div class="chip spam-chip" onclick="use(this)">Congratulations! You've won £500 cash. Call 0845-123-4567</div>
      <div class="chip spam-chip" onclick="use(this)">FREE entry! Text WIN to 88800 to enter our prize draw.</div>
      <div class="chip ham-chip"  onclick="use(this)">Hey, are you joining us for dinner tonight at 7pm?</div>
      <div class="chip ham-chip"  onclick="use(this)">Can you send me the meeting notes from yesterday?</div>
      <div class="chip ham-chip"  onclick="use(this)">Happy birthday! Wishing you a wonderful day.</div>
    </div>
  </div>

  <div class="metrics">
    <div class="metric"><div class="val">98.29%</div><div class="lbl">Accuracy</div></div>
    <div class="metric"><div class="val">92.21%</div><div class="lbl">Precision</div></div>
    <div class="metric"><div class="val">95.30%</div><div class="lbl">Recall</div></div>
    <div class="metric"><div class="val">93.73%</div><div class="lbl">F1-Score</div></div>
    <div class="metric"><div class="val">99.16%</div><div class="lbl">ROC-AUC</div></div>
  </div>
</div>

<footer>
  <a href="https://github.com/Riyanshakeb/Spam-Message-Classifier-NLP-Project" target="_blank">GitHub</a>
  &nbsp;·&nbsp; Logistic Regression · TF-IDF (10k features, 1–2 ngrams) · NLTK preprocessing
</footer>

<script>
function use(el) {
  document.getElementById('msg').value = el.innerText;
  classify();
}
async function classify() {
  const msg = document.getElementById('msg').value.trim();
  if (!msg) return;
  const btn = document.getElementById('btn');
  btn.disabled = true; btn.textContent = 'Classifying…';
  try {
    const res  = await fetch('/predict', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: msg})
    });
    const data = await res.json();
    if (data.error) { alert(data.error); return; }
    showResult(data);
  } finally {
    btn.disabled = false; btn.textContent = 'Classify Message';
  }
}
function showResult(d) {
  const isSpam = d.label === 'spam';
  document.getElementById('verdict-icon').textContent = isSpam ? '🚫' : '✅';
  document.getElementById('verdict-text').textContent = isSpam ? 'This looks like SPAM' : 'This looks like a normal message';
  const badge = document.getElementById('verdict-badge');
  badge.textContent = d.label.toUpperCase();
  badge.className = 'badge ' + (isSpam ? 'badge-spam' : 'badge-ham');
  const sp = (d.spam_probability * 100).toFixed(1);
  const hp = (d.ham_probability  * 100).toFixed(1);
  document.getElementById('spam-pct').textContent = sp + '%';
  document.getElementById('ham-pct').textContent  = hp + '%';
  document.getElementById('spam-bar').style.width = sp + '%';
  document.getElementById('ham-bar').style.width  = hp + '%';
  document.getElementById('cleaned-text').textContent = d.cleaned_text || '—';
  document.getElementById('result').style.display = 'block';
}
</script>
</body>
</html>"""

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True, silent=True) or {}
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"error": "No message provided"}), 400

    try:
        pipeline = get_pipeline()
        cleaned  = preprocess_text(message)
        proba    = pipeline.predict_proba([cleaned])[0]

        # pipeline classes: [ham=0, spam=1]
        classes     = list(pipeline.classes_)
        spam_idx    = classes.index(1)
        ham_idx     = classes.index(0)
        spam_prob   = float(proba[spam_idx])
        ham_prob    = float(proba[ham_idx])
        label       = "spam" if spam_prob >= 0.5 else "ham"

        return jsonify({
            "label":            label,
            "spam_probability": spam_prob,
            "ham_probability":  ham_prob,
            "confidence":       max(spam_prob, ham_prob),
            "cleaned_text":     cleaned,
            "model":            _metadata.get("best_model", "Logistic Regression"),
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": _pipeline is not None})


# ── Dev server ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)
