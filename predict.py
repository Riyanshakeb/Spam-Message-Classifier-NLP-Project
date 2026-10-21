"""
predict.py
----------
Interactive prediction module for the Spam Message Classifier.

Usage
-----
  python predict.py                        # interactive REPL mode
  python predict.py --message "Your msg"   # single message CLI flag
  python predict.py --demo                 # run built-in demo predictions
"""

import sys
import json
import argparse
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).parent.resolve()
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "best_model_pipeline.joblib"

sys.path.insert(0, str(BASE_DIR))
from preprocessing import preprocess_text  # noqa: E402


# ---------------------------------------------------------------------------
# Colour helpers (ANSI codes for terminals that support them)
# ---------------------------------------------------------------------------

_RED = "\033[91m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_CYAN = "\033[96m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _color(text: str, code: str) -> str:
    return f"{code}{text}{_RESET}"


# ---------------------------------------------------------------------------
# Classifier wrapper
# ---------------------------------------------------------------------------


class SpamClassifier:
    """
    Thin wrapper around the saved sklearn Pipeline for inference.

    Parameters
    ----------
    model_path : str | Path
        Path to the joblib-saved pipeline.
    """

    def __init__(self, model_path: str | Path = MODEL_PATH):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at: {model_path}\n"
                "Run `python train.py` first to train and save the model."
            )
        self.pipeline = joblib.load(model_path)
        self._load_metadata()

    def _load_metadata(self) -> None:
        meta_path = MODELS_DIR / "model_metadata.json"
        if meta_path.exists():
            with open(meta_path) as fp:
                self.metadata = json.load(fp)
            self.model_name = self.metadata.get("best_model", "Unknown")
        else:
            self.metadata = {}
            self.model_name = "Unknown"

    # ------------------------------------------------------------------

    def predict(self, message: str) -> dict:
        """
        Classify a single message.

        Parameters
        ----------
        message : str
            Raw SMS / text message to classify.

        Returns
        -------
        dict with keys:
            message      : original message
            cleaned      : preprocessed text
            label        : 'spam' | 'ham'
            confidence   : float in [0, 1] (probability of predicted class)
            spam_prob    : float — raw probability of SPAM
            ham_prob     : float — raw probability of HAM
        """
        cleaned = preprocess_text(message)
        if not cleaned.strip():
            return {
                "message": message,
                "cleaned": cleaned,
                "label": "ham",
                "confidence": 1.0,
                "spam_prob": 0.0,
                "ham_prob": 1.0,
                "note": "Empty after preprocessing — defaulting to ham",
            }

        probs = self.pipeline.predict_proba([cleaned])[0]
        spam_prob = float(probs[1])
        ham_prob = float(probs[0])
        label = "spam" if spam_prob >= 0.5 else "ham"
        confidence = spam_prob if label == "spam" else ham_prob

        return {
            "message": message,
            "cleaned": cleaned,
            "label": label,
            "confidence": confidence,
            "spam_prob": spam_prob,
            "ham_prob": ham_prob,
        }

    def predict_batch(self, messages: list[str]) -> list[dict]:
        """Classify multiple messages."""
        return [self.predict(m) for m in messages]


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------


def _confidence_bar(prob: float, width: int = 20) -> str:
    filled = round(prob * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {prob:.1%}"


def print_prediction(result: dict, idx: int | None = None) -> None:
    label = result["label"]
    label_color = _RED if label == "spam" else _GREEN
    header = f"  Message #{idx}" if idx is not None else "  Message"

    print()
    print(_color("─" * 62, _CYAN))
    print(f"{_BOLD}{header}{_RESET}")
    print(f"  Original : {result['message'][:80]}{'…' if len(result['message']) > 80 else ''}")
    print(f"  Cleaned  : {result['cleaned'][:80]}{'…' if len(result['cleaned']) > 80 else ''}")
    print()
    print(f"  Verdict  : {_color(_BOLD + label.upper(), label_color)}")
    print(f"  Spam     : {_confidence_bar(result['spam_prob'])}")
    print(f"  Ham      : {_confidence_bar(result['ham_prob'])}")
    if "note" in result:
        print(f"  Note     : {result['note']}")


# ---------------------------------------------------------------------------
# Demo predictions
# ---------------------------------------------------------------------------

DEMO_MESSAGES = [
    # --- Clear SPAM ---
    "WINNER!! You've been selected to receive a FREE iPhone. Click NOW: http://win.xyz",
    "Congratulations! Your mobile number has won £500 cash. Call 0845-123-4567 to claim.",
    "URGENT: Your bank account has been locked. Verify at www.secure-bank-verify.com ASAP!",
    "FREE entry! Text WIN to 88800 to enter our weekly prize draw. T&C apply.",
    "You have won a 2-week holiday for 2 to Majorca! To claim call 09061743806.",

    # --- Clear HAM ---
    "Hey, just checking if you're joining us for dinner tonight at 7pm?",
    "Can you please send me the meeting notes from yesterday's session?",
    "The doctor's appointment is confirmed for Wednesday at 10:30 AM.",
    "Happy birthday! Wishing you a wonderful day filled with joy.",
    "Running late, will be there in 15 minutes. Sorry!",

    # --- Edge cases ---
    "Your account statement is ready. Login to view it.",
    "Claim your reward points before they expire next month.",
]


def run_demo(classifier: SpamClassifier) -> None:
    print()
    print(_color("=" * 62, _BOLD))
    print(_color("  SPAM CLASSIFIER — DEMO PREDICTIONS", _BOLD))
    print(f"  Model: {classifier.model_name}")
    print(_color("=" * 62, _BOLD))

    spam_count = 0
    for i, msg in enumerate(DEMO_MESSAGES, 1):
        result = classifier.predict(msg)
        print_prediction(result, idx=i)
        if result["label"] == "spam":
            spam_count += 1

    print()
    print(_color("─" * 62, _CYAN))
    print(f"  Demo complete: {spam_count}/{len(DEMO_MESSAGES)} messages classified as SPAM")
    print(_color("─" * 62, _CYAN))


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------


def run_interactive(classifier: SpamClassifier) -> None:
    print()
    print(_color("=" * 62, _BOLD))
    print(_color("  SPAM CLASSIFIER — INTERACTIVE MODE", _BOLD))
    print(f"  Model: {classifier.model_name}")
    print(_color("=" * 62, _BOLD))
    print(f"\n  Type a message and press {_BOLD}Enter{_RESET} to classify it.")
    print(f"  Type {_BOLD}'demo'{_RESET} to run sample predictions.")
    print(f"  Type {_BOLD}'quit'{_RESET} or {_BOLD}'exit'{_RESET} to leave.\n")

    while True:
        try:
            user_input = input(_color("  > ", _CYAN)).strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n\n{_color('Goodbye!', _GREEN)}\n")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            print(f"\n{_color('Goodbye!', _GREEN)}\n")
            break
        if user_input.lower() == "demo":
            run_demo(classifier)
            continue

        result = classifier.predict(user_input)
        print_prediction(result)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Spam Message Classifier — Inference Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py
  python predict.py --demo
  python predict.py --message "You have won a FREE prize!"
  python predict.py --message "See you at lunch tomorrow" --json
        """,
    )
    parser.add_argument("--message", "-m", type=str, default=None,
                        help="Single message to classify")
    parser.add_argument("--demo", action="store_true",
                        help="Run built-in demo predictions and exit")
    parser.add_argument("--model", type=str, default=str(MODEL_PATH),
                        help="Path to saved model pipeline (.joblib)")
    parser.add_argument("--json", action="store_true",
                        help="Output result as JSON (for --message mode)")
    args = parser.parse_args()

    # Load model
    try:
        classifier = SpamClassifier(model_path=args.model)
    except FileNotFoundError as e:
        print(f"\n{_color('[ERROR]', _RED)} {e}\n")
        sys.exit(1)

    if args.demo:
        run_demo(classifier)
        return

    if args.message:
        result = classifier.predict(args.message)
        if args.json:
            # Remove non-serialisable keys
            output = {k: v for k, v in result.items() if k != "pipeline"}
            print(json.dumps(output, indent=2))
        else:
            print_prediction(result)
        return

    # Default: interactive
    run_interactive(classifier)


if __name__ == "__main__":
    main()
