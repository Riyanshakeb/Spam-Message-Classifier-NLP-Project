"""
preprocessing.py
----------------
Text preprocessing pipeline for the Spam Message Classifier.

Steps:
    1. Lowercase conversion
    2. URL & email removal
    3. Number removal
    4. Punctuation stripping
    5. Tokenization
    6. Stop-word removal
    7. Optional: stemming / lemmatization
"""

import re
import string
from typing import List, Optional

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# ---------------------------------------------------------------------------
# Ensure required NLTK data is present (auto-download if missing)
# ---------------------------------------------------------------------------

_NLTK_RESOURCES = {
    "tokenizers/punkt": "punkt",
    "tokenizers/punkt_tab": "punkt_tab",
    "corpora/stopwords": "stopwords",
    "corpora/wordnet": "wordnet",
}

for _path, _pkg in _NLTK_RESOURCES.items():
    try:
        nltk.data.find(_path)
    except LookupError:
        nltk.download(_pkg, quiet=True)

# ---------------------------------------------------------------------------
# Module-level singletons (instantiated once)
# ---------------------------------------------------------------------------

_STOP_WORDS: set = set(stopwords.words("english"))
_STEMMER = PorterStemmer()
_LEMMATIZER = WordNetLemmatizer()


# ---------------------------------------------------------------------------
# Core preprocessing function
# ---------------------------------------------------------------------------


def preprocess_text(
    text: str,
    *,
    remove_stopwords: bool = True,
    use_stemming: bool = False,
    use_lemmatization: bool = True,
) -> str:
    """
    Full text preprocessing pipeline.

    Parameters
    ----------
    text : str
        Raw input text / SMS message.
    remove_stopwords : bool
        Remove English stop-words (default: True).
    use_stemming : bool
        Apply Porter Stemming (default: False).
    use_lemmatization : bool
        Apply WordNet Lemmatization (default: True).
        Ignored when *use_stemming* is True.

    Returns
    -------
    str
        Cleaned, tokenized (space-joined) string.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs and email addresses
    text = re.sub(r"http\S+|www\.\S+", " url ", text)
    text = re.sub(r"\S+@\S+", " email ", text)

    # 3. Replace phone numbers (sequences of digits with dashes / spaces)
    text = re.sub(r"\b\d[\d\s\-]{5,}\d\b", " phonenumber ", text)

    # 4. Replace standalone numbers
    text = re.sub(r"\b\d+\b", " num ", text)

    # 5. Remove punctuation (keep the placeholder words from steps 2-4)
    text = text.translate(str.maketrans("", "", string.punctuation))

    # 6. Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 7. Tokenize
    tokens: List[str] = word_tokenize(text)

    # 8. Remove stop-words and very short tokens (len <= 1)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]
    else:
        tokens = [t for t in tokens if len(t) > 1]

    # 9. Stemming or Lemmatization
    if use_stemming:
        tokens = [_STEMMER.stem(t) for t in tokens]
    elif use_lemmatization:
        tokens = [_LEMMATIZER.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def preprocess_series(
    series,
    *,
    remove_stopwords: bool = True,
    use_stemming: bool = False,
    use_lemmatization: bool = True,
) -> "pd.Series":
    """
    Apply :func:`preprocess_text` to a pandas Series of messages.

    Parameters
    ----------
    series : pd.Series
        Raw text series.
    remove_stopwords : bool
        Remove English stop-words.
    use_stemming : bool
        Apply Porter Stemming.
    use_lemmatization : bool
        Apply WordNet Lemmatization.

    Returns
    -------
    pd.Series
        Preprocessed text series.
    """
    return series.map(
        lambda t: preprocess_text(
            t,
            remove_stopwords=remove_stopwords,
            use_stemming=use_stemming,
            use_lemmatization=use_lemmatization,
        )
    )


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    samples = [
        "WINNER!! You have been selected to receive a $1000 prize. Call 0800-123-4567 NOW!",
        "Hey, are you free tomorrow at 3pm for lunch?",
        "URGENT: Click http://fakesite.com to claim your FREE gift today!!!",
        "Don't forget to bring the files to the office meeting.",
    ]
    print("=" * 60)
    print("Preprocessing Demo")
    print("=" * 60)
    for msg in samples:
        cleaned = preprocess_text(msg)
        print(f"\nOriginal : {msg}")
        print(f"Cleaned  : {cleaned}")
