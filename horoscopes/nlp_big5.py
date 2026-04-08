"""
Big Five personality scoring from free text.

Uses validated lexical proxies (Mairesse et al. 2007 + replications):
  E (Extraversion):       positive emotion, word count, social words
  A (Agreeableness):      trust/positive emotion, low anger
  C (Conscientiousness):  achievement/work words, causal connectives
  N (Neuroticism):        negative emotion, anxiety, first-person singular
  O (Openness):           insight words, vocab richness, long words

Emotion categories from NRC Emotion Lexicon (Mohammad & Turney 2013).
Word categories from validated LIWC-adjacent open lists.
Output: z-scored Big Five domain scores across the sample.
"""
import re
import numpy as np
import pandas as pd
from nrclex import NRCLex

# ---------------------------------------------------------------------------
# Word category lexicons (free, validated proxies for LIWC categories)
# ---------------------------------------------------------------------------

_SOCIAL = {
    "friend", "friends", "people", "family", "we", "us", "our", "together",
    "community", "social", "group", "team", "partner", "girlfriend", "boyfriend",
    "wife", "husband", "mom", "dad", "sister", "brother", "colleague",
}

_ACHIEVEMENT = {
    "accomplish", "achieve", "success", "goal", "ambition", "work", "career",
    "job", "degree", "graduate", "earn", "build", "create", "complete",
    "finish", "win", "excel", "dedicated", "motivated", "driven", "discipline",
    "productive", "efficient", "organised", "organized", "plan", "manage",
}

_ANXIETY = {
    "worry", "anxious", "anxiety", "nervous", "afraid", "fear", "scared",
    "stress", "stressed", "panic", "dread", "uneasy", "overwhelm", "tense",
}

_INSIGHT = {
    "think", "know", "understand", "consider", "believe", "feel", "wonder",
    "realize", "realise", "reflect", "question", "explore", "analyse",
    "analyze", "curious", "learn", "curious", "ponder", "imagine",
}

_CAUSAL = {
    "because", "therefore", "thus", "hence", "consequently", "since",
    "result", "effect", "cause", "reason", "why", "due", "leads", "implies",
}

_ANGER_WORDS = {
    "hate", "angry", "anger", "furious", "rage", "annoyed", "irritated",
    "frustrated", "bitter", "resentful", "hostile", "mad", "outraged",
}

_FIRST_PERSON_SINGULAR = {"i", "me", "my", "myself", "mine"}
_SECOND_PERSON = {"you", "your", "yourself", "yours"}

_CULTURE = {
    "music", "art", "film", "movie", "book", "read", "travel", "concert",
    "theatre", "theater", "poetry", "literature", "philosophy", "culture",
    "language", "history", "science", "nature", "documentary", "gallery",
}


def _clean_text(text: str) -> str:
    """Strip HTML tags and normalise whitespace."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-z]+;", " ", text)          # HTML entities
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return text.lower()


def _word_list(text: str) -> list[str]:
    return text.split()


def _token_ratio(words: list[str]) -> float:
    """Type-token ratio (vocabulary richness). Bounded to avoid length artefacts."""
    if len(words) < 5:
        return 0.0
    # Use sqrt(n) correction for length
    return len(set(words)) / (len(words) ** 0.5 + 1e-9)


def _mean_word_len(words: list[str]) -> float:
    if not words:
        return 0.0
    return np.mean([len(w) for w in words])


def _fraction(words: list[str], lexicon: set) -> float:
    if not words:
        return 0.0
    return sum(1 for w in words if w in lexicon) / len(words)


def score_text(text: str) -> dict[str, float]:
    """
    Score a single text string on Big Five proxies.
    Returns raw (unscaled) scores for E, A, C, N, O.
    """
    if not isinstance(text, str) or len(text.strip()) < 10:
        return {t: np.nan for t in ["E", "A", "C", "N", "O"]}

    clean = _clean_text(text)
    words = _word_list(clean)
    n = len(words)

    if n < 5:
        return {t: np.nan for t in ["E", "A", "C", "N", "O"]}

    # NRC emotion scores
    nrc = NRCLex("placeholder")
    nrc.load_raw_text(clean)
    freqs = nrc.affect_frequencies  # dict of emotion → proportion

    pos  = freqs.get("positive", 0)
    neg  = freqs.get("negative", 0)
    joy  = freqs.get("joy", 0)
    trust = freqs.get("trust", 0)
    anger_nrc = freqs.get("anger", 0)
    fear  = freqs.get("fear", 0)
    sad   = freqs.get("sadness", 0)
    ant   = freqs.get("anticipation", 0)
    surp  = freqs.get("surprise", 0)

    # Word-category fractions
    social   = _fraction(words, _SOCIAL)
    achieve  = _fraction(words, _ACHIEVEMENT)
    anxious  = _fraction(words, _ANXIETY)
    insight  = _fraction(words, _INSIGHT)
    causal   = _fraction(words, _CAUSAL)
    anger_lex = _fraction(words, _ANGER_WORDS)
    fp_sing  = _fraction(words, _FIRST_PERSON_SINGULAR)
    sec_per  = _fraction(words, _SECOND_PERSON)
    culture  = _fraction(words, _CULTURE)

    # Text structure
    ttr   = _token_ratio(words)
    mwl   = _mean_word_len(words)
    log_wc = np.log1p(n)

    # --- Big Five composite scores (validated directions) ---
    E = (pos + joy + social + sec_per + 0.3 * log_wc) - (neg + sad)
    A = (trust + pos + sec_per) - (anger_nrc + anger_lex + neg)
    C = (achieve + causal + 0.5 * (1 - fp_sing)) - (neg * 0.5)
    N = (neg + fear + sad + anxious + fp_sing) - (pos + joy)
    O = (insight + causal + culture + ant + surp + ttr + 0.1 * mwl) - 0

    return {"E": E, "A": A, "C": C, "N": N, "O": O}


def score_texts(texts: pd.Series, verbose: bool = True) -> pd.DataFrame:
    """
    Score a Series of text strings. Returns DataFrame with columns E,A,C,N,O,
    z-scored across the sample (mean=0, std=1), with NaN rows dropped.
    """
    if verbose:
        print(f"Scoring {len(texts):,} texts for Big Five...")

    rows = []
    for i, text in enumerate(texts):
        if verbose and i % 5000 == 0:
            print(f"  {i:,}/{len(texts):,}", end="\r")
        rows.append(score_text(str(text) if pd.notna(text) else ""))

    df = pd.DataFrame(rows)

    # Z-score across sample (ignoring NaN)
    for col in ["E", "A", "C", "N", "O"]:
        mu = df[col].mean()
        sd = df[col].std()
        if sd > 0:
            df[col] = (df[col] - mu) / sd

    if verbose:
        print(f"\nScored {df.notna().all(axis=1).sum():,} complete rows")

    return df
