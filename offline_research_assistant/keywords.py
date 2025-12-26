from __future__ import annotations

from collections import Counter
from typing import List, Tuple

from .text_utils import tokenize_words, remove_stopwords


def extract_keywords(text: str, top_k: int = 15) -> List[Tuple[str, int]]:
    """Offline keyword extraction using simple frequency with stopword removal."""

    words = remove_stopwords(tokenize_words(text))
    counts = Counter(words)
    return counts.most_common(top_k)


def extract_keyphrases(text: str, top_k: int = 10) -> List[str]:
    """Very lightweight keyphrase extraction.

    Groups consecutive non-stopwords into phrases (mini-RAKE style).
    """

    raw = tokenize_words(text)
    phrases: list[str] = []
    buff: list[str] = []

    stop = set()
    # reuse stopwords from remove_stopwords by filtering; create quickly
    filtered = set(remove_stopwords(raw))
    stop = set(raw) - filtered

    for w in raw:
        if w in stop or len(w) <= 2:
            if buff:
                phrases.append(" ".join(buff))
                buff = []
        else:
            buff.append(w)

    if buff:
        phrases.append(" ".join(buff))

    counts = Counter(phrases)
    return [p for p, _ in counts.most_common(top_k) if len(p.split()) >= 2]
