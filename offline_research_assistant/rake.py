from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .text_utils import tokenize_words, remove_stopwords


@dataclass(frozen=True)
class RakePhrase:
    phrase: str
    score: float


def extract_rake_phrases(text: str, top_k: int = 15) -> List[RakePhrase]:
    """Lightweight RAKE implementation (offline, no dependencies).

    Steps:
    - split into candidate phrases by stopwords
    - score words by degree/frequency
    - score phrases as sum(word scores)
    """

    tokens = tokenize_words(text)
    filtered = set(remove_stopwords(tokens))
    stopwords = set(tokens) - filtered

    # candidate phrases
    phrases: List[List[str]] = []
    buff: List[str] = []
    for w in tokens:
        if w in stopwords or len(w) <= 2:
            if buff:
                phrases.append(buff)
                buff = []
        else:
            buff.append(w)
    if buff:
        phrases.append(buff)

    # word stats
    freq: Dict[str, int] = {}
    degree: Dict[str, int] = {}

    for p in phrases:
        unique = p
        deg = len(p) - 1
        for w in unique:
            freq[w] = freq.get(w, 0) + 1
            degree[w] = degree.get(w, 0) + deg

    word_score: Dict[str, float] = {}
    for w in freq:
        word_score[w] = (degree[w] + freq[w]) / float(freq[w])

    scored: List[RakePhrase] = []
    for p in phrases:
        if len(p) < 2:
            continue
        phrase = " ".join(p)
        score = sum(word_score.get(w, 0.0) for w in p)
        scored.append(RakePhrase(phrase=phrase, score=score))

    # de-duplicate by max score
    best: Dict[str, float] = {}
    for sp in scored:
        best[sp.phrase] = max(best.get(sp.phrase, 0.0), sp.score)

    out = [RakePhrase(phrase=k, score=v) for k, v in best.items()]
    out.sort(key=lambda x: x.score, reverse=True)
    return out[:top_k]
