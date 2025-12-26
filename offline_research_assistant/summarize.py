from __future__ import annotations

from collections import Counter
import math
from typing import List

from .text_utils import split_sentences, tokenize_words, remove_stopwords


def summarize_extractive(text: str, max_sentences: int = 10) -> str:
    """Offline extractive summarization using a lightweight TF-IDF sentence scorer.

    No external NLP models; deterministic; works fully offline.
    """

    sents = split_sentences(text)
    if not sents:
        return ""

    max_sentences = max(1, min(max_sentences, len(sents)))

    sent_tokens: List[List[str]] = []
    df = Counter()

    for s in sents:
        words = remove_stopwords(tokenize_words(s))
        unique = set(words)
        for w in unique:
            df[w] += 1
        sent_tokens.append(words)

    n = len(sents)

    def idf(term: str) -> float:
        # Smooth IDF
        return math.log((n + 1) / (df[term] + 1)) + 1.0

    scores = []
    for i, words in enumerate(sent_tokens):
        if not words:
            scores.append((i, 0.0))
            continue
        tf = Counter(words)
        score = 0.0
        for t, c in tf.items():
            score += (c / len(words)) * idf(t)
        # mild length normalization
        score /= (1.0 + abs(len(sents[i]) - 160) / 160.0)
        scores.append((i, score))

    top = sorted(scores, key=lambda x: x[1], reverse=True)[:max_sentences]
    keep = sorted(i for i, _ in top)
    return " ".join(sents[i] for i in keep)
