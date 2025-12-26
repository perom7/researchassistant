from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .text_utils import split_sentences, tokenize_words, remove_stopwords


@dataclass(frozen=True)
class TextRankResult:
    summary: str
    sentence_scores: List[Tuple[int, float]]
    selected_indices: List[int]


def _build_tfidf_vectors(sent_tokens: List[List[str]]) -> List[Dict[str, float]]:
    # document frequency
    df: Dict[str, int] = {}
    for toks in sent_tokens:
        for t in set(toks):
            df[t] = df.get(t, 0) + 1

    n = len(sent_tokens)
    if n == 0:
        return []

    vectors: List[Dict[str, float]] = []
    for toks in sent_tokens:
        if not toks:
            vectors.append({})
            continue
        tf: Dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        denom = float(len(toks))
        vec: Dict[str, float] = {}
        for t, c in tf.items():
            idf = ( (n + 1) / (df.get(t, 0) + 1) )
            # log smoothing
            idf = 1.0 + __import__("math").log(idf)
            vec[t] = (c / denom) * idf
        vectors.append(vec)
    return vectors


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    # dot
    if len(a) > len(b):
        a, b = b, a
    dot = 0.0
    for k, av in a.items():
        bv = b.get(k)
        if bv is not None:
            dot += av * bv

    na = sum(v * v for v in a.values())
    nb = sum(v * v for v in b.values())
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / ((na ** 0.5) * (nb ** 0.5))


def summarize_textrank(text: str, max_sentences: int = 10, damping: float = 0.85, iters: int = 50) -> TextRankResult:
    sents = split_sentences(text)
    if not sents:
        return TextRankResult(summary="", sentence_scores=[], selected_indices=[])

    max_sentences = max(1, min(max_sentences, len(sents)))

    sent_tokens = [remove_stopwords(tokenize_words(s)) for s in sents]
    vecs = _build_tfidf_vectors(sent_tokens)
    if not vecs:
        # fallback: first N sentences
        keep = list(range(max_sentences))
        return TextRankResult(summary=" ".join(sents[i] for i in keep), sentence_scores=[(i, 0.0) for i in range(len(sents))], selected_indices=keep)

    n = len(sents)
    # similarity matrix rows
    P: List[Dict[int, float]] = []
    for i in range(n):
        sims: Dict[int, float] = {}
        row_sum = 0.0
        for j in range(n):
            if i == j:
                continue
            sc = _cosine(vecs[i], vecs[j])
            if sc > 0.0:
                sims[j] = sc
                row_sum += sc
        if row_sum > 0.0:
            for j in list(sims.keys()):
                sims[j] /= row_sum
        P.append(sims)

    r = [1.0 / n for _ in range(n)]
    teleport = (1.0 - damping) / n

    for _ in range(iters):
        new_r = [teleport for _ in range(n)]
        # r = teleport + damping * P^T r
        for i in range(n):
            if not P[i]:
                continue
            for j, pij in P[i].items():
                new_r[j] += damping * pij * r[i]
        r = new_r

    scores = [(i, float(r[i])) for i in range(n)]
    top = sorted(scores, key=lambda x: x[1], reverse=True)[:max_sentences]
    keep = sorted(i for i, _ in top)

    summary = " ".join(sents[i] for i in keep)
    return TextRankResult(summary=summary, sentence_scores=scores, selected_indices=keep)
