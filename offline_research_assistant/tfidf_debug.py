from __future__ import annotations

from collections import Counter
import math
from dataclasses import dataclass
from typing import List, Tuple

from .sentence_segmenter import split_sentences
from .text_utils import tokenize_words, remove_stopwords


@dataclass(frozen=True)
class TfidfSummaryDebug:
    summary: str
    sentence_scores: List[Tuple[int, float, str]]  # (idx, score, sentence)
    selected_indices: List[int]
    top_terms: List[Tuple[str, float]]
    segmentation_method: str


def summarize_tfidf_debug(text: str, max_sentences: int = 10, segmentation: str = "regex") -> TfidfSummaryDebug:
    seg = split_sentences(text, method=segmentation)  # type: ignore
    sents = seg.sentences
    if not sents:
        return TfidfSummaryDebug(summary="", sentence_scores=[], selected_indices=[], top_terms=[], segmentation_method=seg.method_used)

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
        return math.log((n + 1) / (df[term] + 1)) + 1.0

    # top terms by corpus tf-idf proxy (tf * idf) aggregated
    agg = Counter()
    for words in sent_tokens:
        tf = Counter(words)
        for t, c in tf.items():
            agg[t] += (c / max(1, len(words))) * idf(t)

    top_terms = [(t, float(v)) for t, v in agg.most_common(20)]

    scores = []
    vectors: List[dict[str, float]] = []
    for i, words in enumerate(sent_tokens):
        if not words:
            scores.append((i, 0.0))
            vectors.append({})
            continue
        tf = Counter(words)
        score = 0.0
        vec: dict[str, float] = {}
        for t, c in tf.items():
            w = (c / len(words)) * idf(t)
            score += w
            vec[t] = w
        score /= (1.0 + abs(len(sents[i]) - 160) / 160.0)

        # Mild position bias: earlier sentences often carry definitions/context.
        # Keep it small to avoid washing out relevance.
        pos_boost = 1.0 + (0.12 * (1.0 - (i / max(1, n - 1))))
        score *= pos_boost

        scores.append((i, score))
        vectors.append(vec)

    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        if not a or not b:
            return 0.0
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

    # Select sentences with Maximal Marginal Relevance to reduce redundancy.
    # MMR score = λ * relevance - (1-λ) * max_similarity(selected)
    lam = 0.78
    base = {i: sc for i, sc in scores}
    selected: List[int] = []
    candidates = set(range(n))

    # seed with best base score
    seed = max(base.items(), key=lambda x: x[1])[0]
    selected.append(seed)
    candidates.remove(seed)

    while len(selected) < max_sentences and candidates:
        best_i = None
        best_score = -1e18
        for i in list(candidates):
            sim = 0.0
            for j in selected:
                sim = max(sim, _cosine(vectors[i], vectors[j]))
            mmr = (lam * base[i]) - ((1.0 - lam) * sim)
            if mmr > best_score:
                best_score = mmr
                best_i = i
        if best_i is None:
            break
        selected.append(best_i)
        candidates.remove(best_i)

    keep = sorted(selected)

    summary = " ".join(sents[i] for i in keep)

    sentence_scores = [(i, float(sc), sents[i]) for i, sc in scores]

    return TfidfSummaryDebug(
        summary=summary,
        sentence_scores=sentence_scores,
        selected_indices=keep,
        top_terms=top_terms,
        segmentation_method=seg.method_used,
    )
