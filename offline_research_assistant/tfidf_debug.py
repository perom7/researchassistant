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
    for i, words in enumerate(sent_tokens):
        if not words:
            scores.append((i, 0.0))
            continue
        tf = Counter(words)
        score = 0.0
        for t, c in tf.items():
            score += (c / len(words)) * idf(t)
        score /= (1.0 + abs(len(sents[i]) - 160) / 160.0)
        scores.append((i, score))

    top = sorted(scores, key=lambda x: x[1], reverse=True)[:max_sentences]
    keep = sorted(i for i, _ in top)

    summary = " ".join(sents[i] for i in keep)

    sentence_scores = [(i, float(sc), sents[i]) for i, sc in scores]

    return TfidfSummaryDebug(
        summary=summary,
        sentence_scores=sentence_scores,
        selected_indices=keep,
        top_terms=top_terms,
        segmentation_method=seg.method_used,
    )
