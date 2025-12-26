from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

from .keywords import extract_keywords
from .rake import extract_rake_phrases
from .readability import compute_linguistic_metrics
from .text_utils import tokenize_words


@dataclass(frozen=True)
class EvaluationMetrics:
    compression_ratio: float
    keyword_coverage: float
    reading_grade_before: float
    reading_grade_after: float


def compute_evaluation(full_text: str, summary: str, simplified: str, keywords: List[str]) -> EvaluationMetrics:
    full_words = tokenize_words(full_text)
    sum_words = tokenize_words(summary)

    compression = (len(sum_words) / len(full_words)) if full_words else 0.0

    if keywords:
        present = 0
        s_low = summary.lower()
        for k in keywords:
            if k.lower() in s_low:
                present += 1
        coverage = present / len(keywords)
    else:
        coverage = 0.0

    before = compute_linguistic_metrics(summary).readability.flesch_kincaid_grade
    after = compute_linguistic_metrics(simplified).readability.flesch_kincaid_grade

    return EvaluationMetrics(
        compression_ratio=float(compression),
        keyword_coverage=float(coverage),
        reading_grade_before=float(before),
        reading_grade_after=float(after),
    )


def keyword_sets(text: str, algo: str = "freq", top_k: int = 15, min_freq: int = 1) -> Dict[str, Any]:
    if algo == "rake":
        phrases = extract_rake_phrases(text, top_k=top_k)
        return {
            "algorithm": "rake",
            "keywords": [{"term": p.phrase, "score": p.score} for p in phrases],
        }

    # freq
    kw = [(k, c) for k, c in extract_keywords(text, top_k=top_k) if c >= min_freq]
    return {
        "algorithm": "freq",
        "keywords": [{"term": k, "score": float(c)} for k, c in kw],
    }


def linguistic_analysis(text: str) -> Dict[str, Any]:
    m = compute_linguistic_metrics(text)
    return {
        "readability": asdict(m.readability),
        "avg_sentence_length": m.avg_sentence_length,
        "p95_sentence_length": m.p95_sentence_length,
        "passive_voice_ratio": m.passive_voice_ratio,
        "jargon_density": m.jargon_density,
    }
