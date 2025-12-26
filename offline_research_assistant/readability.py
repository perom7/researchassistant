from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List

from .text_utils import split_sentences, tokenize_words


_VOWELS = set("aeiouy")


def _count_syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    # heuristic syllable count
    count = 0
    prev_vowel = False
    for ch in w:
        is_vowel = ch in _VOWELS
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    # silent e
    if w.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


@dataclass(frozen=True)
class ReadabilityMetrics:
    flesch_reading_ease: float
    flesch_kincaid_grade: float
    words: int
    sentences: int
    syllables: int


def compute_readability(text: str) -> ReadabilityMetrics:
    sents = split_sentences(text)
    words = tokenize_words(text)
    if not words:
        return ReadabilityMetrics(0.0, 0.0, 0, len(sents), 0)

    syllables = sum(_count_syllables(w) for w in words)
    wc = len(words)
    sc = max(1, len(sents))

    # Flesch Reading Ease and FK Grade
    fre = 206.835 - 1.015 * (wc / sc) - 84.6 * (syllables / wc)
    fk = 0.39 * (wc / sc) + 11.8 * (syllables / wc) - 15.59

    return ReadabilityMetrics(
        flesch_reading_ease=float(fre),
        flesch_kincaid_grade=float(fk),
        words=wc,
        sentences=len(sents),
        syllables=syllables,
    )


@dataclass(frozen=True)
class LinguisticMetrics:
    readability: ReadabilityMetrics
    avg_sentence_length: float
    p95_sentence_length: float
    passive_voice_ratio: float
    jargon_density: float


def _percentile(values: List[int], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = int(round((p / 100.0) * (len(values) - 1)))
    k = min(max(k, 0), len(values) - 1)
    return float(values[k])


def compute_linguistic_metrics(text: str) -> LinguisticMetrics:
    sents = split_sentences(text)
    sent_lengths = [len(tokenize_words(s)) for s in sents if s.strip()]
    avg_len = sum(sent_lengths) / len(sent_lengths) if sent_lengths else 0.0
    p95 = _percentile(sent_lengths, 95)

    # Passive voice heuristic: (be-verb) + (past participle-like token)
    passive_hits = 0
    total_sents = max(1, len(sents))
    be = {"am","is","are","was","were","be","been","being"}

    for s in sents:
        w = tokenize_words(s)
        for i in range(len(w) - 1):
            if w[i] in be and (w[i + 1].endswith("ed") or w[i + 1].endswith("en")):
                passive_hits += 1
                break

    passive_ratio = passive_hits / total_sents

    # Jargon density heuristic: long words + uppercase acronyms
    words = tokenize_words(text)
    if not words:
        jd = 0.0
    else:
        long_words = sum(1 for w in words if len(w) >= 12)
        jd = long_words / len(words)

    return LinguisticMetrics(
        readability=compute_readability(text),
        avg_sentence_length=float(avg_len),
        p95_sentence_length=float(p95),
        passive_voice_ratio=float(passive_ratio),
        jargon_density=float(jd),
    )
