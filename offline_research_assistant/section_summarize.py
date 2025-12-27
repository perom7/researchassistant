from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

from collections import defaultdict

from .sections import split_into_sections
from .text_utils import split_sentences, tokenize_words
from .textrank import summarize_textrank
from .tfidf_debug import summarize_tfidf_debug


SummarizerAlgo = Literal["tfidf", "textrank"]


@dataclass(frozen=True)
class SectionSummary:
    section: str
    words: int
    allocated_sentences: int
    summary: str
    selected_indices: List[int]
    top_terms: Optional[List[Tuple[str, float]]] = None  # TF-IDF only


@dataclass(frozen=True)
class SectionSummarizationResult:
    summary: str
    sections: List[SectionSummary]
    sentence_score_rows: List[Dict[str, Any]]  # for CSV export
    summarizer_debug: Dict[str, Any]


def _allocate_sentences(
    section_words: List[Tuple[str, int]],
    total_sentences: int,
    max_per_section: Optional[int],
) -> Dict[str, int]:
    total_sentences = max(1, int(total_sentences))
    words_total = sum(w for _, w in section_words) or 1

    alloc: Dict[str, int] = {}
    for name, w in section_words:
        share = (w / words_total) * total_sentences
        alloc[name] = max(1, int(round(share)))

    if max_per_section is not None:
        cap = max(1, int(max_per_section))
        for k in list(alloc.keys()):
            alloc[k] = min(alloc[k], cap)

    # adjust to match total
    def capped(k: str) -> bool:
        if max_per_section is None:
            return False
        return alloc[k] >= max(1, int(max_per_section))

    # too many -> reduce
    while sum(alloc.values()) > total_sentences:
        # reduce from sections with largest allocation, but never below 1
        candidates = sorted(alloc.items(), key=lambda x: x[1], reverse=True)
        changed = False
        for k, v in candidates:
            if v > 1:
                alloc[k] = v - 1
                changed = True
                break
        if not changed:
            break

    # too few -> add
    while sum(alloc.values()) < total_sentences:
        candidates = sorted(alloc.items(), key=lambda x: x[1])
        changed = False
        for k, v in candidates:
            if not capped(k):
                alloc[k] = v + 1
                changed = True
                break
        if not changed:
            # All sections are capped but we still haven't reached total.
            # Treat max_per_section as a soft cap in this case to honor total_sentences.
            biggest = max(alloc.items(), key=lambda x: x[1])[0]
            alloc[biggest] = alloc[biggest] + 1
            changed = True

    return alloc


def _rebalance_for_paper_structure(alloc: Dict[str, int], *, total_sentences: int) -> Dict[str, int]:
    """Heuristics to avoid summaries dominated by Abstract.

    Research-paper PDFs often have an Abstract that is easy to score highly, which can
    lead to summaries that feel like they're only based on Abstract. This function
    caps Abstract and guarantees some coverage of later sections when present.
    """

    if not alloc:
        return alloc

    # Only apply if we actually have multiple sections.
    if len(alloc) <= 1:
        return alloc

    total_sentences = max(1, int(total_sentences))

    # Cap Abstract to at most ~25% of requested sentences (but at least 1 if present).
    abs_key = None
    for k in alloc.keys():
        if k.strip().lower() == "abstract":
            abs_key = k
            break

    if abs_key is not None and total_sentences >= 6:
        cap = max(1, int(round(total_sentences * 0.25)))
        if alloc.get(abs_key, 0) > cap:
            freed = alloc[abs_key] - cap
            alloc[abs_key] = cap

            # Prefer distributing to later sections if they exist.
            preferred = [
                "results",
                "discussion",
                "conclusion",
                "experiments",
                "method",
                "methodology",
                "introduction",
            ]
            keys_by_pref: List[str] = []
            for p in preferred:
                for k in alloc.keys():
                    if k.strip().lower() == p and k not in keys_by_pref:
                        keys_by_pref.append(k)
            for k in alloc.keys():
                if k != abs_key and k not in keys_by_pref:
                    keys_by_pref.append(k)

            idx = 0
            while freed > 0 and keys_by_pref:
                alloc[keys_by_pref[idx % len(keys_by_pref)]] += 1
                freed -= 1
                idx += 1

    # Guarantee at least 1 sentence from key sections (if present) when budget allows.
    must_have = ["method", "results", "conclusion"]
    present = []
    for m in must_have:
        for k in alloc.keys():
            if k.strip().lower() == m:
                present.append(k)
                break

    if present and total_sentences >= (1 + len(present)):
        for k in present:
            alloc[k] = max(1, alloc.get(k, 0))

        # If we exceeded total, trim from the largest non-must-have sections first.
        while sum(alloc.values()) > total_sentences:
            candidates = sorted(alloc.items(), key=lambda x: x[1], reverse=True)
            changed = False
            for k, v in candidates:
                if k in present:
                    continue
                if v > 1:
                    alloc[k] = v - 1
                    changed = True
                    break
            if not changed:
                break

    return alloc


def summarize_by_sections(
    text: str,
    total_sentences: int,
    summarizer: SummarizerAlgo,
    sentence_segmentation: Literal["regex", "spacy"] = "regex",
    max_sentences_per_section: Optional[int] = None,
    include_references: bool = False,
) -> SectionSummarizationResult:
    split = split_into_sections(text)

    # filter sections
    order = list(split.order)
    sections = dict(split.sections)
    if not include_references:
        sections = {k: v for k, v in sections.items() if k.lower() != "references"}
        order = [k for k in order if k.lower() != "references"]
        if not sections:
            # fallback
            sections = split.sections
            order = split.order

    section_words: List[Tuple[str, int]] = []
    for name in order:
        wc = len(tokenize_words(sections.get(name, "")))
        section_words.append((name, wc))

    alloc = _allocate_sentences(section_words, total_sentences, max_sentences_per_section)
    alloc = _rebalance_for_paper_structure(alloc, total_sentences=int(total_sentences))

    summaries: List[SectionSummary] = []
    sentence_rows: List[Dict[str, Any]] = []
    overall_terms: Dict[str, float] = defaultdict(float)

    for name in order:
        sec_text = sections.get(name, "").strip()
        if not sec_text:
            continue

        k = alloc.get(name, 1)

        if summarizer == "textrank":
            tr = summarize_textrank(sec_text, max_sentences=k)
            sec_sents = split_sentences(sec_text)

            for i, score in tr.sentence_scores:
                sent = sec_sents[i] if 0 <= i < len(sec_sents) else ""
                sentence_rows.append(
                    {
                        "section": name,
                        "sentence_index": i,
                        "score": float(score),
                        "selected": i in tr.selected_indices,
                        "sentence": sent,
                    }
                )

            summaries.append(
                SectionSummary(
                    section=name,
                    words=len(tokenize_words(sec_text)),
                    allocated_sentences=k,
                    summary=tr.summary,
                    selected_indices=list(tr.selected_indices),
                    top_terms=None,
                )
            )
        else:
            td = summarize_tfidf_debug(sec_text, max_sentences=k, segmentation=sentence_segmentation)
            for t, sc in td.top_terms:
                overall_terms[t] += float(sc)
            for i, score, sent in td.sentence_scores:
                sentence_rows.append(
                    {
                        "section": name,
                        "sentence_index": i,
                        "score": float(score),
                        "selected": i in td.selected_indices,
                        "sentence": sent,
                    }
                )

            summaries.append(
                SectionSummary(
                    section=name,
                    words=len(tokenize_words(sec_text)),
                    allocated_sentences=k,
                    summary=td.summary,
                    selected_indices=list(td.selected_indices),
                    top_terms=list(td.top_terms),
                )
            )

    combined_summary = "\n\n".join(s.summary for s in summaries if s.summary.strip()).strip()

    summarizer_debug: Dict[str, Any] = {
        "mode": "sectioned",
        "algorithm": summarizer,
        "total_sentences": int(total_sentences),
        "max_sentences_per_section": max_sentences_per_section,
        "include_references": include_references,
        "llm": (
            {"provider": "gemini", "model": gemini_model, "temperature": 0.3}
            if summarizer == "gemini"
            else None
        ),
        "top_terms": (
            [{"term": t, "score": sc} for t, sc in sorted(overall_terms.items(), key=lambda x: x[1], reverse=True)[:20]]
            if summarizer == "tfidf" and overall_terms
            else None
        ),
        "sections": [
            {
                "section": s.section,
                "words": s.words,
                "allocated_sentences": s.allocated_sentences,
                "selected_indices": s.selected_indices,
                "top_terms": ([{"term": t, "score": sc} for t, sc in (s.top_terms or [])][:10] if s.top_terms else None),
            }
            for s in summaries
        ],
    }

    return SectionSummarizationResult(
        summary=combined_summary,
        sections=summaries,
        sentence_score_rows=sentence_rows,
        summarizer_debug=summarizer_debug,
    )
