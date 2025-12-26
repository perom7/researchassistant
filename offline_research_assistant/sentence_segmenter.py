from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

from .text_utils import split_sentences as regex_split_sentences


SegmentationMethod = Literal["regex", "spacy"]


@dataclass(frozen=True)
class SegmentationResult:
    method_used: SegmentationMethod
    sentences: List[str]


def split_sentences(text: str, method: SegmentationMethod = "regex") -> SegmentationResult:
    """Split text into sentences.

    - `regex`: fast, always available
    - `spacy`: higher quality if spaCy is installed (no model download required for sentencizer)
    """

    if method == "spacy":
        try:
            import spacy  # type: ignore

            nlp = spacy.blank("en")
            nlp.add_pipe("sentencizer")
            doc = nlp(text)
            sents = [s.text.strip() for s in doc.sents if len(s.text.strip()) >= 20]
            return SegmentationResult(method_used="spacy", sentences=sents)
        except Exception:
            # fall back
            pass

    return SegmentationResult(method_used="regex", sentences=regex_split_sentences(text))
