from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List


_DEFAULT_STOPWORDS = {
    "a","an","the","and","or","but","if","then","else","when","while","for","to","of","in","on","at","by","with","as",
    "is","are","was","were","be","been","being","this","that","these","those","it","its","they","their","them","we","our","you",
    "from","into","over","under","between","within","without","can","could","may","might","should","would","will","shall","do","does",
    "did","done","not","no","yes","such","also","however","therefore","thus","more","most","less","least","many","much","some","any",
    "than","very","via","using","use","used","based","across","among","per"
}


@dataclass(frozen=True)
class CleanOptions:
    collapse_whitespace: bool = True
    strip_page_markers: bool = True
    strip_null_bytes: bool = True


def clean_text(text: str, opts: CleanOptions | None = None) -> str:
    if opts is None:
        opts = CleanOptions()

    if opts.strip_null_bytes:
        text = text.replace("\x00", "")

    if opts.strip_page_markers:
        text = re.sub(r"Page\s+\d+\s+of\s+\d+", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\n\s*\d+\s*\n", "\n", text)

    if opts.collapse_whitespace:
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = _SENT_SPLIT.split(text)
    out: List[str] = []
    for p in parts:
        p = p.strip()
        if len(p) < 20:
            continue
        out.append(p)
    return out


_WORD = re.compile(r"[A-Za-z][A-Za-z\-']+")


def tokenize_words(text: str) -> List[str]:
    return [w.lower() for w in _WORD.findall(text)]


def remove_stopwords(words: Iterable[str], stopwords: set[str] | None = None) -> List[str]:
    sw = stopwords or _DEFAULT_STOPWORDS
    return [w for w in words if w not in sw and len(w) > 2]
