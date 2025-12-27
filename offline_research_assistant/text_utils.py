from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List


_SPLIT_LEADING_WORDS = {
    # Very common sentence-leading words that frequently appear split by PDF text extraction
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "for",
    "from",
    "how",
    "if",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "our",
    "over",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "this",
    "those",
    "to",
    "under",
    "us",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "why",
    "with",
    "within",
    "without",
    "would",
    "you",
    "your",
}


_BULLET_PREFIX = r"(?:^|\n)[\t ]*(?:[-*•·●▪▫►‣⁃]|\uf0b7|\uf0a7)\s+"
_SPLIT_LEADING_WORD_RE = re.compile(
    rf"(^|[\n\r]|[.!?:;]\s|{_BULLET_PREFIX})([A-Za-z])\s+([A-Za-z]{{1,40}})"
)

# Words that are safe to split when a PDF extraction accidentally produces
# things like 'T HEPROPOSED' (missing space after 'THE').
_SPLITTABLE_PREFIX_WORDS = {
    "the",
    "this",
    "that",
    "these",
    "those",
    "our",
    "their",
    "your",
    "its",
}


def fix_split_leading_words(text: str) -> str:
    """Fix a common PDF-extraction artifact like 'T HE' -> 'THE'.

    We only join when the joined token matches a small set of very common
    sentence-leading words to avoid damaging legitimate phrases like 'I see'.
    """

    if not text:
        return text

    def repl(m: re.Match[str]) -> str:
        prefix, first, rest = m.group(1), m.group(2), m.group(3)
        joined = (first + rest)
        joined_low = joined.lower()

        # Standard case: 'T HE' -> 'THE'
        if joined_low in _SPLIT_LEADING_WORDS:
            return prefix + joined

        # Common PDF case: 'T HEPROPOSED' -> 'THE PROPOSED'
        for w in _SPLITTABLE_PREFIX_WORDS:
            if joined_low.startswith(w) and len(joined) > len(w):
                nxt = joined[len(w)]
                if nxt.isalpha() and nxt.isupper():
                    return prefix + joined[: len(w)] + " " + joined[len(w) :]
        return m.group(0)

    # Apply a few passes in case multiple artifacts occur back-to-back
    out = text
    for _ in range(3):
        new = _SPLIT_LEADING_WORD_RE.sub(repl, out)
        if new == out:
            break
        out = new
    return out


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

    # Apply PDF artifact normalization after basic whitespace cleanup.
    text = fix_split_leading_words(text)

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
