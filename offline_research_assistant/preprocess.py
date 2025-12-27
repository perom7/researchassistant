from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .text_utils import fix_split_leading_words


@dataclass(frozen=True)
class PreprocessOptions:
    remove_repeated_headers_footers: bool = True
    strip_citations: bool = True
    skip_table_like_lines: bool = True


_CITATION_PATTERNS = [
    re.compile(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]"),  # [1] or [1,2]
    re.compile(r"\(\s*\d{4}\s*\)"),  # (2023)
]


def _strip_citations(text: str) -> str:
    out = text
    for pat in _CITATION_PATTERNS:
        out = pat.sub("", out)
    # common author-year like: Smith et al., 2023
    out = re.sub(r"\b[A-Z][A-Za-z\-]+\s+et\s+al\.,?\s+\d{4}\b", "", out)
    return out


def _is_table_like(line: str) -> bool:
    s = line.strip()
    if not s:
        return False

    # Do not treat obvious math/equations as tables
    if any(op in s for op in ["=", "≤", "≥", "≈", "≠", "∑", "∫", "→", "←", "⇒", "⇔", "±", "×", "·", "^"]):
        return False
    if re.search(r"\b[a-zA-Z]\s*[=<>]\s*\b", s):
        return False

    # pipes or many columns by spacing
    if s.count("|") >= 2:
        return True
    if re.search(r"\b\d+\s{3,}\d+\b", s):
        return True
    if re.search(r"-{3,}\s+-{3,}", s):
        return True
    return False


def _dehyphenate(text: str) -> str:
    # Join words split across line breaks: "inter-\nnational" -> "international"
    return re.sub(r"([A-Za-z])\-\n([A-Za-z])", r"\1\2", text)


def remove_repeated_headers_footers(pages: Sequence[str], top_lines: int = 2, bottom_lines: int = 2, threshold: float = 0.6) -> List[str]:
    """Remove lines that repeat across many pages (simple header/footer noise removal)."""

    if not pages:
        return []

    tops: List[str] = []
    bottoms: List[str] = []
    for p in pages:
        lines = [ln.strip() for ln in p.splitlines() if ln.strip()]
        tops.extend(lines[:top_lines])
        bottoms.extend(lines[-bottom_lines:] if bottom_lines > 0 else [])

    def frequent(lines: List[str]) -> set[str]:
        counts = {}
        for ln in lines:
            counts[ln] = counts.get(ln, 0) + 1
        cut = max(2, int(len(pages) * threshold))
        return {ln for ln, c in counts.items() if c >= cut and len(ln) <= 120}

    bad = frequent(tops) | frequent(bottoms)

    cleaned_pages: List[str] = []
    for p in pages:
        out_lines = []
        for ln in p.splitlines():
            s = ln.strip()
            if s in bad:
                continue
            out_lines.append(ln)
        cleaned_pages.append("\n".join(out_lines))

    return cleaned_pages


def preprocess_pages(pages: Sequence[str], opts: PreprocessOptions) -> Tuple[str, str]:
    """Return (clean_text_for_processing, raw_text_for_export)."""

    raw = "\n\n".join(pages)
    raw = _dehyphenate(raw)

    proc_pages = list(pages)
    if opts.remove_repeated_headers_footers:
        proc_pages = remove_repeated_headers_footers(proc_pages)

    if opts.skip_table_like_lines:
        new_pages = []
        for p in proc_pages:
            keep = [ln for ln in p.splitlines() if not _is_table_like(ln)]
            new_pages.append("\n".join(keep))
        proc_pages = new_pages

    processed = "\n\n".join(proc_pages)
    processed = _dehyphenate(processed)
    if opts.strip_citations:
        processed = _strip_citations(processed)

    # Fix common PDF-extraction artifact: split first letter at sentence starts (e.g., 'T HE').
    processed = fix_split_leading_words(processed)

    return processed, raw
