from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple


_COMMON_HEADINGS = [
    "abstract",
    "introduction",
    "background",
    "related work",
    "method",
    "methodology",
    "materials and methods",
    "experiments",
    "results",
    "discussion",
    "conclusion",
    "limitations",
    "future work",
    "references",
]


@dataclass(frozen=True)
class SectionSplit:
    sections: Dict[str, str]
    order: List[str]


def split_into_sections(text: str) -> SectionSplit:
    """Heuristic section splitter based on common headings.

    Returns a dict of section_name -> section_text.
    """

    lines = text.splitlines()
    indices: List[Tuple[int, str]] = []

    for i, ln in enumerate(lines):
        s = ln.strip().lower()
        s = re.sub(r"\s+", " ", s)
        if not s or len(s) > 60:
            continue

        # match headings with optional numbering
        for h in _COMMON_HEADINGS:
            if s == h or s.endswith(":") and s[:-1] == h:
                indices.append((i, h.title()))
                break
            if re.match(rf"^(\d+\.?\s+)?{re.escape(h)}\b", s):
                indices.append((i, h.title()))
                break

    if not indices:
        return SectionSplit(sections={"Document": text}, order=["Document"])

    # unique by first occurrence
    seen = set()
    uniq = []
    for idx, name in indices:
        if name in seen:
            continue
        seen.add(name)
        uniq.append((idx, name))

    uniq.sort(key=lambda x: x[0])

    sections: Dict[str, str] = {}
    order: List[str] = []

    for j, (start, name) in enumerate(uniq):
        end = uniq[j + 1][0] if j + 1 < len(uniq) else len(lines)
        body = "\n".join(lines[start + 1 : end]).strip()
        if body:
            sections[name] = body
            order.append(name)

    if not sections:
        return SectionSplit(sections={"Document": text}, order=["Document"])

    return SectionSplit(sections=sections, order=order)
