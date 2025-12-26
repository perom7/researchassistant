from __future__ import annotations

import re
from typing import List


_SIMPLE_REPLACEMENTS = {
    "utilize": "use",
    "leverage": "use",
    "facilitate": "help",
    "therefore": "so",
    "however": "but",
    "moreover": "also",
    "approximately": "about",
    "demonstrate": "show",
    "significant": "important",
    "methodology": "method",
    "framework": "system",
    "architecture": "design",
    "implementation": "build",
    "evaluation": "test",
}


def simplify_text(text: str, aggressiveness: int = 1) -> str:
    """Offline simplifier: plain-language replacements + shorter sentences.

    This is not an LLM; it’s deterministic and safe offline.
    """

    aggressiveness = max(1, min(int(aggressiveness), 3))

    out = text
    for k, v in _SIMPLE_REPLACEMENTS.items():
        out = re.sub(rf"\b{k}\b", v, out, flags=re.IGNORECASE)

    if aggressiveness >= 2:
        out = out.replace("e.g.", "for example")
        out = out.replace("i.e.", "that is")

    if aggressiveness >= 3:
        out = re.sub(r"\b(sufficiently|substantially|significantly)\b", "", out, flags=re.IGNORECASE)
        out = re.sub(r"\b(there is|there are)\b", "", out, flags=re.IGNORECASE)

    # soft sentence splitting to avoid huge walls of text
    out = re.sub(r"\s+", " ", out).strip()
    out = out.replace(";", ".")
    out = out.replace(":", ".")

    # Insert line breaks every ~N sentences
    sentences = re.split(r"(?<=[.!?])\s+", out)
    chunks: List[str] = []

    group = 2 if aggressiveness <= 2 else 1
    for i in range(0, len(sentences), group):
        chunk = " ".join(s.strip() for s in sentences[i:i+group] if s.strip())
        if chunk:
            chunks.append(chunk)

    return "\n\n".join(chunks).strip()
