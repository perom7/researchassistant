from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class GeminiConfig:
    api_key: str
    model: str = "gemini-1.5-flash"
    temperature: float = 0.3
    max_output_tokens: int = 1024


class GeminiUnavailableError(RuntimeError):
    pass


def _require_api_key(api_key: Optional[str]) -> str:
    k = (api_key or "").strip()
    if not k:
        raise ValueError("Gemini API key is required when summarizer='gemini'.")
    return k


def generate_text(prompt: str, *, config: GeminiConfig) -> str:
    """Generate text via Gemini using the google-generativeai SDK.

    This function intentionally avoids persisting any secrets.
    """

    try:
        import google.generativeai as genai  # type: ignore
    except Exception as e:  # pragma: no cover
        raise GeminiUnavailableError(
            "Gemini SDK not installed. Add 'google-generativeai' to requirements and reinstall."
        ) from e

    genai.configure(api_key=_require_api_key(config.api_key))

    model = genai.GenerativeModel(config.model)
    resp = model.generate_content(
        prompt,
        generation_config={
            "temperature": float(config.temperature),
            "max_output_tokens": int(config.max_output_tokens),
        },
    )

    text = getattr(resp, "text", None)
    if not text:
        # Try to extract from candidates if needed
        try:
            candidates = getattr(resp, "candidates", None) or []
            parts: list[str] = []
            for c in candidates:
                content = getattr(c, "content", None)
                if not content:
                    continue
                for p in getattr(content, "parts", []) or []:
                    t = getattr(p, "text", None)
                    if t:
                        parts.append(t)
            text = "\n".join(parts).strip()
        except Exception:
            text = ""

    return (text or "").strip()


def build_system_prefix(*, no_markdown: bool = True) -> str:
    if no_markdown:
        return (
            "You are a helpful research assistant.\n"
            "Return plain text only (no Markdown, no code fences).\n"
        )
    return "You are a helpful research assistant.\n"


def safe_strip(text: str) -> str:
    return (text or "").strip().strip("` ")
