from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def _require_pyttsx3():
    try:
        import pyttsx3  # type: ignore
        return pyttsx3
    except Exception as e:
        raise RuntimeError(
            "pyttsx3 is required for offline TTS. Install with: pip install pyttsx3 pywin32"
        ) from e


def synthesize_to_wav(text: str, wav_path: str | Path, voice_name_contains: Optional[str] = None, rate: Optional[int] = None) -> str:
    """Offline TTS to WAV using Windows SAPI5 via pyttsx3."""

    pyttsx3 = _require_pyttsx3()
    wav_path = Path(wav_path)
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    engine = pyttsx3.init()

    if rate is not None:
        engine.setProperty('rate', int(rate))

    if voice_name_contains:
        vc = voice_name_contains.lower()
        for v in engine.getProperty('voices'):
            name = (getattr(v, 'name', '') or '').lower()
            if vc in name:
                engine.setProperty('voice', v.id)
                break

    engine.save_to_file(text, str(wav_path))
    engine.runAndWait()

    if not wav_path.exists() or wav_path.stat().st_size == 0:
        raise RuntimeError("TTS failed to produce output audio.")

    return str(wav_path)


def concat_wavs(wav_paths: list[str | Path], out_wav: str | Path, silence_ms: int = 600) -> str:
    """Concatenate WAVs; uses pydub if available (no ffmpeg needed for wav)."""

    try:
        from pydub import AudioSegment  # type: ignore
    except Exception as e:
        raise RuntimeError("pydub is required for WAV concatenation. Install with: pip install pydub") from e

    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    combined = AudioSegment.silent(duration=0)
    silence = AudioSegment.silent(duration=silence_ms)

    for p in wav_paths:
        seg = AudioSegment.from_wav(str(p))
        combined += seg + silence

    combined.export(str(out_wav), format='wav')
    return str(out_wav)
