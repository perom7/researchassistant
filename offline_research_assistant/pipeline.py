from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from .analysis import compute_evaluation, keyword_sets, linguistic_analysis
from .pdf_utils import extract_pages_from_pdf
from .preprocess import PreprocessOptions, preprocess_pages
from .simplify import simplify_text
from .scripts import generate_podcast_script, podcast_script_to_text, generate_video_script
from .text_utils import clean_text
from .section_summarize import summarize_by_sections
from .tts import synthesize_to_wav, concat_wavs
from .ppt import create_ppt_from_summary
from .gemini_llm import GeminiConfig, generate_text, safe_strip


@dataclass
class PipelineOptions:
    ocr_fallback: bool = False
    max_pages: Optional[int] = None
    summary_sentences: int = 10
    ppt_theme: str = "professional"

    summarizer: Literal["tfidf", "textrank", "gemini"] = "tfidf"
    sentence_segmentation: Literal["regex", "spacy"] = "regex"
    keyword_algorithm: Literal["freq", "rake"] = "freq"
    min_keyword_freq: int = 2

    strip_citations: bool = True
    skip_tables: bool = True
    remove_headers_footers: bool = True

    target_reading_grade: Optional[float] = None
    podcast_target_minutes: Optional[float] = None
    speech_wpm: int = 140

    section_aware: bool = True
    max_sentences_per_section: int = 3
    include_references: bool = False

    # LLM (optional)
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-1.5-flash"


@dataclass
class PipelineOutputs:
    extracted_text_path: str
    raw_text_path: str
    summary_path: str
    simplified_path: str
    podcast_script_path: str
    podcast_audio_path: str
    pptx_path: str
    video_script_path: str
    analysis_json_path: str
    sentence_scores_csv_path: str


def run_pipeline(pdf_path: str | Path, out_dir: str | Path, opts: PipelineOptions | None = None) -> PipelineOutputs:
    if opts is None:
        opts = PipelineOptions()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Extract + preprocess
    pages = extract_pages_from_pdf(pdf_path, ocr_fallback=opts.ocr_fallback, max_pages=opts.max_pages)
    processed, raw = preprocess_pages(
        pages,
        PreprocessOptions(
            remove_repeated_headers_footers=opts.remove_headers_footers,
            strip_citations=opts.strip_citations,
            skip_table_like_lines=opts.skip_tables,
        ),
    )
    text = clean_text(processed)
    raw_text = clean_text(raw)

    extracted_path = out_dir / "extracted_text.txt"
    extracted_path.write_text(text, encoding="utf-8")

    raw_path = out_dir / "raw_text.txt"
    raw_path.write_text(raw_text, encoding="utf-8")

    # 2) Summary (+ debug)
    if opts.section_aware:
        sec = summarize_by_sections(
            text,
            total_sentences=opts.summary_sentences,
            summarizer=opts.summarizer,
            sentence_segmentation=opts.sentence_segmentation,
            max_sentences_per_section=opts.max_sentences_per_section,
            include_references=opts.include_references,
            gemini_api_key=opts.gemini_api_key,
            gemini_model=opts.gemini_model,
        )
        summary = sec.summary
        summary_debug = sec.summarizer_debug
        score_rows = sec.sentence_score_rows
    else:
        # fallback: whole-document mode via section summarizer on a single "Document" section
        sec = summarize_by_sections(
            text,
            total_sentences=opts.summary_sentences,
            summarizer=opts.summarizer,
            sentence_segmentation=opts.sentence_segmentation,
            max_sentences_per_section=None,
            include_references=True,
            gemini_api_key=opts.gemini_api_key,
            gemini_model=opts.gemini_model,
        )
        summary = sec.summary
        summary_debug = {**sec.summarizer_debug, "mode": "whole"}
        score_rows = sec.sentence_score_rows

    summary_path = out_dir / "summary.txt"
    summary_path.write_text(summary, encoding="utf-8")

    # 3) Simplified
    aggressiveness = 1
    if opts.target_reading_grade is not None and opts.target_reading_grade <= 8:
        aggressiveness = 3
    elif opts.target_reading_grade is not None and opts.target_reading_grade <= 10:
        aggressiveness = 2

    if opts.summarizer == "gemini":
        if not (opts.gemini_api_key or "").strip():
            raise ValueError("Gemini API key is required when summarizer='gemini'.")

        grade_hint = ""
        if opts.target_reading_grade is not None:
            grade_hint = f"Target reading level: grade {opts.target_reading_grade:.1f}.\n"

        prompt = (
            "Rewrite the following summary in simpler, clearer English.\n"
            + grade_hint
            + "Rules:\n"
            "- Plain text only\n"
            "- Keep all key facts\n"
            "- Short sentences, minimal jargon\n"
            "- Do not add new information\n\n"
            "SUMMARY:\n"
            + summary
        )

        simplified = safe_strip(
            generate_text(
                prompt,
                config=GeminiConfig(
                    api_key=opts.gemini_api_key,
                    model=opts.gemini_model,
                    temperature=0.25,
                    max_output_tokens=900,
                ),
            )
        )
    else:
        simplified = simplify_text(summary, aggressiveness=aggressiveness)
    simplified_path = out_dir / "simplified.txt"
    simplified_path.write_text(simplified, encoding="utf-8")

    # 4) Keywords (for scripts + analysis)
    kw_bundle = keyword_sets(text, algo=opts.keyword_algorithm, top_k=15, min_freq=opts.min_keyword_freq)
    keywords_only = [k["term"] for k in kw_bundle["keywords"]]

    # 5) Scripts
    # Generate up to 10 exchanges and optionally truncate to match target duration.
    exchanges = 10
    if opts.podcast_target_minutes is not None:
        exchanges = 10

    if opts.summarizer == "gemini":
        prompt = (
            "Create a natural conversation between a host and a guest expert about this research summary.\n"
            f"Write {int(exchanges)} back-and-forth exchanges.\n"
            "Rules:\n"
            "- Do NOT write 'Host:' or 'Expert:' or any speaker labels in the spoken text.\n"
            "- Each turn should be 1-2 sentences.\n"
            "- Keep it factual and faithful to the summary.\n"
            "- Avoid repeating the same opener every turn.\n"
            "Output format MUST be exactly:\n"
            "H: <line>\n"
            "E: <line>\n"
            "(repeat)\n\n"
            "SUMMARY:\n"
            + summary
        )

        raw_dialogue = safe_strip(
            generate_text(
                prompt,
                config=GeminiConfig(
                    api_key=opts.gemini_api_key or "",
                    model=opts.gemini_model,
                    temperature=0.35,
                    max_output_tokens=1200,
                ),
            )
        )

        host_lines = []
        expert_lines = []
        for line in raw_dialogue.splitlines():
            line = line.strip()
            if line.startswith("H:"):
                host_lines.append(line[2:].strip())
            elif line.startswith("E:"):
                expert_lines.append(line[2:].strip())
        # Fallback to rule-based if parsing fails
        if len(host_lines) < 3 or len(expert_lines) < 3:
            podcast_script = generate_podcast_script(
                summary=summary,
                keywords=keywords_only,
                keyphrases=keywords_only[:10],
                exchanges=exchanges,
            )
        else:
            from .scripts import PodcastScript

            podcast_script = PodcastScript(host_lines=host_lines[:exchanges], expert_lines=expert_lines[:exchanges])
    else:
        podcast_script = generate_podcast_script(
            summary=summary,
            keywords=keywords_only,
            keyphrases=keywords_only[:10],
            exchanges=exchanges,
        )

    podcast_initial_exchanges = len(podcast_script.host_lines)

    # Enforce podcast duration target (best-effort) by truncating exchanges
    if opts.podcast_target_minutes is not None:
        target_words = int(opts.podcast_target_minutes * opts.speech_wpm)
        # iteratively reduce until under target
        while True:
            podcast_text_tmp = podcast_script_to_text(podcast_script, include_speaker_labels=False)
            wc = len(podcast_text_tmp.split())
            if wc <= target_words or len(podcast_script.host_lines) <= 3:
                break
            podcast_script = type(podcast_script)(
                host_lines=podcast_script.host_lines[:-1],
                expert_lines=podcast_script.expert_lines[:-1],
            )
    podcast_text = podcast_script_to_text(podcast_script, include_speaker_labels=True)
    podcast_text_for_audio = podcast_script_to_text(podcast_script, include_speaker_labels=False)
    podcast_final_exchanges = len(podcast_script.host_lines)
    podcast_words = len(podcast_text_for_audio.split())
    podcast_estimated_minutes = (podcast_words / opts.speech_wpm) if opts.speech_wpm else None
    podcast_script_path = out_dir / "podcast_script.txt"
    podcast_script_path.write_text(podcast_text, encoding="utf-8")

    # 6) Audio (two voices if available; fallback to one)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    wavs = []
    for i, (h, e) in enumerate(zip(podcast_script.host_lines, podcast_script.expert_lines)):
        # Do NOT say the words "Host"/"Expert" in audio; rely on different voices for separation.
        wavs.append(synthesize_to_wav(h, audio_dir / f"host_{i}.wav", voice_name_contains="zira", rate=190))
        wavs.append(synthesize_to_wav(e, audio_dir / f"expert_{i}.wav", voice_name_contains="david", rate=175))

    podcast_audio_path = out_dir / "podcast.wav"
    try:
        concat_wavs(wavs, podcast_audio_path, silence_ms=650)
    except Exception:
        # If concat fails, create a single narration
        synthesize_to_wav(podcast_text_for_audio, podcast_audio_path, rate=180)

    # 7) PPT
    pptx_path = out_dir / "presentation.pptx"
    create_ppt_from_summary(
        summary,
        pptx_path,
        theme_name=opts.ppt_theme,
        title="Research Paper Summary",
        subtitle="Generated offline",
        keywords=keywords_only[:12],
    )

    # 8) Video script
    if opts.summarizer == "gemini":
        prompt = (
            "Write a short video script for a 30-45 second reel about this research summary.\n"
            "Format with these headings exactly: Hook:, Problem:, Simple idea:, Impact:, Call to action:\n"
            "Plain text only. No markdown. No emojis.\n\n"
            "SUMMARY:\n"
            + summary
        )
        video_script = safe_strip(
            generate_text(
                prompt,
                config=GeminiConfig(
                    api_key=opts.gemini_api_key or "",
                    model=opts.gemini_model,
                    temperature=0.35,
                    max_output_tokens=500,
                ),
            )
        )
    else:
        video_script = generate_video_script(summary, kind="reel")
    video_script_path = out_dir / "video_script.txt"
    video_script_path.write_text(video_script, encoding="utf-8")

    # 9) Analysis artifacts (introspection + metrics)
    evalm = compute_evaluation(text, summary, simplified, keywords_only[:10])

    podcast_hint = None
    if opts.podcast_target_minutes is not None:
        podcast_hint = {
            "target_minutes": opts.podcast_target_minutes,
            "speech_wpm": opts.speech_wpm,
            "target_words": int(opts.podcast_target_minutes * opts.speech_wpm),
        }

    analysis = {
        "options": {
            "summarizer": opts.summarizer,
            "sentence_segmentation": opts.sentence_segmentation,
            "keyword_algorithm": opts.keyword_algorithm,
            "min_keyword_freq": opts.min_keyword_freq,
            "strip_citations": opts.strip_citations,
            "skip_tables": opts.skip_tables,
            "remove_headers_footers": opts.remove_headers_footers,
            "target_reading_grade": opts.target_reading_grade,
            "podcast_target": podcast_hint,
            "section_aware": opts.section_aware,
            "max_sentences_per_section": opts.max_sentences_per_section,
            "include_references": opts.include_references,
        },
        "podcast": {
            "initial_exchanges": podcast_initial_exchanges,
            "final_exchanges": podcast_final_exchanges,
            "estimated_words": podcast_words,
            "estimated_minutes": podcast_estimated_minutes,
        },
        "summarizer": summary_debug,
        "keywords": kw_bundle,
        "metrics": {
            "full_text": linguistic_analysis(text),
            "summary": linguistic_analysis(summary),
            "simplified": linguistic_analysis(simplified),
        },
        "evaluation": {
            "compression_ratio": evalm.compression_ratio,
            "keyword_coverage": evalm.keyword_coverage,
            "reading_grade_before": evalm.reading_grade_before,
            "reading_grade_after": evalm.reading_grade_after,
        },
    }

    analysis_json_path = out_dir / "analysis.json"
    analysis_json_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")

    sentence_scores_csv_path = out_dir / "sentence_scores.csv"
    with open(sentence_scores_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["section", "sentence_index", "score", "selected", "sentence"])
        for r in score_rows:
            w.writerow([
                r.get("section", ""),
                r.get("sentence_index", ""),
                r.get("score", ""),
                r.get("selected", ""),
                r.get("sentence", ""),
            ])

    return PipelineOutputs(
        extracted_text_path=str(extracted_path),
        raw_text_path=str(raw_path),
        summary_path=str(summary_path),
        simplified_path=str(simplified_path),
        podcast_script_path=str(podcast_script_path),
        podcast_audio_path=str(podcast_audio_path),
        pptx_path=str(pptx_path),
        video_script_path=str(video_script_path),
        analysis_json_path=str(analysis_json_path),
        sentence_scores_csv_path=str(sentence_scores_csv_path),
    )
