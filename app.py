from __future__ import annotations

import io
import json
import tempfile
import zipfile
from pathlib import Path

import streamlit as st

from offline_research_assistant.pipeline import run_pipeline, PipelineOptions
from offline_research_assistant.video import overlay_audio_on_video


st.set_page_config(page_title="Research Assistant", layout="wide")

st.title("Research Assistant")
st.write("Runs locally. Optional Gemini backend for higher-quality outputs.")

with st.sidebar:
    st.header("Options")
    summary_sentences = st.slider("Summary length (sentences)", 5, 20, 10, 1)
    ocr_fallback = st.checkbox("Use OCR fallback (slower)", value=False)
    ppt_theme = st.selectbox("PPT theme", ["professional", "modern", "creative"], index=0)
    max_pages = st.number_input("Max pages (0 = all)", min_value=0, max_value=500, value=0, step=1)

    st.divider()
    st.subheader("NLP")
    summarizer = st.selectbox("Summarizer", ["tfidf", "textrank", "gemini"], index=0)
    sentence_segmentation = st.selectbox("Sentence segmentation (TF-IDF)", ["regex", "spacy"], index=0)
    keyword_algorithm = st.selectbox("Keyword extraction", ["freq", "rake"], index=0)
    min_keyword_freq = st.number_input("Min keyword frequency (freq mode)", min_value=1, max_value=20, value=2, step=1)

    gemini_api_key = None
    gemini_model = "gemini-1.5-flash"
    if summarizer == "gemini":
        st.divider()
        st.subheader("Gemini")
        gemini_api_key = st.text_input("Gemini API key", type="password", help="Used only for this run; not saved.")
        gemini_model = st.selectbox("Gemini model", ["gemini-1.5-flash", "gemini-1.5-pro"], index=0)

    st.divider()
    st.subheader("Robustness")
    remove_headers_footers = st.checkbox("Remove repeated headers/footers", value=True)
    strip_citations = st.checkbox("Strip citation markers", value=True)
    skip_tables = st.checkbox("Skip table-like lines", value=True)

    st.divider()
    st.subheader("Constraints")
    target_reading_grade = st.number_input("Target reading grade (0 = off)", min_value=0.0, max_value=20.0, value=0.0, step=0.5)
    podcast_target_minutes = st.number_input("Podcast target minutes (0 = off)", min_value=0.0, max_value=60.0, value=0.0, step=0.5)

    st.divider()
    st.subheader("Structure")
    section_aware = st.checkbox("Section-aware summarization", value=True)
    max_sentences_per_section = st.slider("Max sentences per section", 1, 8, 3, 1)
    include_references = st.checkbox("Include References section", value=False)

pdf_file = st.file_uploader("Upload a research paper PDF", type=["pdf"])
video_file = st.file_uploader("Optional: Upload a video to overlay audio", type=["mp4", "mov", "mkv"])  # optional

if pdf_file is None:
    st.stop()

run = st.button("Generate Outputs")

if not run:
    st.stop()

with st.status("Processing…", expanded=True) as status:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        in_pdf = tmp_path / "input.pdf"
        in_pdf.write_bytes(pdf_file.getvalue())

        status.write("Extracting + summarizing…")
        opts = PipelineOptions(
            ocr_fallback=ocr_fallback,
            max_pages=(None if int(max_pages) == 0 else int(max_pages)),
            summary_sentences=int(summary_sentences),
            ppt_theme=ppt_theme,
            summarizer=summarizer,
            sentence_segmentation=sentence_segmentation,
            keyword_algorithm=keyword_algorithm,
            min_keyword_freq=int(min_keyword_freq),
            strip_citations=strip_citations,
            skip_tables=skip_tables,
            remove_headers_footers=remove_headers_footers,
            target_reading_grade=(None if float(target_reading_grade) == 0.0 else float(target_reading_grade)),
            podcast_target_minutes=(None if float(podcast_target_minutes) == 0.0 else float(podcast_target_minutes)),
            section_aware=bool(section_aware),
            max_sentences_per_section=int(max_sentences_per_section),
            include_references=bool(include_references),
            gemini_api_key=(gemini_api_key or None),
            gemini_model=str(gemini_model),
        )

        out_dir = tmp_path / "outputs"
        outputs = run_pipeline(in_pdf, out_dir, opts=opts)

        status.write("Preparing downloads…")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Preview")
            summary_text = Path(outputs.summary_path).read_text(encoding="utf-8")
            st.text_area("Summary", summary_text, height=220)

            simplified_text = Path(outputs.simplified_path).read_text(encoding="utf-8")
            st.text_area("Simplified", simplified_text, height=220)

            st.subheader("Metrics")
            analysis = json.loads(Path(outputs.analysis_json_path).read_text(encoding="utf-8"))
            m_sum = analysis.get("metrics", {}).get("summary", {}).get("readability", {})
            m_simp = analysis.get("metrics", {}).get("simplified", {}).get("readability", {})
            evalm = analysis.get("evaluation", {})

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Compression ratio", f"{evalm.get('compression_ratio', 0.0):.3f}")
            with c2:
                st.metric("Keyword coverage", f"{evalm.get('keyword_coverage', 0.0):.2f}")
            with c3:
                st.metric("Grade shift", f"{evalm.get('reading_grade_before', 0.0):.1f} → {evalm.get('reading_grade_after', 0.0):.1f}")

            st.caption("Readability (summary vs simplified)")
            st.write(
                {
                    "summary_flesch": round(float(m_sum.get("flesch_reading_ease", 0.0)), 2),
                    "summary_grade": round(float(m_sum.get("flesch_kincaid_grade", 0.0)), 2),
                    "simplified_flesch": round(float(m_simp.get("flesch_reading_ease", 0.0)), 2),
                    "simplified_grade": round(float(m_simp.get("flesch_kincaid_grade", 0.0)), 2),
                }
            )

            st.subheader("Introspection")
            st.caption("Top keywords")
            st.write([k.get("term") for k in analysis.get("keywords", {}).get("keywords", [])[:12]])

            if analysis.get("summarizer", {}).get("algorithm") == "tfidf":
                st.caption("Top TF-IDF terms")
                st.write([t.get("term") for t in (analysis.get("summarizer", {}).get("top_terms") or [])[:12]])

            sec_info = analysis.get("summarizer", {}).get("sections")
            if isinstance(sec_info, list) and sec_info:
                st.caption("Section contribution")
                st.dataframe(
                    [
                        {
                            "section": s.get("section"),
                            "words": s.get("words"),
                            "allocated_sentences": s.get("allocated_sentences"),
                            "selected_count": len(s.get("selected_indices") or []),
                        }
                        for s in sec_info
                    ],
                    use_container_width=True,
                )

        with col2:
            st.subheader("Downloads")

            # Bundle everything into a single ZIP for one-click download
            bundle = io.BytesIO()
            with zipfile.ZipFile(bundle, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
                z.write(outputs.extracted_text_path, arcname="extracted_text.txt")
                z.write(outputs.raw_text_path, arcname="raw_text.txt")
                z.write(outputs.summary_path, arcname="summary.txt")
                z.write(outputs.simplified_path, arcname="simplified.txt")
                z.write(outputs.podcast_script_path, arcname="podcast_script.txt")
                z.write(outputs.podcast_audio_path, arcname="podcast.wav")
                z.write(outputs.pptx_path, arcname="presentation.pptx")
                z.write(outputs.video_script_path, arcname="video_script.txt")
                z.write(outputs.analysis_json_path, arcname="analysis.json")
                z.write(outputs.sentence_scores_csv_path, arcname="sentence_scores.csv")

                maybe_video = out_dir / "final_video.mp4"
                if maybe_video.exists():
                    z.write(str(maybe_video), arcname="final_video.mp4")

            bundle.seek(0)
            st.download_button(
                "Download ALL outputs (ZIP)",
                data=bundle.getvalue(),
                file_name="research_assistant_outputs.zip",
                mime="application/zip",
            )

            st.download_button(
                "Download extracted text",
                data=Path(outputs.extracted_text_path).read_bytes(),
                file_name="extracted_text.txt",
            )
            st.download_button(
                "Download raw text (pre-clean)",
                data=Path(outputs.raw_text_path).read_bytes(),
                file_name="raw_text.txt",
            )
            st.download_button(
                "Download summary",
                data=Path(outputs.summary_path).read_bytes(),
                file_name="summary.txt",
            )
            st.download_button(
                "Download simplified",
                data=Path(outputs.simplified_path).read_bytes(),
                file_name="simplified.txt",
            )
            st.download_button(
                "Download podcast script",
                data=Path(outputs.podcast_script_path).read_bytes(),
                file_name="podcast_script.txt",
            )
            st.download_button(
                "Download podcast audio (WAV)",
                data=Path(outputs.podcast_audio_path).read_bytes(),
                file_name="podcast.wav",
            )
            st.download_button(
                "Download presentation (PPTX)",
                data=Path(outputs.pptx_path).read_bytes(),
                file_name="presentation.pptx",
            )
            st.download_button(
                "Download video script",
                data=Path(outputs.video_script_path).read_bytes(),
                file_name="video_script.txt",
            )

            st.download_button(
                "Download NLP analysis (JSON)",
                data=Path(outputs.analysis_json_path).read_bytes(),
                file_name="analysis.json",
            )
            st.download_button(
                "Download sentence scores (CSV)",
                data=Path(outputs.sentence_scores_csv_path).read_bytes(),
                file_name="sentence_scores.csv",
            )

        if video_file is not None:
            status.write("Overlaying audio on video…")
            in_video = tmp_path / "input_video.mp4"
            in_video.write_bytes(video_file.getvalue())
            out_video = out_dir / "final_video.mp4"
            overlay_audio_on_video(in_video, outputs.podcast_audio_path, out_video)

            st.subheader("Video")
            st.video(out_video.read_bytes())
            st.download_button(
                "Download final video",
                data=out_video.read_bytes(),
                file_name="final_video.mp4",
            )

        status.update(label="Done", state="complete", expanded=False)
