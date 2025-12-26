from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_AUTO_SIZE, PP_ALIGN
from pptx.util import Inches, Pt

from .text_utils import split_sentences


THEMES: Dict[str, Dict[str, object]] = {
    "professional": {
        "title_font": "Calibri",
        "title_size": Pt(40),
        "title_color": RGBColor(31, 78, 121),
        "title_bold": True,
        "body_font": "Calibri",
        "body_size": Pt(18),
        "body_color": RGBColor(64, 64, 64),
        "background_color": RGBColor(255, 255, 255),
    },
    "modern": {
        "title_font": "Arial",
        "title_size": Pt(38),
        "title_color": RGBColor(47, 84, 150),
        "title_bold": True,
        "body_font": "Arial",
        "body_size": Pt(16),
        "body_color": RGBColor(89, 89, 89),
        "background_color": RGBColor(248, 248, 248),
    },
    "creative": {
        "title_font": "Georgia",
        "title_size": Pt(38),
        "title_color": RGBColor(192, 80, 77),
        "title_bold": True,
        "body_font": "Georgia",
        "body_size": Pt(18),
        "body_color": RGBColor(79, 79, 79),
        "background_color": RGBColor(252, 248, 227),
    },
}


def _shorten_for_bullet(text: str, max_chars: int = 180) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    cut = t[: max_chars - 1]
    for sep in ("; ", ". ", ", ", ": ", " - "):
        idx = cut.rfind(sep)
        if idx >= int(max_chars * 0.6):
            cut = cut[:idx]
            break
    return cut.rstrip(" ,;:-") + "…"


def _apply_slide_background(slide, background_color: RGBColor) -> None:
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = background_color


def _add_top_bar(slide, color: RGBColor) -> None:
    bar = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        left=Inches(0),
        top=Inches(0),
        width=Inches(10),
        height=Inches(0.32),
    )
    bar.fill.solid()
    bar.fill.fore_color.rgb = color
    bar.line.fill.background()


def _add_title(slide, title: str, theme: Dict[str, object]) -> None:
    _apply_slide_background(slide, theme["background_color"])  # type: ignore
    _add_top_bar(slide, theme["title_color"])  # type: ignore

    box = slide.shapes.add_textbox(
        left=Inches(0.7),
        top=Inches(0.55),
        width=Inches(8.9),
        height=Inches(0.8),
    )
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = (title or "").strip() or "Slide"
    p.alignment = PP_ALIGN.LEFT
    p.font.name = theme["title_font"]  # type: ignore
    p.font.size = Pt(34)
    p.font.bold = True
    p.font.color.rgb = theme["title_color"]  # type: ignore


def _add_bullets(slide, bullets: Sequence[str], theme: Dict[str, object]) -> None:
    box = slide.shapes.add_textbox(
        left=Inches(0.85),
        top=Inches(1.5),
        width=Inches(8.95),
        height=Inches(5.6),
    )
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE

    added = 0
    for b in bullets:
        b = _shorten_for_bullet(b)
        if not b:
            continue
        p = tf.paragraphs[0] if added == 0 else tf.add_paragraph()
        p.text = b
        p.level = 0
        p.space_after = Pt(6)
        p.line_spacing = 1.1
        p.font.name = theme["body_font"]  # type: ignore
        p.font.size = Pt(20)
        p.font.color.rgb = theme["body_color"]  # type: ignore
        added += 1


def _add_footer(slide, text: str) -> None:
    box = slide.shapes.add_textbox(Inches(0.5), Inches(7.05), Inches(9), Inches(0.3))
    tf = box.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(10)
    p.font.color.rgb = RGBColor(150, 150, 150)
    p.alignment = PP_ALIGN.CENTER


def _bucket_sentences(summary: str) -> Dict[str, List[str]]:
    s = split_sentences(summary)
    if not s:
        return {"Context": [], "Method": [], "Results": [], "Limitations": [], "Conclusion": []}

    buckets = {"Context": [], "Method": [], "Results": [], "Limitations": [], "Conclusion": []}
    for sent in s:
        low = sent.lower()
        if any(k in low for k in ["method", "dataset", "experiment", "architecture", "model", "approach"]):
            buckets["Method"].append(sent)
        elif any(k in low for k in ["result", "accuracy", "improve", "outperform", "achieve", "metric"]):
            buckets["Results"].append(sent)
        elif any(k in low for k in ["limitation", "however", "constraint", "future work", "future"]):
            buckets["Limitations"].append(sent)
        elif any(k in low for k in ["conclude", "overall", "in summary", "we show", "this paper"]):
            buckets["Conclusion"].append(sent)
        else:
            buckets["Context"].append(sent)

    return buckets


def create_ppt_from_summary(
    summary: str,
    output_path: str | Path,
    theme_name: str = "professional",
    title: str = "Research Summary",
    subtitle: str | None = None,
    keywords: Sequence[str] | None = None,
) -> Path:
    theme = THEMES.get(theme_name, THEMES["professional"])
    out_pptx = Path(output_path)
    out_pptx.parent.mkdir(parents=True, exist_ok=True)

    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(7.5)

    buckets = _bucket_sentences(summary)
    summary_sents = [s.strip() for s in split_sentences(summary) if s.strip()]

    # 1) Title slide (clean, centered)
    s1 = prs.slides.add_slide(prs.slide_layouts[6])
    _apply_slide_background(s1, theme["background_color"])  # type: ignore
    _add_top_bar(s1, theme["title_color"])  # type: ignore

    title_box = s1.shapes.add_textbox(Inches(0.9), Inches(2.15), Inches(8.2), Inches(1.2))
    ttf = title_box.text_frame
    ttf.clear()
    ttf.word_wrap = True
    p = ttf.paragraphs[0]
    p.text = (title or "").strip() or "Research Summary"
    p.alignment = PP_ALIGN.CENTER
    p.font.name = theme["title_font"]  # type: ignore
    p.font.size = Pt(44)
    p.font.bold = True
    p.font.color.rgb = theme["title_color"]  # type: ignore

    sub_text = (subtitle or "Generated offline")
    sub_box = s1.shapes.add_textbox(Inches(1.1), Inches(3.45), Inches(7.8), Inches(0.6))
    stf = sub_box.text_frame
    stf.clear()
    sp = stf.paragraphs[0]
    sp.text = sub_text
    sp.alignment = PP_ALIGN.CENTER
    sp.font.name = theme["body_font"]  # type: ignore
    sp.font.size = Pt(22)
    sp.font.color.rgb = theme["body_color"]  # type: ignore

    # 2) Agenda
    s2 = prs.slides.add_slide(prs.slide_layouts[6])
    _add_title(s2, "Agenda", theme)
    _add_bullets(
        s2,
        [
            "Executive summary",
            "Research context",
            "Methodology",
            "Key findings",
            "Limitations",
            "Conclusion",
        ],
        theme,
    )

    # 3) Executive summary
    s3 = prs.slides.add_slide(prs.slide_layouts[6])
    _add_title(s3, "Executive Summary", theme)
    _add_bullets(s3, summary_sents[:6] if summary_sents else ["(No summary text provided)"] , theme)

    # 4-8) Core sections
    section_map = [
        ("Research Context", buckets["Context"]),
        ("Methodology", buckets["Method"]),
        ("Key Findings", buckets["Results"]),
        ("Limitations & Future Work", buckets["Limitations"]),
        ("Conclusion", buckets["Conclusion"]),
    ]
    for slide_title, bullets in section_map:
        if not bullets:
            continue
        s = prs.slides.add_slide(prs.slide_layouts[6])
        _add_title(s, slide_title, theme)
        _add_bullets(s, bullets[:7], theme)

    # 9) Keywords (optional)
    if keywords:
        kw = [k.strip() for k in keywords if k and k.strip()]
        if kw:
            s = prs.slides.add_slide(prs.slide_layouts[6])
            _add_title(s, "Keywords", theme)
            _add_bullets(s, kw[:12], theme)

    # Footer on all slides
    for slide in prs.slides:
        _add_footer(slide, "Offline Research Assistant")

    prs.save(str(out_pptx))
    return out_pptx
