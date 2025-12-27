# Research Assistant (Offline)

Local-first research paper assistant that takes a PDF and generates:

- Full-paper extractive summary (section-aware)
- Simplified version
- Podcast-style script + optional audio (TTS)
- A formatted PPT deck
- Optional video script + artifacts (scores/metrics)

No API keys required.

## Quickstart (Windows)

### 1) Create and activate a venv

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```powershell
pip install -r requirements.txt
```

### 3) Run the Streamlit app

```powershell
python -m streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`). Upload a PDF and download the generated artifacts.

## Outputs

Each run writes an output folder (and the app also offers a ZIP download) containing:

- `summary.txt`
- `simplified.txt`
- `podcast_script.txt`
- `podcast.wav` (if TTS succeeds)
- `presentation.pptx`
- `video_script.txt`
- `analysis.json`
- `sentence_scores.csv`

## How it works (high level)

1. **PDF extraction**: reads all pages (embedded text). If needed, falls back to OCR.
2. **Preprocessing**: de-hyphenation, header/footer removal, citation stripping, and common PDF artifact fixes.
3. **Summarization**: TF‑IDF + MMR diversity (or TextRank). Optional section-aware allocation to avoid “abstract-only” summaries.
4. **Deliverables**: simplified text, podcast script/audio, PPT deck, and introspection artifacts.

## Troubleshooting

### OCR fallback (pdf2image + Tesseract)

If your PDFs are scanned images, OCR is required:

- Install **Tesseract OCR** and ensure `tesseract` is on your PATH.
- Install **Poppler** for `pdf2image` (Windows) and ensure `pdftoppm` is on your PATH.

If OCR is missing, the app should still work for PDFs that already contain selectable text.

### Audio / video dependencies (FFmpeg)

`pydub` and `moviepy` rely on **FFmpeg**.

- Install FFmpeg and add it to PATH.
- If FFmpeg is missing, you may see warnings and audio/video generation can fail, but text outputs and PPT should still generate.

## Repo notes

- This project is intended to keep generated artifacts out of git (see `.gitignore`).
- If you want to ship a sample PDF, place it under an `examples/` folder and remove `*.pdf` from `.gitignore`.
