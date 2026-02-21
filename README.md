# Ecuador News NLP Pipeline (Docling-based)

Research-oriented document processing pipeline for large-scale historical Ecuadorian news archives.

This project implements and compares two document-to-text workflows:

1. Baseline OCR pipeline (Surya-based extraction)
2. Docling-based structured extraction

The Docling-based pipeline was adopted after empirical observation of significantly cleaner text, better structural preservation, and higher-quality chunk segmentation for downstream NLP tasks.

---

## Pipeline Overview

### 1. Extraction
- `docling_originals_to_ministral.py`
- Baseline: `split_surya_results_to_txt.py`

### 2. Cleaning
- `post_clean_docling.py`
- Baseline: `post_clean_txts.py`

### 3. Chunking
- `chunk_texts.py`

### 4. Quality Filtering
- `filter_chunks.py`

### 5. Embedding (optional downstream step)
- `embed_faiss.py`

### 6. Retrieval & Visualization
- `query.py`
- `app.py`

---

## Why Docling?

The baseline OCR pipeline produced noisier outputs, layout artifacts, and less stable segmentation. The Docling-based workflow significantly improved structural consistency and reduced noise, resulting in higher-quality chunks for embedding and retrieval.

---

## Notes

This repository shares the pipeline code only. Original datasets, OCR outputs, and FAISS indices are excluded.
