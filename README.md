# Ecuador News NLP Pipeline (Docling-based)

Research-oriented document processing pipeline for large-scale historical Ecuadorian news archives (2001–2022).

This project implements and empirically compares two document-to-text workflows:

1. Baseline OCR pipeline (Surya-based extraction)
2. Docling-based structured document parsing

The Docling-based workflow was adopted after systematic observation of cleaner text outputs, improved structural preservation, and more stable chunk segmentation for downstream NLP tasks such as embedding-based retrieval.

---

## Motivation

Historical newspaper archives present significant challenges for NLP:

- Noisy OCR artifacts
- Layout distortions
- Broken paragraph boundaries
- Inconsistent structural metadata

The goal of this pipeline is to transform raw document outputs into high-quality, retrieval-ready text representations suitable for large-scale semantic indexing.

---

## Pipeline Overview

### 1. Extraction
- `docling_originals_to_ministral.py`
- Baseline: `split_surya_results_to_txt.py`

Structured parsing (Docling) or OCR-based extraction.

### 2. Cleaning
- `post_clean_docling.py`
- Baseline: `post_clean_txts.py`

Noise reduction, normalization, and structural cleanup.

### 3. Chunking
- `chunk_texts.py`

Semantic segmentation with configurable overlap to improve retrieval robustness.

### 4. Quality Filtering
- `filter_chunks.py`

Heuristic-based scoring to remove low-quality or structurally unstable chunks.

### 5. Embedding
- `embed_faiss.py`

Vector representation and FAISS indexing for similarity-based retrieval.

### 6. Retrieval & Visualization
- `query.py`
- `app.py`

Similarity search interface and interactive querying layer.

---

## Architecture

The pipeline is modular and stage-based:

Extraction → Cleaning → Chunking → Filtering → Embedding → Retrieval → Application

Each stage is independently executable, enabling:

- Controlled experimentation
- Pipeline ablation studies
- Reproducibility
- Comparative evaluation between workflows

---

## Dataset Context

The system was designed to process large-scale Ecuadorian newspaper archives spanning 2001–2022.

Experiments were conducted on subsets ranging from thousands to tens of thousands of documents to evaluate:

- Text cleanliness
- Structural preservation
- Chunk coherence
- Downstream retrieval stability

---

## Why Docling?

Empirical comparison against the baseline OCR pipeline revealed:

- Reduced layout artifacts
- More coherent paragraph segmentation
- Improved structural consistency
- Higher-quality chunk boundaries

These improvements directly impacted embedding stability and semantic retrieval performance.

---

## Notes

This repository contains the pipeline implementation only. Original datasets, OCR outputs, and FAISS indices are excluded for size and licensing reasons.
