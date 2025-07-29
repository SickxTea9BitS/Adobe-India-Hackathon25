# Adobe India Hackathon 2025 – Connecting the Dots

## Overview

This repository contains the complete solution for the **Adobe India Hackathon 2025** focused on enhancing PDF interaction. Our AI-driven platform extracts document structure and provides persona- and task-driven insights across single or multiple PDFs, enabling an intelligent research companion experience.

The project consists of two main phases:

- **Round 1A – PDF Outline Extraction**  
  Extracts precise hierarchical outlines from PDFs by detecting Titles and Section Headers (H1–H3) using a custom YOLOv11x model combined with offline OCR. Outputs a clean JSON outlining document structure, serving as a foundation for downstream intelligence.

- **Round 1B – Persona-Driven Document Section Analyst**  
  Consumes Round 1A outputs (or equivalent tagged JSONs) plus persona and job/task prompts to rank and extract the most relevant document sections and fine-grained snippets leveraging lightweight semantic embeddings and similarity ranking.

Both pipelines are optimized for resource-constrained, fully offline execution with CPU-only compatibility.

---

## Main Features

- **Round 1A:**  
  - High-precision, YOLO-based detection of Titles and hierarchical section headers in PDFs  
  - Offline OCR (EasyOCR) for text extraction inside detected regions  
  - Outputs hierarchical JSON with document title and outline entries (level, text, page)

- **Round 1B:**  
  - Leverages tagged JSON outputs from Round 1A or similar  
  - Encodes persona and job prompts and document sections using `all-MiniLM-L6-v2` lightweight sentence transformer embeddings  
  - Ranks and extracts top relevant document sections and their most pertinent paragraph chunks  
  - Outputs enriched JSON detailing ranked sections and fine-grained textual snippets with relevance scores  

- **General:**  
  - Fully offline, CPU-only, and resource-efficient (≤1GB RAM)  
  - Supports multi-document, multilingual inputs  
  - Modular architecture enables independent or sequential use of each stage  
  - Containerized execution via Docker images for portability and reproducibility

---

## Technologies and Libraries

- **Programming Language:** Python 3.8 or later
- **PDF / Document Handling:** PyMuPDF (`fitz`), `pdfminer.six` (Round 1A uses PyMuPDF to render pages)  
- **Computer Vision / Detection:** YOLOv11x (custom-trained on DocLayNet) with PyTorch backend  
- **OCR:** EasyOCR (offline text extraction for detected regions)  
- **NLP:**  
  - `sentence-transformers` (all-MiniLM-L6-v2 for embeddings)  
  - `spaCy` (optional for entity/keyword extraction)  
  - `scikit-learn` (traditional ML for hierarchy classification in Round 1A)  
- **Data Handling:** pandas, numpy, json  
- **Other:** Docker (for containerization), Flask (optional web service in Round 1B), tqdm for progress display

---

## Setup Instructions

1. **Clone the repository:**

git clone https://github.com/SickxTea9BitS/Adobe-India-Hackathon25.git
cd Adobe-India-Hackathon25


2. **Build & Run per Round:**

- For **Round 1A (PDF Outline Extraction)**, please see `/Round1A/README.md`  
  Contains scripts, models, Dockerfile, and instructions for heading detection and outline JSON extraction.

- For **Round 1B (Persona-Driven Document Section Analyst)**, please see `/Round1B/README.md`  
  Details about consuming tag outputs, semantic ranking, subsection chunking, and generating prioritized JSON output.

---

## Team

- [Meet Vachhani](https://github.com/x0x-Lucifer-x0x) — Team Lead, ML Pipeline
- [Aromal Kunnel](https://github.com/SickxTea9BitS) — Backend & Dockerization
- [Sushil Shetty](https://github.com/SUSHIL-0711) — PDF Parsing & NLP

---

## Notes

- This repository is designed for seamless chaining: outputs of Round 1A feed into Round 1B.  
- All processing is strictly offline with no external API calls or internet dependencies at runtime.  
- Models and weights included/downloaded at build time to respect offline constraints.  
- Both rounds use optimized, modular Python pipelines enabling extensibility and rapid experimentation.


---

**Happy Document Understanding!**  
*—  TEAM AVINYAKRITI*

