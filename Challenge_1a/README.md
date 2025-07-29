# Round 1A – PDF Outline Extraction

## Overview

This project extracts a structured **outline** from PDF documents by detecting Titles and Section Headers using a custom-trained YOLOv11x model (trained on the DocLayNet dataset) combined with offline OCR (EasyOCR). Heading levels (H1, H2, H3, etc.) are precisely assigned using both detected numbering patterns and structural heuristics. The output is a JSON file containing the document title and a hierarchical outline with heading levels and page numbers.

The solution is **fully offline-ready** and designed to run efficiently within a Docker container.

---

## Approach

- **Detection**: Uses a custom YOLOv11x model trained on DocLayNet to detect "Title" and "Section-header" boxes in high-resolution renders of each PDF page.
- **OCR**: For each detected bounding box, text is extracted using EasyOCR with local pre-downloaded model weights (no network required).
- **Hierarchy Assignment**:
    - Recognizes unlimited-depth headings via numbering (e.g., `1.`, `1.1.`, `1.1.1.`, producing H1–Hn).
    - Falls back to rules based on bounding box area and vertical position if numbering is absent.
- **Batch Processing**: All PDFs found in the input directory are processed in one run, and each receives its own outline JSON.

---

## Models and Libraries Used

- **YOLOv11x (Ultralytics)** — for layout detection (titles/section-headers)
- **DocLayNet dataset** — for model training
- **PyMuPDF (`fitz`)** — converts PDF pages to images for detection
- **EasyOCR** — for text extraction from detected regions (runs purely offline)
- **Torch, OpenCV, NumPy, PIL** — vision and data utilities
- **Python Built-ins** — `os`, `pathlib`, `json`, `re`, etc.

---

### Build Command
```bash
docker build --platform linux/amd64 -t sickxtea9bits.adobe-challenge-1a .
```

### Run Command
```bash
docker run --rm -v $(pwd)/input:/app/input:ro -v $(pwd)/output/repoidentifier/:/app/output docker run --rm -v $(pwd)/input:/app/input:ro -v $(pwd)/output/document-outline-extractor/:/app/output --network none sickxtea9bits.adobe-challenge-1a

```


**Notes:**
- Place PDFs to be processed inside the local `./input` directory.
- Output outline JSON files will be written into `./output/document-outline-extractor/`.
- The container runs fully offline and expects both YOLO model and EasyOCR weights to be present at build time in `/model/` (see Dockerfile).
- The `:ro` flag on your input mount ensures PDFs are read-only from inside the container for safety.
- `--network none` ensures strictly offline execution.

---

## Output Example

After running, for each input PDF (e.g., `thesis.pdf`), you’ll find an outline file:

output/document-outline-extractor/thesis.json


Sample structure:

{
"title": "Sample Document Title",
"outline": [
{ "level": "H1", "text": "Introduction", "page": 1 },
{ "level": "H2", "text": "Background", "page": 2 },
{ "level": "H3", "text": "Detailed History", "page": 3 }
]
}


---

## Troubleshooting

- Ensure YOLOv11x and EasyOCR model weights are present in `/model/` and `/model/easyocr/` inside the Docker image.
- Check volume paths and that PDFs are present in your local `./input/` directory.
- Error and process messages are printed in the container logs for each file processed.

---
