# Round 1B – Persona-Driven Document Section Analyst

## Overview

This project implements a **persona- and task-aware document analysis pipeline** that processes PDF documents annotated with tagged text blocks, ranks and extracts the most relevant document sections and their subsections, and outputs a structured JSON suitable for downstream consumption.

It builds upon pre-annotated PDFs (e.g., from Round 1A outputs) and utilizes semantic text embeddings and similarity ranking to deliver contextually relevant, hierarchy-aware extractions tailored to specific personas and tasks.

Designed for **CPU-only deployment (<1GB RAM)** and optimized for throughput (typically under 20 seconds for 3–5 PDFs), this solution requires **no model training** and uses lightweight, high-performance models.

## Key Features

- Leverages tagged JSON outputs from document parsing/tagging models that annotate titles, section headers (H1, H2, H3), lists, footnotes, etc.
- Ranks sections using semantic similarity with a persona + job description prompt.
- Fine-grained ranking of paragraph chunks inside sections for detailed snippet extraction.
- Modular and extensible architecture for easy integration.
- Configurable parallelism and batching for speed optimization on CPU.
- Produces clean, annotated JSON with document metadata, relevance scores, and hierarchical structure.

## Project Structure

round1b_doc_analyst/
├── main.py # Pipeline entry-point CLI script
├── config.py # Configuration constants and hyperparameters
├── parsing/
│ └── doc_tag_parser.py # Loading and preparing tagged blocks JSON
├── embedding/
│ └── embedder.py # SentenceTransformer embedding wrapper
├── ranking/
│ └── section_ranker.py # Persona-aware section ranking logic
├── chunking/
│ └── subchunker.py # Paragraph chunking and ranking inside sections
├── output/
│ └── formatter.py # Output JSON assembly
├── utils/
│ └── fast_filter.py # Utility methods (flattening, deduplication)
├── requirements.txt # Python dependencies
├── README.md # This documentation

```bash
# This command builds the image and tags it as 'challenge1b-analyzer'
docker build -t sickxtea9bits.challenge1b-analyzer .
```
### Run Command
```bash
docker run --rm -v "$(pwd)/Collection 1:/app/input_output" sickxtea9bits.challenge1b-analyzer --input /app/input_output/challenge1b_input.json --output /app/input_output/challenge1b_output.json

```



## Input Details

- **`--doc_inputs`**: Triplets of PDF file path, corresponding tagged JSON path, and document name. Provide one triplet per document.
- **`--persona`**: Natural language description of the user profile.
- **`--job`**: Task or job description specifying extraction focus.
- **`--outpath`**: Output JSON file path.

Tagged JSON files are expected to contain blocks with:
- Page number
- Tag type (e.g., Title, Section-header)
- Extracted text
- Header level information (optional)
- Unique block identifiers

## Output

A JSON file containing:

- Persona and job descriptions.
- List of processed documents.
- An array of extracted sections ranked by relevance with metadata:
  - Document name
  - Section title and header level
  - Page number
  - Importance rank and similarity score
  - Subsection analysis with top paragraph snippets and scores

Example snippet:

{
"persona_description": "Senior Research Scientist",
"job_to_be_done": "Extract catalytic sections",
"timestamp": "...",
"documents": [{"name": "Doc1", "pdf_path": "..."}],
"extracted_sections": [
{
"document": "Doc1",
"section_title": "Catalytic Reaction Mechanisms",
"section_level": 1,
"page_number": 5,
"importance_rank": 1,
"similarity_score": 0.87,
"subsection_analysis": [
{"score": 0.91, "refined_text": "..."},
{"score": 0.89, "refined_text": "..."}
]
}
]
}

text

## Notes

- Designed to integrate seamlessly with Round 1A outputs or other similar PDF tag extraction methods.
- Fully CPU compatible and optimized for low memory and runtime efficiency.
- Parallel processing and batch embedding are configurable via `config.py`.

---
