# Adobe India Hackathon 2025 – Connecting the Dots

## Overview

This repository contains our complete solution for the **Adobe India Hackathon 2025**, focused on reimagining how people interact with PDFs. Our AI-driven platform extracts structure, surfaces personalized insights, and provides a research companion-like experience for a wide spectrum of users and documents.

The project has two main phases:

- **Round 1A:** Extracts structured outlines (Title, H1-H3) of a PDF in a hierarchical JSON format, forming the foundation for further document intelligence.
- **Round 1B:** Acts as an intelligent document analyst, extracting and prioritizing the most relevant sections across a collection of PDFs based on a persona and their specific task ("job-to-be-done").

## Repository Structure

- `/Round1A/` — Solution and documentation for heading and outline extraction (Round 1A)
- `/Round1B/` — Solution and documentation for persona-driven document intelligence (Round 1B)
- `README.md` — Main repository overview (this file)
- Other scripts and supporting files

## Main Features

- **AI-powered PDF outline (title/headings) extraction**
- **Persona & goal-driven intelligent section extraction from multiple documents**
- **Fast, offline, and modular architecture**
- **Handles various document domains and potential multilingual input**

## Technologies Used

- **Languages**: Python 3.x
- **Libraries**: 
  - PDF parsing: `PyMuPDF` (fitz), `pdfminer.six`
  - NLP: `spaCy`, `Transformers`, `sentence-transformers`, `scikit-learn`
  - Data handling: `pandas`, `json`, `datetime`
  - Others: `Flask` (for optional web/service layer), `Docker` for containerization
- **AI/ML Models**: Lightweight transformer and embedding models (model size constraints met)
- **No external API calls; entirely offline**

## Setup Instructions

1. **Clone the Repo:**
git clone https://github.com/SickxTea9BitS/Adobe-India-Hackathon25.git
cd Adobe-India-Hackathon25


2. **Build & run the solution for a specific round:**

- See `/Round1A/README.md` for outline extraction
- See `/Round1B/README.md` for persona-driven analysis

## Example Usage

Each round supports Dockerized, offline, resource-contained execution. See each round’s README for details and expected input/output.

## Team

<!-- Add contributor names here -->
- [Your Name]  
