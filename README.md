# Adobe India Hackathon 2025

## Overview
This repository contains solutions for the Adobe India Hackathon 2025, featuring advanced PDF processing and analysis capabilities. The project demonstrates innovative approaches to document understanding, content extraction, and intelligent information retrieval.

## Project Structure
```
Adobe-India-Hackathon25/
├── Challenge_1a/                    # Basic PDF Processing Solution
│   ├── dataset/
│   │   ├── outputs/                # Generated JSON files
│   │   └── pdfs/                   # Input PDF files
│   ├── Dockerfile                  # Container configuration
│   ├── process_pdfs.py            # Processing script
│   └── README.md                  # Challenge 1a documentation
├── Challenge_1b/                    # Advanced Multi-Collection Analysis
│   ├── Collection 1/              # Travel Planning Collection
│   ├── Collection 2/              # Adobe Acrobat Learning Collection
│   ├── Collection 3/              # Recipe Collection
│   └── README.md                  # Challenge 1b documentation
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

## Challenges

### [Challenge 1a: PDF Processing Solution](./Challenge_1a/README.md)
**Focus**: Basic PDF processing with Docker containerization

**Key Features**:
- PDF document processing and structure extraction
- JSON output generation with hierarchical data
- Docker containerization for easy deployment
- Batch processing capabilities

**Technology Stack**:
- Python 3.10
- Docker
- JSON data formatting

**Use Cases**:
- Document structure analysis
- Content extraction from PDFs
- Automated document processing pipelines

### [Challenge 1b: Multi-Collection PDF Analysis](./Challenge_1b/README.md)
**Focus**: Advanced persona-based content analysis

**Key Features**:
- Multi-collection document processing
- Persona-based content filtering
- Importance ranking algorithms
- Refined text extraction and analysis

**Technology Stack**:
- Advanced PDF processing
- Natural language processing
- Persona-based filtering
- Structured JSON analysis

**Use Cases**:
- Travel planning content analysis
- Learning material organization
- Recipe and meal planning
- Intelligent content recommendation

## Collections in Challenge 1b

### Collection 1: Travel Planning
- **Persona**: Travel Planner
- **Documents**: 7 South of France travel guides
- **Use Case**: Plan a 4-day trip for 10 college friends

### Collection 2: Adobe Acrobat Learning
- **Persona**: Learning Professional  
- **Documents**: 15 Acrobat tutorials and guides
- **Use Case**: Comprehensive Acrobat training and skill development

### Collection 3: Recipe Collection
- **Persona**: Home Chef
- **Documents**: 9 cooking and meal planning guides
- **Use Case**: Meal planning and recipe discovery

## Technical Highlights

### Innovation Features
1. **Persona-Based Analysis**: Content extraction tailored to specific user types
2. **Importance Ranking**: Intelligent prioritization of relevant content
3. **Multi-Format Support**: Handles diverse document collections
4. **Scalable Architecture**: Docker-based deployment for easy scaling
5. **Structured Output**: Consistent JSON formatting for downstream processing

### Advanced Capabilities
- **Content Contextualization**: Understanding document purpose and audience
- **Hierarchical Extraction**: Maintaining document structure and relationships
- **Quality Assurance**: Validation and verification of extracted content
- **Performance Optimization**: Efficient processing of large document sets

## Getting Started

### Prerequisites
- Python 3.10+
- Docker (for Challenge 1a)
- Access to PDF documents for processing

### Quick Start
1. **Challenge 1a**: Follow the [Challenge 1a README](./Challenge_1a/README.md) for basic PDF processing
2. **Challenge 1b**: Follow the [Challenge 1b README](./Challenge_1b/README.md) for advanced analysis

### Running the Solutions
```bash
# Challenge 1a - Docker-based processing
cd Challenge_1a
docker build -t pdf-processor .
docker run -v /path/to/input:/app/input -v /path/to/output:/app/output pdf-processor

# Challenge 1b - Multi-collection analysis
cd Challenge_1b
python process_collection.py --collection 1 --persona "Travel Planner"
```

## Output Examples

### Challenge 1a Output
```json
{
  "title": "Document Title",
  "outline": [
    {
      "level": "H1",
      "text": "Section Title",
      "page": 1
    }
  ]
}
```

### Challenge 1b Output
```json
{
  "metadata": {
    "persona": "Travel Planner",
    "job_to_be_done": "Plan a trip of 4 days for a group of 10 college friends"
  },
  "extracted_sections": [
    {
      "section_title": "Comprehensive Guide to Major Cities",
      "importance_rank": 1,
      "page_number": 1
    }
  ]
}
```

## Development Roadmap

### Planned Enhancements
- **Machine Learning Integration**: Advanced content ranking and classification
- **Multi-Language Support**: Processing documents in multiple languages
- **Real-Time Processing**: Stream processing capabilities
- **API Development**: RESTful APIs for integration
- **User Interface**: Web-based document analysis interface

### Future Capabilities
- **OCR Enhancement**: Improved text extraction from scanned documents
- **Content Summarization**: AI-powered document summarization
- **Interactive Analysis**: Real-time content exploration tools
- **Integration APIs**: Connect with content management systems

## Contributing
This project was developed for the Adobe India Hackathon 2025. For questions or contributions, please refer to the individual challenge documentation.

## License
This project is developed for the Adobe India Hackathon 2025 and follows the hackathon's terms and conditions.

---

**Note**: Each challenge directory contains detailed documentation and specific implementation details. Please refer to the individual README files for comprehensive information about each solution.