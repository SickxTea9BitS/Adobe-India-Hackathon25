# parsing/doc_tag_parser.py
import fitz
import json

def parse_pdf_to_blocks(pdf_path: str, model_tagged_json_path: str, doc_name: str) -> list:
    """
    Loads the output from your custom model (tagged elements per page, JSON format)
    and returns canonical block dicts per element with: doc, page, tag, (if sec) header level, text, etc.
    """
    with open(model_tagged_json_path, "r", encoding="utf-8") as f:
        tag_data = json.load(f)  # [{page, tag, ...}, ...]

    blocks = []
    for item in tag_data:
        tag_type = item.get("tag_type")
        if tag_type in ["Section-header", "Title"]:
            header_level = item.get("header_level", None)
        else:
            header_level = None

        block = {
            'document': doc_name,
            'page_number': item["page"],
            'tag_type': tag_type,
            'header_level': header_level,
            'text': item["text"],
            'block_id': f'{doc_name}|{item["page"]}|{item.get("block_number", 0)}',
        }
        blocks.append(block)
    return blocks
