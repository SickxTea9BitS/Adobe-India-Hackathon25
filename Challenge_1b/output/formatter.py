# output/formatter.py
import time

def build_output_json(docs_metadata, persona_desc, job_desc, selected_sections, sub_analysis_map):
    """
    Constructs the output JSON as specification: includes document names, page, rank, top chunks, etc.
    """
    output = {
        "persona_description": persona_desc,
        "job_to_be_done": job_desc,
        "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S'),
        "documents": docs_metadata,
        "extracted_sections": [],
    }
    for sec in selected_sections:
        obj = {
            "document": sec["document"],
            "section_title": sec.get("text", ""),
            "section_level": sec.get("header_level", None),
            "page_number": sec["page_number"],
            "importance_rank": sec["importance_rank"],
            "similarity_score": sec["similarity_score"],
            "subsection_analysis": sub_analysis_map.get(sec["block_id"], []),
        }
        output["extracted_sections"].append(obj)
    return output
