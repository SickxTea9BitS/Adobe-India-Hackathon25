# main.py
import argparse
import multiprocessing as mp
from parsing.doc_tag_parser import parse_pdf_to_blocks
from embedding.embedder import EmbeddingEngine
from ranking.section_ranker import rank_sections
from chunking.subchunker import rank_chunks
from output.formatter import build_output_json
from utils.fast_filter import flatten_doc_blocks
from config import *

def parse_all_pdfs(pdf_model_input_list):
    """Parallel parse all input PDFs to tagged blocks."""
    results = []
    with mp.Pool(N_PARSE_THREADS) as pool:
        jobs = [pool.apply_async(parse_pdf_to_blocks, args=(pdf, tag_json, doc_nm))
                for (pdf, tag_json, doc_nm) in pdf_model_input_list]
        for j in jobs:
            results.extend(j.get())
    return results

def main():
    parser = argparse.ArgumentParser(description="Persona-driven document section analyst (Round1b)")
    parser.add_argument("--doc_inputs", nargs="+", metavar=('PDF', "TAGGED_JSON", "DOCNAME"),
                        help="List of tuples: PDF path, model-tagged-JSON-path, document name")
    parser.add_argument("--persona", required=True, help="Persona description string")
    parser.add_argument("--job", required=True, help="Job-to-be-done/task string")
    parser.add_argument("--outpath", required=True, help="Where to write output JSON")
    args = parser.parse_args()

    # Example for N documents: args.doc_inputs =
    #   sample1.pdf sample1_tagged.json "Sample 1" sample2.pdf sample2_tagged.json "Sample 2" ...

    # ---- Parse input PDF+JSON pairs
    pdfs = []
    for i in range(0, len(args.doc_inputs), 3):
        pdf_file, tag_json, doc_nm = args.doc_inputs[i:i+3]
        pdfs.append((pdf_file, tag_json, doc_nm))
    doc_blocks = parse_all_pdfs(pdfs)
    docs_metadata = [{"name": d[2], "pdf_path": d[0]} for d in pdfs]

    # ---- Flatten blocks for batch embedding/scoring
    blocks = flatten_doc_blocks(doc_blocks)

    # ---- Embedding setup
    embedder = EmbeddingEngine(EMBEDDING_MODEL_NAME)
    persona_job_prompt = f"{args.persona}\n\n{args.job}"
    prompt_embed = embedder.embed_one(persona_job_prompt)
    block_texts = []
    for b in blocks:
        if b["header_level"]:
            block_text = f"{b['tag_type']} {b['header_level']}: {b['text']}"
        else:
            block_text = f"{b['tag_type']}: {b['text']}"
        block_texts.append(block_text)

    block_embeds = embedder.embed_many(block_texts, batch_size=BATCH_EMBED_SIZE)

    # ---- Section-level ranking
    selected_sections = rank_sections(blocks, block_embeds, prompt_embed)

    # ---- Fine-grained chunking within each section
    sub_analysis_map = {}  # block_id -> list
    for sec in selected_sections:
        chunks = rank_chunks(sec["text"], embedder, prompt_embed)
        sub_analysis_map[sec["block_id"]] = chunks

    # ---- Output JSON
    output = build_output_json(docs_metadata, args.persona, args.job, selected_sections, sub_analysis_map)
    with open(args.outpath, "w", encoding="utf-8") as f:
        import json
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Extracted/Ranked analysis written to {args.outpath}")

if __name__ == "__main__":
    main()
