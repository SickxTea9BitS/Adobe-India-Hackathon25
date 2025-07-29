# chunking/subchunker.py
import re
from config import TOP_M_CHUNKS_PER_SECTION

def paragraph_chunks(text):
    # Naive split on double newline, fallback to single
    paras = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    if len(paras) == 1:
        paras = [p.strip() for p in text.split('\n') if p.strip()]
    return paras

def rank_chunks(section_text, embedder, prompt_embed):
    paras = paragraph_chunks(section_text)
    if not paras:
        return []
    para_embeds = embedder.embed_many(paras)
    scores = [float((e @ prompt_embed)) for e in para_embeds]
    ranked = sorted(zip(scores, paras), key=lambda x: -x[0])
    selected = []
    seen = set()
    for score, chunk_text in ranked:
        key = chunk_text[:60]
        if key in seen:
            continue
        seen.add(key)
        selected.append({"score": score, "refined_text": chunk_text})
        if len(selected) == TOP_M_CHUNKS_PER_SECTION:
            break
    return selected
