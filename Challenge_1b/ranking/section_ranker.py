# ranking/section_ranker.py
import numpy as np
from config import DROP_TAGS, TOP_N_SECTIONS, MIN_SECTION_CHAR_LEN

def score_one_block(block_embed, prompt_embed):
    return float(np.dot(block_embed, prompt_embed))  # cosine similarity (embeddings normalized)

def rank_sections(blocks, block_embeds, prompt_embed):
    """Returns top-N ranked sections with all relevant metadata."""
    scored_blocks = []
    for block, embed in zip(blocks, block_embeds):
        if block['tag_type'] in DROP_TAGS:
            continue
        if len(block['text']) < MIN_SECTION_CHAR_LEN:
            continue
        s = score_one_block(embed, prompt_embed)
        scored_blocks.append((s, block))
    # Sort by score descending; remove near-duplicate sections by block_id/text hash if needed
    scored_blocks = sorted(scored_blocks, key=lambda x: -x[0])
    results = []
    seen = set()
    for rank, (score, block) in enumerate(scored_blocks):
        hash_ = hash(block["text"][:80])
        if hash_ in seen:
            continue
        seen.add(hash_)
        d = dict(block)
        d["importance_rank"] = rank + 1
        d["similarity_score"] = score
        results.append(d)
        if len(results) == TOP_N_SECTIONS:
            break
    return results
