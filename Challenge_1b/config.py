# config.py
import os

# Model config
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Candidate/Chunk limits & thresholds
MIN_SECTION_CHAR_LEN = 45
DROP_TAGS = set(["Page-footer", "Page-header", "Formula", "Picture"])
TOP_N_SECTIONS = 5
TOP_M_CHUNKS_PER_SECTION = 2

# Parallelism (tune as needed per hardware)
N_PARSE_THREADS = int(os.environ.get("N_PARSE_THREADS", 4))
N_EMBED_THREADS = int(os.environ.get("N_EMBED_THREADS", 4))
BATCH_EMBED_SIZE = 32
