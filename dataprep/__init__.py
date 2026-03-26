"""dataprep — text-to-dataset generation library.

Pure library. No FastAPI, no database, no config files.
Import and call from any context (server, CLI, notebook).

Usage:
    from dataprep import DatasetGenerator, extract_text, chunk_text

    text = extract_text("book.pdf")
    chunks = chunk_text(text, chunk_size=500, overlap=50)

    gen = DatasetGenerator(model_dir="./models")
    for chunk in chunks:
        pairs = gen.generate_pairs(chunk, pairs_per_chunk=3)
"""

from dataprep.text_extractor import extract_text
from dataprep.chunker import chunk_text
from dataprep.generator import DatasetGenerator, MODEL_REGISTRY
