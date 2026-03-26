import logging
import re

logger = logging.getLogger("dataprep.chunker")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into chunks of approximately chunk_size words with overlap.
    Tries to break at sentence boundaries when possible."""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    text = text.strip()

    if not text:
        return []

    words = text.split()
    total_words = len(words)

    if total_words <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < total_words:
        end = min(start + chunk_size, total_words)
        chunk_words = words[start:end]
        chunk_str = " ".join(chunk_words)

        # Try to break at sentence boundary (not at the very end of text)
        if end < total_words:
            last_period = chunk_str.rfind('. ')
            last_newline = chunk_str.rfind('\n')
            break_at = max(last_period, last_newline)

            if break_at > len(chunk_str) * 0.5:
                chunk_str = chunk_str[:break_at + 1].strip()

        if chunk_str:
            chunks.append(chunk_str)

        start = end - overlap
        if start >= total_words:
            break

    logger.info(f"Split {total_words} words into {len(chunks)} chunks "
                f"(size={chunk_size}, overlap={overlap})")
    return chunks
