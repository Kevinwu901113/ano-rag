from typing import Dict, List

from config import config
from utils import TextUtils


def make_chunks(doc_id: str, text: str, chunk_id_prefix: str = "p") -> List[Dict]:
    sentences = TextUtils.split_by_sentence(text)
    n_sent = int(config.get("chunk.n_sent", 2))
    overlap = int(config.get("chunk.overlap", 0))
    step = max(1, n_sent - overlap)

    chunks: List[Dict] = []
    idx = 0
    cursor = 0
    while cursor < len(sentences):
        group = sentences[cursor : cursor + n_sent]
        if not group:
            break
        chunk_text = " ".join(group)
        chunks.append(
            {
                "doc_id": doc_id,
                "chunk_id": f"{chunk_id_prefix}{idx:04d}",
                "text": chunk_text,
            }
        )
        idx += 1
        cursor += step
    return chunks
