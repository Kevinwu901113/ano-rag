from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from relrag.doc.chunker import make_chunks

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

class Chunker(ABC):
    @abstractmethod
    def chunk(self, doc_id: str, text: str, doc_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split document into chunks.
        
        Args:
            doc_id: Document ID.
            text: Full document text.
            doc_meta: Document metadata (may contain pre-split sentences).
            
        Returns:
            List of chunk dicts with keys: chunk_id, doc_id, text, meta.
        """
        pass

class SentenceAwareChunker(Chunker):
    """
    Sentence-aware chunking with sentence overlap.
    Uses relrag.doc.chunker.make_chunks (n_sent/overlap from relrag config).
    """
    def chunk(self, doc_id: str, text: str, doc_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        title = str(doc_meta.get("title") or doc_id)
        sentences = doc_meta.get("sentences")
        if isinstance(sentences, list) and sentences:
            source_text = " ".join(str(s).strip() for s in sentences if str(s).strip()).strip()
        else:
            source_text = str(text or "").strip()
        if not source_text:
            return []

        chunks = make_chunks(
            doc_id=doc_id,
            text=source_text,
            chunk_id_prefix="c",
            doc_title=title,
        )

        normalized: List[Dict[str, Any]] = []
        for idx, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                continue
            chunk_text = str(chunk.get("text") or "").strip()
            if not chunk_text:
                continue
            meta = chunk.get("meta") if isinstance(chunk.get("meta"), dict) else {}
            sent_spans_raw = meta.get("sent_spans") if isinstance(meta.get("sent_spans"), list) else []
            sent_spans: List[Dict[str, Any]] = []
            for sidx, span in enumerate(sent_spans_raw):
                if isinstance(span, dict):
                    fixed = dict(span)
                    fixed.setdefault("idx", sidx)
                    sent_spans.append(fixed)
            meta["sent_spans"] = sent_spans
            meta.setdefault("title", title)
            meta.setdefault("doc_title", title)

            normalized.append(
                {
                    "chunk_id": str(chunk.get("chunk_id") or f"c{idx:04d}_{doc_id}"),
                    "doc_id": str(chunk.get("doc_id") or doc_id),
                    "text": chunk_text,
                    "meta": meta,
                }
            )
        return normalized

class FixedWindowChunker(Chunker):
    """
    Splits text into fixed-size windows with overlap, ignoring sentence boundaries.
    """
    def __init__(self, tokenizer_name: str = "gpt2", chunk_size: int = 256, overlap: int = 32):
        self.chunk_size = chunk_size
        self.overlap = overlap
        if AutoTokenizer:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
            except:
                # Fallback to simple whitespace splitting if tokenizer fails or not found
                self.tokenizer = None
        else:
            self.tokenizer = None

    def chunk(self, doc_id: str, text: str, doc_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        title = str(doc_meta.get("title") or doc_id)
        
        if self.tokenizer:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
        else:
            # Fallback: simple whitespace tokenization
            tokens = text.split()
            
        if not tokens:
            return []
            
        chunks = []
        stride = self.chunk_size - self.overlap
        
        # If text is shorter than chunk size, just one chunk
        if len(tokens) <= self.chunk_size:
            chunk_text = self._decode(tokens)
            chunks.append(self._make_chunk(doc_id, 0, chunk_text, title))
        else:
            for i in range(0, len(tokens), stride):
                window = tokens[i : i + self.chunk_size]
                chunk_text = self._decode(window)
                chunks.append(self._make_chunk(doc_id, i, chunk_text, title))
                
                if i + self.chunk_size >= len(tokens):
                    break
                    
        return chunks

    def _decode(self, tokens: Union[List[int], List[str]]) -> str:
        if self.tokenizer:
            return self.tokenizer.decode(tokens, skip_special_tokens=True)
        else:
            return " ".join(tokens)

    def _make_chunk(self, doc_id: str, idx: int, text: str, title: str) -> Dict[str, Any]:
        return {
            "chunk_id": f"c{idx:04d}_{doc_id}",
            "doc_id": doc_id,
            "text": text,
            "meta": {
                "title": title,
                "doc_title": title,
                # Fixed chunking destroys sentence boundaries, so we don't provide reliable sent_spans.
                # However, to avoid breaking downstream code that expects 'sent_spans',
                # we can provide a dummy span or try to reconstruct.
                # For strict ablation, we should probably OMIT them or provide the whole text as one span.
                "sent_spans": [{"idx": 0, "text": text}] 
            }
        }
