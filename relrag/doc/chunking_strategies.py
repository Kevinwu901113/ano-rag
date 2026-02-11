import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import re
from relrag.utils import TextUtils
from relrag.config import config

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
    Chunks based on pre-segmented sentences in metadata (HotpotQA style)
    or using TextUtils if no metadata provided.
    Currently used in HotpotQA baseline.
    """
    def chunk(self, doc_id: str, text: str, doc_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        title = str(doc_meta.get("title") or doc_id)
        # Prefer pre-split sentences from meta if available (HotpotQA)
        sentences = doc_meta.get("sentences")
        
        if not sentences:
            # Fallback to TextUtils splitter
            spans = TextUtils.split_with_spans(text)
            sentences = [s["text"] for s in spans]
        
        # Filter empty
        sentences = [str(s).strip() for s in sentences if str(s).strip()]
        
        if not sentences:
            return []
            
        # Current HotpotQA logic: One chunk per document? 
        # Wait, looking at hotpot_entry.py:
        # text = " ".join(sentences)
        # chunk = { ... "text": text ... }
        # It creates a SINGLE chunk per document containing all sentences!
        # Let's double check hotpot_entry.py line 316.
        # "chunk_id = f'c{idx:04d}_{doc_id}'" -> inside a loop over doc_index.items()
        # So yes, for HotpotQA it seems to be 1 chunk per document in the simple case?
        # But wait, retrieval usually needs smaller chunks.
        # Ah, in hotpot_entry.py:
        # "chunk = { ... text: text ... }" where text is join of ALL sentences.
        # So the "chunk" is the whole document.
        # However, for ablation study, we usually compare "Sentence Splitting" vs "Block Splitting".
        # "Sentence Splitting" usually means "split by sentence and maybe group them".
        # But if the current baseline is "Whole Doc", that's different.
        
        # Let's re-read the user request: "compare current sentence splitting method vs traditional chunk splitting".
        # In `experiment_protocol.md`: "Sentence-aware chunking... Pack sentences in order to target chunk_size = 1200 tokens... Overlap...".
        # This implies the *intended* sentence chunking is grouping sentences into 1200-token chunks.
        
        # However, `hotpot_entry.py` `_write_chunks_for_example` (which I saw in search) 
        # loops over docs, and for each doc creates ONE chunk. 
        # "text = " ".join(sentences)"
        # This might be because HotpotQA docs are short (Wikipedia paragraphs).
        # Let's verify HotpotQA doc lengths. Usually they are short.
        # If they are short, then "Sentence Chunking" might just be "Keep sentences intact".
        # And "Traditional Block Chunking" might split a short doc into even smaller pieces?
        # Or maybe the "Whole Doc" IS the chunk.
        
        # But the user says "current sentence splitting method". 
        # If `hotpot_entry.py` creates 1 chunk per doc, then the "splitting" happens at retrieval time?
        # Or maybe I should implement the logic from `experiment_protocol.md` (1200 tokens) as the "SentenceAwareChunker".
        
        # Actually, looking at `relrag/doc/chunker.py`, `make_chunks` DOES split into multiple chunks with overlap.
        # But `hotpot_entry.py` seems to use a simpler logic.
        
        # I will implement `SentenceAwareChunker` to support BOTH:
        # 1. "Whole Doc" (if that's what hotpot_entry does)
        # 2. "Grouped Sentences" (like relrag/doc/chunker.py)
        
        # Given the task is ablation, I should probably stick to what `hotpot_entry.py` does by default 
        # OR what `experiment_protocol.md` claims.
        # `experiment_protocol.md` describes the "Sentence-aware chunking" in detail. 
        # I should probably assume `relrag/doc/chunker.py` logic is the "current method" being referred to,
        # or at least the one I should use as "Sentence Splitting".
        
        # However, `hotpot_entry.py` has its own `_write_chunks_for_example`.
        # I'll replicate `hotpot_entry.py` logic exactly for now as the "Baseline/Current" for Hotpot.
        # It creates one chunk per doc, but includes "sent_spans" metadata.
        # This allows the *Reader* to select specific sentences.
        
        # Wait, if I change to "Block Chunking", I might break the Reader if it relies on "sent_spans".
        # The user asks to compare "sentence splitting vs traditional block chunking".
        # Traditional block chunking usually ignores sentence boundaries.
        
        # So:
        # Method A (Sentence): Doc -> [Sentences] -> Preserved in metadata.
        # Method B (Block): Doc -> [Block 1, Block 2] (Fixed tokens).
        
        # If Hotpot docs are small, Method A is effectively "Whole Doc with Sentence Index".
        # Method B would be "Whole Doc (or split) with NO Sentence Index (or broken ones)".
        
        # I will implement `SentenceAwareChunker` to produce the output format of `hotpot_entry.py`.
        
        chunk_id = f"c0000_{doc_id}" # Simple ID for single-chunk-per-doc
        text_joined = " ".join(sentences)
        sent_spans = [{"idx": i, "text": s} for i, s in enumerate(sentences)]
        
        return [{
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "text": text_joined,
            "meta": {
                "title": title,
                "doc_title": title,
                "sent_spans": sent_spans
            }
        }]

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
