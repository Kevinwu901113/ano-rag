import pytest
from relrag.doc.chunking_strategies import SentenceAwareChunker, FixedWindowChunker

class TestSentenceAwareChunker:
    def test_chunk_with_meta_sentences(self):
        chunker = SentenceAwareChunker()
        doc_id = "doc1"
        text = "Ignored text"
        meta = {
            "title": "Title1",
            "sentences": ["Sent 1.", "Sent 2."]
        }
        
        chunks = chunker.chunk(doc_id, text, meta)
        
        assert len(chunks) == 1
        assert chunks[0]["doc_id"] == "doc1"
        assert chunks[0]["text"] == "Sent 1. Sent 2."
        assert chunks[0]["meta"]["title"] == "Title1"
        assert len(chunks[0]["meta"]["sent_spans"]) == 2
        assert chunks[0]["meta"]["sent_spans"][0]["text"] == "Sent 1."
        assert chunks[0]["meta"]["sent_spans"][1]["idx"] == 1

    def test_chunk_without_meta_sentences(self):
        chunker = SentenceAwareChunker()
        doc_id = "doc2"
        text = "Sent A. Sent B."
        meta = {"title": "Title2"}
        
        chunks = chunker.chunk(doc_id, text, meta)
        
        assert len(chunks) == 1
        assert "Sent A." in chunks[0]["text"]
        assert len(chunks[0]["meta"]["sent_spans"]) >= 2

class TestFixedWindowChunker:
    def test_chunk_simple_split(self):
        # Using whitespace fallback for test simplicity (mocking tokenizer is complex)
        chunker = FixedWindowChunker(tokenizer_name="gpt2", chunk_size=3, overlap=1)
        # Force no tokenizer for this test to use split() logic
        chunker.tokenizer = None 
        
        doc_id = "doc3"
        text = "A B C D E"
        meta = {"title": "Title3"}
        
        chunks = chunker.chunk(doc_id, text, meta)
        
        # Expected: [A B C], [C D E] (stride 3-1=2)
        # i=0: A B C
        # i=2: C D E
        # i=4: E (if logic allows, but loop breaks if i+size >= len)
        
        assert len(chunks) == 2
        assert chunks[0]["text"] == "A B C"
        assert chunks[1]["text"] == "C D E"
        assert chunks[0]["chunk_id"].startswith("c0000")
        assert chunks[1]["chunk_id"].startswith("c0002")
        
    def test_chunk_small_text(self):
        chunker = FixedWindowChunker(chunk_size=10, overlap=2)
        chunker.tokenizer = None
        
        doc_id = "doc4"
        text = "A B C"
        meta = {"title": "Title4"}
        
        chunks = chunker.chunk(doc_id, text, meta)
        
        assert len(chunks) == 1
        assert chunks[0]["text"] == "A B C"
