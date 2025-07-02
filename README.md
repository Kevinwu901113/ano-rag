# ano-rag

This repository contains utilities for working with retrieval augmented generation (RAG).

## Embedding Utilities

The `src/embeddings` package provides an `EmbeddingModel` class capable of loading
`bge-m3` or `nomic-embed` models from HuggingFace, a local path or an Ollama
instance. Batch embedding is supported and can optionally return `cudf` data
frames when GPU acceleration via RAPIDS is available.  Example usage:

```python
from embeddings.model import EmbeddingModel
from embeddings.vector_store import VectorStore

model = EmbeddingModel("bge-m3", provider="huggingface")
embs = model.embed_batch(["hello", "world"], batch_size=2)
store = VectorStore(dim=len(embs[0]))
store.add(embs, ["hello", "world"])
store.save()
```

The `VectorStore` class wraps a simple FAISS index which can be persisted to
local files.
