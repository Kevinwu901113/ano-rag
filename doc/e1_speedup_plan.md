## Step A: Static location (embedding/index build paths)

### Dense/Hybrid entry points used by E1
- `experiments/manifests/methods.yaml`: `dense_rag` / `hybrid_rag` map to:
  - HotpotQA: `scripts/hotpotqa/baselines/run_vanilla_rag.py`
  - MuSiQue: `scripts/musique/baselines/run_vanilla_rag.py`
  - MIRAGE: `scripts/mirage/run_vanilla_rag.py`
- `experiments/run.py`: no `build_index` for these methods; index build happens inside entry scripts.

### HotpotQA (dense/hybrid) – per-question in-memory index
- `scripts/hotpotqa/baselines/run_vanilla_rag.py`
  - `InMemoryVanillaRetriever.build_index_for_question(...)` builds per-question vectors by calling
    `TransformerEmbedder.encode(...)` (from `scripts/hotpotqa/baselines/baseline_utils.py`).
  - This is executed for every question and repeated across jobs (budgets/methods).

### MuSiQue (dense/hybrid) – per-question in-memory index
- `scripts/musique/baselines/run_vanilla_rag.py`
  - `InMemoryVanillaRetriever.build_index_for_question(...)` calls the encoder to embed passages.
  - Like HotpotQA, this builds a fresh in-memory index per question/job.

### MIRAGE (dense/hybrid) – doc-pool FAISS index build
- `scripts/mirage/run_vanilla_rag.py`
  - If `artifacts/vanilla_rag_index.faiss` / `artifacts/vanilla_rag_chunk_store.pkl` are missing,
    it loads `doc_pool.json` and calls `VanillaRAGIndexer.build(...)`.
- `baselines/vanilla_rag/index.py`
  - `VanillaRAGIndexer.build(...)`:
    - chunks docs with `VanillaChunker(target_tokens=512, max_tokens=600, overlap_tokens=50)`
    - embeds chunks via `utils.embedding_utils.EmbeddingEncoder.encode(...)`
    - builds `faiss.IndexFlatIP` and writes:
      - index: `vanilla_rag_index.faiss`
      - chunk store: `vanilla_rag_chunk_store.pkl`
      - meta: `vanilla_rag_index.faiss.meta.pkl`

## Step B: Cache design (cross-job reuse)

### Cache location (shared across jobs)
- `result_relrag/cache/<cache_key>/`

### Cache key (must include all embedding/index factors)
Example schema:
`<dataset>__<embed_model>__<device>__len<max_len>__norm<0/1>__dtype<dtype>__chunk<...>__faiss<kind>__data<hash>`

Concrete example (HotpotQA):
`hotpotqa_distractor_200__home_wjk_models_qwen3-emb__cpu__len512__norm1__dtypeauto__batch4__maxctx10__indexin_memory_cosine__data1a2b3c4d5e6f`

Concrete example (MIRAGE):
`mirage_sample_200__home_wjk_models_qwen3-emb__cpu__len256__norm1__dtypeauto__provqwen3__chunk512x600x50__faissFlatIP__doc1a2b3c4d5e6f`

### Cache contents (auditable)
- `embeddings.npy` (precomputed vectors)
- `faiss.index` (real FAISS index for MIRAGE; placeholder marker for in-memory HotpotQA cache)
- `offsets.json` (HotpotQA passage offsets)
- `metadata.json` (dataset hash, params, counts)
- `build.log`

### New cache-aware build paths
- HotpotQA:
  - `scripts/hotpotqa/baselines/run_vanilla_rag.py` builds a dataset-level passage embedding cache and
    reuses it across budgets/jobs.
- MIRAGE:
  - `scripts/mirage/run_vanilla_rag.py` builds the FAISS index into the shared cache and reuses it.
