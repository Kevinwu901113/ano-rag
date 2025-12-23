# LLM/Embedding 统一与最终验收修复审计报告

## 1. 残留 BF16 引用清单 (BF16 Model ID: Qwen/Qwen3-30B-A3B)

仅发现一处残留，其他关键脚本均已正确指向 GPTQ-Int4 版本。

- **`utils/llm_client.py`**:
  - line 17: `HF_MODEL_ID = "Qwen/Qwen3-30B-A3B"`
  - **建议修复**: 修改为 `"Qwen/Qwen3-30B-A3B-GPTQ-Int4"`。

**已确认正确的文件 (无需修改)**:
- `scripts/llm/start_vllm_qwen3_30b_a3b.sh`: `MODEL_ID="Qwen/Qwen3-30B-A3B-GPTQ-Int4"`
- `scripts/mirage/build_notes.sh`: `VLLM_MODEL="Qwen/Qwen3-30B-A3B-GPTQ-Int4"`
- `scripts/hotpotqa/baselines/run_graphrag_with_vllm.sh`: `VLLM_MODEL="Qwen/Qwen3-30B-A3B-GPTQ-Int4"`
- `scripts/musique/run_musique.sh`: `VLLM_MODEL="Qwen/Qwen3-30B-A3B-GPTQ-Int4"`
- `scripts/modal_vllm_launcher.py`: `model_name = "Qwen/Qwen3-30B-A3B-GPTQ-Int4"`

## 2. Extract Profile 覆盖情况

已检查以下提取/构图/索引相关组件，确认 Profile 使用情况：

| 组件文件 | 类/方法 | 当前 Profile | 状态 |
| :--- | :--- | :--- | :--- |
| `generator/note_generator.py` | `NoteGenerator.__init__` | `llm_profile="extract"` | ✅ 合规 |
| `structrag/structurizer.py` | `Structurizer._build_graph` | `llm_profile="extract"` | ✅ 合规 |
| `generator/extractor.py` | `EvidenceExtractor.__init__` | `llm_profile="extract"` | ✅ 合规 |
| `retriever/rerank.py` | `LLMReranker.__init__` | `llm_profile="extract"` | ✅ 合规 |
| `generator/extractor.py` | `EvidenceExtractor.judge_and_compress` | `llm_profile="extract"` | ✅ 合规 |

**注意**: 其他生成/回答组件 (如 `generator/answerer.py`, `baselines/naive_rag/runner.py`) 使用 `generate` profile，符合预期。

## 3. Embedding 默认值风险清单

以下文件包含 `Qwen/Qwen3-Embedding-8B` 硬编码默认值，需统一替换：

1.  `baselines/common/model_clients.py`: `base = model or "Qwen/Qwen3-Embedding-8B"`
2.  `scripts/hotpotqa/evaluate_embedonly_recall.py`: `return "Qwen/Qwen3-Embedding-8B"`
3.  `config/config_loader.py`: `"model": "Qwen/Qwen3-Embedding-8B"` (config default)
4.  `indexer/embedding_index.py`: `base = self.embed_cfg.get("model", "Qwen/Qwen3-Embedding-8B")`
5.  `baselines/vanilla_rag/retriever.py`: `base = embed_cfg.get("model", "Qwen/Qwen3-Embedding-8B")`
6.  `baselines/vanilla_rag/index.py`: `model = self._resolve_model_name()` (inherits default logic or hardcoded)
7.  `baselines/fid_rag/indexer.py`: (Inherits similar logic)
8.  `baselines/naive_rag/index.py`: (Inherits similar logic)

## 4. 本地 Embedding 模型发现与选择

**扫描路径**: `/home/wjk/models/`

**发现模型**:
1.  **`qwen3-emb`**: 位于 `/home/wjk/models/qwen3-emb`。
    - 推测为 `Qwen/Qwen3-Embedding-8B`。
    - **缺点**: 显存占用大 (8B)，CPU 推理慢，不适合作为默认 Baseline 配置。
2.  **`all-MiniLM-L6-v2`**: 位于 `/home/wjk/models/hf-cache/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf`。
    - **优点**: 轻量级 (MiniLM)，CPU 极其友好，适合作为 Baseline 默认值。

**最终选择**:
- **DEFAULT_EMBED_MODEL**: `/home/wjk/models/hf-cache/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf`
- **理由**: 满足“量化/轻量、CPU 友好、已在本地”的所有要求。

## 5. 其他检查
- **LMStudio**: 全仓未发现 `lmstudio`, `:1234`, `1234/v1` 相关残留。
- **vLLM 启动模板**: `scripts/llm/start_vllm_qwen3_30b_a3b.sh` 配置正确 (Model, TP, GPTQ, Download Dir)。
