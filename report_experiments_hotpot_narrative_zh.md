# 检索器与模型实验配置报告

## 1. 需求完成情况

| ID | 需求 | 状态 | 备注 |
|----|------|------|------|
| 1.1 | 入口脚本支持 `--config` | **已完成** | 已在 `hotpot_entry.py` 与 `narrativeqa_entry.py` 中实现。 |
| 1.2 | 配置迁移至 `relrag/config/` | **已完成** | 已创建 `relrag/config/config.yaml`，并更新 `ConfigLoader`。 |
| 1.3 | 实验专用配置文件 | **已完成** | 已在 `relrag/config/` 中创建 4 个文件。 |
| 1.4 | 路径依赖与 base_dir 规则 | **已完成** | `ConfigLoader` 将路径按 **仓库根目录** 进行归一化。 |
| 1.5 | Markdown 报告 | **已完成** | 即本文件。 |
| 1.6 | 冒烟运行证据 | **已完成** | HotpotQA 与 NarrativeQA（vLLM）冒烟测试通过。 |

## 2. 配置系统更新

### 配置文件
已在 `relrag/config/` 中创建以下配置文件：

*   **`config.yaml`**：主配置（已移动/迁移）。
*   **`exp_hotpot_retrievers.yaml`**：HotpotQA 检索器实验（BM25 / Dense / Hybrid，vLLM reader）。
*   **`exp_narrative_retrievers.yaml`**：NarrativeQA 检索器实验（BM25 / Dense / Hybrid，vLLM reader）。
*   **`exp_hotpot_models.yaml`**：HotpotQA 模型对比（Hybrid 检索器，vLLM / OpenAI readers）。
*   **`exp_narrative_models.yaml`**：NarrativeQA 模型对比（Hybrid 检索器，vLLM / OpenAI readers）。

### 配置加载优先级
`ConfigLoader` 现在按以下优先级顺序加载配置：
1.  **CLI 参数**：`--config <path>`
2.  **环境变量**：`ANO_RAG_CONFIG`
3.  **默认（新增）**：`relrag/config/config.yaml`
4.  **旧版兜底**：`relrag/config.yaml`

### 路径归一化策略
**策略**：配置文件中的所有相对路径都以 **仓库根目录** 为基准解析。
**会被归一化的键**：
*   `notes.out_path`, `notes.indexes_dir`
*   `retriever.embedding.cache_dir`, `download_dir`, `model_path_override`, `offline_index_path`, `meta_path`
*   `retriever.bm25.store_path`
*   `hotpot_entry.data`, `cache_dir`, `output_dir`, `debug_dir`
*   `narrativeqa_entry.qaps`, `summaries`, `stories_dir`, `cache_dir`, `output_dir`

## 3. 实验命令

使用以下命令运行实验。按需调整 `--limit` 或 `--output_dir`。

### HotpotQA 检索器实验
*对比 BM25、Dense 与 Hybrid 检索器，使用 vLLM reader。*
```bash
python hotpot_entry.py --config relrag/config/exp_hotpot_retrievers.yaml
```

### NarrativeQA 检索器实验
*对比 BM25、Dense 与 Hybrid 检索器，使用 vLLM reader。*
```bash
python narrativeqa_entry.py --config relrag/config/exp_narrative_retrievers.yaml --qaps <path_to_qaps> --summaries <path_to_summaries>
```

### HotpotQA 模型对比
*在使用 Hybrid 检索器的情况下，对比 vLLM 与 OpenAI 模型。*
```bash
# openai reader 需要设置 OPENAI_API_KEY
export OPENAI_API_KEY=sk-...
python hotpot_entry.py --config relrag/config/exp_hotpot_models.yaml
```

### NarrativeQA 模型对比
*在使用 Hybrid 检索器的情况下，对比 vLLM 与 OpenAI 模型。*
```bash
# openai reader 需要设置 OPENAI_API_KEY
export OPENAI_API_KEY=sk-...
python narrativeqa_entry.py --config relrag/config/exp_narrative_models.yaml --qaps <path_to_qaps> --summaries <path_to_summaries>
```

### 结构化模式说明
*   **支持情况**：流水线已完整支持。
*   **默认状态**：在这些实验中默认关闭，以便聚焦于基线检索器对比（BM25 / Dense / Hybrid）。
*   **启用方式**：在任意配置文件的 `retrievers` 列表中加入 `"structured"`（例如 `retrievers: ["bm25", "dense", "hybrid", "structured"]`）。同时确保 `relrag/config/config.yaml` 中设置了 `retriever.structured.enabled: true`。

## 4. 冒烟测试证据

### HotpotQA 冒烟运行
**命令**：
```bash
python hotpot_entry.py --config relrag/config/exp_hotpot_retrievers.yaml --limit 2 --output_dir result_smoke_hotpot
```
**状态**：成功（vLLM reader，BM25 / Dense / Hybrid 模式）
**产物**：
*   `result_smoke_hotpot/pred_dev_bm25.jsonl`
*   `result_smoke_hotpot/pred_dev_dense.jsonl`
*   `result_smoke_hotpot/pred_dev_hybrid.jsonl`
*   `result_smoke_hotpot/summary_dev.json`

### NarrativeQA 冒烟运行
**命令**：
```bash
export OPENAI_API_KEY=sk-fake  # 使用假 key 以允许启动
python narrativeqa_entry.py --config relrag/config/exp_narrative_models.yaml --limit 2 --qaps data/narrativeqa_smoke/qaps.csv --summaries data/narrativeqa_smoke/summaries.csv --output_dir result_smoke_narrative
```
**状态**：
*   **vLLM Reader**：成功（处理 2 个样例）。
*   **OpenAI Reader**：符合预期地失败（由于假 key 导致网络/鉴权错误）。
**产物**：
*   `result_smoke_narrative/pred_valid_vllm.jsonl`（包含结果）
*   `result_smoke_narrative/pred_valid_openai.jsonl`（为空/失败）
*   `result_smoke_narrative/summary_valid.json`

## 5. 输出校验
要验证输出，请检查生成的 `.jsonl` 文件中是否存在 `metrics` 字段，其中包含 `bleu1`、`bleu4`、`rougeL` 和 `meteor`。`summary_*.json` 文件会按 reader 与模式汇总这些指标。
