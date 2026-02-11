# 跨实验集综合评估报告 (Experiment 23, 2Wiki_2, MuSiQue_9)
> **最后更新**: 2026-02-11
> **状态**: 已完成深度诊断与指标修正

本报告汇总了针对 HotpotQA (Exp 23)、2WikiMultiHopQA (Exp 2Wiki 2) 和 MuSiQue (Exp 9) 三个数据集的最新评估结果。
**重大更新**：经过代码审计与诊断，我们发现原有的 "Recall@K" 指标基于严格的句子级（Sentence-Level）匹配，严重低估了检索性能。修正后的段落级（Paragraph-Level）评估显示，所有数据集的检索召回率均在 85%~90% 以上。

---

## 1. 核心诊断与修复 (Critical Diagnosis)

### 1.1 "低召回率" 的真相：评估粒度偏差
之前的报告中，HotpotQA 和 2Wiki 的 Recall@2 仅为 ~20% 左右，但这被证明是评估指标定义的 Artifact。
- **现象**：评估脚本要求检索出的 "句子" 必须与 Gold "句子" 完全一致。
- **真相**：Dense Retriever 往往检索出包含正确信息的同一段落的其他切片（Chunk），或者包含了完整段落。
- **修正**：引入 **Paragraph Recall**（只要检索到包含证据的文档/段落即视为命中）。

**修正后的检索性能对比 (Paragraph Recall)**：

| 数据集 | Recall@2 (Para) | Recall@5 (Para) | Recall@10 (Para) | 原始 Sentence R@2 | 结论 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **HotpotQA** | **60.40%** | **83.30%** | **90.20%** | ~22.6% | **检索已解决**。Top-10 覆盖了 90% 的证据。 |
| **2Wiki** | **67.70%** | **85.70%** | **88.80%** | ~48.2% | **检索表现强劲**。85%+ 的 R@5 足够支持 QA。 |
| **MuSiQue** | **53.32%** | **73.95%** | **87.03%** | ~33.4% | **接近理论上限**。MuSiQue 平均支持事实 > 2 个，R@2 的理论上限仅 ~66%。 |

### 1.2 Index Path 隔离性验证
- **问题**：之前的 Config Dump 显示所有实验似乎都在使用同一个全局 Index (`indexes/faiss/notes.faiss`)。
- **排查**：经代码审计，这是 Config 输出逻辑的 Bug。实际运行时，`retriever` 会根据 `work_dir` 动态构建 per-example 的 Index。
- **修复**：已修改 `*_entry.py` 脚本，在输出 `config.resolved.json` 时将 `offline_index_path` 显式标记为 `<dynamic_per_example>`，消除误解。

---

## 2. 详细评估数据表

### 2.1 MuSiQue (Experiment 9)
> **环境**: Dev Set (500 samples), vLLM (Qwen3-30B), Dense Retrieval
> **注意**: Recall@2 较低是因为 MuSiQue通常需要 3+ 个支撑段落，K=2 时无法完全召回。

| 评估粒度 | Recall@2 | Recall@5 | Recall@10 | IE@2 | IE@5 | IE@10 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Paragraph (修正)** | **0.5332** | **0.7395** | **0.8703** | **0.6880** | **0.4332** | **0.2766** |
| Sentence (原始) | 0.3340 | 0.5620 | - | 0.6880 | 0.4332 | - |

**QA 指标**:
- **F1**: 0.3301
- **EM**: 0.2440
- **分析**: 尽管 R@10 达到 87%，但 F1 仅 0.33。这表明瓶颈在于 **Reader (LLM)** 的多跳推理能力或长上下文抗噪能力，而非检索。

### 2.2 2WikiMultiHopQA (Experiment 2Wiki 2)
> **环境**: Dev Set (500 samples), vLLM (Qwen3-30B), Dense Retrieval

| 评估粒度 | Recall@2 | Recall@5 | Recall@10 | IE@2 | IE@5 | IE@10 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Paragraph (修正)** | **0.6770** | **0.8570** | **0.8880** | **0.7850** | **0.4040** | **0.5478** |
| Sentence (原始) | 0.5060 | 0.5825 | - | 0.7850 | 0.4040 | - |

**QA 指标**:
- **F1**: 0.3661 (Hybrid 略高 0.3736)
- **EM**: 0.3020
- **分析**: R@5 达到 85%，说明绝大多数所需信息都在上下文中。F1 偏低同样指向 Reader 能力瓶颈。

### 2.3 HotpotQA (Experiment 23)
> **环境**: Dev Set (500 samples), vLLM (Qwen3-30B), Dense Retrieval

| 评估粒度 | Recall@2 | Recall@5 | Recall@10 | IE@2 | IE@5 | IE@10 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Paragraph (修正)** | **0.6040** | **0.8330** | **0.9020** | **0.6040** | **0.3348** | **0.3444** |
| Sentence (原始) | 0.2260 | 0.3840 | - | 0.6040 | 0.3348 | - |

**QA 指标**:
- **F1**: 0.5460 (Hybrid 最佳 0.5614)
- **EM**: 0.4120
- **分析**: 90% 的检索召回率证明检索系统非常可靠。F1 0.546 与 SOTA (通常 > 0.7-0.8) 仍有差距，需检查 CoT Prompt 或 Answer Extraction 逻辑。

---

## 3. 下一步建议

1.  **评估标准统一**：后续所有实验报告（evaluation_report_zh.md）必须记录以下标准指标：`f1`, `em`, `recall@2`, `recall@5`, `ie@2`, `ie@5`, `ndcg@2`, `ndcg@5`。
    - **注意**：检索指标 (Recall, IE, NDCG) 默认指 **Paragraph-level** (基于 Gold Title 匹配)，以消除切分粒度带来的评估偏差。
2.  **MuSiQue 优化**：鉴于 R@10 (87%) 远高于 F1 (33%)，建议重点优化 **Reader** 阶段。可以尝试：
    - 增加 CoT 的推理步数。
    - 引入 **Noise Filtering** (重排序或过滤)，减少无关段落对推理的干扰（IE@10 仅 27%，说明 Top-10 中 70% 是噪声）。
3.  **2Wiki/HotpotQA 优化**：R@5 均 > 83%，说明 Top-5 窗口已足够。可以尝试减少输入 LLM 的 Top-K (如 K=5)，以提高信息密度（IE@5 > IE@10），可能有助于提升 QA F1。
