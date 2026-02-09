# RelRAG 实验结果汇总报告

## 1. 概述
本报告汇总了 `result/` 目录下各数据集的最佳实验结果。通过遍历所有实验版本（`experiment_01` 至 `experiment_19` 及各数据集特定目录），选出了在关键指标上表现最优的配置。

## 2. 最佳结果汇总表

| 数据集 (Dataset) | 最佳实验/版本 (Best Version) | 模型 (Model) | 检索方式 (Retriever) | 核心指标 (Key Metrics) | 备注 (Notes) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **NarrativeQA** | `experiment_19` | DeepSeek-Chat (OpenAI) | Dense | **BLEU-4: 0.6692**<br>Rouge-L: 0.5251 | 经过 19 个版本的迭代，该版本表现最优。vLLM (Qwen3-30b) 在 Rouge-L 上略优 (0.548) 但 BLEU-4 稍低。 |
| **HotpotQA** | `hotpot_dev500_vllm` | Qwen3-30b (vLLM) | Hybrid | **Ans F1: 0.2140**<br>Ans EM: 0.1560 | 官方指标优于 OpenAI (F1 ~0.1976)。<br>**瓶颈**: SP EM 为 0，证据召回率极低 (Gold SP coverage ~2%)。 |
| **MuSiQue** | `musique_experiment_1` | DeepSeek-Chat (OpenAI) | Dense/Hybrid | **F1: 0.498**<br>EM: 0.498 | 显著优于 vLLM (F1 ~0.38)。OpenAI 在不同检索方式下表现一致，可能是模型本身知识覆盖较强或评估集特性导致。 |
| **MIRAGE** | N/A | - | - | - | 暂无实验结果 (`result/mirage` 仅包含缓存)。 |

## 3. 详细分析

### 3.1 NarrativeQA
- **迭代路径**: 从 `experiment_01` 到 `experiment_19`，性能稳步提升。
- **最佳配置**: `experiment_19` 中 `openai + dense` 组合达到最高 BLEU-4 (0.6692)。
- **模型对比**: DeepSeek-Chat 在 BLEU 指标上优于 Qwen3-30b，但 Qwen 在 Rouge-L 上有竞争力。

### 3.2 HotpotQA
- **数据来源**: 基于 `hotpot_dev500_*` 系列实验（500样本，Dev split）。
- **指标差异**:
  - **官方指标 (F1/EM)**: vLLM + Hybrid 最佳 (F1 0.2140)。
  - **生成指标 (BLEU)**: OpenAI + Dense 在 BLEU-4 上较高 (0.5465 vs vLLM 0.5378)，但官方问答准确率不如 vLLM。
- **主要问题**: 所有实验的 Supporting Facts EM (SP EM) 均为 0，Joint EM 为 0。说明检索模块未能有效召回完整的证据链，严重限制了最终问答性能。

### 3.3 MuSiQue
- **现状**: `musique_experiment_1` 显示 OpenAI 模型在 F1/EM 上达到 ~0.5，远高于 vLLM 的 ~0.38。
- **疑点**: OpenAI 在 BM25、Dense、Hybrid 下指标完全一致，需进一步核查是否为模型内部知识直接作答（幻觉或过拟合）而非依赖检索。

## 4. 结论与建议
1. **RelRAG 策略**: 在 HotpotQA 上，Hybrid 检索（Dense + BM25）带来了微弱的提升（F1 0.214 vs Dense 0.207），但在 SP 召回极低的情况下，改进检索召回率是当务之急。
2. **模型选择**:
   - 生成长文本（NarrativeQA）：推荐 OpenAI (DeepSeek)。
   - 精确问答（HotpotQA）：推荐 vLLM (Qwen) 配合 Hybrid 检索。
   - 多跳推理（MuSiQue）：OpenAI 表现更佳，但需确认检索有效性。
