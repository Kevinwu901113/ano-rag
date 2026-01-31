# HotpotQA dev-distractor-500 实验结果汇总（OpenAI vs vLLM）

## 目录
- 1. 背景与运行命令
- 2. 结果文件位置（如何定位）
- 3. 指标口径
- 4. 总表（6 个 run，重新评估结果）
- 5. 关键结论（中文解读）
- 6. 复现“官方指标”计算的方法
- 7. FAQ

## 1. 背景与运行命令

本报告对应以下两次运行（同一数据集、不同 reader）：

- OpenAI：
  - `python hotpot_entry.py --data data/hotpot_dev_distractor_500_jsonl.jsonl --reader openai --output_dir result/hotpot_dev500_openai --cache_dir result/cache_hotpot_dev500_openai --debug_dir result/debug_hotpot_dev500_openai`
- vLLM：
  - `python hotpot_entry.py --data data/hotpot_dev_distractor_500_jsonl.jsonl --reader vllm --output_dir result/hotpot_dev500_vllm --cache_dir result/cache_hotpot_dev500_vllm --debug_dir result/debug_hotpot_dev500_vllm`

说明：
- 数据 split：`dev`
- 样本数：`500`
- retriever 模式：`bm25 / dense / hybrid`
- top_k：`10`（脚本内部对检索做 overfetch + 去重，最终有效 top_k_final 见下表）

## 2. 结果文件位置（如何定位）

每个 run 的目录结构（以 `vllm_hybrid` 为例）：

- 运行目录：`result/hotpot_dev500_vllm/`
  - 生成输出（逐样本）：`pred_dev_hybrid.jsonl`
  - 对齐与官方评估输入：`align_pred_dev_hybrid/official_pred.json`、`align_pred_dev_hybrid/official_gold.json`
  - 对齐审计：`align_pred_dev_hybrid/alignment_audit_report.md`

两个 output_root 的汇总文件：
- OpenAI：`result/hotpot_dev500_openai/summary_dev.json`
- vLLM：`result/hotpot_dev500_vllm/summary_dev.json`

注意：
- `alignment_audit_report.md` 中包含 `llm_final` 与 `gold_sp∈topk` 的统计口径。

## 3. 指标口径

本实验同时包含两类指标：

1) **官方 HotpotQA 指标**（来自 `align_pred_dev_*/official_pred.json` vs `official_gold.json`，按 `eval/hotpot_evaluate_v1.py` 的口径计算）
- Answer EM / Answer F1：答案匹配
- SP EM / SP F1：Supporting Facts（证据句）匹配
- Joint EM / Joint F1：Answer 与 SP 的联合分数

2) **生成文本指标**（来自 `summary_dev.json`）
- BLEU1、BLEU4、ROUGE-L、METEOR

补充诊断指标（来自 `alignment_audit_report.md`）：
- `llm_final` 占比：回答来自正常解析（非 fallback）的比例
- `gold_sp∈topk` 占比：`gold supporting_facts` 是否被包含在检索 topk 证据集合中的比例（subset 口径）
- `top_k_final` 均值：检索去重后实际保留的证据条数均值

## 4. 总表（6 个 run，重新评估结果）

| run | reader | retriever | model | Answer EM | Answer F1 | SP EM | SP F1 | Joint EM | Joint F1 | BLEU1 | BLEU4 | ROUGE-L | METEOR | top_k_final均值 | llm_final占比 | gold_sp∈topk占比 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| openai_bm25 | openai | bm25 | deepseek-chat | 0.1420 | 0.1872 | 0.0000 | 0.1607 | 0.0000 | 0.0817 | 0.3684 | 0.5465 | 0.1865 | 0.1350 | 1.76 | 0.724 | 0.012 |
| openai_dense | openai | dense | deepseek-chat | 0.1440 | 0.1885 | 0.0000 | 0.1620 | 0.0000 | 0.0818 | 0.3674 | 0.5445 | 0.1873 | 0.1344 | 1.76 | 0.740 | 0.014 |
| openai_hybrid | openai | hybrid | deepseek-chat | 0.1460 | 0.1976 | 0.0000 | 0.2034 | 0.0000 | 0.0855 | 0.3385 | 0.4775 | 0.1994 | 0.1552 | 2.38 | 0.634 | 0.020 |
| vllm_bm25 | vllm | bm25 | qwen3-30b-a3b | 0.1580 | 0.2125 | 0.0000 | 0.1607 | 0.0000 | 0.0815 | 0.3728 | 0.5347 | 0.2131 | 0.1602 | 1.76 | 0.974 | 0.012 |
| vllm_dense | vllm | dense | qwen3-30b-a3b | 0.1560 | 0.2076 | 0.0000 | 0.1620 | 0.0000 | 0.0826 | 0.3736 | 0.5378 | 0.2089 | 0.1537 | 1.76 | 0.986 | 0.014 |
| vllm_hybrid | vllm | hybrid | qwen3-30b-a3b | 0.1560 | 0.2140 | 0.0000 | 0.2034 | 0.0000 | 0.0860 | 0.3378 | 0.4639 | 0.2161 | 0.1685 | 2.38 | 0.826 | 0.020 |

## 5. 关键结论（中文解读）

1) **vLLM 在官方 Answer 指标上整体高于 OpenAI**
- Answer F1：vLLM ~0.2076–0.2140；OpenAI ~0.1872–0.1976
- Answer EM：vLLM ~0.1560–0.1580；OpenAI ~0.1420–0.1460

2) **证据（Supporting Facts）是当前最主要瓶颈**
- 所有 run 的 SP EM 都是 0，导致 Joint EM 也为 0。
- `gold_sp∈topk` 为 ~1.2%–2.0%（500 条里 6–10 条样本，金标 supporting_facts 完整落在检索 topk 中）。
- 这意味着：即使 reader 输出再好，缺少/不匹配证据会严重限制联合指标。

3) **retriever 之间差异较小，hybrid 在 Joint F1 上略占优**
- vLLM：hybrid 的 Joint F1 略高（0.0860），dense/bm25 接近（0.0826/0.0815）。
- OpenAI：hybrid 的 Joint F1 略高（0.0855），bm25/dense 接近（0.0817/0.0818）。

4) **fallback 行为差异明显**
- vLLM：`llm_final` ~0.826–0.986（fallback 很少）
- OpenAI：`llm_final` ~0.634–0.740（fallback 明显更多）

## 6. 复现“官方指标”计算的方法

以 `vllm_hybrid` 为例：

```bash
python eval/hotpot_evaluate_v1.py \
  result/hotpot_dev500_vllm/align_pred_dev_hybrid/official_pred.json \
  result/hotpot_dev500_vllm/align_pred_dev_hybrid/official_gold.json
```

同理可替换为其他 run 的 `align_pred_dev_*/official_pred.json` 与 `official_gold.json`。

## 7. FAQ

**Q1: 为什么表中没有 metrics.json 的来源？**  
A1: 本次目录结构中未包含独立的 `metrics.json`，生成文本指标来自 `summary_dev.json` 的聚合结果。

**Q2: 为什么 hybrid 的 top_k_final 均值更高？**  
A2: `alignment_audit_report.md` 显示 hybrid 的去重后有效证据条数更高（2.38），与检索融合产生的候选覆盖更广有关。
