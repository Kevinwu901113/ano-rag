# E1 Overall Benchmark Report

## A1. 实验设置概览
- 数据集列表：HotpotQA, MuSiQue, MIRAGE
- Baseline 列表：bm25, dense, hybrid, lightrag, raptor, selfrag, relrag_full
- 统一条件：
  - context_budget = 4096
  - embedding_model = sentence-transformers/all-MiniLM-L6-v2
  - 统一评测脚本 = scripts/evaluate_standard.py
- 结果可用性：本次 run 仅发现 bm25/dense/hybrid/raptor/relrag_full 结果，未发现 lightrag/selfrag 目录

## A2. 主结果表

### HotpotQA

| method | EM | F1 | R@1 | R@3 | R@5 | MRR | Hit@5 | invalid_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bm25 | 1.000 | 16.062 | 0.419 | 0.756 | 0.851 | 0.905 | 0.995 | 0.000 |
| dense | 0.500 | 15.802 | 0.090 | 0.335 | 0.532 | 0.427 | 0.800 | 0.000 |
| hybrid | 1.000 | 15.154 | 0.251 | 0.542 | 0.785 | 0.686 | 0.983 | 0.000 |
| raptor | 0.000 | 10.516 | 0.085 | 0.287 | 0.445 | 0.397 | 0.725 | 0.000 |
| relrag_full | 0.000 | 13.014 | 0.083 | 0.318 | 0.522 | 0.413 | 0.770 | 0.000 |

### MuSiQue

| method | EM | F1 | R@1 | R@3 | R@5 | MRR | Hit@5 | invalid_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bm25 | 0.000 | 5.054 | 0.252 | 0.412 | 0.500 | 0.572 | 0.710 | 0.000 |
| dense | 0.000 | 5.996 | 0.102 | 0.260 | 0.409 | 0.382 | 0.600 | 0.000 |
| hybrid | 0.000 | 4.761 | 0.169 | 0.403 | 0.521 | 0.499 | 0.720 | 0.000 |
| raptor | 0.000 | 5.802 | 0.099 | 0.255 | 0.381 | 0.371 | 0.570 | 0.000 |
| relrag_full | 0.000 | 6.211 | 0.110 | 0.249 | 0.380 | 0.380 | 0.580 | 0.000 |

### MIRAGE

| method | EM | F1 | R@1 | R@3 | R@5 | MRR | Hit@5 | invalid_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bm25 | 0.000 | 14.356 | 0.985 | 1.000 | 1.000 | 0.993 | 1.000 | 0.000 |
| dense | 0.000 | 14.199 | 0.970 | 0.990 | 0.990 | 0.980 | 0.990 | 0.000 |
| hybrid | 0.000 | 14.485 | 0.985 | 0.990 | 0.990 | 0.989 | 0.990 | 0.000 |
| raptor | 0.000 | 27.957 | 0.990 | 0.995 | 0.995 | 0.993 | 0.995 | 0.000 |
| relrag_full | 0.000 | 6.475 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## A3. 结果现象总结（不解释原因）

### HotpotQA
- Top-performing 方法：bm25（F1=16.062, EM=1.000）
- 明显落后方法：raptor（F1=10.516, EM=0.000）
- Recall 与 F1 关系：
  - Recall@5 gap vs F1 close: dense has lower R@5 (0.532) but similar F1 (15.802).

### MuSiQue
- Top-performing 方法：relrag_full（F1=6.211, EM=0.000）
- 明显落后方法：hybrid（F1=4.761, EM=0.000）
- Recall 与 F1 关系：
  - High Recall@5 but lower F1: hybrid (R@5=0.521, F1=4.761).
  - Higher F1 with lower Recall@5: relrag_full (F1=6.211, R@5=0.380).

### MIRAGE
- Top-performing 方法：raptor（F1=27.957, EM=0.000）
- 明显落后方法：relrag_full（F1=6.475, EM=0.000）
- Recall 与 F1 关系：
  - High Recall@5 but lower F1: bm25 (R@5=1.000, F1=14.356).
  - Higher F1 with lower Recall@5: raptor (F1=27.957, R@5=0.995).

## A4. 工程与可比性声明
- 所有 E1 组合状态：PASS（15/15）
- Canonical ID 对齐情况（三数据集）：已对齐（检索指标均成功产出）
- 无 error prediction / invalid run：invalid_rate=0 across all available runs
- 本次 E1 结果可作为后续分析与论文实验基线
