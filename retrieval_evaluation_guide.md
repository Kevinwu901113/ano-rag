# 检索评测改进指南

## 问题描述
当前的检索评测结果显示 Recall@10 和 Recall@20 都卡在了 0.276，这表明检索系统没有输出完整的 top-k 候选文档列表。

## 解决方案：在 Pipeline 中添加 retrieved_doc_ids

### 1. 修改输出格式
在你的 pipeline 中，在阅读器/生成器之前，把检索得到的 top-k 段落 doc_id 列表一并写入结果项。

**期望的输出格式：**
```json
{
  "id": "8851020722386421469",
  "retrieved_doc_ids": [
    "8851020722386421469_28",  // ← 理想情况：gold 索引要能出现在这里
    "8851020722386421469_12",
    "8851020722386421469_5",
    "8851020722386421469_41",
    // ... 更多候选文档，建议输出 top-20
  ],
  "predicted_support_idxs": [41, 6],        // 这是"阅读器选的支持"，可以保留
  "predicted_answer": "..."
}
```

### 2. Doc ID 命名规范
确保 doc_id 的命名规则严格遵循：`f"{qid}_{cand_idx}"`

### 3. 修改位置建议
在以下文件中寻找合适的修改点：
- `main.py` 或 `main_musique.py` - 主要的 pipeline 入口
- `query/query_processor.py` - 查询处理器
- `retrieval/` 目录下的检索相关模块

### 4. 实现步骤
1. 在检索阶段保存完整的 top-k 候选文档 ID 列表
2. 在输出结果时，将这个列表添加到 `retrieved_doc_ids` 字段
3. 确保 doc_id 格式与 qrels 文件中的格式完全一致

### 5. 验证效果
修改后重新运行评测：
```bash
python eval_retrieval_nq.py --results result/24/musique_results.jsonl --qrels nq_dev_rag_raw/qrels_dev.tsv
```

预期改进：
- Recall@10 和 Recall@20 应该会有明显提升
- 不再"卡在5"的限制
- 能够真正反映检索系统的性能

### 6. 调试建议
如果修改后仍有问题，可以：
1. 先运行 `check_input_coverage.py` 确认数据准备阶段的覆盖率
2. 运行 `check_alignment_report.py` 分析具体的对齐问题
3. 检查 doc_id 命名是否与 qrels 完全一致