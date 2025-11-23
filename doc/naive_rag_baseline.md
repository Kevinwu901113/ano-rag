# Naive RAG 基线：构建索引 & QA 运行命令

> 仅适用于 MIRAGE 示例数据；使用 `baselines/naive_rag` 下的脚本。

## 构建向量索引（FAISS + chunks.jsonl）

```bash
python scripts/mirage/build_naive_index.py \
  --doc-pool data/mirage_sample/doc_pool.json \
  --out-dir result/mirage_naive
```

- 可选参数：`--target-tokens/--max-tokens/--overlap-tokens` 控制分块长度；`--no-title` 不在块前加标题。

## 跑 QA（无结构检索，只用朴素向量检索）

```bash
python scripts/mirage/run_naive_rag.py \
  --dataset-path data/mirage_sample/dataset.json \
  --index-dir result/mirage_naive \
  --topk 5 \
  --result-root result \
  --new \
  --lmstudio-endpoint http://127.0.0.1:1234/v1 \
  --lmstudio-model openai/gpt-oss-20b
```

- `--index-path/--chunks-path` 可显式指定索引文件；`--limit` 限制样本数；`--no-debug` 关闭调试输出；`--new` 强制创建新的结果目录（如 `result/mirage_naive_000`）。***
