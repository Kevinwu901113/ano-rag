## Embedding-only baselines：GPU→CPU 自动回退验证命令

说明：
- 这些命令不依赖 LM Studio / vLLM（仅做 embedding + 检索/评测）。
- 为避免网络下载，请把 `--embed-model` 指向本机已存在的 embedding 模型路径（示例：`/home/wjk/models/qwen3-emb`）。
- `--embed-device auto`（默认）会优先用 GPU（若可用），遇到 OOM / CUDA 错误等会自动打印 `fallback to cpu because <reason>` 并回退 CPU（只重试一次）。

### 1) 已验证：无 GPU（强制 CPU）跑通 build_index + run_retrieval + metrics（MIRAGE naive）

构建索引（doc embedding）：
```bash
CUDA_VISIBLE_DEVICES="" \
python scripts/mirage/build_naive_index.py \
  --doc-pool data/mirage_sample/doc_pool.json \
  --out-dir result/verify_mirage_naive_cpu \
  --embed-model /home/wjk/models/qwen3-emb \
  --embed-device auto \
  --embed-batch-size 2 \
  --embed-max-length 256 \
  --embed-normalize
```

评测检索 recall（query embedding + FAISS search）：
```bash
CUDA_VISIBLE_DEVICES="" \
python scripts/evaluate_mirage_retrieval.py naive \
  --dataset data/mirage_sample/dataset.json \
  --index-dir result/verify_mirage_naive_cpu \
  --ks 1,3,5 \
  --limit 10
```

### 2) 已验证：无 GPU（强制 CPU）跑通 HotpotQA retrieval-only（不调用 LLM）

```bash
CUDA_VISIBLE_DEVICES="" \
python scripts/hotpotqa/baselines/run_vanilla_rag.py \
  --dataset data/hotpotqa/dataset_distractor_200.json \
  --retrieval-only \
  --limit 2 \
  --num-workers 1 \
  --work-dir result/verify_hotpot_vanilla_cpu \
  --embed-model /home/wjk/models/qwen3-emb \
  --embed-device auto \
  --embed-batch-size 2 \
  --embed-max-length 256 \
  --embed-normalize
```

### 3) 示例：GPU 显存不足触发 OOM → 自动回退 CPU

在“有 GPU”的机器上运行（不要设置 `CUDA_VISIBLE_DEVICES=""`），把 `--embed-batch-size` 调大到能触发 OOM（不同显存大小需要不同数值；示例仅供参考）：

构建索引阶段触发回退（doc embedding）：
```bash
python scripts/mirage/build_naive_index.py \
  --doc-pool data/mirage_sample/doc_pool.json \
  --out-dir result/verify_mirage_naive_fallback \
  --embed-model /home/wjk/models/qwen3-emb \
  --embed-device auto \
  --embed-batch-size 256 \
  --embed-max-length 256 \
  --embed-normalize
```

检索评测阶段触发回退（query embedding）：
```bash
python scripts/evaluate_mirage_retrieval.py naive \
  --dataset data/mirage_sample/dataset.json \
  --index-dir result/verify_mirage_naive_fallback \
  --embed-device auto \
  --embed-batch-size 256 \
  --limit 10
```

### 4) 示例：其它 embedding-only 索引构建命令（FiD）

```bash
CUDA_VISIBLE_DEVICES="" \
python scripts/mirage/build_fid_index.py \
  --doc-pool data/mirage_sample/doc_pool.json \
  --out-dir result/verify_mirage_fid_cpu \
  --embed-model /home/wjk/models/qwen3-emb \
  --embed-device auto \
  --embed-batch-size 2 \
  --embed-max-length 256 \
  --embed-normalize
```

### 5) 示例：HotpotQA 其它 retrieval-only baselines

```bash
CUDA_VISIBLE_DEVICES="" \
python scripts/hotpotqa/baselines/run_selfrag.py \
  --dataset data/hotpotqa/dataset_distractor_200.json \
  --retrieval-only \
  --limit 2 \
  --num-workers 1 \
  --work-dir result/verify_hotpot_selfrag_cpu \
  --embed-model /home/wjk/models/qwen3-emb \
  --embed-device auto \
  --embed-batch-size 2 \
  --embed-max-length 256 \
  --embed-normalize
```

```bash
CUDA_VISIBLE_DEVICES="" \
python scripts/hotpotqa/baselines/run_raptor.py \
  --dataset data/hotpotqa/dataset_distractor_200.json \
  --retrieval-only \
  --limit 2 \
  --num-workers 1 \
  --work-dir result/verify_hotpot_raptor_cpu \
  --embed-model /home/wjk/models/qwen3-emb \
  --embed-device auto \
  --embed-batch-size 2 \
  --embed-max-length 256 \
  --embed-normalize
```

```bash
CUDA_VISIBLE_DEVICES="" \
python scripts/hotpotqa/baselines/run_relrag.py \
  --dataset data/hotpotqa/dataset_distractor_200.json \
  --retrieval-only \
  --limit 2 \
  --num-workers 1 \
  --work-dir result/verify_hotpot_relrag_cpu \
  --embed-model /home/wjk/models/qwen3-emb \
  --embed-device auto \
  --embed-batch-size 2 \
  --embed-max-length 256 \
  --embed-normalize
```

### 6) 查看产物 meta.json（确认 used device / fallback reason）

MIRAGE 索引目录：
```bash
cat result/verify_mirage_naive_cpu/meta.json
```

HotpotQA work_dir：
```bash
cat result/verify_hotpot_vanilla_cpu/meta.json
```

`meta.json` 关键字段（至少）：
- `embed_model`
- `embed_device_used`
- `fallback_reason`（无回退则为 `null` / `""`）
- `normalize`
- `dim`
