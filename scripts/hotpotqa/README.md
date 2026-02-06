HotpotQA RAPTOR/GraphRAG/LightRAG 对比跑法

**前置条件**
- 本地 vLLM LLM 服务已启动（OpenAI-compatible），例如 `http://127.0.0.1:8000/v1`
- 本地 vLLM Embedding 服务已启动（建议单独模型），例如 `http://127.0.0.1:8001/v1`
- 安装依赖：
  - `pip install graphrag lightrag-hku transformers openai pyyaml`

**数据集**
- distractor（closed-context）：`data/hotpot_dev_distractor_500_jsonl.jsonl`
- fullwiki（**仅使用样本自带 context，不等价 open-domain**）：`data/hotpot_dev_fullwiki_500_jsonl.jsonl`

**运行（distractor / strict）**
```bash
python scripts/hotpotqa/run_hotpot_compare.py \
  --data data/hotpot_dev_distractor_500_jsonl.jsonl \
  --out_dir result/hotpot_compare_distractor \
  --mode strict \
  --limit 50 \
  --llm_base_url http://127.0.0.1:8000/v1 \
  --llm_model qwen3-30b-a3b \
  --embed_base_url http://127.0.0.1:8001/v1 \
  --embed_model qwen3-embedding \
  --tokenizer_model cyankiwi/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit
```

**使用配置文件（推荐）**
```bash
python scripts/hotpotqa/run_hotpot_compare.py \
  --config scripts/hotpotqa/hotpot_compare_config.yaml
```

配置文件字段见：`scripts/hotpotqa/hotpot_compare_config.yaml`

**并行运行三种方法（注意会占满 GPU）**
```bash
python scripts/hotpotqa/run_hotpot_compare.py \
  --config scripts/hotpotqa/hotpot_compare_config.yaml \
  --parallel
```

如 RAPTOR 仍需 Python 3.9，可在 3.10/3.11 环境中运行对比脚本，并通过 `--raptor_python` 指定 3.9 解释器：
```bash
python scripts/hotpotqa/run_hotpot_compare.py \
  --data data/hotpot_dev_distractor_500_jsonl.jsonl \
  --out_dir result/hotpot_compare_distractor \
  --mode strict \
  --limit 50 \
  --raptor_python /path/to/python3.9
```

**运行（distractor / merged，速度更快但非严格闭集）**
```bash
python scripts/hotpotqa/run_hotpot_compare.py \
  --data data/hotpot_dev_distractor_500_jsonl.jsonl \
  --out_dir result/hotpot_compare_distractor_merged \
  --mode merged \
  --limit 500
```

**运行（fullwiki / context-only）**
```bash
python scripts/hotpotqa/run_hotpot_compare.py \
  --data data/hotpot_dev_fullwiki_500_jsonl.jsonl \
  --out_dir result/hotpot_compare_fullwiki_context \
  --mode strict \
  --limit 500
```

**输出**
- `gold_official.json`：官方评测 gold
- `{method}_{mode}/raptor_pred.json` / `graphrag_pred.json` / `lightrag_pred.json`
- `{method}_{mode}/metrics.json`：`eval/hotpot_evaluate_v1.py` 口径的 EM/F1/Sp/Joint

**注意**
- GraphRAG/LightRAG 默认不输出 sentence-level supporting facts，本脚本预测 `sp=[]`，因此 sp/joint 指标会非常低（通常为 0）。如需 sp/joint，请实现证据句对齐与抽取。
- `--mode merged` 不是严格闭集设定，仅用于快速 sanity check。
