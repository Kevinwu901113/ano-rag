# Baseline 实施说明与运行手册（LightRAG / GraphRAG / RAPTOR）

## 1. 我实际做了什么

本次在仓库根目录新增了独立 `baseline/` 体系，和主项目流程解耦，主要内容如下：

1. 统一数据中间层
- 新增 `baseline/tools/build_intermediate.py`
- 一次性把 3 个数据集转换为统一格式：
  - `baseline/data/<dataset>/corpus.json`
  - `baseline/data/<dataset>/corpus.txt`
  - `baseline/data/<dataset>/qa.jsonl`
  - `baseline/data/<dataset>/meta.json`

2. 三套 baseline runner
- `baseline/runners/run_lightrag_qa.py`
- `baseline/runners/run_graphrag_qa.py`
- `baseline/runners/run_raptor_qa.py`
- 统一参数风格：
  - `--dataset {hotpotqa,musique,2wiki}`
  - `--llm_backend {qwen,deepseek}`
  - `--data_root baseline/data`
  - `--output_root baseline/results`
  - `--limit`（默认 0=全量）
  - `--rebuild_index`
  - `--max_docs`（默认 0=全量；用于 smoke）

3. 评测与汇总
- `baseline/eval/score_hotpot.py`
- `baseline/eval/score_squad_style.py`
- `baseline/eval/score_official_proxy.py`
- `baseline/eval/score_all.py`
- 汇总输出：`baseline/results/metrics.csv`

4. 服务预检
- `baseline/tools/check_services.py`
- 检查：
  - `http://127.0.0.1:8000/v1`（`qwen3-30b-a3b`）
  - `http://127.0.0.1:8001/v1`（`qwen3-embedding`）
  - embedding 维度是否为 `4096`

5. 结果目录规范
- 预测：
  - `baseline/results/<method>/<dataset>/<llm_backend>/pred.jsonl`
- 指标：
  - `baseline/results/metrics.csv`
- 工作区/索引：
  - `baseline/workspaces/<method>/...`

---

## 2. 三套方法是否使用官方代码/仓库

### LightRAG
- 使用官方包：`lightrag-hku==1.4.9.11`
- 方式：在 runner 中直接调用官方 `LightRAG` API（`ainsert/aquery`）
- 结论：**是官方实现 + 项目内薄封装 runner**（未改官方包源码）

### GraphRAG
- 使用官方包：`graphrag==3.0.1`（CLI：`graphrag init/index/query`）
- 方式：runner 自动生成/修补 `settings.yaml` 并调用官方 CLI
- 结论：**是官方实现 + 项目内配置/流程封装**（未改官方包源码）

### RAPTOR
- 使用本地 `RAPTOR/raptor`（官方仓库代码路径）
- 方式：基于官方类 `BaseEmbeddingModel/BaseSummarizationModel/BaseQAModel/RetrievalAugmentation`，新增 OpenAI-compatible 适配器
- 结论：**是官方实现 + 项目内适配器封装**（未改 RAPTOR 源码文件）

---

## 3. 统一模型接口约定

1. 本地 vLLM Chat
- Base URL: `http://127.0.0.1:8000/v1`
- Model: `qwen3-30b-a3b`
- API Key: `EMPTY`

2. 本地 vLLM Embedding
- Base URL: `http://127.0.0.1:8001/v1`
- Model: `qwen3-embedding`
- 维度：`4096`
- API Key: `EMPTY`

3. DeepSeek（OpenAI 模式）
- Base URL: `https://api.deepseek.com/v1`
- Model: `deepseek-chat`
- API Key: 运行时通过环境变量注入（`OPENAI_API_KEY`）

---

## 4. 环境与依赖（当前约定）

1. Conda 环境（已复用）
- `baseline-lightrag`（Python 3.11）
- `baseline-graphrag`（Python 3.11）
- `baseline-raptor`（Python 3.9）

2. 环境导出文件
- `baseline/envs/baseline-lightrag.yml`
- `baseline/envs/baseline-graphrag.yml`
- `baseline/envs/baseline-raptor.yml`
- 对应 pip 冻结：`baseline/envs/*.requirements.txt`

---

## 5. 后续运行命令（可直接复制）

## 5.1 预检与数据准备

```bash
# 1) 服务可用性检查
python baseline/tools/check_services.py

# 2) 构建统一中间层（Hotpot/MuSiQue/2Wiki）
python baseline/tools/build_intermediate.py
```

## 5.2 可选：注入 DeepSeek Key

```bash
# 运行 deepseek backend 前执行（示例）
export OPENAI_API_KEY="你的_deepseek_key"
```

## 5.3 Hotpot 全量 6 组（正式跑数）

```bash
# LightRAG
conda run -n baseline-lightrag python baseline/runners/run_lightrag_qa.py --dataset hotpotqa --llm_backend qwen --rebuild_index
conda run -n baseline-lightrag python baseline/runners/run_lightrag_qa.py --dataset hotpotqa --llm_backend deepseek --rebuild_index

# RAPTOR
conda run -n baseline-raptor python baseline/runners/run_raptor_qa.py --dataset hotpotqa --llm_backend qwen --rebuild_index
conda run -n baseline-raptor python baseline/runners/run_raptor_qa.py --dataset hotpotqa --llm_backend deepseek --rebuild_index

# GraphRAG
conda run -n baseline-graphrag python baseline/runners/run_graphrag_qa.py --dataset hotpotqa --llm_backend qwen --rebuild_index
conda run -n baseline-graphrag python baseline/runners/run_graphrag_qa.py --dataset hotpotqa --llm_backend deepseek --rebuild_index
```

## 5.4 MuSiQue 全量 6 组

```bash
# LightRAG
conda run -n baseline-lightrag python baseline/runners/run_lightrag_qa.py --dataset musique --llm_backend qwen --rebuild_index
conda run -n baseline-lightrag python baseline/runners/run_lightrag_qa.py --dataset musique --llm_backend deepseek --rebuild_index

# RAPTOR
conda run -n baseline-raptor python baseline/runners/run_raptor_qa.py --dataset musique --llm_backend qwen --rebuild_index
conda run -n baseline-raptor python baseline/runners/run_raptor_qa.py --dataset musique --llm_backend deepseek --rebuild_index

# GraphRAG
conda run -n baseline-graphrag python baseline/runners/run_graphrag_qa.py --dataset musique --llm_backend qwen --rebuild_index
conda run -n baseline-graphrag python baseline/runners/run_graphrag_qa.py --dataset musique --llm_backend deepseek --rebuild_index
```

## 5.5 2Wiki 全量 6 组

```bash
# LightRAG
conda run -n baseline-lightrag python baseline/runners/run_lightrag_qa.py --dataset 2wiki --llm_backend qwen --rebuild_index
conda run -n baseline-lightrag python baseline/runners/run_lightrag_qa.py --dataset 2wiki --llm_backend deepseek --rebuild_index

# RAPTOR
conda run -n baseline-raptor python baseline/runners/run_raptor_qa.py --dataset 2wiki --llm_backend qwen --rebuild_index
conda run -n baseline-raptor python baseline/runners/run_raptor_qa.py --dataset 2wiki --llm_backend deepseek --rebuild_index

# GraphRAG
conda run -n baseline-graphrag python baseline/runners/run_graphrag_qa.py --dataset 2wiki --llm_backend qwen --rebuild_index
conda run -n baseline-graphrag python baseline/runners/run_graphrag_qa.py --dataset 2wiki --llm_backend deepseek --rebuild_index
```

## 5.6 评测与汇总

```bash
python baseline/eval/score_all.py
cat baseline/results/metrics.csv
```

---

## 6. Smoke（快速冒烟）命令模板

适合先验证链路（小规模），不是正式指标。

```bash
# 示例：Hotpot + LightRAG + qwen，10题、10文档
conda run -n baseline-lightrag python baseline/runners/run_lightrag_qa.py \
  --dataset hotpotqa --llm_backend qwen --limit 10 --max_docs 10 --rebuild_index
```

---

## 7. 当前已完成状态（截至本次）

1. 已完成：
- `baseline/` 全套脚本与目录落地
- 三数据中间层已生成
- Hotpot 的三方法双后端已跑通并出 `pred.jsonl`
- `metrics.csv` 已可正常汇总

2. 注意：
- 这轮我实际跑通 Hotpot 时用过 smoke 参数做链路验证；你后续正式跑数请按“全量命令”执行（不传 `--limit/--max_docs` 或保持其默认 `0`）。
