# LLM 全链路迁移报告（vLLM + Qwen3-30B-A3B）

目标：移除旧本地服务与多 provider 逻辑，统一 vLLM OpenAI-compatible 接口，提取/生成同模型（Qwen3-30B-A3B），并记录 extract/generate profile 参数。

## 删除/替换清单（路径 + 简述）
- `utils/llm_client.py`: 新增统一 vLLM 客户端（固定 endpoint/model，extract/generate profiles，extract 关闭 thinking）。
- `generator/answerer.py`, `generator/extractor.py`, `generator/note_generator.py`, `retriever/rerank.py`: 全部改用 `LLMChatClient`，移除旧直连调用。
- `baselines/common/model_clients.py`, `baselines/*/runner.py`, `structrag/*.py`, `rag_core/llm_client.py`: 统一走 `LLMChatClient`，去掉多 provider 路由。
- `scripts/mirage/*.py`, `scripts/musique/*.py`, `scripts/hotpotqa/*.py`, `main_mirage.py`, `main_query.py`, `run_fid_custom.py`, `run_baselines.sh`: 旧本地 LLM 参数统一替换为 `--lm-endpoint/--lm-model`，默认 `http://127.0.0.1:8000/v1` + `qwen3-30b-a3b`。
- `scripts/mirage/build_notes.sh`, `scripts/musique/run_musique.sh`, `scripts/hotpotqa/baselines/run_graphrag_with_vllm.sh`, `scripts/modal_vllm_launcher.py`: vLLM 启动模型/端口/缓存目录统一固定，显式设置代理用于首次下载。
- `config.yaml`, `config/config_loader.py`, `experiments/configs/*.yaml`: endpoint/model 固定为 vLLM + qwen3-30b-a3b，移除其他 provider 配置。
- `README.md`, `intro.md`, `doc/*.md`: 文档与示例命令统一为 vLLM 与固定模型。
- `analysis/hotpotqa_relrag_case_study_20/config.json`, `analysis/hotpotqa_relrag_case_study_20/cases.jsonl`: 旧 endpoint/model 字符串替换为新值。

## 新增资产
- `scripts/llm/start_vllm_qwen3_30b_a3b.sh`: 多卡 vLLM 启动模板（自动 TP=GPU 数，端口 8000，download-dir 固定缓存目录，默认 `--enforce-eager`，启动前设置代理）。

> 说明：首次启动会通过 `--download-dir` 下载模型到本地缓存目录；后续复用该目录，不会重复下载。

## 残留检查
- 旧 provider 关键字与旧端口字符串检查 → 0 处残留

## 最小验收命令（不含 healthcheck）
1) 启动 vLLM（含代理）：
```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/llm/start_vllm_qwen3_30b_a3b.sh
```

2) 跑最小实验（audit_full.yaml，limit=20）：
```bash
python experiments/run.py --config experiments/configs/audit_full.yaml
```

3) 产物验证（抽样 3 条 pred_raw，且不含 "error"）：
```bash
RUN_DIR=$(ls -td result_relrag/audit_full_matrix/run_* | head -1)
python - <<'PY'
import json, glob, os
files = glob.glob(os.path.join("result_relrag", "audit_full_matrix", "run_*", "*", "*", "*", "*", "*", "preds", "pred_raw.jsonl"))
path = files[0] if files else None
print("pred_raw sample from:", path)
if path:
    with open(path, "r", encoding="utf-8") as f:
        for i, line in zip(range(3), f):
            print(json.loads(line).get("pred_raw"))
PY
rg -n "error" "${RUN_DIR}"/*/*/*/*/*/preds/pred_raw.jsonl
```
