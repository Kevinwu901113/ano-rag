# 项目清理记录

本次已移除全部旧的向量 / 混合检索相关模块和脚本，保留的代码只围绕结构化三元组管线（`doc/chunker.py` → `generator/` → `indexer/` → `retriever/` → `main.py`）运行。

## 已删除的目录/脚本
- 目录：`adapters/`, `answer/`, `chunker/`, `configs/`, `context/`, `eval/`, `graph/`, `llm/`, `parallel/`, `reasoning/`, `retrieval/`, `scripts/`, `support/`, `training/`, `vector_store/`, `test_notes/`, `MIRAGE/`
- 旧入口及工具脚本：`answer_selector.py`, `convert_to_official_format.py`, `enhanced_evaluator.py`, `extract_dev200.py`, `install_requirements.py`, `main_build_notes.py`, `main_mirage.py`, `main_musique.py`, `main_query.py`, `run_evaluation.py`, `test_mirage_small.py`
- 旧测试与迁移文档：`tests/`、`README_EFSA.md`、`VLLM_MIGRATION_NOTES.md`

## 当前仍保留但待评估的资源
- `docs/`：多为旧架构示意图/说明，如需继续沿用需更新内容。
- `data/`、`logs/`, `indexes/`, `notes/`：示例数据与运行产物，可视需求归档或清空。
- `validator/`（与 `validators/` 并存的旧目录）：若确认无引用，可在下一轮移除。

> 如需回溯旧实现，建议在清理前参考 Git 历史或创建专门的 `legacy` 分支。
