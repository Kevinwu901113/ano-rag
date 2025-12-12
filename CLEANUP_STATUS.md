# 项目清理记录 / 当前现状

仓库已完成一次大整理：主结构化三元组笔记管线集中在  
`doc/chunker.py` → `generator/` → `indexer/` → `retriever/` → `query/` → `main.py`，  
并在 `scripts/` 中提供 MIRAGE / HotpotQA / MuSiQue 的一键运行与基线评测。

## 已完成的清理

- 旧的向量检索/混合检索栈已从主链路剥离（详见 Git 历史）。
- 配置入口统一为 `config.yaml` + `config/config_loader.py`。
- 对比基线集中在 `baselines/` 与 `scripts/*/baselines/`。

## 仍保留的实验/Legacy 组件

- `main_build_notes.py` 与 `adapters/`：目前主要被 `scripts/mirage/build_notes.sh` 使用，属于旧入口；如需完全统一到 `main.py process`，可在后续迭代中合并/替换。
- `retriever/retrieve.py`、`rag_core/`、`analysis/` 等：不走主链路或仅用于实验，可视需求归档/删减。
- 运行产物目录（建议保持在 `.gitignore` 中）：`notes/`、`indexes/`、`result/`、`logs/`、`ans/` 等。

> 需要回溯旧实现或清理背景，请参考 Git 历史或建立 `legacy` 分支。  
