# 冗余代码盘点（2024-11-10）

## 方法
- 以 `main.py` → `pipeline/structured_builder.py` → `generator/` → `indexer/` → `retriever/` → `query/query_processor.py` 为主路径，梳理真实执行链。
- 使用 `rg`/`nl` 校验每个可疑模块的引用情况（例如 `rg -n "get_adapter"`、`rg --no-ignore -n "NotesCache"`、`rg -n "retriever\.retrieve"`）。
- 将只被 Git 忽略脚本引用或完全无引用的模块判定为冗余，将与现行架构冲突的脚本划入“待移除”。

## 冗余 / 待删除项

### 1. 旧版数据构建栈仍残留
- **路径**：`main_build_notes.py`、`adapters/`、`scripts/mirage/build_notes.sh`
- **证据**：
  - `CLEANUP_STATUS.md:5-8` 声明 `adapters/`、`main_build_notes.py` 已删除，但仓库仍存在。
  - `adapters/__init__.py:1-16` 仅注册 `mirage` 适配器，`rg -n "get_adapter"` 显示唯一调用点为 `main_build_notes.py:11-37`。
  - `scripts/mirage/build_notes.sh:201-221` 继续通过 `python main_build_notes.py` 驱动旧流水线，与 `main.py process` 双轨。
- **影响**：持续维护两套入口，易导致配置漂移；`CLEANUP_STATUS` 与实际不符。
- **建议**：统一到 `StructuredBuilder`，删除 `main_build_notes.py` 与 `adapters/`，同时替换/废弃 `scripts/mirage` 里的旧调用。

### 2. 重复的查询入口
- **路径**：`main_query.py`
- **证据**：文件 `main_query.py:1-27` 只是 `QueryProcessor` 的薄封装，功能与 `main.py query` 完全一致；除 `CLEANUP_STATUS.md:7` 外无任何引用。
- **建议**：移除该文件或将其内容改成简单的 `main.py` wrapper，避免 CLI 混乱。

### 3. 被 `.gitignore` 屏蔽的 Musique 自动化及其依赖
- **路径**：`scripts/musique/`（整目录）、`utils/notes_cache.py`、`utils/vllm_server_manager.py`
- **证据**：
  - `.gitignore:173-201` 明确忽略 `musique/`，意味着该目录不会被版本控制或CI覆盖。
  - `scripts/musique/run.py:27-38` 仍在导入 `NotesCache` 与 `VLLMServerManager`，并包含 800+ 行独立执行逻辑。
  - `rg --no-ignore -n "NotesCache"` / `"VLLMServerManager"` 仅在 `scripts/musique/run.py` 与对应 utils( `utils/notes_cache.py:15-118`、`utils/vllm_server_manager.py:13-78` ) 中出现。
- **影响**：这套工具既不受版本控制，又强依赖 pandas/requests/本地 vLLM 管理，长期漂移且无法复用。
- **建议**：若不再需要 Musique 管线，将整个目录与专属 utils 移除；若需保留，至少把脚本移出忽略列表并补充 README/测试。

### 4. “增强”配置与实体规范化子系统
- **路径**：`config/enhanced_config.py`、`utils/enhanced_ner.py`、`utils/enhanced_relation_extractor.py`、`utils/note_normalizer.py`、`utils/summary_auditor.py`、`utils/notes_quality_filter.py`、`utils/notes_parser.py`、`utils/note_completeness.py`
- **证据**：
  - `rg -n "enhanced_config"` 无任何外部引用；`config/enhanced_config.py:1-59` 仅定义静态字典。
  - `utils/enhanced_relation_extractor.py:3-20` 依赖早已删除的 `graph.relation_extractor`（同样在 `CLEANUP_STATUS.md:5` 被标记），导入即失败。
  - `rg --no-ignore -n "EnhancedNER"`, `"NoteNormalizer"`, `"SummaryAuditor"` 仅命中这些文件内部，未被主流程使用。
  - `utils/notes_quality_filter.py` 和 `utils/notes_parser.py` 互相引用，但外部没有入口。
- **影响**：大量 HuggingFace / regex / pandas 逻辑闲置，占用维护心智，还可能误导新人以为仍有增强管线。
- **建议**：整体归档至 `legacy/` 或直接删除；若未来需要其中部分能力，应重新设计并接入现行流水线。

### 5. 离线归一化脚本与文档
- **路径**：`utils/offline_normalization_script.py`、`utils/README_offline_normalization.md`
- **证据**：`rg --no-ignore -n "offline_normalization_script"` 仅在 README 中出现；脚本自身 (`utils/offline_normalization_script.py:1-78`) 依赖上一节的 `EntityPredicateNormalizer` / `notes_parser` 等冗余模块。
- **影响**：脚本无法单独运行（依赖已弃用模块），但 README 仍提示可用，容易造成误导。
- **建议**：同步删除或在 README 中标注“legacy/不可用”；若需保留功能，需先恢复/精简依赖。

### 6. 过时的检索实现
- **路径**：`retriever/retrieve.py`
- **证据**：`rg -n "from retriever\.retrieve"` 无任何命中，`QueryProcessor` 只调用 `retriever/pipeline.py:84-143`，表明 `retriever/retrieve.py` 已完全游离。
- **影响**：与现行 `retrieve_answer` 逻辑（支持 IR、Hybrid）重复且易混淆。
- **建议**：删除该文件，避免贡献者在选择入口时踩坑。

### 7. 生成产物与文档不一致
- **路径**：`result/`, `logs/`, `indexes/tmp/`, `data/**`
- **证据**：`CLEANUP_STATUS.md:10-13` 将其列为“可视需要归档”，但仓库依旧携带大量 run 产物（`result/000-...` 等）。
- **影响**：增加仓库体积且掩盖真实的最小示例。
- **建议**：将样例数据移至独立存储或以脚本生成，仅保留最小示例并在 README 中说明。

## 其他备注
- `CLEANUP_STATUS.md` 与实际状态不符（例如声称 `adapters/` 已移除、仍存在 `validator/` 目录却未见），需要更新该文档以防信息错配。
