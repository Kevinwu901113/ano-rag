# ANO-RAG: 结构化构建与检索两阶段方法报告

本文基于仓库 `Kevinwu901113/ano-rag` 的实际代码实现，对项目的两阶段（构建与检索）进行系统性分析，并以论文式结构呈现：动机、相关工作、算法结构（构建、检索）、实验设置与结果。分析严格依据代码文件与索引结构，不引用 README 的描述。为便于讨论，Mirage 数据集仅在第四节用于说明实验设置与结果。

## 1. 动机

- 传统 RAG 大多依赖非结构化文本块，检索与答案生成的解释性与可控性不足。ANO-RAG 的核心动机是通过将文档转化为结构化的事实笔记（subj–pred–obj 三元组），构建轻量知识图索引以实现“先检索后推理”的透明流程。
- 结构化范式的优势：
  - 可解释性：检索路径由显式边和节点构成（`graph_edges.jsonl`），证据可回溯至 `note_id` 与 `evidence` 字段。
  - 精准约束：通过类型边索引（`type_edge_index.json`）与谓词归一规则限制合法跳转，提高候选质量。
  - 别名稳健性：实体别名索引（`entity_alias_index.json`）在绑定阶段提升召回，支持归一、包含与松匹配。
  - 结构内兜底：在结构范围内进行 BM25 的“弱信号补全”，比纯文本全局检索更具针对性与可控性。

## 2. 相关工作

- KGQA 与图检索：知识图问答通过图路径匹配与约束求解提升解释性。相关方法强调从实体绑定到路径扩展的检索-推理范式。
- 结构化 RAG：结合三元组索引实现检索与答案生成的分离，常见技巧包括三元组线性化、别名词典与属性归一，将非结构化证据映射到结构通道。
- 传统 BM25/倒排检索：在结构化范围内应用 BM25 可作为弱信号与噪声鲁棒的补全手段；ANO-RAG 用 `FieldWeightedBM25` 与简化 BM25 实现并支持字段权重。
- 别名与实体链接：别名字典与归一策略（括号压缩、Unicode 规范化、大小写折叠）在提升跨文体与噪声下的实体匹配上有显著作用。

说明：本节观点与术语与本仓库代码一致，引用的技术手段在后文对应代码实现处给出。

## 3. 算法结构

### 3.1 构建阶段（文档→笔记→索引）

- 目标：将输入文档切分为段落/句子块，交给 LLM 提取事实笔记，随后进行验证与归一，最终产出用于检索的结构化索引。
- 关键组件与文件：
  - 文档切分：`doc/chunker.py::make_chunks`（基于句子窗口与重叠），赋予 `doc_id`/`chunk_id`。
  - 笔记生成：`generator/note_generator.py::NoteGenerator` 构造严格 JSON 输出（`subj/pred/obj/subj_type/obj_type/evidence/meta`），调用 LLM 提取事实。
  - 验证与归一：`validators/note_validator.py::validate_and_normalize` 对谓词与类型组合进行校验与规范（`PRED_SYNONYM_SETS`, `PRED2ATTR`），计算质量分。
  - 索引构建：`indexer/index_builder.py::IndexBuilder` 生成：
    - `entity_to_notes.json`：实体到 `note_id` 的倒排；
    - `predicate_to_notes.json`：谓词到笔记；
    - `graph_edges.jsonl`/`inverse_edges.jsonl`：有向边与反向边；
    - `type_edge_index.json`：`(subj_type, pred, obj_type)` 到笔记；
    - `field_index.json`：属性值归一后的倒排（如 occupation）；
    - `entity_alias_index.json`：别名到实体集合。

- 归一化要点（摘自代码）：
  - 别名归一（`_normalize_alias` 与检索侧 `_normalize_alias_query`）
    - 去括号、破折号/点号转空格、Unicode 规范化、多空格压缩、小写折叠。
  - 属性值归一（`_normalize_field_value`）
    - 去括号和标点、Unicode 规范化、性别与复数统一（职业如 actress→actor，cartoon artist→cartoonist）；
    - 提供简单词干 `_stem_occupation`，多键写入提升召回。

- 构建阶段伪代码：

```
for chunk in make_chunks(doc):
    note = llm_extract_fact(chunk)  # subj, pred, obj, types, evidence, meta
    note_norm = validate_and_normalize(note)  # 类型/谓词规范、质量分
    index_builder.add_note(note_norm)  # 写入多类倒排与图索引

index_builder.dump(out_dir)
```

- 结构性重要性：图边与类型约束为检索阶段提供了可解释的“跳转空间”；别名与属性值归一保证实体绑定与属性检索的稳定性；质量分与置信度为后续重排提供信号。

### 3.2 检索阶段（问题→解析→绑定→扩展→打分→兜底）

- 关键组件与文件：
  - 意图识别：`retriever/intent_detector.py::AnswerIntentDetector` 推断 `entity/attribute/question_type/entity_type/confidence`。
  - 问题解析：`retriever/parser.py::parse_question` 将自然语言映射为 `QueryIR`（种子、谓词链、目标类型、fanout、max_hops）。
  - 绑定与扩展：`retriever/operators.py::BIND/EXPAND_from` 完成实体别名匹配与图邻接遍历。
  - 路径打分：`retriever/scorer.py::score_path` 按跳数惩罚；目标类型命中加分见 `pipeline._walk_chain`。
  - 管线总控：`retriever/pipeline.py::retrieve_answer` 组织结构检索与兜底流程（属性收集、BM25结构内检索、候选重排）。

- 检索阶段伪代码：

```
intent = detect_intent(question)
ir = parse_question(question)
if ir is None:
    return fallback_lookup(intent, indexes, note_store, None, "parse_failed")

seeds = bind_seeds(ir.seeds)  # BIND: 别名精确/归一精确/包含/松匹配
if not seeds:
    return fallback_lookup(intent, indexes, note_store, ir, "no_seed_match")

if ir.pred_chain is empty:
    candidates = collect_entity_mentions(seeds)
else:
    states = [{entity: e, path: []} for e in seeds]
    for step in ir.pred_chain[:ir.max_hops]:
        next_states = []
        for s in states:
            for (obj, note_id) in EXPAND_from(indexes, s.entity, step.pred, step.direction, ir.fanout):
                next_states.append({entity: obj, path: s.path + edge(s.entity, step.pred, obj, note_id)})
        states = next_states[:ir.fanout]
    candidates = score_and_make_candidates(states)

if not candidates:
    structured = structured_fallback(seeds, intent)  # 结构范围内 BM25
    if structured: return structured
    return fallback_lookup(intent, indexes, note_store, ir, "no_path")

final = rerank(candidates, intent.attribute)  # occupation 优先
return assemble_result(ir, final, evidences)
```

- 路径打分与目标类型约束：

公式（来自 `scorer.py` 与 `pipeline._walk_chain`）：

```
score(path) = 1.0 - 0.05 * (len(path) - 1) + type_bonus
type_bonus = 0.1 if final_note.obj_type == ir.target_type else 0
```

- 结构化兜底（BM25）策略：
  - 在绑定得到的实体作用域 `entities[:10]` 内收集其相关笔记 `[:200]`，形成 `scoped_notes`。
  - 构建 BM25 语料（`evidence + obj` 拼接），查询用 `intent.entity`。
  - Occupation 优先：对 `attribute in {occupation, title}` 的笔记加分或优先排序。
  - 返回单边“结构化命中路径”（标识 `fallback: true`）与证据 Top-5。

- 别名绑定细节（`BIND`）：
  - 层次顺序：精确匹配 → 归一精确 → 包含匹配（仅当前两者无命中）→ 对 `entity_to_notes` 键的包含/松匹配（3 字符以上 token）。
  - 检索端归一规则（`_normalize_alias_query`）与构建端 `_normalize_alias` 协同，保证键对齐。

- Occupation 重排（`_rerank_candidates`）：
  - 如果意图属性为 `occupation`，则对候选路径末边谓词为 `occupation/title` 的候选加 `+0.05`，仅在候选集合内部重排。

### 3.3 结构检索范式总结

- 范式要点：
  - 从自然语言到结构 IR（`QueryIR`）的映射是检索的入口；
  - 绑定（BIND）将别名与实体对齐，扩展（EXPAND）在图上沿指定谓词链遍历；
  - 打分以“少跳数+类型一致”为偏好，Occupation 类问题设有域内加权；
  - 兜底在结构范围内做 BM25，提高在弱解析或弱结构信号下的鲁棒性；
  - 全程证据链透明，结果可通过 `note_id` 回溯。

## 4. 实验设置与结果（Mirage）

- 设置：
  - 数据集：`data/mirage/dataset`（问题集），对应结果位于 `result/019-mirage/answers.json`。
  - 索引：使用上述构建流程得到的 `indexes/`（别名、实体倒排、图边、类型边、字段倒排）。
  - 参数：`fanout` 典型为 12–15；BM25 使用简化实现或 `rank_bm25`（如可用），`k1=1.5, b=0.75`。

- 统计（由 `answers.json` 解析）：
  - 总问题数：`20`
  - 结构化命中（不使用兜底）：`15`
  - 使用兜底：`5`（全部为 `no_seed_match` 场景）
  - 问题属性分布：`occupation` 20/20；问题类型：`what` 20/20
  - 平均路径长度：`1.0`（单边）
  - 平均证据条数：`4.25`

- 示例（摘自 `answers.json`）：
  - “What is John Mayne's occupation?”
    - 结构 IR：`intent=open_entity_query`，`seeds=['John Mayne']`，`pred_chain=[]`，`fanout=15`
    - 路径：单边 `__mention__` 命中多条证据；未触发兜底（`fallback.used=false`）
    - 证据：来自多 `note_id` 的 `evidence` 字段，如“Scottish printer, journalist and poet…”。

- 观察：
  - Occupation 类问题多为单边检索（mention→属性），结构化策略在别名与字段倒排的配合下命中较稳。
  - 兜底场景集中在实体绑定失败时，通过结构范围内 BM25 仍可提取合理答案值。

## 5. 讨论与局限

- 结构化的优势：
  - 更强可解释性与约束性；
  - 边与类型索引可过滤不合理跳转；
  - 别名与属性归一提升跨语体与噪声鲁棒性。

- 局限与改进方向：
  - 谓词库（`PREDICATE_LIBRARY`）与复合规则有限，复杂问题需扩充规则或学习型解析；
  - 路径打分较简洁（固定线性惩罚），可引入基于质量分、实体流行度、属性一致性等多特征学习重排；
  - 结构兜底目前以 BM25 为主，可加入属性感知的加权或向量检索与重排融合。

## 6. 结论

ANO-RAG 以“结构化构建—结构检索—结构内兜底”的范式，形成了端到端可解释的 QA 流程。构建阶段通过验证与归一保证结构质量；检索阶段以别名绑定、图扩展与类型约束实现稳定召回；兜底在结构范围内以 BM25 提升弱信号场景鲁棒性。实验表明，Occupation 类问题在该范式下获得良好的结构命中率与证据覆盖，体现了结构化索引与检索的实用性与可扩展价值。

——

附：关键公式与伪代码均来自本仓库代码（`indexer/`, `retriever/`, `utils/` 等）中的具体实现，包括 BM25 参数与路径打分项。若需扩展为更复杂任务，可在不改变范式的前提下扩充谓词库与类型约束，并引入学习型重排模块。