# HotpotQA 运行诊断与证据整合报告（基于 result/result_1768126196）

## 1. 范围与材料
- 数据集: `data/hotpot_dev_distractor_5_jsonl.jsonl`（5 题）
- 输出: `result/result_1768126196.jsonl`, `result/result_1768126196_official.json`
- 中间产物:
  - `result/debug/<qid>.json`（包含 doc_index、note_ids、record/intermediate）
  - `result/cache/<qid>/docs/*.txt`（原始段落）
  - `result/cache/<qid>/notes.jsonl`（抽取到的 notes）
  - `result/cache/<qid>/indexes/`（索引）
- 说明: debug 文件只保存“被检索/引用”的 notes；完整 notes 需要直接看 `notes.jsonl`。

## 2. 总体结果概览
| qid | 问题 | 标准答案 | 本次输出 | 结果 |
| --- | --- | --- | --- | --- |
| 5ae143ed55429920d5234360 | In what year was the university where Sergei Aleksandrovich Tokarev was a professor founded? | 1755 | Insufficient evidence | MISS |
| 5abc19705542993a06baf86e | Black Book starred the actress and writer of what heritage? | Dutch | Insufficient evidence | MISS |
| 5ac3e0f7554299194317388b | Which actor does American Beauty and American Beauty have in common? | Kevin Spacey | Kevin Spacey | OK |
| 5ae518655542993aec5ec139 | Ken Pruitt was a Republican member of an upper house of the legislature with how many members? | 40 members | Insufficient evidence | MISS |
| 5ab985eb554299131ca42360 | Between Greyia and Calibanus, which genus contains more species? | Greyia | Insufficient evidence | MISS |

## 3. 初始异常与直接证据（来自运行日志/用户提供）
- 依赖缺失导致无法启动:
  - `ModuleNotFoundError: No module named 'loguru'`
- BM25 后端未实现导致降级:
  - `BM25 backend 'pyserini' is not implemented; falling back to rank_bm25.`
- LLM 请求超上下文长度导致生成失败（触发重试，可能拖住线程池）:
  - `HTTP error ... status=400 ... 'max_tokens' is too large ... request has 2128 input tokens (1975 > 4096 - 2128)`
- JSON schema 校验失败（birth 类型错误）:
  - `1903 is not of type 'string', 'null' ... subject_profile.birth: 1903`
- 异常后变量未初始化导致崩溃:
  - `cannot access local variable 'pronoun_notes' where it is not associated with a value`
- “CPU/GPU 空闲但程序不结束”的主要触发路径:
  - LLM 调用重试/backoff + 线程池 wait 阻塞，导致主进程无明显 CPU/GPU 活跃却仍在等待未完成任务。

## 4. 全局结构性原因（基于中间产物）
- 解析/IR 误判: 多题的 `intermediate.ir.seeds` 抽取错误，导致检索起点偏离目标实体。
- 缺少多跳链路: 例如 Tokarev 题需要 `works_for -> founded_on`，但 IR 只有单跳 `founded_on`。
- Schema 覆盖不足: 缺少 “成员数/物种数” 等数值型谓词，导致即便原文有数字也无法结构化对比。
- 抽取不完整: 多实体列表（演员名单）未完全展开为独立 notes（缺少 Halina Reijn 的 performed_by）。
- 结构化检索被错误证据主导: Tokarev 题的 hybrid 共识集中在 nationality（Russian），偏离问题核心。

## 5. 分题分析（含原始证据与中间内容）

### 5ae143ed55429920d5234360（Tokarev → MSU founded_on）
- 标准答案: 1755
- 原始证据（supporting facts）:
  - `Sergei Aleksandrovich Tokarev` [0] Sergei Aleksandrovich Tokarev ... professor at Moscow State University.
  - `Moscow State University` [3] It was founded on January 25, 1755 by Mikhail Lomonosov.
- 抽取到的关键 notes（来自 `result/cache/5ae143ed55429920d5234360/notes.jsonl`）:
  - `...tokarev#c0001#0` pred=works_for obj=Moscow State University | evidence="professor at Moscow State University"
  - `...moscow_state_university#c0001#3` pred=founded_on obj=1755-01-25 | evidence="It was founded on January 25, 1755 by Mikhail Lomonosov."
- 中间解析（`result/debug/5ae143ed55429920d5234360.json`）:
```json
{
  "intent": {
    "entity": "the university where Sergei Aleksandrovich Tokarev was a professor",
    "entity_type": "ORG",
    "attribute": "founded_on",
    "question_type": "where",
    "confidence": 0.95
  },
  "ir": {
    "intent": "relation_query",
    "seeds": [{"text": "Sergei Aleksandrovich Tokarev", "type_hint": "ORG"}],
    "pred_chain": [{"pred": "founded_on", "direction": "out", "target_hint": "TIME"}],
    "target_type": "TIME",
    "max_hops": 1,
    "fallback": false
  }
}
```
- 检索结果关键迹象:
  - `support_note_ids` 只有 nationality note；`hybrid.consensus_label = "Russian"`，`support_pool_size = 1`
- 失效原因:
  - 需要两跳（Tokarev → works_for → MSU → founded_on），但 IR 只有单跳 founded_on。
  - seed 类型错误（Tokarev 被当作 ORG），导致结构化检索路径偏离。

### 5abc19705542993a06baf86e（Black Book → Halina Reijn → Dutch）
- 标准答案: Dutch
- 原始证据（supporting facts）:
  - `Black Book (film)` [0] ... starring ... Halina Reijn.
  - `Halina Reijn` [0] Halina Reijn ... is a Dutch actress and writer.
- 抽取到的关键 notes（来自 `result/cache/5abc19705542993a06baf86e/notes.jsonl`）:
  - `...black_book_film#c0000#6` pred=performed_by obj=Carice van Houten | evidence 包含 Halina Reijn（演员列表）
  - `...halina_reijn#c0000#0` pred=occupation obj=actor | evidence="Dutch actress and writer"
  - `...halina_reijn#c0000#1` pred=nationality obj=Dutch | evidence="Dutch actress and writer"
- 中间解析（`result/debug/5abc19705542993a06baf86e.json`）:
```json
{
  "intent": {
    "entity": "Black Book starred the actress and writer of what",
    "entity_type": "PERSON",
    "attribute": "nationality",
    "question_type": "what",
    "confidence": 0.95
  },
  "ir": {
    "intent": "relation_query",
    "seeds": [{"text": "the actress and writer of", "type_hint": "WORK"}],
    "pred_chain": [{"pred": "acted_in", "direction": "in", "target_hint": "PERSON"}],
    "target_type": "PERSON",
    "max_hops": 1,
    "fallback": false
  }
}
```
- 检索结果关键迹象:
  - `support_note_ids` 为空，`reason = no_path`。
- 失效原因:
  - 解析 seed 错误，未把 “Black Book” 识别为主体实体。
  - performed_by 未明确生成 Halina Reijn 的独立 note（演员列表未完全展开）。

### 5ac3e0f7554299194317388b（American Beauty 共用演员）
- 标准答案: Kevin Spacey
- 原始证据（supporting facts）:
  - `American Beauty (soundtrack)` [0] ... soundtrack album to the 1999 film starring Kevin Spacey ...
  - `American Beauty (1999 film)` [1] Kevin Spacey stars as Lester Burnham ...
- 抽取到的关键 notes（示例）:
  - `...accolades...#c0000#6` pred=acted_in obj=Kevin Spacey | evidence="The film stars Kevin Spacey"
  - `...american_beauty_soundtrack#c0000#0` evidence 包含 “starring Kevin Spacey”
- 中间解析: `open_entity_query` fallback，但 BM25/embedding 仍命中 Kevin Spacey，最终答案正确。

### 5ae518655542993aec5ec139（Ken Pruitt → Florida Senate → 40 members）
- 标准答案: 40 members
- 原始证据（supporting facts）:
  - `Ken Pruitt` [0] ... Republican member of the Florida Senate ...
  - `Florida Senate` [0] The Florida Senate is the upper house ...
  - `Florida Senate` [2] The Senate has 40 members ...
- 抽取到的关键 notes:
  - `...ken_pruitt#c0000#0` pred=works_for obj=Florida Senate | evidence="Republican member of the Florida Senate"
  - `...ken_pruitt#c0000#1` pred=member_of obj=Florida Senate | same evidence
  - Florida Senate 没有 “member_count / size” 类谓词
- 中间解析:
```json
{
  "intent": {
    "entity": "Republican",
    "entity_type": null,
    "attribute": null,
    "question_type": "how",
    "confidence": 0.75
  },
  "ir": {
    "intent": "relation_query",
    "seeds": [{"text": "an upper house of the legislature with", "type_hint": "PERSON"}],
    "pred_chain": [{"pred": "member_of", "direction": "out", "target_hint": "ORG"}],
    "target_type": "ORG",
    "max_hops": 1,
    "fallback": false
  }
}
```
- 检索结果关键迹象:
  - `support_note_ids` 来自无关实体（Murtaza Ahmed Khan），`reason = no_path`。
- 失效原因:
  - seed 抽取失败，未锁定 Ken Pruitt / Florida Senate。
  - Schema 缺失“成员数”谓词，无法读取 “40 members”。

### 5ab985eb554299131ca42360（Greyia vs Calibanus 物种数对比）
- 标准答案: Greyia
- 原始证据（supporting facts）:
  - `Greyia` [0] Greyia is a genus of plant ...
  - `Greyia` [1] It contains three species:
  - `Calibanus` [0] Calibanus is a genus of two species ...
- 抽取到的关键 notes:
  - `...greyia#c0000#0` pred=type obj=genus | evidence="Greyia is a genus ..."
  - `...calibanus#c0000#0` pred=type obj=GENUS | evidence="genus of two species ..."
  - 没有 species_count / species_num 类谓词
- 中间解析:
```json
{
  "intent": {
    "entity": "Calibanus",
    "entity_type": null,
    "attribute": null,
    "question_type": "which",
    "confidence": 0.75
  },
  "ir": {
    "intent": "open_entity_query",
    "seeds": [
      {"text": "Greyia", "type_hint": null},
      {"text": "Calibanus", "type_hint": null}
    ],
    "pred_chain": [],
    "target_type": null,
    "max_hops": 1,
    "fallback": true
  }
}
```
- 检索结果关键迹象:
  - `support_note_ids` 为空，`reason = no_path`。
- 失效原因:
  - 需要可比较的“物种数量”结构化属性，当前 schema 不支持。

## 6. 可复用的排错路径（中间产物对照）
- 看原文证据: `result/cache/<qid>/docs/*.txt` 或 `result/debug/<qid>.json` 里的 doc_index。
- 看抽取是否覆盖: `result/cache/<qid>/notes.jsonl`（是否存在目标谓词/数值）。
- 看解析/检索是否跑偏: `result/debug/<qid>.json` → `record.intermediate.intent` / `ir` / `retrieve_result`。
- 看最终落地: `result/result_1768126196.jsonl`（每题完整输出），`result/result_1768126196_official.json`（官方格式）。
