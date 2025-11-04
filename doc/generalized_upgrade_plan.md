# 后续泛化改造行动清单

## 检索与索引
- [ ] 将 `retriever/pipeline.py` 中的兜底逻辑与字段化索引 (`field_index.json`) 打通，支持属性值倒排查找与实体候选交叉验证。
- [ ] 引入定义句/首句检测特征，对 `note_store` 返回的证据打标，配合重排器优先选择定义型证据。
- [ ] 构建图路径重排模块，综合 `score_path`、`meta.quality_score` 和字段命中情况，输出结构化得分报告。

## 上下文调度
- [ ] 在 `query/query_processor.py` 中引入上下文角色（definition/support/context）调度，重新组织传给生成模型的证据顺序。
- [ ] 将 `fallback` 产出的候选摘要与结构化检索结果合并，避免重复证据并限制 token。

## 生成与后处理
- [ ] 设计受约束解码接口，使 `generator/note_generator.py` 支持仅返回目标槽位集合或 JSON，便于 LM 最终回答阶段复用。
- [ ] 在 `validators/note_validator.py` 之后追加标准化聚合层，支持多值集合去重、排序及向量化回写别名。

## 置信度与拒答
- [ ] 结合 `meta.final_conf`、`meta.quality_score`、证据数量等信号训练可回答性打分器，触发 `unknown` 拒答策略。
- [ ] 在输出中记录置信度决策信息，方便 A/B 测试与链路观测。

## 观测与评估
- [ ] 接入指标采集（解析命中率、兜底触发率、字段命中率、受约束遵从率等），并在日志或仪表盘输出。
- [ ] 设计小规模自动化评测数据集，监控 occupation/nationality/title 等属性问答的集合 F1 与正确拒答率。
