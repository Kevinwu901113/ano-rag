# Skills 使用说明（面向本仓库 Agent）

本仓库提供一组可复用 Skills，位于 `.trae/skills/`。每个 Skill 都是一个目录，包含一个 `SKILL.md`，其内容是“启用该 Skill 时应遵循的工作流与输出约定”。当用户点名某个 Skill 或引用对应 `SKILL.md` 路径时，应优先读取并遵循该说明。

## 引用方式
- 用户直接引用：`@.trae/skills/<skill-name>/SKILL.md`
- 用户点名：例如“用 relrag-pipeline-debugger ……”或“启用 webapp-testing ……”等

## 选择建议
- 新增技能：优先用 `skill-creator`
- 复杂任务先规划：用 `create-plan`
- 需要代码审查：用 `code-reviewer`
- 需要补齐测试：用 `test-writer`（或 `webapp-testing` 做端到端/接口回归）
- 需要定位 bug：用 `bug-triage`
- 需要安全审查：用 `security-reviewer`
- 需要性能定位：用 `performance-profiler`
- 需要重构拆分：用 `refactor-assistant`
- 需要写上手文档：用 `repo-onboarding`
- 需要定位 CI 失败：用 `ci-debugger`
- 需要发版材料：用 `git-release`
- 需要规范提交信息：用 `git-commit-writer`
- 需要依赖升级评估：用 `dependency-auditor`
- 需要外部工具/服务接入：用 `mcp-builder`
- 需要验证改动是否有效：用 `webapp-testing` 或项目内测试
- RelRAG 相关问题（召回、格式、指标回归）：优先选择 `relrag-*` 与 `hotpot-eval-analyst`
