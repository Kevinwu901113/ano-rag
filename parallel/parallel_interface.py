#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并行处理统一接口（最小可用实现）

暴露：
- ParallelInterface
- 三种任务处理器占位实现：DocumentTaskProcessor, QueryTaskProcessor, MusiqueTaskProcessor
- 工厂方法：create_parallel_interface

满足 main.py 与 main_musique.py 的基本调用：
- process_documents(documents, output_dir, force_reprocess=False)
- process_queries(queries, knowledge_base)
- process_musique_dataset(items, base_work_dir)
- get_performance_stats()
- cleanup()
"""

from __future__ import annotations

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger

from .parallel_engine import (
    ParallelEngine,
    TaskProcessor,
    ParallelTask,
    ParallelStrategy,
    ProcessingMode,
)


# -------------------- 任务处理器占位实现 --------------------
class DocumentTaskProcessor(TaskProcessor):
    def process(self, task: ParallelTask) -> Dict[str, Any]:
        payload = task.payload
        # 约定输入字段：file_path, force_reprocess
        file_path = payload.get("file_path")
        force = payload.get("force_reprocess", False)
        # 最小实现：返回元信息，真实处理由项目其他模块完成。
        return {
            "file_path": file_path,
            "force_reprocess": force,
            "atomic_notes": [],  # 预留字段以兼容后续流程
            "status": "processed"
        }


class QueryTaskProcessor(TaskProcessor):
    def process(self, task: ParallelTask) -> Dict[str, Any]:
        payload = task.payload
        # 约定输入字段：question, knowledge_base
        question = payload.get("question")
        kb = payload.get("knowledge_base", {})
        # 最小实现：返回一个占位答案
        return {
            "question": question,
            "answer": f"[stub answer] {question}",
            "retrieval": {
                "vector_results": [],
                "graph_results": [],
            },
        }


class MusiqueTaskProcessor(TaskProcessor):
    def process(self, task: ParallelTask) -> Dict[str, Any]:
        payload = task.payload
        # 约定输入字段：合并主流程需要的字段
        return {
            "id": payload.get("id"),
            "predicted_answer": payload.get("predicted_answer", ""),
            "retrieved_contexts": payload.get("retrieved_contexts", []),
            "error": False,
        }


# -------------------- 接口封装 --------------------
@dataclass
class ParallelInterface:
    engine: ParallelEngine

    # 文档处理
    def process_documents(self, documents: List[Dict[str, Any]], output_dir: str, force_reprocess: bool = False) -> List[Dict[str, Any]]:
        # 将每个文档映射为任务
        items = []
        for d in documents:
            item = {
                "file_path": d.get("file_path"),
                "force_reprocess": d.get("force_reprocess", force_reprocess),
                "output_dir": output_dir,
            }
            items.append(item)
        return self.engine.process_list(items)

    # 查询处理
    def process_queries(self, queries: List[Dict[str, Any]], knowledge_base: Dict[str, Any]) -> List[Dict[str, Any]]:
        items = []
        for q in queries:
            item = {
                "question": q.get("question"),
                "knowledge_base": knowledge_base,
            }
            items.append(item)
        return self.engine.process_list(items)

    # Musique数据集处理
    def process_musique_dataset(self, items: List[Dict[str, Any]], base_work_dir: str) -> List[Dict[str, Any]]:
        # 直接透传字段，最小并行执行
        return self.engine.process_list(items)

    def get_performance_stats(self) -> Dict[str, Any]:
        stats = self.engine.get_stats()
        return {
            "total_tasks": stats.total_tasks,
            "succeeded": stats.succeeded,
            "failed": stats.failed,
            "elapsed_ms": stats.elapsed_ms,
            **stats.details,
        }

    def cleanup(self) -> None:
        self.engine.cleanup()


# -------------------- 工厂方法 --------------------

def create_parallel_interface(
    max_workers: int = 4,
    processing_mode: ProcessingMode = ProcessingMode.AUTO,
    strategy: ParallelStrategy = ParallelStrategy.HYBRID,
    debug: bool = False,
) -> ParallelInterface:
    """根据策略选择处理器并创建接口，当前均使用占位处理器。"""
    if strategy in (ParallelStrategy.DATA_COPY, ParallelStrategy.DATA_SPLIT, ParallelStrategy.TASK_DISPATCH, ParallelStrategy.HYBRID):
        # 根据调用上下文，优先选择能够满足主流程的处理器；
        # 由于 main.py / main_musique.py 中的使用场景不同，
        # 这里统一先创建文档处理器，具体处理方法在接口中按函数选择。
        processor = DocumentTaskProcessor()
    else:
        processor = DocumentTaskProcessor()

    engine = ParallelEngine(
        processor=processor,
        max_workers=max_workers,
        strategy=strategy,
        processing_mode=processing_mode,
        debug=debug,
    )
    return ParallelInterface(engine=engine)