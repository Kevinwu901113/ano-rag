#!/usr/bin/env python3
"""
Note Graph V2 构建驱动脚本

使用 v2 模块串联构建笔记图谱：
1. candidate_gate.build_candidates_for_note() → 候选 ≤ topB
2. dbtes_select_edges() → 边选择
3. mmr_select() → ≤ keep_k
4. materialize.write_snapshot() → 产出 GraphML + JSONL

命令例子：
PYTHONPATH=. python scripts/build_note_graph_v2.py \
  --notes data/notes.jsonl \
  --out graph_v2/snapshots \
  --config config.yaml
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Callable
import yaml
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from retrieval_v2.candidate_gate import build_candidates_for_note
from atomic_v2.dbtes import dbtes_select_edges
from retrieval_v2.mmr import mmr_select
from graph_v2.materialize import write_snapshot
from llm.lmstudio_client import LMStudioClient
from llm.ollama_client import OllamaClient
from llm.router import build_router_from_config


class NoteGraphBuilder:
    """Note Graph V2 构建器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.router = self._init_router()
        
        # 从配置中获取参数
        self.retrieval_v2_config = config.get("retrieval_v2", {})
        self.dbtes_config = self.retrieval_v2_config.get("dbtes", {})
        self.mmr_config = config.get("mmr", {})
        self.graph_v2_config = config.get("graph_v2", {})
        
        # 并行处理配置
        llm_pool_config = config.get("llm_pool", {})
        backends = llm_pool_config.get("backends", [])
        total_workers = sum(backend.get("max_inflight", 1) for backend in backends) * 2
        self.max_workers = max(1, total_workers)  # 确保至少有1个worker
        
        # 运行统计信息
        self.run_stats = {
            "start_time": None,
            "end_time": None,
            "total_duration_s": 0,
            "note_count": 0,
            "candidate_count": 0,
            "selected_edge_count": 0,
            "llm_pool_stats": {}
        }
        
        logger.info(f"初始化 NoteGraphBuilder，使用路由器，并行度: {self.max_workers}")
    
    def _init_router(self):
        """初始化路由器"""
        try:
            # 创建 LMStudio 调用函数
            def lmstudio_call(prompt: str) -> str:
                try:
                    lmstudio_config = self.config.get("atomic_note_generation", {}).get("lmstudio", {})
                    client = LMStudioClient()
                    messages = [{"role": "user", "content": prompt}]
                    response = client.chat(messages)
                    return response.strip() if response else ""
                except Exception as e:
                    logger.error(f"LMStudio 调用失败: {e}")
                    return ""
            
            # 创建 Ollama 调用函数（可选）
            def ollama_call(prompt: str) -> str:
                try:
                    client = OllamaClient()
                    response = client.generate(prompt)
                    return response.strip() if response else ""
                except Exception as e:
                    logger.error(f"Ollama 调用失败: {e}")
                    return ""
            
            # 检查是否有 Ollama 配置
            ollama_config = self.config.get("atomic_note_generation", {}).get("ollama", {})
            has_ollama = bool(ollama_config.get("base_url"))
            
            # 构建路由器
            router = build_router_from_config(
                self.config, 
                lmstudio_call, 
                ollama_call if has_ollama else None
            )
            
            logger.info(f"路由器初始化成功，后端: {list(router.backends.keys())}")
            return router
            
        except Exception as e:
            logger.error(f"路由器初始化失败: {e}")
            raise RuntimeError(f"无法初始化路由器: {e}")
    
    def _init_llm_client(self) -> Any:
        """初始化 LLM 客户端"""
        # 优先使用 lmstudio，回退到 ollama
        try:
            lmstudio_config = self.config.get("atomic_note_generation", {}).get("lmstudio", {})
            if lmstudio_config.get("base_url"):
                logger.info("尝试初始化 LMStudio 客户端")
                client = LMStudioClient()
                # 简单健康检查
                if hasattr(client, 'is_available') and client.is_available():
                    logger.info("LMStudio 客户端初始化成功")
                    return client
                else:
                    logger.warning("LMStudio 不可用，尝试 Ollama")
        except Exception as e:
            logger.warning(f"LMStudio 初始化失败: {e}，尝试 Ollama")
        
        try:
            ollama_config = self.config.get("atomic_note_generation", {}).get("ollama", {})
            logger.info("初始化 Ollama 客户端")
            client = OllamaClient()
            if hasattr(client, 'is_available') and client.is_available():
                logger.info("Ollama 客户端初始化成功")
                return client
            else:
                logger.error("Ollama 不可用")
        except Exception as e:
            logger.error(f"Ollama 初始化失败: {e}")
        
        raise RuntimeError("无法初始化任何 LLM 客户端")
    
    def _estimate_tokens_per_edge(self, edge: Dict[str, Any]) -> int:
        """粗估边的 token 数量（按单词数近似）"""
        try:
            # 提取边中的文本内容
            text_content = ""
            
            # 从边的各个字段中提取文本
            if "reasoning" in edge:
                text_content += str(edge["reasoning"]) + " "
            if "evidence" in edge:
                text_content += str(edge["evidence"]) + " "
            if "source_content" in edge:
                text_content += str(edge["source_content"]) + " "
            if "target_content" in edge:
                text_content += str(edge["target_content"]) + " "
            
            # 简单的单词计数（中英文混合）
            # 英文单词按空格分割，中文字符按字符计算
            words = text_content.split()
            chinese_chars = sum(1 for char in text_content if '\u4e00' <= char <= '\u9fff')
            
            # 粗估：英文单词 * 1.3 + 中文字符 * 1.5
            estimated_tokens = int(len(words) * 1.3 + chinese_chars * 1.5)
            
            return estimated_tokens
        except Exception as e:
            logger.warning(f"估算边 token 数量失败: {e}")
            return 0
    
    def _check_budget_warning(self, edges: List[Dict[str, Any]]) -> None:
        """检查预算并打印警告"""
        try:
            total_estimated_tokens = 0
            high_cost_edges = []
            
            for edge in edges:
                tokens = self._estimate_tokens_per_edge(edge)
                total_estimated_tokens += tokens
                
                if tokens > 800:
                    high_cost_edges.append({
                        "edge_id": edge.get("id", "unknown"),
                        "source": edge.get("source", "unknown"),
                        "target": edge.get("target", "unknown"),
                        "estimated_tokens": tokens
                    })
            
            # 打印预算警告
            if high_cost_edges:
                logger.warning(f"[WARN] 发现 {len(high_cost_edges)} 条高成本边（>800 tokens）:")
                for edge_info in high_cost_edges:
                    logger.warning(f"[WARN] 边 {edge_info['edge_id']}: {edge_info['source']} -> {edge_info['target']}, 估算 tokens: {edge_info['estimated_tokens']}")
            
            logger.info(f"总估算 tokens: {total_estimated_tokens}, 高成本边数量: {len(high_cost_edges)}")
            
        except Exception as e:
            logger.error(f"预算检查失败: {e}")

    def _process_single_note(self, center_note: Dict[str, Any], notes: List[Dict[str, Any]], 
                           notes_dict: Dict[str, Dict[str, Any]], weak_graph: Dict[str, Any]) -> List[Dict[str, Any]]:
        """处理单个笔记，生成相关边"""
        center_id = center_note["id"]
        
        try:
            # 1. 构建候选
            candidates = build_candidates_for_note(
                center_note=center_note,
                notes=notes,
                config=self.retrieval_v2_config
            )
            
            # 更新候选数统计
            self.run_stats["candidate_count"] += len(candidates)
            
            if not candidates:
                logger.debug(f"笔记 {center_id} 没有候选")
                return []
            
            # 2. DBTES 边选择
            selected_edges = dbtes_select_edges(
                center_note=center_note,
                candidates=candidates,
                notes_dict=notes_dict,
                weak_graph=weak_graph,
                llm_call=self.router.route,
                config=self.dbtes_config
            )
            
            if not selected_edges:
                logger.debug(f"笔记 {center_id} DBTES 未选中任何边")
                return []
            
            # 3. MMR 选择
            final_edges = mmr_select(
                edges=selected_edges,
                config=self.mmr_config
            )
            
            return final_edges
            
        except Exception as e:
            logger.error(f"处理笔记 {center_id} 时出错: {e}")
            return []

    def load_notes(self, notes_path: str) -> List[Dict[str, Any]]:
        """加载笔记文件"""
        notes_path = Path(notes_path)
        
        if not notes_path.exists():
            raise FileNotFoundError(f"笔记文件不存在: {notes_path}")
        
        notes = []
        
        if notes_path.suffix == '.jsonl':
            # JSONL 格式
            with open(notes_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        note = json.loads(line)
                        notes.append(note)
                    except json.JSONDecodeError as e:
                        logger.warning(f"跳过无效 JSON 行 {line_num}: {e}")
        
        elif notes_path.suffix == '.json':
            # JSON 格式
            with open(notes_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    notes = data
                else:
                    raise ValueError("JSON 文件应包含笔记数组")
        
        else:
            raise ValueError(f"不支持的文件格式: {notes_path.suffix}")
        
        logger.info(f"加载了 {len(notes)} 条笔记")
        return notes

    def build_graph(self, notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """构建笔记图谱 - 使用并行处理"""
        logger.info("开始构建笔记图谱（并行模式）")
        
        # 记录开始时间
        self.run_stats["start_time"] = time.time()
        self.run_stats["note_count"] = len(notes)
        
        # 创建笔记索引
        notes_dict = {note["id"]: note for note in notes}
        
        # 初始化弱图（空图，struct_prior 需要）
        weak_graph = {}
        
        all_edges = []
        total_notes = len(notes)
        
        # 使用 ThreadPoolExecutor 并行处理
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_note = {}
            for center_note in notes:
                future = executor.submit(
                    self._process_single_note, 
                    center_note, 
                    notes, 
                    notes_dict, 
                    weak_graph
                )
                future_to_note[future] = center_note
            
            # 收集结果
            completed = 0
            for future in as_completed(future_to_note):
                center_note = future_to_note[future]
                center_id = center_note["id"]
                completed += 1
                
                try:
                    edges = future.result()
                    all_edges.extend(edges)
                    logger.info(f"完成笔记 {completed}/{total_notes}: ID={center_id}, 生成 {len(edges)} 条边")
                except Exception as e:
                    logger.error(f"处理笔记 {center_id} 失败: {e}")
        
        # 记录结束时间和统计信息
        self.run_stats["end_time"] = time.time()
        self.run_stats["total_duration_s"] = self.run_stats["end_time"] - self.run_stats["start_time"]
        self.run_stats["selected_edge_count"] = len(all_edges)
        
        # 检查预算警告
        self._check_budget_warning(all_edges)
        
        logger.info(f"图谱构建完成，共生成 {len(all_edges)} 条边，用时 {self.run_stats['total_duration_s']:.2f} 秒")
        return all_edges
    
    def save_graph(self, edges: List[Dict[str, Any]], notes: List[Dict[str, Any]], 
                   output_dir: str) -> str:
        """保存图谱到文件"""
        logger.info(f"保存图谱到: {output_dir}")
        
        # 读取度裁剪配置，如果存在则复用
        degree_cap = self.graph_v2_config.get("degree_cap", 20)
        logger.info(f"使用度裁剪配置: degree_cap = {degree_cap}")
        
        snapshot_dir = write_snapshot(
            edges=edges,
            notes=notes,
            out_dir=output_dir,
            degree_cap=degree_cap
        )
        
        # 获取 LLM 池统计信息
        try:
            self.run_stats["llm_pool_stats"] = self.router.stats
        except Exception as e:
            logger.warning(f"获取 LLM 池统计信息失败: {e}")
            self.run_stats["llm_pool_stats"] = {}
        
        # 保存运行日志
        run_log_file = Path(snapshot_dir) / "run_log.json"
        try:
            with open(run_log_file, 'w', encoding='utf-8') as f:
                json.dump(self.run_stats, f, indent=2, ensure_ascii=False)
            logger.info(f"运行日志已保存到: {run_log_file}")
        except Exception as e:
            logger.error(f"保存运行日志失败: {e}")
        
        # 保存 LLM 池统计信息（保持原有功能）
        stats_file = Path(snapshot_dir) / "llm_pool_stats.json"
        try:
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.router.stats, f, indent=2, ensure_ascii=False)
            logger.info(f"LLM 池统计信息已保存到: {stats_file}")
        except Exception as e:
            logger.error(f"保存 LLM 池统计信息失败: {e}")
        
        logger.info(f"图谱已保存到: {snapshot_dir}")
        return snapshot_dir


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.suffix in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        elif config_path.suffix == '.json':
            return json.load(f)
        else:
            raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="构建 Note Graph V2")
    parser.add_argument("--notes", required=True, help="笔记文件路径 (.jsonl 或 .json)")
    parser.add_argument("--out", required=True, help="输出目录")
    parser.add_argument("--config", required=True, help="配置文件路径")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细日志")
    
    args = parser.parse_args()
    
    # 配置日志
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.remove()
        logger.add(sys.stderr, level="INFO")
    
    try:
        # 加载配置
        config = load_config(args.config)
        
        # 创建构建器
        builder = NoteGraphBuilder(config)
        
        # 加载笔记
        notes = builder.load_notes(args.notes)
        
        # 构建图谱
        edges = builder.build_graph(notes)
        
        # 保存结果
        snapshot_dir = builder.save_graph(edges, notes, args.out)
        
        logger.info("构建完成！")
        logger.info(f"输出目录: {snapshot_dir}")
        
        # 打印统计信息
        logger.info(f"LLM 池统计: {builder.router.stats}")
        
    except Exception as e:
        logger.error(f"构建失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()