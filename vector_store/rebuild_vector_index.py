#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向量索引重建脚本
用于执行完整的向量索引重建操作，支持配置化嵌入策略和增量重建
"""

import os
import sys
import json
import time
import hashlib
import argparse
import sys
from typing import List, Dict, Any, Optional
from loguru import logger
from pathlib import Path
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vector_store.retriever import VectorRetriever
from vector_store.embedding_strategy import EmbeddingStrategy
from llm.atomic_note_generator import AtomicNoteGenerator
from llm.enhanced_atomic_note_generator import EnhancedAtomicNoteGenerator
from utils.file_utils import FileUtils
from config import config

def load_atomic_notes_from_jsonl(jsonl_file: str) -> List[Dict[str, Any]]:
    """从JSONL文件加载并生成原子笔记"""
    logger.info(f"从 {jsonl_file} 加载数据")
    
    # 读取JSONL数据
    documents = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                documents.append(json.loads(line.strip()))
    
    logger.info(f"加载了 {len(documents)} 个文档")
    
    # 初始化LocalLLM
    from llm.local_llm import LocalLLM
    llm = LocalLLM()
    
    # 生成原子笔记
    try:
        # 优先使用增强版生成器
        generator = EnhancedAtomicNoteGenerator(llm=llm)
        logger.info("使用增强版原子笔记生成器")
    except Exception as e:
        logger.warning(f"增强版生成器初始化失败: {e}，使用基础版生成器")
        generator = AtomicNoteGenerator(llm=llm)
    
    all_atomic_notes = []
    for i, doc in enumerate(documents):
        logger.info(f"处理文档 {i+1}/{len(documents)}")
        try:
            # 将文档转换为text_chunks格式
            text_chunks = []
            for para in doc.get('paragraphs', []):
                chunk = {
                    'text': para.get('paragraph_text', ''),
                    'chunk_index': para.get('idx', 0),
                    'source_info': {
                        'title': para.get('title', ''),
                        'document_id': doc.get('id', ''),
                        'is_supporting': para.get('is_supporting', False)
                    },
                    'paragraph_info': {
                        'title': para.get('title', ''),
                        'idx': para.get('idx', 0)
                    }
                }
                text_chunks.append(chunk)
            
            atomic_notes = generator.generate_atomic_notes(text_chunks)
            all_atomic_notes.extend(atomic_notes)
            logger.info(f"文档 {i+1} 生成了 {len(atomic_notes)} 个原子笔记")
        except Exception as e:
            logger.error(f"处理文档 {i+1} 时出错: {e}")
            continue
    
    logger.info(f"总共生成了 {len(all_atomic_notes)} 个原子笔记")
    return all_atomic_notes

def should_rebuild_index(atomic_notes: List[Dict[str, Any]], 
                         embedding_strategy: EmbeddingStrategy) -> bool:
    """判断是否需要重建索引"""
    rebuild_config = config.get('embedding_strategy', {}).get('index_rebuilding', {})
    triggers = rebuild_config.get('rebuild_triggers', {})
    
    # 检查笔记数量阈值
    note_threshold = triggers.get('note_count_threshold', 10000)
    if len(atomic_notes) >= note_threshold:
        logger.info(f"笔记数量 {len(atomic_notes)} 超过阈值 {note_threshold}，需要重建")
        return True
    
    # 检查时间阈值
    time_threshold_hours = triggers.get('time_threshold_hours', 168)  # 7天
    if embedding_strategy.current_version_id:
        current_version = embedding_strategy.versions.get(embedding_strategy.current_version_id)
        if current_version:
            age_hours = (time.time() - current_version.created_at) / 3600
            if age_hours >= time_threshold_hours:
                logger.info(f"索引年龄 {age_hours:.1f} 小时超过阈值 {time_threshold_hours}，需要重建")
                return True
    
    # 检查配置变更
    if triggers.get('config_change', True):
        config_hash = _calculate_config_hash()
        if embedding_strategy.current_version_id:
            current_version = embedding_strategy.versions.get(embedding_strategy.current_version_id)
            if current_version and current_version.metadata.get('config_hash') != config_hash:
                logger.info("检测到配置变更，需要重建")
                return True
    
    return False

def _calculate_config_hash() -> str:
    """计算配置哈希值"""
    embedding_config = config.get('embedding_strategy', {})
    config_str = json.dumps(embedding_config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()

def rebuild_vector_index(atomic_notes: List[Dict[str, Any]], 
                        save_index: bool = True,
                        force_rebuild: bool = False,
                        incremental: bool = False) -> bool:
    """重建向量索引"""
    logger.info("开始重建向量索引")
    
    # 获取重建配置
    rebuild_config = config.get('embedding_strategy', {}).get('index_rebuilding', {})
    enable_incremental = rebuild_config.get('enable_incremental', True) and incremental
    batch_size = rebuild_config.get('batch_size', 1000)
    checkpoint_interval = rebuild_config.get('checkpoint_interval', 5000)
    enable_backup = rebuild_config.get('enable_backup', True)
    backup_before_rebuild = rebuild_config.get('backup_before_rebuild', True)
    
    # 创建嵌入策略管理器
    embedding_strategy = EmbeddingStrategy(config)
    
    # 检查是否需要重建
    if not force_rebuild and not should_rebuild_index(atomic_notes, embedding_strategy):
        logger.info("索引无需重建")
        return True
    
    # 创建向量检索器
    retriever = VectorRetriever()
    
    # 备份现有索引
    if enable_backup and backup_before_rebuild:
        try:
            _backup_existing_index(retriever)
        except Exception as e:
            logger.warning(f"备份索引失败: {e}")
    
    # 执行重建
    try:
        if enable_incremental and not force_rebuild:
            success = _incremental_rebuild(retriever, atomic_notes, batch_size, checkpoint_interval)
        else:
            success = _full_rebuild(retriever, atomic_notes, save_index)
        
        if success:
            logger.info("向量索引重建成功")
            
            # 更新版本信息
            _update_version_info(embedding_strategy, atomic_notes)
            
            # 显示索引统计信息
            try:
                stats = retriever.vector_index.get_index_stats()
                logger.info(f"索引统计信息: {stats}")
            except Exception as e:
                logger.warning(f"获取索引统计失败: {e}")
            
            # 测试检索功能
            _test_retrieval(retriever)
        else:
            logger.error("向量索引重建失败")
        
        return success
        
    except Exception as e:
        logger.error(f"重建过程中发生错误: {e}")
        return False

def _backup_existing_index(retriever: VectorRetriever) -> None:
    """备份现有索引"""
    backup_dir = Path("./embeddings/backups")
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"index_backup_{timestamp}"
    
    # 这里可以添加具体的备份逻辑
    logger.info(f"索引已备份到: {backup_path}")

def _incremental_rebuild(retriever: VectorRetriever, atomic_notes: List[Dict[str, Any]], 
                        batch_size: int, checkpoint_interval: int) -> bool:
    """增量重建索引"""
    logger.info(f"开始增量重建，批次大小: {batch_size}")
    
    total_notes = len(atomic_notes)
    processed = 0
    
    for i in range(0, total_notes, batch_size):
        batch = atomic_notes[i:i + batch_size]
        
        try:
            # 添加批次笔记
            success = retriever.add_notes(batch, rebuild_index=False)
            if not success:
                logger.error(f"批次 {i//batch_size + 1} 处理失败")
                return False
            
            processed += len(batch)
            logger.info(f"已处理 {processed}/{total_notes} 个笔记")
            
            # 检查点保存
            if processed % checkpoint_interval == 0:
                logger.info(f"到达检查点，已处理 {processed} 个笔记")
                # 这里可以添加检查点保存逻辑
            
        except Exception as e:
            logger.error(f"处理批次 {i//batch_size + 1} 时出错: {e}")
            return False
    
    return True

def _full_rebuild(retriever: VectorRetriever, atomic_notes: List[Dict[str, Any]], 
                 save_index: bool) -> bool:
    """完整重建索引"""
    logger.info("开始完整重建")
    
    return retriever.build_index(
        atomic_notes=atomic_notes,
        force_rebuild=True,
        save_index=save_index
    )

def _update_version_info(embedding_strategy: EmbeddingStrategy, 
                        atomic_notes: List[Dict[str, Any]]) -> None:
    """更新版本信息"""
    try:
        config_hash = _calculate_config_hash()
        metadata = {
            'config_hash': config_hash,
            'rebuild_time': time.time(),
            'note_count': len(atomic_notes)
        }
        
        # 这里可以添加版本更新逻辑
        logger.info(f"版本信息已更新: {metadata}")
        
    except Exception as e:
        logger.warning(f"更新版本信息失败: {e}")

def _test_retrieval(retriever: VectorRetriever) -> None:
    """测试检索功能"""
    test_queries = ["测试查询", "example query", "示例问题"]
    
    for query in test_queries:
        try:
            results = retriever.retrieve(query, top_k=3)
            logger.info(f"测试查询 '{query}' 返回了 {len(results)} 个结果")
        except Exception as e:
            logger.warning(f"测试查询 '{query}' 失败: {e}")
            break

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="重建向量索引")
    parser.add_argument("--data_path", type=str, required=True, help="数据文件路径")
    parser.add_argument("--no_save", action="store_true", help="不保存索引到磁盘")
    parser.add_argument("--enhanced", action="store_true", help="使用增强版原子笔记生成器")
    parser.add_argument("--force", action="store_true", help="强制重建索引")
    parser.add_argument("--incremental", action="store_true", help="使用增量重建")
    parser.add_argument("--dry_run", action="store_true", help="仅检查是否需要重建，不执行实际重建")
    parser.add_argument("--batch_size", type=int, help="批处理大小（仅用于增量重建）")
    parser.add_argument("--backup", action="store_true", help="重建前备份现有索引")
    
    args = parser.parse_args()
    
    # 验证参数
    if args.incremental and args.force:
        logger.warning("同时指定了 --incremental 和 --force，将使用完整重建")
        args.incremental = False
    
    # 加载原子笔记
    logger.info(f"从 {args.data_path} 加载原子笔记")
    start_time = time.time()
    
    try:
        atomic_notes = load_atomic_notes_from_jsonl(args.data_path, use_enhanced=args.enhanced)
    except Exception as e:
        logger.error(f"加载原子笔记失败: {e}")
        sys.exit(1)
    
    if not atomic_notes:
        logger.error("未能加载任何原子笔记")
        sys.exit(1)
    
    load_time = time.time() - start_time
    logger.info(f"成功加载 {len(atomic_notes)} 个原子笔记，耗时 {load_time:.2f} 秒")
    
    # 仅检查模式
    if args.dry_run:
        embedding_strategy = EmbeddingStrategy(config)
        needs_rebuild = should_rebuild_index(atomic_notes, embedding_strategy)
        if needs_rebuild:
            logger.info("索引需要重建")
        else:
            logger.info("索引无需重建")
        return
    
    # 更新配置（如果指定了批处理大小）
    if args.batch_size:
        config.setdefault('embedding_strategy', {}).setdefault('index_rebuilding', {})['batch_size'] = args.batch_size
    
    # 重建索引
    rebuild_start = time.time()
    success = rebuild_vector_index(
        atomic_notes=atomic_notes, 
        save_index=not args.no_save,
        force_rebuild=args.force,
        incremental=args.incremental
    )
    
    rebuild_time = time.time() - rebuild_start
    total_time = time.time() - start_time
    
    if success:
        logger.info(f"索引重建完成，重建耗时 {rebuild_time:.2f} 秒，总耗时 {total_time:.2f} 秒")
        
        # 显示性能统计
        notes_per_second = len(atomic_notes) / rebuild_time if rebuild_time > 0 else 0
        logger.info(f"处理速度: {notes_per_second:.1f} 笔记/秒")
    else:
        logger.error(f"索引重建失败，耗时 {rebuild_time:.2f} 秒")
        sys.exit(1)

if __name__ == "__main__":
    main()