#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线标准化脚本

基于raw_span和raw_span_evidence字段使用正则表达式填充normalized_entities和predicates字段

功能：
1. 从raw_span和raw_span_evidence中提取实体
2. 从raw_span和raw_span_evidence中提取谓词
3. 标准化提取的实体和谓词
4. 批量处理atomic_notes数据
5. 支持增量更新和全量更新
"""

import json
import re
import logging
import argparse
from typing import Dict, List, Any, Set, Tuple, Optional
from pathlib import Path
from collections import defaultdict, Counter
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.entity_predicate_normalizer import EntityPredicateNormalizer, create_entity_predicate_normalizer
from utils.text_utils import TextUtils

logger = logging.getLogger(__name__)

class OfflineNormalizer:
    """
    离线标准化处理器
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化离线标准化器
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            config = self._get_default_config()
        
        # 初始化实体谓词标准化器
        self.normalizer = create_entity_predicate_normalizer(config.get('normalizer', {}))
        
        # 实体提取正则模式
        self.entity_patterns = [
            # 人名模式
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # John Smith
            r'\b[A-Z][a-z]+\s+[A-Z]\.\s+[A-Z][a-z]+\b',  # John A. Smith
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # John Michael Smith
            
            # 组织机构模式
            r'\b[A-Z][a-zA-Z\s&]+(?:Inc|Corp|Ltd|LLC|Company|Corporation|Organization|University|College|Institute)\b',
            r'\b(?:The\s+)?[A-Z][a-zA-Z\s&]+(?:Foundation|Association|Society|Group|Team|Department)\b',
            
            # 地点模式
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:City|State|Country|Province|County|District))\b',
            r'\b(?:New\s+York|Los\s+Angeles|San\s+Francisco|Washington\s+D\.C\.|United\s+States|United\s+Kingdom)\b',
            
            # 产品/品牌模式
            r'\b[A-Z][a-zA-Z0-9\s]+(?:™|®)\b',
            r'\b(?:iPhone|Android|Windows|MacOS|Linux|Microsoft|Apple|Google|Amazon|Facebook|Twitter)\b',
        ]
        
        # 谓词提取正则模式
        self.predicate_patterns = [
            # 创建/建立关系
            (r'(\w+)\s+(?:founded|established|created|co-founded|cofounded)\s+(\w+)', 'founded'),
            (r'(\w+)\s+(?:co-founded|cofounded)\s+(\w+)', 'co_founded'),
            
            # 位置关系
            (r'(\w+)\s+(?:located|based|situated)\s+in\s+(\w+)', 'located_in'),
            (r'(\w+)\s+is\s+in\s+(\w+)', 'located_in'),
            
            # 工作关系
            (r'(\w+)\s+(?:works?\s+for|employed\s+by|works?\s+at)\s+(\w+)', 'works_for'),
            (r'(\w+)\s+is\s+(?:a|an)\s+(?:employee|worker|member)\s+of\s+(\w+)', 'works_for'),
            
            # 成员关系
            (r'(\w+)\s+(?:member\s+of|belongs\s+to)\s+(\w+)', 'member_of'),
            (r'(\w+)\s+is\s+(?:a|an)\s+member\s+of\s+(\w+)', 'member_of'),
            
            # 部分关系
            (r'(\w+)\s+(?:part\s+of|component\s+of)\s+(\w+)', 'part_of'),
            (r'(\w+)\s+is\s+(?:a|an)\s+part\s+of\s+(\w+)', 'part_of'),
            
            # 实例关系
            (r'(\w+)\s+(?:instance\s+of|type\s+of|kind\s+of)\s+(\w+)', 'instance_of'),
            (r'(\w+)\s+is\s+(?:a|an)\s+(\w+)', 'instance_of'),
        ]
        
        # 统计信息
        self.stats = {
            'processed_notes': 0,
            'entities_extracted': 0,
            'predicates_extracted': 0,
            'entities_normalized': 0,
            'predicates_normalized': 0,
            'errors': 0
        }
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'normalizer': {
                'entity_normalizer': {
                    'fuzzy_threshold': 0.8,
                    'enable_fuzzy_matching': True,
                    'case_sensitive': False
                },
                'predicate_normalizer': {
                    'fuzzy_threshold': 0.8,
                    'enable_fuzzy_matching': True,
                    'case_sensitive': False
                }
            }
        }
    
    def extract_entities_from_text(self, text: str) -> List[str]:
        """
        从文本中提取实体
        
        Args:
            text: 输入文本
            
        Returns:
            提取的实体列表
        """
        entities = set()
        
        for pattern in self.entity_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    entities.update(match)
                else:
                    entities.add(match)
        
        # 过滤和清理实体
        cleaned_entities = []
        for entity in entities:
            entity = entity.strip()
            if len(entity) > 1 and not entity.isdigit():
                cleaned_entities.append(entity)
        
        return cleaned_entities
    
    def extract_predicates_from_text(self, text: str) -> List[Tuple[str, str, str]]:
        """
        从文本中提取谓词关系
        
        Args:
            text: 输入文本
            
        Returns:
            提取的谓词关系列表 [(subject, predicate, object), ...]
        """
        relations = []
        
        for pattern, predicate_type in self.predicate_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) >= 2:
                    subject = match[0].strip()
                    obj = match[1].strip()
                    if subject and obj and subject != obj:
                        relations.append((subject, predicate_type, obj))
        
        return relations
    
    def normalize_note(self, note: Dict[str, Any], force_update: bool = False) -> Dict[str, Any]:
        """
        标准化单个笔记
        
        Args:
            note: 原始笔记
            force_update: 是否强制更新已有的标准化字段
            
        Returns:
            标准化后的笔记
        """
        try:
            # 复制笔记避免修改原始数据
            normalized_note = note.copy()
            
            # 获取文本内容
            raw_span = note.get('raw_span', '')
            raw_span_evidence = note.get('raw_span_evidence', '')
            content = note.get('content', '')
            
            # 合并所有文本用于提取
            combined_text = ' '.join(filter(None, [raw_span, raw_span_evidence, content]))
            
            # 提取实体
            extracted_entities = self.extract_entities_from_text(combined_text)
            
            # 合并现有实体
            existing_entities = note.get('entities', [])
            all_entities = list(set(existing_entities + extracted_entities))
            
            # 标准化实体
            normalized_entities = []
            if force_update or not note.get('normalized_entities'):
                for entity in all_entities:
                    normalized_entity, confidence = self.normalizer.normalize_entity(entity)
                    if confidence > 0.5:  # 只保留高置信度的标准化结果
                        normalized_entities.append(normalized_entity)
                    else:
                        normalized_entities.append(entity)  # 保留原始实体
                
                normalized_note['normalized_entities'] = list(set(normalized_entities))
                self.stats['entities_normalized'] += len(normalized_entities)
            
            # 提取谓词关系
            extracted_relations = self.extract_predicates_from_text(combined_text)
            
            # 标准化谓词
            normalized_predicates = []
            if force_update or not note.get('normalized_predicates'):
                predicates_set = set()
                
                # 从提取的关系中获取谓词
                for subject, predicate, obj in extracted_relations:
                    predicates_set.add(predicate)
                
                # 从现有谓词中获取
                existing_predicates = note.get('predicates', [])
                predicates_set.update(existing_predicates)
                
                # 标准化谓词
                for predicate in predicates_set:
                    normalized_predicate, confidence = self.normalizer.normalize_predicate(predicate)
                    if confidence > 0.5:
                        normalized_predicates.append(normalized_predicate)
                    else:
                        normalized_predicates.append(predicate)
                
                normalized_note['normalized_predicates'] = list(set(normalized_predicates))
                self.stats['predicates_normalized'] += len(normalized_predicates)
            
            # 更新统计
            self.stats['processed_notes'] += 1
            self.stats['entities_extracted'] += len(extracted_entities)
            self.stats['predicates_extracted'] += len(extracted_relations)
            
            return normalized_note
            
        except Exception as e:
            logger.error(f"Error normalizing note: {e}")
            self.stats['errors'] += 1
            return note
    
    def process_notes_file(self, input_file: str, output_file: str, force_update: bool = False) -> None:
        """
        处理笔记文件
        
        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径
            force_update: 是否强制更新已有的标准化字段
        """
        logger.info(f"Processing notes file: {input_file}")
        
        # 读取输入文件
        with open(input_file, 'r', encoding='utf-8') as f:
            notes = json.load(f)
        
        if not isinstance(notes, list):
            raise ValueError("Input file must contain a list of notes")
        
        # 处理每个笔记
        normalized_notes = []
        for i, note in enumerate(notes):
            if i % 100 == 0:
                logger.info(f"Processing note {i+1}/{len(notes)}")
            
            normalized_note = self.normalize_note(note, force_update)
            normalized_notes.append(normalized_note)
        
        # 保存结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(normalized_notes, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Processed {len(notes)} notes, saved to {output_file}")
        self.print_stats()
    
    def print_stats(self) -> None:
        """打印统计信息"""
        logger.info("Normalization Statistics:")
        logger.info(f"  Processed notes: {self.stats['processed_notes']}")
        logger.info(f"  Entities extracted: {self.stats['entities_extracted']}")
        logger.info(f"  Predicates extracted: {self.stats['predicates_extracted']}")
        logger.info(f"  Entities normalized: {self.stats['entities_normalized']}")
        logger.info(f"  Predicates normalized: {self.stats['predicates_normalized']}")
        logger.info(f"  Errors: {self.stats['errors']}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='离线标准化脚本')
    parser.add_argument('input_file', help='输入的atomic_notes JSON文件路径')
    parser.add_argument('output_file', help='输出的标准化JSON文件路径')
    parser.add_argument('--config', help='配置文件路径')
    parser.add_argument('--force-update', action='store_true', help='强制更新已有的标准化字段')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别')
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 检查输入文件
    if not Path(args.input_file).exists():
        logger.error(f"Input file not found: {args.input_file}")
        return 1
    
    # 创建输出目录
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # 初始化标准化器
        normalizer = OfflineNormalizer(args.config)
        
        # 处理文件
        normalizer.process_notes_file(args.input_file, args.output_file, args.force_update)
        
        logger.info("Normalization completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error during normalization: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())