import os
import json
import random
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
from collections import defaultdict

from config import config
from .enhanced_ner import EnhancedNER


class SummaryAuditor:
    """两阶段摘要校验器
    
    阶段一（快速审查）：基于NER工具抽取原文与摘要中的实体，进行结构化比对
    阶段二（精准判断）：当第一阶段发现疑似丢失信息时，调用LLM进一步判断
    """
    
    def __init__(self):
        self.config = config.get('summary_auditor', {})
        self.enabled = self.config.get('enabled', True)
        self.llm_check_ratio = self.config.get('llm_check_ratio', 0.2)
        self.ner_model = self.config.get('ner_model', 'spacy')
        self.entity_similarity_threshold = self.config.get('entity_similarity_threshold', 0.8)
        self.missing_entity_threshold = self.config.get('missing_entity_threshold', 0.3)
        
        # 初始化NER工具
        if self.ner_model == 'spacy':
            self.ner = EnhancedNER()
        elif self.ner_model == 'ltp':
            # 如果需要LTP支持，可以在这里添加LTP实现
            logger.warning("LTP NER not implemented, falling back to spacy")
            self.ner = EnhancedNER()
        else:
            logger.warning(f"Unknown NER model: {self.ner_model}, using spacy")
            self.ner = EnhancedNER()
        
        # 初始化LLM（延迟加载）
        self._llm = None
        
        # 统计信息
        self.audit_stats = {
            'total_audited': 0,
            'stage1_flagged': 0,
            'stage2_checked': 0,
            'rewrite_recommended': 0
        }
    
    @property
    def llm(self):
        """延迟加载LLM"""
        if self._llm is None:
            from llm.local_llm import LocalLLM
            self._llm = LocalLLM()
        return self._llm
    
    def audit_summary(self, original_text: str, summary: str, note_id: str = None) -> Dict[str, Any]:
        """审核摘要质量
        
        Args:
            original_text: 原文
            summary: 摘要
            note_id: 笔记ID（可选）
            
        Returns:
            审核结果字典，包含是否需要重写、缺失信息等
        """
        if not self.enabled:
            return {
                'needs_rewrite': False,
                'audit_enabled': False,
                'note_id': note_id
            }
        
        self.audit_stats['total_audited'] += 1
        
        # 阶段一：快速审查
        stage1_result = self._stage1_entity_comparison(original_text, summary)
        
        audit_result = {
            'note_id': note_id,
            'needs_rewrite': False,
            'stage1_result': stage1_result,
            'stage2_result': None,
            'missing_entities': stage1_result.get('missing_entities', []),
            'missing_ratio': stage1_result.get('missing_ratio', 0.0),
            'audit_timestamp': datetime.now().isoformat()
        }
        
        # 如果阶段一发现问题，进入阶段二
        if stage1_result.get('has_missing_info', False):
            self.audit_stats['stage1_flagged'] += 1
            
            # 根据配置的比例决定是否进行LLM检查
            if random.random() < self.llm_check_ratio:
                self.audit_stats['stage2_checked'] += 1
                stage2_result = self._stage2_llm_judgment(original_text, summary, stage1_result)
                audit_result['stage2_result'] = stage2_result
                
                if stage2_result.get('needs_rewrite', False):
                    audit_result['needs_rewrite'] = True
                    audit_result['rewrite_reason'] = stage2_result.get('reason', '')
                    self.audit_stats['rewrite_recommended'] += 1
        
        # 记录失败或需要重写的案例
        if audit_result['needs_rewrite'] or stage1_result.get('has_missing_info', False):
            self._log_audit_failure(original_text, summary, audit_result)
        
        return audit_result
    
    def _stage1_entity_comparison(self, original_text: str, summary: str) -> Dict[str, Any]:
        """阶段一：基于实体识别的快速审查"""
        try:
            # 提取原文和摘要中的实体
            original_entities = self.ner.extract_entities(original_text, filter_non_persons=False)
            summary_entities = self.ner.extract_entities(summary, filter_non_persons=False)
            
            # 实体归一化
            original_normalized = self.ner.normalize_entities(original_entities, self.entity_similarity_threshold)
            summary_normalized = self.ner.normalize_entities(summary_entities, self.entity_similarity_threshold)
            
            # 获取代表性实体
            original_repr = set(original_normalized.keys())
            summary_repr = set(summary_normalized.keys())
            
            # 计算缺失的实体
            missing_entities = original_repr - summary_repr
            missing_ratio = len(missing_entities) / len(original_repr) if original_repr else 0.0
            
            # 分类实体类型
            entity_analysis = self._analyze_entity_types(original_entities, summary_entities)
            
            has_missing_info = missing_ratio > self.missing_entity_threshold
            
            return {
                'original_entities': list(original_repr),
                'summary_entities': list(summary_repr),
                'missing_entities': list(missing_entities),
                'missing_ratio': missing_ratio,
                'has_missing_info': has_missing_info,
                'entity_analysis': entity_analysis,
                'extraction_method': 'ner_comparison'
            }
            
        except Exception as e:
            logger.error(f"Stage 1 entity comparison failed: {e}")
            return {
                'error': str(e),
                'has_missing_info': False,
                'missing_entities': [],
                'missing_ratio': 0.0
            }
    
    def _analyze_entity_types(self, original_entities: List[str], summary_entities: List[str]) -> Dict[str, Any]:
        """分析实体类型分布"""
        def categorize_entities(entities):
            categories = {
                'persons': [],
                'organizations': [],
                'works': [],
                'dates': [],
                'others': []
            }
            
            for entity in entities:
                if self.ner._is_likely_person(entity):
                    categories['persons'].append(entity)
                elif any(indicator in entity.lower() for indicator in ['company', 'corporation', 'organization']):
                    categories['organizations'].append(entity)
                elif any(indicator in entity.lower() for indicator in ['album', 'book', 'movie', 'show']):
                    categories['works'].append(entity)
                elif any(char.isdigit() for char in entity):
                    categories['dates'].append(entity)
                else:
                    categories['others'].append(entity)
            
            return categories
        
        original_categories = categorize_entities(original_entities)
        summary_categories = categorize_entities(summary_entities)
        
        return {
            'original_categories': original_categories,
            'summary_categories': summary_categories
        }
    
    def _stage2_llm_judgment(self, original_text: str, summary: str, stage1_result: Dict[str, Any]) -> Dict[str, Any]:
        """阶段二：LLM精准判断"""
        try:
            missing_entities = stage1_result.get('missing_entities', [])
            
            # 构建LLM提示词
            prompt = self._build_llm_audit_prompt(original_text, summary, missing_entities)
            system_prompt = self._get_llm_audit_system_prompt()
            
            # 调用LLM
            response = self.llm.generate(prompt, system_prompt)
            
            # 解析LLM响应
            llm_result = self._parse_llm_audit_response(response)
            
            return {
                'llm_response': response,
                'needs_rewrite': llm_result.get('needs_rewrite', False),
                'reason': llm_result.get('reason', ''),
                'missing_info_type': llm_result.get('missing_info_type', ''),
                'confidence': llm_result.get('confidence', 0.0),
                'extraction_method': 'llm_judgment'
            }
            
        except Exception as e:
            logger.error(f"Stage 2 LLM judgment failed: {e}")
            return {
                'error': str(e),
                'needs_rewrite': False,
                'reason': 'LLM judgment failed'
            }
    
    def _build_llm_audit_prompt(self, original_text: str, summary: str, missing_entities: List[str]) -> str:
        """构建LLM审核提示词"""
        prompt = f"""请审核以下摘要是否遗漏了重要信息：

原文：
{original_text}

摘要：
{summary}

实体分析发现可能缺失的实体：{', '.join(missing_entities) if missing_entities else '无'}

请判断：
1. 摘要是否遗漏了重要的人物关系、协作关系或关键细节？
2. 缺失的信息是否影响对原文的理解？
3. 是否建议重写摘要？

请以JSON格式回答：
{{
    "needs_rewrite": true/false,
    "reason": "具体原因",
    "missing_info_type": "缺失信息类型（如：人物关系、时间信息、协作关系等）",
    "confidence": 0.0-1.0
}}"""
        return prompt
    
    def _get_llm_audit_system_prompt(self) -> str:
        """获取LLM审核系统提示词"""
        return """你是一个专业的文本摘要质量审核员。你的任务是判断摘要是否遗漏了原文中的重要信息，特别关注：
1. 人物关系和协作关系
2. 重要的时间、地点信息
3. 关键的事实细节
4. 作品名称和创作信息

请基于原文内容进行客观判断，避免过度严格或过度宽松。只有当缺失的信息确实影响对原文核心内容的理解时，才建议重写。"""
    
    def _parse_llm_audit_response(self, response: str) -> Dict[str, Any]:
        """解析LLM审核响应"""
        try:
            # 尝试提取JSON
            import re
            json_match = re.search(r'\{[^}]*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)
                return result
            else:
                # 如果没有找到JSON，尝试从文本中提取信息
                needs_rewrite = 'true' in response.lower() or '需要重写' in response or '建议重写' in response
                return {
                    'needs_rewrite': needs_rewrite,
                    'reason': response[:200],
                    'confidence': 0.5
                }
        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return {
                'needs_rewrite': False,
                'reason': 'Failed to parse LLM response',
                'confidence': 0.0
            }
    
    def _log_audit_failure(self, original_text: str, summary: str, audit_result: Dict[str, Any]):
        """记录审核失败或需要重写的案例"""
        try:
            # 创建日志目录
            today = datetime.now().strftime('%Y%m%d')
            log_dir = os.path.join(config.get('storage.result_root', 'result'), f'summary_audit_log_{today}')
            os.makedirs(log_dir, exist_ok=True)
            
            # 准备日志记录
            log_record = {
                'timestamp': audit_result.get('audit_timestamp'),
                'note_id': audit_result.get('note_id'),
                'original_text': original_text,
                'summary': summary,
                'audit_result': audit_result,
                'needs_rewrite': audit_result.get('needs_rewrite', False)
            }
            
            # 写入日志文件
            log_file = os.path.join(log_dir, 'audit_failures.jsonl')
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_record, ensure_ascii=False) + '\n')
                
        except Exception as e:
            logger.error(f"Failed to log audit failure: {e}")
    
    def batch_audit_summaries(self, notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量审核摘要"""
        if not self.enabled:
            logger.info("Summary auditor is disabled")
            return notes
        
        logger.info(f"Starting batch audit of {len(notes)} summaries")
        
        audited_notes = []
        for note in notes:
            try:
                original_text = note.get('original_text', '')
                content = note.get('content', '')
                note_id = note.get('note_id', '')
                
                if not original_text or not content:
                    logger.warning(f"Missing text or content for note {note_id}")
                    audited_notes.append(note)
                    continue
                
                # 执行审核
                audit_result = self.audit_summary(original_text, content, note_id)
                
                # 将审核结果添加到笔记中
                note_copy = note.copy()
                note_copy['audit_result'] = audit_result
                
                audited_notes.append(note_copy)
                
            except Exception as e:
                logger.error(f"Failed to audit note {note.get('note_id', 'unknown')}: {e}")
                audited_notes.append(note)
        
        # 记录统计信息
        self._log_audit_statistics()
        
        return audited_notes
    
    def _log_audit_statistics(self):
        """记录审核统计信息"""
        stats = self.audit_stats
        logger.info(f"Summary audit statistics:")
        logger.info(f"  Total audited: {stats['total_audited']}")
        
        if stats['total_audited'] > 0:
            logger.info(f"  Stage 1 flagged: {stats['stage1_flagged']} ({stats['stage1_flagged']/stats['total_audited']*100:.1f}%)")
            logger.info(f"  Stage 2 checked: {stats['stage2_checked']} ({stats['stage2_checked']/stats['total_audited']*100:.1f}%)")
            logger.info(f"  Rewrite recommended: {stats['rewrite_recommended']} ({stats['rewrite_recommended']/stats['total_audited']*100:.1f}%)")
        else:
            logger.info("  Stage 1 flagged: 0 (0.0%)")
            logger.info("  Stage 2 checked: 0 (0.0%)")
            logger.info("  Rewrite recommended: 0 (0.0%)")
    
    def save_flagged_summaries(self, notes: List[Dict[str, Any]], output_dir: str):
        """保存被标记为需要重写的摘要"""
        try:
            flagged_notes = []
            for note in notes:
                audit_result = note.get('audit_result', {})
                if audit_result.get('needs_rewrite', False):
                    flagged_notes.append(note)
            
            if flagged_notes:
                today = datetime.now().strftime('%Y%m%d')
                flagged_dir = os.path.join(output_dir, f'grag_{today}')
                os.makedirs(flagged_dir, exist_ok=True)
                
                flagged_file = os.path.join(flagged_dir, 'flagged_summaries.json')
                with open(flagged_file, 'w', encoding='utf-8') as f:
                    json.dump(flagged_notes, f, ensure_ascii=False, indent=2)
                
                logger.info(f"Saved {len(flagged_notes)} flagged summaries to {flagged_file}")
            
        except Exception as e:
            logger.error(f"Failed to save flagged summaries: {e}")