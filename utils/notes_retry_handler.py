"""
原子笔记重试与回退处理器
实现解析失败时的重试机制和规则回退逻辑
"""

from typing import List, Dict, Any, Optional, Callable
from loguru import logger
from config import config


class NotesRetryHandler:
    """原子笔记重试与回退处理器"""
    
    def __init__(self):
        """初始化重试处理器"""
        # 从配置加载参数
        self.retry_enabled = config.get('notes_llm.retry_once_on_parse_error', True)
        self.shorten_on_retry_chars = config.get('notes_llm.shorten_on_retry_chars', 1000)
        self.enable_rule_fallback = config.get('notes_llm.enable_rule_fallback', True)
        
        # 统计信息
        self.stats = {
            'total_attempts': 0,
            'first_attempt_success': 0,
            'retry_attempts': 0,
            'retry_success': 0,
            'fallback_used': 0,
            'final_failures': 0
        }
    
    def process_chunk_with_retry(
        self,
        chunk_data: Dict[str, Any],
        llm_generate_func: Callable[[str, str], str],
        parse_func: Callable[[str], Optional[List[Dict[str, Any]]]],
        system_prompt: str,
        user_prompt: str
    ) -> tuple[Optional[List[Dict[str, Any]]], Dict[str, Any]]:
        """
        处理单个chunk，包含重试和回退逻辑
        
        Args:
            chunk_data: chunk数据
            llm_generate_func: LLM生成函数
            parse_func: 解析函数
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            
        Returns:
            tuple: (解析结果, 处理元数据)
        """
        self.stats['total_attempts'] += 1
        
        # 第一次尝试
        logger.debug("First attempt to generate notes")
        response = llm_generate_func(user_prompt, system_prompt)
        parsed_result = parse_func(response)
        
        metadata = {
            'attempts': 1,
            'used_retry': False,
            'used_fallback': False,
            'original_chunk_length': len(chunk_data.get('text', '')),
            'final_chunk_length': len(chunk_data.get('text', ''))
        }
        
        if parsed_result is not None:
            # 第一次尝试成功
            self.stats['first_attempt_success'] += 1
            logger.debug("First attempt successful")
            return parsed_result, metadata
        
        # 第一次尝试失败，检查是否启用重试
        if not self.retry_enabled:
            logger.debug("Retry disabled, proceeding to fallback")
            return self._handle_fallback(chunk_data, metadata)
        
        # 重试逻辑
        logger.debug("First attempt failed, starting retry")
        self.stats['retry_attempts'] += 1
        metadata['used_retry'] = True
        metadata['attempts'] = 2
        
        # 缩短chunk文本
        shortened_chunk = self._shorten_chunk(chunk_data)
        metadata['final_chunk_length'] = len(shortened_chunk.get('text', ''))
        
        # 生成新的用户提示词
        shortened_prompt = user_prompt.replace(chunk_data.get('text', ''), shortened_chunk.get('text', ''))
        
        # 第二次尝试
        retry_response = llm_generate_func(shortened_prompt, system_prompt)
        retry_parsed = parse_func(retry_response)
        
        if retry_parsed is not None:
            # 重试成功
            self.stats['retry_success'] += 1
            logger.debug("Retry attempt successful")
            return retry_parsed, metadata
        
        # 重试也失败，进入回退逻辑
        logger.debug("Retry attempt failed, proceeding to fallback")
        return self._handle_fallback(shortened_chunk, metadata)
    
    def _shorten_chunk(self, chunk_data: Dict[str, Any]) -> Dict[str, Any]:
        """缩短chunk文本到指定长度"""
        text = chunk_data.get('text', '')
        
        if len(text) <= self.shorten_on_retry_chars:
            return chunk_data
        
        # 缩短到指定字符数，尽量在句子边界截断
        shortened_text = text[:self.shorten_on_retry_chars]
        
        # 尝试在句号处截断
        last_period = shortened_text.rfind('.')
        if last_period > self.shorten_on_retry_chars * 0.7:  # 至少保留70%的内容
            shortened_text = shortened_text[:last_period + 1]
        
        # 创建新的chunk_data
        shortened_chunk = chunk_data.copy()
        shortened_chunk['text'] = shortened_text
        
        logger.debug(f"Shortened chunk from {len(text)} to {len(shortened_text)} characters")
        return shortened_chunk
    
    def _handle_fallback(self, chunk_data: Dict[str, Any], metadata: Dict[str, Any]) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """处理回退逻辑"""
        if not self.enable_rule_fallback:
            # 回退被禁用，返回空结果
            self.stats['final_failures'] += 1
            logger.warning("Rule fallback disabled, returning empty result")
            return [], metadata
        
        # 使用规则回退生成基本笔记
        self.stats['fallback_used'] += 1
        metadata['used_fallback'] = True
        
        logger.debug("Using rule fallback to generate basic note")
        fallback_note = self._generate_rule_fallback_note(chunk_data)
        
        return [fallback_note], metadata
    
    def _generate_rule_fallback_note(self, chunk_data: Dict[str, Any]) -> Dict[str, Any]:
        """使用规则回退生成基本笔记"""
        text = chunk_data.get('text', '')
        
        # 基本的规则提取
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        sent_count = min(len(sentences), 3)
        
        # 简单的关键词提取（取最长的几个词）
        words = text.split()
        keywords = sorted([w for w in words if len(w) > 4], key=len, reverse=True)[:3]
        
        # 简单的实体提取（大写开头的词）
        entities = list(set([w for w in words if w[0].isupper() and len(w) > 2]))[:5]
        
        # 计算基本的显著性分数
        salience = min(0.6, len(text) / 1000)  # 基于长度的简单评分
        
        fallback_note = {
            'text': text[:200] + ('...' if len(text) > 200 else ''),  # 截断到200字符
            'sent_count': sent_count,
            'salience': salience,
            'local_spans': [],
            'entities': entities,
            'years': [],
            'quality_flags': ['OK', 'RULE_FALLBACK']  # 标记为规则回退生成
        }
        
        logger.debug(f"Generated rule fallback note with salience {salience}")
        return fallback_note
    
    def get_stats(self) -> Dict[str, int]:
        """获取统计信息"""
        stats = self.stats.copy()
        
        # 计算成功率
        if stats['total_attempts'] > 0:
            stats['first_attempt_success_rate'] = stats['first_attempt_success'] / stats['total_attempts']
            stats['overall_success_rate'] = (stats['first_attempt_success'] + stats['retry_success']) / stats['total_attempts']
        else:
            stats['first_attempt_success_rate'] = 0.0
            stats['overall_success_rate'] = 0.0
        
        if stats['retry_attempts'] > 0:
            stats['retry_success_rate'] = stats['retry_success'] / stats['retry_attempts']
        else:
            stats['retry_success_rate'] = 0.0
        
        return stats
    
    def reset_stats(self):
        """重置统计信息"""
        for key in self.stats:
            self.stats[key] = 0


def process_with_retry(
    chunk_data: Dict[str, Any],
    llm_generate_func: Callable[[str, str], str],
    parse_func: Callable[[str], Optional[List[Dict[str, Any]]]],
    system_prompt: str,
    user_prompt: str
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    便捷函数：处理单个chunk的重试逻辑
    
    Args:
        chunk_data: chunk数据
        llm_generate_func: LLM生成函数
        parse_func: 解析函数
        system_prompt: 系统提示词
        user_prompt: 用户提示词
        
    Returns:
        tuple: (解析结果, 处理元数据)
    """
    handler = NotesRetryHandler()
    result, metadata = handler.process_chunk_with_retry(
        chunk_data, llm_generate_func, parse_func, system_prompt, user_prompt
    )
    
    # 确保返回的是列表
    if result is None:
        result = []
    
    return result, metadata