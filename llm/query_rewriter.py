import json
from typing import List, Dict, Any, Optional
from loguru import logger
from .local_llm import LocalLLM
from config import config
from .prompts import (
    QUERY_ANALYSIS_SYSTEM_PROMPT,
    QUERY_ANALYSIS_PROMPT,
    SPLIT_QUERY_SYSTEM_PROMPT,
    SPLIT_QUERY_PROMPT,
    OPTIMIZE_QUERY_SYSTEM_PROMPT,
    OPTIMIZE_QUERY_PROMPT,
    ENHANCE_QUERY_SYSTEM_PROMPT,
    ENHANCE_QUERY_PROMPT,
)

class QueryRewriter:
    """查询重写器，用于优化和拆分用户查询"""
    
    def __init__(self, llm: LocalLLM = None):
        self.llm = llm or LocalLLM()
        self.enable_rewrite = config.get('query.rewrite_enabled', True)
        self.split_multi_queries = config.get('query.split_multi_queries', True)
        self.add_prior_knowledge = config.get('query.add_prior_knowledge', False)
        
    def rewrite_query(self, query: str) -> Dict[str, Any]:
        """重写查询"""
        if not self.enable_rewrite:
            return {
                'original_query': query,
                'rewritten_queries': [query],
                'query_type': 'simple',
                'enhancements': []
            }
        
        logger.info(f"Rewriting query: {query[:100]}...")
        
        # 分析查询类型和复杂度
        query_analysis = self._analyze_query(query)
        
        # 根据分析结果进行重写
        if query_analysis['is_multi_question'] and self.split_multi_queries:
            rewritten_queries = self._split_multi_queries(query)
        else:
            rewritten_queries = self._optimize_single_query(query)
        
        # 添加先验知识（如果启用）
        if self.add_prior_knowledge:
            rewritten_queries = self._enhance_with_prior_knowledge(rewritten_queries)
        
        result = {
            'original_query': query,
            'rewritten_queries': rewritten_queries,
            'query_type': query_analysis['query_type'],
            'complexity': query_analysis['complexity'],
            'enhancements': query_analysis.get('enhancements', []),
            'is_multi_question': query_analysis['is_multi_question']
        }
        
        logger.info(f"Query rewritten into {len(rewritten_queries)} queries")
        return result
    
    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """分析查询的类型和复杂度"""
        system_prompt = QUERY_ANALYSIS_SYSTEM_PROMPT
        
        prompt = QUERY_ANALYSIS_PROMPT.format(query=query)
        
        try:
            response = self.llm.generate(prompt, system_prompt)
            # 清理响应，移除可能的markdown代码块标记
            cleaned_response = self._clean_json_response(response)
            analysis = json.loads(cleaned_response)
            
            # 验证和补充默认值
            return {
                'query_type': analysis.get('query_type', 'factual'),
                'complexity': analysis.get('complexity', 'medium'),
                'is_multi_question': analysis.get('is_multi_question', False),
                'key_concepts': analysis.get('key_concepts', []),
                'intent': analysis.get('intent', ''),
                'domain': analysis.get('domain', 'general'),
                'enhancements': []
            }
        except Exception as e:
            logger.warning(f"Query analysis failed: {e}")
            return {
                'query_type': 'factual',
                'complexity': 'medium',
                'is_multi_question': '？' in query or '，' in query or 'and' in query.lower(),
                'key_concepts': [],
                'intent': '',
                'domain': 'general',
                'enhancements': []
            }
    
    def _split_multi_queries(self, query: str) -> List[str]:
        """拆分多个查询"""
        system_prompt = SPLIT_QUERY_SYSTEM_PROMPT
        
        prompt = SPLIT_QUERY_PROMPT.format(query=query)
        
        try:
            response = self.llm.generate(prompt, system_prompt)
            cleaned_response = self._clean_json_response(response)
            result = json.loads(cleaned_response)
            sub_queries = result.get('sub_queries', [query])
            
            # 验证拆分结果
            if not sub_queries or len(sub_queries) == 0:
                return [query]
            
            return sub_queries
            
        except Exception as e:
            logger.warning(f"Query splitting failed: {e}")
            # 简单的拆分逻辑作为备用
            return self._simple_split_query(query)
    
    def _simple_split_query(self, query: str) -> List[str]:
        """简单的查询拆分逻辑"""
        # 基于标点符号的简单拆分
        import re
        
        # 按照问号、句号、分号拆分
        parts = re.split(r'[？?。；;]', query)
        
        # 清理和过滤
        sub_queries = []
        for part in parts:
            part = part.strip()
            if part and len(part) > 3:
                # 如果不是完整的问句，添加适当的疑问词
                if not any(word in part for word in ['什么', '如何', '为什么', '哪里', '什么时候', 'what', 'how', 'why', 'where', 'when']):
                    part = f"关于{part}的信息"
                sub_queries.append(part)
        
        return sub_queries if sub_queries else [query]
    
    def _optimize_single_query(self, query: str) -> List[str]:
        """优化单个查询"""
        system_prompt = OPTIMIZE_QUERY_SYSTEM_PROMPT
        
        prompt = OPTIMIZE_QUERY_PROMPT.format(query=query)
        
        try:
            response = self.llm.generate(prompt, system_prompt)
            cleaned_response = self._clean_json_response(response)
            result = json.loads(cleaned_response)
            optimized_queries = result.get('optimized_queries', [query])
            
            # 确保包含原始查询
            if query not in optimized_queries:
                optimized_queries.insert(0, query)
            
            return optimized_queries
            
        except Exception as e:
            logger.warning(f"Query optimization failed: {e}")
            return [query]
    
    def _enhance_with_prior_knowledge(self, queries: List[str]) -> List[str]:
        """使用先验知识增强查询（需要防止幻觉）"""
        if not self.add_prior_knowledge:
            return queries
        
        enhanced_queries = []
        
        for query in queries:
            system_prompt = ENHANCE_QUERY_SYSTEM_PROMPT

            prompt = ENHANCE_QUERY_PROMPT.format(query=query)
            
            try:
                response = self.llm.generate(prompt, system_prompt)
                cleaned_response = self._clean_json_response(response)
                result = json.loads(cleaned_response)
                
                confidence = result.get('confidence', 0.0)
                
                # 只有在高置信度时才使用增强查询
                if confidence > 0.7:
                    enhanced_query = result.get('enhanced_query', query)
                    enhanced_queries.append(enhanced_query)
                else:
                    enhanced_queries.append(query)
                    
            except Exception as e:
                logger.warning(f"Query enhancement failed: {e}")
                enhanced_queries.append(query)
        
        return enhanced_queries
    
    def batch_rewrite_queries(self, queries: List[str]) -> List[Dict[str, Any]]:
        """批量重写查询"""
        results = []
        
        for query in queries:
            try:
                result = self.rewrite_query(query)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to rewrite query '{query}': {e}")
                # 返回原始查询作为备用
                results.append({
                    'original_query': query,
                    'rewritten_queries': [query],
                    'query_type': 'unknown',
                    'enhancements': []
                })
        
        return results
    
    def validate_rewritten_queries(self, rewrite_result: Dict[str, Any]) -> Dict[str, Any]:
        """验证重写结果的质量"""
        original_query = rewrite_result['original_query']
        rewritten_queries = rewrite_result['rewritten_queries']
        
        # 基本验证
        valid_queries = []
        for query in rewritten_queries:
            if isinstance(query, str) and len(query.strip()) > 3:
                valid_queries.append(query.strip())
        
        # 如果没有有效的重写查询，使用原始查询
        if not valid_queries:
            valid_queries = [original_query]
        
        rewrite_result['rewritten_queries'] = valid_queries
        rewrite_result['validation_passed'] = len(valid_queries) > 0
        
        return rewrite_result
    
    def _clean_json_response(self, response: str) -> str:
        """清理LLM响应，移除markdown代码块标记和其他格式"""
        if not response:
            return "{}"
        
        # 清理控制字符
        response = self._clean_control_characters(response)
        
        # 移除markdown代码块标记
        response = response.strip()
        if response.startswith('```json'):
            response = response[7:]
        elif response.startswith('```'):
            response = response[3:]
        
        if response.endswith('```'):
            response = response[:-3]
        
        # 移除可能的前后空白和换行
        response = response.strip()
        
        # 尝试提取JSON对象
        import re
        # 查找第一个完整的JSON对象
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response)
        if json_match:
            response = json_match.group(0)
        
        # 如果响应为空或不是JSON格式，返回空对象
        if not response or not (response.startswith('{') or response.startswith('[')):
            return "{}"
        
        return response
    
    def _clean_control_characters(self, text: str) -> str:
        """清理字符串中的无效控制字符"""
        import re
        
        # 移除或替换无效的控制字符，但保留有效的空白字符（空格、制表符、换行符）
        # 保留 \t (\x09), \n (\x0A), \r (\x0D) 和普通空格 (\x20)
        cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # 替换一些常见的问题字符
        cleaned = cleaned.replace('\u0000', '')  # NULL字符
        cleaned = cleaned.replace('\u0001', '')  # SOH字符
        cleaned = cleaned.replace('\u0002', '')  # STX字符
        
        return cleaned
