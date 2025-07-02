import json
from typing import List, Dict, Any, Optional
from loguru import logger
from .local_llm import LocalLLM
from config import config

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
        system_prompt = """
你是一个专业的查询分析专家。请分析用户查询的特点和类型。

分析维度：
1. query_type: factual/conceptual/procedural/comparative/analytical
2. complexity: simple/medium/complex
3. is_multi_question: 是否包含多个问题
4. key_concepts: 关键概念列表
5. intent: 用户意图
6. domain: 领域分类

请以JSON格式返回分析结果。
"""
        
        prompt = f"""
请分析以下查询：

查询：{query}

请返回JSON格式的分析结果：
"""
        
        try:
            response = self.llm.generate(prompt, system_prompt)
            analysis = json.loads(response)
            
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
        system_prompt = """
你是一个专业的查询拆分专家。请将包含多个问题的查询拆分为独立的子查询。

要求：
1. 每个子查询应该是独立完整的
2. 保持原始查询的语义
3. 确保子查询之间的逻辑关系
4. 避免信息丢失

请以JSON格式返回拆分结果：{"sub_queries": ["查询1", "查询2", ...]}
"""
        
        prompt = f"""
请将以下查询拆分为独立的子查询：

原始查询：{query}

请返回JSON格式的拆分结果：
"""
        
        try:
            response = self.llm.generate(prompt, system_prompt)
            result = json.loads(response)
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
        system_prompt = """
你是一个专业的查询优化专家。请优化用户查询以提高检索效果。

优化策略：
1. 明确查询意图
2. 补充关键信息
3. 使用更精确的术语
4. 保持查询的简洁性
5. 考虑同义词和相关概念

请以JSON格式返回优化结果：{"optimized_queries": ["优化查询1", "优化查询2", ...]}
"""
        
        prompt = f"""
请优化以下查询以提高检索效果：

原始查询：{query}

请返回JSON格式的优化结果：
"""
        
        try:
            response = self.llm.generate(prompt, system_prompt)
            result = json.loads(response)
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
            system_prompt = """
你是一个知识增强专家。请基于你的知识为查询添加相关的背景信息和上下文。

重要要求：
1. 只添加确定性很高的知识
2. 避免推测和假设
3. 如果不确定，请保持原查询不变
4. 标明添加的信息来源于常识

请以JSON格式返回：{"enhanced_query": "增强后的查询", "confidence": 0.8, "added_context": "添加的上下文"}
"""
            
            prompt = f"""
请为以下查询添加相关的背景知识（仅在确定的情况下）：

查询：{query}

请返回JSON格式的结果：
"""
            
            try:
                response = self.llm.generate(prompt, system_prompt)
                result = json.loads(response)
                
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