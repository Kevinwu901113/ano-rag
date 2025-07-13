from typing import List, Dict, Any, Optional
from loguru import logger
from .local_llm import LocalLLM
from utils import BatchProcessor, TextUtils, extract_json_from_response, clean_control_characters
from config import config
from .prompts import (
    ATOMIC_NOTEGEN_SYSTEM_PROMPT,
    ATOMIC_NOTEGEN_PROMPT,
)

class AtomicNoteGenerator:
    """原子笔记生成器，专门用于文档处理阶段的原子笔记构建"""
    
    def __init__(self, llm: LocalLLM = None):
        self.llm = llm or LocalLLM()
        self.batch_processor = BatchProcessor(
            batch_size=config.get('document.batch_size', 32),
            use_gpu=config.get('performance.use_gpu', True)
        )
        
    def generate_atomic_notes(self, text_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """从文本块生成原子笔记"""
        logger.info(f"Generating atomic notes for {len(text_chunks)} text chunks")
        
        # 准备提示词模板
        system_prompt = self._get_atomic_note_system_prompt()
        
        def process_batch(batch):
            results = []
            for chunk_data in batch:
                try:
                    note = self._generate_single_atomic_note(chunk_data, system_prompt)
                    results.append(note)
                except Exception as e:
                    logger.error(f"Failed to generate atomic note: {e}")
                    # 创建基本的原子笔记
                    results.append(self._create_fallback_note(chunk_data))
            return results
        
        atomic_notes = self.batch_processor.process_batches(
            text_chunks,
            process_batch,
            desc="Generating atomic notes"
        )
        
        # 后处理：添加ID和元数据
        for i, note in enumerate(atomic_notes):
            note['note_id'] = f"note_{i:06d}"
            note['created_at'] = self._get_timestamp()
        
        logger.info(f"Generated {len(atomic_notes)} atomic notes")
        return atomic_notes
    
    def _generate_single_atomic_note(self, chunk_data: Dict[str, Any], system_prompt: str) -> Dict[str, Any]:
        """生成单个原子笔记"""
        text = chunk_data.get('text', '')
        
        prompt = ATOMIC_NOTEGEN_PROMPT.format(text=text)
        
        response = self.llm.generate(prompt, system_prompt)
        
        try:
            import json
            import re
            
            # 清理响应，提取JSON部分
            cleaned_response = extract_json_from_response(response)
            
            if not cleaned_response:
                logger.warning(f"No valid JSON found in response: {response[:200]}...")
                return self._create_fallback_note(chunk_data)
            
            note_data = json.loads(cleaned_response)
            
            # 提取相关的paragraph idx信息
            paragraph_idx_mapping = chunk_data.get('paragraph_idx_mapping', {})
            relevant_idxs = self._extract_relevant_paragraph_idxs(text, paragraph_idx_mapping)
            
            # 验证和清理数据
            atomic_note = {
                'original_text': text,
                'content': note_data.get('content', text),
                'summary': note_data.get('summary', ''),
                'keywords': self._clean_list(note_data.get('keywords', [])),
                'entities': self._clean_list(note_data.get('entities', [])),
                'concepts': self._clean_list(note_data.get('concepts', [])),
                'importance_score': float(note_data.get('importance_score', 0.5)),
                'note_type': note_data.get('note_type', 'fact'),
                'source_info': chunk_data.get('source_info', {}),
                'chunk_index': chunk_data.get('chunk_index', 0),
                'length': len(text),
                'paragraph_idxs': relevant_idxs  # 添加相关的paragraph idx信息
            }
            
            # 提取额外的实体（如果LLM没有提取到）
            if not atomic_note['entities']:
                atomic_note['entities'] = TextUtils.extract_entities(text)

            # 确保包含主要实体
            primary_entity = chunk_data.get('primary_entity')
            if primary_entity and primary_entity not in atomic_note['entities']:
                atomic_note['entities'].insert(0, primary_entity)

            return atomic_note
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse JSON response: {e}. Response: {response[:200]}...")
            return self._create_fallback_note(chunk_data)
    
    
    def _try_fix_truncated_json(self, response: str) -> str:
        """尝试修复截断的JSON响应"""
        import re
        import json
        
        # 清理响应
        cleaned = clean_control_characters(response.strip())
        
        # 移除markdown标记
        cleaned = re.sub(r'```(?:json)?\s*', '', cleaned)
        cleaned = re.sub(r'```\s*$', '', cleaned)
        
        # 查找JSON开始位置
        start_idx = cleaned.find('{')
        if start_idx == -1:
            return ""
        
        # 提取从开始到最后的内容
        json_part = cleaned[start_idx:]
        
        # 如果JSON看起来被截断了（以...结尾或没有闭合括号）
        if json_part.endswith('...') or json_part.count('{') > json_part.count('}'):
            # 尝试构建一个最小的有效JSON
            try:
                # 移除...结尾
                if json_part.endswith('...'):
                    json_part = json_part[:-3]
                
                # 尝试找到content字段的值
                content_match = re.search(r'"content"\s*:\s*"([^"]*)', json_part)
                if content_match:
                    content = content_match.group(1)
                    # 构建最小的有效JSON
                    minimal_json = {
                        "content": content,
                        "summary": content[:100] if len(content) > 100 else content,
                        "keywords": [],
                        "entities": [],
                        "concepts": [],
                        "importance_score": 0.5,
                        "note_type": "fact"
                    }
                    return json.dumps(minimal_json, ensure_ascii=False)
            except Exception:
                pass
        
        return ""
    
    def _create_fallback_note(self, chunk_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建备用的原子笔记（当LLM生成失败时）"""
        text = chunk_data.get('text', '')
        primary_entity = chunk_data.get('primary_entity')
        entities = TextUtils.extract_entities(text)
        if not entities and primary_entity:
            entities = [primary_entity]

        return {
            'original_text': text,
            'content': text,
            'summary': text[:100] + '...' if len(text) > 100 else text,
            'keywords': [],
            'entities': entities,
            'concepts': [],
            'importance_score': 0.5,
            'note_type': 'fact',
            'source_info': chunk_data.get('source_info', {}),
            'chunk_index': chunk_data.get('chunk_index', 0),
            'length': len(text)
        }
    
    def _get_atomic_note_system_prompt(self) -> str:
        """获取原子笔记生成的系统提示词"""
        return ATOMIC_NOTEGEN_SYSTEM_PROMPT
    
    def _clean_list(self, items: List[str]) -> List[str]:
        """清理列表，去除空值和重复项"""
        if not isinstance(items, list):
            return []
        
        cleaned = []
        for item in items:
            if isinstance(item, str) and item.strip():
                cleaned_item = item.strip()
                if cleaned_item not in cleaned:
                    cleaned.append(cleaned_item)
        
        return cleaned
    
    def _extract_relevant_paragraph_idxs(self, text: str, paragraph_idx_mapping: Dict[str, int]) -> List[int]:
        """从文本中提取相关的paragraph idx"""
        relevant_idxs = []
        
        if not paragraph_idx_mapping:
            return relevant_idxs
        
        # 对于每个段落文本，检查是否在当前chunk的文本中
        for paragraph_text, idx in paragraph_idx_mapping.items():
            # 多种匹配策略
            match_found = False
            
            # 1. 直接文本包含检查
            if paragraph_text in text:
                match_found = True
            
            # 2. 检查段落的前100个字符是否在文本中
            elif len(paragraph_text) > 100:
                prefix = paragraph_text[:100]
                if prefix in text:
                    match_found = True
            
            # 3. 按句子分割检查（针对长段落）
            if not match_found:
                sentences = [s.strip() for s in paragraph_text.split('.') if len(s.strip()) > 30]
                for sentence in sentences[:3]:  # 只检查前3个句子
                    if sentence in text:
                        match_found = True
                        break
            
            # 4. 关键词匹配（提取段落中的关键词）
            if not match_found and len(paragraph_text) > 50:
                # 简单的关键词提取：长度大于5的单词
                words = [w.strip('.,!?;:"()[]{}') for w in paragraph_text.split() if len(w) > 5]
                if len(words) >= 3:
                    # 如果文本中包含段落的多个关键词，认为相关
                    word_matches = sum(1 for word in words[:10] if word.lower() in text.lower())
                    if word_matches >= min(3, len(words) // 2):
                        match_found = True
            
            if match_found:
                relevant_idxs.append(idx)
        
        return sorted(list(set(relevant_idxs)))  # 去重并排序
    
    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def enhance_notes_with_relations(self, atomic_notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """增强原子笔记，添加关系信息"""
        logger.info("Enhancing atomic notes with relations")
        
        for i, note in enumerate(atomic_notes):
            # 计算与其他笔记的相似度
            note['related_notes'] = []
            
            for j, other_note in enumerate(atomic_notes):
                if i != j:
                    similarity = TextUtils.calculate_similarity_keywords(
                        note['content'], other_note['content']
                    )
                    
                    if similarity > 0.3:  # 相似度阈值
                        note['related_notes'].append({
                            'note_id': other_note.get('note_id', f"note_{j:06d}"),
                            'similarity': similarity,
                            'relation_type': 'content_similarity'
                        })
            
            # 实体共现关系
            note['entity_relations'] = self._find_entity_relations(note, atomic_notes)
        
        return atomic_notes
    
    def _find_entity_relations(self, note: Dict[str, Any], all_notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """查找实体关系"""
        relations = []
        note_entities = set(note.get('entities', []))
        
        if not note_entities:
            return relations
        
        for other_note in all_notes:
            if other_note.get('note_id') == note.get('note_id'):
                continue
            
            other_entities = set(other_note.get('entities', []))
            common_entities = note_entities.intersection(other_entities)
            
            if common_entities:
                relations.append({
                    'target_note_id': other_note.get('note_id'),
                    'common_entities': list(common_entities),
                    'relation_type': 'entity_coexistence'
                })
        
        return relations
    
    def validate_atomic_notes(self, atomic_notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """验证原子笔记的质量"""
        valid_notes = []
        
        for note in atomic_notes:
            # 基本验证
            if not note.get('content') or len(note['content'].strip()) < 10:
                logger.warning(f"Skipping note with insufficient content: {note.get('note_id')}")
                continue
            
            # 重要性评分验证
            if note.get('importance_score', 0) < 0.1:
                logger.warning(f"Note has very low importance score: {note.get('note_id')}")
            
            valid_notes.append(note)
        
        logger.info(f"Validated {len(valid_notes)} out of {len(atomic_notes)} atomic notes")
        return valid_notes
