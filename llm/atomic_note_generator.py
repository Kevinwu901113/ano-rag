from typing import List, Dict, Any, Optional
from loguru import logger
from .local_llm import LocalLLM
from utils import BatchProcessor, TextUtils
from config import config

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
        
        prompt = f"""
请将以下文本转换为原子笔记。每个原子笔记应该包含一个独立的知识点。

文本内容：
{text}

请按照以下JSON格式返回：
{{
    "content": "原子笔记的主要内容",
    "summary": "简要总结",
    "keywords": ["关键词1", "关键词2"],
    "entities": ["实体1", "实体2"],
    "concepts": ["概念1", "概念2"],
    "importance_score": 0.8,
    "note_type": "fact/concept/procedure/example"
}}
"""
        
        response = self.llm.generate(prompt, system_prompt)
        
        try:
            import json
            note_data = json.loads(response)
            
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
                'length': len(text)
            }
            
            # 提取额外的实体（如果LLM没有提取到）
            if not atomic_note['entities']:
                atomic_note['entities'] = TextUtils.extract_entities(text)
            
            return atomic_note
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            return self._create_fallback_note(chunk_data)
    
    def _create_fallback_note(self, chunk_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建备用的原子笔记（当LLM生成失败时）"""
        text = chunk_data.get('text', '')
        
        return {
            'original_text': text,
            'content': text,
            'summary': text[:100] + '...' if len(text) > 100 else text,
            'keywords': [],
            'entities': TextUtils.extract_entities(text),
            'concepts': [],
            'importance_score': 0.5,
            'note_type': 'fact',
            'source_info': chunk_data.get('source_info', {}),
            'chunk_index': chunk_data.get('chunk_index', 0),
            'length': len(text)
        }
    
    def _get_atomic_note_system_prompt(self) -> str:
        """获取原子笔记生成的系统提示词"""
        return """
你是一个专业的知识提取和整理专家。你的任务是将给定的文本转换为高质量的原子笔记。

原子笔记的特点：
1. 每个笔记包含一个独立、完整的知识点
2. 内容简洁明了，避免冗余
3. 保留关键信息和必要的上下文
4. 便于后续的检索和组合

提取要求：
1. content: 提取核心知识点，保持完整性和准确性
2. summary: 用一句话概括主要内容
3. keywords: 提取3-5个关键词，有助于检索
4. entities: 识别人名、地名、机构名、专业术语等
5. concepts: 识别重要概念和理论
6. importance_score: 评估内容重要性（0-1分）
7. note_type: 分类为fact（事实）、concept（概念）、procedure（流程）、example（示例）

请确保返回有效的JSON格式。
"""
    
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