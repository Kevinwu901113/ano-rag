from typing import List, Dict, Any, Optional
from loguru import logger
from config import config
from utils.batch_processor import BatchProcessor
from utils.json_utils import extract_json_from_response
from .openai_client import OpenAIClient
from .prompts import (
    ATOMIC_NOTE_SYSTEM_PROMPT,
    ATOMIC_NOTE_PROMPT,
    EXTRACT_ENTITIES_SYSTEM_PROMPT,
    EXTRACT_ENTITIES_PROMPT,
)


class OnlineLLM:
    """在线LLM类，支持OpenAI API和其他兼容的在线API服务"""
    
    def __init__(self, provider: str = "openai", model_name: str = None, api_key: str = None, base_url: str = None):
        self.provider = provider.lower()
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        
        # 根据provider设置默认配置
        if self.provider == "openai":
            self.model_name = self.model_name or config.get('llm.openai.model', 'gpt-3.5-turbo')
            self.api_key = self.api_key or config.get('llm.openai.api_key')
            self.base_url = self.base_url or config.get('llm.openai.base_url')
            self.temperature = config.get('llm.openai.temperature', 0.7)
            self.max_tokens = config.get('llm.openai.max_tokens', 4096)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
        
        self.client = None
        self.batch_processor = BatchProcessor()
        
    def load_model(self):
        """初始化客户端"""
        try:
            logger.info(f"Initializing {self.provider} client with model: {self.model_name}")
            
            if self.provider == "openai":
                self.client = OpenAIClient(
                    api_key=self.api_key,
                    model=self.model_name,
                    base_url=self.base_url
                )
                
                # 测试连接
                if not self.client.is_available():
                    raise Exception(f"Failed to connect to {self.provider} API")
                
                logger.info(f"{self.provider} client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize {self.provider} client: {e}")
            raise
    
    def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """生成文本"""
        if self.client is None:
            self.load_model()
        
        try:
            return self.client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                **kwargs
            )
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return ""
    
    def batch_generate(self, prompts: List[str], system_prompt: str = None, **kwargs) -> List[str]:
        """批量生成文本"""
        def process_batch(batch_prompts):
            results = []
            for prompt in batch_prompts:
                result = self.generate(prompt, system_prompt, **kwargs)
                results.append(result)
            return results
        
        return self.batch_processor.process_batches(prompts, process_batch)
    
    def generate_atomic_notes(self, text_chunks: List[str]) -> List[Dict[str, Any]]:
        """生成原子笔记"""
        atomic_notes = []
        
        for chunk in text_chunks:
            try:
                prompt = ATOMIC_NOTE_PROMPT.format(chunk=chunk)
                response = self.generate(prompt, ATOMIC_NOTE_SYSTEM_PROMPT)
                
                # 提取JSON响应
                json_data = extract_json_from_response(response)
                if json_data:
                    atomic_notes.append({
                        'content': json_data.get('content', ''),
                        'keywords': json_data.get('keywords', []),
                        'entities': json_data.get('entities', []),
                        'concepts': json_data.get('concepts', []),
                        'importance_score': json_data.get('importance_score', 0.5),
                        'note_type': json_data.get('note_type', 'fact')
                    })
                else:
                    # 如果JSON解析失败，创建基本的原子笔记
                    atomic_notes.append({
                        'content': chunk,
                        'keywords': [],
                        'entities': [],
                        'concepts': [],
                        'importance_score': 0.5,
                        'note_type': 'fact'
                    })
                    
            except Exception as e:
                logger.error(f"Failed to generate atomic note for chunk: {e}")
                continue
        
        return atomic_notes
    
    def extract_entities_and_relations(self, text: str) -> Dict[str, Any]:
        """提取实体和关系"""
        try:
            prompt = EXTRACT_ENTITIES_PROMPT.format(text=text)
            response = self.generate(prompt, EXTRACT_ENTITIES_SYSTEM_PROMPT)
            
            # 提取JSON响应
            json_data = extract_json_from_response(response)
            if json_data:
                return json_data
            else:
                return {'entities': [], 'relations': []}
                
        except Exception as e:
            logger.error(f"Failed to extract entities and relations: {e}")
            return {'entities': [], 'relations': []}
    
    def is_available(self) -> bool:
        """检查客户端是否可用"""
        try:
            if self.client is None:
                # 创建临时客户端进行检查
                if self.provider == "openai":
                    temp_client = OpenAIClient(
                        api_key=self.api_key,
                        model=self.model_name,
                        base_url=self.base_url
                    )
                    return temp_client.is_available()
                return False
            return self.client.is_available()
        except Exception:
            return False
    
    def cleanup(self):
        """清理资源"""
        if self.client:
            # OpenAI客户端通常不需要特殊清理
            self.client = None
        logger.info(f"{self.provider} client cleaned up")
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'provider': self.provider,
            'model_name': self.model_name,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'base_url': self.base_url,
            'is_available': self.is_available()
        }
    
    def list_available_models(self) -> List[str]:
        """列出可用的模型"""
        try:
            if self.client is None:
                self.load_model()
            return self.client.list_models()
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []