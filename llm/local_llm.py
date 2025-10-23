import torch
from typing import List, Dict, Any, Optional
from loguru import logger
from config import config
from utils.batch_processor import BatchProcessor
from utils.json_utils import extract_json_from_response
from .openai_client import OpenAIClient
from .lmstudio_client import LMStudioClient
from .prompts import (
    ATOMIC_NOTE_SYSTEM_PROMPT,
    ATOMIC_NOTE_SYSTEM_PROMPT_V2,
    ATOMIC_NOTE_PROMPT,
    EXTRACT_ENTITIES_SYSTEM_PROMPT,
    EXTRACT_ENTITIES_PROMPT,
)
from .streaming_early_stop import create_early_stop_stream

class LocalLLM:
    """统一LLM类，支持本地和在线模型，用于原子笔记生成和查询重写等任务
    
    支持的provider类型：
    - openai: OpenAI API或兼容的在线服务
    - lmstudio: LM Studio本地服务
    """

    def __init__(self, model_name: str = None, device: str = None, provider: str = None, api_key: str = None, base_url: str = None):
        # 获取provider配置
        self.provider = provider or config.get('llm.provider') or config.get('llm.local_model.provider', 'lmstudio')
        self.api_key = api_key
        self.base_url = base_url
        
        # 初始化客户端变量
        self.openai_client = None
        self.lmstudio_client = None
        self.batch_processor = BatchProcessor()
        
        # 不再支持混合模式
        self.is_hybrid_mode = False
        
        # 根据provider选择配置段和默认值
        if self.provider == 'openai':
            config_section = 'llm.openai'
            default_model = 'gpt-3.5-turbo'
        elif self.provider == 'lmstudio':
            config_section = 'llm.lmstudio'
            default_model = 'gpt-oss-20b'
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
        
        self.model_name = model_name or config.get('llm.model') or config.get(f'{config_section}.model', default_model)
        self.device = device or config.get('llm.local_model.device', 'auto')
        self.temperature = config.get('llm.local_model.temperature') or config.get(f'{config_section}.temperature', 0.1)
        self.max_tokens = config.get('llm.local_model.max_tokens') or config.get(f'{config_section}.max_tokens', 2048)
        
        self.is_openai_model = (self.provider == 'openai')
        self.is_lmstudio_model = (self.provider == 'lmstudio')
        
        logger.info(f"Initialized LocalLLM with provider: {self.provider}, model: {self.model_name}")
        self.load_model()

    def load_model(self):
        """加载模型"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            if self.is_openai_model:
                api_key = self.api_key or config.get('llm.openai.api_key')
                base_url = self.base_url or config.get('llm.openai.base_url')
                self.openai_client = OpenAIClient(api_key=api_key, model=self.model_name, base_url=base_url)
                if not self.openai_client.is_available():
                    raise Exception("Failed to connect to OpenAI API")
                logger.info("OpenAI model loaded and tested successfully")
            elif self.is_lmstudio_model:
                base_url = config.get('llm.lmstudio.base_url', f'http://localhost:{config.get("llm.lmstudio.port", 1234)}/v1')
                self.lmstudio_client = LMStudioClient(base_url=base_url, model=self.model_name)
                if not self.lmstudio_client.is_available():
                    raise Exception("Failed to connect to LM Studio API")
                logger.info("LM Studio model loaded and tested successfully")
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
            
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """生成文本"""
        try:
            if self.is_openai_model and self.openai_client:
                return self.openai_client.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=kwargs.get('temperature', self.temperature),
                    max_tokens=kwargs.get('max_tokens', self.max_tokens)
                )
            elif self.is_lmstudio_model and self.lmstudio_client:
                return self.lmstudio_client.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=kwargs.get('temperature', self.temperature),
                    max_tokens=kwargs.get('max_tokens', self.max_tokens)
                )
            else:
                raise ValueError(f"Unsupported provider for generation: {self.provider}")
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
