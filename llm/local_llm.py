import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import List, Dict, Any, Optional
from loguru import logger
from config import config
from utils.batch_processor import BatchProcessor
from utils.json_utils import extract_json_from_response
from .ollama_client import OllamaClient
from .openai_client import OpenAIClient
from .lmstudio_client import LMStudioClient
from .multi_model_client import MultiModelClient
from .factory import LLMFactory
from .prompts import (
    ATOMIC_NOTE_SYSTEM_PROMPT,
    ATOMIC_NOTE_PROMPT,
    EXTRACT_ENTITIES_SYSTEM_PROMPT,
    EXTRACT_ENTITIES_PROMPT,
)

class LocalLLM:
    """统一LLM类，支持本地和在线模型，用于原子笔记生成和查询重写等任务
    
    支持的provider类型：
    - ollama: 本地Ollama服务
    - openai: OpenAI API或兼容的在线服务
    - lmstudio: LM Studio本地服务
    - transformers: 直接使用transformers库加载本地模型
    """
    
    def __init__(self, model_name: str = None, device: str = None, provider: str = None, api_key: str = None, base_url: str = None):
        # 获取provider配置，支持新的配置结构
        self.provider = provider or config.get('llm.provider') or config.get('llm.local_model.provider', 'ollama')
        
        # 支持在线服务的额外参数
        self.api_key = api_key
        self.base_url = base_url
        
        # 初始化客户端变量
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.ollama_client = None
        self.openai_client = None
        self.lmstudio_client = None
        self.multi_model_client = None
        self.batch_processor = BatchProcessor()
        
        # 根据provider选择配置段和默认值
        if self.provider == 'openai':
            config_section = 'llm.openai'
            default_model = 'gpt-3.5-turbo'
        elif self.provider == 'lmstudio':
            config_section = 'llm.lmstudio'
            default_model = 'gpt-oss-20b'
        else:  # ollama (默认)
            config_section = 'llm.ollama'
            default_model = 'gpt-oss:latest'
        
        # 从对应配置段获取参数，local_model的参数可以覆盖
        # 优先从 llm.model 读取，然后从特定 provider 配置段读取
        self.model_name = model_name or config.get('llm.model') or config.get(f'{config_section}.model', default_model)
        self.device = device or config.get('llm.local_model.device', 'auto')
        self.temperature = config.get('llm.local_model.temperature') or config.get(f'{config_section}.temperature', 0.1)
        self.max_tokens = config.get('llm.local_model.max_tokens') or config.get(f'{config_section}.max_tokens', 2048)
        
        # 根据provider确定模型类型
        self.is_openai_model = (self.provider == 'openai')
        self.is_ollama_model = (self.provider == 'ollama')
        self.is_lmstudio_model = (self.provider == 'lmstudio')
        
        logger.info(f"Initialized LocalLLM with provider: {self.provider}, model: {self.model_name}")
        
        # 立即加载模型，避免延迟加载导致的多次加载
        self.load_model()
        
    def load_model(self):
        """加载模型"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            if self.is_ollama_model:
                # 使用单实例Ollama客户端
                base_url = config.get('llm.ollama.base_url', 'http://localhost:11434')
                self.ollama_client = OllamaClient(base_url=base_url, model=self.model_name)
                
                # 直接测试连接和生成能力，避免递归调用
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        # 直接测试生成能力，这会自动检查服务可用性
                        test_response = self.ollama_client.generate("Hello")
                        if test_response:
                            logger.info("Ollama model loaded and tested successfully")
                            break
                        else:
                            raise Exception("Model test generation failed")
                    except Exception as e:
                        if attempt < max_retries - 1:
                            logger.warning(f"Attempt {attempt + 1} failed: {e}, retrying...")
                            import time
                            time.sleep(2)  # 等待2秒后重试
                        else:
                            raise Exception(f"Failed to connect to Ollama after {max_retries} attempts: {e}")
            elif self.is_openai_model:
                # 使用OpenAI客户端，支持参数覆盖
                api_key = self.api_key or config.get('llm.openai.api_key')
                base_url = self.base_url or config.get('llm.openai.base_url')
                timeout = config.get('llm.openai.timeout', 60)
                max_retries = config.get('llm.openai.max_retries', 3)
                
                self.openai_client = OpenAIClient(
                    api_key=api_key, 
                    model=self.model_name, 
                    base_url=base_url
                )
                
                # 测试OpenAI连接
                if not self.openai_client.is_available():
                    raise Exception("Failed to connect to OpenAI API")
                
                logger.info("OpenAI model loaded and tested successfully")
            elif self.is_lmstudio_model:
                # 检查是否启用多模型并行
                multi_model_enabled = config.get('llm.multi_model.enabled', False)
                
                if multi_model_enabled:
                    # 使用多模型客户端
                    logger.info("Multi-model mode enabled, initializing MultiModelClient...")
                    self.multi_model_client = MultiModelClient()
                    logger.info("MultiModelClient initialized successfully")
                else:
                    # 使用单个LM Studio客户端
                    base_url = config.get('llm.lmstudio.base_url', f'http://localhost:{config.get("llm.lmstudio.port", 1234)}/v1')
                    self.lmstudio_client = LMStudioClient(base_url=base_url, model=self.model_name)
                    
                    # 测试LM Studio连接
                    if not self.lmstudio_client.is_available():
                        raise Exception("Failed to connect to LM Studio API")
                    
                    logger.info("LM Studio model loaded and tested successfully")
            else:
                # 使用transformers直接加载
                import torch
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map=self.device
                )
                
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """生成文本"""
        
        try:
            if self.is_ollama_model and self.ollama_client:
                # 使用Ollama客户端生成
                return self.ollama_client.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=kwargs.get('temperature', self.temperature),
                    max_tokens=kwargs.get('max_tokens', self.max_tokens)
                )
            elif self.is_openai_model and self.openai_client:
                # 使用OpenAI客户端生成
                return self.openai_client.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=kwargs.get('temperature', self.temperature),
                    max_tokens=kwargs.get('max_tokens', self.max_tokens)
                )
            elif self.is_lmstudio_model:
                if self.multi_model_client:
                    # 使用多模型客户端生成
                    return self.multi_model_client.generate(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        temperature=kwargs.get('temperature', self.temperature),
                        max_tokens=kwargs.get('max_tokens', self.max_tokens)
                    )
                elif self.lmstudio_client:
                    # 使用单个LM Studio客户端生成
                    return self.lmstudio_client.generate(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        temperature=kwargs.get('temperature', self.temperature),
                        max_tokens=kwargs.get('max_tokens', self.max_tokens)
                    )
            else:
                # 使用transformers生成
                if system_prompt:
                    full_prompt = f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"
                else:
                    full_prompt = prompt
                
                inputs = self.tokenizer(full_prompt, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=kwargs.get('max_tokens', self.max_tokens),
                        temperature=kwargs.get('temperature', self.temperature),
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                return generated_text[len(full_prompt):].strip()
                
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
        
        return self.batch_processor.process_batches(
            prompts, 
            process_batch,
            desc="Generating text"
        )
    
    def generate_atomic_notes(self, text_chunks: List[str]) -> List[Dict[str, Any]]:
        """生成原子笔记"""
        system_prompt = ATOMIC_NOTE_SYSTEM_PROMPT
        
        def process_chunk(chunk):
            prompt = ATOMIC_NOTE_PROMPT.format(chunk=chunk)
            
            response = self.generate(prompt, system_prompt)
            
            try:
                import json
                cleaned_response = extract_json_from_response(response)
                note_data = json.loads(cleaned_response)
                return {
                    'original_text': chunk,
                    'content': note_data.get('content', chunk),
                    'keywords': note_data.get('keywords', []),
                    'entities': note_data.get('entities', []),
                    'summary': note_data.get('summary', ''),
                    'length': len(chunk)
                }
            except json.JSONDecodeError:
                # 如果JSON解析失败，返回基本格式
                return {
                    'original_text': chunk,
                    'content': response or chunk,
                    'keywords': [],
                    'entities': [],
                    'summary': '',
                    'length': len(chunk)
                }
        
        return self.batch_processor.process_batches(
            text_chunks,
            lambda batch: [process_chunk(chunk) for chunk in batch],
            desc="Generating atomic notes"
        )
    
    def extract_entities_and_relations(self, text: str) -> Dict[str, Any]:
        """提取实体和关系"""
        system_prompt = EXTRACT_ENTITIES_SYSTEM_PROMPT
        
        prompt = EXTRACT_ENTITIES_PROMPT.format(text=text)
        
        response = self.generate(prompt, system_prompt)
        
        try:
            import json
            cleaned_response = extract_json_from_response(response)
            return json.loads(cleaned_response) if cleaned_response else {'entities': [], 'relations': []}
        except json.JSONDecodeError:
            return {'entities': [], 'relations': []}
    
    def is_available(self) -> bool:
        """检查模型是否可用"""
        try:
            if self.is_ollama_model:
                # 避免递归调用load_model，直接创建临时客户端检查
                if self.ollama_client is None:
                    base_url = config.get('llm.ollama.base_url', 'http://localhost:11434')
                    temp_client = OllamaClient(base_url=base_url, model=self.model_name)
                    return temp_client.is_available()
                return self.ollama_client.is_available()
            elif self.is_openai_model:
                # 检查OpenAI客户端可用性
                if self.openai_client is None:
                    api_key = config.get('llm.openai.api_key')
                    base_url = config.get('llm.openai.base_url')
                    temp_client = OpenAIClient(api_key=api_key, model=self.model_name, base_url=base_url)
                    return temp_client.is_available()
                return self.openai_client.is_available()
            elif self.is_lmstudio_model:
                # 检查LM Studio客户端可用性
                if self.lmstudio_client is None:
                    base_url = config.get('llm.lmstudio.base_url', f'http://localhost:{config.get("llm.lmstudio.port", 1234)}/v1')
                    temp_client = LMStudioClient(base_url=base_url, model=self.model_name)
                    return temp_client.is_available()
                return self.lmstudio_client.is_available()
            else:
                if self.model is None:
                    self.load_model()
                return True
        except Exception:
            return False
    
    
    def cleanup(self):
        """清理模型资源"""
        if self.ollama_client is not None:
            self.ollama_client = None
            
        if self.openai_client is not None:
            self.openai_client = None
            
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
            
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            
        if self.ollama_client is not None:
            self.ollama_client = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("LocalLLM resources cleaned up")
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'provider': self.provider,
            'model_name': self.model_name,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'device': self.device,
            'is_available': self.is_available(),
            'api_key_configured': bool(self.api_key) if self.is_openai_model else None,
            'base_url': self.base_url if self.is_openai_model else None
        }
    
    def list_available_models(self) -> List[str]:
        """列出可用的模型"""
        try:
            if self.is_ollama_model and self.ollama_client:
                return self.ollama_client.list_models()
            elif self.is_openai_model and self.openai_client:
                return self.openai_client.list_models()
            elif self.is_lmstudio_model and self.lmstudio_client:
                return self.lmstudio_client.list_models()
            else:
                # 对于transformers模型，返回当前模型
                return [self.model_name] if self.model_name else []
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
