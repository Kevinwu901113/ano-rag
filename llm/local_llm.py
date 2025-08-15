import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import List, Dict, Any, Optional
from loguru import logger
from config import config
from utils.batch_processor import BatchProcessor
from utils.json_utils import extract_json_from_response
from .ollama_client import OllamaClient
from .multi_ollama_client import MultiOllamaClient
from .openai_client import OpenAIClient
from .prompts import (
    ATOMIC_NOTE_SYSTEM_PROMPT,
    ATOMIC_NOTE_PROMPT,
    EXTRACT_ENTITIES_SYSTEM_PROMPT,
    EXTRACT_ENTITIES_PROMPT,
)

class LocalLLM:
    """本地LLM类，用于原子笔记生成和查询重写等任务"""
    
    def __init__(self, model_name: str = None, device: str = None):
        # 获取provider配置
        self.provider = config.get('llm.local_model.provider', 'ollama')
        
        # 根据provider选择配置段
        if self.provider == 'openai':
            config_section = 'llm.openai'
            default_model = 'gpt-3.5-turbo'
        else:  # ollama
            config_section = 'llm.ollama'
            default_model = 'llama3.1:8b'
        
        # 从对应配置段获取参数，local_model的参数可以覆盖
        self.model_name = model_name or config.get(f'{config_section}.model', default_model)
        self.device = device or config.get('llm.local_model.device', 'auto')
        self.temperature = config.get('llm.local_model.temperature') or config.get(f'{config_section}.temperature', 0.1)
        self.max_tokens = config.get('llm.local_model.max_tokens') or config.get(f'{config_section}.max_tokens', 2048)
        
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.ollama_client = None
        self.openai_client = None
        self.batch_processor = BatchProcessor()
        
        # 根据provider确定模型类型
        self.is_openai_model = (self.provider == 'openai')
        self.is_ollama_model = (self.provider == 'ollama')
        
        logger.info(f"Initialized LocalLLM with provider: {self.provider}, model: {self.model_name}")
        
    def load_model(self):
        """加载模型"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            if self.is_ollama_model:
                # 使用多实例Ollama客户端，支持负载均衡
                base_url = config.get('llm.ollama.base_url', 'http://localhost:11434')
                self.ollama_client = MultiOllamaClient(base_url=base_url, model=self.model_name)
                
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
                # 使用OpenAI客户端，从openai配置段获取参数
                api_key = config.get('llm.openai.api_key')
                base_url = config.get('llm.openai.base_url')
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
        if self.ollama_client is None and self.openai_client is None and self.model is None:
            self.load_model()
        
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
                    temp_client = MultiOllamaClient(base_url=base_url, model=self.model_name)
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
            else:
                if self.model is None:
                    self.load_model()
                return True
        except Exception:
            return False
    
    
    def cleanup(self):
        """清理模型资源"""
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
