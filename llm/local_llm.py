import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import List, Dict, Any, Optional
from loguru import logger
from config import config
from utils import BatchProcessor
from .ollama_client import OllamaClient
from .prompts import (
    ATOMIC_NOTE_SYSTEM_PROMPT,
    ATOMIC_NOTE_PROMPT,
    EXTRACT_ENTITIES_SYSTEM_PROMPT,
    EXTRACT_ENTITIES_PROMPT,
)

class LocalLLM:
    """本地LLM类，用于原子笔记生成和查询重写等任务"""
    
    def __init__(self, model_name: str = None, device: str = None):
        self.model_name = model_name or config.get('llm.local_model.model', 'llama3.1:8b')
        self.device = device or config.get('llm.local_model.device', 'auto')
        self.temperature = config.get('llm.local_model.temperature', 0.1)
        self.max_tokens = config.get('llm.local_model.max_tokens', 2048)
        
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.ollama_client = None
        self.batch_processor = BatchProcessor()
        
        # 检查是否是Ollama模型格式
        self.is_ollama_model = ':' in self.model_name
        
    def load_model(self):
        """加载模型"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            if self.is_ollama_model:
                # 使用Ollama客户端
                base_url = config.get('llm.local_model.base_url', 'http://localhost:11434')
                self.ollama_client = OllamaClient(base_url=base_url, model=self.model_name)
                
                # 直接测试连接和生成能力，避免递归调用
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        # 直接测试生成能力，这会自动检查服务可用性
                        test_response = self.ollama_client.generate("Hello", timeout=10)
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
        if self.ollama_client is None and self.model is None:
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
                note_data = json.loads(response)
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
            return json.loads(response)
        except json.JSONDecodeError:
            return {'entities': [], 'relations': []}
    
    def is_available(self) -> bool:
        """检查模型是否可用"""
        try:
            if self.is_ollama_model:
                # 避免递归调用load_model，直接创建临时客户端检查
                if self.ollama_client is None:
                    base_url = config.get('llm.local_model.base_url', 'http://localhost:11434')
                    temp_client = OllamaClient(base_url=base_url, model=self.model_name)
                    return temp_client.is_available()
                return self.ollama_client.is_available()
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
