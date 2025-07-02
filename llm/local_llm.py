import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import List, Dict, Any, Optional
from loguru import logger
from config import config
from utils import BatchProcessor

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
        self.batch_processor = BatchProcessor()
        
    def load_model(self):
        """加载模型"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # 对于Ollama模型，我们使用pipeline方式
            if 'llama' in self.model_name.lower() or ':' in self.model_name:
                # 使用text-generation pipeline
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model_name,
                    device=0 if self.device == 'cuda' and torch.cuda.is_available() else -1,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
            else:
                # 使用transformers直接加载
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
        if self.pipeline is None and self.model is None:
            self.load_model()
        
        try:
            if system_prompt:
                full_prompt = f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"
            else:
                full_prompt = prompt
            
            if self.pipeline:
                # 使用pipeline生成
                outputs = self.pipeline(
                    full_prompt,
                    max_new_tokens=kwargs.get('max_tokens', self.max_tokens),
                    temperature=kwargs.get('temperature', self.temperature),
                    do_sample=True,
                    pad_token_id=self.pipeline.tokenizer.eos_token_id
                )
                return outputs[0]['generated_text'][len(full_prompt):].strip()
            else:
                # 使用transformers直接生成
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
        system_prompt = """
你是一个专业的知识提取专家。请将给定的文本块转换为原子笔记。

原子笔记要求：
1. 每个笔记包含一个独立的知识点
2. 内容简洁明了，易于理解
3. 保留关键信息和上下文
4. 使用结构化的格式

请以JSON格式返回，包含以下字段：
- content: 笔记内容
- keywords: 关键词列表
- entities: 实体列表
- summary: 简要总结
"""
        
        def process_chunk(chunk):
            prompt = f"""
请将以下文本转换为原子笔记：

{chunk}

请返回JSON格式的原子笔记：
"""
            
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
        system_prompt = """
你是一个专业的实体关系提取专家。请从给定文本中提取实体和它们之间的关系。

请以JSON格式返回，包含：
- entities: 实体列表，每个实体包含name和type
- relations: 关系列表，每个关系包含source, target, relation_type
"""
        
        prompt = f"""
请从以下文本中提取实体和关系：

{text}

请返回JSON格式的结果：
"""
        
        response = self.generate(prompt, system_prompt)
        
        try:
            import json
            return json.loads(response)
        except json.JSONDecodeError:
            return {'entities': [], 'relations': []}
    
    def is_available(self) -> bool:
        """检查模型是否可用"""
        try:
            if self.pipeline is None and self.model is None:
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
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()