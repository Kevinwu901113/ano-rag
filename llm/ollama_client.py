import requests
import json
from typing import List, Dict, Any, Optional, Union
from loguru import logger
from config import config

class OllamaClient:
    """Ollama客户端，用于与本地Ollama服务通信"""
    
    def __init__(self, base_url: str = None, model: str = None):
        self.base_url = base_url or config.get('llm.ollama.base_url', 'http://localhost:11434')
        self.model = model or config.get('llm.ollama.model', 'llama3.1:8b')
        self.temperature = config.get('llm.ollama.temperature', 0.7)
        self.max_tokens = config.get('llm.ollama.max_tokens', 4096)
        
    def _make_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """发送请求到Ollama服务"""
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = requests.post(url, json=data, timeout=300)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama request failed: {e}")
            raise
    
    def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """生成文本"""
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get('temperature', self.temperature),
                "num_predict": kwargs.get('max_tokens', self.max_tokens)
            }
        }
        
        if system_prompt:
            data["system"] = system_prompt
        
        try:
            response = self._make_request("api/generate", data)
            return response.get('response', '')
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return ""
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """聊天模式生成"""
        data = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": kwargs.get('temperature', self.temperature),
                "num_predict": kwargs.get('max_tokens', self.max_tokens)
            }
        }
        
        try:
            response = self._make_request("api/chat", data)
            return response.get('message', {}).get('content', '')
        except Exception as e:
            logger.error(f"Chat failed: {e}")
            return ""
    
    def generate_final_answer(self, context: str, query: str) -> str:
        """生成最终答案"""
        system_prompt = """
你是一个专业的知识问答助手。请根据提供的上下文信息回答用户的问题。

要求：
1. 基于提供的上下文信息进行回答
2. 如果上下文信息不足以回答问题，请明确说明
3. 回答要准确、简洁、有条理
4. 不要编造不存在于上下文中的信息
"""
        
        prompt = f"""
上下文信息：
{context}

用户问题：{query}

请基于上述上下文信息回答用户问题：
"""
        
        return self.generate(prompt, system_prompt)
    
    def evaluate_answer(self, query: str, context: str, answer: str) -> Dict[str, float]:
        """评估答案质量"""
        system_prompt = """
你是一个专业的答案质量评估专家。请从以下几个维度评估答案的质量：

1. 相关性 (0-1)：答案与问题的相关程度
2. 准确性 (0-1)：答案基于上下文的准确程度
3. 完整性 (0-1)：答案的完整程度
4. 清晰度 (0-1)：答案的表达清晰程度

请以JSON格式返回评分，例如：
{"relevance": 0.9, "accuracy": 0.8, "completeness": 0.7, "clarity": 0.9}
"""
        
        prompt = f"""
问题：{query}

上下文：{context}

答案：{answer}

请评估上述答案的质量：
"""
        
        try:
            response = self.generate(prompt, system_prompt)
            # 尝试解析JSON响应
            scores = json.loads(response)
            return scores
        except Exception as e:
            logger.error(f"Answer evaluation failed: {e}")
            # 返回默认分数
            return {
                "relevance": 0.5,
                "accuracy": 0.5,
                "completeness": 0.5,
                "clarity": 0.5
            }
    
    def batch_generate(self, prompts: List[str], system_prompt: str = None, **kwargs) -> List[str]:
        """批量生成（串行处理，因为Ollama通常是单实例）"""
        results = []
        
        for prompt in prompts:
            try:
                result = self.generate(prompt, system_prompt, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch generation failed for prompt: {e}")
                results.append("")
        
        return results
    
    def is_available(self) -> bool:
        """检查Ollama服务是否可用"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def list_models(self) -> List[str]:
        """列出可用的模型"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            data = response.json()
            return [model['name'] for model in data.get('models', [])]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []