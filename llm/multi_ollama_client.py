import json
import random
import time
import threading
from typing import Any, Dict, Iterator, List, Union, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum

import ollama
import requests
from loguru import logger

from config import config
from .prompts import (
    FINAL_ANSWER_SYSTEM_PROMPT,
    FINAL_ANSWER_PROMPT,
    EVALUATE_ANSWER_SYSTEM_PROMPT,
    EVALUATE_ANSWER_PROMPT,
)


class LoadBalancingStrategy(Enum):
    """负载均衡策略"""
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    LEAST_BUSY = "least_busy"


@dataclass
class OllamaInstance:
    """Ollama实例配置"""
    base_url: str
    model: str
    client: Optional[ollama.Client] = None
    is_healthy: bool = True
    last_health_check: float = 0
    active_requests: int = 0
    total_requests: int = 0
    error_count: int = 0
    
    def __post_init__(self):
        if self.client is None:
            self.client = ollama.Client(host=self.base_url)


class MultiOllamaClient:
    """支持多实例负载均衡的Ollama客户端"""

    def __init__(self, base_url: str | None = None, model: str | None = None):
        # 检查是否启用多实例配置
        self.multi_instance_enabled = config.get("llm.ollama.multiple_instances.enabled", False)
        
        if self.multi_instance_enabled:
            self._init_multi_instance()
        else:
            self._init_single_instance(base_url, model)
        
        # 通用配置
        self.temperature = config.get("llm.ollama.temperature", 0.7)
        self.max_tokens = config.get("llm.ollama.max_tokens", 4096)
        self.num_ctx = config.get("llm.ollama.num_ctx", 32768)
        self.timeout = config.get("llm.ollama.timeout", 60)
        
        # 默认选项
        self.default_options = {
            "num_ctx": self.num_ctx,
            "temperature": self.temperature,
            "num_predict": self.max_tokens,
        }
        
        # 负载均衡相关
        self._round_robin_index = 0
        self._lock = threading.Lock()
        
        # 启动健康检查线程
        if self.multi_instance_enabled:
            self._start_health_check_thread()
    
    def _init_single_instance(self, base_url: str | None, model: str | None):
        """初始化单实例模式"""
        base_url = base_url or config.get("llm.ollama.base_url", "http://localhost:11434")
        model = model or config.get("llm.ollama.model", "gemma3:4b-it-fp16")
        
        self.instances = [OllamaInstance(base_url=base_url, model=model)]
        self.load_balancing_strategy = LoadBalancingStrategy.ROUND_ROBIN
        
        # 设置重试配置，确保单实例模式下也有这个属性
        self.max_retries_per_instance = config.get("llm.ollama.multiple_instances.max_retries_per_instance", 2)
        
        logger.info(f"Initialized single Ollama instance: {base_url}")
    
    def _init_multi_instance(self):
        """初始化多实例模式"""
        instances_config = config.get("llm.ollama.multiple_instances.instances", [])
        if not instances_config:
            logger.warning("Multi-instance enabled but no instances configured, falling back to single instance")
            self._init_single_instance(None, None)
            return
        
        self.instances = []
        for instance_config in instances_config:
            instance = OllamaInstance(
                base_url=instance_config["base_url"],
                model=instance_config["model"]
            )
            self.instances.append(instance)
        
        # 负载均衡策略
        strategy_str = config.get("llm.ollama.multiple_instances.load_balancing", "round_robin")
        self.load_balancing_strategy = LoadBalancingStrategy(strategy_str)
        
        # 健康检查配置
        self.health_check_interval = config.get("llm.ollama.multiple_instances.health_check_interval", 30)
        self.max_retries_per_instance = config.get("llm.ollama.multiple_instances.max_retries_per_instance", 2)
        
        logger.info(f"Initialized {len(self.instances)} Ollama instances with {strategy_str} load balancing")
    
    def _start_health_check_thread(self):
        """启动健康检查线程"""
        def health_check_worker():
            while True:
                try:
                    self._perform_health_checks()
                    time.sleep(self.health_check_interval)
                except Exception as e:
                    logger.error(f"Health check thread error: {e}")
                    time.sleep(5)  # 出错时短暂等待
        
        thread = threading.Thread(target=health_check_worker, daemon=True)
        thread.start()
        logger.info("Health check thread started")
    
    def _perform_health_checks(self):
        """执行健康检查"""
        current_time = time.time()
        
        for instance in self.instances:
            if current_time - instance.last_health_check > self.health_check_interval:
                try:
                    # 快速健康检查
                    response = requests.get(f"{instance.base_url}/api/version", timeout=3)
                    instance.is_healthy = response.status_code == 200
                    if not instance.is_healthy:
                        logger.warning(f"Instance {instance.base_url} health check failed: status {response.status_code}")
                except Exception as e:
                    instance.is_healthy = False
                    instance.error_count += 1
                    logger.warning(f"Instance {instance.base_url} health check failed: {e}")
                
                instance.last_health_check = current_time
    
    def _select_instance(self) -> OllamaInstance:
        """根据负载均衡策略选择实例"""
        healthy_instances = [inst for inst in self.instances if inst.is_healthy]
        
        if not healthy_instances:
            logger.warning("No healthy instances available, using first instance")
            return self.instances[0]
        
        if self.load_balancing_strategy == LoadBalancingStrategy.ROUND_ROBIN:
            with self._lock:
                instance = healthy_instances[self._round_robin_index % len(healthy_instances)]
                self._round_robin_index += 1
                return instance
        
        elif self.load_balancing_strategy == LoadBalancingStrategy.RANDOM:
            return random.choice(healthy_instances)
        
        elif self.load_balancing_strategy == LoadBalancingStrategy.LEAST_BUSY:
            return min(healthy_instances, key=lambda x: x.active_requests)
        
        return healthy_instances[0]
    
    def _execute_with_retry(self, func, *args, **kwargs):
        """带重试的执行函数"""
        last_exception = None
        
        for attempt in range(self.max_retries_per_instance + 1):
            instance = self._select_instance()
            
            try:
                # 更新实例状态
                instance.active_requests += 1
                instance.total_requests += 1
                
                # 执行请求
                result = func(instance, *args, **kwargs)
                return result
                
            except Exception as e:
                last_exception = e
                instance.error_count += 1
                instance.is_healthy = False  # 标记为不健康
                logger.warning(f"Request failed on {instance.base_url} (attempt {attempt + 1}): {e}")
                
                if attempt < self.max_retries_per_instance:
                    time.sleep(0.5 * (attempt + 1))  # 指数退避
            
            finally:
                instance.active_requests = max(0, instance.active_requests - 1)
        
        # 所有重试都失败
        raise last_exception or Exception("All instances failed")
    
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> str:
        """生成文本"""
        def _generate(instance: OllamaInstance, prompt: str, system_prompt: str | None, stream: bool, **kwargs):
            # 合并选项
            options = self.default_options.copy()
            options.update({
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", self.max_tokens),
                "num_ctx": kwargs.get("num_ctx", self.num_ctx),
            })
            
            # 准备完整提示
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            # 执行生成
            response = instance.client.generate(
                model=instance.model,
                prompt=full_prompt,
                stream=stream,
                options=options,
            )
            
            text: str = ""
            if stream:
                for chunk in response:
                    if hasattr(chunk, "response"):
                        text += chunk.response
            else:
                if hasattr(response, "response"):
                    text = response.response
                else:
                    text = str(response)
            
            return self._clean_response(text)
        
        return self._execute_with_retry(_generate, prompt, system_prompt, stream, **kwargs)
    
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[str, Iterator[str]]:
        """聊天接口"""
        def _chat(instance: OllamaInstance, messages: list[dict[str, str]], stream: bool, **kwargs):
            options = self.default_options.copy()
            options.update({
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", self.max_tokens),
                "num_ctx": kwargs.get("num_ctx", self.num_ctx),
            })
            
            response = instance.client.chat(
                model=instance.model,
                messages=messages,
                stream=stream,
                options=options,
            )
            
            if stream:
                return (chunk.message.content for chunk in response)
            
            result = response.message.content if hasattr(response, 'message') else str(response)
            return self._clean_response(result)
        
        return self._execute_with_retry(_chat, messages, stream, **kwargs)
    
    def batch_generate(
        self,
        prompts: List[str],
        system_prompt: str | None = None,
        *,
        stream: bool = False,
        max_workers: int = None,
        **kwargs: Any,
    ) -> List[str]:
        """批量生成（并行处理）"""
        if not self.multi_instance_enabled or len(prompts) <= 1:
            # 单实例或单个提示，使用串行处理
            results = []
            for prompt in prompts:
                try:
                    result = self.generate(prompt, system_prompt, stream=stream, **kwargs)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Batch generation failed for prompt: {e}")
                    results.append("")
            return results
        
        # 多实例并行处理
        max_workers = max_workers or min(len(self.instances), len(prompts))
        results = [None] * len(prompts)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_index = {
                executor.submit(self.generate, prompt, system_prompt, stream=stream, **kwargs): i
                for i, prompt in enumerate(prompts)
            }
            
            # 收集结果
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    logger.error(f"Batch generation failed for prompt {index}: {e}")
                    results[index] = ""
        
        return results
    
    def generate_final_answer(self, context: str, query: str) -> str:
        """生成最终答案"""
        system_prompt = FINAL_ANSWER_SYSTEM_PROMPT
        prompt = FINAL_ANSWER_PROMPT.format(context=context, query=query)
        return self.generate(prompt, system_prompt)
    
    def evaluate_answer(self, query: str, context: str, answer: str) -> Dict[str, float]:
        """评估答案"""
        system_prompt = EVALUATE_ANSWER_SYSTEM_PROMPT
        prompt = EVALUATE_ANSWER_PROMPT.format(query=query, context=context, answer=answer)
        try:
            text = self.generate(prompt, system_prompt)
            cleaned_text = self._clean_json_response(text)
            return json.loads(cleaned_text)
        except Exception as e:
            logger.error(f"Answer evaluation failed: {e}")
            return {"relevance": 0.5, "accuracy": 0.5, "completeness": 0.5, "clarity": 0.5}
    
    def _clean_response(self, response: str) -> str:
        """清理响应文本"""
        if not response:
            return ""
        
        # 移除控制字符
        response = self._clean_control_characters(response)
        
        # 移除多余的空白字符
        lines = response.split('\n')
        cleaned_lines = [line.rstrip() for line in lines]
        
        # 移除连续的空行
        result_lines = []
        prev_empty = False
        for line in cleaned_lines:
            if line.strip():
                result_lines.append(line)
                prev_empty = False
            elif not prev_empty:
                result_lines.append(line)
                prev_empty = True
        
        return '\n'.join(result_lines).strip()
    
    def _clean_json_response(self, response: str) -> str:
        """清理JSON响应"""
        if not response:
            return "{}"
        
        response = self._clean_control_characters(response)
        response = response.strip()
        
        # 移除markdown代码块标记
        if response.startswith('```json'):
            response = response[7:]
        elif response.startswith('```'):
            response = response[3:]
        
        if response.endswith('```'):
            response = response[:-3]
        
        response = response.strip()
        
        # 提取JSON对象
        import re
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response)
        if json_match:
            response = json_match.group(0)
        
        if not response or not (response.startswith('{') or response.startswith('[')):
            return "{}"
        
        return response
    
    def _clean_control_characters(self, text: str) -> str:
        """清理控制字符"""
        if not text:
            return ""
        
        # 移除常见的控制字符，但保留换行符和制表符
        import re
        # 移除除了\n, \t, \r之外的控制字符
        cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        return cleaned
    
    def is_available(self) -> bool:
        """检查服务可用性"""
        healthy_instances = [inst for inst in self.instances if inst.is_healthy]
        return len(healthy_instances) > 0
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            "multi_instance_enabled": self.multi_instance_enabled,
            "total_instances": len(self.instances),
            "healthy_instances": len([inst for inst in self.instances if inst.is_healthy]),
            "load_balancing_strategy": self.load_balancing_strategy.value,
            "instances": []
        }
        
        for i, instance in enumerate(self.instances):
            instance_stats = {
                "index": i,
                "base_url": instance.base_url,
                "model": instance.model,
                "is_healthy": instance.is_healthy,
                "active_requests": instance.active_requests,
                "total_requests": instance.total_requests,
                "error_count": instance.error_count,
                "last_health_check": instance.last_health_check
            }
            stats["instances"].append(instance_stats)
        
        return stats
    
    def list_models(self) -> List[str]:
        """列出可用模型"""
        try:
            instance = self._select_instance()
            response = instance.client.list()
            return [m.model for m in response.models]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []