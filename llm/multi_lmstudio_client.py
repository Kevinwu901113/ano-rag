import json
import random
import time
import threading
from typing import Any, Dict, Iterator, List, Union, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum

import openai
import requests
from loguru import logger

from config import config
from .lmstudio_client import LMStudioClient
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
class LMStudioInstance:
    """LM Studio实例配置"""
    base_url: str
    model: str
    port: int
    client: Optional[LMStudioClient] = None
    is_healthy: bool = True
    last_health_check: float = 0
    active_requests: int = 0
    total_requests: int = 0
    error_count: int = 0
    
    def __post_init__(self):
        if self.client is None:
            self.client = LMStudioClient(base_url=self.base_url, model=self.model, port=self.port)


class MultiLMStudioClient:
    """支持多实例负载均衡的LM Studio客户端
    
    当LM Studio加载多个相同模型实例时，系统能自动识别并实现并发调用以提高效率。
    支持多种负载均衡策略和健康检查机制。
    """

    def __init__(self, base_url: str = None, model: str = None, port: int = None):
        # 检查是否启用多实例配置
        self.multi_instance_enabled = config.get("llm.lmstudio.multiple_instances.enabled", False)
        
        if self.multi_instance_enabled:
            self._init_multi_instance()
        else:
            self._init_single_instance(base_url, model, port)
        
        # 通用配置
        self.temperature = config.get("llm.lmstudio.temperature", 0.7)
        self.max_tokens = config.get("llm.lmstudio.max_tokens", 4096)
        self.timeout = config.get("llm.lmstudio.timeout", 60)
        
        # 默认选项
        self.default_options = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        
        # 负载均衡相关
        self._round_robin_index = 0
        self._lock = threading.Lock()
        
        # 启动健康检查线程
        if self.multi_instance_enabled:
            self._start_health_check_thread()
    
    def _init_single_instance(self, base_url: str = None, model: str = None, port: int = None):
        """初始化单实例模式"""
        port = port or config.get("llm.lmstudio.port", 1234)
        base_url = base_url or config.get("llm.lmstudio.base_url", f"http://localhost:{port}/v1")
        model = model or config.get("llm.lmstudio.model", "default-model")
        
        self.instances = [LMStudioInstance(base_url=base_url, model=model, port=port)]
        self.load_balancing_strategy = LoadBalancingStrategy.ROUND_ROBIN
        
        # 设置重试配置，确保单实例模式下也有这个属性
        self.max_retries_per_instance = config.get("llm.lmstudio.multiple_instances.max_retries_per_instance", 2)
        
        logger.info(f"Initialized single LM Studio instance: {base_url}")
    
    def _init_multi_instance(self):
        """初始化多实例模式"""
        # 优先尝试自动检测LM Studio中的模型实例
        auto_detected_instances = self._auto_detect_lmstudio_instances()
        
        if auto_detected_instances:
            logger.info(f"Auto-detected {len(auto_detected_instances)} LM Studio model instances")
            self.instances = auto_detected_instances
        else:
            # 回退到配置文件中的实例配置
            instances_config = config.get("llm.lmstudio.multiple_instances.instances", [])
            if not instances_config:
                logger.warning("Multi-instance enabled but no instances detected or configured, falling back to single instance")
                self._init_single_instance(None, None, None)
                return
            
            self.instances = []
            for instance_config in instances_config:
                port = instance_config.get("port", 1234)
                base_url = instance_config.get("base_url", f"http://localhost:{port}/v1")
                model = instance_config["model"]
                
                instance = LMStudioInstance(
                    base_url=base_url,
                    model=model,
                    port=port
                )
                self.instances.append(instance)
        
        # 负载均衡策略
        strategy_str = config.get("llm.lmstudio.multiple_instances.load_balancing", "round_robin")
        self.load_balancing_strategy = LoadBalancingStrategy(strategy_str)
        
        # 健康检查配置
        self.health_check_interval = config.get("llm.lmstudio.multiple_instances.health_check_interval", 30)
        self.max_retries_per_instance = config.get("llm.lmstudio.multiple_instances.max_retries_per_instance", 2)
        
        logger.info(f"Initialized {len(self.instances)} LM Studio instances with {strategy_str} load balancing")
    
    def _auto_detect_lmstudio_instances(self):
        """自动检测LM Studio中的模型实例"""
        port = config.get("llm.lmstudio.port", 1234)
        base_url = f"http://localhost:{port}/v1"
        
        try:
            # 尝试连接到LM Studio并获取模型列表
            response = requests.get(f"{base_url}/models", timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                models = models_data.get("data", [])
                
                if not models:
                    logger.warning("No models found in LM Studio")
                    return []
                
                instances = []
                model_count = {}
                
                for model_info in models:
                    model_id = model_info.get("id", "unknown-model")
                    
                    # 处理相同模型名称的情况，添加序号后缀
                    if model_id in model_count:
                        model_count[model_id] += 1
                        instance_model_name = f"{model_id}:{model_count[model_id]}"
                    else:
                        model_count[model_id] = 1
                        instance_model_name = model_id
                    
                    instance = LMStudioInstance(
                        base_url=base_url,
                        model=instance_model_name,
                        port=port
                    )
                    instances.append(instance)
                    logger.info(f"Detected LM Studio model instance: {instance_model_name}")
                
                return instances
            else:
                logger.warning(f"Failed to connect to LM Studio at {base_url}, status code: {response.status_code}")
                return []
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"Could not connect to LM Studio at {base_url}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error detecting LM Studio instances: {e}")
            return []
    
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
        logger.info("LM Studio health check thread started")
    
    def _perform_health_checks(self):
        """执行健康检查"""
        current_time = time.time()
        
        for instance in self.instances:
            if current_time - instance.last_health_check > self.health_check_interval:
                try:
                    # 使用LM Studio的健康检查接口
                    health_url = instance.base_url.replace('/v1', '') + '/v1/models'
                    response = requests.get(health_url, timeout=3)
                    instance.is_healthy = response.status_code == 200
                    if not instance.is_healthy:
                        logger.warning(f"LM Studio instance {instance.base_url} health check failed: status {response.status_code}")
                except Exception as e:
                    instance.is_healthy = False
                    instance.error_count += 1
                    logger.warning(f"LM Studio instance {instance.base_url} health check failed: {e}")
                
                instance.last_health_check = current_time
    
    def _select_instance(self) -> LMStudioInstance:
        """根据负载均衡策略选择实例"""
        healthy_instances = [inst for inst in self.instances if inst.is_healthy]
        
        if not healthy_instances:
            logger.warning("No healthy LM Studio instances available, using first instance")
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
                logger.warning(f"Request failed on LM Studio {instance.base_url} (attempt {attempt + 1}): {e}")
                
                if attempt < self.max_retries_per_instance:
                    time.sleep(0.5 * (attempt + 1))  # 指数退避
            
            finally:
                instance.active_requests = max(0, instance.active_requests - 1)
        
        # 所有重试都失败
        raise last_exception or Exception("All LM Studio instances failed")
    
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> str:
        """生成文本"""
        def _generate(instance: LMStudioInstance, prompt: str, system_prompt: str | None, stream: bool, **kwargs):
            return instance.client.generate(prompt, system_prompt, stream=stream, **kwargs)
        
        return self._execute_with_retry(_generate, prompt, system_prompt, stream, **kwargs)
    
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[str, Iterator[str]]:
        """聊天接口"""
        def _chat(instance: LMStudioInstance, messages: list[dict[str, str]], stream: bool, **kwargs):
            return instance.client.chat(messages, stream=stream, **kwargs)
        
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
        prompt = FINAL_ANSWER_PROMPT.format(context=context, query=query)
        return self.generate(prompt, FINAL_ANSWER_SYSTEM_PROMPT)
    
    def evaluate_answer(self, query: str, context: str, answer: str) -> Dict[str, float]:
        """评估答案质量"""
        prompt = EVALUATE_ANSWER_PROMPT.format(query=query, context=context, answer=answer)
        response = self.generate(prompt, EVALUATE_ANSWER_SYSTEM_PROMPT)
        
        try:
            return json.loads(self._clean_json_response(response))
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse evaluation response: {response}")
            return {"relevance": 0.5, "completeness": 0.5, "accuracy": 0.5}
    
    def _clean_response(self, response: str) -> str:
        """清理和标准化响应文本"""
        if not response:
            return ""
        
        # 移除常见的伪影
        response = response.strip()
        
        # 移除控制字符
        response = self._clean_control_characters(response)
        
        return response
    
    def _clean_json_response(self, response: str) -> str:
        """通过提取有效JSON来清理JSON响应"""
        response = response.strip()
        
        # 尝试在响应中找到JSON
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1
        
        if start_idx != -1 and end_idx > start_idx:
            json_str = response[start_idx:end_idx]
            
            # 清理常见问题
            json_str = json_str.replace('\n', ' ')
            json_str = json_str.replace('\t', ' ')
            
            # 移除多个空格
            import re
            json_str = re.sub(r'\s+', ' ', json_str)
            
            return json_str
        
        return response
    
    def _clean_control_characters(self, text: str) -> str:
        """从文本中移除控制字符"""
        import re
        # 移除控制字符，除了换行符和制表符
        cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        return cleaned
    
    def is_available(self) -> bool:
        """检查是否有可用的LM Studio实例"""
        return any(instance.client.is_available() for instance in self.instances)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取实例统计信息"""
        total_requests = sum(inst.total_requests for inst in self.instances)
        total_errors = sum(inst.error_count for inst in self.instances)
        healthy_count = sum(1 for inst in self.instances if inst.is_healthy)
        
        return {
            "total_instances": len(self.instances),
            "healthy_instances": healthy_count,
            "total_requests": total_requests,
            "total_errors": total_errors,
            "error_rate": total_errors / max(total_requests, 1),
            "load_balancing_strategy": self.load_balancing_strategy.value,
            "multi_instance_enabled": self.multi_instance_enabled,
            "instances": [
                {
                    "base_url": inst.base_url,
                    "model": inst.model,
                    "port": inst.port,
                    "is_healthy": inst.is_healthy,
                    "active_requests": inst.active_requests,
                    "total_requests": inst.total_requests,
                    "error_count": inst.error_count,
                }
                for inst in self.instances
            ]
        }
    
    def list_models(self) -> List[str]:
        """列出所有实例的可用模型"""
        all_models = set()
        for instance in self.instances:
            if instance.is_healthy:
                try:
                    models = instance.client.list_models()
                    all_models.update(models)
                except Exception as e:
                    logger.error(f"Failed to list models from {instance.base_url}: {e}")
        return list(all_models)
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'provider': 'lmstudio',
            'multi_instance_enabled': self.multi_instance_enabled,
            'instances_count': len(self.instances),
            'healthy_instances_count': sum(1 for inst in self.instances if inst.is_healthy),
            'load_balancing_strategy': self.load_balancing_strategy.value,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'is_available': self.is_available(),
            'stats': self.get_stats()
        }