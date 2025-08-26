import json
import random
import time
import threading
from typing import Any, Dict, Iterator, List, Union, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
import subprocess
import psutil
import GPUtil

import openai
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
    """负载均衡策略枚举"""
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    LEAST_BUSY = "least_busy"
    GPU_OPTIMAL = "gpu_optimal"  # 基于GPU使用率的优化策略
    PERFORMANCE_BASED = "performance_based"  # 基于性能指标的策略


@dataclass
class ModelInstance:
    """模型实例配置"""
    model_name: str
    port: int
    base_url: str
    gpu_id: Optional[int] = None
    process: Optional[subprocess.Popen] = None
    client: Optional[openai.OpenAI] = None
    is_healthy: bool = False
    is_loading: bool = False
    last_health_check: float = 0
    active_requests: int = 0
    total_requests: int = 0
    error_count: int = 0
    avg_response_time: float = 0.0
    gpu_memory_usage: float = 0.0
    
    def __post_init__(self):
        if self.client is None and not self.is_loading:
            self._init_client()
    
    def _init_client(self):
        """初始化OpenAI客户端"""
        try:
            client_kwargs = {
                "api_key": "lm-studio",
                "base_url": self.base_url,
                "timeout": config.get("llm.multi_model.timeout", 60),
                "max_retries": config.get("llm.multi_model.max_retries", 3),
            }
            self.client = openai.OpenAI(**client_kwargs)
            logger.info(f"Initialized client for model {self.model_name} on port {self.port}")
        except Exception as e:
            logger.error(f"Failed to initialize client for {self.model_name}: {e}")
            self.is_healthy = False


class GPUResourceManager:
    """GPU资源管理器"""
    
    def __init__(self):
        self.gpu_info = self._get_gpu_info()
        self.allocated_memory = {gpu_id: 0 for gpu_id in range(len(self.gpu_info))}
        self._lock = threading.Lock()
    
    def _get_gpu_info(self) -> List[Dict[str, Any]]:
        """获取GPU信息"""
        try:
            gpus = GPUtil.getGPUs()
            return [
                {
                    "id": gpu.id,
                    "name": gpu.name,
                    "memory_total": gpu.memoryTotal,
                    "memory_free": gpu.memoryFree,
                    "memory_used": gpu.memoryUsed,
                    "load": gpu.load,
                }
                for gpu in gpus
            ]
        except Exception as e:
            logger.warning(f"Failed to get GPU info: {e}")
            return []
    
    def allocate_gpu(self, memory_required: float = 4096) -> Optional[int]:
        """分配GPU资源"""
        with self._lock:
            for gpu in self.gpu_info:
                gpu_id = gpu["id"]
                available_memory = gpu["memory_total"] - self.allocated_memory[gpu_id]
                if available_memory >= memory_required:
                    self.allocated_memory[gpu_id] += memory_required
                    logger.info(f"Allocated {memory_required}MB on GPU {gpu_id}")
                    return gpu_id
            return None
    
    def release_gpu(self, gpu_id: int, memory_amount: float):
        """释放GPU资源"""
        with self._lock:
            if gpu_id in self.allocated_memory:
                self.allocated_memory[gpu_id] = max(0, self.allocated_memory[gpu_id] - memory_amount)
                logger.info(f"Released {memory_amount}MB from GPU {gpu_id}")
    
    def get_gpu_usage(self) -> Dict[int, Dict[str, float]]:
        """获取GPU使用情况"""
        try:
            gpus = GPUtil.getGPUs()
            return {
                gpu.id: {
                    "memory_used": gpu.memoryUsed,
                    "memory_total": gpu.memoryTotal,
                    "load": gpu.load,
                    "allocated": self.allocated_memory.get(gpu.id, 0)
                }
                for gpu in gpus
            }
        except Exception as e:
            logger.warning(f"Failed to get GPU usage: {e}")
            return {}


class MultiModelClient:
    """多模型并行客户端
    
    支持同时加载和运行多个独立的模型实例：
    - 多模型实例：每个模型运行在独立的端口和进程中
    - GPU资源管理：智能分配GPU资源给不同模型
    - 负载均衡：根据模型性能和GPU使用情况分发请求
    - 健康监控：监控每个模型实例的健康状态和性能
    
    提供统一的API接口，支持高效的并发处理。
    """

    def __init__(self):
        # 基础配置
        self.temperature = config.get("llm.multi_model.temperature", 0.7)
        self.max_tokens = config.get("llm.multi_model.max_tokens", 4096)
        self.timeout = config.get("llm.multi_model.timeout", 60)
        self.max_retries = config.get("llm.multi_model.max_retries", 3)
        
        # 多模型配置
        self.model_instances: List[ModelInstance] = []
        self.gpu_manager = GPUResourceManager()
        
        # 负载均衡配置
        strategy_str = config.get("llm.multi_model.load_balancing", "gpu_optimal")
        self.load_balancing_strategy = LoadBalancingStrategy(strategy_str)
        self._round_robin_index = 0
        self._lock = threading.Lock()
        
        # 健康检查配置
        self.health_check_interval = config.get("llm.multi_model.health_check_interval", 30)
        self.health_check_thread = None
        
        # 默认选项
        self.default_options = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        
        # 初始化模型实例
        self._init_model_instances()
        
        # 启动健康检查
        self._start_health_check_thread()
        
        logger.info(f"Initialized MultiModelClient with {len(self.model_instances)} model instances")
    
    def _init_model_instances(self):
        """初始化多个模型实例
        
        利用LM Studio的自动命名机制：
        - 重复加载相同模型时，LM Studio会自动为模型分配不同名称
        - 例如：gpt-oss-20b, gpt-oss-20b:2, gpt-oss-20b:3 等
        - 所有实例共享同一个端口（1234），通过模型名称区分
        """
        models_config = config.get("llm.multi_model.instances", [])
        base_port = config.get("llm.multi_model.base_port", 1234)
        base_url = f"http://localhost:{base_port}/v1"
        
        if not models_config:
            # 如果没有配置，创建默认的多实例配置
            model_count = config.get("llm.multi_model.instance_count", 2)
            base_model_name = config.get("llm.multi_model.model_name", "gpt-oss-20b")
            
            for i in range(model_count):
                # 第一个实例使用原始模型名，后续实例使用LM Studio的自动命名
                if i == 0:
                    model_name = base_model_name
                else:
                    model_name = f"{base_model_name}:{i+1}"
                
                instance = ModelInstance(
                    model_name=model_name,
                    port=base_port,  # 所有实例使用同一个端口
                    base_url=base_url,
                    gpu_id=self.gpu_manager.allocate_gpu()
                )
                self.model_instances.append(instance)
        else:
            # 使用配置文件中的实例配置
            for model_config in models_config:
                port = model_config.get("port", base_port)
                instance = ModelInstance(
                    model_name=model_config["model_name"],
                    port=port,
                    base_url=model_config.get("base_url", f"http://localhost:{port}/v1"),
                    gpu_id=model_config.get("gpu_id") or self.gpu_manager.allocate_gpu()
                )
                self.model_instances.append(instance)
        
        # 检查模型实例是否已在LM Studio中加载
        self._check_and_load_model_instances()
    
    def _check_and_load_model_instances(self):
        """检查并加载模型实例
        
        检查LM Studio中是否已加载所需的模型实例：
        1. 直接测试每个配置的模型名称是否可用
        2. 如果模型可用，直接初始化客户端
        3. 如果模型不可用，标记为未加载
        """
        try:
            # 获取当前LM Studio中已加载的模型列表（仅用于日志）
            available_models = self._get_available_models()
            logger.info(f"Available models in LM Studio: {available_models}")
            
            for instance in self.model_instances:
                # 直接测试模型是否可用，而不依赖于/v1/models接口的返回
                if self._test_model_availability(instance.model_name):
                    logger.info(f"Model {instance.model_name} is available and responding")
                    instance._init_client()
                    instance.is_healthy = True
                else:
                    logger.warning(f"Model {instance.model_name} is not available or not responding")
                    instance.is_healthy = False
            
            # 检查是否有足够的健康实例
            healthy_count = sum(1 for inst in self.model_instances if inst.is_healthy)
            if healthy_count == 0:
                logger.error("No healthy model instances available. Please load models in LM Studio manually.")
                self._print_loading_instructions()
            elif healthy_count < len(self.model_instances):
                logger.warning(f"Only {healthy_count}/{len(self.model_instances)} model instances are healthy")
                self._print_loading_instructions()
            else:
                logger.info(f"All {healthy_count} model instances are ready")
                
        except Exception as e:
            logger.error(f"Failed to check model instances: {e}")
            # 如果无法连接到LM Studio，尝试初始化所有实例
            for instance in self.model_instances:
                instance._init_client()
    
    def _get_available_models(self) -> List[str]:
        """获取LM Studio中可用的模型列表"""
        try:
            base_port = config.get("llm.multi_model.base_port", 1234)
            response = requests.get(f"http://localhost:{base_port}/v1/models", timeout=10)
            if response.status_code == 200:
                models_data = response.json()
                models = [model["id"] for model in models_data.get("data", [])]
                logger.info(f"在端口 {base_port} 找到 {len(models)} 个模型: {models}")
                return models
            else:
                logger.warning(f"Failed to get models list: HTTP {response.status_code}")
                return []
        except Exception as e:
            logger.warning(f"Failed to connect to LM Studio: {e}")
            return []
    
    def _test_model_availability(self, model_name: str) -> bool:
        """直接测试模型是否可用
        
        通过发送一个简单的请求来测试模型是否真正可用
        """
        try:
            base_port = config.get("llm.multi_model.base_port", 1234)
            url = f"http://localhost:{base_port}/v1/chat/completions"
            
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 1,
                "temperature": 0
            }
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                logger.debug(f"Model {model_name} test successful")
                return True
            else:
                logger.debug(f"Model {model_name} test failed with status {response.status_code}")
                return False
                
        except Exception as e:
            logger.debug(f"Model {model_name} test failed with exception: {e}")
            return False
    
    def _print_loading_instructions(self):
        """打印模型加载指导"""
        base_model_name = config.get("llm.multi_model.model_name", "gpt-oss-20b")
        model_count = len(self.model_instances)
        
        logger.info("\n" + "="*60)
        logger.info("多模型加载指导:")
        logger.info(f"需要启动 {model_count} 个独立的LM Studio实例")
        logger.info("每个实例运行在不同的端口上:")
        for i, instance in enumerate(self.model_instances):
            status = "✓" if instance.is_healthy else "✗"
            logger.info(f"  {status} 实例 {i+1}: 端口 {instance.port} - 模型 {instance.model_name}")
        logger.info("\n加载步骤:")
        logger.info("1. 启动第一个LM Studio实例 (默认端口1234)")
        logger.info(f"   - 加载模型: {base_model_name}")
        for i in range(1, model_count):
            port = self.model_instances[i].port
            logger.info(f"2. 启动第{i+1}个LM Studio实例 (端口{port})")
            logger.info(f"   - 在LM Studio设置中修改服务器端口为 {port}")
            logger.info(f"   - 加载模型: {base_model_name}")
        logger.info(f"{model_count+1}. 确保所有模型实例都已启动并可访问")
        logger.info(f"{model_count+2}. 重新运行程序")
        logger.info("="*60 + "\n")
    

    
    def _start_health_check_thread(self):
        """启动健康检查线程"""
        if self.health_check_thread is None or not self.health_check_thread.is_alive():
            self.health_check_thread = threading.Thread(
                target=self._health_check_loop,
                daemon=True
            )
            self.health_check_thread.start()
            logger.info("Health check thread started")
    
    def _health_check_loop(self):
        """健康检查循环"""
        while True:
            try:
                self._perform_health_checks()
                time.sleep(self.health_check_interval)
            except Exception as e:
                logger.error(f"Health check error: {e}")
                time.sleep(5)
    
    def _perform_health_checks(self):
        """执行健康检查"""
        current_time = time.time()
        
        for instance in self.model_instances:
            if instance.is_loading:
                continue
                
            try:
                # 检查模型实例是否响应
                start_time = time.time()
                response = requests.get(f"{instance.base_url}/models", timeout=10)
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    instance.is_healthy = True
                    # 更新平均响应时间
                    if instance.avg_response_time == 0:
                        instance.avg_response_time = response_time
                    else:
                        instance.avg_response_time = (instance.avg_response_time * 0.8 + response_time * 0.2)
                else:
                    instance.is_healthy = False
                    
            except Exception as e:
                instance.is_healthy = False
                logger.warning(f"Health check failed for {instance.model_name}: {e}")
            
            instance.last_health_check = current_time
            
            # 更新GPU使用情况
            if instance.gpu_id is not None:
                gpu_usage = self.gpu_manager.get_gpu_usage()
                if instance.gpu_id in gpu_usage:
                    instance.gpu_memory_usage = gpu_usage[instance.gpu_id]["memory_used"]
    
    def _select_instance(self) -> Optional[ModelInstance]:
        """根据负载均衡策略选择实例"""
        healthy_instances = [inst for inst in self.model_instances if inst.is_healthy and not inst.is_loading]
        
        if not healthy_instances:
            logger.warning("No healthy model instances available")
            return None
        
        if self.load_balancing_strategy == LoadBalancingStrategy.ROUND_ROBIN:
            with self._lock:
                instance = healthy_instances[self._round_robin_index % len(healthy_instances)]
                self._round_robin_index += 1
                return instance
        
        elif self.load_balancing_strategy == LoadBalancingStrategy.RANDOM:
            return random.choice(healthy_instances)
        
        elif self.load_balancing_strategy == LoadBalancingStrategy.LEAST_BUSY:
            return min(healthy_instances, key=lambda x: x.active_requests)
        
        elif self.load_balancing_strategy == LoadBalancingStrategy.GPU_OPTIMAL:
            # 选择GPU使用率最低的实例，如果相同则选择活跃请求最少的
            def gpu_score(instance):
                return (instance.gpu_memory_usage, instance.active_requests, instance.total_requests)
            return min(healthy_instances, key=gpu_score)
        
        elif self.load_balancing_strategy == LoadBalancingStrategy.PERFORMANCE_BASED:
            # 基于响应时间和活跃请求数的综合评分
            def calculate_score(instance):
                response_penalty = instance.avg_response_time * 10
                load_penalty = instance.active_requests * 5
                return response_penalty + load_penalty
            
            return min(healthy_instances, key=calculate_score)
        
        return healthy_instances[0]
    
    def _execute_with_retry(self, func, *args, **kwargs):
        """带重试的执行函数"""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            instance = self._select_instance()
            
            if instance is None:
                raise Exception("No healthy model instances available")
            
            try:
                # 更新实例状态
                instance.active_requests += 1
                instance.total_requests += 1
                
                start_time = time.time()
                result = func(instance, *args, **kwargs)
                response_time = time.time() - start_time
                
                # 更新平均响应时间
                if instance.avg_response_time == 0:
                    instance.avg_response_time = response_time
                else:
                    instance.avg_response_time = (instance.avg_response_time * 0.9 + response_time * 0.1)
                
                return result
                
            except Exception as e:
                last_exception = e
                instance.error_count += 1
                instance.is_healthy = False  # 标记为不健康
                logger.warning(f"Request failed on model {instance.model_name} (attempt {attempt + 1}): {e}")
                
                if attempt < self.max_retries:
                    time.sleep(0.5 * (attempt + 1))  # 指数退避
            
            finally:
                instance.active_requests = max(0, instance.active_requests - 1)
        
        # 所有重试都失败
        raise last_exception or Exception("All model instances failed")
    
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> str:
        """生成文本"""
        def _generate(instance: ModelInstance, prompt: str, system_prompt: str | None, stream: bool, **kwargs):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            options = {**self.default_options, **kwargs}
            
            response = instance.client.chat.completions.create(
                model=instance.model_name,
                messages=messages,
                stream=stream,
                **options
            )
            
            if stream:
                return response
            else:
                return response.choices[0].message.content
        
        return self._execute_with_retry(_generate, prompt, system_prompt, stream, **kwargs)
    
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[str, Iterator[str]]:
        """对话接口"""
        def _chat(instance: ModelInstance, messages: list[dict[str, str]], stream: bool, **kwargs):
            options = {**self.default_options, **kwargs}
            
            response = instance.client.chat.completions.create(
                model=instance.model_name,
                messages=messages,
                stream=stream,
                **options
            )
            
            if stream:
                return response
            else:
                return response.choices[0].message.content
        
        return self._execute_with_retry(_chat, messages, stream, **kwargs)
    
    def generate_concurrent(
        self,
        prompts: List[str],
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> List[str]:
        """并发生成多个文本（使用负载均衡）"""
        if not prompts:
            return []
        
        # 使用所有健康的实例进行并发处理
        healthy_instances = [inst for inst in self.model_instances if inst.is_healthy and not inst.is_loading]
        max_workers = min(len(healthy_instances), len(prompts))
        
        if max_workers == 0:
            raise Exception("No healthy model instances available for concurrent processing")
        
        results = [None] * len(prompts)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交任务
            future_to_index = {}
            for i, prompt in enumerate(prompts):
                future = executor.submit(self.generate, prompt, system_prompt, **kwargs)
                future_to_index[future] = i
            
            # 收集结果
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    logger.error(f"Concurrent generation failed for prompt {index}: {e}")
                    results[index] = f"Error: {str(e)}"
        
        return results
    
    def generate_parallel(
        self,
        prompts: List[str],
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> List[str]:
        """真正的并行生成多个文本（直接分配到实例）
        
        与generate_concurrent不同，这个方法确保多个实例真正同时处理请求，
        而不是通过负载均衡策略选择单个实例。
        
        Args:
            prompts: 提示列表
            system_prompt: 系统提示
            **kwargs: 其他参数
            
        Returns:
            响应列表
        """
        if not prompts:
            return []
        
        # 获取健康实例
        healthy_instances = [inst for inst in self.model_instances if inst.is_healthy and not inst.is_loading]
        
        if not healthy_instances:
            raise Exception("No healthy model instances available for parallel processing")
        
        logger.info(f"🚀 开始并行处理 {len(prompts)} 个请求，使用 {len(healthy_instances)} 个实例")
        
        # 如果提示数量少于或等于实例数量，每个实例处理一个
        if len(prompts) <= len(healthy_instances):
            return self._execute_direct_parallel(prompts, healthy_instances, system_prompt, **kwargs)
        else:
            # 如果提示数量多于实例数量，需要分批处理
            return self._execute_batch_parallel(prompts, healthy_instances, system_prompt, **kwargs)
    
    def _execute_direct_parallel(self, prompts: List[str], instances: List, system_prompt: str = None, **kwargs) -> List[str]:
        """直接并行执行（提示数 <= 实例数）"""
        results = [None] * len(prompts)
        
        with ThreadPoolExecutor(max_workers=len(instances)) as executor:
            # 为每个提示分配一个实例
            future_to_index = {}
            for i, (prompt, instance) in enumerate(zip(prompts, instances)):
                future = executor.submit(self._execute_single_request, instance, prompt, system_prompt, **kwargs)
                future_to_index[future] = i
            
            # 收集结果
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    logger.error(f"Parallel request {index} failed: {e}")
                    results[index] = f"Error: {str(e)}"
        
        return results
    
    def _execute_batch_parallel(self, prompts: List[str], instances: List, system_prompt: str = None, **kwargs) -> List[str]:
        """批量并行执行（提示数 > 实例数）"""
        results = [None] * len(prompts)
        num_instances = len(instances)
        
        # 将提示分配给实例
        instance_tasks = [[] for _ in range(num_instances)]
        for i, prompt in enumerate(prompts):
            instance_idx = i % num_instances
            instance_tasks[instance_idx].append((i, prompt))
        
        with ThreadPoolExecutor(max_workers=num_instances) as executor:
            # 为每个实例提交一个批量任务
            futures = []
            for instance_idx, tasks in enumerate(instance_tasks):
                if tasks:  # 只处理有任务的实例
                    future = executor.submit(
                        self._execute_instance_batch, 
                        instances[instance_idx], 
                        tasks, 
                        system_prompt, 
                        **kwargs
                    )
                    futures.append((future, instance_idx))
            
            # 收集结果
            for future, instance_idx in futures:
                try:
                    batch_results = future.result()
                    for original_index, result in batch_results:
                        results[original_index] = result
                except Exception as e:
                    logger.error(f"Batch execution failed for instance {instance_idx}: {e}")
                    # 为该实例的所有任务设置错误结果
                    for original_index, _ in instance_tasks[instance_idx]:
                        results[original_index] = f"Error: {str(e)}"
        
        return results
    
    def _execute_instance_batch(self, instance, tasks: List[tuple], system_prompt: str = None, **kwargs) -> List[tuple]:
        """单个实例执行批量任务"""
        results = []
        for original_index, prompt in tasks:
            try:
                result = self._execute_single_request(instance, prompt, system_prompt, **kwargs)
                results.append((original_index, result))
            except Exception as e:
                logger.error(f"Single request failed in batch: {e}")
                results.append((original_index, f"Error: {str(e)}"))
        return results
    
    def _execute_single_request(self, instance, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """执行单个请求"""
        try:
            # 更新实例状态
            instance.active_requests += 1
            
            # 准备消息
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # 合并选项
            options = {**self.default_options, **kwargs}
            
            # 执行请求
            start_time = time.time()
            response = instance.client.chat.completions.create(
                model=instance.model_name,
                messages=messages,
                **options
            )
            end_time = time.time()
            
            # 更新统计信息
            instance.total_requests += 1
            response_time = end_time - start_time
            instance.avg_response_time = (
                (instance.avg_response_time * (instance.total_requests - 1) + response_time) / 
                instance.total_requests
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            instance.error_count += 1
            raise e
        finally:
            instance.active_requests -= 1
    
    def get_status(self) -> Dict[str, Any]:
        """获取多模型系统状态"""
        healthy_count = sum(1 for inst in self.model_instances if inst.is_healthy)
        loading_count = sum(1 for inst in self.model_instances if inst.is_loading)
        total_requests = sum(inst.total_requests for inst in self.model_instances)
        total_errors = sum(inst.error_count for inst in self.model_instances)
        
        return {
            "total_instances": len(self.model_instances),
            "healthy_instances": healthy_count,
            "loading_instances": loading_count,
            "total_requests": total_requests,
            "total_errors": total_errors,
            "error_rate": total_errors / max(total_requests, 1),
            "load_balancing_strategy": self.load_balancing_strategy.value,
            "gpu_usage": self.gpu_manager.get_gpu_usage(),
            "instances": [
                {
                    "model_name": inst.model_name,
                    "port": inst.port,
                    "gpu_id": inst.gpu_id,
                    "is_healthy": inst.is_healthy,
                    "is_loading": inst.is_loading,
                    "active_requests": inst.active_requests,
                    "total_requests": inst.total_requests,
                    "error_count": inst.error_count,
                    "avg_response_time": inst.avg_response_time,
                    "gpu_memory_usage": inst.gpu_memory_usage,
                }
                for inst in self.model_instances
            ]
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'provider': 'multi_model',
            'instance_count': len(self.model_instances),
            'healthy_instances': sum(1 for inst in self.model_instances if inst.is_healthy),
            'load_balancing_strategy': self.load_balancing_strategy.value,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'gpu_info': self.gpu_manager.gpu_info,
            'instances': [{
                'model_name': inst.model_name,
                'port': inst.port,
                'gpu_id': inst.gpu_id,
                'is_healthy': inst.is_healthy,
                'is_loading': inst.is_loading,
            } for inst in self.model_instances]
        }
    
    def shutdown(self):
        """关闭多模型系统"""
        logger.info("Shutting down MultiModelClient...")
        
        # 停止健康检查线程
        if self.health_check_thread and self.health_check_thread.is_alive():
            # 这里需要实现线程停止机制
            pass
        
        # 停止所有模型实例
        for instance in self.model_instances:
            try:
                if instance.process and instance.process.poll() is None:
                    instance.process.terminate()
                    instance.process.wait(timeout=10)
                
                # 释放GPU资源
                if instance.gpu_id is not None:
                    self.gpu_manager.release_gpu(instance.gpu_id, 4096)  # 假设每个模型占用4GB
                    
            except Exception as e:
                logger.error(f"Error shutting down instance {instance.model_name}: {e}")
        
        logger.info("MultiModelClient shutdown complete")


# 为了向后兼容，保留原有的类名别名
MultiLMStudioClient = MultiModelClient