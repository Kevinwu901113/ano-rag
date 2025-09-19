"""
LLM路由器实现
支持lmstudio + ollama并发分流，包含加权轮询、熔断机制、并发控制等功能
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from typing import Dict, Any, Callable, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class BackendConfig:
    """后端配置"""
    name: str
    weight: int
    max_inflight: int
    timeout_s: int


@dataclass
class BreakerConfig:
    """熔断器配置"""
    fail_threshold: int
    cool_down_s: int


class CircuitBreaker:
    """熔断器实现"""
    
    def __init__(self, fail_threshold: int, cool_down_s: int):
        self.fail_threshold = fail_threshold
        self.cool_down_s = cool_down_s
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.circuit_break_count = 0  # 熔断次数统计
        self._lock = threading.Lock()
    
    def can_execute(self) -> bool:
        """检查是否可以执行请求"""
        with self._lock:
            if self.state == "CLOSED":
                return True
            elif self.state == "OPEN":
                if time.time() - self.last_failure_time >= self.cool_down_s:
                    self.state = "HALF_OPEN"
                    return True
                return False
            else:  # HALF_OPEN
                return True
    
    def record_success(self):
        """记录成功"""
        with self._lock:
            self.failure_count = 0
            self.state = "CLOSED"
    
    def record_failure(self):
        """记录失败"""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.fail_threshold:
                if self.state != "OPEN":  # 只在状态变化时计数
                    self.circuit_break_count += 1
                self.state = "OPEN"


class BackendManager:
    """后端管理器"""
    
    def __init__(self, config: BackendConfig, call_func: Callable[[str], str], breaker_config: BreakerConfig):
        self.config = config
        self.call_func = call_func
        self.breaker = CircuitBreaker(breaker_config.fail_threshold, breaker_config.cool_down_s)
        self.semaphore = threading.Semaphore(config.max_inflight)
        self.active_requests = 0
        self.total_requests = 0
        self.failed_requests = 0  # 失败请求计数
        self._lock = threading.Lock()
    
    def is_available(self) -> bool:
        """检查后端是否可用"""
        return self.breaker.can_execute() and self.active_requests < self.config.max_inflight
    
    def execute(self, prompt: str) -> str:
        """执行请求"""
        if not self.breaker.can_execute():
            raise RuntimeError(f"Backend {self.config.name} is circuit broken")
        
        # 获取信号量
        if not self.semaphore.acquire(blocking=False):
            raise RuntimeError(f"Backend {self.config.name} is at max capacity")
        
        try:
            with self._lock:
                self.active_requests += 1
                self.total_requests += 1
            
            # 用原生线程 + join 超时，避免每次建池
            res, err = {}, {}
            def _run():
                try:
                    res["v"] = self.call_func(prompt)
                except Exception as e:
                    err["e"] = e
            
            t = threading.Thread(target=_run, daemon=True)
            t.start()
            t.join(self.config.timeout_s)
            
            if t.is_alive():
                with self._lock:
                    self.failed_requests += 1
                self.breaker.record_failure()
                raise RuntimeError(f"Backend {self.config.name} timeout after {self.config.timeout_s}s")
            if "e" in err:
                with self._lock:
                    self.failed_requests += 1
                self.breaker.record_failure()
                raise RuntimeError(f"Backend {self.config.name} failed: {err['e']}")
            
            self.breaker.record_success()
            return res.get("v", "")
        finally:
            with self._lock:
                self.active_requests -= 1
            self.semaphore.release()


class LLMRouter:
    """LLM路由器
    
    支持加权轮询选择后端，尊重每端max_inflight限制，
    失败/超时按breaker熔断并回退到其他端
    """
    
    def __init__(self, pool_cfg: Dict[str, Any], lmstudio_call: Callable[[str], str], ollama_call: Optional[Callable[[str], str]] = None):
        """初始化路由器
        
        Args:
            pool_cfg: llm_pool配置，包含backends和breaker
            lmstudio_call: lmstudio调用函数
            ollama_call: ollama调用函数（可选）
        """
        self.backends = {}
        self.backend_weights = []
        self.current_index = 0
        self._lock = threading.Lock()
        
        # 统计信息
        self.stats = {
            "calls": 0,
            "failed_calls": 0,  # 失败调用计数
            "by_backend": {}
        }
        
        # 解析breaker配置
        breaker_cfg = BreakerConfig(
            fail_threshold=pool_cfg["breaker"]["fail_threshold"],
            cool_down_s=pool_cfg["breaker"]["cool_down_s"]
        )
        
        # 初始化后端
        for backend_cfg in pool_cfg["backends"]:
            backend_config = BackendConfig(
                name=backend_cfg["name"],
                weight=backend_cfg["weight"],
                max_inflight=backend_cfg["max_inflight"],
                timeout_s=backend_cfg["timeout_s"]
            )
            
            # 根据后端名称选择调用函数
            if backend_config.name == "lmstudio":
                call_func = lmstudio_call
            elif backend_config.name == "ollama" and ollama_call is not None:
                call_func = ollama_call
            else:
                logger.warning(f"Skipping backend {backend_config.name}: no call function provided")
                continue
            
            # 创建后端管理器
            manager = BackendManager(backend_config, call_func, breaker_cfg)
            self.backends[backend_config.name] = manager
            
            # 根据权重添加到轮询列表
            for _ in range(backend_config.weight):
                self.backend_weights.append(backend_config.name)
            
            # 初始化统计
            self.stats["by_backend"][backend_config.name] = {
                "success": 0,
                "failed": 0,
                "circuit_breaks": 0
            }
        
        if not self.backends:
            raise ValueError("No valid backends configured")
        
        logger.info(f"LLMRouter initialized with backends: {list(self.backends.keys())}")
    
    def _select_backend(self) -> Optional[str]:
        """加权轮询选择后端"""
        if not self.backend_weights:
            return None
        
        with self._lock:
            # 尝试找到可用的后端，最多尝试所有后端一轮
            attempts = 0
            max_attempts = len(self.backend_weights)
            
            while attempts < max_attempts:
                backend_name = self.backend_weights[self.current_index]
                self.current_index = (self.current_index + 1) % len(self.backend_weights)
                attempts += 1
                
                if backend_name in self.backends and self.backends[backend_name].is_available():
                    return backend_name
            
            return None
    
    def call(self, prompt: str) -> str:
        """调用LLM生成响应
        
        Args:
            prompt: 输入提示
            
        Returns:
            str: 生成的响应
            
        Raises:
            RuntimeError: 所有后端都不可用时
        """
        with self._lock:
            self.stats["calls"] += 1
        
        # 尝试所有后端
        tried_backends = set()
        
        while len(tried_backends) < len(self.backends):
            backend_name = self._select_backend()
            
            if backend_name is None or backend_name in tried_backends:
                # 如果没有可用后端或已经尝试过，尝试其他后端
                for name in self.backends:
                    if name not in tried_backends and self.backends[name].is_available():
                        backend_name = name
                        break
                else:
                    break
            
            tried_backends.add(backend_name)
            
            try:
                result = self.backends[backend_name].execute(prompt)
                
                # 更新统计
                with self._lock:
                    self.stats["by_backend"][backend_name]["success"] += 1
                
                logger.debug(f"Request completed by backend: {backend_name}")
                return result
                
            except Exception as e:
                # 更新失败统计
                with self._lock:
                    self.stats["by_backend"][backend_name]["failed"] += 1
                    # 检查是否是熔断导致的失败
                    if "circuit broken" in str(e):
                        self.stats["by_backend"][backend_name]["circuit_breaks"] = self.backends[backend_name].breaker.circuit_break_count
                
                logger.warning(f"Backend {backend_name} failed: {str(e)}")
                continue
        
        # 所有后端都失败
        with self._lock:
            self.stats["failed_calls"] += 1
        raise RuntimeError("All backends are unavailable")
    
    def route(self, prompt: str) -> str:
        """路由方法的别名，与 call 方法相同"""
        return self.call(prompt)


def build_router_from_config(cfg: Dict[str, Any], lmstudio_call: Callable[[str], str], ollama_call: Optional[Callable[[str], str]] = None) -> LLMRouter:
    """从配置构建路由器
    
    Args:
        cfg: 完整配置字典
        lmstudio_call: lmstudio调用函数
        ollama_call: ollama调用函数（可选）
        
    Returns:
        LLMRouter: 配置好的路由器实例
    """
    pool_cfg = cfg.get("llm_pool", {})
    if not pool_cfg:
        raise ValueError("llm_pool configuration not found")
    
    return LLMRouter(pool_cfg, lmstudio_call, ollama_call)


# 自测代码片段
"""
自测示例：

# 模拟调用函数
def mock_lmstudio_call(prompt: str) -> str:
    import time
    time.sleep(0.1)  # 模拟处理时间
    return f"LMStudio response to: {prompt[:50]}..."

def mock_ollama_call(prompt: str) -> str:
    import time
    time.sleep(0.1)  # 模拟处理时间
    return f"Ollama response to: {prompt[:50]}..."

# 测试配置
test_config = {
    "llm_pool": {
        "backends": [
            {"name": "lmstudio", "weight": 3, "max_inflight": 4, "timeout_s": 30},
            {"name": "ollama", "weight": 1, "max_inflight": 2, "timeout_s": 30}
        ],
        "breaker": {
            "fail_threshold": 3,
            "cool_down_s": 60
        }
    }
}

# 测试1：只有lmstudio后端
single_backend_config = {
    "llm_pool": {
        "backends": [
            {"name": "lmstudio", "weight": 1, "max_inflight": 2, "timeout_s": 30}
        ],
        "breaker": {
            "fail_threshold": 3,
            "cool_down_s": 60
        }
    }
}

# 单后端测试
router1 = build_router_from_config(single_backend_config, mock_lmstudio_call)
result1 = router1.call("Test prompt")
print(f"Single backend result: {result1}")
print(f"Single backend stats: {router1.stats}")
# 预期：stats["by_backend"]["lmstudio"] > 0, 没有ollama

# 双后端测试
router2 = build_router_from_config(test_config, mock_lmstudio_call, mock_ollama_call)
for i in range(10):
    result = router2.call(f"Test prompt {i}")
    print(f"Request {i}: {result}")

print(f"Dual backend stats: {router2.stats}")
# 预期：stats["by_backend"]中lmstudio和ollama都有非零值，且lmstudio约为ollama的3倍（权重比）
"""