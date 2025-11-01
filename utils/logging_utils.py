from loguru import logger
import os
import sys
import json
import time
from typing import Dict, Any, Optional, Callable
from functools import wraps
from contextlib import contextmanager
from datetime import datetime


class StructuredLogger:
    """结构化日志记录器，提供统一的日志格式和性能监控"""
    
    def __init__(self, component_name: str = "RAG_System"):
        self.component_name = component_name
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def log_structured(self, level: str, message: str, **kwargs):
        """记录结构化日志"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "component": self.component_name,
            "session_id": self.session_id,
            "message": message,
            **kwargs
        }
        
        log_message = f"{message} | {json.dumps(log_data, ensure_ascii=False)}"
        getattr(logger, level.lower())(log_message)
    
    def info(self, message: str, **kwargs):
        self.log_structured("INFO", message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        self.log_structured("DEBUG", message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        self.log_structured("WARNING", message, **kwargs)
    
    def error(self, message: str, **kwargs):
        self.log_structured("ERROR", message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        self.log_structured("CRITICAL", message, **kwargs)


def setup_logging(log_file: str, log_level: str = "INFO", enable_json: bool = False):
    """配置增强的loguru日志系统"""
    # 从环境变量读取日志级别，优先级高于参数
    env_log_level = os.environ.get('LOGURU_LEVEL')
    if env_log_level:
        log_level = env_log_level.upper()
    
    os.makedirs(os.path.dirname(log_file) or '.', exist_ok=True)
    logger.remove()
    
    # 控制台输出格式
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    
    # 文件输出格式
    if enable_json:
        file_format = (
            '{"timestamp": "{time:YYYY-MM-DD HH:mm:ss}", '
            '"level": "{level}", '
            '"module": "{name}", '
            '"function": "{function}", '
            '"line": {line}, '
            '"message": "{message}"}'
        )
    else:
        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
            "{name}:{function}:{line} | {message}"
        )
    
    # 添加控制台处理器
    logger.add(
        sys.stderr,
        level=log_level,
        format=console_format,
        colorize=True
    )
    
    def _add_file_sink(path: str, *, level: str, rotation: str, retention: str, compression: Optional[str] = None):
        sink_kwargs = dict(
            level=level,
            format=file_format,
            rotation=rotation,
            retention=retention,
            compression=compression,
            encoding="utf-8",
            enqueue=True,
            backtrace=True,
            diagnose=True,
        )
        try:
            logger.add(path, **sink_kwargs)
        except (PermissionError, OSError) as exc:
            sink_kwargs["enqueue"] = False
            logger.warning(
                f"Failed to enable async logging for {path} ({exc}); falling back to synchronous writes."
            )
            logger.add(path, **sink_kwargs)
    
    # 添加文件处理器
    _add_file_sink(
        log_file,
        level=log_level,
        rotation="50 MB",
        retention="30 days",
        compression="zip",
    )
    
    # 添加错误文件处理器
    error_file = log_file.replace('.log', '_error.log')
    _add_file_sink(
        error_file,
        level="ERROR",
        rotation="10 MB",
        retention="60 days",
        compression=None,
    )
    
    return logger


def log_performance(func_name: str = None):
    """性能监控装饰器"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            function_name = func_name or f"{func.__module__}.{func.__name__}"
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                logger.info(
                    f"Performance: {function_name} completed",
                    extra={
                        "execution_time": f"{execution_time:.3f}s",
                        "function": function_name,
                        "status": "success"
                    }
                )
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(
                    f"Performance: {function_name} failed",
                    extra={
                        "execution_time": f"{execution_time:.3f}s",
                        "function": function_name,
                        "status": "error",
                        "error": str(e)
                    }
                )
                raise
        
        return wrapper
    return decorator


@contextmanager
def log_operation(operation_name: str, **context):
    """操作日志上下文管理器"""
    start_time = time.time()
    logger.info(f"Starting operation: {operation_name}", extra=context)
    
    try:
        yield
        execution_time = time.time() - start_time
        logger.info(
            f"Operation completed: {operation_name}",
            extra={**context, "execution_time": f"{execution_time:.3f}s", "status": "success"}
        )
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(
            f"Operation failed: {operation_name}",
            extra={**context, "execution_time": f"{execution_time:.3f}s", "status": "error", "error": str(e)}
        )
        raise


def log_retrieval_metrics(query: str, results_count: int, **metrics):
    """记录检索指标"""
    logger.info(
        f"Retrieval metrics for query",
        extra={
            "query_length": len(query),
            "results_count": results_count,
            "query_hash": hash(query) % 10000,
            **metrics
        }
    )


def log_diversity_metrics(candidates_count: int, selected_count: int, diversity_score: float, **metrics):
    """记录多样性调度指标"""
    logger.info(
        f"Diversity scheduling metrics",
        extra={
            "candidates_count": candidates_count,
            "selected_count": selected_count,
            "diversity_score": f"{diversity_score:.3f}",
            "selection_ratio": f"{selected_count/max(candidates_count,1):.3f}",
            **metrics
        }
    )


def log_path_aware_metrics(candidates_count: int, path_enhanced_count: int, avg_path_score: float, **metrics):
    """记录路径感知排序指标"""
    logger.info(
        f"PathAware ranking metrics",
        extra={
            "candidates_count": candidates_count,
            "path_enhanced_count": path_enhanced_count,
            "avg_path_score": f"{avg_path_score:.3f}",
            "enhancement_ratio": f"{path_enhanced_count/max(candidates_count,1):.3f}",
            **metrics
        }
    )
