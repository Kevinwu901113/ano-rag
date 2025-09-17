"""
性能监控和告警机制模块

提供并行任务处理的性能监控、统计分析和告警功能
"""

import time
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from loguru import logger
from collections import deque, defaultdict
import json


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    timestamp: datetime
    task_type: str  # 'ollama', 'lmstudio', 'fallback'
    duration: float
    success: bool
    error_type: Optional[str] = None
    retry_count: int = 0
    chunk_size: int = 0


@dataclass
class AlertThresholds:
    """告警阈值配置"""
    error_rate_threshold: float = 0.1  # 错误率阈值
    avg_duration_threshold: float = 30.0  # 平均处理时间阈值（秒）
    fallback_rate_threshold: float = 0.2  # 回退率阈值
    consecutive_failures_threshold: int = 5  # 连续失败次数阈值
    monitoring_window_minutes: int = 10  # 监控窗口（分钟）


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, thresholds: Optional[AlertThresholds] = None):
        self.thresholds = thresholds or AlertThresholds()
        self._metrics: deque = deque(maxlen=1000)  # 保留最近1000条记录
        self._lock = threading.Lock()
        
        # 统计计数器
        self._counters = defaultdict(int)
        self._durations = defaultdict(list)
        
        # 告警状态
        self._alert_callbacks: List[Callable] = []
        self._last_alert_time = defaultdict(lambda: datetime.min)
        self._alert_cooldown = timedelta(minutes=5)  # 告警冷却时间
        
        # 连续失败跟踪
        self._consecutive_failures = defaultdict(int)
        
        logger.info("Performance monitor initialized")
    
    def record_task(self, task_type: str, duration: float, success: bool, 
                   error_type: Optional[str] = None, retry_count: int = 0, 
                   chunk_size: int = 0):
        """记录任务执行情况"""
        metric = PerformanceMetrics(
            timestamp=datetime.now(),
            task_type=task_type,
            duration=duration,
            success=success,
            error_type=error_type,
            retry_count=retry_count,
            chunk_size=chunk_size
        )
        
        with self._lock:
            self._metrics.append(metric)
            
            # 更新计数器
            self._counters[f"{task_type}_total"] += 1
            if success:
                self._counters[f"{task_type}_success"] += 1
                self._consecutive_failures[task_type] = 0
            else:
                self._counters[f"{task_type}_error"] += 1
                self._consecutive_failures[task_type] += 1
            
            if retry_count > 0:
                self._counters[f"{task_type}_retry"] += retry_count
            
            # 记录处理时间
            self._durations[task_type].append(duration)
            if len(self._durations[task_type]) > 100:  # 保留最近100次的时间记录
                self._durations[task_type].pop(0)
        
        # 检查告警条件
        self._check_alerts(task_type)
    
    def record_fallback(self, from_client: str, to_client: str, success: bool):
        """记录回退操作"""
        with self._lock:
            self._counters[f"fallback_{from_client}_to_{to_client}"] += 1
            if success:
                self._counters[f"fallback_{from_client}_to_{to_client}_success"] += 1
            else:
                self._counters[f"fallback_{from_client}_to_{to_client}_error"] += 1
        
        # 检查回退率告警
        self._check_fallback_rate_alert()
    
    def get_current_stats(self) -> Dict[str, Any]:
        """获取当前统计信息"""
        with self._lock:
            now = datetime.now()
            window_start = now - timedelta(minutes=self.thresholds.monitoring_window_minutes)
            
            # 过滤窗口内的指标
            recent_metrics = [m for m in self._metrics if m.timestamp >= window_start]
            
            stats = {
                'timestamp': now.isoformat(),
                'monitoring_window_minutes': self.thresholds.monitoring_window_minutes,
                'total_tasks': len(recent_metrics),
                'counters': dict(self._counters),
                'task_types': {}
            }
            
            # 按任务类型统计
            for task_type in ['ollama', 'lmstudio', 'fallback']:
                type_metrics = [m for m in recent_metrics if m.task_type == task_type]
                if type_metrics:
                    success_count = sum(1 for m in type_metrics if m.success)
                    error_count = len(type_metrics) - success_count
                    durations = [m.duration for m in type_metrics]
                    
                    stats['task_types'][task_type] = {
                        'total': len(type_metrics),
                        'success': success_count,
                        'error': error_count,
                        'error_rate': error_count / len(type_metrics) if type_metrics else 0,
                        'avg_duration': sum(durations) / len(durations) if durations else 0,
                        'min_duration': min(durations) if durations else 0,
                        'max_duration': max(durations) if durations else 0,
                        'consecutive_failures': self._consecutive_failures[task_type]
                    }
            
            # 计算回退率
            total_tasks = stats['total_tasks']
            fallback_count = sum(v for k, v in self._counters.items() if k.startswith('fallback_') and not k.endswith('_success') and not k.endswith('_error'))
            stats['fallback_rate'] = fallback_count / total_tasks if total_tasks > 0 else 0
            
            return stats
    
    def add_alert_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """添加告警回调函数"""
        self._alert_callbacks.append(callback)
    
    def _check_alerts(self, task_type: str):
        """检查告警条件"""
        now = datetime.now()
        
        # 检查连续失败告警
        if self._consecutive_failures[task_type] >= self.thresholds.consecutive_failures_threshold:
            self._trigger_alert(
                f"consecutive_failures_{task_type}",
                f"{task_type} has {self._consecutive_failures[task_type]} consecutive failures",
                {
                    'task_type': task_type,
                    'consecutive_failures': self._consecutive_failures[task_type],
                    'threshold': self.thresholds.consecutive_failures_threshold
                }
            )
        
        # 检查错误率告警（需要足够的样本）
        if self._counters[f"{task_type}_total"] >= 10:
            error_rate = self._counters[f"{task_type}_error"] / self._counters[f"{task_type}_total"]
            if error_rate > self.thresholds.error_rate_threshold:
                self._trigger_alert(
                    f"error_rate_{task_type}",
                    f"{task_type} error rate {error_rate:.2%} exceeds threshold {self.thresholds.error_rate_threshold:.2%}",
                    {
                        'task_type': task_type,
                        'error_rate': error_rate,
                        'threshold': self.thresholds.error_rate_threshold,
                        'total_tasks': self._counters[f"{task_type}_total"],
                        'error_count': self._counters[f"{task_type}_error"]
                    }
                )
        
        # 检查平均处理时间告警
        if task_type in self._durations and len(self._durations[task_type]) >= 5:
            avg_duration = sum(self._durations[task_type]) / len(self._durations[task_type])
            if avg_duration > self.thresholds.avg_duration_threshold:
                self._trigger_alert(
                    f"avg_duration_{task_type}",
                    f"{task_type} average duration {avg_duration:.2f}s exceeds threshold {self.thresholds.avg_duration_threshold}s",
                    {
                        'task_type': task_type,
                        'avg_duration': avg_duration,
                        'threshold': self.thresholds.avg_duration_threshold,
                        'sample_count': len(self._durations[task_type])
                    }
                )
    
    def _check_fallback_rate_alert(self):
        """检查回退率告警"""
        stats = self.get_current_stats()
        if stats['total_tasks'] >= 10 and stats['fallback_rate'] > self.thresholds.fallback_rate_threshold:
            self._trigger_alert(
                "fallback_rate",
                f"Fallback rate {stats['fallback_rate']:.2%} exceeds threshold {self.thresholds.fallback_rate_threshold:.2%}",
                {
                    'fallback_rate': stats['fallback_rate'],
                    'threshold': self.thresholds.fallback_rate_threshold,
                    'total_tasks': stats['total_tasks']
                }
            )
    
    def _trigger_alert(self, alert_type: str, message: str, data: Dict[str, Any]):
        """触发告警"""
        now = datetime.now()
        
        # 检查冷却时间
        if now - self._last_alert_time[alert_type] < self._alert_cooldown:
            return
        
        self._last_alert_time[alert_type] = now
        
        alert_data = {
            'type': alert_type,
            'message': message,
            'timestamp': now.isoformat(),
            'data': data
        }
        
        logger.warning(f"Performance Alert: {message}")
        
        # 调用告警回调
        for callback in self._alert_callbacks:
            try:
                callback(alert_type, alert_data)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def export_metrics(self, format: str = 'json') -> str:
        """导出指标数据"""
        stats = self.get_current_stats()
        
        if format == 'json':
            return json.dumps(stats, indent=2, ensure_ascii=False)
        elif format == 'csv':
            # 简化的CSV格式
            lines = ['timestamp,task_type,duration,success,error_type']
            with self._lock:
                for metric in self._metrics:
                    lines.append(f"{metric.timestamp.isoformat()},{metric.task_type},{metric.duration},{metric.success},{metric.error_type or ''}")
            return '\n'.join(lines)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def reset_stats(self):
        """重置统计信息"""
        with self._lock:
            self._metrics.clear()
            self._counters.clear()
            self._durations.clear()
            self._consecutive_failures.clear()
        logger.info("Performance monitor stats reset")


# 全局监控器实例
_global_monitor: Optional[PerformanceMonitor] = None


def get_monitor() -> PerformanceMonitor:
    """获取全局监控器实例"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor


def setup_monitor(thresholds: Optional[AlertThresholds] = None) -> PerformanceMonitor:
    """设置全局监控器"""
    global _global_monitor
    _global_monitor = PerformanceMonitor(thresholds)
    return _global_monitor


# 便捷函数
def record_task_performance(task_type: str, duration: float, success: bool, 
                          error_type: Optional[str] = None, retry_count: int = 0, 
                          chunk_size: int = 0):
    """记录任务性能（便捷函数）"""
    get_monitor().record_task(task_type, duration, success, error_type, retry_count, chunk_size)


def record_fallback_operation(from_client: str, to_client: str, success: bool):
    """记录回退操作（便捷函数）"""
    get_monitor().record_fallback(from_client, to_client, success)


def get_performance_stats() -> Dict[str, Any]:
    """获取性能统计（便捷函数）"""
    return get_monitor().get_current_stats()


def add_performance_alert_callback(callback: Callable[[str, Dict[str, Any]], None]):
    """添加性能告警回调（便捷函数）"""
    get_monitor().add_alert_callback(callback)


def reset_monitor():
    """重置全局监控器实例（便捷函数）"""
    get_monitor().reset_stats()