"""
原子笔记统计与监控日志记录器
用于记录和导出原子笔记生成过程的统计信息
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
from loguru import logger
from config import config


class NotesStatsLogger:
    """原子笔记统计日志记录器"""
    
    def __init__(self, log_file: Optional[str] = None):
        """
        初始化统计日志记录器
        
        Args:
            log_file: 日志文件路径，默认为notes_stats.json
        """
        self.log_file = log_file or "notes_stats.json"
        self.session_stats = {
            'session_id': self._generate_session_id(),
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'total_processing_time': 0.0,
            'notes_zero_count': 0,
            'sentinel_rate': 0.0,
            'parse_fail_rate': 0.0,
            'rule_fallback_rate': 0.0,
            'avg_notes_per_chunk': 0.0,
            'total_chunks_processed': 0,
            'total_notes_generated': 0,
            'quality_filter_stats': {},
            'retry_handler_stats': {},
            'performance_metrics': {}
        }
        
        # 累积统计信息
        self.cumulative_stats = self._load_cumulative_stats()
    
    def _generate_session_id(self) -> str:
        """生成会话ID"""
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def _load_cumulative_stats(self) -> Dict[str, Any]:
        """加载累积统计信息"""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get('cumulative_stats', {})
            except Exception as e:
                logger.warning(f"Failed to load cumulative stats: {e}")
        
        return {
            'total_sessions': 0,
            'total_chunks_processed': 0,
            'total_notes_generated': 0,
            'average_sentinel_rate': 0.0,
            'average_parse_fail_rate': 0.0,
            'average_rule_fallback_rate': 0.0,
            'last_updated': None
        }
    
    def update_session_stats(self, stats: Dict[str, Any]):
        """更新会话统计信息"""
        for key, value in stats.items():
            if key in self.session_stats:
                self.session_stats[key] = value
        
        logger.debug(f"Updated session stats: {stats}")
    
    def add_quality_filter_stats(self, filter_stats: Dict[str, Any]):
        """添加质量过滤统计信息"""
        if 'quality_filter_stats' not in self.session_stats:
            self.session_stats['quality_filter_stats'] = {}
        
        # 累积过滤统计
        for key, value in filter_stats.items():
            if key in self.session_stats['quality_filter_stats']:
                self.session_stats['quality_filter_stats'][key] += value
            else:
                self.session_stats['quality_filter_stats'][key] = value
    
    def add_retry_handler_stats(self, retry_stats: Dict[str, Any]):
        """添加重试处理器统计信息"""
        if 'retry_handler_stats' not in self.session_stats:
            self.session_stats['retry_handler_stats'] = {}
        
        # 累积重试统计
        for key, value in retry_stats.items():
            if key in self.session_stats['retry_handler_stats']:
                if isinstance(value, (int, float)):
                    self.session_stats['retry_handler_stats'][key] += value
                else:
                    self.session_stats['retry_handler_stats'][key] = value
            else:
                self.session_stats['retry_handler_stats'][key] = value
    
    def add_performance_metrics(self, metrics: Dict[str, Any]):
        """添加性能指标"""
        self.session_stats['performance_metrics'].update(metrics)
    
    def finalize_session(self, processing_time: float = 0.0):
        """结束会话并计算最终统计"""
        self.session_stats['end_time'] = datetime.now().isoformat()
        self.session_stats['total_processing_time'] = processing_time
        
        # 计算比率
        if self.session_stats['total_chunks_processed'] > 0:
            self.session_stats['sentinel_rate'] = (
                self.session_stats['notes_zero_count'] / 
                self.session_stats['total_chunks_processed']
            )
            
            # 计算平均每块笔记数
            self.session_stats['avg_notes_per_chunk'] = (
                self.session_stats['total_notes_generated'] / 
                self.session_stats['total_chunks_processed']
            )
            
            # 从重试统计中计算解析失败率
            retry_stats = self.session_stats.get('retry_handler_stats', {})
            total_attempts = retry_stats.get('total_attempts', 0)
            final_failures = retry_stats.get('final_failures', 0)
            
            if total_attempts > 0:
                self.session_stats['parse_fail_rate'] = final_failures / total_attempts
            
            fallback_used = retry_stats.get('fallback_used', 0)
            if total_attempts > 0:
                self.session_stats['rule_fallback_rate'] = fallback_used / total_attempts
        
        logger.info(f"Session {self.session_stats['session_id']} finalized")
    
    def export_stats(self, export_format: str = 'json') -> str:
        """
        导出统计信息
        
        Args:
            export_format: 导出格式，支持 'json', 'yaml', 'txt'
            
        Returns:
            str: 导出的文件路径
        """
        # 更新累积统计
        self._update_cumulative_stats()
        
        # 准备导出数据
        export_data = {
            'session_stats': self.session_stats,
            'cumulative_stats': self.cumulative_stats,
            'export_time': datetime.now().isoformat()
        }
        
        if export_format.lower() == 'json':
            return self._export_json(export_data)
        elif export_format.lower() == 'yaml':
            return self._export_yaml(export_data)
        elif export_format.lower() == 'txt':
            return self._export_txt(export_data)
        else:
            logger.warning(f"Unsupported export format: {export_format}, using JSON")
            return self._export_json(export_data)
    
    def _update_cumulative_stats(self):
        """更新累积统计信息"""
        self.cumulative_stats['total_sessions'] += 1
        self.cumulative_stats['total_chunks_processed'] += self.session_stats['total_chunks_processed']
        self.cumulative_stats['total_notes_generated'] += self.session_stats['total_notes_generated']
        
        # 计算平均值
        total_sessions = self.cumulative_stats['total_sessions']
        if total_sessions > 0:
            # 使用加权平均更新比率
            old_weight = (total_sessions - 1) / total_sessions
            new_weight = 1 / total_sessions
            
            self.cumulative_stats['average_sentinel_rate'] = (
                self.cumulative_stats['average_sentinel_rate'] * old_weight +
                self.session_stats['sentinel_rate'] * new_weight
            )
            
            self.cumulative_stats['average_parse_fail_rate'] = (
                self.cumulative_stats['average_parse_fail_rate'] * old_weight +
                self.session_stats['parse_fail_rate'] * new_weight
            )
            
            self.cumulative_stats['average_rule_fallback_rate'] = (
                self.cumulative_stats['average_rule_fallback_rate'] * old_weight +
                self.session_stats['rule_fallback_rate'] * new_weight
            )
        
        self.cumulative_stats['last_updated'] = datetime.now().isoformat()
    
    def _export_json(self, data: Dict[str, Any]) -> str:
        """导出为JSON格式"""
        filename = self.log_file
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Stats exported to JSON: {filename}")
            return filename
        except Exception as e:
            logger.error(f"Failed to export JSON stats: {e}")
            return ""
    
    def _export_yaml(self, data: Dict[str, Any]) -> str:
        """导出为YAML格式"""
        try:
            import yaml
            filename = self.log_file.replace('.json', '.yaml')
            with open(filename, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"Stats exported to YAML: {filename}")
            return filename
        except ImportError:
            logger.warning("PyYAML not installed, falling back to JSON export")
            return self._export_json(data)
        except Exception as e:
            logger.error(f"Failed to export YAML stats: {e}")
            return ""
    
    def _export_txt(self, data: Dict[str, Any]) -> str:
        """导出为文本格式"""
        filename = self.log_file.replace('.json', '.txt')
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("=== 原子笔记生成统计报告 ===\n\n")
                
                # 会话统计
                session = data['session_stats']
                f.write(f"会话ID: {session['session_id']}\n")
                f.write(f"开始时间: {session['start_time']}\n")
                f.write(f"结束时间: {session['end_time']}\n")
                f.write(f"处理时间: {session['total_processing_time']:.2f}秒\n\n")
                
                f.write("=== 处理统计 ===\n")
                f.write(f"总处理块数: {session['total_chunks_processed']}\n")
                f.write(f"总生成笔记数: {session['total_notes_generated']}\n")
                f.write(f"零笔记数量: {session['notes_zero_count']}\n")
                f.write(f"哨兵字符率: {session['sentinel_rate']:.2%}\n")
                f.write(f"解析失败率: {session['parse_fail_rate']:.2%}\n")
                f.write(f"规则回退率: {session['rule_fallback_rate']:.2%}\n")
                f.write(f"平均每块笔记数: {session['avg_notes_per_chunk']:.2f}\n\n")
                
                # 累积统计
                cumulative = data['cumulative_stats']
                f.write("=== 累积统计 ===\n")
                f.write(f"总会话数: {cumulative['total_sessions']}\n")
                f.write(f"累积处理块数: {cumulative['total_chunks_processed']}\n")
                f.write(f"累积生成笔记数: {cumulative['total_notes_generated']}\n")
                f.write(f"平均哨兵字符率: {cumulative['average_sentinel_rate']:.2%}\n")
                f.write(f"平均解析失败率: {cumulative['average_parse_fail_rate']:.2%}\n")
                f.write(f"平均规则回退率: {cumulative['average_rule_fallback_rate']:.2%}\n")
                
            logger.info(f"Stats exported to TXT: {filename}")
            return filename
        except Exception as e:
            logger.error(f"Failed to export TXT stats: {e}")
            return ""
    
    def get_session_summary(self) -> Dict[str, Any]:
        """获取会话摘要"""
        return {
            'session_id': self.session_stats['session_id'],
            'chunks_processed': self.session_stats['total_chunks_processed'],
            'notes_generated': self.session_stats['total_notes_generated'],
            'sentinel_rate': self.session_stats['sentinel_rate'],
            'parse_fail_rate': self.session_stats['parse_fail_rate'],
            'rule_fallback_rate': self.session_stats['rule_fallback_rate'],
            'avg_notes_per_chunk': self.session_stats['avg_notes_per_chunk'],
            'processing_time': self.session_stats['total_processing_time']
        }


# 全局统计记录器实例
_global_stats_logger: Optional[NotesStatsLogger] = None


def get_global_stats_logger() -> NotesStatsLogger:
    """获取全局统计记录器实例"""
    global _global_stats_logger
    if _global_stats_logger is None:
        _global_stats_logger = NotesStatsLogger()
    return _global_stats_logger


def log_notes_stats(stats: Dict[str, Any]):
    """记录笔记统计信息到全局记录器"""
    logger_instance = get_global_stats_logger()
    logger_instance.update_session_stats(stats)


def finalize_notes_session(processing_time: float = 0.0) -> Dict[str, Any]:
    """结束笔记生成会话并返回摘要"""
    logger_instance = get_global_stats_logger()
    logger_instance.finalize_session(processing_time)
    
    # 导出统计信息
    export_format = config.get('notes_llm.stats_export_format', 'json')
    logger_instance.export_stats(export_format)
    
    return logger_instance.get_session_summary()