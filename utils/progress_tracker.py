from typing import Optional, Any
from tqdm import tqdm
from loguru import logger
import threading

class ProgressTracker:
    """进度跟踪器，用于显示JSONL文件处理的总行数进度"""
    
    def __init__(self, total_items: int, description: str = "Processing", show_rate: bool = True):
        """
        初始化进度跟踪器
        
        Args:
            total_items: 总项目数（如JSONL文件的总行数）
            description: 进度条描述
            show_rate: 是否显示处理速率
        """
        self.total_items = total_items
        self.description = description
        self.show_rate = show_rate
        self.current_count = 0
        self._lock = threading.Lock()
        self._pbar = None
        
    def __enter__(self):
        """上下文管理器入口"""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()
        
    def start(self):
        """启动进度条"""
        # 自定义格式：显示 (当前数量/总数量) 格式
        bar_format = f"{self.description}: ({{n}}/{{total}}) |{{bar}}|"
        if self.show_rate:
            bar_format += " {rate_fmt} {elapsed}"
        else:
            bar_format += " {elapsed}"
            
        self._pbar = tqdm(
            total=self.total_items,
            desc=self.description,
            bar_format=bar_format,
            ncols=100,
            unit="items"
        )
        logger.info(f"Started progress tracking for {self.total_items} items")
        
    def update(self, count: int = 1):
        """更新进度
        
        Args:
            count: 增加的项目数，默认为1
        """
        with self._lock:
            if self._pbar is not None:
                self._pbar.update(count)
                self.current_count += count
                
    def set_description(self, description: str):
        """设置进度条描述
        
        Args:
            description: 新的描述文本
        """
        if self._pbar is not None:
            self._pbar.set_description(description)
            
    def set_postfix(self, **kwargs):
        """设置进度条后缀信息
        
        Args:
            **kwargs: 要显示的后缀信息
        """
        if self._pbar is not None:
            self._pbar.set_postfix(**kwargs)
            
    def get_progress_info(self) -> dict:
        """获取当前进度信息
        
        Returns:
            包含进度信息的字典
        """
        return {
            'current': self.current_count,
            'total': self.total_items,
            'percentage': (self.current_count / self.total_items * 100) if self.total_items > 0 else 0,
            'remaining': self.total_items - self.current_count
        }
        
    def close(self):
        """关闭进度条"""
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None
            logger.info(f"Progress tracking completed: {self.current_count}/{self.total_items} items processed")
            
    def is_complete(self) -> bool:
        """检查是否已完成
        
        Returns:
            如果已完成返回True，否则返回False
        """
        return self.current_count >= self.total_items


class JSONLProgressTracker(ProgressTracker):
    """专门用于JSONL文件处理的进度跟踪器"""
    
    def __init__(self, jsonl_file_path: str, description: str = "Processing JSONL"):
        """
        初始化JSONL进度跟踪器
        
        Args:
            jsonl_file_path: JSONL文件路径
            description: 进度条描述
        """
        # 计算JSONL文件的总行数
        total_lines = self._count_jsonl_lines(jsonl_file_path)
        super().__init__(total_lines, description, show_rate=True)
        self.jsonl_file_path = jsonl_file_path
        
    def _count_jsonl_lines(self, file_path: str) -> int:
        """计算JSONL文件的总行数
        
        Args:
            file_path: JSONL文件路径
            
        Returns:
            文件总行数
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return sum(1 for _ in f)
        except Exception as e:
            logger.error(f"Failed to count lines in {file_path}: {e}")
            return 0
            
    def update_with_item_info(self, item_info: Optional[dict] = None):
        """更新进度并显示当前处理项目的信息
        
        Args:
            item_info: 当前处理项目的信息字典
        """
        self.update(1)
        
        if item_info and self._pbar is not None:
            # 显示当前处理项目的一些关键信息
            postfix_info = {}
            if 'question' in item_info:
                # 截取问题的前30个字符作为显示
                question_preview = item_info['question'][:30] + "..." if len(item_info['question']) > 30 else item_info['question']
                postfix_info['current'] = question_preview
            elif 'title' in item_info:
                title_preview = item_info['title'][:30] + "..." if len(item_info['title']) > 30 else item_info['title']
                postfix_info['current'] = title_preview
                
            if postfix_info:
                self.set_postfix(**postfix_info)