"""
流式处理早停机制
实现基于首字符和累计字符的早停逻辑
"""

from typing import Iterator, Optional
from loguru import logger


class StreamingEarlyStop:
    """流式处理早停机制"""
    
    def __init__(self, sentinel_char: str = "~", max_chars_before_check: int = 16):
        """
        初始化早停机制
        
        Args:
            sentinel_char: 哨兵字符，表示零笔记
            max_chars_before_check: 在检查非[字符前允许的最大字符数
        """
        self.sentinel_char = sentinel_char
        self.max_chars_before_check = max_chars_before_check
    
    def apply_early_stop(self, stream: Iterator[str]) -> Iterator[str]:
        """
        对流式输出应用早停机制
        
        Args:
            stream: 原始流式输出
            
        Yields:
            str: 处理后的流式输出
        """
        accumulated_text = ""
        first_char_received = False
        
        for chunk in stream:
            if not chunk:
                continue
            
            # 累积文本
            accumulated_text += chunk
            
            # 检查首字符
            if not first_char_received and accumulated_text.strip():
                first_char = accumulated_text.strip()[0]
                first_char_received = True
                
                # 如果首字符是哨兵字符，立即停止
                if first_char == self.sentinel_char:
                    logger.debug(f"Early stop triggered: first character is sentinel '{self.sentinel_char}'")
                    yield chunk
                    return
            
            # 检查累计字符数和非期望字符
            if len(accumulated_text.strip()) > self.max_chars_before_check:
                stripped_text = accumulated_text.strip()
                if stripped_text and stripped_text[0] not in ['[', self.sentinel_char]:
                    logger.debug(f"Early stop triggered: accumulated {len(stripped_text)} chars, first char is '{stripped_text[0]}'")
                    # 不再yield更多内容，但保留已有内容
                    yield chunk
                    return
            
            yield chunk
    
    def should_early_stop(self, accumulated_text: str) -> tuple[bool, str]:
        """
        检查是否应该早停
        
        Args:
            accumulated_text: 累积的文本
            
        Returns:
            tuple: (是否应该早停, 早停原因)
        """
        stripped_text = accumulated_text.strip()
        
        if not stripped_text:
            return False, ""
        
        first_char = stripped_text[0]
        
        # 首字符是哨兵字符
        if first_char == self.sentinel_char:
            return True, f"First character is sentinel '{self.sentinel_char}'"
        
        # 累计字符超过阈值且首字符不是期望字符
        if len(stripped_text) > self.max_chars_before_check:
            if first_char not in ['[', self.sentinel_char]:
                return True, f"Accumulated {len(stripped_text)} chars, first char is '{first_char}'"
        
        return False, ""


def create_early_stop_stream(
    original_stream: Iterator[str], 
    sentinel_char: str = "~", 
    max_chars_before_check: int = 16
) -> Iterator[str]:
    """
    创建带早停机制的流式输出
    
    Args:
        original_stream: 原始流式输出
        sentinel_char: 哨兵字符
        max_chars_before_check: 检查前的最大字符数
        
    Yields:
        str: 处理后的流式输出
    """
    early_stop = StreamingEarlyStop(sentinel_char, max_chars_before_check)
    yield from early_stop.apply_early_stop(original_stream)