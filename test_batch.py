#!/usr/bin/env python3
"""
测试批量处理脚本
使用example.jsonl作为测试数据
"""

import os
import sys
from pathlib import Path
from batch_test import BatchProcessor
from loguru import logger

def main():
    # 设置项目根目录
    project_root = Path(__file__).parent
    
    # 使用example.jsonl作为测试输入
    input_file = project_root / "example.jsonl"
    output_file = project_root / "test_result.json"
    
    if not input_file.exists():
        logger.error(f"Test input file not found: {input_file}")
        return
    
    logger.info("Starting batch test with example.jsonl")
    
    # 创建批量处理器
    processor = BatchProcessor(str(input_file), str(output_file))
    
    # 运行批量处理（限制为1个项目进行测试）
    processor.run_batch(limit=1)
    
    logger.info(f"Test completed. Check results in {output_file}")

if __name__ == '__main__':
    main()