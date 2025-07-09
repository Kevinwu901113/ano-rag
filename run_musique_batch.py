#!/usr/bin/env python3
"""
专门用于处理musique数据集的批量脚本
优化版本，包含更好的错误处理和进度显示
"""

import json
import os
import sys
from pathlib import Path
from batch_test import BatchProcessor
from loguru import logger
import time

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run batch processing on musique dataset')
    parser.add_argument('--limit', '-l', type=int, default=None,
                       help='Limit number of items to process (for testing)')
    parser.add_argument('--start', '-s', type=int, default=0,
                       help='Start from item index (0-based)')
    parser.add_argument('--output', '-o', default='anorag.json',
                       help='Output file name')
    parser.add_argument('--resume', '-r', action='store_true',
                       help='Resume from existing output file')
    
    args = parser.parse_args()
    
    # 设置项目根目录
    project_root = Path(__file__).parent
    
    # 输入文件路径
    input_file = project_root / "data" / "musique_ans_v1.0_train.jsonl"
    output_file = project_root / args.output
    
    # 检查输入文件
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        logger.info("Please ensure the musique_ans_v1.0_train.jsonl file is in the data/ directory")
        return 1
    
    # 处理恢复模式
    existing_results = []
    start_index = args.start
    
    if args.resume and output_file.exists():
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
            start_index = len(existing_results)
            logger.info(f"Resuming from item {start_index} (found {len(existing_results)} existing results)")
        except Exception as e:
            logger.warning(f"Could not load existing results: {e}")
            existing_results = []
    
    # 读取输入数据
    logger.info(f"Reading input file: {input_file}")
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON at line {line_num + 1}: {e}")
    
    total_items = len(data)
    logger.info(f"Loaded {total_items} items from input file")
    
    # 应用开始索引和限制
    if start_index > 0:
        data = data[start_index:]
        logger.info(f"Starting from item {start_index}")
    
    if args.limit:
        data = data[:args.limit]
        logger.info(f"Limited to {args.limit} items")
    
    if not data:
        logger.info("No items to process")
        return 0
    
    logger.info(f"Will process {len(data)} items (from index {start_index} to {start_index + len(data) - 1})")
    
    # 创建批量处理器
    processor = BatchProcessor(str(input_file), str(output_file))
    
    # 处理数据
    results = existing_results.copy()
    start_time = time.time()
    
    for i, item in enumerate(data):
        current_index = start_index + i
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing item {current_index + 1}/{total_items} (batch {i + 1}/{len(data)})")
            logger.info(f"ID: {item.get('id', 'unknown')}")
            logger.info(f"Question: {item.get('question', 'No question')[:100]}...")
            
            item_start_time = time.time()
            result = processor.process_single_item(item, current_index)
            item_duration = time.time() - item_start_time
            
            results.append(result)
            
            # 计算进度和预估时间
            elapsed_time = time.time() - start_time
            avg_time_per_item = elapsed_time / (i + 1)
            remaining_items = len(data) - (i + 1)
            estimated_remaining_time = avg_time_per_item * remaining_items
            
            logger.info(f"Completed item {current_index + 1}: {result.get('id')}")
            logger.info(f"Answer: {result.get('predicted_answer', 'No answer')[:100]}...")
            logger.info(f"Item processing time: {item_duration:.1f}s")
            logger.info(f"Average time per item: {avg_time_per_item:.1f}s")
            logger.info(f"Estimated remaining time: {estimated_remaining_time/60:.1f} minutes")
            
            # 每处理5个项目保存一次结果
            if (i + 1) % 5 == 0 or i == len(data) - 1:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                logger.info(f"Intermediate results saved to {output_file}")
            
        except KeyboardInterrupt:
            logger.info("\nReceived interrupt signal. Saving current progress...")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Progress saved to {output_file}")
            return 1
            
        except Exception as e:
            logger.error(f"Failed to process item {current_index + 1}: {e}")
            error_result = {
                "id": item.get('id', f'item_{current_index}'),
                "predicted_answer": "Processing Error",
                "predicted_support_idxs": [],
                "predicted_answerable": False,
                "error": str(e)
            }
            results.append(error_result)
    
    # 保存最终结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    total_time = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"Batch processing completed!")
    logger.info(f"Total items processed: {len(data)}")
    logger.info(f"Total time: {total_time/60:.1f} minutes")
    logger.info(f"Average time per item: {total_time/len(data):.1f} seconds")
    logger.info(f"Results saved to: {output_file}")
    
    # 统计结果
    successful = sum(1 for r in results if r.get('predicted_answerable', False))
    errors = sum(1 for r in results if 'error' in r)
    logger.info(f"Successful answers: {successful}/{len(results)}")
    logger.info(f"Errors: {errors}/{len(results)}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())