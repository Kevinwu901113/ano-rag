#!/usr/bin/env python3
"""
批量测试脚本
读取data/musique_ans_v1.0_train.jsonl文件，对每一行数据运行完整的处理流程
"""

import json
import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger


class BatchProcessor:
    def __init__(self, input_file: str, output_file: str = "anorag.json"):
        self.input_file = input_file
        self.output_file = output_file
        self.project_root = Path(__file__).parent
        self.main_py = self.project_root / "main.py"
        
    def read_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """读取JSONL文件"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
    
    def create_temp_data_file(self, data_item: Dict[str, Any]) -> str:
        """为单个数据项创建临时数据文件"""
        # 创建临时目录
        temp_dir = tempfile.mkdtemp(prefix="anorag_batch_")
        temp_data_dir = os.path.join(temp_dir, "data")
        os.makedirs(temp_data_dir, exist_ok=True)
        
        # 直接保留原始记录，确保包含paragraphs字段
        temp_file = os.path.join(temp_data_dir, "temp_data.jsonl")
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(data_item, ensure_ascii=False) + '\n')
        
        return temp_dir
    
    def run_process(self, temp_dir: str) -> bool:
        """运行文档处理"""
        try:
            # 设置环境变量，指定数据源目录
            env = os.environ.copy()
            env['PYTHONPATH'] = str(self.project_root)
            
            # 运行处理命令
            cmd = ["python", str(self.main_py), "process", "--new", "--force"]
            
            # 临时修改配置以使用临时数据目录
            original_cwd = os.getcwd()
            os.chdir(self.project_root)
            
            # 修改配置文件中的数据源路径
            config_file = self.project_root / "config.yaml"
            if config_file.exists():
                # 备份原配置
                backup_config = config_file.with_suffix(".yaml.backup")
                shutil.copy2(config_file, backup_config)
                
                # 读取并修改配置
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_content = f.read()
                
                # 替换数据源路径
                data_dir = os.path.join(temp_dir, "data")
                modified_config = config_content.replace(
                    'source_docs_dir: "./data"',
                    f'source_docs_dir: "{data_dir}"'
                ).replace(
                    "source_docs_dir: './data'",
                    f"source_docs_dir: '{data_dir}'"
                ).replace(
                    "source_docs_dir: ./data",
                    f"source_docs_dir: {data_dir}"
                ).replace(
                    "source_docs_dir: data",
                    f"source_docs_dir: {data_dir}"
                )
                
                with open(config_file, 'w', encoding='utf-8') as f:
                    f.write(modified_config)
            
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)
            
            # 恢复原配置
            if config_file.exists() and backup_config.exists():
                shutil.move(backup_config, config_file)
            
            os.chdir(original_cwd)
            
            if result.returncode != 0:
                logger.error(f"Process failed: {result.stderr}")
                return False
            
            logger.info("Document processing completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in process: {e}")
            return False
    
    def run_query(self, question: str, work_dir: str = None) -> Dict[str, Any]:
        """运行查询"""
        try:
            env = os.environ.copy()
            env['PYTHONPATH'] = str(self.project_root)
            
            cmd = ["python", str(self.main_py), "query", question]
            if work_dir:
                cmd.extend(["--work-dir", work_dir])
            
            original_cwd = os.getcwd()
            os.chdir(self.project_root)
            
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)
            
            os.chdir(original_cwd)
            
            if result.returncode != 0:
                logger.error(f"Query failed: {result.stderr}")
                return {"answer": "Error", "error": result.stderr}
            
            # 解析输出，提取答案
            answer = result.stdout.strip()
            return {"answer": answer}
            
        except Exception as e:
            logger.error(f"Error in query: {e}")
            return {"answer": "Error", "error": str(e)}
    
    def get_latest_work_dir(self) -> str:
        """获取最新的工作目录"""
        result_root = self.project_root / "result"
        if not result_root.exists():
            return None
        
        subdirs = [d for d in result_root.iterdir() if d.is_dir() and d.name.isdigit()]
        if not subdirs:
            return None
        
        latest = max(subdirs, key=lambda x: int(x.name))
        return str(latest)
    
    def process_single_item(self, data_item: Dict[str, Any], index: int) -> Dict[str, Any]:
        """处理单个数据项"""
        logger.info(f"Processing item {index + 1}: {data_item.get('id', 'unknown')}")
        
        # 创建临时数据文件
        temp_dir = self.create_temp_data_file(data_item)
        
        try:
            # 运行文档处理
            if not self.run_process(temp_dir):
                return {
                    "id": data_item.get('id', f'item_{index}'),
                    "predicted_answer": "Processing Error",
                    "predicted_support_idxs": [],
                    "predicted_answerable": False,
                    "error": "Document processing failed"
                }
            
            # 获取最新工作目录
            work_dir = self.get_latest_work_dir()
            
            # 运行查询
            question = data_item.get('question', '')
            query_result = self.run_query(question, work_dir)
            
            # 构造结果
            result = {
                "id": data_item.get('id', f'item_{index}'),
                "predicted_answer": query_result.get('answer', 'No answer'),
                "predicted_support_idxs": [],  # 这里可以根据实际需要提取支持段落索引
                "predicted_answerable": True if query_result.get('answer') and 'Error' not in query_result.get('answer', '') else False
            }
            
            if 'error' in query_result:
                result['error'] = query_result['error']
            
            return result
            
        finally:
            # 清理临时目录
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to clean up temp dir {temp_dir}: {e}")
    
    def run_batch(self, limit: int = None) -> None:
        """运行批量处理"""
        logger.info(f"Starting batch processing of {self.input_file}")
        
        # 读取输入数据
        data = self.read_jsonl(self.input_file)
        logger.info(f"Loaded {len(data)} items")
        
        if limit:
            data = data[:limit]
            logger.info(f"Limited to first {limit} items")
        
        results = []
        
        for i, item in enumerate(data):
            try:
                result = self.process_single_item(item, i)
                results.append(result)
                logger.info(f"Completed item {i + 1}/{len(data)}: {result.get('id')}")
            except Exception as e:
                logger.error(f"Failed to process item {i + 1}: {e}")
                results.append({
                    "id": item.get('id', f'item_{i}'),
                    "predicted_answer": "Processing Error",
                    "predicted_support_idxs": [],
                    "predicted_answerable": False,
                    "error": str(e)
                })
        
        # 保存结果
        output_path = self.project_root / self.output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Batch processing completed. Results saved to {output_path}")
        logger.info(f"Processed {len(results)} items total")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch test script for Ano-RAG')
    parser.add_argument('--input', '-i', default='data/musique_ans_v1.0_train.jsonl',
                       help='Input JSONL file path')
    parser.add_argument('--output', '-o', default='anorag.json',
                       help='Output JSON file path')
    parser.add_argument('--limit', '-l', type=int,
                       help='Limit number of items to process (for testing)')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return
    
    processor = BatchProcessor(args.input, args.output)
    processor.run_batch(args.limit)


if __name__ == '__main__':
    main()