#!/usr/bin/env python3
"""
MuSiQue评测自动化脚本
"""
import os
import sys
import subprocess
import json
from pathlib import Path

def get_latest_workdir(result_root: str = "result") -> str:
    """获取最新的工作目录"""
    if not os.path.exists(result_root):
        raise FileNotFoundError(f"Result directory {result_root} not found")
    
    subdirs = [d for d in os.listdir(result_root) if os.path.isdir(os.path.join(result_root, d))]
    if not subdirs:
        raise FileNotFoundError("No work directories found in result")
    
    # 按数字排序
    numeric_subdirs = [d for d in subdirs if d.isdigit()]
    if numeric_subdirs:
        latest = str(max(int(d) for d in numeric_subdirs))
    else:
        latest = sorted(subdirs)[-1]
    
    return os.path.join(result_root, latest)

def convert_to_official_format(input_file: str, output_file: str) -> bool:
    """转换为官方格式"""
    print(f"转换预测格式: {input_file} -> {output_file}")
    
    try:
        result = subprocess.run([
            sys.executable, "convert_to_official_format.py", 
            input_file, output_file
        ], capture_output=True, text=True, check=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"格式转换失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False

def run_evaluation(pred_file: str, gold_file: str, output_file: str) -> bool:
    """运行官方评测"""
    print(f"运行评测: {pred_file} vs {gold_file}")
    
    try:
        result = subprocess.run([
            sys.executable, "musique/musique/evaluate_v1.0.py", 
            pred_file, gold_file
        ], capture_output=True, text=True, check=True)
        
        # 保存评测输出
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\n--- STDERR ---\n")
                f.write(result.stderr)
        
        print("评测完成，结果:")
        print(result.stdout)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"评测失败: {e}")
        print(f"错误输出: {e.stderr}")
        
        # 即使失败也保存错误信息
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"评测失败: {e}\n")
            f.write(f"错误输出: {e.stderr}\n")
            if e.stdout:
                f.write(f"标准输出: {e.stdout}\n")
        
        return False

def main():
    try:
        # 1. 获取最新工作目录
        latest_workdir = get_latest_workdir()
        print(f"使用工作目录: {latest_workdir}")
        
        # 2. 检查预测结果文件
        pred_raw_file = os.path.join(latest_workdir, "musique_results.jsonl")
        if not os.path.exists(pred_raw_file):
            print(f"错误: 预测结果文件不存在: {pred_raw_file}")
            print("请先运行主流程生成预测结果")
            return False
        
        # 3. 转换为官方格式
        pred_official_file = os.path.join(latest_workdir, "musique_pred_official.jsonl")
        if not convert_to_official_format(pred_raw_file, pred_official_file):
            return False
        
        # 4. 运行评测
        gold_file = "data/musique_ans_v1.0_dev_200.jsonl"
        eval_output_file = os.path.join(latest_workdir, "eval_out.txt")
        
        if not run_evaluation(pred_official_file, gold_file, eval_output_file):
            print("评测失败，但已保存错误信息")
        
        # 5. 输出最终路径
        print("\n" + "="*60)
        print("评测完成！文件路径:")
        print(f"预测原始文件: {pred_raw_file}")
        print(f"预测官方格式: {pred_official_file}")
        print(f"评测文本: {eval_output_file}")
        print(f"DONE {os.path.basename(latest_workdir)}={latest_workdir} PRED={pred_official_file} GOLD={gold_file}")
        
        return True
        
    except Exception as e:
        print(f"评测过程出错: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)