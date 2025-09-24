#!/usr/bin/env python3
"""
将预测结果转换为MuSiQue官方格式
"""
import json
import os
import sys
from typing import Dict, Any, List

def convert_to_official_format(input_file: str, output_file: str) -> bool:
    """
    将预测结果转换为MuSiQue官方格式
    
    输入格式示例：
    {"id": "...", "answer": "...", "supporting_notes": [...]}
    
    输出格式：
    {"id": "...", "predicted_answer": "...", "predicted_support_idxs": [...]}
    """
    if not os.path.exists(input_file):
        print(f"错误：输入文件 {input_file} 不存在")
        return False
    
    converted_count = 0
    skipped_lines = []
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
                
                # 检查必要字段
                if 'id' not in data:
                    print(f"警告：第{line_num}行缺少id字段")
                    skipped_lines.append(line_num)
                    continue
                
                # 构造官方格式
                official_data = {
                    "id": data["id"]
                }
                
                # 处理答案字段
                if "predicted_answer" in data:
                    # 已经是官方格式
                    official_data["predicted_answer"] = data["predicted_answer"]
                elif "answer" in data:
                    # 转换格式
                    official_data["predicted_answer"] = data["answer"]
                else:
                    print(f"警告：第{line_num}行缺少答案字段")
                    official_data["predicted_answer"] = ""
                
                # 处理支持证据字段
                if "predicted_support_idxs" in data:
                    # 已经是官方格式
                    official_data["predicted_support_idxs"] = data["predicted_support_idxs"]
                elif "supporting_notes" in data and isinstance(data["supporting_notes"], list):
                    # 尝试从supporting_notes提取索引
                    # 如果supporting_notes包含索引信息，提取之；否则给空列表
                    support_idxs = []
                    for note in data["supporting_notes"]:
                        if isinstance(note, dict) and "paragraph_idx" in note:
                            support_idxs.append(note["paragraph_idx"])
                        elif isinstance(note, int):
                            support_idxs.append(note)
                    official_data["predicted_support_idxs"] = support_idxs
                else:
                    # 没有支持证据信息，给空列表
                    official_data["predicted_support_idxs"] = []
                
                # 处理predicted_answerable字段
                if "predicted_answerable" in data:
                    official_data["predicted_answerable"] = data["predicted_answerable"]
                else:
                    # 默认为True（可回答）
                    official_data["predicted_answerable"] = True
                
                # 写入转换后的数据
                outfile.write(json.dumps(official_data, ensure_ascii=False) + '\n')
                converted_count += 1
                
            except json.JSONDecodeError as e:
                print(f"警告：第{line_num}行JSON解析失败: {e}")
                skipped_lines.append(line_num)
                continue
            except Exception as e:
                print(f"警告：第{line_num}行处理失败: {e}")
                skipped_lines.append(line_num)
                continue
    
    print(f"成功转换 {converted_count} 条记录到 {output_file}")
    if skipped_lines:
        print(f"跳过的行号: {skipped_lines}")
    
    return True

def main():
    if len(sys.argv) != 3:
        print("用法: python convert_to_official_format.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    success = convert_to_official_format(input_file, output_file)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()