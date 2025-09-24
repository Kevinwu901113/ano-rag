#!/usr/bin/env python3
"""
提取MuSiQue dev数据集的前200行
"""
import json
import os

def extract_dev200():
    input_file = "data/musique_ans_v1.0_dev.jsonl"
    output_file = "data/musique_ans_v1.0_dev_200.jsonl"
    
    if not os.path.exists(input_file):
        print(f"错误：输入文件 {input_file} 不存在")
        return False
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    count = 0
    skipped_lines = []
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            if count >= 200:
                break
                
            line = line.strip()
            if not line:
                continue
                
            try:
                # 验证JSON格式并检查必要字段
                data = json.loads(line)
                if 'id' not in data or 'question' not in data:
                    print(f"警告：第{line_num}行缺少必要字段 (id 或 question)")
                    skipped_lines.append(line_num)
                    continue
                
                # 写入输出文件
                outfile.write(line + '\n')
                count += 1
                
            except json.JSONDecodeError as e:
                print(f"警告：第{line_num}行JSON解析失败: {e}")
                skipped_lines.append(line_num)
                continue
    
    print(f"成功提取 {count} 条样本到 {output_file}")
    if skipped_lines:
        print(f"跳过的行号: {skipped_lines}")
    
    return True

if __name__ == "__main__":
    extract_dev200()