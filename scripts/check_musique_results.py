#!/usr/bin/env python3
"""
快速自检脚本，用于检查MuSiQue结果的质量

检查项目：
1. 禁词比例（应接近0）
2. 空答案比例（应为0）
3. support平均长度（应≈2-3）
4. 抽样检查answer是否是support_idxs段落的子串
"""

import json
import re
import argparse
from typing import Dict, List, Any
from pathlib import Path
from loguru import logger

# 禁用的答案短语
FORBIDDEN_PHRASES = {"insufficient information", "no spouse mentioned"}

def load_results(file_path: str) -> List[Dict[str, Any]]:
    """加载结果文件"""
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line: {line[:100]}... Error: {e}")
    return results

def check_forbidden_phrases(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """检查禁词比例"""
    total = len(results)
    forbidden_count = 0
    forbidden_examples = []
    
    for result in results:
        answer = result.get('predicted_answer', '').lower()
        for phrase in FORBIDDEN_PHRASES:
            if phrase in answer:
                forbidden_count += 1
                forbidden_examples.append({
                    'id': result.get('id', 'unknown'),
                    'answer': result.get('predicted_answer', ''),
                    'phrase': phrase
                })
                break
    
    return {
        'total': total,
        'forbidden_count': forbidden_count,
        'forbidden_ratio': forbidden_count / total if total > 0 else 0,
        'examples': forbidden_examples[:5]  # 只显示前5个例子
    }

def check_empty_answers(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """检查空答案比例"""
    total = len(results)
    empty_count = 0
    empty_examples = []
    
    for result in results:
        answer = result.get('predicted_answer', '').strip()
        if not answer:
            empty_count += 1
            empty_examples.append({
                'id': result.get('id', 'unknown'),
                'answer': repr(answer)
            })
    
    return {
        'total': total,
        'empty_count': empty_count,
        'empty_ratio': empty_count / total if total > 0 else 0,
        'examples': empty_examples[:5]
    }

def check_support_lengths(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """检查support平均长度"""
    support_lengths = []
    empty_support_count = 0
    
    for result in results:
        support_idxs = result.get('predicted_support_idxs', [])
        if isinstance(support_idxs, list):
            support_lengths.append(len(support_idxs))
            if len(support_idxs) == 0:
                empty_support_count += 1
        else:
            # 可能是其他格式，尝试解析
            try:
                if isinstance(support_idxs, str):
                    parsed = json.loads(support_idxs)
                    if isinstance(parsed, list):
                        support_lengths.append(len(parsed))
                    else:
                        support_lengths.append(0)
                else:
                    support_lengths.append(0)
            except:
                support_lengths.append(0)
    
    avg_length = sum(support_lengths) / len(support_lengths) if support_lengths else 0
    
    return {
        'total': len(results),
        'empty_support_count': empty_support_count,
        'empty_support_ratio': empty_support_count / len(results) if results else 0,
        'average_length': avg_length,
        'length_distribution': {
            '0': support_lengths.count(0),
            '1': support_lengths.count(1),
            '2': support_lengths.count(2),
            '3': support_lengths.count(3),
            '3+': sum(1 for x in support_lengths if x > 3)
        }
    }

def sample_check_substring_match(results: List[Dict[str, Any]], sample_size: int = 50) -> Dict[str, Any]:
    """抽样检查answer是否是support_idxs段落的子串
    
    注意：这个检查需要原始段落数据，这里只是一个框架
    """
    import random
    
    if len(results) < sample_size:
        sample_size = len(results)
    
    sampled = random.sample(results, sample_size)
    
    # 这里需要访问原始段落数据来进行子串匹配检查
    # 由于我们没有原始段落数据，这里只是统计有答案和support的样本
    valid_samples = 0
    for result in sampled:
        answer = result.get('predicted_answer', '').strip()
        support_idxs = result.get('predicted_support_idxs', [])
        if answer and support_idxs:
            valid_samples += 1
    
    return {
        'sample_size': sample_size,
        'valid_samples': valid_samples,
        'note': 'Substring matching requires original paragraph data, not available in this check'
    }

def analyze_json_parsing_issues(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """分析JSON解析相关的问题"""
    json_string_answers = 0
    malformed_json_answers = 0
    
    for result in results:
        answer = result.get('predicted_answer', '')
        
        # 检查是否是JSON字符串被当作答案
        if answer.startswith('{') and answer.endswith('}'):
            try:
                parsed = json.loads(answer)
                if isinstance(parsed, dict) and 'answer' in parsed:
                    json_string_answers += 1
            except json.JSONDecodeError:
                malformed_json_answers += 1
    
    return {
        'total': len(results),
        'json_string_answers': json_string_answers,
        'malformed_json_answers': malformed_json_answers,
        'json_issues_ratio': (json_string_answers + malformed_json_answers) / len(results) if results else 0
    }

def main():
    parser = argparse.ArgumentParser(description='Check MuSiQue results quality')
    parser.add_argument('result_file', help='Path to the result JSONL file')
    parser.add_argument('--sample-size', type=int, default=50, help='Sample size for substring check')
    
    args = parser.parse_args()
    
    if not Path(args.result_file).exists():
        logger.error(f"Result file not found: {args.result_file}")
        return
    
    logger.info(f"Loading results from {args.result_file}")
    results = load_results(args.result_file)
    
    if not results:
        logger.error("No valid results found")
        return
    
    logger.info(f"Loaded {len(results)} results")
    
    # 执行各项检查
    print("\n" + "="*60)
    print("MuSiQue Results Quality Check Report")
    print("="*60)
    
    # 1. 禁词检查
    forbidden_check = check_forbidden_phrases(results)
    print(f"\n1. Forbidden Phrases Check:")
    print(f"   Total results: {forbidden_check['total']}")
    print(f"   Forbidden count: {forbidden_check['forbidden_count']}")
    print(f"   Forbidden ratio: {forbidden_check['forbidden_ratio']:.4f} ({forbidden_check['forbidden_ratio']*100:.2f}%)")
    if forbidden_check['examples']:
        print(f"   Examples:")
        for ex in forbidden_check['examples']:
            print(f"     - {ex['id']}: '{ex['answer']}' (contains '{ex['phrase']}')")
    
    # 2. 空答案检查
    empty_check = check_empty_answers(results)
    print(f"\n2. Empty Answers Check:")
    print(f"   Total results: {empty_check['total']}")
    print(f"   Empty count: {empty_check['empty_count']}")
    print(f"   Empty ratio: {empty_check['empty_ratio']:.4f} ({empty_check['empty_ratio']*100:.2f}%)")
    if empty_check['examples']:
        print(f"   Examples:")
        for ex in empty_check['examples']:
            print(f"     - {ex['id']}: {ex['answer']}")
    
    # 3. Support长度检查
    support_check = check_support_lengths(results)
    print(f"\n3. Support Lengths Check:")
    print(f"   Total results: {support_check['total']}")
    print(f"   Empty support count: {support_check['empty_support_count']}")
    print(f"   Empty support ratio: {support_check['empty_support_ratio']:.4f} ({support_check['empty_support_ratio']*100:.2f}%)")
    print(f"   Average length: {support_check['average_length']:.2f}")
    print(f"   Length distribution:")
    for length, count in support_check['length_distribution'].items():
        print(f"     - {length}: {count}")
    
    # 4. JSON解析问题检查
    json_check = analyze_json_parsing_issues(results)
    print(f"\n4. JSON Parsing Issues Check:")
    print(f"   Total results: {json_check['total']}")
    print(f"   JSON string answers: {json_check['json_string_answers']}")
    print(f"   Malformed JSON answers: {json_check['malformed_json_answers']}")
    print(f"   JSON issues ratio: {json_check['json_issues_ratio']:.4f} ({json_check['json_issues_ratio']*100:.2f}%)")
    
    # 5. 抽样检查
    substring_check = sample_check_substring_match(results, args.sample_size)
    print(f"\n5. Substring Match Check (Sample):")
    print(f"   Sample size: {substring_check['sample_size']}")
    print(f"   Valid samples (has answer & support): {substring_check['valid_samples']}")
    print(f"   Note: {substring_check['note']}")
    
    # 总结
    print(f"\n" + "="*60)
    print("Summary:")
    print(f"- Forbidden phrases: {forbidden_check['forbidden_ratio']*100:.2f}% (target: ~0%)")
    print(f"- Empty answers: {empty_check['empty_ratio']*100:.2f}% (target: 0%)")
    print(f"- Empty support: {support_check['empty_support_ratio']*100:.2f}% (target: ~0%)")
    print(f"- Average support length: {support_check['average_length']:.2f} (target: 2-3)")
    print(f"- JSON issues: {json_check['json_issues_ratio']*100:.2f}% (target: ~0%)")
    print("="*60)

if __name__ == '__main__':
    main()