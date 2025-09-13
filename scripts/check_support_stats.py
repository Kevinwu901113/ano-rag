#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
支持段落统计检查脚本

作用：跑完一批预测后，快速验证本次改造是否生效。
检查项：
1. len(predicted_support_idxs)的分布（应集中在2–4）
2. support_idxs[0]所指段落是否包含答案子串
3. 按问题估计的K与实际len(support_idxs)的一致率
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import Counter, defaultdict
from loguru import logger
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.k_estimator import estimate_required_k
from utils.text_utils import TextUtils


class SupportStatsChecker:
    """支持段落统计检查器"""
    
    def __init__(self):
        """初始化检查器"""
        self.stats = {
            'total_samples': 0,
            'support_length_distribution': Counter(),
            'answer_in_first_support': 0,
            'k_estimation_matches': 0,
            'k_estimation_attempts': 0,
            'ghost_id_count': 0,
            'ghost_id_items': [],
            'unique_ghost_ids': set(),
            'errors': []
        }
    
    def load_results_file(self, file_path: str) -> List[Dict[str, Any]]:
        """加载预测结果文件
        
        Args:
            file_path: 结果文件路径（支持.json和.jsonl格式）
            
        Returns:
            预测结果列表
        """
        results = []
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Results file not found: {file_path}")
        
        try:
            if file_path.suffix == '.jsonl':
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_no, line in enumerate(f, 1):
                        line = line.strip()
                        if line:
                            try:
                                result = json.loads(line)
                                results.append(result)
                            except json.JSONDecodeError as e:
                                logger.warning(f"Failed to parse line {line_no}: {e}")
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        results = data
                    else:
                        results = [data]
        except Exception as e:
            logger.error(f"Failed to load results file: {e}")
            raise
        
        logger.info(f"Loaded {len(results)} results from {file_path}")
        return results
    
    def extract_passages_from_result(self, result: Dict[str, Any]) -> Optional[Dict[int, str]]:
        """从结果中提取段落信息
        
        Args:
            result: 单个预测结果
            
        Returns:
            段落索引到内容的映射，如果无法提取返回None
        """
        passages_by_idx = {}
        
        # 尝试从notes字段提取
        notes = result.get('notes', [])
        if notes:
            for note in notes:
                paragraph_idxs = note.get('paragraph_idxs', [])
                content = note.get('content', '')
                if paragraph_idxs and content:
                    primary_idx = paragraph_idxs[0]
                    passages_by_idx[primary_idx] = content
        
        # 如果没有找到，尝试其他可能的字段
        if not passages_by_idx:
            context = result.get('context', '')
            if context:
                # 尝试解析[P{idx}]格式的上下文
                import re
                pattern = r'\[P(\d+)\]\s*([^\[]*?)(?=\[P\d+\]|$)'
                matches = re.findall(pattern, context, re.DOTALL)
                for idx_str, content in matches:
                    try:
                        idx = int(idx_str)
                        passages_by_idx[idx] = content.strip()
                    except ValueError:
                        continue
        
        return passages_by_idx if passages_by_idx else None
    
    def check_answer_in_first_support(self, answer: str, support_idxs: List[int], 
                                     passages_by_idx: Dict[int, str]) -> bool:
        """检查答案是否在第一个支持段落中
        
        Args:
            answer: 预测答案
            support_idxs: 支持段落索引列表
            passages_by_idx: 段落索引到内容的映射
            
        Returns:
            True如果答案在第一个支持段落中
        """
        if not support_idxs or not answer:
            return False
        
        first_idx = support_idxs[0]
        first_passage = passages_by_idx.get(first_idx, '')
        
        if not first_passage:
            return False
        
        # 检查答案是否在段落中（忽略大小写）
        return answer.lower().strip() in first_passage.lower()
    
    def estimate_k_for_sample(self, result: Dict[str, Any], 
                            passages_by_idx: Dict[int, str]) -> Optional[int]:
        """为单个样本估计K值
        
        Args:
            result: 预测结果
            passages_by_idx: 段落索引到内容的映射
            
        Returns:
            估计的K值，如果估计失败返回None
        """
        try:
            question = result.get('query', result.get('question', ''))
            answer = result.get('predicted_answer', result.get('answer', ''))
            
            if not question or not answer:
                return None
            
            # 构建packed_order（按索引排序）
            packed_order = sorted(passages_by_idx.keys())
            
            estimated_k = estimate_required_k(
                question=question,
                answer=answer,
                passages_by_idx=passages_by_idx,
                packed_order=packed_order
            )
            
            return estimated_k
            
        except Exception as e:
            logger.debug(f"K estimation failed: {e}")
            return None
    
    def check_ghost_ids(self, support_idxs: List[int], passages_by_idx: Dict[int, str]) -> List[int]:
        """检查幽灵id（不存在于passages_by_idx中的id）
        
        Args:
            support_idxs: 支持段落索引列表
            passages_by_idx: 段落索引到内容的映射
            
        Returns:
            幽灵id列表
        """
        ghost_ids = []
        for idx in support_idxs:
            try:
                idx_int = int(idx)
                if idx_int not in passages_by_idx:
                    ghost_ids.append(idx_int)
            except (ValueError, TypeError):
                # 非整数id也算作幽灵id
                ghost_ids.append(idx)
        return ghost_ids
    
    def check_single_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """检查单个预测结果
        
        Args:
            result: 预测结果
            
        Returns:
            检查结果统计
        """
        check_result = {
            'support_length': 0,
            'answer_in_first_support': False,
            'estimated_k': None,
            'actual_k': 0,
            'k_match': False,
            'ghost_ids': [],
            'has_ghost_ids': False,
            'error': None
        }
        
        try:
            # 提取基本信息
            support_idxs = result.get('predicted_support_idxs', [])
            answer = result.get('predicted_answer', result.get('answer', ''))
            
            check_result['support_length'] = len(support_idxs)
            check_result['actual_k'] = len(support_idxs)
            
            # 提取段落信息
            passages_by_idx = self.extract_passages_from_result(result)
            
            if passages_by_idx:
                # 检查幽灵id
                ghost_ids = self.check_ghost_ids(support_idxs, passages_by_idx)
                check_result['ghost_ids'] = ghost_ids
                check_result['has_ghost_ids'] = len(ghost_ids) > 0
                
                # 检查答案是否在第一个支持段落中
                check_result['answer_in_first_support'] = self.check_answer_in_first_support(
                    answer, support_idxs, passages_by_idx
                )
                
                # 估计K值
                estimated_k = self.estimate_k_for_sample(result, passages_by_idx)
                if estimated_k is not None:
                    check_result['estimated_k'] = estimated_k
                    check_result['k_match'] = (estimated_k == len(support_idxs))
            
        except Exception as e:
            check_result['error'] = str(e)
            logger.debug(f"Error checking result: {e}")
        
        return check_result
    
    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析预测结果
        
        Args:
            results: 预测结果列表
            
        Returns:
            分析统计结果
        """
        self.stats['total_samples'] = len(results)
        
        for i, result in enumerate(results):
            try:
                check_result = self.check_single_result(result)
                
                # 更新统计
                support_length = check_result['support_length']
                self.stats['support_length_distribution'][support_length] += 1
                
                if check_result['answer_in_first_support']:
                    self.stats['answer_in_first_support'] += 1
                
                if check_result['estimated_k'] is not None:
                    self.stats['k_estimation_attempts'] += 1
                    if check_result['k_match']:
                        self.stats['k_estimation_matches'] += 1
                
                # 统计幽灵id
                if check_result['has_ghost_ids']:
                    self.stats['ghost_id_count'] += 1
                    self.stats['ghost_id_items'].append({
                        'sample_index': i,
                        'id': result.get('id', f'sample_{i}'),
                        'ghost_ids': check_result['ghost_ids'],
                        'all_support_idxs': result.get('predicted_support_idxs', [])
                    })
                    # 记录唯一的幽灵id
                    for ghost_id in check_result['ghost_ids']:
                        self.stats['unique_ghost_ids'].add(ghost_id)
                
                if check_result['error']:
                    self.stats['errors'].append({
                        'sample_index': i,
                        'error': check_result['error']
                    })
                    
            except Exception as e:
                logger.error(f"Failed to analyze sample {i}: {e}")
                self.stats['errors'].append({
                    'sample_index': i,
                    'error': str(e)
                })
        
        return self.generate_report()
    
    def generate_report(self) -> Dict[str, Any]:
        """生成分析报告
        
        Returns:
            分析报告
        """
        total = self.stats['total_samples']
        
        # 计算百分比
        support_length_percentages = {}
        for length, count in self.stats['support_length_distribution'].items():
            support_length_percentages[length] = (count / total * 100) if total > 0 else 0
        
        answer_in_first_percentage = (self.stats['answer_in_first_support'] / total * 100) if total > 0 else 0
        
        k_match_percentage = 0
        if self.stats['k_estimation_attempts'] > 0:
            k_match_percentage = (self.stats['k_estimation_matches'] / self.stats['k_estimation_attempts'] * 100)
        
        # 检查是否满足验收标准
        target_lengths = {2, 3, 4}
        target_samples = sum(self.stats['support_length_distribution'][length] for length in target_lengths)
        target_percentage = (target_samples / total * 100) if total > 0 else 0
        
        # 幽灵id统计
        ghost_id_percentage = (self.stats['ghost_id_count'] / total * 100) if total > 0 else 0
        
        report = {
            'summary': {
                'total_samples': total,
                'target_length_samples': target_samples,
                'target_length_percentage': target_percentage,
                'answer_in_first_support_percentage': answer_in_first_percentage,
                'k_estimation_success_rate': k_match_percentage,
                'ghost_id_count': self.stats['ghost_id_count'],
                'ghost_id_percentage': ghost_id_percentage,
                'unique_ghost_ids_count': len(self.stats['unique_ghost_ids'])
            },
            'support_length_distribution': dict(self.stats['support_length_distribution']),
            'support_length_percentages': support_length_percentages,
            'k_estimation_stats': {
                'attempts': self.stats['k_estimation_attempts'],
                'matches': self.stats['k_estimation_matches'],
                'success_rate': k_match_percentage
            },
            'ghost_id_stats': {
                'ghost_id_items': self.stats['ghost_id_items'][:10],  # 前10个幽灵id样本
                'unique_ghost_ids': sorted(list(self.stats['unique_ghost_ids']))
            },
            'validation_results': {
                'structure_check': target_percentage >= 95.0,  # ≥95% 样本满足 len(support_idxs) ∈ {2,3,4}
                'verifiability_check': answer_in_first_percentage >= 95.0,  # ≥95% 样本满足 answer ∈ paragraph[support_idxs[0]]
                'k_estimation_reasonable': k_match_percentage >= 70.0,  # K估计合理性检查
                'ghost_id_check': self.stats['ghost_id_count'] == 0  # 幽灵id数量必须为0
            },
            'errors': self.stats['errors'][:10]  # 只显示前10个错误
        }
        
        return report
    
    def print_report(self, report: Dict[str, Any]):
        """打印分析报告
        
        Args:
            report: 分析报告
        """
        print("\n" + "="*60)
        print("支持段落统计检查报告")
        print("="*60)
        
        summary = report['summary']
        print(f"\n📊 总体统计:")
        print(f"  总样本数: {summary['total_samples']}")
        print(f"  目标长度样本数 (2-4): {summary['target_length_samples']}")
        print(f"  目标长度占比: {summary['target_length_percentage']:.2f}%")
        print(f"  答案在首段占比: {summary['answer_in_first_support_percentage']:.2f}%")
        print(f"  K估计成功率: {summary['k_estimation_success_rate']:.2f}%")
        print(f"  幽灵ID样本数: {summary['ghost_id_count']}")
        print(f"  幽灵ID占比: {summary['ghost_id_percentage']:.2f}%")
        print(f"  唯一幽灵ID数: {summary['unique_ghost_ids_count']}")
        
        print(f"\n📈 支持段落长度分布:")
        for length in sorted(report['support_length_distribution'].keys()):
            count = report['support_length_distribution'][length]
            percentage = report['support_length_percentages'][length]
            print(f"  长度 {length}: {count} 样本 ({percentage:.2f}%)")
        
        print(f"\n🔍 K估计统计:")
        k_stats = report['k_estimation_stats']
        print(f"  尝试估计: {k_stats['attempts']} 样本")
        print(f"  估计匹配: {k_stats['matches']} 样本")
        print(f"  成功率: {k_stats['success_rate']:.2f}%")
        
        print(f"\n✅ 验收标准检查:")
        validation = report['validation_results']
        structure_status = "✅ 通过" if validation['structure_check'] else "❌ 未通过"
        verifiability_status = "✅ 通过" if validation['verifiability_check'] else "❌ 未通过"
        k_estimation_status = "✅ 合理" if validation['k_estimation_reasonable'] else "⚠️ 需优化"
        ghost_id_status = "✅ 通过" if validation['ghost_id_check'] else "❌ 发现幽灵ID"
        
        print(f"  结构检查 (≥95% 长度2-4): {structure_status}")
        print(f"  可验证性检查 (≥95% 答案在首段): {verifiability_status}")
        print(f"  K估计合理性 (≥70% 匹配): {k_estimation_status}")
        print(f"  幽灵ID检查 (数量=0): {ghost_id_status}")
        
        # 显示幽灵ID详细信息
        ghost_stats = report['ghost_id_stats']
        if ghost_stats['unique_ghost_ids']:
            print(f"\n👻 幽灵ID详细信息:")
            print(f"  发现的唯一幽灵ID: {ghost_stats['unique_ghost_ids']}")
            print(f"  前10个异常样本:")
            for item in ghost_stats['ghost_id_items']:
                print(f"    ID: {item['id']} - 幽灵ID: {item['ghost_ids']} - 所有支持ID: {item['all_support_idxs']}")
        
        if report['errors']:
            print(f"\n⚠️ 错误样本 (显示前10个):")
            for error in report['errors']:
                print(f"  样本 {error['sample_index']}: {error['error']}")
        
        print("\n" + "="*60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='检查支持段落统计')
    parser.add_argument('results_file', help='预测结果文件路径 (.json 或 .jsonl)')
    parser.add_argument('--output', '-o', help='输出报告文件路径 (可选)')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.remove()
        logger.add(sys.stderr, level="INFO")
    
    try:
        # 创建检查器
        checker = SupportStatsChecker()
        
        # 加载结果
        logger.info(f"Loading results from {args.results_file}")
        results = checker.load_results_file(args.results_file)
        
        # 分析结果
        logger.info("Analyzing results...")
        report = checker.analyze_results(results)
        
        # 打印报告
        checker.print_report(report)
        
        # 保存报告
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            logger.info(f"Report saved to {args.output}")
        
        # 返回退出码
        validation = report['validation_results']
        if all(validation.values()):
            logger.info("All validation checks passed!")
            sys.exit(0)
        else:
            logger.warning("Some validation checks failed.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()