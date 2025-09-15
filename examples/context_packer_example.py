#!/usr/bin/env python3
"""Example usage of ContextPacker for dual-view context assembly.

This example demonstrates how to use the ContextPacker to create
both fact-based and paragraph-based context views from atomic notes.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from query.context_packer import ContextPacker
from loguru import logger


def create_sample_atomic_notes():
    """Create sample atomic notes for testing."""
    return [
        {
            'id': 'note_1',
            'content': '张三是北京大学的教授，专门研究人工智能。',
            'entities': ['张三', '北京大学', '人工智能'],
            'relations': [
                {'subject': '张三', 'predicate': '工作于', 'object': '北京大学'},
                {'subject': '张三', 'predicate': '研究', 'object': '人工智能'}
            ],
            'original_text': '张三是北京大学的教授，专门研究人工智能领域的深度学习算法。他在2020年发表了多篇重要论文。',
            'span_info': {'start': 0, 'end': 45}
        },
        {
            'id': 'note_2', 
            'content': '李四在2023年加入了清华大学计算机系。',
            'entities': ['李四', '清华大学', '计算机系'],
            'relations': [
                {'subject': '李四', 'predicate': '加入', 'object': '清华大学计算机系'}
            ],
            'original_text': '李四博士在2023年3月正式加入了清华大学计算机系，担任副教授职位。',
            'span_info': {'start': 46, 'end': 85}
        },
        {
            'id': 'note_3',
            'content': '人工智能技术在医疗领域有广泛应用。',
            'entities': ['人工智能', '医疗领域'],
            'relations': [
                {'subject': '人工智能技术', 'predicate': '应用于', 'object': '医疗领域'}
            ],
            'original_text': '人工智能技术在医疗领域有广泛应用，包括医学影像分析、药物发现和疾病诊断等方面。',
            'span_info': {'start': 86, 'end': 125}
        },
        {
            'id': 'note_4',
            'content': '深度学习是机器学习的一个重要分支。',
            'entities': ['深度学习', '机器学习'],
            'relations': [
                {'subject': '深度学习', 'predicate': '是', 'object': '机器学习的分支'}
            ],
            'original_text': '深度学习是机器学习的一个重要分支，通过多层神经网络来学习数据的复杂模式。',
            'span_info': {'start': 126, 'end': 165}
        }
    ]


def main():
    """Main example function."""
    logger.info("Starting ContextPacker example")
    
    # Initialize context packer
    packer = ContextPacker()
    
    # Create sample data
    atomic_notes = create_sample_atomic_notes()
    query = "人工智能在大学的研究情况"
    
    logger.info(f"Processing {len(atomic_notes)} atomic notes for query: '{query}'")
    
    # Example 1: Basic dual-view context packing
    print("\n" + "="*60)
    print("示例 1: 基础双视图上下文拼装")
    print("="*60)
    
    dual_context = packer.pack_dual_view_context(
        atomic_notes=atomic_notes,
        query=query,
        token_budget=1500
    )
    
    print(f"\n事实数量: {dual_context.fact_count}")
    print(f"片段数量: {dual_context.span_count}")
    print(f"覆盖实体: {len(dual_context.coverage_entities)}")
    print(f"预估Token: {dual_context.total_tokens}")
    
    print("\n--- 事实视图 ---")
    print(dual_context.note_view)
    
    print("\n--- 段落视图 ---")
    print(dual_context.paragraph_view)
    
    # Example 2: Formatted context for LLM
    print("\n" + "="*60)
    print("示例 2: LLM格式化上下文")
    print("="*60)
    
    formatted_context = packer.format_context_for_llm(
        atomic_notes=atomic_notes,
        query=query,
        token_budget=1200,
        include_metadata=True
    )
    
    print(formatted_context)
    
    # Example 3: Context summary statistics
    print("\n" + "="*60)
    print("示例 3: 上下文统计信息")
    print("="*60)
    
    summary = packer.get_context_summary(dual_context)
    
    print("上下文统计:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Example 4: Token allocation optimization
    print("\n" + "="*60)
    print("示例 4: Token分配优化")
    print("="*60)
    
    optimized_context, optimization_stats = packer.optimize_token_allocation(
        atomic_notes=atomic_notes,
        query=query,
        target_token_budget=1000,
        iterations=2
    )
    
    print(f"\n优化结果:")
    print(f"  最佳分配: {optimization_stats['best_allocation']}")
    print(f"  收敛状态: {optimization_stats['convergence']}")
    print(f"  迭代次数: {len(optimization_stats['iterations'])}")
    
    print("\n优化后的上下文:")
    print(f"  事实数量: {optimized_context.fact_count}")
    print(f"  片段数量: {optimized_context.span_count}")
    print(f"  总Token: {optimized_context.total_tokens}")
    
    # Example 5: Different token budgets comparison
    print("\n" + "="*60)
    print("示例 5: 不同Token预算对比")
    print("="*60)
    
    budgets = [500, 1000, 1500, 2000]
    
    for budget in budgets:
        context = packer.pack_dual_view_context(
            atomic_notes=atomic_notes,
            query=query,
            token_budget=budget
        )
        
        print(f"\nToken预算 {budget}:")
        print(f"  事实: {context.fact_count}, 片段: {context.span_count}")
        print(f"  实际Token: {context.total_tokens}")
        print(f"  实体覆盖: {len(context.coverage_entities)}")
    
    logger.info("ContextPacker example completed successfully")


if __name__ == "__main__":
    main()