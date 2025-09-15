#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import yaml
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from context.packer import ContextPacker

def test_dual_view():
    # 加载配置
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print("=== 双视图配置检查 ===")
    atomic_note_config = config.get('atomic_note_generator', {})
    context_config = atomic_note_config.get('context', {})
    print(f"原子笔记生成器配置存在: {'atomic_note_generator' in config}")
    print(f"上下文配置存在: {'context' in atomic_note_config}")
    print(f"上下文配置内容: {context_config}")
    dual_view_config = context_config.get('dual_view_packing', {})
    print(f"双视图配置: {dual_view_config}")
    print(f"双视图启用状态: {dual_view_config.get('enabled', False)}")
    print(f"事实比例: {dual_view_config.get('facts_ratio', 0.7)}")
    print(f"原文比例: {dual_view_config.get('original_ratio', 0.3)}")
    
    # 创建 ContextPacker 实例
    try:
        # 提取双视图配置并重新构造为ContextPacker期望的格式
        dual_view_config = context_config.get('dual_view_packing', {})
        packer_config = {
            'context': {
                'dual_view_packing': dual_view_config
            }
        }
        packer = ContextPacker(config=packer_config)
        print("\n=== ContextPacker 初始化成功 ===")
        
        # 模拟一些测试数据
        test_notes = [
            {
                'note_id': 'test_1',
                'content': '人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。',
                'title': '人工智能定义',
                'dense_score': 0.8,
                'bm25_score': 0.7
            },
            {
                'note_id': 'test_2', 
                'content': '机器学习是人工智能的一个子领域，通过算法让计算机从数据中学习模式。',
                'title': '机器学习',
                'dense_score': 0.75,
                'bm25_score': 0.6
            }
        ]
        
        test_query = "什么是人工智能？"
        
        print(f"\n=== 测试查询: {test_query} ===")
        
        # 测试双视图打包
        if dual_view_config.get('enabled', False):
            print("\n=== 调用双视图打包 ===")
            try:
                result = packer.pack_dual_view_context(
                    notes=test_notes,
                    question=test_query,
                    token_budget=1800
                )
                print("双视图打包成功！")
                
                # pack_dual_view_context返回元组 (prompt, passages_by_idx, packed_order)
                if isinstance(result, tuple) and len(result) >= 1:
                    prompt = result[0]
                    print(f"生成的prompt长度: {len(prompt)}")
                    
                    # 检查结果内容
                    if "证据清单" in prompt or "facts" in prompt.lower():
                        print("✓ 检测到事实视图内容")
                    else:
                        print("✗ 未检测到事实视图内容")
                    
                    if "原文" in prompt or "paragraph" in prompt.lower():
                        print("✓ 检测到段落视图内容")
                    else:
                        print("✗ 未检测到段落视图内容")
                    
                    print("\n=== 生成的上下文 ===")
                    print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
                else:
                    print(f"意外的返回类型: {type(result)}")
                    print(f"返回值: {result}")
                
            except Exception as e:
                print(f"双视图打包失败: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("双视图功能未启用")
            
    except Exception as e:
        print(f"ContextPacker 初始化失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dual_view()