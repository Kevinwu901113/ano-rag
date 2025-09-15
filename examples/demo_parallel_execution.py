#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模型并行执行演示
展示如何使用优化的并行执行功能提高整体执行效率
"""

import time
from typing import List
from loguru import logger
from optimized_multi_model_client import OptimizedMultiModelClient
from llm.multi_model_client import MultiModelClient

def demo_performance_comparison():
    """演示性能对比：串行 vs 并发 vs 真正并行"""
    logger.info("🎯 多模型并行执行性能对比演示")
    logger.info("="*60)
    
    # 准备测试数据
    test_prompts = [
        "请用一句话描述人工智能的发展前景。",
        "解释什么是深度学习，用简单的语言。",
        "推荐一个学习编程的有效方法。",
        "描述云计算对企业的主要优势。",
        "什么是区块链技术的核心特点？",
        "介绍机器学习在医疗领域的应用。"
    ]
    
    try:
        # 创建客户端
        standard_client = MultiModelClient()
        optimized_client = OptimizedMultiModelClient()
        
        logger.info(f"📊 测试数据: {len(test_prompts)} 个提示")
        logger.info(f"🖥️ 可用实例: {len(standard_client.model_instances)} 个")
        
        # 方法1: 串行执行
        logger.info("\n🐌 方法1: 串行执行")
        start_time = time.time()
        serial_results = []
        for i, prompt in enumerate(test_prompts):
            logger.info(f"   处理请求 {i+1}/{len(test_prompts)}...")
            result = standard_client.generate(prompt, max_tokens=50)
            serial_results.append(result)
        serial_time = time.time() - start_time
        logger.info(f"   ⏱️ 串行执行耗时: {serial_time:.2f}s")
        
        # 方法2: 标准并发执行
        logger.info("\n🔄 方法2: 标准并发执行 (generate_concurrent)")
        start_time = time.time()
        concurrent_results = standard_client.generate_concurrent(test_prompts, max_tokens=50)
        concurrent_time = time.time() - start_time
        logger.info(f"   ⏱️ 并发执行耗时: {concurrent_time:.2f}s")
        
        # 方法3: 优化并行执行
        logger.info("\n🚀 方法3: 优化并行执行 (generate_parallel)")
        start_time = time.time()
        parallel_results = optimized_client.generate_parallel(test_prompts, max_tokens=50)
        parallel_time = time.time() - start_time
        logger.info(f"   ⏱️ 并行执行耗时: {parallel_time:.2f}s")
        
        # 性能分析
        logger.info("\n📈 性能分析:")
        logger.info("="*40)
        
        concurrent_speedup = serial_time / concurrent_time if concurrent_time > 0 else 1.0
        parallel_speedup = serial_time / parallel_time if parallel_time > 0 else 1.0
        parallel_vs_concurrent = concurrent_time / parallel_time if parallel_time > 0 else 1.0
        
        logger.info(f"📊 串行执行:     {serial_time:.2f}s (基准)")
        logger.info(f"📊 标准并发:     {concurrent_time:.2f}s (提升 {concurrent_speedup:.2f}x)")
        logger.info(f"📊 优化并行:     {parallel_time:.2f}s (提升 {parallel_speedup:.2f}x)")
        logger.info(f"🎉 并行 vs 并发: 快 {parallel_vs_concurrent:.2f}x ({(parallel_vs_concurrent-1)*100:.1f}% 更快)")
        
        # 验证结果一致性
        logger.info("\n🔍 结果验证:")
        success_count = sum(1 for r in parallel_results if r and not r.startswith("Error:"))
        logger.info(f"✅ 成功处理: {success_count}/{len(test_prompts)} 个请求")
        
        # 显示部分结果
        logger.info("\n📝 部分结果展示:")
        for i, (prompt, result) in enumerate(zip(test_prompts[:3], parallel_results[:3])):
            logger.info(f"   Q{i+1}: {prompt[:30]}...")
            if result and not result.startswith("Error:"):
                logger.info(f"   A{i+1}: {result[:60]}...")
            else:
                logger.info(f"   A{i+1}: [处理失败]")
        
        return {
            'serial_time': serial_time,
            'concurrent_time': concurrent_time,
            'parallel_time': parallel_time,
            'parallel_speedup': parallel_speedup,
            'success_rate': success_count / len(test_prompts)
        }
        
    except Exception as e:
        logger.error(f"❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def demo_real_world_scenario():
    """演示真实世界场景：批量文档处理"""
    logger.info("\n🌍 真实场景演示：批量文档摘要生成")
    logger.info("="*50)
    
    # 模拟文档内容
    documents = [
        "人工智能技术在过去十年中取得了显著进展，特别是在深度学习、自然语言处理和计算机视觉领域。这些技术的发展不仅推动了科技行业的创新，也在医疗、金融、教育等多个领域产生了深远影响。",
        "云计算作为现代信息技术的重要组成部分，为企业提供了灵活、可扩展的计算资源。通过云服务，企业可以降低IT成本，提高运营效率，并快速响应市场变化。主要的云服务模式包括IaaS、PaaS和SaaS。",
        "区块链技术以其去中心化、不可篡改的特性，在数字货币、供应链管理、数字身份验证等领域展现出巨大潜力。这项技术有望重塑传统的商业模式和信任机制。",
        "机器学习算法能够从大量数据中自动发现模式和规律，为决策提供支持。监督学习、无监督学习和强化学习是三种主要的机器学习范式，各自适用于不同类型的问题。",
        "物联网技术将物理世界与数字世界连接起来，通过传感器、网络和数据分析，实现智能化的监控和控制。这项技术在智慧城市、工业4.0和智能家居等领域有着广泛应用。",
        "量子计算利用量子力学原理进行信息处理，在某些特定问题上具有指数级的计算优势。虽然目前仍处于发展阶段，但量子计算有望在密码学、优化问题和科学模拟等领域带来革命性突破。"
    ]
    
    # 生成摘要提示
    summary_prompts = [
        f"请为以下文档生成一个简洁的摘要（不超过50字）：\n\n{doc}"
        for doc in documents
    ]
    
    try:
        client = OptimizedMultiModelClient()
        
        logger.info(f"📄 待处理文档: {len(documents)} 个")
        logger.info(f"🖥️ 可用实例: {len(client.model_instances)} 个")
        
        # 并行生成摘要
        logger.info("\n🚀 开始并行生成摘要...")
        start_time = time.time()
        summaries = client.generate_parallel(summary_prompts, max_tokens=60)
        processing_time = time.time() - start_time
        
        logger.info(f"✅ 处理完成，耗时: {processing_time:.2f}s")
        logger.info(f"⚡ 平均每个文档: {processing_time/len(documents):.2f}s")
        
        # 显示结果
        logger.info("\n📋 文档摘要结果:")
        logger.info("-" * 60)
        
        for i, (doc, summary) in enumerate(zip(documents, summaries), 1):
            logger.info(f"\n📄 文档 {i}:")
            logger.info(f"   原文: {doc[:80]}...")
            if summary and not summary.startswith("Error:"):
                logger.info(f"   摘要: {summary.strip()}")
            else:
                logger.info(f"   摘要: [生成失败]")
        
        # 统计信息
        success_count = sum(1 for s in summaries if s and not s.startswith("Error:"))
        logger.info(f"\n📊 处理统计:")
        logger.info(f"   成功率: {success_count}/{len(documents)} ({success_count/len(documents)*100:.1f}%)")
        logger.info(f"   总耗时: {processing_time:.2f}s")
        logger.info(f"   吞吐量: {len(documents)/processing_time:.1f} 文档/秒")
        
        return {
            'documents_processed': len(documents),
            'processing_time': processing_time,
            'success_rate': success_count / len(documents),
            'throughput': len(documents) / processing_time
        }
        
    except Exception as e:
        logger.error(f"❌ 真实场景演示失败: {e}")
        return None

def demo_scalability_test():
    """演示可扩展性测试：不同负载下的性能"""
    logger.info("\n📈 可扩展性测试：不同负载下的性能表现")
    logger.info("="*50)
    
    try:
        client = OptimizedMultiModelClient()
        
        # 测试不同的负载大小
        load_sizes = [2, 4, 6, 8, 10]
        base_prompt = "请简单解释以下概念："
        concepts = ["人工智能", "机器学习", "深度学习", "神经网络", "自然语言处理", 
                   "计算机视觉", "强化学习", "数据挖掘", "大数据", "云计算"]
        
        results = []
        
        for load_size in load_sizes:
            logger.info(f"\n🧪 测试负载: {load_size} 个请求")
            
            # 准备测试提示
            test_prompts = [f"{base_prompt}{concepts[i % len(concepts)]}" 
                          for i in range(load_size)]
            
            # 执行测试
            start_time = time.time()
            responses = client.generate_parallel(test_prompts, max_tokens=40)
            execution_time = time.time() - start_time
            
            # 统计结果
            success_count = sum(1 for r in responses if r and not r.startswith("Error:"))
            throughput = load_size / execution_time
            
            result = {
                'load_size': load_size,
                'execution_time': execution_time,
                'success_count': success_count,
                'success_rate': success_count / load_size,
                'throughput': throughput
            }
            results.append(result)
            
            logger.info(f"   ⏱️ 执行时间: {execution_time:.2f}s")
            logger.info(f"   ✅ 成功率: {success_count}/{load_size} ({result['success_rate']*100:.1f}%)")
            logger.info(f"   🚀 吞吐量: {throughput:.1f} 请求/秒")
        
        # 分析可扩展性
        logger.info("\n📊 可扩展性分析:")
        logger.info("-" * 40)
        
        for result in results:
            efficiency = result['throughput'] / results[0]['throughput'] if results[0]['throughput'] > 0 else 1.0
            logger.info(f"负载 {result['load_size']:2d}: {result['execution_time']:5.2f}s, "
                       f"{result['throughput']:5.1f} req/s, 效率 {efficiency:.2f}x")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ 可扩展性测试失败: {e}")
        return None

def main():
    """主演示函数"""
    logger.info("🎬 多模型并行执行完整演示")
    logger.info("="*80)
    
    # 演示1: 性能对比
    perf_results = demo_performance_comparison()
    
    # 演示2: 真实场景
    real_world_results = demo_real_world_scenario()
    
    # 演示3: 可扩展性测试
    scalability_results = demo_scalability_test()
    
    # 总结
    logger.info("\n🎉 演示总结")
    logger.info("="*40)
    
    if perf_results:
        logger.info(f"✅ 性能提升: {perf_results['parallel_speedup']:.2f}x")
        logger.info(f"✅ 成功率: {perf_results['success_rate']*100:.1f}%")
    
    if real_world_results:
        logger.info(f"✅ 文档处理吞吐量: {real_world_results['throughput']:.1f} 文档/秒")
    
    if scalability_results:
        max_throughput = max(r['throughput'] for r in scalability_results)
        logger.info(f"✅ 最大吞吐量: {max_throughput:.1f} 请求/秒")
    
    logger.info("\n🚀 多模型并行系统已准备就绪，可以显著提高处理效率！")

if __name__ == "__main__":
    main()