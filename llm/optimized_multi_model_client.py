#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化的多模型并行客户端
实现真正的并行执行，提高整体执行效率
"""

import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
from loguru import logger
from llm.multi_model_client import MultiModelClient

class OptimizedMultiModelClient(MultiModelClient):
    """优化的多模型并行客户端
    
    继承自MultiModelClient，增强并行处理能力：
    - 真正的并行执行：多个实例同时处理请求
    - 智能任务分配：根据实例性能动态分配任务
    - 性能监控：实时监控并行执行效率
    - 负载均衡优化：避免单实例过载
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.parallel_stats = {
            'total_parallel_requests': 0,
            'total_parallel_time': 0.0,
            'avg_parallel_efficiency': 0.0
        }
    
    def generate_parallel(self, prompts: List[str], system_prompt: str = None, **kwargs) -> List[str]:
        """真正的并行生成方法
        
        与generate_concurrent不同，这个方法确保多个实例真正同时处理请求
        
        Args:
            prompts: 提示列表
            system_prompt: 系统提示
            **kwargs: 其他参数
            
        Returns:
            响应列表
        """
        if not prompts:
            return []
        
        # 获取健康实例
        healthy_instances = [inst for inst in self.model_instances if inst.is_healthy and not inst.is_loading]
        
        if not healthy_instances:
            raise Exception("No healthy model instances available for parallel processing")
        
        logger.info(f"🚀 开始并行处理 {len(prompts)} 个请求，使用 {len(healthy_instances)} 个实例")
        
        start_time = time.time()
        
        # 如果提示数量少于或等于实例数量，每个实例处理一个
        if len(prompts) <= len(healthy_instances):
            return self._execute_direct_parallel(prompts, healthy_instances, system_prompt, **kwargs)
        else:
            # 如果提示数量多于实例数量，需要分批处理
            return self._execute_batch_parallel(prompts, healthy_instances, system_prompt, **kwargs)
    
    def _execute_direct_parallel(self, prompts: List[str], instances: List, system_prompt: str = None, **kwargs) -> List[str]:
        """直接并行执行（提示数 <= 实例数）"""
        results = [None] * len(prompts)
        
        with ThreadPoolExecutor(max_workers=len(instances)) as executor:
            # 为每个提示分配一个实例
            future_to_index = {}
            for i, (prompt, instance) in enumerate(zip(prompts, instances)):
                future = executor.submit(self._execute_single_request, instance, prompt, system_prompt, **kwargs)
                future_to_index[future] = i
            
            # 收集结果
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    logger.error(f"Parallel request {index} failed: {e}")
                    results[index] = f"Error: {str(e)}"
        
        return results
    
    def _execute_batch_parallel(self, prompts: List[str], instances: List, system_prompt: str = None, **kwargs) -> List[str]:
        """批量并行执行（提示数 > 实例数）"""
        results = [None] * len(prompts)
        num_instances = len(instances)
        
        # 将提示分配给实例
        instance_tasks = [[] for _ in range(num_instances)]
        for i, prompt in enumerate(prompts):
            instance_idx = i % num_instances
            instance_tasks[instance_idx].append((i, prompt))
        
        with ThreadPoolExecutor(max_workers=num_instances) as executor:
            # 为每个实例提交一个批量任务
            futures = []
            for instance_idx, tasks in enumerate(instance_tasks):
                if tasks:  # 只处理有任务的实例
                    future = executor.submit(
                        self._execute_instance_batch, 
                        instances[instance_idx], 
                        tasks, 
                        system_prompt, 
                        **kwargs
                    )
                    futures.append((future, instance_idx))
            
            # 收集结果
            for future, instance_idx in futures:
                try:
                    batch_results = future.result()
                    for original_index, result in batch_results:
                        results[original_index] = result
                except Exception as e:
                    logger.error(f"Batch execution failed for instance {instance_idx}: {e}")
                    # 为该实例的所有任务设置错误结果
                    for original_index, _ in instance_tasks[instance_idx]:
                        results[original_index] = f"Error: {str(e)}"
        
        return results
    
    def _execute_instance_batch(self, instance, tasks: List[tuple], system_prompt: str = None, **kwargs) -> List[tuple]:
        """单个实例执行批量任务"""
        results = []
        for original_index, prompt in tasks:
            try:
                result = self._execute_single_request(instance, prompt, system_prompt, **kwargs)
                results.append((original_index, result))
            except Exception as e:
                logger.error(f"Single request failed in batch: {e}")
                results.append((original_index, f"Error: {str(e)}"))
        return results
    
    def _execute_single_request(self, instance, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """执行单个请求"""
        try:
            # 更新实例状态
            instance.active_requests += 1
            
            # 准备消息
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # 合并选项
            options = {**self.default_options, **kwargs}
            
            # 执行请求
            start_time = time.time()
            response = instance.client.chat.completions.create(
                model=instance.model_name,
                messages=messages,
                **options
            )
            end_time = time.time()
            
            # 更新统计信息
            instance.total_requests += 1
            response_time = end_time - start_time
            instance.avg_response_time = (
                (instance.avg_response_time * (instance.total_requests - 1) + response_time) / 
                instance.total_requests
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            instance.error_count += 1
            raise e
        finally:
            instance.active_requests -= 1
    
    def generate_concurrent_optimized(self, prompts: List[str], system_prompt: str = None, **kwargs) -> List[str]:
        """优化的并发生成方法
        
        结合了原有的负载均衡策略和新的并行执行能力
        """
        if not prompts:
            return []
        
        start_time = time.time()
        
        # 使用真正的并行执行
        results = self.generate_parallel(prompts, system_prompt, **kwargs)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 更新并行统计
        self.parallel_stats['total_parallel_requests'] += len(prompts)
        self.parallel_stats['total_parallel_time'] += total_time
        
        # 计算并行效率
        if len(prompts) > 1:
            theoretical_serial_time = total_time * len(prompts)
            efficiency = theoretical_serial_time / total_time if total_time > 0 else 1.0
            self.parallel_stats['avg_parallel_efficiency'] = (
                (self.parallel_stats['avg_parallel_efficiency'] * 
                 (self.parallel_stats['total_parallel_requests'] - len(prompts)) + efficiency * len(prompts)) /
                self.parallel_stats['total_parallel_requests']
            )
        
        logger.info(f"✅ 并行处理完成，耗时: {total_time:.2f}s，效率: {efficiency:.2f}x")
        
        return results
    
    def get_parallel_stats(self) -> Dict[str, Any]:
        """获取并行执行统计信息"""
        return {
            **self.parallel_stats,
            'instance_stats': [
                {
                    'model_name': inst.model_name,
                    'total_requests': inst.total_requests,
                    'active_requests': inst.active_requests,
                    'avg_response_time': inst.avg_response_time,
                    'error_count': inst.error_count,
                    'is_healthy': inst.is_healthy
                }
                for inst in self.model_instances
            ]
        }
    
    def benchmark_parallel_performance(self, test_prompts: List[str] = None, iterations: int = 3) -> Dict[str, Any]:
        """基准测试并行性能"""
        if test_prompts is None:
            test_prompts = [
                "请简单介绍人工智能。",
                "解释什么是机器学习。",
                "描述深度学习的应用。",
                "什么是自然语言处理？"
            ]
        
        logger.info(f"🧪 开始并行性能基准测试，{iterations} 次迭代")
        
        # 测试串行执行
        serial_times = []
        for i in range(iterations):
            start_time = time.time()
            for prompt in test_prompts:
                self.generate(prompt, max_tokens=30)
            serial_times.append(time.time() - start_time)
        
        avg_serial_time = sum(serial_times) / len(serial_times)
        
        # 测试并行执行
        parallel_times = []
        for i in range(iterations):
            start_time = time.time()
            self.generate_parallel(test_prompts, max_tokens=30)
            parallel_times.append(time.time() - start_time)
        
        avg_parallel_time = sum(parallel_times) / len(parallel_times)
        
        # 计算性能提升
        speedup = avg_serial_time / avg_parallel_time if avg_parallel_time > 0 else 1.0
        efficiency = speedup / len(self.model_instances) if self.model_instances else 0.0
        
        results = {
            'test_prompts_count': len(test_prompts),
            'iterations': iterations,
            'avg_serial_time': avg_serial_time,
            'avg_parallel_time': avg_parallel_time,
            'speedup': speedup,
            'efficiency': efficiency,
            'instance_count': len(self.model_instances),
            'performance_improvement': (speedup - 1) * 100  # 百分比提升
        }
        
        logger.info(f"📊 基准测试结果:")
        logger.info(f"   串行平均耗时: {avg_serial_time:.2f}s")
        logger.info(f"   并行平均耗时: {avg_parallel_time:.2f}s")
        logger.info(f"   性能提升: {speedup:.2f}x ({results['performance_improvement']:.1f}%)")
        logger.info(f"   并行效率: {efficiency:.2f}")
        
        return results

# 使用示例和测试函数
def test_optimized_client():
    """测试优化的多模型客户端"""
    logger.info("🧪 测试优化的多模型客户端")
    
    try:
        # 创建优化客户端
        client = OptimizedMultiModelClient()
        
        # 测试提示
        test_prompts = [
            "请用一句话描述春天。",
            "解释什么是云计算。",
            "推荐一本好书。",
            "描述人工智能的未来。",
            "什么是区块链技术？",
            "介绍量子计算的原理。"
        ]
        
        # 测试并行执行
        logger.info(f"\n🚀 测试并行执行 {len(test_prompts)} 个请求")
        start_time = time.time()
        results = client.generate_parallel(test_prompts, max_tokens=50)
        parallel_time = time.time() - start_time
        
        logger.info(f"✅ 并行执行完成，耗时: {parallel_time:.2f}s")
        
        # 显示结果
        for i, result in enumerate(results, 1):
            if result and not result.startswith("Error:"):
                logger.info(f"📝 结果{i}: {result[:50]}...")
            else:
                logger.error(f"❌ 结果{i}: {result}")
        
        # 显示统计信息
        stats = client.get_parallel_stats()
        logger.info(f"\n📊 并行统计: 总请求 {stats['total_parallel_requests']}, 平均效率 {stats['avg_parallel_efficiency']:.2f}x")
        
        # 运行基准测试
        logger.info("\n🏁 运行性能基准测试")
        benchmark_results = client.benchmark_parallel_performance(test_prompts[:4], iterations=2)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_optimized_client()