#!/usr/bin/env python3
"""
vLLM 压力测试脚本

支持环境变量配置，统计延迟并输出 JSON 结果
"""

import os
import sys
import json
import time
import asyncio
import statistics
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from openai import OpenAI
from loguru import logger


class VLLMBenchmark:
    """vLLM 压力测试类"""
    
    def __init__(self, base_url: str, model: str, concurrency: int = 64, num_requests: int = 256):
        self.base_url = base_url
        self.model = model
        self.concurrency = concurrency
        self.num_requests = num_requests
        
        # 初始化 OpenAI 客户端
        self.client = OpenAI(
            base_url=base_url,
            api_key="EMPTY",  # vLLM 不需要真实的 API key
            timeout=60
        )
        
        # 测试消息
        self.test_messages = [
            [{"role": "user", "content": "请简要介绍一下人工智能的发展历史。"}],
            [{"role": "user", "content": "解释一下什么是机器学习？"}],
            [{"role": "user", "content": "Python 和 Java 有什么区别？"}],
            [{"role": "user", "content": "如何优化数据库查询性能？"}],
            [{"role": "user", "content": "请介绍一下云计算的主要特点。"}],
        ]
    
    def single_request(self, request_id: int) -> Dict[str, Any]:
        """执行单个请求
        
        Args:
            request_id: 请求ID
            
        Returns:
            dict: 请求结果
        """
        result = {
            'request_id': request_id,
            'success': False,
            'latency': None,
            'tokens': 0,
            'error': None
        }
        
        try:
            # 随机选择测试消息
            messages = self.test_messages[request_id % len(self.test_messages)]
            
            start_time = time.time()
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
                max_tokens=256,
                top_p=1.0
            )
            
            end_time = time.time()
            
            result['success'] = True
            result['latency'] = (end_time - start_time) * 1000  # ms
            
            if response.usage:
                result['tokens'] = response.usage.completion_tokens or 0
            
            # 记录响应内容长度（用于验证）
            if response.choices and response.choices[0].message.content:
                result['content_length'] = len(response.choices[0].message.content)
            
        except Exception as e:
            result['error'] = str(e)
            result['latency'] = (time.time() - start_time) * 1000 if 'start_time' in locals() else 0
        
        return result
    
    def run_benchmark(self) -> Dict[str, Any]:
        """运行压力测试
        
        Returns:
            dict: 测试结果
        """
        print(f"开始压力测试...")
        print(f"目标: {self.base_url}")
        print(f"模型: {self.model}")
        print(f"并发数: {self.concurrency}")
        print(f"请求数: {self.num_requests}")
        print("=" * 50)
        
        start_time = time.time()
        results = []
        
        # 使用线程池执行并发请求
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            # 提交所有任务
            future_to_id = {
                executor.submit(self.single_request, i): i 
                for i in range(self.num_requests)
            }
            
            # 收集结果
            completed = 0
            for future in as_completed(future_to_id):
                result = future.result()
                results.append(result)
                completed += 1
                
                # 进度显示
                if completed % 10 == 0 or completed == self.num_requests:
                    progress = (completed / self.num_requests) * 100
                    print(f"进度: {completed}/{self.num_requests} ({progress:.1f}%)")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 统计结果
        return self._analyze_results(results, total_time)
    
    def _analyze_results(self, results: List[Dict[str, Any]], total_time: float) -> Dict[str, Any]:
        """分析测试结果
        
        Args:
            results: 所有请求的结果
            total_time: 总耗时
            
        Returns:
            dict: 分析结果
        """
        successful_results = [r for r in results if r['success']]
        failed_results = [r for r in results if not r['success']]
        
        success_count = len(successful_results)
        failure_count = len(failed_results)
        
        # 延迟统计（只统计成功的请求）
        latencies = [r['latency'] for r in successful_results if r['latency'] is not None]
        
        latency_stats = {}
        if latencies:
            latencies.sort()
            latency_stats = {
                'min': round(min(latencies), 2),
                'max': round(max(latencies), 2),
                'mean': round(statistics.mean(latencies), 2),
                'median': round(statistics.median(latencies), 2),
                'p95': round(latencies[int(len(latencies) * 0.95)], 2),
                'p99': round(latencies[int(len(latencies) * 0.99)], 2),
            }
        
        # Token 统计
        total_tokens = sum(r.get('tokens', 0) for r in successful_results)
        
        # 错误统计
        error_types = {}
        for result in failed_results:
            error = result.get('error', 'Unknown')
            error_types[error] = error_types.get(error, 0) + 1
        
        analysis = {
            'benchmark_config': {
                'base_url': self.base_url,
                'model': self.model,
                'concurrency': self.concurrency,
                'num_requests': self.num_requests
            },
            'summary': {
                'total_requests': len(results),
                'successful_requests': success_count,
                'failed_requests': failure_count,
                'success_rate': round((success_count / len(results)) * 100, 2),
                'total_time_seconds': round(total_time, 2),
                'requests_per_second': round(len(results) / total_time, 2),
                'successful_rps': round(success_count / total_time, 2)
            },
            'latency_ms': latency_stats,
            'tokens': {
                'total_completion_tokens': total_tokens,
                'avg_tokens_per_request': round(total_tokens / success_count, 2) if success_count > 0 else 0
            },
            'errors': error_types
        }
        
        return analysis


def main():
    """主函数"""
    # 从环境变量获取配置
    base_url = os.getenv('VLLM_BASE', 'http://127.0.0.1:8002/v1')
    model = os.getenv('VLLM_MODEL', 'gpt_oss_20b')
    concurrency = int(os.getenv('CONC', '64'))
    num_requests = int(os.getenv('NREQ', '256'))
    
    # 创建并运行基准测试
    benchmark = VLLMBenchmark(
        base_url=base_url,
        model=model,
        concurrency=concurrency,
        num_requests=num_requests
    )
    
    try:
        results = benchmark.run_benchmark()
        
        # 输出结果
        print("\n" + "=" * 50)
        print("测试完成！")
        print("\n📊 结果摘要:")
        print(f"成功率: {results['summary']['success_rate']}%")
        print(f"总 RPS: {results['summary']['requests_per_second']}")
        print(f"成功 RPS: {results['summary']['successful_rps']}")
        
        if results['latency_ms']:
            print(f"\n⏱️  延迟统计 (ms):")
            print(f"平均: {results['latency_ms']['mean']}")
            print(f"中位数: {results['latency_ms']['median']}")
            print(f"P95: {results['latency_ms']['p95']}")
            print(f"P99: {results['latency_ms']['p99']}")
        
        if results['errors']:
            print(f"\n❌ 错误统计:")
            for error, count in results['errors'].items():
                print(f"  {error}: {count}")
        
        # 输出完整 JSON 结果
        print("\n" + "=" * 50)
        print("完整 JSON 结果:")
        print(json.dumps(results, indent=2, ensure_ascii=False))
        
    except KeyboardInterrupt:
        print("\n测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n测试失败: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()