#!/usr/bin/env python3
"""
vLLM 压测脚本

使用 asyncio + aiohttp 并发请求测试 vLLM 性能
统计 p50、p95、平均延迟和吞吐量
"""

import asyncio
import aiohttp
import time
import json
import statistics
import argparse
from typing import List, Dict, Any
from dataclasses import dataclass
from loguru import logger


@dataclass
class BenchmarkResult:
    """压测结果数据类"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_time: float
    avg_latency: float
    p50_latency: float
    p95_latency: float
    p99_latency: float
    throughput: float  # requests per second
    error_rate: float
    latencies: List[float]


class VLLMBenchmark:
    """vLLM 压测类"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8001/v1", model: str = "qwen2_5_0_5b"):
        self.base_url = base_url
        self.model = model
        self.api_key = "EMPTY"
        
        # 测试用的消息模板
        self.test_messages = [
            "请简单介绍一下人工智能。",
            "什么是机器学习？",
            "深度学习有哪些应用？",
            "请解释一下神经网络的工作原理。",
            "自然语言处理技术有哪些？",
            "计算机视觉的主要任务是什么？",
            "强化学习是如何工作的？",
            "大语言模型的优势是什么？",
            "AI 在医疗领域有哪些应用？",
            "请谈谈 AI 的发展前景。"
        ]
    
    async def single_request(self, session: aiohttp.ClientSession, message: str, 
                           max_tokens: int = 256, temperature: float = 0.0) -> Dict[str, Any]:
        """发送单个请求
        
        Args:
            session: aiohttp 会话
            message: 请求消息
            max_tokens: 最大 token 数
            temperature: 温度参数
            
        Returns:
            包含响应时间和结果的字典
        """
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": message}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        
        start_time = time.time()
        
        try:
            async with session.post(url, headers=headers, json=payload, timeout=60) as response:
                end_time = time.time()
                latency = end_time - start_time
                
                if response.status == 200:
                    result = await response.json()
                    content = result['choices'][0]['message']['content']
                    return {
                        'success': True,
                        'latency': latency,
                        'response_length': len(content),
                        'content': content[:100] + '...' if len(content) > 100 else content
                    }
                else:
                    error_text = await response.text()
                    return {
                        'success': False,
                        'latency': latency,
                        'error': f"HTTP {response.status}: {error_text}"
                    }
                    
        except Exception as e:
            end_time = time.time()
            latency = end_time - start_time
            return {
                'success': False,
                'latency': latency,
                'error': str(e)
            }
    
    async def run_benchmark(self, concurrency: int = 32, total_requests: int = 100, 
                          max_tokens: int = 256, temperature: float = 0.0) -> BenchmarkResult:
        """运行压测
        
        Args:
            concurrency: 并发数
            total_requests: 总请求数
            max_tokens: 最大 token 数
            temperature: 温度参数
            
        Returns:
            压测结果
        """
        print(f"\n开始压测: 并发数={concurrency}, 总请求数={total_requests}")
        print(f"参数: max_tokens={max_tokens}, temperature={temperature}")
        print("-" * 60)
        
        # 创建请求任务
        messages = [self.test_messages[i % len(self.test_messages)] for i in range(total_requests)]
        
        # 限制并发数
        semaphore = asyncio.Semaphore(concurrency)
        
        async def bounded_request(session, message):
            async with semaphore:
                return await self.single_request(session, message, max_tokens, temperature)
        
        # 执行压测
        start_time = time.time()
        
        connector = aiohttp.TCPConnector(limit=concurrency * 2, limit_per_host=concurrency * 2)
        timeout = aiohttp.ClientTimeout(total=120)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = [bounded_request(session, message) for message in messages]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 统计结果
        successful_results = []
        failed_results = []
        latencies = []
        
        for result in results:
            if isinstance(result, Exception):
                failed_results.append({'error': str(result), 'latency': 0})
            elif result['success']:
                successful_results.append(result)
                latencies.append(result['latency'])
            else:
                failed_results.append(result)
                latencies.append(result['latency'])
        
        # 计算统计指标
        successful_count = len(successful_results)
        failed_count = len(failed_results)
        
        if latencies:
            avg_latency = statistics.mean(latencies)
            p50_latency = statistics.median(latencies)
            p95_latency = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies)
            p99_latency = statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else max(latencies)
        else:
            avg_latency = p50_latency = p95_latency = p99_latency = 0
        
        throughput = successful_count / total_time if total_time > 0 else 0
        error_rate = failed_count / total_requests * 100
        
        return BenchmarkResult(
            total_requests=total_requests,
            successful_requests=successful_count,
            failed_requests=failed_count,
            total_time=total_time,
            avg_latency=avg_latency,
            p50_latency=p50_latency,
            p95_latency=p95_latency,
            p99_latency=p99_latency,
            throughput=throughput,
            error_rate=error_rate,
            latencies=latencies
        )
    
    def print_result(self, result: BenchmarkResult, concurrency: int):
        """打印压测结果
        
        Args:
            result: 压测结果
            concurrency: 并发数
        """
        print(f"\n📊 压测结果 (并发数: {concurrency})")
        print("=" * 60)
        print(f"总请求数:     {result.total_requests}")
        print(f"成功请求数:   {result.successful_requests}")
        print(f"失败请求数:   {result.failed_requests}")
        print(f"错误率:       {result.error_rate:.2f}%")
        print(f"总耗时:       {result.total_time:.2f}s")
        print("")
        print("📈 延迟统计:")
        print(f"平均延迟:     {result.avg_latency:.3f}s")
        print(f"P50 延迟:     {result.p50_latency:.3f}s")
        print(f"P95 延迟:     {result.p95_latency:.3f}s")
        print(f"P99 延迟:     {result.p99_latency:.3f}s")
        print("")
        print("🚀 吞吐量:")
        print(f"QPS:          {result.throughput:.2f} requests/s")
        print(f"每分钟请求数: {result.throughput * 60:.0f} requests/min")
    
    async def progressive_benchmark(self, concurrency_levels: List[int] = None, 
                                  requests_per_level: int = 100):
        """渐进式压测
        
        Args:
            concurrency_levels: 并发级别列表
            requests_per_level: 每个级别的请求数
        """
        if concurrency_levels is None:
            concurrency_levels = [1, 4, 8, 16, 32, 64, 128]
        
        print("🔥 开始渐进式压测")
        print(f"并发级别: {concurrency_levels}")
        print(f"每级别请求数: {requests_per_level}")
        
        results = []
        
        for concurrency in concurrency_levels:
            try:
                result = await self.run_benchmark(
                    concurrency=concurrency,
                    total_requests=requests_per_level
                )
                results.append((concurrency, result))
                self.print_result(result, concurrency)
                
                # 检查错误率，如果太高就停止
                if result.error_rate > 50:
                    print(f"\n⚠️  错误率过高 ({result.error_rate:.1f}%)，停止测试")
                    break
                
                # 短暂休息
                await asyncio.sleep(2)
                
            except Exception as e:
                print(f"\n❌ 并发数 {concurrency} 测试失败: {e}")
                break
        
        # 输出汇总
        self.print_summary(results)
    
    def print_summary(self, results: List[tuple]):
        """打印汇总结果
        
        Args:
            results: 结果列表，每个元素为 (concurrency, BenchmarkResult)
        """
        if not results:
            return
        
        print("\n" + "=" * 80)
        print("📋 压测汇总")
        print("=" * 80)
        print(f"{'并发数':<8} {'QPS':<10} {'平均延迟':<12} {'P95延迟':<12} {'错误率':<10} {'状态':<8}")
        print("-" * 80)
        
        best_qps = 0
        best_concurrency = 0
        
        for concurrency, result in results:
            status = "✅" if result.error_rate < 5 else "⚠️" if result.error_rate < 20 else "❌"
            
            print(f"{concurrency:<8} {result.throughput:<10.2f} {result.avg_latency:<12.3f} "
                  f"{result.p95_latency:<12.3f} {result.error_rate:<10.1f}% {status:<8}")
            
            if result.error_rate < 10 and result.throughput > best_qps:
                best_qps = result.throughput
                best_concurrency = concurrency
        
        print("-" * 80)
        if best_concurrency > 0:
            print(f"🏆 最佳配置: 并发数 {best_concurrency}, QPS {best_qps:.2f}")
        
        print("\n💡 建议:")
        print("- 监控 GPU 使用率: nvidia-smi 或 nvtop")
        print("- 观察内存使用情况")
        print("- 根据错误率调整并发数")
        print("- 考虑增加 --tensor-parallel-size 以利用多GPU")


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="vLLM 压测工具")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001/v1", help="vLLM API 基础URL")
    parser.add_argument("--model", default="qwen2_5_0_5b", help="模型名称")
    parser.add_argument("--concurrency", type=int, help="单次测试并发数")
    parser.add_argument("--requests", type=int, default=100, help="总请求数")
    parser.add_argument("--progressive", action="store_true", help="渐进式压测")
    parser.add_argument("--max-tokens", type=int, default=256, help="最大token数")
    parser.add_argument("--temperature", type=float, default=0.0, help="温度参数")
    
    args = parser.parse_args()
    
    benchmark = VLLMBenchmark(base_url=args.base_url, model=args.model)
    
    # 首先检查服务可用性
    print("🔍 检查 vLLM 服务可用性...")
    try:
        async with aiohttp.ClientSession() as session:
            test_result = await benchmark.single_request(session, "Hello", 10, 0.0)
            if not test_result['success']:
                print(f"❌ vLLM 服务不可用: {test_result.get('error', 'Unknown error')}")
                print("\n请确保 vLLM 服务正在运行:")
                print(f"curl {args.base_url.replace('/v1', '')}/v1/models")
                return
            else:
                print("✅ vLLM 服务可用")
    except Exception as e:
        print(f"❌ 无法连接到 vLLM 服务: {e}")
        return
    
    if args.progressive:
        # 渐进式压测
        await benchmark.progressive_benchmark(requests_per_level=args.requests)
    elif args.concurrency:
        # 单次压测
        result = await benchmark.run_benchmark(
            concurrency=args.concurrency,
            total_requests=args.requests,
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )
        benchmark.print_result(result, args.concurrency)
    else:
        # 默认渐进式压测
        await benchmark.progressive_benchmark(requests_per_level=args.requests)


if __name__ == "__main__":
    asyncio.run(main())