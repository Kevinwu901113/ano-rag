#!/usr/bin/env python3
"""
vLLM 端点测试脚本
用于验证双实例服务的健康状态和基本功能
"""

import asyncio
import aiohttp
import json
import time
from typing import List, Dict, Any

# 测试端点配置
ENDPOINTS = [
    "http://127.0.0.1:8000/v1",
    "http://127.0.0.1:8001/v1"
]

# 测试消息
TEST_MESSAGES = [
    {
        "role": "system",
        "content": "你是一个有用的助手。请简洁回答问题。"
    },
    {
        "role": "user", 
        "content": "什么是人工智能？请用一句话回答。"
    }
]

async def test_models_endpoint(session: aiohttp.ClientSession, endpoint: str) -> Dict[str, Any]:
    """测试 /v1/models 端点"""
    try:
        url = f"{endpoint}/models"
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
            if response.status == 200:
                data = await response.json()
                return {
                    "endpoint": endpoint,
                    "status": "success",
                    "models": [model["id"] for model in data.get("data", [])]
                }
            else:
                return {
                    "endpoint": endpoint,
                    "status": "error",
                    "error": f"HTTP {response.status}"
                }
    except Exception as e:
        return {
            "endpoint": endpoint,
            "status": "error", 
            "error": str(e)
        }

async def test_chat_completion(session: aiohttp.ClientSession, endpoint: str) -> Dict[str, Any]:
    """测试 chat completions 端点"""
    try:
        url = f"{endpoint}/chat/completions"
        payload = {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "messages": TEST_MESSAGES,
            "max_tokens": 50,
            "temperature": 0.1,
            "stream": False
        }
        
        start_time = time.time()
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as response:
            end_time = time.time()
            
            if response.status == 200:
                data = await response.json()
                content = data["choices"][0]["message"]["content"]
                return {
                    "endpoint": endpoint,
                    "status": "success",
                    "response_time": round(end_time - start_time, 2),
                    "content": content[:100] + "..." if len(content) > 100 else content,
                    "usage": data.get("usage", {})
                }
            else:
                error_text = await response.text()
                return {
                    "endpoint": endpoint,
                    "status": "error",
                    "error": f"HTTP {response.status}: {error_text}"
                }
    except Exception as e:
        return {
            "endpoint": endpoint,
            "status": "error",
            "error": str(e)
        }

async def test_concurrent_requests(session: aiohttp.ClientSession, endpoint: str, num_requests: int = 5) -> Dict[str, Any]:
    """测试并发请求处理能力"""
    try:
        url = f"{endpoint}/chat/completions"
        payload = {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "messages": [
                {"role": "user", "content": f"请回答数字 {i}"} 
                for i in range(num_requests)
            ][0:1] + [{"role": "user", "content": "简单回答：1+1=?"}],
            "max_tokens": 10,
            "temperature": 0.0
        }
        
        start_time = time.time()
        
        # 创建并发任务
        tasks = []
        for i in range(num_requests):
            task_payload = payload.copy()
            task_payload["messages"] = [{"role": "user", "content": f"计算 {i}+1=?"}]
            tasks.append(session.post(url, json=task_payload, timeout=aiohttp.ClientTimeout(total=30)))
        
        # 等待所有请求完成
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        success_count = 0
        error_count = 0
        
        for response in responses:
            if isinstance(response, Exception):
                error_count += 1
            else:
                if response.status == 200:
                    success_count += 1
                else:
                    error_count += 1
                response.close()
        
        return {
            "endpoint": endpoint,
            "status": "success",
            "total_requests": num_requests,
            "successful_requests": success_count,
            "failed_requests": error_count,
            "total_time": round(end_time - start_time, 2),
            "avg_time_per_request": round((end_time - start_time) / num_requests, 2)
        }
        
    except Exception as e:
        return {
            "endpoint": endpoint,
            "status": "error",
            "error": str(e)
        }

async def main():
    """主测试函数"""
    print("🚀 Starting vLLM endpoints testing...")
    print(f"Testing endpoints: {ENDPOINTS}")
    print("=" * 60)
    
    async with aiohttp.ClientSession() as session:
        # 1. 测试 models 端点
        print("\n📋 Testing /v1/models endpoints...")
        models_tasks = [test_models_endpoint(session, endpoint) for endpoint in ENDPOINTS]
        models_results = await asyncio.gather(*models_tasks)
        
        for result in models_results:
            if result["status"] == "success":
                print(f"✅ {result['endpoint']}: {result['models']}")
            else:
                print(f"❌ {result['endpoint']}: {result['error']}")
        
        # 2. 测试 chat completions
        print("\n💬 Testing chat completions...")
        chat_tasks = [test_chat_completion(session, endpoint) for endpoint in ENDPOINTS]
        chat_results = await asyncio.gather(*chat_tasks)
        
        for result in chat_results:
            if result["status"] == "success":
                print(f"✅ {result['endpoint']}: {result['response_time']}s")
                print(f"   Response: {result['content']}")
                print(f"   Usage: {result['usage']}")
            else:
                print(f"❌ {result['endpoint']}: {result['error']}")
        
        # 3. 测试并发处理
        print("\n⚡ Testing concurrent requests (5 requests per endpoint)...")
        concurrent_tasks = [test_concurrent_requests(session, endpoint, 5) for endpoint in ENDPOINTS]
        concurrent_results = await asyncio.gather(*concurrent_tasks)
        
        for result in concurrent_results:
            if result["status"] == "success":
                print(f"✅ {result['endpoint']}: {result['successful_requests']}/{result['total_requests']} successful")
                print(f"   Total time: {result['total_time']}s, Avg: {result['avg_time_per_request']}s/req")
            else:
                print(f"❌ {result['endpoint']}: {result['error']}")
    
    print("\n" + "=" * 60)
    print("🎉 Testing completed!")

if __name__ == "__main__":
    asyncio.run(main())