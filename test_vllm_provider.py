#!/usr/bin/env python3
"""
vLLM Provider 测试脚本

用于测试 vLLM OpenAI 兼容接口的连接和功能
"""

import asyncio
import time
from typing import List
from loguru import logger

from llm.factory import LLMFactory
from llm.providers.vllm_openai import VLLMOpenAIProvider
from config import config


def test_vllm_connection():
    """测试 vLLM 连接"""
    print("\n=== 测试 vLLM 连接 ===")
    
    try:
        # 直接创建 vLLM provider
        provider = VLLMOpenAIProvider(
            base_url="http://127.0.0.1:8001/v1",
            model="qwen2_5_0_5b",
            api_key="EMPTY"
        )
        
        # 检查可用性
        is_available = provider.is_available()
        print(f"vLLM 服务可用性: {is_available}")
        
        if is_available:
            # 获取模型列表
            models = provider.list_models()
            print(f"可用模型: {models}")
            
            # 获取模型信息
            info = provider.get_model_info()
            print(f"模型信息: {info}")
        
        return is_available
        
    except Exception as e:
        print(f"连接测试失败: {e}")
        return False


def test_vllm_chat():
    """测试 vLLM 聊天功能"""
    print("\n=== 测试 vLLM 聊天功能 ===")
    
    try:
        provider = VLLMOpenAIProvider(
            base_url="http://127.0.0.1:8001/v1",
            model="qwen2_5_0_5b",
            api_key="EMPTY"
        )
        
        # 测试简单对话
        messages = [
            {"role": "user", "content": "你好，请简单介绍一下自己。"}
        ]
        
        start_time = time.time()
        response = provider.chat(messages)
        end_time = time.time()
        
        print(f"响应时间: {end_time - start_time:.2f}秒")
        print(f"回复内容: {response}")
        
        return True
        
    except Exception as e:
        print(f"聊天测试失败: {e}")
        return False


def test_vllm_stream():
    """测试 vLLM 流式输出"""
    print("\n=== 测试 vLLM 流式输出 ===")
    
    try:
        provider = VLLMOpenAIProvider(
            base_url="http://127.0.0.1:8001/v1",
            model="qwen2_5_0_5b",
            api_key="EMPTY"
        )
        
        messages = [
            {"role": "user", "content": "请写一首关于人工智能的短诗。"}
        ]
        
        print("流式输出:")
        start_time = time.time()
        full_response = ""
        
        for chunk in provider.stream(messages):
            print(chunk, end="", flush=True)
            full_response += chunk
        
        end_time = time.time()
        print(f"\n\n响应时间: {end_time - start_time:.2f}秒")
        print(f"完整回复长度: {len(full_response)} 字符")
        
        return True
        
    except Exception as e:
        print(f"流式输出测试失败: {e}")
        return False


async def test_vllm_async():
    """测试 vLLM 异步功能"""
    print("\n=== 测试 vLLM 异步功能 ===")
    
    try:
        provider = VLLMOpenAIProvider(
            base_url="http://127.0.0.1:8001/v1",
            model="qwen2_5_0_5b",
            api_key="EMPTY"
        )
        
        # 并发测试
        messages_list = [
            [{"role": "user", "content": f"请用一句话描述数字 {i}"}]
            for i in range(1, 6)
        ]
        
        start_time = time.time()
        
        # 并发执行
        tasks = [provider.async_chat(messages) for messages in messages_list]
        responses = await asyncio.gather(*tasks)
        
        end_time = time.time()
        
        print(f"并发请求数: {len(tasks)}")
        print(f"总响应时间: {end_time - start_time:.2f}秒")
        print(f"平均响应时间: {(end_time - start_time) / len(tasks):.2f}秒")
        
        for i, response in enumerate(responses, 1):
            print(f"响应 {i}: {response}")
        
        return True
        
    except Exception as e:
        print(f"异步测试失败: {e}")
        return False


def test_factory_integration():
    """测试工厂模式集成"""
    print("\n=== 测试工厂模式集成 ===")
    
    try:
        # 通过工厂创建 vLLM provider
        provider = LLMFactory.create_provider('vllm_openai')
        
        # 测试基本功能
        response = provider.generate("请说一句话")
        print(f"工厂创建的 provider 响应: {response}")
        
        # 测试可用性检查
        available_providers = LLMFactory.get_available_providers()
        print(f"可用的 providers: {available_providers}")
        
        # 获取最佳 provider
        best_provider = LLMFactory.get_best_available_provider()
        print(f"最佳 provider: {best_provider}")
        
        return True
        
    except Exception as e:
        print(f"工厂集成测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("vLLM Provider 测试开始")
    print("=" * 50)
    
    # 测试结果
    results = {}
    
    # 1. 连接测试
    results['connection'] = test_vllm_connection()
    
    # 只有连接成功才继续其他测试
    if results['connection']:
        # 2. 聊天功能测试
        results['chat'] = test_vllm_chat()
        
        # 3. 流式输出测试
        results['stream'] = test_vllm_stream()
        
        # 4. 异步功能测试
        results['async'] = asyncio.run(test_vllm_async())
        
        # 5. 工厂集成测试
        results['factory'] = test_factory_integration()
    else:
        print("\n⚠️  vLLM 服务不可用，跳过其他测试")
        print("请确保 vLLM 服务正在运行：")
        print("python -m vllm.entrypoints.openai.api_server \\")
        print("  --model Qwen/Qwen2.5-0.5B-Instruct \\")
        print("  --served-model-name qwen2_5_0_5b \\")
        print("  --dtype float16 \\")
        print("  --max-model-len 4096 \\")
        print("  --gpu-memory-utilization 0.80 \\")
        print("  --port 8001")
    
    # 输出测试结果
    print("\n" + "=" * 50)
    print("测试结果汇总:")
    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
    
    # 总体结果
    passed_tests = sum(results.values())
    total_tests = len(results)
    print(f"\n总体结果: {passed_tests}/{total_tests} 测试通过")
    
    if passed_tests == total_tests:
        print("🎉 所有测试通过！vLLM provider 集成成功")
    else:
        print("⚠️  部分测试失败，请检查配置和服务状态")


if __name__ == "__main__":
    main()