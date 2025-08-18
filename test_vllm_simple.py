#!/usr/bin/env python3
"""
简化的 vLLM Provider 测试脚本
避免复杂的依赖问题，只测试核心功能
"""

import sys
import os
import asyncio
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 直接导入 vLLM provider
from llm.providers.vllm_openai import VLLMOpenAIProvider

def test_vllm_provider():
    """测试 vLLM Provider 基本功能"""
    print("=== vLLM Provider 测试 ===")
    
    # 测试配置
    config = {
        'base_url': 'http://127.0.0.1:8001/v1',
        'model': 'qwen2_5_0_5b',
        'api_key': 'EMPTY',
        'temperature': 0.0,
        'max_tokens': 256
    }
    
    try:
        # 创建 provider 实例
        provider = VLLMOpenAIProvider(**config)
        print(f"✓ Provider 创建成功")
        
        # 测试连接
        if provider.is_available():
            print("✓ vLLM 服务连接成功")
            
            # 测试模型列表
            models = provider.list_models()
            print(f"✓ 可用模型: {models}")
            
            # 测试聊天功能
            messages = [
                {"role": "user", "content": "你好，请简单介绍一下你自己。"}
            ]
            
            response = provider.chat(messages)
            print(f"✓ 聊天测试成功: {response[:100]}...")
            
            # 测试流式输出
            print("\n=== 流式输出测试 ===")
            for chunk in provider.stream(messages):
                print(chunk, end='', flush=True)
            print("\n✓ 流式输出测试完成")
            
        else:
            print("✗ vLLM 服务不可用")
            print("请确保 vLLM 服务已启动:")
            print("python -m vllm.entrypoints.openai.api_server \\")
            print("  --model Qwen/Qwen2.5-0.5B-Instruct \\")
            print("  --served-model-name qwen2_5_0_5b \\")
            print("  --dtype float16 \\")
            print("  --max-model-len 4096 \\")
            print("  --gpu-memory-utilization 0.80 \\")
            print("  --port 8001")
            return False
            
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False
    
    return True

async def test_async_functionality():
    """测试异步功能"""
    print("\n=== 异步功能测试 ===")
    
    config = {
        'base_url': 'http://127.0.0.1:8001/v1',
        'model': 'qwen2_5_0_5b',
        'api_key': 'EMPTY',
        'temperature': 0.0,
        'max_tokens': 128
    }
    
    try:
        provider = VLLMOpenAIProvider(**config)
        
        if not provider.is_available():
            print("✗ vLLM 服务不可用，跳过异步测试")
            return False
        
        messages = [
            {"role": "user", "content": "请用一句话介绍人工智能。"}
        ]
        
        # 测试异步聊天
        response = await provider.async_chat(messages)
        print(f"✓ 异步聊天测试成功: {response}")
        
        # 测试异步流式输出
        print("\n=== 异步流式输出测试 ===")
        async for chunk in provider.async_stream(messages):
            print(chunk, end='', flush=True)
        print("\n✓ 异步流式输出测试完成")
        
        return True
        
    except Exception as e:
        print(f"✗ 异步测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始 vLLM Provider 测试...\n")
    
    # 同步测试
    sync_success = test_vllm_provider()
    
    # 异步测试
    async_success = asyncio.run(test_async_functionality())
    
    print("\n=== 测试总结 ===")
    print(f"同步功能: {'✓ 通过' if sync_success else '✗ 失败'}")
    print(f"异步功能: {'✓ 通过' if async_success else '✗ 失败'}")
    
    if sync_success and async_success:
        print("\n🎉 所有测试通过！vLLM Provider 工作正常。")
        return 0
    else:
        print("\n❌ 部分测试失败，请检查 vLLM 服务状态。")
        return 1

if __name__ == "__main__":
    exit(main())