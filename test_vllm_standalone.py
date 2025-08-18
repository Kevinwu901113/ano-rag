#!/usr/bin/env python3
"""
独立的 vLLM Provider 测试脚本
直接测试 vLLM 的 OpenAI 兼容接口
"""

import asyncio
import aiohttp
import requests
import json
from typing import List, Dict, Any, Optional, AsyncGenerator

class SimpleVLLMClient:
    """简化的 vLLM 客户端，用于测试"""
    
    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY"):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def is_available(self) -> bool:
        """检查服务是否可用"""
        try:
            response = requests.get(f"{self.base_url}/models", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def list_models(self) -> List[str]:
        """获取可用模型列表"""
        try:
            response = requests.get(f"{self.base_url}/models", headers=self.headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return [model["id"] for model in data.get("data", [])]
            return []
        except Exception as e:
            print(f"获取模型列表失败: {e}")
            return []
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """聊天接口"""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.0),
            "max_tokens": kwargs.get("max_tokens", 256),
            "top_p": kwargs.get("top_p", 1.0)
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            else:
                raise Exception(f"API 错误: {response.status_code} - {response.text}")
                
        except Exception as e:
            raise Exception(f"聊天请求失败: {e}")
    
    def stream(self, messages: List[Dict[str, str]], **kwargs):
        """流式聊天接口"""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.0),
            "max_tokens": kwargs.get("max_tokens", 256),
            "top_p": kwargs.get("top_p", 1.0),
            "stream": True
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                stream=True,
                timeout=30
            )
            
            if response.status_code != 200:
                raise Exception(f"API 错误: {response.status_code} - {response.text}")
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data_str = line[6:]
                        if data_str.strip() == '[DONE]':
                            break
                        try:
                            data = json.loads(data_str)
                            if 'choices' in data and len(data['choices']) > 0:
                                delta = data['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    yield delta['content']
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            raise Exception(f"流式请求失败: {e}")
    
    async def async_chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """异步聊天接口"""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.0),
            "max_tokens": kwargs.get("max_tokens", 256),
            "top_p": kwargs.get("top_p", 1.0)
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data["choices"][0]["message"]["content"]
                    else:
                        text = await response.text()
                        raise Exception(f"API 错误: {response.status} - {text}")
                        
        except Exception as e:
            raise Exception(f"异步聊天请求失败: {e}")
    
    async def async_stream(self, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[str, None]:
        """异步流式聊天接口"""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.0),
            "max_tokens": kwargs.get("max_tokens", 256),
            "top_p": kwargs.get("top_p", 1.0),
            "stream": True
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status != 200:
                        text = await response.text()
                        raise Exception(f"API 错误: {response.status} - {text}")
                    
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if line.startswith('data: '):
                            data_str = line[6:]
                            if data_str.strip() == '[DONE]':
                                break
                            try:
                                data = json.loads(data_str)
                                if 'choices' in data and len(data['choices']) > 0:
                                    delta = data['choices'][0].get('delta', {})
                                    if 'content' in delta:
                                        yield delta['content']
                            except json.JSONDecodeError:
                                continue
                                
        except Exception as e:
            raise Exception(f"异步流式请求失败: {e}")

def test_vllm_basic():
    """测试基本功能"""
    print("=== vLLM 基本功能测试 ===")
    
    client = SimpleVLLMClient(
        base_url="http://127.0.0.1:8001/v1",
        model="qwen2_5_0_5b"
    )
    
    try:
        # 测试连接
        if not client.is_available():
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
        
        print("✓ vLLM 服务连接成功")
        
        # 测试模型列表
        models = client.list_models()
        print(f"✓ 可用模型: {models}")
        
        # 测试聊天
        messages = [{"role": "user", "content": "你好，请简单介绍一下你自己。"}]
        response = client.chat(messages, max_tokens=128)
        print(f"✓ 聊天测试成功: {response[:100]}...")
        
        # 测试流式输出
        print("\n=== 流式输出测试 ===")
        messages = [{"role": "user", "content": "请用一句话介绍人工智能。"}]
        for chunk in client.stream(messages, max_tokens=64):
            print(chunk, end='', flush=True)
        print("\n✓ 流式输出测试完成")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False

async def test_vllm_async():
    """测试异步功能"""
    print("\n=== vLLM 异步功能测试 ===")
    
    client = SimpleVLLMClient(
        base_url="http://127.0.0.1:8001/v1",
        model="qwen2_5_0_5b"
    )
    
    try:
        if not client.is_available():
            print("✗ vLLM 服务不可用，跳过异步测试")
            return False
        
        # 测试异步聊天
        messages = [{"role": "user", "content": "请用一句话介绍机器学习。"}]
        response = await client.async_chat(messages, max_tokens=64)
        print(f"✓ 异步聊天测试成功: {response}")
        
        # 测试异步流式输出
        print("\n=== 异步流式输出测试 ===")
        messages = [{"role": "user", "content": "请简单解释什么是深度学习。"}]
        async for chunk in client.async_stream(messages, max_tokens=64):
            print(chunk, end='', flush=True)
        print("\n✓ 异步流式输出测试完成")
        
        return True
        
    except Exception as e:
        print(f"✗ 异步测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始 vLLM 独立测试...\n")
    
    # 基本功能测试
    basic_success = test_vllm_basic()
    
    # 异步功能测试
    async_success = asyncio.run(test_vllm_async())
    
    print("\n=== 测试总结 ===")
    print(f"基本功能: {'✓ 通过' if basic_success else '✗ 失败'}")
    print(f"异步功能: {'✓ 通过' if async_success else '✗ 失败'}")
    
    if basic_success and async_success:
        print("\n🎉 所有测试通过！vLLM 服务工作正常。")
        return 0
    else:
        print("\n❌ 部分测试失败，请检查 vLLM 服务状态。")
        return 1

if __name__ == "__main__":
    exit(main())