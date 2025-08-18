#!/usr/bin/env python3
"""
vLLM Provider 使用示例

展示如何在 anorag 中使用 vLLM provider
"""

import asyncio
import time
from loguru import logger

# 导入 anorag 组件
from llm.factory import LLMFactory
from llm.providers.vllm_openai import VLLMOpenAIProvider
from config import config


def example_direct_usage():
    """直接使用 vLLM Provider 的示例"""
    print("\n=== 直接使用 vLLM Provider ===")
    
    # 直接创建 vLLM provider
    provider = VLLMOpenAIProvider(
        base_url="http://127.0.0.1:8001/v1",
        model="qwen2_5_0_5b",
        api_key="EMPTY",
        temperature=0.1,
        max_tokens=512
    )
    
    # 检查可用性
    if not provider.is_available():
        print("❌ vLLM 服务不可用，请先启动服务")
        return False
    
    print("✅ vLLM 服务可用")
    
    # 1. 简单文本生成
    print("\n1. 简单文本生成:")
    response = provider.generate("请简单介绍一下人工智能的发展历程。")
    print(f"回复: {response}")
    
    # 2. 聊天对话
    print("\n2. 聊天对话:")
    messages = [
        {"role": "system", "content": "你是一个有用的AI助手。"},
        {"role": "user", "content": "什么是机器学习？"}
    ]
    response = provider.chat(messages)
    print(f"回复: {response}")
    
    # 3. 流式输出
    print("\n3. 流式输出:")
    messages = [
        {"role": "user", "content": "请写一首关于春天的短诗。"}
    ]
    print("流式回复: ", end="")
    for chunk in provider.stream(messages):
        print(chunk, end="", flush=True)
    print("\n")
    
    return True


def example_factory_usage():
    """使用工厂模式的示例"""
    print("\n=== 使用工厂模式 ===")
    
    try:
        # 通过工厂创建 provider
        provider = LLMFactory.create_provider('vllm_openai')
        
        # 使用配置中的路由
        provider_with_route = LLMFactory.create_provider('vllm_openai', route_name='tiny_qwen')
        
        print(f"✅ 成功创建 vLLM provider")
        print(f"模型信息: {provider.get_model_info()}")
        
        # 测试生成
        response = provider.generate("请用一句话总结深度学习的核心思想。")
        print(f"回复: {response}")
        
        return True
        
    except Exception as e:
        print(f"❌ 工厂模式创建失败: {e}")
        return False


async def example_async_usage():
    """异步使用示例"""
    print("\n=== 异步使用示例 ===")
    
    provider = VLLMOpenAIProvider(
        base_url="http://127.0.0.1:8001/v1",
        model="qwen2_5_0_5b",
        api_key="EMPTY"
    )
    
    if not provider.is_available():
        print("❌ vLLM 服务不可用")
        return False
    
    # 并发请求测试
    questions = [
        "什么是人工智能？",
        "机器学习有哪些类型？",
        "深度学习的优势是什么？",
        "自然语言处理的应用有哪些？",
        "计算机视觉技术如何工作？"
    ]
    
    print(f"并发处理 {len(questions)} 个问题...")
    start_time = time.time()
    
    # 并发执行
    tasks = [provider.async_generate(q) for q in questions]
    responses = await asyncio.gather(*tasks)
    
    end_time = time.time()
    
    print(f"总耗时: {end_time - start_time:.2f}秒")
    print(f"平均耗时: {(end_time - start_time) / len(questions):.2f}秒/问题")
    
    for i, (question, response) in enumerate(zip(questions, responses), 1):
        print(f"\n问题 {i}: {question}")
        print(f"回答 {i}: {response[:100]}..." if len(response) > 100 else f"回答 {i}: {response}")
    
    return True


def example_integration_with_anorag():
    """与 anorag 系统集成的示例"""
    print("\n=== 与 anorag 系统集成 ===")
    
    try:
        # 模拟原子笔记生成
        from llm.atomic_note_generator import AtomicNoteGenerator
        
        # 创建原子笔记生成器，使用 vLLM
        # 注意：这需要修改 AtomicNoteGenerator 以支持工厂模式
        print("模拟原子笔记生成...")
        
        provider = LLMFactory.create_provider('vllm_openai')
        
        # 示例文档
        document = """
        人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，
        它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
        该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。
        """
        
        # 生成原子笔记
        prompt = f"""请将以下文档分解为原子笔记，每个笔记应该包含一个独立的概念或事实：
        
        文档：{document}
        
        请以JSON格式返回，包含以下字段：
        - title: 笔记标题
        - content: 笔记内容
        - keywords: 关键词列表
        """
        
        response = provider.generate(prompt, max_tokens=1024)
        print(f"原子笔记生成结果:\n{response}")
        
        return True
        
    except Exception as e:
        print(f"❌ 集成测试失败: {e}")
        return False


def example_performance_comparison():
    """性能对比示例"""
    print("\n=== 性能对比示例 ===")
    
    # 获取可用的 providers
    available_providers = LLMFactory.get_available_providers()
    print(f"可用的 providers: {available_providers}")
    
    test_prompt = "请简单解释什么是深度学习。"
    results = {}
    
    for provider_name, is_available in available_providers.items():
        if not is_available:
            print(f"⏭️  跳过不可用的 provider: {provider_name}")
            continue
        
        try:
            print(f"\n测试 {provider_name}...")
            provider = LLMFactory.create_provider(provider_name)
            
            start_time = time.time()
            response = provider.generate(test_prompt)
            end_time = time.time()
            
            results[provider_name] = {
                'latency': end_time - start_time,
                'response_length': len(response),
                'response': response[:100] + '...' if len(response) > 100 else response
            }
            
            print(f"✅ {provider_name} - 延迟: {results[provider_name]['latency']:.2f}s")
            
        except Exception as e:
            print(f"❌ {provider_name} 测试失败: {e}")
            results[provider_name] = {'error': str(e)}
    
    # 输出对比结果
    print("\n📊 性能对比结果:")
    print("-" * 60)
    for provider_name, result in results.items():
        if 'error' in result:
            print(f"{provider_name:<15} ❌ {result['error']}")
        else:
            print(f"{provider_name:<15} ⏱️  {result['latency']:.2f}s  📝 {result['response_length']} chars")
    
    return True


def main():
    """主函数"""
    print("🚀 vLLM Provider 使用示例")
    print("=" * 50)
    
    # 检查配置
    print(f"当前配置的 provider: {config.get('llm.provider', 'not set')}")
    print(f"默认路由: {config.get('llm.default_route', 'not set')}")
    
    results = []
    
    # 1. 直接使用示例
    results.append(("直接使用", example_direct_usage()))
    
    # 2. 工厂模式示例
    results.append(("工厂模式", example_factory_usage()))
    
    # 3. 异步使用示例
    results.append(("异步使用", asyncio.run(example_async_usage())))
    
    # 4. 系统集成示例
    results.append(("系统集成", example_integration_with_anorag()))
    
    # 5. 性能对比示例
    results.append(("性能对比", example_performance_comparison()))
    
    # 输出总结
    print("\n" + "=" * 50)
    print("📋 示例运行结果:")
    for name, success in results:
        status = "✅ 成功" if success else "❌ 失败"
        print(f"  {name:<12} {status}")
    
    successful_count = sum(results[i][1] for i in range(len(results)))
    print(f"\n总体结果: {successful_count}/{len(results)} 个示例成功")
    
    if successful_count == len(results):
        print("🎉 所有示例运行成功！vLLM 集成完成")
    else:
        print("⚠️  部分示例失败，请检查 vLLM 服务状态")
        print("\n💡 启动 vLLM 服务:")
        print("./scripts/start_vllm.sh start-tiny")
        print("\n💡 测试服务:")
        print("./scripts/start_vllm.sh test")


if __name__ == "__main__":
    main()