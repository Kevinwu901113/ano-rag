#!/usr/bin/env python3
"""
示例：使用OpenAI API进行原子笔记生成

本示例展示如何配置和使用OpenAI API来生成原子笔记。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm import LocalLLM, OnlineLLM, OpenAIClient
from config import config
from loguru import logger


def example_openai_direct():
    """直接使用OpenAI客户端的示例"""
    print("=== 直接使用OpenAI客户端 ===")
    
    # 创建OpenAI客户端
    # 注意：需要设置环境变量 OPENAI_API_KEY 或在config.yaml中配置
    client = OpenAIClient(
        api_key="your-api-key-here",  # 替换为你的API密钥
        model="gpt-3.5-turbo"
    )
    
    # 检查可用性
    if not client.is_available():
        print("OpenAI客户端不可用，请检查API密钥配置")
        return
    
    # 生成文本
    prompt = "请解释什么是机器学习"
    system_prompt = "你是一个专业的AI助手，请用简洁明了的语言回答问题。"
    
    response = client.generate(prompt, system_prompt)
    print(f"生成的回答：{response}")
    
    # 批量生成
    prompts = [
        "什么是深度学习？",
        "什么是神经网络？",
        "什么是自然语言处理？"
    ]
    
    responses = client.batch_generate(prompts, system_prompt)
    for i, response in enumerate(responses):
        print(f"问题 {i+1}: {prompts[i]}")
        print(f"回答: {response}\n")


def example_online_llm():
    """使用OnlineLLM类的示例"""
    print("=== 使用OnlineLLM类 ===")
    
    # 创建在线LLM实例
    online_llm = OnlineLLM(
        provider="openai",
        model_name="gpt-3.5-turbo",
        api_key="your-api-key-here"  # 替换为你的API密钥
    )
    
    # 检查可用性
    if not online_llm.is_available():
        print("OnlineLLM不可用，请检查API密钥配置")
        return
    
    # 获取模型信息
    model_info = online_llm.get_model_info()
    print(f"模型信息：{model_info}")
    
    # 生成原子笔记
    text_chunks = [
        "机器学习是人工智能的一个分支，它使计算机能够在没有明确编程的情况下学习和改进。",
        "深度学习是机器学习的一个子集，它使用多层神经网络来模拟人脑的工作方式。",
        "自然语言处理是计算机科学和人工智能的一个分支，专注于计算机与人类语言之间的交互。"
    ]
    
    atomic_notes = online_llm.generate_atomic_notes(text_chunks)
    
    print("生成的原子笔记：")
    for i, note in enumerate(atomic_notes):
        print(f"笔记 {i+1}:")
        print(f"  内容: {note['content']}")
        print(f"  关键词: {note['keywords']}")
        print(f"  实体: {note['entities']}")
        print(f"  重要性评分: {note['importance_score']}")
        print(f"  笔记类型: {note['note_type']}\n")


def example_local_llm_with_openai():
    """使用LocalLLM类配置OpenAI的示例"""
    print("=== 使用LocalLLM类配置OpenAI ===")
    
    # 创建LocalLLM实例，指定OpenAI模型
    local_llm = LocalLLM(model_name="gpt-3.5-turbo")
    
    # 检查可用性
    if not local_llm.is_available():
        print("LocalLLM (OpenAI)不可用，请检查配置")
        return
    
    # 生成文本
    prompt = "请简要介绍人工智能的发展历史"
    response = local_llm.generate(prompt)
    print(f"生成的回答：{response}")
    
    # 提取实体和关系
    text = "苹果公司由史蒂夫·乔布斯、史蒂夫·沃兹尼亚克和罗纳德·韦恩于1976年创立。"
    entities_relations = local_llm.extract_entities_and_relations(text)
    print(f"提取的实体和关系：{entities_relations}")


def example_configuration():
    """配置示例"""
    print("=== 配置示例 ===")
    
    print("要使用OpenAI API，你需要：")
    print("1. 在config.yaml中配置：")
    print("   llm:")
    print("     openai:")
    print("       api_key: 'your-openai-api-key'")
    print("       model: 'gpt-3.5-turbo'")
    print("       temperature: 0.7")
    print("       max_tokens: 4096")
    print()
    print("2. 或者设置环境变量：")
    print("   export OPENAI_API_KEY='your-openai-api-key'")
    print()
    print("3. 在LocalLLM中使用OpenAI：")
    print("   # 方法1：通过模型名称自动识别")
    print("   llm = LocalLLM(model_name='gpt-3.5-turbo')")
    print()
    print("   # 方法2：通过配置provider")
    print("   # 在config.yaml中设置 llm.local_model.provider: 'openai'")
    print("   llm = LocalLLM()")
    print()
    print("4. 直接使用OnlineLLM：")
    print("   online_llm = OnlineLLM(provider='openai', model_name='gpt-3.5-turbo', api_key='your-key')")


if __name__ == "__main__":
    print("OpenAI API 集成示例\n")
    
    # 显示配置信息
    example_configuration()
    print("\n" + "="*50 + "\n")
    
    # 注意：以下示例需要有效的OpenAI API密钥才能运行
    # 请取消注释并提供有效的API密钥
    
    # example_openai_direct()
    # print("\n" + "="*50 + "\n")
    
    # example_online_llm()
    # print("\n" + "="*50 + "\n")
    
    # example_local_llm_with_openai()
    
    print("示例完成！")
    print("\n注意：要运行实际的API调用示例，请：")
    print("1. 获取OpenAI API密钥")
    print("2. 在代码中替换 'your-api-key-here'")
    print("3. 取消注释相应的示例函数调用")