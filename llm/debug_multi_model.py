#!/usr/bin/env python3
"""
调试多模型客户端的实际行为
检查LM Studio中的模型实例和负载均衡情况
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import requests
import time
from loguru import logger
from config import config
from llm.multi_model_client import MultiModelClient

def check_lmstudio_models_detailed():
    """详细检查LM Studio中的模型"""
    logger.info("详细检查LM Studio模型...")
    
    base_port = config.get("llm.multi_model.base_port", 1234)
    
    try:
        response = requests.get(f"http://localhost:{base_port}/v1/models", timeout=10)
        if response.status_code == 200:
            models_data = response.json()
            logger.info(f"原始响应数据: {models_data}")
            
            models = models_data.get("data", [])
            logger.info(f"找到 {len(models)} 个模型:")
            
            for i, model in enumerate(models):
                logger.info(f"  模型 {i+1}:")
                logger.info(f"    ID: {model.get('id', 'N/A')}")
                logger.info(f"    Object: {model.get('object', 'N/A')}")
                logger.info(f"    Created: {model.get('created', 'N/A')}")
                logger.info(f"    Owned by: {model.get('owned_by', 'N/A')}")
            
            return [model["id"] for model in models]
        else:
            logger.error(f"HTTP错误: {response.status_code}")
            logger.error(f"响应内容: {response.text}")
            return []
    except Exception as e:
        logger.error(f"连接LM Studio失败: {e}")
        return []

def test_model_name_matching():
    """测试模型名称匹配"""
    logger.info("测试模型名称匹配...")
    
    # 获取实际的模型列表
    actual_models = check_lmstudio_models_detailed()
    
    # 获取配置的模型名称
    model_count = config.get("llm.multi_model.instance_count", 2)
    base_model_name = config.get("llm.multi_model.model_name", "gpt-oss-20b")
    
    expected_models = []
    for i in range(model_count):
        if i == 0:
            model_name = base_model_name
        else:
            model_name = f"{base_model_name}:{i+1}"
        expected_models.append(model_name)
    
    logger.info(f"期望的模型名称: {expected_models}")
    logger.info(f"实际的模型名称: {actual_models}")
    
    # 检查匹配情况
    for expected in expected_models:
        if expected in actual_models:
            logger.info(f"✓ 模型 '{expected}' 匹配成功")
        else:
            logger.warning(f"✗ 模型 '{expected}' 未找到")
            # 检查是否有相似的模型名
            similar = [m for m in actual_models if base_model_name in m]
            if similar:
                logger.info(f"  相似的模型: {similar}")

def test_multi_model_initialization():
    """测试多模型初始化"""
    logger.info("测试多模型初始化...")
    
    try:
        client = MultiModelClient()
        status = client.get_status()
        
        logger.info(f"总实例数: {status['total_instances']}")
        logger.info(f"健康实例数: {status['healthy_instances']}")
        logger.info(f"负载均衡策略: {status['load_balancing_strategy']}")
        
        # 显示每个实例的详细信息
        for i, instance_info in enumerate(status['instances']):
            logger.info(f"实例 {i+1}:")
            logger.info(f"  模型名: {instance_info['model_name']}")
            logger.info(f"  端口: {instance_info['port']}")
            logger.info(f"  健康状态: {instance_info['is_healthy']}")
            logger.info(f"  活跃请求: {instance_info['active_requests']}")
            logger.info(f"  总请求数: {instance_info['total_requests']}")
        
        return client
        
    except Exception as e:
        logger.error(f"多模型初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_actual_generation(client):
    """测试实际的文本生成"""
    if not client:
        logger.error("客户端未初始化，跳过生成测试")
        return
    
    logger.info("测试实际文本生成...")
    
    try:
        # 发送几个测试请求
        test_prompts = [
            "你好，请简单介绍一下自己。",
            "1+1等于多少？",
            "请用一句话描述人工智能。"
        ]
        
        for i, prompt in enumerate(test_prompts):
            logger.info(f"\n发送请求 {i+1}: {prompt}")
            try:
                response = client.generate(prompt)
                logger.info(f"响应: {response[:100]}..." if len(response) > 100 else f"响应: {response}")
            except Exception as e:
                logger.error(f"请求 {i+1} 失败: {e}")
        
        # 检查最终状态
        final_status = client.get_status()
        logger.info("\n最终状态:")
        for i, instance_info in enumerate(final_status['instances']):
            logger.info(f"实例 {i+1}: 总请求数 {instance_info['total_requests']}, 错误数 {instance_info['error_count']}")
            
    except Exception as e:
        logger.error(f"生成测试失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    logger.info("开始详细调试多模型客户端...")
    
    # 检查配置
    multi_model_enabled = config.get("llm.multi_model.enabled", False)
    logger.info(f"多模型配置启用: {multi_model_enabled}")
    
    if not multi_model_enabled:
        logger.error("多模型功能未启用，请检查配置文件")
        return
    
    print("\n" + "="*60)
    logger.info("步骤1: 检查LM Studio模型详情")
    check_lmstudio_models_detailed()
    
    print("\n" + "="*60)
    logger.info("步骤2: 测试模型名称匹配")
    test_model_name_matching()
    
    print("\n" + "="*60)
    logger.info("步骤3: 测试多模型初始化")
    client = test_multi_model_initialization()
    
    print("\n" + "="*60)
    logger.info("步骤4: 测试实际文本生成")
    test_actual_generation(client)

if __name__ == "__main__":
    main()