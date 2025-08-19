#!/usr/bin/env python3
"""
vLLM 健康检查脚本

从配置文件读取 vLLM 路由配置，逐个探测并打印状态
"""

import sys
import os
import json
import requests
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import config
from loguru import logger


def check_health(base_url: str, timeout: int = 5) -> dict:
    """检查单个 vLLM 实例的健康状态
    
    Args:
        base_url: vLLM 实例的 base_url
        timeout: 超时时间
        
    Returns:
        dict: 健康检查结果
    """
    result = {
        'base_url': base_url,
        'healthy': False,
        'models': [],
        'error': None,
        'response_time': None
    }
    
    try:
        # 移除 /v1 后缀以避免重复
        clean_url = base_url.rstrip('/v1').rstrip('/')
        models_url = f"{clean_url}/v1/models"
        
        import time
        start_time = time.time()
        
        response = requests.get(models_url, timeout=timeout)
        response_time = time.time() - start_time
        result['response_time'] = round(response_time * 1000, 2)  # ms
        
        if response.status_code == 200:
            data = response.json()
            if 'data' in data and len(data['data']) > 0:
                result['healthy'] = True
                result['models'] = [model['id'] for model in data['data']]
            else:
                result['error'] = "No models available"
        else:
            result['error'] = f"HTTP {response.status_code}: {response.text}"
            
    except requests.exceptions.Timeout:
        result['error'] = f"Timeout after {timeout}s"
    except requests.exceptions.ConnectionError:
        result['error'] = "Connection refused"
    except Exception as e:
        result['error'] = str(e)
    
    return result


def main():
    """主函数"""
    print("vLLM 健康检查工具")
    print("=" * 50)
    
    # 获取路由配置
    routes = config.get('llm.routes', {})
    if not routes:
        print("❌ 未找到 vLLM 路由配置")
        sys.exit(1)
    
    default_route = config.get('llm.default_route', 'gpt20_a')
    fallback_route = config.get('llm.fallback_route')
    
    print(f"默认路由: {default_route}")
    if fallback_route:
        print(f"回退路由: {fallback_route}")
    print(f"总路由数: {len(routes)}")
    print()
    
    # 检查每个路由
    results = []
    healthy_count = 0
    
    for route_name, route_config in routes.items():
        base_url = route_config.get('base_url')
        model = route_config.get('model')
        timeout = route_config.get('timeout', 5)
        
        print(f"检查路由: {route_name}")
        print(f"  URL: {base_url}")
        print(f"  模型: {model}")
        
        result = check_health(base_url, timeout)
        results.append({
            'route_name': route_name,
            'expected_model': model,
            **result
        })
        
        if result['healthy']:
            healthy_count += 1
            status = "✅ 健康"
            if result['response_time']:
                status += f" ({result['response_time']}ms)"
            print(f"  状态: {status}")
            print(f"  可用模型: {', '.join(result['models'])}")
            
            # 检查期望的模型是否可用
            if model not in result['models']:
                print(f"  ⚠️  警告: 期望的模型 '{model}' 不在可用模型列表中")
        else:
            print(f"  状态: ❌ 不健康")
            print(f"  错误: {result['error']}")
        
        print()
    
    # 总结
    print("=" * 50)
    print(f"健康检查完成: {healthy_count}/{len(routes)} 个路由健康")
    
    if healthy_count == 0:
        print("❌ 所有路由都不可用！")
        sys.exit(1)
    elif healthy_count < len(routes):
        print("⚠️  部分路由不可用")
    else:
        print("✅ 所有路由都健康")
    
    # 输出 JSON 格式结果（便于脚本调用）
    if '--json' in sys.argv:
        print("\n" + "=" * 50)
        print("JSON 结果:")
        print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()