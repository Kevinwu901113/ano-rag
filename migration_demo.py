#!/usr/bin/env python3
"""
Anorag LLM 迁移演示脚本
展示从 Ollama 到 vLLM 的完整迁移流程
"""

import sys
import os
import yaml
from pathlib import Path
from typing import Dict, Any

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def load_config() -> Dict[str, Any]:
    """加载配置文件"""
    config_path = project_root / "config.yaml"
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return {}
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def show_config_migration():
    """展示配置迁移"""
    print("=== 配置迁移演示 ===")
    print("\n1. 原始 Ollama 配置:")
    print("```yaml")
    print("llm:")
    print("  provider: ollama")
    print("  ollama:")
    print("    base_url: http://localhost:11434")
    print("    model: qwen2.5:0.5b")
    print("```")
    
    print("\n2. 新增 vLLM 配置:")
    print("```yaml")
    print("llm:")
    print("  provider: vllm_openai  # 切换到 vLLM")
    print("  vllm_openai:")
    print("    routes:")
    print("      tiny_qwen:")
    print("        base_url: http://127.0.0.1:8001/v1")
    print("        model: qwen2_5_0_5b")
    print("      large_qwen:")
    print("        base_url: http://127.0.0.1:8002/v1")
    print("        model: qwen2_5_7b")
    print("    default_route: tiny_qwen")
    print("    params:")
    print("      temperature: 0.0")
    print("      max_tokens: 2048")
    print("```")
    
    # 显示当前配置
    config = load_config()
    if config:
        print("\n3. 当前项目配置:")
        llm_config = config.get('llm', {})
        provider = llm_config.get('provider', 'unknown')
        print(f"   当前提供商: {provider}")
        
        if provider == 'vllm_openai':
            vllm_config = llm_config.get('vllm_openai', {})
            routes = vllm_config.get('routes', {})
            default_route = vllm_config.get('default_route', 'unknown')
            print(f"   默认路由: {default_route}")
            print(f"   可用路由: {list(routes.keys())}")
            print("   ✅ vLLM 配置已就绪")
        else:
            print("   ⚠️  当前使用传统配置")

def show_provider_architecture():
    """展示提供商架构"""
    print("\n=== 提供商架构演示 ===")
    print("\n1. 工厂模式设计:")
    print("   📁 llm/factory.py - LLM 工厂类")
    print("   📁 llm/providers/ - 提供商实现")
    print("   ├── __init__.py")
    print("   └── vllm_openai.py - vLLM OpenAI 兼容提供商")
    
    print("\n2. 提供商注册:")
    print("   - OllamaClient (传统)")
    print("   - MultiOllamaClient (多实例)")
    print("   - OpenAIClient (OpenAI API)")
    print("   - VLLMOpenAIProvider (新增 vLLM)")
    
    print("\n3. 自动回退机制:")
    print("   vLLM 不可用 → 自动回退到 Ollama")
    print("   确保服务连续性和兼容性")

def show_usage_examples():
    """展示使用示例"""
    print("\n=== 使用示例演示 ===")
    
    print("\n1. 直接使用 vLLM Provider:")
    print("```python")
    print("from llm.providers.vllm_openai import VLLMOpenAIProvider")
    print("")
    print("provider = VLLMOpenAIProvider(")
    print("    base_url='http://127.0.0.1:8001/v1',")
    print("    model='qwen2_5_0_5b'")
    print(")")
    print("")
    print("response = provider.chat([")
    print("    {'role': 'user', 'content': '你好'}")
    print("])")
    print("```")
    
    print("\n2. 通过工厂模式使用:")
    print("```python")
    print("from llm.factory import LLMFactory")
    print("")
    print("provider = LLMFactory.create_provider('vllm_openai', config)")
    print("response = provider.chat(messages)")
    print("```")
    
    print("\n3. 集成到 LocalLLM:")
    print("```python")
    print("from llm import LocalLLM")
    print("")
    print("# 配置文件中设置 provider: vllm_openai")
    print("llm = LocalLLM(provider='vllm_openai')")
    print("response = llm.generate('你好，请介绍一下你自己')")
    print("```")

def show_performance_benefits():
    """展示性能优势"""
    print("\n=== 性能优势演示 ===")
    
    print("\n1. vLLM 优势:")
    print("   🚀 更高的推理吞吐量")
    print("   ⚡ 更低的延迟")
    print("   🔧 更好的 GPU 利用率")
    print("   📊 支持批处理推理")
    print("   🔄 动态批处理")
    
    print("\n2. 兼容性:")
    print("   ✅ OpenAI API 完全兼容")
    print("   ✅ 支持流式输出")
    print("   ✅ 支持异步调用")
    print("   ✅ 支持多模型路由")
    
    print("\n3. 压测脚本:")
    print("   📄 benchmark_vllm.py - 性能测试")
    print("   📊 支持并发测试")
    print("   📈 延迟和吞吐量统计")
    print("   🔍 GPU 利用率监控")

def show_deployment_guide():
    """展示部署指南"""
    print("\n=== 部署指南演示 ===")
    
    print("\n1. 启动 vLLM 服务:")
    print("```bash")
    print("# 使用启动脚本")
    print("./scripts/start_vllm.sh start-tiny    # 启动小模型")
    print("./scripts/start_vllm.sh start-medium  # 启动中等模型")
    print("./scripts/start_vllm.sh status        # 检查状态")
    print("```")
    
    print("\n2. 手动启动:")
    print("```bash")
    print("python -m vllm.entrypoints.openai.api_server \\")
    print("  --model Qwen/Qwen2.5-0.5B-Instruct \\")
    print("  --served-model-name qwen2_5_0_5b \\")
    print("  --dtype float16 \\")
    print("  --max-model-len 4096 \\")
    print("  --gpu-memory-utilization 0.80 \\")
    print("  --port 8001")
    print("```")
    
    print("\n3. 多 GPU 部署:")
    print("```bash")
    print("./scripts/start_vllm.sh start-large \\")
    print("  --tensor-parallel 2 \\")
    print("  --gpu-memory 0.90")
    print("```")

def show_testing_guide():
    """展示测试指南"""
    print("\n=== 测试指南演示 ===")
    
    print("\n1. 独立测试:")
    print("```bash")
    print("python test_vllm_standalone.py  # 独立 API 测试")
    print("```")
    
    print("\n2. 集成测试:")
    print("```bash")
    print("python examples/vllm_example.py  # 完整集成示例")
    print("```")
    
    print("\n3. 性能测试:")
    print("```bash")
    print("python benchmark_vllm.py \\")
    print("  --base-url http://127.0.0.1:8001/v1 \\")
    print("  --model qwen2_5_0_5b \\")
    print("  --concurrency 10 \\")
    print("  --requests 100")
    print("```")

def show_migration_checklist():
    """展示迁移检查清单"""
    print("\n=== 迁移检查清单 ===")
    
    checklist = [
        ("✅", "创建 vLLM Provider", "llm/providers/vllm_openai.py"),
        ("✅", "注册到工厂模式", "llm/factory.py"),
        ("✅", "更新配置文件", "config.yaml"),
        ("✅", "集成到 LocalLLM", "llm/local_llm.py"),
        ("✅", "创建启动脚本", "scripts/start_vllm.sh"),
        ("✅", "编写测试脚本", "test_vllm_*.py"),
        ("✅", "创建压测工具", "benchmark_vllm.py"),
        ("✅", "编写使用示例", "examples/vllm_example.py"),
        ("⚠️", "启动 vLLM 服务", "需要网络连接下载模型"),
        ("⚠️", "性能验证", "需要 vLLM 服务运行")
    ]
    
    for status, task, note in checklist:
        print(f"   {status} {task:<20} - {note}")
    
    print("\n📝 注意事项:")
    print("   - 确保有足够的 GPU 内存")
    print("   - 网络连接用于下载模型")
    print("   - 配置文件备份")
    print("   - 渐进式迁移测试")

def main():
    """主演示函数"""
    print("🚀 Anorag LLM 迁移演示")
    print("从 Ollama 到 vLLM 的完整迁移流程\n")
    
    # 展示各个部分
    show_config_migration()
    show_provider_architecture()
    show_usage_examples()
    show_performance_benefits()
    show_deployment_guide()
    show_testing_guide()
    show_migration_checklist()
    
    print("\n🎉 迁移演示完成！")
    print("\n📚 相关文件:")
    print("   - llm/providers/vllm_openai.py - vLLM 提供商实现")
    print("   - llm/factory.py - LLM 工厂模式")
    print("   - config.yaml - 配置文件")
    print("   - scripts/start_vllm.sh - 启动脚本")
    print("   - benchmark_vllm.py - 性能测试")
    print("   - examples/vllm_example.py - 使用示例")
    
    print("\n🔧 下一步:")
    print("   1. 确保网络连接")
    print("   2. 启动 vLLM 服务")
    print("   3. 运行测试脚本")
    print("   4. 执行性能对比")
    print("   5. 生产环境部署")

if __name__ == "__main__":
    main()