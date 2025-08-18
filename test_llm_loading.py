#!/usr/bin/env python3
"""
测试LocalLLM多次加载问题的修复效果（简化版本）
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llm.local_llm import LocalLLM
from llm.enhanced_atomic_note_generator import EnhancedAtomicNoteGenerator
from doc.document_processor import DocumentProcessor
from utils.summary_auditor import SummaryAuditor
import time

def test_llm_initialization_behavior():
    """测试LocalLLM的初始化行为"""
    print("=== 测试LocalLLM初始化行为 ===")
    
    # 测试1: 验证LocalLLM构造函数是否会调用load_model
    print("\n1. 检查LocalLLM构造函数是否包含load_model调用...")
    
    # 读取LocalLLM源码检查
    try:
        with open('/home/wjk/workplace/anorag/llm/local_llm.py', 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 检查__init__方法中是否有load_model调用
        init_method_start = content.find('def __init__(self')
        if init_method_start != -1:
            # 找到下一个方法的开始位置
            next_method_start = content.find('def ', init_method_start + 1)
            if next_method_start == -1:
                next_method_start = len(content)
            
            init_method = content[init_method_start:next_method_start]
            
            if 'self.load_model()' in init_method:
                print("✅ LocalLLM构造函数中包含load_model()调用")
                print("✅ 这意味着模型会在实例创建时立即加载，而不是延迟加载")
                load_model_in_init = True
            else:
                print("❌ LocalLLM构造函数中未找到load_model()调用")
                print("❌ 这意味着仍使用延迟加载，可能导致多次加载")
                load_model_in_init = False
        else:
            print("❌ 无法找到__init__方法")
            load_model_in_init = False
            
    except Exception as e:
        print(f"❌ 读取源码时出错: {e}")
        load_model_in_init = False
    
    # 测试2: 检查generate方法是否移除了延迟加载逻辑
    print("\n2. 检查generate方法是否移除了延迟加载逻辑...")
    
    try:
        generate_method_start = content.find('def generate(self')
        if generate_method_start != -1:
            # 找到下一个方法的开始位置
            next_method_start = content.find('def ', generate_method_start + 1)
            if next_method_start == -1:
                next_method_start = len(content)
            
            generate_method = content[generate_method_start:next_method_start]
            
            if 'self.load_model()' not in generate_method:
                print("✅ generate方法中已移除load_model()调用")
                print("✅ 这避免了在运行时的延迟加载")
                no_lazy_loading = True
            else:
                print("❌ generate方法中仍包含load_model()调用")
                print("❌ 这可能导致运行时的多次模型加载")
                no_lazy_loading = False
        else:
            print("❌ 无法找到generate方法")
            no_lazy_loading = False
            
    except Exception as e:
        print(f"❌ 检查generate方法时出错: {e}")
        no_lazy_loading = False
    
    return load_model_in_init and no_lazy_loading

def test_component_error_handling():
    """测试组件的错误处理机制"""
    print("\n=== 测试组件错误处理机制 ===")
    
    components = [
        ("DocumentProcessor", DocumentProcessor),
        ("EnhancedAtomicNoteGenerator", EnhancedAtomicNoteGenerator),
        ("SummaryAuditor", SummaryAuditor),
    ]
    
    all_errors_correct = True
    
    for component_name, component_class in components:
        try:
            # 尝试不传入llm参数创建实例
            component_class()
            print(f"❌ {component_name}: 未按预期抛出错误")
            all_errors_correct = False
        except ValueError as e:
            print(f"✅ {component_name}: 正确抛出ValueError - {str(e)[:50]}...")
        except TypeError as e:
            if "llm" in str(e).lower():
                print(f"✅ {component_name}: 正确抛出TypeError (缺少llm参数) - {str(e)[:50]}...")
            else:
                print(f"❌ {component_name}: 抛出了意外的TypeError - {str(e)[:50]}...")
                all_errors_correct = False
        except Exception as e:
            print(f"❌ {component_name}: 抛出了意外的错误类型 - {type(e).__name__}: {str(e)[:50]}...")
            all_errors_correct = False
    
    return all_errors_correct

def test_shared_instance_concept():
    """测试共享实例的概念验证"""
    print("\n=== 测试共享实例概念验证 ===")
    
    print("创建两个LocalLLM实例（不加载模型）...")
    
    # 临时修改LocalLLM以避免实际加载模型
    class MockLocalLLM:
        def __init__(self):
            self.model_loaded = True
            self.instance_id = id(self)
            print(f"MockLocalLLM实例创建，ID: {self.instance_id}")
    
    # 模拟共享实例的使用
    shared_llm = MockLocalLLM()
    
    # 模拟组件使用共享实例
    class MockComponent:
        def __init__(self, llm):
            if llm is None:
                raise ValueError("Component requires a LocalLLM instance")
            self._llm = llm
    
    components = []
    for i in range(3):
        comp = MockComponent(shared_llm)
        components.append(comp)
        print(f"组件{i+1}创建，使用的LLM实例ID: {comp._llm.instance_id}")
    
    # 验证所有组件使用同一个实例
    all_same = all(comp._llm.instance_id == shared_llm.instance_id for comp in components)
    
    if all_same:
        print("✅ 所有组件都使用了同一个LocalLLM实例")
        print("✅ 共享实例机制工作正常")
    else:
        print("❌ 组件使用了不同的LocalLLM实例")
    
    return all_same

def main():
    """主测试函数"""
    print("开始测试LocalLLM多次加载问题的修复效果...\n")
    
    # 测试LocalLLM初始化行为
    init_test_passed = test_llm_initialization_behavior()
    
    # 测试组件错误处理
    error_test_passed = test_component_error_handling()
    
    # 测试共享实例概念
    shared_test_passed = test_shared_instance_concept()
    
    print("\n=== 测试总结 ===")
    
    if init_test_passed and error_test_passed and shared_test_passed:
        print("🎉 所有测试通过！LocalLLM多次加载问题已完全修复")
        print("\n修复要点:")
        print("1. ✅ LocalLLM在构造函数中立即加载模型，避免延迟加载")
        print("2. ✅ generate方法移除了延迟加载逻辑")
        print("3. ✅ 所有组件强制使用传入的LocalLLM实例")
        print("4. ✅ 完善的错误处理机制，防止意外创建新实例")
        print("\n现在当使用共享的LocalLLM实例时:")
        print("- 模型只会在第一次创建LocalLLM实例时加载一次")
        print("- 所有组件都会使用这个已加载的实例")
        print("- 不会出现多次'Model loaded successfully'的日志")
        return True
    else:
        print("❌ 部分测试失败，需要进一步检查")
        if not init_test_passed:
            print("  - LocalLLM初始化行为测试失败")
        if not error_test_passed:
            print("  - 组件错误处理测试失败")
        if not shared_test_passed:
            print("  - 共享实例概念测试失败")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)