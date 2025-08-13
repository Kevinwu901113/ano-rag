#!/usr/bin/env python3
"""
依赖安装脚本
自动检测系统环境并安装合适的依赖包
"""

import subprocess
import sys
import platform
import importlib.util
from pathlib import Path

def check_cuda_available():
    """检查CUDA是否可用"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def install_requirements():
    """安装基础依赖"""
    print("正在安装基础依赖...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def install_gpu_dependencies():
    """安装GPU相关依赖"""
    if platform.machine() != "x86_64":
        print("非x86_64架构，跳过GPU依赖安装")
        return
    
    if not check_cuda_available():
        print("CUDA不可用，跳过GPU依赖安装")
        return
    
    print("检测到CUDA环境，安装GPU加速依赖...")
    gpu_packages = [
        "cudf-cu12>=23.10.0",
        "cuml-cu12>=23.10.0", 
        "cugraph-cu12>=23.10.0",
        "cupy-cuda12x>=12.0.0"
    ]
    
    for package in gpu_packages:
        try:
            print(f"安装 {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError as e:
            print(f"警告: {package} 安装失败: {e}")
            print("您可以稍后手动安装GPU依赖")

def install_spacy_model():
    """安装spaCy模型"""
    try:
        print("安装spaCy英文模型...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    except subprocess.CalledProcessError as e:
        print(f"警告: spaCy模型安装失败: {e}")
        print("请手动运行: python -m spacy download en_core_web_sm")

def main():
    """主安装流程"""
    print("=== AnoRAG 依赖安装脚本 ===")
    print(f"Python版本: {sys.version}")
    print(f"系统架构: {platform.machine()}")
    print(f"操作系统: {platform.system()}")
    print()
    
    try:
        # 安装基础依赖
        install_requirements()
        
        # 安装GPU依赖（如果适用）
        install_gpu_dependencies()
        
        # 安装spaCy模型
        install_spacy_model()
        
        print("\n=== 安装完成 ===")
        print("所有依赖已成功安装！")
        
    except Exception as e:
        print(f"\n安装过程中出现错误: {e}")
        print("请检查错误信息并手动安装相关依赖")
        sys.exit(1)

if __name__ == "__main__":
    main()