#!/usr/bin/env python3
"""
示例：在不同端口运行多个 Ollama 实例并行生成文本

使用前请先在多个端口启动 Ollama 服务，例如：
    ollama serve --port 11434
    ollama serve --port 11435

每个服务都会独立加载模型，请确保机器具有足够的 VRAM/内存。
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor

# 允许脚本直接运行
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm import LocalLLM


def generate_with_port(port: int, prompt: str) -> str:
    """使用指定端口的 Ollama 服务生成文本"""
    llm = LocalLLM(base_url=f"http://localhost:{port}")
    return llm.generate(prompt)


def main() -> None:
    prompts = [
        "简要介绍机器学习",  # 将由11434端口处理
        "解释什么是深度学习"  # 将由11435端口处理
    ]
    ports = [11434, 11435]

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(generate_with_port, port, prompt)
            for port, prompt in zip(ports, prompts)
        ]
        for idx, future in enumerate(futures, start=1):
            print(f"任务{idx} (端口 {ports[idx-1]}) 的结果：{future.result()}")


if __name__ == "__main__":
    main()
