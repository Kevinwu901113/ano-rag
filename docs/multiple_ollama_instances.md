# 多端口运行多个 Ollama 实例

为了在本地实现并行处理，可以在不同端口启动多个 Ollama 服务，并在任务脚本中为每个并行任务分配独立端口。

## 启动多个服务
```bash
ollama serve --port 11434
ollama serve --port 11435
```

## 在脚本中分配端口
```python
from llm import LocalLLM

llm_a = LocalLLM(base_url="http://localhost:11434")
llm_b = LocalLLM(base_url="http://localhost:11435")
```

## 注意事项
- 每个服务都会独立加载模型，请确保机器拥有足够的 VRAM/内存。
- 参考 `examples/ollama_parallel_example.py` 了解如何使用 `ThreadPoolExecutor` 进行并行调用。
