# Provider切换指南

本指南介绍如何在OpenAI和Ollama之间轻松切换，只需修改一个配置项即可。

## 快速切换

### 切换到OpenAI
在 `config.yaml` 中修改：
```yaml
llm:
  local_model:
    provider: "openai"  # 设置为openai
```

### 切换到Ollama
在 `config.yaml` 中修改：
```yaml
llm:
  local_model:
    provider: "ollama"  # 设置为ollama
```

## 配置结构

### 完整配置示例
```yaml
llm:
  # 主要配置 - 只需修改provider即可切换
  local_model:
    provider: "openai"  # 或 "ollama"
    temperature: 0.1    # 可选：覆盖具体provider的设置
    max_tokens: 4096    # 可选：覆盖具体provider的设置
    
  # Ollama特定配置
  ollama:
    base_url: "http://localhost:11434"
    model: "qwen2.5:7b-instruct-fp16"
    temperature: 0.7
    max_tokens: 4096
    num_ctx: 32768
    max_async: 4
    timeout: 60
    
  # OpenAI特定配置
  openai:
    api_key: "your-api-key-here"
    base_url: "https://api.deepseek.com/v1"  # 可选：用于兼容API
    model: "deepseek-chat"
    temperature: 0.7
    max_tokens: 4096
    timeout: 60
    max_retries: 3
```

## 配置优先级

1. **local_model** 中的设置具有最高优先级
2. 具体provider段（ollama/openai）中的设置作为默认值

例如：
- 如果 `local_model.temperature = 0.1` 且 `openai.temperature = 0.7`
- 使用OpenAI时，实际temperature为 `0.1`（local_model覆盖）

## 使用示例

### Python代码
```python
from llm.local_llm import LocalLLM

# 自动根据配置选择provider
llm = LocalLLM()
llm.load_model()

# 生成文本
response = llm.generate("Hello, how are you?")
print(response)

# 检查当前使用的provider
print(f"当前provider: {llm.provider}")
print(f"模型名称: {llm.model_name}")
```

### 验证切换
运行测试脚本验证切换是否成功：
```bash
python test_provider_switch.py
```

## 注意事项

1. **API密钥安全**：确保OpenAI API密钥安全存储
2. **网络连接**：OpenAI需要网络连接，Ollama可以本地运行
3. **模型可用性**：确保配置的模型在对应服务中可用
4. **成本控制**：OpenAI按使用量计费，注意控制成本

## 故障排除

### 常见问题

1. **OpenAI连接失败**
   - 检查API密钥是否正确
   - 检查网络连接
   - 验证base_url是否正确

2. **Ollama连接失败**
   - 确保Ollama服务正在运行
   - 检查base_url和端口
   - 验证模型是否已下载

3. **模型不存在**
   - 检查模型名称是否正确
   - 对于Ollama，使用 `ollama list` 查看可用模型
   - 对于OpenAI，确认模型名称符合API规范

### 调试命令
```bash
# 测试当前配置
python test_provider_switch.py

# 检查Ollama服务
curl http://localhost:11434/api/tags

# 测试OpenAI连接
python -c "from llm.openai_client import OpenAIClient; print(OpenAIClient().is_available())"
```

## 最佳实践

1. **开发环境**：使用Ollama进行本地开发和测试
2. **生产环境**：根据需求选择OpenAI或自托管模型
3. **配置管理**：使用环境变量管理敏感信息
4. **监控**：监控API使用量和成本
5. **备份方案**：配置多个provider作为备选