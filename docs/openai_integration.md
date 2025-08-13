# OpenAI API 集成指南

本文档介绍如何在 anorag 项目中使用 OpenAI API 进行原子笔记生成和其他LLM任务。

## 概述

我们为 anorag 项目新增了对 OpenAI API 的支持，包括：

1. **OpenAIClient**: 直接的 OpenAI API 客户端
2. **OnlineLLM**: 统一的在线LLM接口，支持多种在线API服务
3. **LocalLLM**: 扩展了对 OpenAI API 的支持

## 安装依赖

确保安装了 OpenAI Python 库：

```bash
pip install openai>=1.0.0
```

或者使用项目的 requirements.txt：

```bash
pip install -r requirements.txt
```

## 配置

### 方法1: 配置文件

在 `config.yaml` 中添加 OpenAI 配置：

```yaml
llm:
  # OpenAI API Configuration
  openai:
    api_key: "your-openai-api-key"  # 你的 OpenAI API 密钥
    base_url: null                  # 可选：用于兼容的API服务
    model: "gpt-3.5-turbo"         # 默认模型
    temperature: 0.7
    max_tokens: 4096
    timeout: 60
    max_retries: 3
  
  # 在 local_model 中使用 OpenAI
  local_model:
    provider: "openai"              # 指定使用 OpenAI
    api_key: "your-openai-api-key" # API 密钥
    model: "gpt-3.5-turbo"         # 模型名称
    temperature: 0.1
```

### 方法2: 环境变量

设置环境变量：

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

## 使用方法

### 1. 直接使用 OpenAIClient

```python
from llm import OpenAIClient

# 创建客户端
client = OpenAIClient(
    api_key="your-api-key",
    model="gpt-3.5-turbo"
)

# 生成文本
response = client.generate(
    prompt="请解释什么是机器学习",
    system_prompt="你是一个专业的AI助手"
)
print(response)

# 批量生成
prompts = ["什么是深度学习？", "什么是神经网络？"]
responses = client.batch_generate(prompts)
```

### 2. 使用 OnlineLLM 类

```python
from llm import OnlineLLM

# 创建在线LLM实例
online_llm = OnlineLLM(
    provider="openai",
    model_name="gpt-3.5-turbo",
    api_key="your-api-key"
)

# 生成原子笔记
text_chunks = [
    "机器学习是人工智能的一个分支...",
    "深度学习是机器学习的一个子集..."
]

atomic_notes = online_llm.generate_atomic_notes(text_chunks)
for note in atomic_notes:
    print(f"内容: {note['content']}")
    print(f"关键词: {note['keywords']}")
    print(f"重要性: {note['importance_score']}")
```

### 3. 在 LocalLLM 中使用 OpenAI

```python
from llm import LocalLLM

# 方法1: 通过模型名称自动识别
llm = LocalLLM(model_name="gpt-3.5-turbo")

# 方法2: 通过配置文件指定 provider
# 在 config.yaml 中设置 llm.local_model.provider: "openai"
llm = LocalLLM()

# 使用
response = llm.generate("请介绍人工智能")
print(response)

# 生成原子笔记
text_chunks = ["你的文本内容..."]
atomic_notes = llm.generate_atomic_notes(text_chunks)
```

## 支持的模型

### OpenAI 官方模型
- `gpt-4`
- `gpt-4-turbo`
- `gpt-3.5-turbo`
- `text-davinci-003`
- 其他 OpenAI 模型

### 兼容的API服务

通过设置 `base_url` 参数，可以使用兼容 OpenAI API 的服务：

```python
client = OpenAIClient(
    api_key="your-key",
    model="your-model",
    base_url="https://your-compatible-api.com/v1"
)
```

## 功能特性

### 1. 原子笔记生成

支持将文本块转换为结构化的原子笔记：

```python
# 生成的原子笔记包含：
{
    'content': '笔记内容',
    'keywords': ['关键词1', '关键词2'],
    'entities': ['实体1', '实体2'],
    'concepts': ['概念1', '概念2'],
    'importance_score': 0.8,
    'note_type': 'fact'  # fact, concept, procedure, example
}
```

### 2. 实体关系提取

```python
text = "苹果公司由史蒂夫·乔布斯创立于1976年。"
result = llm.extract_entities_and_relations(text)
# 返回实体和关系的结构化数据
```

### 3. 批量处理

支持批量生成，提高处理效率：

```python
prompts = ["问题1", "问题2", "问题3"]
responses = client.batch_generate(prompts)
```

### 4. 错误处理

内置完善的错误处理机制：
- API 密钥验证
- 网络连接检查
- 速率限制处理
- 模型可用性检查

## 最佳实践

### 1. API 密钥安全

- 不要在代码中硬编码 API 密钥
- 使用环境变量或配置文件
- 定期轮换 API 密钥

### 2. 成本控制

```python
# 设置合理的 max_tokens 限制
client = OpenAIClient(
    model="gpt-3.5-turbo",
    max_tokens=1000  # 根据需要调整
)

# 使用更便宜的模型进行测试
test_client = OpenAIClient(model="gpt-3.5-turbo")
production_client = OpenAIClient(model="gpt-4")
```

### 3. 性能优化

```python
# 批量处理以减少API调用次数
text_chunks = ["文本1", "文本2", "文本3"]
atomic_notes = online_llm.generate_atomic_notes(text_chunks)

# 设置合理的超时时间
client = OpenAIClient(timeout=30)
```

## 故障排除

### 常见问题

1. **API 密钥错误**
   ```
   错误: API Key error: Invalid or missing OpenAI API key
   解决: 检查 API 密钥是否正确设置
   ```

2. **模型不存在**
   ```
   错误: Model error: Model 'xxx' not found
   解决: 检查模型名称是否正确
   ```

3. **速率限制**
   ```
   错误: Rate limit error: OpenAI API rate limit exceeded
   解决: 减少请求频率或升级API计划
   ```

### 调试技巧

```python
# 检查客户端可用性
if not client.is_available():
    print("客户端不可用，请检查配置")

# 获取模型信息
model_info = online_llm.get_model_info()
print(f"模型信息: {model_info}")

# 列出可用模型
available_models = client.list_models()
print(f"可用模型: {available_models}")
```

## 示例代码

完整的示例代码请参考 `examples/openai_example.py` 文件。

## 扩展支持

如需支持其他在线API服务（如 Anthropic Claude、Google PaLM 等），可以：

1. 创建对应的客户端类（参考 `OpenAIClient`）
2. 在 `OnlineLLM` 中添加新的 provider 支持
3. 更新配置文件和文档

## 贡献

欢迎提交 Issue 和 Pull Request 来改进 OpenAI 集成功能。