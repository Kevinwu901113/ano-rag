# vLLM OpenAI 兼容 Provider 使用指南

本文档介绍如何在 anorag 项目中使用 vLLM OpenAI 兼容 Provider，包括配置、部署、测试和故障排除。

## 概述

vLLM OpenAI 兼容 Provider 是一个高性能的 LLM 推理服务接口，支持：

- ✅ OpenAI 兼容的 API 接口
- ✅ 多路由负载均衡
- ✅ 健康检查与自动故障转移
- ✅ 同步和异步调用
- ✅ 流式响应
- ✅ 重试机制与指数退避
- ✅ 详细的性能监控

## 快速开始

### 1. 启动 vLLM 服务

#### 单实例启动（推荐用于开发）

```bash
# 启动 gpt-oss-20b 模型
python -m vllm.entrypoints.openai.api_server \
  --model openai/gpt-oss-20b \
  --served-model-name gpt_oss_20b \
  --dtype float16 \
  --tensor-parallel-size 2 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --port 8002
```

#### 多实例启动（用于生产负载均衡）

```bash
# 实例 A (端口 8002)
python -m vllm.entrypoints.openai.api_server \
  --model openai/gpt-oss-20b \
  --served-model-name gpt_oss_20b \
  --dtype float16 \
  --tensor-parallel-size 2 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --port 8002 &

# 实例 B (端口 8202)
python -m vllm.entrypoints.openai.api_server \
  --model openai/gpt-oss-20b \
  --served-model-name gpt_oss_20b \
  --dtype float16 \
  --tensor-parallel-size 2 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --port 8202 &
```

#### 启动参数说明

- `--model`: 模型路径或 HuggingFace 模型名
- `--served-model-name`: API 中使用的模型名称
- `--dtype`: 模型精度，推荐 `float16` 或 `bfloat16`
- `--tensor-parallel-size`: 张量并行大小（GPU 数量）
- `--max-model-len`: 最大序列长度
- `--gpu-memory-utilization`: GPU 显存利用率
- `--port`: 服务端口

#### 显存优化建议

```bash
# 单卡或显存紧张时
python -m vllm.entrypoints.openai.api_server \
  --model openai/gpt-oss-20b \
  --served-model-name gpt_oss_20b \
  --dtype float16 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.85 \
  --kv-cache-dtype fp8 \
  --port 8002
```

### 2. 配置 config.yaml

#### 基础配置

```yaml
llm:
  # 使用 vLLM Provider
  provider: vllm_openai
  
  # 路由配置
  routes:
    gpt20_a:
      base_url: http://127.0.0.1:8002/v1
      model: gpt_oss_20b
      timeout: 60
  
  # 默认路由
  default_route: gpt20_a
  
  # 负载均衡策略
  lb_policy: round_robin
  
  # 全局参数
  params:
    temperature: 0.0
    max_tokens: 1024
    top_p: 1.0
    timeout: 60
    max_retries: 3
```

#### 多实例负载均衡配置

```yaml
llm:
  provider: vllm_openai
  
  routes:
    gpt20_a:
      base_url: http://127.0.0.1:8002/v1
      model: gpt_oss_20b
      timeout: 60
    gpt20_b:
      base_url: http://127.0.0.1:8202/v1
      model: gpt_oss_20b
      timeout: 60
    # 备用小模型
    tiny_qwen:
      base_url: http://127.0.0.1:8001/v1
      model: qwen2_5_0_5b
      timeout: 60
  
  default_route: gpt20_a
  fallback_route: tiny_qwen  # 主路由失败时的回退
  lb_policy: least_latency   # 选择延迟最低的实例
  
  params:
    temperature: 0.0
    max_tokens: 1024
    top_p: 1.0
    timeout: 60
    max_retries: 3
```

### 3. 健康检查

运行健康检查脚本验证所有实例状态：

```bash
# 基础健康检查
python scripts/healthcheck_vllm.py

# 输出 JSON 格式结果
python scripts/healthcheck_vllm.py --json
```

期望输出：
```
vLLM 健康检查工具
==================================================
默认路由: gpt20_a
总路由数: 1

检查路由: gpt20_a
  URL: http://127.0.0.1:8002/v1
  模型: gpt_oss_20b
  状态: ✅ 健康 (245.67ms)
  可用模型: gpt_oss_20b

==================================================
健康检查完成: 1/1 个路由健康
✅ 所有路由都健康
```

### 4. 冒烟测试

```python
# test_vllm_smoke.py
import sys
sys.path.append('.')

from llm.factory import LLMFactory

# 创建 provider
provider = LLMFactory.create_provider('vllm_openai')

# 测试同步调用
response = provider.chat([
    {"role": "user", "content": "你好，请简单介绍一下自己。"}
])
print(f"同步响应: {response}")

# 测试流式调用
print("\n流式响应: ", end="")
for chunk in provider.stream([
    {"role": "user", "content": "请用一句话总结人工智能的重要性。"}
]):
    print(chunk, end="", flush=True)
print("\n")

# 检查健康状态
print(f"Provider 可用性: {provider.is_available()}")
print(f"可用模型: {provider.list_models()}")
```

运行测试：
```bash
python test_vllm_smoke.py
```

### 5. 压力测试

使用内置的压测脚本：

```bash
# 默认配置：64 并发，256 请求
python scripts/bench_vllm.py

# 自定义配置
VLLM_BASE=http://127.0.0.1:8002/v1 \
VLLM_MODEL=gpt_oss_20b \
CONC=32 \
NREQ=128 \
python scripts/bench_vllm.py

# 高并发测试
CONC=128 NREQ=512 python scripts/bench_vllm.py
```

期望输出示例：
```json
{
  "benchmark_config": {
    "base_url": "http://127.0.0.1:8002/v1",
    "model": "gpt_oss_20b",
    "concurrency": 64,
    "num_requests": 256
  },
  "summary": {
    "total_requests": 256,
    "successful_requests": 256,
    "failed_requests": 0,
    "success_rate": 100.0,
    "total_time_seconds": 45.67,
    "requests_per_second": 5.61,
    "successful_rps": 5.61
  },
  "latency_ms": {
    "min": 2341.23,
    "max": 8765.43,
    "mean": 4532.11,
    "median": 4234.56,
    "p95": 6789.01,
    "p99": 7890.12
  }
}
```

## 高级功能

### 负载均衡策略

#### Round Robin（轮询）
```yaml
lb_policy: round_robin
```
按顺序轮流使用各个健康的实例。

#### Least Latency（最低延迟）
```yaml
lb_policy: least_latency
```
选择响应延迟最低的实例。

### 故障转移

当主路由失败时，系统会自动：
1. 标记失败的路由为不健康
2. 尝试使用其他健康的路由
3. 如果配置了 `fallback_route`，最终回退到该路由
4. 使用指数退避进行重试

### 环境变量覆盖

可以通过环境变量覆盖配置：

```bash
# 覆盖默认路由的 base_url
export VLLM_BASE_URL=http://192.168.1.100:8002/v1

# 覆盖模型名称
export VLLM_MODEL=gpt_oss_120b

# 运行应用
python main.py
```

## 故障排除

### 常见问题

#### 1. 连接被拒绝
```
Connection refused
```
**解决方案：**
- 检查 vLLM 服务是否启动
- 验证端口号是否正确
- 确认防火墙设置

#### 2. 未授权错误
```
HTTP 401: Unauthorized
```
**解决方案：**
- vLLM 需要非空的 API key，配置中已设置为 "EMPTY"
- 如果使用自定义 vLLM 部署，确认 API key 配置

#### 3. 模型不存在
```
Model 'gpt_oss_20b' not found
```
**解决方案：**
- 检查 `--served-model-name` 参数
- 使用健康检查脚本查看可用模型
- 确认模型加载成功

#### 4. 显存不足
```
CUDA out of memory
```
**解决方案：**
- 减少 `--max-model-len`
- 降低 `--gpu-memory-utilization`
- 使用 `--kv-cache-dtype fp8`
- 考虑使用更小的模型

#### 5. 超时错误
```
Timeout after 60s
```
**解决方案：**
- 增加 `timeout` 配置
- 检查服务器负载
- 优化模型推理参数

### 调试技巧

#### 启用详细日志
```python
from loguru import logger
logger.add("vllm_debug.log", level="DEBUG")
```

#### 监控实例状态
```bash
# 持续监控健康状态
watch -n 5 'python scripts/healthcheck_vllm.py'
```

#### 检查 GPU 使用情况
```bash
# 监控 GPU 状态
watch -n 1 nvidia-smi
```

## 性能优化建议

### 1. 硬件配置
- **GPU**: 推荐 A100/H100，至少 24GB 显存
- **内存**: 推荐 64GB+ 系统内存
- **存储**: 使用 NVMe SSD 存储模型文件

### 2. vLLM 参数调优
```bash
# 高性能配置示例
python -m vllm.entrypoints.openai.api_server \
  --model openai/gpt-oss-20b \
  --served-model-name gpt_oss_20b \
  --dtype bfloat16 \
  --tensor-parallel-size 2 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.95 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --disable-log-requests \
  --port 8002
```

### 3. 并发配置
- 根据 GPU 数量调整并发数
- 监控 GPU 利用率，避免过载
- 使用多实例负载均衡提高吞吐量

### 4. 网络优化
- 使用本地部署减少网络延迟
- 配置合适的超时时间
- 启用 HTTP/2 支持（如果可用）

## 与 Ollama 的对比

| 特性 | vLLM | Ollama |
|------|------|--------|
| 性能 | 极高 | 中等 |
| 显存效率 | 优秀 | 良好 |
| 并发支持 | 原生支持 | 需要多实例 |
| 部署复杂度 | 中等 | 简单 |
| 模型支持 | 广泛 | 有限 |
| API 兼容性 | OpenAI 标准 | 自定义 |

## 总结

vLLM OpenAI 兼容 Provider 为 anorag 项目提供了高性能、可靠的 LLM 推理能力。通过合理的配置和优化，可以实现：

- **高吞吐量**: 支持高并发请求处理
- **低延迟**: 优化的推理引擎和缓存机制
- **高可用性**: 多实例负载均衡和故障转移
- **易维护**: 完善的监控和调试工具

建议在生产环境中使用多实例部署，并配置适当的监控和告警机制。