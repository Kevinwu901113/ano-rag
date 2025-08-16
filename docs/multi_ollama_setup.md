# 多Ollama实例配置指南

本指南介绍如何配置和使用多个Ollama实例来提升RAG系统的并行处理效率。

## 概述

当处理大量数据（如2000+行的输入文件）时，单个Ollama实例可能成为性能瓶颈。通过配置多个Ollama实例，系统可以并行处理多个请求，显著提升处理效率。

## 快速开始

### 1. 启动多个Ollama实例

使用提供的脚本快速启动多个Ollama实例：

```bash
# 启动4个实例（默认）
./scripts/start_multiple_ollama.sh

# 启动6个实例，从端口11434开始
./scripts/start_multiple_ollama.sh 6 11434 "gpt-oss:latest"

# 参数说明：
# 参数1: 实例数量（默认4）
# 参数2: 起始端口（默认11434）
# 参数3: 模型名称（默认"gpt-oss:latest"）
```

### 2. 配置config.yaml

在`config.yaml`中启用多实例配置：

```yaml
llm:
  ollama:
    # 单实例配置（向后兼容）
    base_url: "http://localhost:11434"
    model: "gpt-oss:latest"
    temperature: 0.1
    max_tokens: 4096
    num_ctx: 8192
    max_async: 8
    timeout: 300
    
    # 多实例配置
    multiple_instances:
      enabled: true  # 启用多实例模式
      load_balancing: "round_robin"  # 负载均衡策略
      health_check_interval: 30  # 健康检查间隔（秒）
      max_retries: 3  # 每个实例最大重试次数
      instances:
        - base_url: "http://localhost:11434"
          model: "gpt-oss:latest"
        - base_url: "http://localhost:11435"
          model: "gpt-oss:latest"
        - base_url: "http://localhost:11436"
          model: "gpt-oss:latest"
        - base_url: "http://localhost:11437"
          model: "gpt-oss:latest"

# 并行处理配置
performance:
  num_workers: 8  # 建议设置为实例数的1.5-2倍
```

### 3. 运行批量处理

```bash
# 使用多实例处理大量数据
python main_musique.py data/input.jsonl results/output.jsonl --workers 8

# workers数量建议设置为Ollama实例数的1.5-2倍
```

## 负载均衡策略

系统支持多种负载均衡策略：

### 1. 轮询（round_robin）
- **默认策略**
- 按顺序轮流分配请求到各个实例
- 适合实例性能相近的场景

### 2. 随机（random）
- 随机选择可用实例
- 简单高效，适合大多数场景

### 3. 最少活跃请求（least_active）
- 选择当前活跃请求数最少的实例
- 适合请求处理时间差异较大的场景

配置示例：
```yaml
llm:
  ollama:
    multiple_instances:
      load_balancing: "least_active"  # 或 "random"
```

## 健康检查和故障恢复

系统会定期检查各个实例的健康状态：

- **健康检查间隔**：默认30秒
- **自动故障转移**：不健康的实例会被暂时移除
- **自动恢复**：故障实例恢复后会重新加入负载均衡
- **重试机制**：请求失败时会自动重试其他实例

## 脚本使用

### 统一管理工具（推荐）

使用 `ollama_manager.sh` 统一管理所有Ollama实例：

```bash
# 查看帮助
./scripts/ollama_manager.sh help

# 启动4个实例
./scripts/ollama_manager.sh start 4

# 快速并行下载模型（推荐）
./scripts/ollama_manager.sh quick-download 4 11434 gpt-oss:latest

# 重启并完整配置实例
./scripts/ollama_manager.sh restart 4

# 检查状态
./scripts/ollama_manager.sh status

# 停止所有实例
./scripts/ollama_manager.sh stop

# 测试模型
./scripts/ollama_manager.sh test 11434 gpt-oss:latest

# 查看日志
./scripts/ollama_manager.sh logs 11434

# 清理临时文件
./scripts/ollama_manager.sh cleanup
```

### 单独脚本使用

#### 启动实例
```bash
./scripts/start_multiple_ollama.sh [实例数] [起始端口] [模型名]

# 启动4个实例，从端口11434开始，使用gpt-oss:latest模型
./scripts/start_multiple_ollama.sh 4 11434 gpt-oss:latest

# 使用默认参数（4个实例，端口11434-11437，gpt-oss:latest模型）
./scripts/start_multiple_ollama.sh
```

#### 停止所有实例
```bash
./scripts/stop_multiple_ollama.sh
```

##### 检查状态
```bash
./scripts/check_ollama_status.sh
```

#### 下载模型

**串行下载（稳定但较慢）：**
```bash
# 为4个实例下载gpt-oss:latest模型
./scripts/download_models_for_instances.sh 4 11434 gpt-oss:latest
```

**并行下载（快速但占用资源多）：**
```bash
# 快速并行下载，适合多核服务器
./scripts/quick_download_models.sh 4 11434 gpt-oss:latest
```

## 脚本功能对比

| 脚本名称 | 功能 | 特点 | 适用场景 |
|---------|------|------|----------|
| `ollama_manager.sh` | 统一管理工具 | 集成所有功能，易用 | 日常管理（推荐） |
| `start_multiple_ollama.sh` | 启动实例 | 自动配置，日志记录 | 初始化部署 |
| `stop_multiple_ollama.sh` | 停止实例 | 强制清理，端口检查 | 维护重启 |
| `check_ollama_status.sh` | 状态检查 | 详细信息，资源监控 | 故障诊断 |
| `download_models_for_instances.sh` | 串行下载 | 稳定可靠，进度显示 | 网络较慢时 |
| `quick_download_models.sh` | 并行下载 | 速度快，资源占用高 | 高性能服务器 |

## 性能优化建议

### 1. 实例数量配置
- **CPU密集型**：实例数 = CPU核心数
- **内存充足**：实例数 = CPU核心数 × 1.5
- **大内存模型**：根据可用内存调整

### 2. 并行工作线程
- 设置为实例数的1.5-2倍
- 避免设置过高导致资源竞争

### 3. 系统资源监控
```bash
# 监控资源使用
./scripts/check_ollama_status.sh

# 查看实时日志
tail -f logs/ollama/ollama_11434.log
```

## 故障排除

### 常见问题

1. **端口被占用**
   ```bash
   # 检查端口占用
   lsof -i :11434
   
   # 停止占用进程
   ./scripts/stop_multiple_ollama.sh
   ```

2. **模型未下载**
   ```bash
   # 手动下载模型
   ollama pull gpt-oss:latest
   ```

3. **内存不足**
   - 减少实例数量
   - 使用更小的模型
   - 增加系统内存

4. **连接超时**
   - 增加timeout配置
   - 检查网络连接
   - 查看实例日志

### 日志分析
```bash
# 查看所有实例状态
./scripts/check_ollama_status.sh

# 查看特定实例日志
tail -f logs/ollama/ollama_11434.log

# 搜索错误信息
grep -i error logs/ollama/*.log
```

## 性能对比

| 配置 | 单实例 | 4实例 | 8实例 |
|------|--------|-------|-------|
| 2000行数据处理时间 | ~4小时 | ~1小时 | ~30分钟 |
| CPU利用率 | 25% | 80% | 95% |
| 内存使用 | 4GB | 16GB | 32GB |
| 推荐场景 | 小数据集 | 中等数据集 | 大数据集 |

## 注意事项

1. **内存需求**：每个实例需要2-4GB内存（取决于模型大小）
2. **CPU使用**：确保有足够的CPU核心支持多实例
3. **磁盘空间**：模型文件会被多个实例共享
4. **网络带宽**：本地实例通常不受网络限制
5. **模型一致性**：确保所有实例使用相同版本的模型

## 高级配置

### 混合模型配置
```yaml
llm:
  ollama:
    multiple_instances:
      instances:
        - base_url: "http://localhost:11434"
          model: "gpt-oss:latest"  # 大模型，用于复杂任务
        - base_url: "http://localhost:11435"
          model: "gpt-oss:7b"      # 小模型，用于简单任务
```

### 自定义健康检查
```yaml
llm:
  ollama:
    multiple_instances:
      health_check_interval: 15  # 更频繁的检查
      max_retries: 5            # 更多重试次数
```

通过合理配置多Ollama实例，您可以显著提升大规模数据处理的效率，将原本需要数小时的任务缩短到几十分钟完成。