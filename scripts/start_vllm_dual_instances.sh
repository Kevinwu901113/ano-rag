#!/bin/bash

# vLLM 双实例启动脚本
# 在两张 4090 上分别启动 vLLM OpenAI 兼容服务

set -e

# 配置参数（与 config.yaml 对齐）
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"  # 与 llm.note_generator.model 对齐
SERVED_MODEL_NAME="qwen2.5:7b"          # OpenAI 端展示的模型别名
DTYPE="float16"
GPU_MEMORY_UTIL="0.92"
MAX_MODEL_LEN="4096"
MAX_BATCHED_TOKENS="32768"

# 端口配置
GPU0_PORT="8000"
GPU1_PORT="8001"

# 日志目录
LOG_DIR="logs/vllm"
mkdir -p "$LOG_DIR"

echo "Starting vLLM dual instances..."
echo "Model: $MODEL_NAME (served as: $SERVED_MODEL_NAME)"
echo "GPU0 Port: $GPU0_PORT"
echo "GPU1 Port: $GPU1_PORT"

# 检查端口是否被占用
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "Warning: Port $port is already in use"
        return 1
    fi
    return 0
}

# 启动 GPU0 实例
start_gpu0() {
    echo "Starting vLLM instance on GPU0 (port $GPU0_PORT)..."
    CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_NAME" \
        --served-model-name "$SERVED_MODEL_NAME" \
        --dtype "$DTYPE" \
        --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
        --port "$GPU0_PORT" \
        --host 0.0.0.0 \
        > "$LOG_DIR/gpu0_port${GPU0_PORT}.log" 2>&1 &
    
    local pid=$!
    echo "GPU0 instance started with PID: $pid"
    echo "$pid" > "$LOG_DIR/gpu0.pid"
}

# 启动 GPU1 实例
start_gpu1() {
    echo "Starting vLLM instance on GPU1 (port $GPU1_PORT)..."
    CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_NAME" \
        --served-model-name "$SERVED_MODEL_NAME" \
        --dtype "$DTYPE" \
        --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
        --port "$GPU1_PORT" \
        --host 0.0.0.0 \
        > "$LOG_DIR/gpu1_port${GPU1_PORT}.log" 2>&1 &
    
    local pid=$!
    echo "GPU1 instance started with PID: $pid"
    echo "$pid" > "$LOG_DIR/gpu1.pid"
}

# 健康检查
health_check() {
    local port=$1
    local max_attempts=30
    local attempt=1
    
    echo "Checking health of service on port $port..."
    
    while [ $attempt -le $max_attempts ]; do
        # 检查 models 列表以及 served 模型别名是否存在
        if curl -s "http://127.0.0.1:$port/v1/models" | grep -q "$SERVED_MODEL_NAME"; then
            echo "✓ Service on port $port is healthy (served: $SERVED_MODEL_NAME)"
            return 0
        fi
        
        echo "Attempt $attempt/$max_attempts: Service on port $port not ready yet..."
        sleep 5
        ((attempt++))
    done
    
    echo "✗ Service on port $port failed to start within timeout"
    return 1
}

# 主执行流程
main() {
    # 检查端口
    if ! check_port "$GPU0_PORT"; then
        echo "Port $GPU0_PORT is in use. Please stop the service or use a different port."
        exit 1
    fi
    
    if ! check_port "$GPU1_PORT"; then
        echo "Port $GPU1_PORT is in use. Please stop the service or use a different port."
        exit 1
    fi
    
    # 启动实例
    start_gpu0
    sleep 2  # 避免同时启动造成资源竞争
    start_gpu1
    
    echo "Waiting for services to start..."
    sleep 10
    
    # 健康检查
    if health_check "$GPU0_PORT" && health_check "$GPU1_PORT"; then
        echo ""
        echo "✓ vLLM dual instances started successfully!"
        echo "GPU0 service: http://127.0.0.1:$GPU0_PORT/v1"
        echo "GPU1 service: http://127.0.0.1:$GPU1_PORT/v1"
        echo ""
        echo "Logs:"
        echo "  GPU0: $LOG_DIR/gpu0_port${GPU0_PORT}.log"
        echo "  GPU1: $LOG_DIR/gpu1_port${GPU1_PORT}.log"
        echo ""
        echo "To stop services, run: ./scripts/stop_vllm_dual_instances.sh"
    else
        echo "✗ Failed to start vLLM dual instances"
        exit 1
    fi
}

# 执行主函数
main "$@"