#!/bin/bash

# GPU内存优化的Ollama多实例启动脚本
# 用法: ./start_ollama_gpu_optimized.sh [实例数量] [起始端口] [模型名称]

# 默认参数
NUM_INSTANCES=${1:-2}  # 默认启动2个实例（避免GPU内存不足）
START_PORT=${2:-11434} # 默认起始端口
MODEL_NAME=${3:-"gpt-oss:latest"}  # 默认模型

echo "启动 $NUM_INSTANCES 个GPU优化的Ollama实例，起始端口: $START_PORT，模型: $MODEL_NAME"

# 检查ollama是否已安装
if ! command -v ollama &> /dev/null; then
    echo "错误: ollama未安装，请先安装ollama"
    echo "安装命令: curl -fsSL https://ollama.ai/install.sh | sh"
    exit 1
fi

# 创建日志目录
LOG_DIR="./logs/ollama"
mkdir -p "$LOG_DIR"

# 停止现有的ollama进程
echo "停止现有的ollama进程..."
pkill -f "ollama serve" || true
sleep 3

# GPU内存分配策略
# 每张RTX 4090有24GB显存，减去系统占用约2GB，可用22GB
# 13GB模型需要约14GB显存（包括推理缓存），每张卡最多1个完整实例
# 使用交替GPU分配策略

echo "检测到GPU配置:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits

# 启动多个ollama实例
for ((i=0; i<$NUM_INSTANCES; i++)); do
    PORT=$((START_PORT + i))
    LOG_FILE="$LOG_DIR/ollama_$PORT.log"
    
    # 根据实例编号分配GPU
    GPU_ID=$((i % 2))  # 在两张GPU之间轮换
    
    echo "启动Ollama实例 $((i+1))/$NUM_INSTANCES 在端口 $PORT (GPU $GPU_ID)..."
    
    # 设置GPU特定的环境变量
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    export OLLAMA_GPU_LAYERS=35  # 限制GPU层数，避免内存不足
    export OLLAMA_NUM_PARALLEL=1
    export OLLAMA_MAX_LOADED_MODELS=1
    
    # 启动ollama服务
    OLLAMA_HOST="0.0.0.0:$PORT" nohup ollama serve > "$LOG_FILE" 2>&1 &
    
    # 记录进程ID
    echo $! > "$LOG_DIR/ollama_$PORT.pid"
    
    # 等待服务启动
    sleep 5
    
    # 检查服务是否启动成功
    if curl -s "http://localhost:$PORT/api/version" > /dev/null; then
        echo "✓ Ollama实例在端口 $PORT 启动成功 (GPU $GPU_ID)"
    else
        echo "✗ Ollama实例在端口 $PORT 启动失败"
    fi
done

echo ""
echo "所有Ollama实例启动完成！"
echo "端口范围: $START_PORT - $((START_PORT + NUM_INSTANCES - 1))"
echo "日志目录: $LOG_DIR"
echo "GPU分配策略: 轮换使用GPU 0和GPU 1"
echo ""
echo "现在预加载模型到各个实例..."

# 预加载模型（串行加载避免GPU内存竞争）
for ((i=0; i<$NUM_INSTANCES; i++)); do
    PORT=$((START_PORT + i))
    GPU_ID=$((i % 2))
    
    echo "预加载模型到端口 $PORT (GPU $GPU_ID)..."
    
    # 设置超时并在后台加载模型
    timeout 60s bash -c "
        export CUDA_VISIBLE_DEVICES='$GPU_ID'
        OLLAMA_HOST='localhost:$PORT' ollama run '$MODEL_NAME' '测试' > /dev/null 2>&1
    " &
    
    # 等待当前实例加载完成再启动下一个
    wait
    
    # 检查模型是否成功加载
    if OLLAMA_HOST="localhost:$PORT" ollama ps | grep -q "$MODEL_NAME"; then
        echo "✓ 模型已成功加载到端口 $PORT"
    else
        echo "⚠ 端口 $PORT 模型加载可能失败，请检查日志"
    fi
    
    sleep 2
done

echo ""
echo "GPU优化启动完成！"
echo "查看GPU使用情况: nvidia-smi"
echo "查看实例状态: ./scripts/ollama_manager.sh status"
echo "停止所有实例: ./scripts/ollama_manager.sh stop"