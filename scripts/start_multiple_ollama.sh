#!/bin/bash

# 启动多个Ollama实例的脚本
# 用法: ./start_multiple_ollama.sh [实例数量] [起始端口] [模型名称]

# 默认参数
NUM_INSTANCES=${1:-4}  # 默认启动4个实例
START_PORT=${2:-11434} # 默认起始端口
MODEL_NAME=${3:-"gpt-oss:latest"}  # 默认模型

echo "启动 $NUM_INSTANCES 个Ollama实例，起始端口: $START_PORT，模型: $MODEL_NAME"

# 检查ollama是否已安装
if ! command -v ollama &> /dev/null; then
    echo "错误: ollama未安装，请先安装ollama"
    echo "安装命令: curl -fsSL https://ollama.ai/install.sh | sh"
    exit 1
fi

# 检查模型是否存在
echo "检查模型 $MODEL_NAME 是否存在..."
if ! ollama list | grep -q "$MODEL_NAME"; then
    echo "模型 $MODEL_NAME 不存在，正在下载..."
    ollama pull "$MODEL_NAME"
    if [ $? -ne 0 ]; then
        echo "错误: 模型下载失败"
        exit 1
    fi
fi

# 创建日志目录
LOG_DIR="./logs/ollama"
mkdir -p "$LOG_DIR"

# 停止现有的ollama进程
echo "停止现有的ollama进程..."
pkill -f "ollama serve" || true
sleep 2

# 启动多个ollama实例
for ((i=0; i<$NUM_INSTANCES; i++)); do
    PORT=$((START_PORT + i))
    LOG_FILE="$LOG_DIR/ollama_$PORT.log"
    
    echo "启动Ollama实例 $((i+1))/$NUM_INSTANCES 在端口 $PORT..."
    
    # 设置环境变量并启动ollama服务
    OLLAMA_HOST="0.0.0.0:$PORT" nohup ollama serve > "$LOG_FILE" 2>&1 &
    
    # 记录进程ID
    echo $! > "$LOG_DIR/ollama_$PORT.pid"
    
    # 等待服务启动
    sleep 3
    
    # 检查服务是否启动成功
    if curl -s "http://localhost:$PORT/api/version" > /dev/null; then
        echo "✓ Ollama实例在端口 $PORT 启动成功"
    else
        echo "✗ Ollama实例在端口 $PORT 启动失败"
    fi
done

echo ""
echo "所有Ollama实例启动完成！"
echo "端口范围: $START_PORT - $((START_PORT + NUM_INSTANCES - 1))"
echo "日志目录: $LOG_DIR"
echo ""
echo "配置config.yaml以启用多实例:"
echo "llm:"
echo "  ollama:"
echo "    multiple_instances:"
echo "      enabled: true"
echo "      instances:"
for ((i=0; i<$NUM_INSTANCES; i++)); do
    PORT=$((START_PORT + i))
    echo "        - base_url: \"http://localhost:$PORT\""
    echo "          model: \"$MODEL_NAME\""
done
echo ""
echo "停止所有实例: ./stop_multiple_ollama.sh"
echo "查看状态: ./check_ollama_status.sh"