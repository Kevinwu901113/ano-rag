#!/bin/bash

# 修复GPU加速问题的Ollama多实例启动脚本
# 用法: ./start_ollama_fixed.sh [实例数量] [起始端口] [模型名称]

# 默认参数
NUM_INSTANCES=${1:-2}  # 默认启动2个实例
START_PORT=${2:-11440} # 使用11440开始，避免与系统ollama冲突
MODEL_NAME=${3:-"gpt-oss:latest"}  # 默认模型

echo "=== Ollama GPU加速多实例启动 ==="
echo "实例数量: $NUM_INSTANCES"
echo "起始端口: $START_PORT"
echo "模型: $MODEL_NAME"
echo "时间: $(date)"
echo ""

# 检查ollama是否已安装
if ! command -v ollama &> /dev/null; then
    echo "错误: ollama未安装，请先安装ollama"
    exit 1
fi

# 检查GPU状态
echo "GPU状态检查:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
echo ""

# 创建日志目录
LOG_DIR="./logs/ollama_fixed"
mkdir -p "$LOG_DIR"

# 停止可能存在的用户级ollama进程（避免端口冲突）
echo "清理可能的端口冲突..."
for ((i=0; i<$NUM_INSTANCES; i++)); do
    PORT=$((START_PORT + i))
    PID=$(lsof -ti:$PORT 2>/dev/null)
    if [ ! -z "$PID" ]; then
        echo "终止占用端口 $PORT 的进程 $PID"
        kill $PID 2>/dev/null || true
        sleep 1
    fi
done

echo "启动多个GPU加速的Ollama实例..."
echo ""

# 启动多个ollama实例
for ((i=0; i<$NUM_INSTANCES; i++)); do
    PORT=$((START_PORT + i))
    LOG_FILE="$LOG_DIR/ollama_$PORT.log"
    PID_FILE="$LOG_DIR/ollama_$PORT.pid"
    
    # 根据实例编号分配GPU（在两张GPU之间轮换）
    GPU_ID=$((i % 2))
    
    echo "启动实例 $((i+1))/$NUM_INSTANCES:"
    echo "  端口: $PORT"
    echo "  GPU: $GPU_ID"
    echo "  日志: $LOG_FILE"
    
    # 设置GPU环境变量
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    export OLLAMA_GPU_LAYERS=35  # 限制GPU层数，避免内存不足
    export OLLAMA_NUM_PARALLEL=1
    export OLLAMA_MAX_LOADED_MODELS=1
    
    # 启动ollama服务
    OLLAMA_HOST="0.0.0.0:$PORT" nohup ollama serve > "$LOG_FILE" 2>&1 &
    INSTANCE_PID=$!
    
    # 记录进程ID
    echo $INSTANCE_PID > "$PID_FILE"
    echo "  PID: $INSTANCE_PID"
    
    # 等待服务启动
    echo "  等待启动..."
    sleep 5
    
    # 检查服务是否启动成功
    if curl -s "http://localhost:$PORT/api/version" > /dev/null; then
        echo "  ✓ 启动成功"
    else
        echo "  ✗ 启动失败，请检查日志: $LOG_FILE"
        continue
    fi
    
    echo ""
done

echo "所有实例启动完成！"
echo ""
echo "现在预加载模型..."

# 预加载模型到各个实例（串行加载避免GPU内存竞争）
for ((i=0; i<$NUM_INSTANCES; i++)); do
    PORT=$((START_PORT + i))
    GPU_ID=$((i % 2))
    
    echo "预加载模型到端口 $PORT (GPU $GPU_ID)..."
    
    # 检查服务是否可用
    if ! curl -s "http://localhost:$PORT/api/version" > /dev/null; then
        echo "  ⚠ 端口 $PORT 服务不可用，跳过"
        continue
    fi
    
    # 设置环境变量并加载模型
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    
    # 使用超时防止卡住
    timeout 60s bash -c "
        OLLAMA_HOST='localhost:$PORT' ollama run '$MODEL_NAME' '测试GPU加速' > /dev/null 2>&1
    " &
    LOAD_PID=$!
    
    # 等待加载完成
    wait $LOAD_PID
    LOAD_STATUS=$?
    
    if [ $LOAD_STATUS -eq 0 ]; then
        echo "  ✓ 模型加载成功"
        
        # 检查是否使用GPU
        PROCESSOR_INFO=$(OLLAMA_HOST="localhost:$PORT" ollama ps | grep "$MODEL_NAME" | awk '{print $4}')
        echo "  处理器: $PROCESSOR_INFO"
    else
        echo "  ⚠ 模型加载超时或失败"
    fi
    
    sleep 2
done

echo ""
echo "=== 启动完成 ==="
echo "端口范围: $START_PORT - $((START_PORT + NUM_INSTANCES - 1))"
echo "日志目录: $LOG_DIR"
echo ""
echo "实例状态检查:"
for ((i=0; i<$NUM_INSTANCES; i++)); do
    PORT=$((START_PORT + i))
    echo "端口 $PORT:"
    OLLAMA_HOST="localhost:$PORT" ollama ps 2>/dev/null || echo "  连接失败"
done

echo ""
echo "当前GPU使用情况:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | awk '{print "GPU利用率: " $1 "%, 显存: " $2 "/" $3 "MB"}'

echo ""
echo "管理命令:"
echo "  查看状态: ./scripts/ollama_manager.sh status"
echo "  停止实例: for port in {$START_PORT..$((START_PORT + NUM_INSTANCES - 1))}; do kill \$(cat $LOG_DIR/ollama_\$port.pid 2>/dev/null) 2>/dev/null; done"
echo "  查看日志: tail -f $LOG_DIR/ollama_*.log"