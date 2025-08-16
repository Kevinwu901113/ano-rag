#!/bin/bash

# 快速并行下载模型脚本
# 用法: ./quick_download_models.sh [实例数量] [起始端口] [模型名称]

# 默认参数
NUM_INSTANCES=${1:-2}
START_PORT=${2:-11434}
MODEL_NAME=${3:-"gpt-oss:latest"}

echo "快速并行下载模式"
echo "实例数量: $NUM_INSTANCES"
echo "端口范围: $START_PORT - $((START_PORT + NUM_INSTANCES - 1))"
echo "模型: $MODEL_NAME"
echo "=============================="

# 创建日志目录
LOG_DIR="./logs/parallel_download"
mkdir -p "$LOG_DIR"

# 并行下载函数
download_for_port() {
    local PORT=$1
    local MODEL=$2
    local LOG_FILE="$LOG_DIR/download_$PORT.log"
    
    echo "[端口 $PORT] 开始下载..." | tee -a "$LOG_FILE"
    
    # 检查实例是否运行
    if ! curl -s "http://localhost:$PORT/api/version" > /dev/null 2>&1; then
        echo "[端口 $PORT] ✗ 实例未运行" | tee -a "$LOG_FILE"
        return 1
    fi
    
    # 检查模型是否已存在
    if OLLAMA_HOST="localhost:$PORT" ollama list 2>/dev/null | grep -q "$MODEL"; then
        echo "[端口 $PORT] ✓ 模型已存在" | tee -a "$LOG_FILE"
        return 0
    fi
    
    # 下载模型
    echo "[端口 $PORT] 正在下载模型 $MODEL..." | tee -a "$LOG_FILE"
    if timeout 1800 bash -c "OLLAMA_HOST=localhost:$PORT ollama pull '$MODEL'" >> "$LOG_FILE" 2>&1; then
        echo "[端口 $PORT] ✓ 下载成功" | tee -a "$LOG_FILE"
        return 0
    else
        local EXIT_CODE=$?
        if [ $EXIT_CODE -eq 124 ]; then
            echo "[端口 $PORT] ✗ 下载超时" | tee -a "$LOG_FILE"
        else
            echo "[端口 $PORT] ✗ 下载失败 (退出码: $EXIT_CODE)" | tee -a "$LOG_FILE"
        fi
        return $EXIT_CODE
    fi
}

# 导出函数以供并行使用
export -f download_for_port
export LOG_DIR

# 记录开始时间
START_TIME=$(date +%s)

echo "开始并行下载..."
echo "提示: 可以使用 'tail -f $LOG_DIR/download_*.log' 查看实时进度"
echo ""

# 生成端口列表并并行执行
PORT_LIST=""
for ((i=0; i<$NUM_INSTANCES; i++)); do
    PORT=$((START_PORT + i))
    PORT_LIST="$PORT_LIST $PORT"
done

# 使用xargs并行下载（最多4个并行任务）
echo $PORT_LIST | tr ' ' '\n' | xargs -n 1 -P 4 -I {} bash -c "download_for_port {} '$MODEL_NAME'"

# 计算总耗时
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
MINUTES=$((TOTAL_TIME / 60))
SECONDS=$((TOTAL_TIME % 60))

echo ""
echo "=============================="
echo "并行下载完成！总耗时: ${MINUTES}分${SECONDS}秒"
echo ""

# 最终验证
echo "验证下载结果:"
echo "------------------------------"
SUCCESS_COUNT=0
for ((i=0; i<$NUM_INSTANCES; i++)); do
    PORT=$((START_PORT + i))
    if curl -s "http://localhost:$PORT/api/version" > /dev/null 2>&1; then
        if OLLAMA_HOST="localhost:$PORT" ollama list 2>/dev/null | grep -q "$MODEL_NAME"; then
            echo "端口 $PORT: ✓ 模型可用"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            echo "端口 $PORT: ✗ 模型不可用"
        fi
    else
        echo "端口 $PORT: ✗ 实例未运行"
    fi
done

echo ""
echo "成功: $SUCCESS_COUNT/$NUM_INSTANCES 个实例"
if [ $SUCCESS_COUNT -eq $NUM_INSTANCES ]; then
    echo "🎉 所有实例模型下载成功！"
else
    echo "⚠️  部分实例下载失败，请检查日志: $LOG_DIR/"
fi

echo ""
echo "查看详细日志: ls -la $LOG_DIR/"
echo "测试模型: OLLAMA_HOST=\"localhost:[端口]\" ollama run $MODEL_NAME"