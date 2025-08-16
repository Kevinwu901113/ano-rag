#!/bin/bash

# 为多个Ollama实例下载模型的脚本
# 用法: ./download_models_for_instances.sh [实例数量] [起始端口] [模型名称]

# 默认参数
NUM_INSTANCES=${1:-2}  # 默认2个实例
START_PORT=${2:-11434} # 默认起始端口
MODEL_NAME=${3:-"gpt-oss:latest"}  # 默认模型

echo "为 $NUM_INSTANCES 个Ollama实例下载模型: $MODEL_NAME"
echo "端口范围: $START_PORT - $((START_PORT + NUM_INSTANCES - 1))"
echo "=============================="

# 检查ollama是否已安装
if ! command -v ollama &> /dev/null; then
    echo "错误: ollama未安装，请先安装ollama"
    echo "安装命令: curl -fsSL https://ollama.ai/install.sh | sh"
    exit 1
fi

# 创建日志目录
LOG_DIR="./logs/model_download"
mkdir -p "$LOG_DIR"

# 记录开始时间
START_TIME=$(date +%s)

# 为每个实例下载模型
for ((i=0; i<$NUM_INSTANCES; i++)); do
    PORT=$((START_PORT + i))
    LOG_FILE="$LOG_DIR/download_$PORT.log"
    
    echo "[$((i+1))/$NUM_INSTANCES] 为端口 $PORT 下载模型 $MODEL_NAME..."
    
    # 检查实例是否在运行
    if ! curl -s "http://localhost:$PORT/api/version" > /dev/null 2>&1; then
        echo "  ✗ 警告: 端口 $PORT 上的Ollama实例未运行"
        echo "  请先启动实例: OLLAMA_HOST=\"0.0.0.0:$PORT\" ollama serve"
        continue
    fi
    
    # 检查模型是否已存在
    echo "  检查模型是否已存在..."
    if OLLAMA_HOST="localhost:$PORT" ollama list 2>/dev/null | grep -q "$MODEL_NAME"; then
        echo "  ✓ 模型 $MODEL_NAME 已存在于端口 $PORT"
        continue
    fi
    
    # 下载模型（显示实时进度）
    echo "  开始下载模型..."
    echo "  提示: 大模型下载可能需要较长时间，请耐心等待"
    
    # 使用timeout防止卡死，同时显示进度
    (
        # 启动下载进程
        OLLAMA_HOST="localhost:$PORT" timeout 1800 ollama pull "$MODEL_NAME" 2>&1 | tee "$LOG_FILE" &
        DOWNLOAD_PID=$!
        
        # 显示进度监控
        while kill -0 $DOWNLOAD_PID 2>/dev/null; do
            if [ -f "$LOG_FILE" ]; then
                # 检查是否有进度信息
                PROGRESS=$(tail -1 "$LOG_FILE" | grep -o '[0-9]\+%' | tail -1)
                if [ -n "$PROGRESS" ]; then
                    echo -ne "\r  下载进度: $PROGRESS"
                else
                    echo -ne "\r  下载中..."
                fi
            fi
            sleep 2
        done
        echo ""  # 换行
        
        # 等待下载完成
        wait $DOWNLOAD_PID
        echo $? > "/tmp/download_exit_$PORT"
    )
    
    # 检查下载结果
    EXIT_CODE=$(cat "/tmp/download_exit_$PORT" 2>/dev/null || echo "1")
    rm -f "/tmp/download_exit_$PORT"
    
    if [ "$EXIT_CODE" -eq 0 ]; then
        echo "  ✓ 模型下载成功"
    elif [ "$EXIT_CODE" -eq 124 ]; then
        echo "  ✗ 模型下载超时（30分钟），可能网络较慢或模型过大"
        echo "  建议: 手动执行 OLLAMA_HOST=\"localhost:$PORT\" ollama pull \"$MODEL_NAME\""
    else
        echo "  ✗ 模型下载失败，查看日志: $LOG_FILE"
        echo "  最近错误信息:"
        tail -3 "$LOG_FILE" | sed 's/^/    /'
    fi
    
    echo ""
done

# 计算总耗时
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
MINUTES=$((TOTAL_TIME / 60))
SECONDS=$((TOTAL_TIME % 60))

echo "=============================="
echo "模型下载完成！总耗时: ${MINUTES}分${SECONDS}秒"
echo ""
echo "验证模型下载状态:"
echo "------------------------------"

# 验证每个实例的模型状态
for ((i=0; i<$NUM_INSTANCES; i++)); do
    PORT=$((START_PORT + i))
    echo "端口 $PORT:"
    
    if curl -s "http://localhost:$PORT/api/version" > /dev/null 2>&1; then
        if OLLAMA_HOST="localhost:$PORT" ollama list 2>/dev/null | grep -q "$MODEL_NAME"; then
            echo "  ✓ 实例运行正常，模型 $MODEL_NAME 可用"
        else
            echo "  ✗ 实例运行正常，但模型 $MODEL_NAME 不可用"
        fi
    else
        echo "  ✗ 实例未运行"
    fi
done

echo ""
echo "使用说明:"
echo "- 查看下载日志: ls -la $LOG_DIR/"
echo "- 测试模型: OLLAMA_HOST=\"localhost:[端口]\" ollama run $MODEL_NAME"
echo "- 列出模型: OLLAMA_HOST=\"localhost:[端口]\" ollama list"