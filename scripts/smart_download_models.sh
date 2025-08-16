#!/bin/bash

# 智能模型下载脚本
# 解决并行下载超时问题，采用分阶段下载策略

# 默认参数
NUM_INSTANCES=${1:-4}
START_PORT=${2:-11434}
MODEL_NAME=${3:-"gpt-oss:latest"}

echo "智能模型下载脚本"
echo "实例数量: $NUM_INSTANCES"
echo "端口范围: $START_PORT - $((START_PORT + NUM_INSTANCES - 1))"
echo "模型: $MODEL_NAME"
echo "=============================="

# 创建日志目录
LOG_DIR="./logs/smart_download"
mkdir -p "$LOG_DIR"

# 检查网络带宽和系统资源
check_system_resources() {
    echo "检查系统资源..."
    
    # 检查可用内存
    local available_mem=$(free -m | awk 'NR==2{printf "%.0f", $7}')
    echo "可用内存: ${available_mem}MB"
    
    # 检查磁盘空间
    local available_disk=$(df -BG . | awk 'NR==2{print $4}' | sed 's/G//')
    echo "可用磁盘: ${available_disk}GB"
    
    # 根据资源情况调整并发数
    if [ "$available_mem" -lt 8000 ]; then
        echo "⚠️  内存不足8GB，建议串行下载"
        return 1
    elif [ "$available_disk" -lt 50 ]; then
        echo "⚠️  磁盘空间不足50GB，可能影响下载"
        return 1
    fi
    
    return 0
}

# 检查模型是否已存在
check_model_exists() {
    local PORT=$1
    local MODEL=$2
    
    if ! curl -s "http://localhost:$PORT/api/version" > /dev/null 2>&1; then
        return 1
    fi
    
    if OLLAMA_HOST="localhost:$PORT" ollama list 2>/dev/null | grep -q "$MODEL"; then
        return 0
    fi
    
    return 1
}

# 单个实例下载函数
download_single_instance() {
    local PORT=$1
    local MODEL=$2
    local LOG_FILE="$LOG_DIR/download_$PORT.log"
    local MAX_RETRIES=3
    local RETRY_COUNT=0
    
    echo "[端口 $PORT] 开始下载..." | tee -a "$LOG_FILE"
    
    # 检查实例是否运行
    if ! curl -s "http://localhost:$PORT/api/version" > /dev/null 2>&1; then
        echo "[端口 $PORT] ✗ 实例未运行" | tee -a "$LOG_FILE"
        return 1
    fi
    
    # 检查模型是否已存在
    if check_model_exists "$PORT" "$MODEL"; then
        echo "[端口 $PORT] ✓ 模型已存在" | tee -a "$LOG_FILE"
        return 0
    fi
    
    # 重试下载
    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        echo "[端口 $PORT] 尝试下载 (第$((RETRY_COUNT + 1))次)..." | tee -a "$LOG_FILE"
        
        # 使用更长的超时时间，并添加进度监控
        if timeout 3600 bash -c "
            OLLAMA_HOST=localhost:$PORT ollama pull '$MODEL' 2>&1 | 
            while IFS= read -r line; do
                echo \"[$(date '+%H:%M:%S')] \$line\" | tee -a '$LOG_FILE'
                # 检查是否有进度更新
                if echo \"\$line\" | grep -q '%'; then
                    # 重置超时计数器
                    touch /tmp/progress_$PORT
                fi
            done
        "; then
            if check_model_exists "$PORT" "$MODEL"; then
                echo "[端口 $PORT] ✓ 下载成功" | tee -a "$LOG_FILE"
                rm -f "/tmp/progress_$PORT"
                return 0
            else
                echo "[端口 $PORT] ⚠️  下载完成但模型验证失败" | tee -a "$LOG_FILE"
            fi
        else
            local EXIT_CODE=$?
            if [ $EXIT_CODE -eq 124 ]; then
                echo "[端口 $PORT] ⚠️  下载超时，准备重试" | tee -a "$LOG_FILE"
            else
                echo "[端口 $PORT] ⚠️  下载失败 (退出码: $EXIT_CODE)，准备重试" | tee -a "$LOG_FILE"
            fi
        fi
        
        RETRY_COUNT=$((RETRY_COUNT + 1))
        if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
            local WAIT_TIME=$((RETRY_COUNT * 30))
            echo "[端口 $PORT] 等待 ${WAIT_TIME}秒 后重试..." | tee -a "$LOG_FILE"
            sleep $WAIT_TIME
        fi
    done
    
    echo "[端口 $PORT] ✗ 下载失败，已达到最大重试次数" | tee -a "$LOG_FILE"
    rm -f "/tmp/progress_$PORT"
    return 1
}

# 智能下载策略
smart_download_strategy() {
    local TOTAL_INSTANCES=$1
    local START_PORT=$2
    local MODEL=$3
    
    # 统计需要下载的实例
    local NEED_DOWNLOAD=()
    local ALREADY_HAVE=()
    
    for ((i=0; i<$TOTAL_INSTANCES; i++)); do
        local PORT=$((START_PORT + i))
        if check_model_exists "$PORT" "$MODEL"; then
            ALREADY_HAVE+=("$PORT")
        else
            NEED_DOWNLOAD+=("$PORT")
        fi
    done
    
    echo "模型状态统计:"
    echo "  已有模型: ${#ALREADY_HAVE[@]} 个实例 (${ALREADY_HAVE[*]})"
    echo "  需要下载: ${#NEED_DOWNLOAD[@]} 个实例 (${NEED_DOWNLOAD[*]})"
    echo ""
    
    if [ ${#NEED_DOWNLOAD[@]} -eq 0 ]; then
        echo "🎉 所有实例都已有模型，无需下载！"
        return 0
    fi
    
    # 检查系统资源
    if ! check_system_resources; then
        echo "采用串行下载策略（资源受限）"
        download_serial "${NEED_DOWNLOAD[@]}"
    else
        echo "采用分批并行下载策略"
        download_batch_parallel "${NEED_DOWNLOAD[@]}"
    fi
}

# 串行下载
download_serial() {
    local PORTS=("$@")
    echo "开始串行下载 ${#PORTS[@]} 个实例..."
    
    local SUCCESS_COUNT=0
    for PORT in "${PORTS[@]}"; do
        if download_single_instance "$PORT" "$MODEL_NAME"; then
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        fi
        echo "进度: $SUCCESS_COUNT/${#PORTS[@]}"
        echo ""
    done
    
    return $SUCCESS_COUNT
}

# 分批并行下载
download_batch_parallel() {
    local PORTS=("$@")
    local BATCH_SIZE=2  # 每批最多2个并行
    local TOTAL_PORTS=${#PORTS[@]}
    local SUCCESS_COUNT=0
    
    echo "开始分批并行下载，批大小: $BATCH_SIZE"
    
    for ((i=0; i<$TOTAL_PORTS; i+=BATCH_SIZE)); do
        local BATCH_PORTS=()
        local BATCH_END=$((i + BATCH_SIZE))
        if [ $BATCH_END -gt $TOTAL_PORTS ]; then
            BATCH_END=$TOTAL_PORTS
        fi
        
        # 构建当前批次
        for ((j=i; j<BATCH_END; j++)); do
            BATCH_PORTS+=("${PORTS[j]}")
        done
        
        echo "批次 $((i/BATCH_SIZE + 1)): 下载端口 ${BATCH_PORTS[*]}"
        
        # 并行下载当前批次
        local PIDS=()
        for PORT in "${BATCH_PORTS[@]}"; do
            download_single_instance "$PORT" "$MODEL_NAME" &
            PIDS+=("$!")
        done
        
        # 等待当前批次完成
        for PID in "${PIDS[@]}"; do
            if wait "$PID"; then
                SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
            fi
        done
        
        echo "批次完成，当前成功: $SUCCESS_COUNT/$TOTAL_PORTS"
        echo ""
        
        # 批次间休息
        if [ $BATCH_END -lt $TOTAL_PORTS ]; then
            echo "批次间休息 10秒..."
            sleep 10
        fi
    done
    
    return $SUCCESS_COUNT
}

# 主函数
main() {
    local START_TIME=$(date +%s)
    
    echo "开始智能下载..."
    smart_download_strategy "$NUM_INSTANCES" "$START_PORT" "$MODEL_NAME"
    local SUCCESS_COUNT=$?
    
    # 计算总耗时
    local END_TIME=$(date +%s)
    local TOTAL_TIME=$((END_TIME - START_TIME))
    local MINUTES=$((TOTAL_TIME / 60))
    local SECONDS=$((TOTAL_TIME % 60))
    
    echo ""
    echo "=============================="
    echo "智能下载完成！总耗时: ${MINUTES}分${SECONDS}秒"
    echo ""
    
    # 最终验证
    echo "最终验证结果:"
    echo "------------------------------"
    local FINAL_SUCCESS=0
    for ((i=0; i<$NUM_INSTANCES; i++)); do
        local PORT=$((START_PORT + i))
        if check_model_exists "$PORT" "$MODEL_NAME"; then
            echo "端口 $PORT: ✓ 模型可用"
            FINAL_SUCCESS=$((FINAL_SUCCESS + 1))
        else
            echo "端口 $PORT: ✗ 模型不可用"
        fi
    done
    
    echo ""
    echo "最终成功: $FINAL_SUCCESS/$NUM_INSTANCES 个实例"
    if [ $FINAL_SUCCESS -eq $NUM_INSTANCES ]; then
        echo "🎉 所有实例模型下载成功！"
    else
        echo "⚠️  部分实例下载失败"
        echo "故障排除建议:"
        echo "1. 检查网络连接: ping -c 3 ollama.ai"
        echo "2. 检查磁盘空间: df -h"
        echo "3. 检查实例状态: ./scripts/ollama_manager.sh status"
        echo "4. 查看详细日志: ls -la $LOG_DIR/"
        echo "5. 手动重试失败的实例"
    fi
    
    # 清理临时文件
    rm -f /tmp/progress_*
}

# 执行主函数
main