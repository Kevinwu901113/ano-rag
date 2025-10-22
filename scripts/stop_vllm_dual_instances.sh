#!/bin/bash

# vLLM 双实例停止脚本

set -e

LOG_DIR="logs/vllm"

echo "Stopping vLLM dual instances..."

# 停止函数
stop_instance() {
    local gpu_name=$1
    local pid_file="$LOG_DIR/${gpu_name}.pid"
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            echo "Stopping $gpu_name instance (PID: $pid)..."
            kill "$pid"
            
            # 等待进程结束
            local count=0
            while kill -0 "$pid" 2>/dev/null && [ $count -lt 30 ]; do
                sleep 1
                ((count++))
            done
            
            if kill -0 "$pid" 2>/dev/null; then
                echo "Force killing $gpu_name instance..."
                kill -9 "$pid"
            fi
            
            echo "✓ $gpu_name instance stopped"
        else
            echo "✓ $gpu_name instance was not running"
        fi
        rm -f "$pid_file"
    else
        echo "✓ No PID file found for $gpu_name instance"
    fi
}

# 停止所有实例
stop_instance "gpu0"
stop_instance "gpu1"

# 额外清理：通过端口查找并停止可能的残留进程
cleanup_by_port() {
    local port=$1
    local pids=$(lsof -ti:$port 2>/dev/null || true)
    
    if [ -n "$pids" ]; then
        echo "Found processes on port $port: $pids"
        echo "$pids" | xargs -r kill -9
        echo "✓ Cleaned up processes on port $port"
    fi
}

cleanup_by_port 8000
cleanup_by_port 8001

echo "✓ All vLLM instances stopped"