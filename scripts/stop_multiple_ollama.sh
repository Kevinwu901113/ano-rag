#!/bin/bash

# 停止多个Ollama实例的脚本

echo "停止所有Ollama实例..."

# 日志目录
LOG_DIR="./logs/ollama"

# 停止所有ollama进程
echo "终止ollama进程..."
pkill -f "ollama serve"

# 等待进程完全停止
sleep 2

# 清理PID文件
if [ -d "$LOG_DIR" ]; then
    echo "清理PID文件..."
    rm -f "$LOG_DIR"/*.pid
fi

# 检查是否还有残留进程
REMAINING=$(pgrep -f "ollama serve" | wc -l)
if [ $REMAINING -gt 0 ]; then
    echo "警告: 仍有 $REMAINING 个ollama进程在运行"
    echo "强制终止残留进程..."
    pkill -9 -f "ollama serve"
    sleep 1
fi

echo "所有Ollama实例已停止"

# 显示端口占用情况
echo ""
echo "检查端口占用情况:"
for port in {11434..11440}; do
    if lsof -i :$port > /dev/null 2>&1; then
        echo "端口 $port: 仍被占用"
    else
        echo "端口 $port: 空闲"
    fi
done