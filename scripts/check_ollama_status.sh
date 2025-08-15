#!/bin/bash

# 检查Ollama实例状态的脚本

echo "检查Ollama实例状态..."
echo "=============================="

# 检查ollama进程
OLLAMA_PROCESSES=$(pgrep -f "ollama serve" | wc -l)
echo "运行中的ollama进程数: $OLLAMA_PROCESSES"

if [ $OLLAMA_PROCESSES -gt 0 ]; then
    echo ""
    echo "进程详情:"
    ps aux | grep "ollama serve" | grep -v grep
fi

echo ""
echo "端口状态检查:"
echo "------------------------------"

# 检查常用端口范围
for port in {11434..11440}; do
    if lsof -i :$port > /dev/null 2>&1; then
        # 端口被占用，检查是否是ollama服务
        if curl -s "http://localhost:$port/api/version" > /dev/null 2>&1; then
            VERSION=$(curl -s "http://localhost:$port/api/version" | grep -o '"version":"[^"]*"' | cut -d'"' -f4)
            echo "端口 $port: ✓ Ollama服务正常 (版本: $VERSION)"
        else
            echo "端口 $port: ✗ 端口被占用但非Ollama服务"
        fi
    else
        echo "端口 $port: - 空闲"
    fi
done

echo ""
echo "模型列表检查:"
echo "------------------------------"

# 检查第一个可用的ollama实例的模型
for port in {11434..11440}; do
    if curl -s "http://localhost:$port/api/version" > /dev/null 2>&1; then
        echo "从端口 $port 获取模型列表:"
        OLLAMA_HOST="localhost:$port" ollama list 2>/dev/null || echo "无法获取模型列表"
        break
    fi
done

echo ""
echo "日志文件:"
echo "------------------------------"

LOG_DIR="./logs/ollama"
if [ -d "$LOG_DIR" ]; then
    echo "日志目录: $LOG_DIR"
    ls -la "$LOG_DIR"/*.log 2>/dev/null || echo "无日志文件"
    
    echo ""
    echo "最近的错误 (如果有):"
    for log_file in "$LOG_DIR"/*.log; do
        if [ -f "$log_file" ]; then
            echo "--- $(basename "$log_file") ---"
            tail -5 "$log_file" | grep -i error || echo "无错误信息"
        fi
    done
else
    echo "日志目录不存在: $LOG_DIR"
fi

echo ""
echo "系统资源使用:"
echo "------------------------------"
echo "内存使用:"
ps aux | grep "ollama" | grep -v grep | awk '{sum+=$6} END {printf "Ollama总内存使用: %.2f MB\n", sum/1024}'

echo ""
echo "CPU使用:"
ps aux | grep "ollama" | grep -v grep | awk '{sum+=$3} END {printf "Ollama总CPU使用: %.1f%%\n", sum}'