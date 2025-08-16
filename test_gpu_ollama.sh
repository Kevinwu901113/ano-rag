#!/bin/bash

# 测试Ollama GPU加速的脚本

echo "=== Ollama GPU加速测试 ==="
echo "测试时间: $(date)"
echo ""

# 检查GPU状态
echo "1. GPU状态检查:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo ""

# 检查CUDA环境
echo "2. CUDA环境检查:"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "CUDA版本: $(nvcc --version | grep 'release' | awk '{print $6}')"
echo ""

# 启动一个测试实例
echo "3. 启动测试Ollama实例 (端口 11440)..."
export CUDA_VISIBLE_DEVICES="1"  # 使用GPU 1
export OLLAMA_GPU_LAYERS=35
export OLLAMA_NUM_PARALLEL=1

# 启动ollama服务
OLLAMA_HOST="0.0.0.0:11440" ollama serve > ./test_ollama.log 2>&1 &
TEST_PID=$!
echo "测试实例PID: $TEST_PID"

# 等待服务启动
echo "等待服务启动..."
sleep 5

# 检查服务状态
if curl -s "http://localhost:11440/api/version" > /dev/null; then
    echo "✓ 测试实例启动成功"
else
    echo "✗ 测试实例启动失败"
    kill $TEST_PID 2>/dev/null
    exit 1
fi

echo ""
echo "4. 测试模型推理 (GPU加速)..."
echo "发送测试请求..."

# 记录开始时间
start_time=$(date +%s.%N)

# 发送测试请求
OLLAMA_HOST="localhost:11440" timeout 30s ollama run gpt-oss:latest "请简单介绍一下人工智能" > ./test_response.txt 2>&1 &
RUN_PID=$!

# 监控GPU使用情况
echo "监控GPU使用情况 (10秒)..."
for i in {1..10}; do
    echo "第 $i 秒:"
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | awk '{print "  GPU利用率: " $1 "%, 显存使用: " $2 "MB"}'
    sleep 1
done

# 等待推理完成
wait $RUN_PID
end_time=$(date +%s.%N)

# 计算耗时
duration=$(echo "$end_time - $start_time" | bc)

echo ""
echo "5. 测试结果:"
echo "推理耗时: ${duration}秒"

if [ -f "./test_response.txt" ] && [ -s "./test_response.txt" ]; then
    echo "✓ 模型响应成功"
    echo "响应内容:"
    head -5 ./test_response.txt
else
    echo "✗ 模型响应失败"
fi

echo ""
echo "6. 检查模型是否使用GPU:"
OLLAMA_HOST="localhost:11440" ollama ps

echo ""
echo "7. 最终GPU状态:"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | awk '{print "GPU利用率: " $1 "%, 显存使用: " $2 "MB"}'

echo ""
echo "8. 清理测试实例..."
kill $TEST_PID 2>/dev/null
sleep 2

echo "测试完成！"
echo "日志文件: ./test_ollama.log"
echo "响应文件: ./test_response.txt"