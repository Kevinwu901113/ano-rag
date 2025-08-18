#!/bin/bash

# vLLM 服务启动脚本
# 支持启动不同规模的模型和多GPU配置

set -e

# 默认配置
DEFAULT_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_SERVED_NAME="qwen2_5_0_5b"
DEFAULT_PORT=8001
DEFAULT_GPU_MEMORY=0.80
DEFAULT_MAX_MODEL_LEN=4096
DEFAULT_DTYPE="float16"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志目录
LOG_DIR="./logs/vllm"
mkdir -p "$LOG_DIR"

# 打印帮助信息
show_help() {
    echo "vLLM 服务启动脚本"
    echo ""
    echo "用法: $0 [选项] [命令]"
    echo ""
    echo "命令:"
    echo "  start-tiny     启动 Qwen2.5-0.5B 模型 (默认)"
    echo "  start-small    启动 Qwen2.5-1.5B 模型"
    echo "  start-medium   启动 Qwen2.5-7B 模型"
    echo "  start-large    启动 Qwen2.5-14B 模型"
    echo "  start-custom   启动自定义模型"
    echo "  stop           停止 vLLM 服务"
    echo "  status         查看服务状态"
    echo "  test           测试服务可用性"
    echo "  logs           查看日志"
    echo "  benchmark      运行压测"
    echo "  help           显示此帮助信息"
    echo ""
    echo "选项:"
    echo "  --model MODEL              模型路径或名称"
    echo "  --served-name NAME          服务模型名称"
    echo "  --port PORT                 服务端口 (默认: 8001)"
    echo "  --gpu-memory RATIO          GPU内存使用率 (默认: 0.80)"
    echo "  --max-model-len LENGTH      最大模型长度 (默认: 4096)"
    echo "  --tensor-parallel SIZE      张量并行大小 (多GPU)"
    echo "  --dtype DTYPE               数据类型 (默认: float16)"
    echo "  --background                后台运行"
    echo ""
    echo "示例:"
    echo "  $0 start-tiny                    # 启动小模型"
    echo "  $0 start-medium --port 8002      # 在8002端口启动中等模型"
    echo "  $0 start-custom --model /path/to/model --tensor-parallel 2"
    echo "  $0 test --port 8001              # 测试8001端口的服务"
}

# 解析命令行参数
MODEL="$DEFAULT_MODEL"
SERVED_NAME="$DEFAULT_SERVED_NAME"
PORT="$DEFAULT_PORT"
GPU_MEMORY="$DEFAULT_GPU_MEMORY"
MAX_MODEL_LEN="$DEFAULT_MAX_MODEL_LEN"
DTYPE="$DEFAULT_DTYPE"
TENSOR_PARALLEL=""
BACKGROUND=false
COMMAND="start-tiny"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --served-name)
            SERVED_NAME="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --gpu-memory)
            GPU_MEMORY="$2"
            shift 2
            ;;
        --max-model-len)
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        --tensor-parallel)
            TENSOR_PARALLEL="$2"
            shift 2
            ;;
        --dtype)
            DTYPE="$2"
            shift 2
            ;;
        --background)
            BACKGROUND=true
            shift
            ;;
        start-tiny|start-small|start-medium|start-large|start-custom|stop|status|test|logs|benchmark|help)
            COMMAND="$1"
            shift
            ;;
        *)
            echo "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# 检查 vLLM 是否安装
check_vllm() {
    if ! python -c "import vllm" 2>/dev/null; then
        echo -e "${RED}错误: vLLM 未安装${NC}"
        echo "请安装 vLLM: pip install vllm"
        exit 1
    fi
}

# 检查 GPU
check_gpu() {
    if ! nvidia-smi >/dev/null 2>&1; then
        echo -e "${YELLOW}警告: 未检测到 NVIDIA GPU${NC}"
        echo "vLLM 将使用 CPU 模式，性能可能较低"
    else
        echo -e "${GREEN}检测到 NVIDIA GPU:${NC}"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | nl
    fi
}

# 获取进程ID
get_vllm_pid() {
    local port=$1
    pgrep -f "vllm.entrypoints.openai.api_server.*--port $port" || echo ""
}

# 启动 vLLM 服务
start_vllm() {
    local model=$1
    local served_name=$2
    local port=$3
    
    echo -e "${BLUE}启动 vLLM 服务...${NC}"
    echo "模型: $model"
    echo "服务名称: $served_name"
    echo "端口: $port"
    echo "GPU内存使用率: $GPU_MEMORY"
    echo "最大模型长度: $MAX_MODEL_LEN"
    echo "数据类型: $DTYPE"
    
    # 检查端口是否被占用
    if lsof -i :$port >/dev/null 2>&1; then
        echo -e "${RED}错误: 端口 $port 已被占用${NC}"
        echo "请使用其他端口或停止占用该端口的服务"
        exit 1
    fi
    
    # 构建启动命令
    local cmd="python -m vllm.entrypoints.openai.api_server"
    cmd="$cmd --model $model"
    cmd="$cmd --served-model-name $served_name"
    cmd="$cmd --dtype $DTYPE"
    cmd="$cmd --max-model-len $MAX_MODEL_LEN"
    cmd="$cmd --gpu-memory-utilization $GPU_MEMORY"
    cmd="$cmd --port $port"
    
    # 添加张量并行
    if [[ -n "$TENSOR_PARALLEL" ]]; then
        cmd="$cmd --tensor-parallel-size $TENSOR_PARALLEL"
        echo "张量并行大小: $TENSOR_PARALLEL"
    fi
    
    local log_file="$LOG_DIR/vllm_${served_name}_${port}.log"
    
    echo "日志文件: $log_file"
    echo "启动命令: $cmd"
    echo ""
    
    if [[ "$BACKGROUND" == "true" ]]; then
        echo -e "${YELLOW}后台启动中...${NC}"
        nohup $cmd > "$log_file" 2>&1 &
        local pid=$!
        echo "进程ID: $pid"
        echo "使用 '$0 logs --port $port' 查看日志"
        echo "使用 '$0 status --port $port' 检查状态"
        
        # 等待服务启动
        echo "等待服务启动..."
        for i in {1..30}; do
            if curl -s "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
                echo -e "${GREEN}✅ vLLM 服务启动成功！${NC}"
                echo "API 地址: http://127.0.0.1:$port/v1"
                return 0
            fi
            sleep 2
            echo -n "."
        done
        echo -e "\n${YELLOW}⚠️  服务启动时间较长，请检查日志${NC}"
    else
        echo -e "${YELLOW}前台启动中... (按 Ctrl+C 停止)${NC}"
        $cmd
    fi
}

# 停止 vLLM 服务
stop_vllm() {
    local port=${1:-$PORT}
    echo -e "${BLUE}停止端口 $port 上的 vLLM 服务...${NC}"
    
    local pid=$(get_vllm_pid $port)
    if [[ -n "$pid" ]]; then
        echo "找到进程 ID: $pid"
        kill $pid
        sleep 2
        
        # 强制杀死如果还在运行
        if kill -0 $pid 2>/dev/null; then
            echo "强制停止进程..."
            kill -9 $pid
        fi
        
        echo -e "${GREEN}✅ vLLM 服务已停止${NC}"
    else
        echo -e "${YELLOW}⚠️  未找到运行在端口 $port 的 vLLM 服务${NC}"
    fi
}

# 查看服务状态
show_status() {
    local port=${1:-$PORT}
    echo -e "${BLUE}检查端口 $port 上的 vLLM 服务状态...${NC}"
    
    local pid=$(get_vllm_pid $port)
    if [[ -n "$pid" ]]; then
        echo -e "${GREEN}✅ 服务正在运行${NC}"
        echo "进程ID: $pid"
        echo "端口: $port"
        
        # 检查API可用性
        if curl -s "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
            echo -e "${GREEN}✅ API 可访问${NC}"
            echo "API 地址: http://127.0.0.1:$port/v1"
        else
            echo -e "${YELLOW}⚠️  API 暂不可访问（可能正在启动中）${NC}"
        fi
    else
        echo -e "${RED}❌ 服务未运行${NC}"
    fi
}

# 测试服务
test_service() {
    local port=${1:-$PORT}
    echo -e "${BLUE}测试端口 $port 上的 vLLM 服务...${NC}"
    
    # 测试模型列表
    echo "1. 测试模型列表 API..."
    if curl -s "http://127.0.0.1:$port/v1/models" | jq . 2>/dev/null; then
        echo -e "${GREEN}✅ 模型列表 API 正常${NC}"
    else
        echo -e "${RED}❌ 模型列表 API 失败${NC}"
        return 1
    fi
    
    # 测试聊天完成
    echo "\n2. 测试聊天完成 API..."
    local test_response=$(curl -s "http://127.0.0.1:$port/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer EMPTY" \
        -d '{
            "model": "'$SERVED_NAME'",
            "messages": [{"role": "user", "content": "Hello!"}],
            "max_tokens": 50,
            "temperature": 0.0
        }')
    
    if echo "$test_response" | jq -e '.choices[0].message.content' >/dev/null 2>&1; then
        echo -e "${GREEN}✅ 聊天完成 API 正常${NC}"
        echo "响应: $(echo "$test_response" | jq -r '.choices[0].message.content')"
    else
        echo -e "${RED}❌ 聊天完成 API 失败${NC}"
        echo "响应: $test_response"
        return 1
    fi
    
    echo -e "\n${GREEN}🎉 所有测试通过！${NC}"
}

# 查看日志
show_logs() {
    local port=${1:-$PORT}
    local log_file="$LOG_DIR/vllm_${SERVED_NAME}_${port}.log"
    
    if [[ -f "$log_file" ]]; then
        echo -e "${BLUE}显示日志: $log_file${NC}"
        tail -f "$log_file"
    else
        echo -e "${YELLOW}⚠️  日志文件不存在: $log_file${NC}"
        echo "可用的日志文件:"
        ls -la "$LOG_DIR/" 2>/dev/null || echo "无日志文件"
    fi
}

# 运行压测
run_benchmark() {
    local port=${1:-$PORT}
    echo -e "${BLUE}运行 vLLM 压测...${NC}"
    
    if [[ -f "benchmark_vllm.py" ]]; then
        python benchmark_vllm.py --base-url "http://127.0.0.1:$port/v1" --model "$SERVED_NAME" --progressive
    else
        echo -e "${RED}错误: 未找到 benchmark_vllm.py${NC}"
        echo "请确保压测脚本存在"
    fi
}

# 主逻辑
case $COMMAND in
    start-tiny)
        check_vllm
        check_gpu
        MODEL="Qwen/Qwen2.5-0.5B-Instruct"
        SERVED_NAME="qwen2_5_0_5b"
        BACKGROUND=true
        start_vllm "$MODEL" "$SERVED_NAME" "$PORT"
        ;;
    start-small)
        check_vllm
        check_gpu
        MODEL="Qwen/Qwen2.5-1.5B-Instruct"
        SERVED_NAME="qwen2_5_1_5b"
        PORT=${PORT:-8002}
        BACKGROUND=true
        start_vllm "$MODEL" "$SERVED_NAME" "$PORT"
        ;;
    start-medium)
        check_vllm
        check_gpu
        MODEL="Qwen/Qwen2.5-7B-Instruct"
        SERVED_NAME="qwen2_5_7b"
        PORT=${PORT:-8003}
        BACKGROUND=true
        start_vllm "$MODEL" "$SERVED_NAME" "$PORT"
        ;;
    start-large)
        check_vllm
        check_gpu
        MODEL="Qwen/Qwen2.5-14B-Instruct"
        SERVED_NAME="qwen2_5_14b"
        PORT=${PORT:-8004}
        TENSOR_PARALLEL=${TENSOR_PARALLEL:-2}
        BACKGROUND=true
        start_vllm "$MODEL" "$SERVED_NAME" "$PORT"
        ;;
    start-custom)
        check_vllm
        check_gpu
        BACKGROUND=true
        start_vllm "$MODEL" "$SERVED_NAME" "$PORT"
        ;;
    stop)
        stop_vllm "$PORT"
        ;;
    status)
        show_status "$PORT"
        ;;
    test)
        test_service "$PORT"
        ;;
    logs)
        show_logs "$PORT"
        ;;
    benchmark)
        run_benchmark "$PORT"
        ;;
    help)
        show_help
        ;;
    *)
        echo "未知命令: $COMMAND"
        show_help
        exit 1
        ;;
esac