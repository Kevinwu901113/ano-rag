#!/bin/bash

# Ollama多实例管理工具
# 统一管理Ollama实例的启动、停止、状态检查和模型下载

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 显示帮助信息
show_help() {
    echo "Ollama多实例管理工具"
    echo "用法: $0 <命令> [参数...]"
    echo ""
    echo "命令:"
    echo "  start <实例数> [起始端口] [模型名]    启动多个Ollama实例"
    echo "  stop                                停止所有Ollama实例"
    echo "  status                              检查实例状态"
    echo "  download <实例数> [起始端口] [模型名] 为实例下载模型（串行）"
    echo "  quick-download <实例数> [起始端口] [模型名] 快速并行下载模型"
    echo "  smart-download <实例数> [起始端口] [模型名] 智能下载模型（推荐）"
    echo "  gpu-start <实例数> [起始端口] [模型名]  GPU优化启动（解决显卡占用0问题）"
    echo "  restart <实例数> [起始端口] [模型名]  重启实例并下载模型"
    echo "  logs [端口]                         查看日志"
    echo "  test [端口] [模型名]                测试模型"
    echo "  cleanup                             清理日志和临时文件"
    echo ""
    echo "示例:"
    echo "  $0 start 4                         # 启动4个实例"
    echo "  $0 download 4 11434 gpt-oss:latest # 为4个实例下载模型"
    echo "  $0 smart-download 4                # 智能下载（推荐）"
    echo "  $0 quick-download 4                # 快速并行下载"
    echo "  $0 restart 4                       # 重启并配置4个实例"
    echo "  $0 test 11434 gpt-oss:latest       # 测试指定实例的模型"
    echo ""
    echo "默认参数:"
    echo "  实例数: 4"
    echo "  起始端口: 11434"
    echo "  模型名: gpt-oss:latest"
}

# 检查依赖
check_dependencies() {
    local missing_deps=()
    
    if ! command -v ollama &> /dev/null; then
        missing_deps+=("ollama")
    fi
    
    if ! command -v curl &> /dev/null; then
        missing_deps+=("curl")
    fi
    
    if [ ${#missing_deps[@]} -gt 0 ]; then
        echo "错误: 缺少依赖项: ${missing_deps[*]}"
        echo "请先安装这些工具"
        exit 1
    fi
}

# 启动实例
start_instances() {
    local num_instances=${1:-4}
    local start_port=${2:-11434}
    local model_name=${3:-"gpt-oss:latest"}
    
    echo "启动 $num_instances 个Ollama实例..."
    "$SCRIPT_DIR/start_multiple_ollama.sh" "$num_instances" "$start_port" "$model_name"
}

# 停止实例
stop_instances() {
    echo "停止所有Ollama实例..."
    "$SCRIPT_DIR/stop_multiple_ollama.sh"
}

# 检查状态
check_status() {
    echo "检查Ollama实例状态..."
    "$SCRIPT_DIR/check_ollama_status.sh"
}

# 下载模型（串行）
download_models() {
    local num_instances=${1:-4}
    local start_port=${2:-11434}
    local model_name=${3:-"gpt-oss:latest"}
    
    echo "为 $num_instances 个实例下载模型（串行模式）..."
    "$SCRIPT_DIR/download_models_for_instances.sh" "$num_instances" "$start_port" "$model_name"
}

# 快速下载模型（并行）
quick_download_models() {
    local num_instances=${1:-4}
    local start_port=${2:-11434}
    local model_name=${3:-"gpt-oss:latest"}
    
    echo "为 $num_instances 个实例快速下载模型（并行模式）..."
    "$SCRIPT_DIR/quick_download_models.sh" "$num_instances" "$start_port" "$model_name"
}

# 智能下载模型（推荐）
smart_download_models() {
    local num_instances=${1:-4}
    local start_port=${2:-11434}
    local model_name=${3:-"gpt-oss:latest"}
    
    echo "为 $num_instances 个实例智能下载模型（推荐模式）..."
    "$SCRIPT_DIR/smart_download_models.sh" "$num_instances" "$start_port" "$model_name"
}

# 重启并配置
restart_and_setup() {
    local num_instances=${1:-4}
    local start_port=${2:-11434}
    local model_name=${3:-"gpt-oss:latest"}
    
    echo "重启并配置 $num_instances 个Ollama实例..."
    echo "1. 停止现有实例"
    stop_instances
    sleep 3
    
    echo "2. 启动新实例"
    start_instances "$num_instances" "$start_port" "$model_name"
    sleep 5
    
    echo "3. 下载模型"
    smart_download_models "$num_instances" "$start_port" "$model_name"
    
    echo "4. 验证配置"
    check_status
}

# 查看日志
view_logs() {
    local port=${1}
    local log_dir="./logs/ollama"
    
    if [ -n "$port" ]; then
        local log_file="$log_dir/ollama_$port.log"
        if [ -f "$log_file" ]; then
            echo "查看端口 $port 的日志:"
            tail -f "$log_file"
        else
            echo "错误: 日志文件不存在: $log_file"
        fi
    else
        echo "可用的日志文件:"
        ls -la "$log_dir"/*.log 2>/dev/null || echo "无日志文件"
        echo ""
        echo "使用方法: $0 logs <端口号>"
    fi
}

# 测试模型
test_model() {
    local port=${1:-11434}
    local model_name=${2:-"gpt-oss:latest"}
    
    echo "测试端口 $port 的模型 $model_name..."
    
    # 检查实例是否运行
    if ! curl -s "http://localhost:$port/api/version" > /dev/null 2>&1; then
        echo "错误: 端口 $port 上的Ollama实例未运行"
        return 1
    fi
    
    # 检查模型是否存在
    if ! OLLAMA_HOST="localhost:$port" ollama list 2>/dev/null | grep -q "$model_name"; then
        echo "错误: 模型 $model_name 在端口 $port 上不可用"
        echo "可用模型:"
        OLLAMA_HOST="localhost:$port" ollama list
        return 1
    fi
    
    # 测试生成
    echo "发送测试请求..."
    echo "你好，请简单介绍一下自己。" | OLLAMA_HOST="localhost:$port" ollama run "$model_name"
}

# 清理文件
cleanup() {
    echo "清理日志和临时文件..."
    
    # 清理日志
    if [ -d "./logs" ]; then
        echo "清理日志目录..."
        rm -rf ./logs/ollama/*.log
        rm -rf ./logs/model_download/*.log
        rm -rf ./logs/parallel_download/*.log
    fi
    
    # 清理临时文件
    rm -f /tmp/download_exit_*
    
    echo "清理完成"
}

# 主函数
main() {
    if [ $# -eq 0 ]; then
        show_help
        exit 1
    fi
    
    check_dependencies
    
    case "$1" in
        "start")
            start_instances "$2" "$3" "$4"
            ;;
        "stop")
            stop_instances
            ;;
        "status")
            check_status
            ;;
        "download")
            download_models "$2" "$3" "$4"
            ;;
        "quick-download")
            quick_download_models "$2" "$3" "$4"
            ;;
        "smart-download")
            smart_download_models "$2" "$3" "$4"
            ;;
        "gpu-start")
            echo "启动GPU优化的Ollama实例..."
            "$SCRIPT_DIR/start_ollama_fixed.sh" "${2:-2}" "${3:-11440}" "${4:-gpt-oss:latest}"
            ;;
        "restart")
            restart_and_setup "$2" "$3" "$4"
            ;;
        "logs")
            view_logs "$2"
            ;;
        "test")
            test_model "$2" "$3"
            ;;
        "cleanup")
            cleanup
            ;;
        "help" | "-h" | "--help")
            show_help
            ;;
        *)
            echo "错误: 未知命令 '$1'"
            echo "使用 '$0 help' 查看帮助信息"
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@"