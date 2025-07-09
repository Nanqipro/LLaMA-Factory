#!/bin/bash

# TensorBoard 启动脚本
# 用法：bash start_tensorboard.sh

LOG_DIR="logs/qwen2.5-3b-bespoke-stratos/lora/sft"
PORT=6006

echo "🌐 启动 TensorBoard 可视化工具"
echo "=========================================="

# 检查日志目录是否存在
if [ ! -d "$LOG_DIR" ]; then
    echo "❌ 日志目录不存在: $LOG_DIR"
    echo "请先启动训练: bash train_bespoke_stratos.sh"
    exit 1
fi

# 检查是否已有TensorBoard进程在运行
if pgrep -f "tensorboard.*$PORT" >/dev/null; then
    echo "⚠️  TensorBoard 已在端口 $PORT 上运行"
    echo "请访问: http://localhost:$PORT"
    echo "如需重启，请先执行: pkill -f tensorboard"
    exit 1
fi

# 检查端口是否被占用
if netstat -tuln 2>/dev/null | grep ":$PORT " >/dev/null; then
    echo "⚠️  端口 $PORT 已被占用，尝试使用其他端口..."
    PORT=$((PORT + 1))
    while netstat -tuln 2>/dev/null | grep ":$PORT " >/dev/null; do
        PORT=$((PORT + 1))
        if [ $PORT -gt 6020 ]; then
            echo "❌ 无法找到可用端口"
            exit 1
        fi
    done
    echo "✅ 使用端口: $PORT"
fi

echo "📂 日志目录: $LOG_DIR"
echo "🌐 TensorBoard地址: http://localhost:$PORT"
echo "🖥️  如果是远程服务器，请使用端口转发："
echo "   ssh -L $PORT:localhost:$PORT username@your_server_ip"
echo "=========================================="
echo "📊 启动中..."
echo "按 Ctrl+C 停止TensorBoard"
echo ""

# 启动TensorBoard
tensorboard --logdir="$LOG_DIR" --port=$PORT --host=0.0.0.0 --reload_interval=30

echo ""
echo "�� TensorBoard 已停止" 