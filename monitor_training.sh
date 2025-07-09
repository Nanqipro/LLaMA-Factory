#!/bin/bash

# 训练监控脚本
# 用法：bash monitor_training.sh

LOG_DIR="logs/qwen2.5-3b-bespoke-stratos/lora/sft"

echo "🔍 Bespoke-Stratos-17k 训练监控工具"
echo "=========================================="

# 检查日志目录是否存在
if [ ! -d "$LOG_DIR" ]; then
    echo "❌ 日志目录不存在: $LOG_DIR"
    echo "请先启动训练: bash train_bespoke_stratos.sh"
    exit 1
fi

# 查找最新的日志文件
LATEST_LOG=$(find "$LOG_DIR" -name "training_*.log" -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)

if [ -z "$LATEST_LOG" ]; then
    echo "❌ 未找到训练日志文件"
    echo "可用的日志文件:"
    ls -la "$LOG_DIR"/*.log 2>/dev/null || echo "无日志文件"
    exit 1
fi

echo "📋 正在监控最新日志: $(basename "$LATEST_LOG")"
echo "📂 完整路径: $LATEST_LOG"
echo "=========================================="

# 显示训练选项
echo "请选择监控方式:"
echo "1) 实时显示最新日志 (tail -f)"
echo "2) 查看完整日志内容"
echo "3) 查看训练进度和损失"
echo "4) 查看GPU使用情况"
echo "5) 启动TensorBoard"
echo "6) 查看错误日志"
echo "7) 显示训练统计信息"
echo "=========================================="
read -p "请输入选择 [1-7]: " choice

case $choice in
    1)
        echo "🔄 实时显示训练日志 (按 Ctrl+C 退出):"
        echo "----------------------------------------"
        tail -f "$LATEST_LOG"
        ;;
    2)
        echo "📖 完整日志内容:"
        echo "----------------------------------------"
        cat "$LATEST_LOG"
        ;;
    3)
        echo "📊 训练进度和损失:"
        echo "----------------------------------------"
        # 提取训练步数和损失信息
        grep -E "(Step|train_loss|eval_loss|epoch)" "$LATEST_LOG" 2>/dev/null || echo "暂无训练进度信息"
        ;;
    4)
        echo "🖥️  GPU使用情况:"
        echo "----------------------------------------"
        watch -n 2 nvidia-smi
        ;;
    5)
        echo "🌐 启动TensorBoard..."
        echo "----------------------------------------"
        echo "TensorBoard地址: http://localhost:6006"
        echo "按 Ctrl+C 停止TensorBoard"
        tensorboard --logdir="$LOG_DIR" --port=6006 --host=0.0.0.0
        ;;
    6)
        echo "🚨 错误日志:"
        echo "----------------------------------------"
        LATEST_ERROR_LOG=$(find "$LOG_DIR" -name "error_*.log" -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
        if [ -n "$LATEST_ERROR_LOG" ]; then
            cat "$LATEST_ERROR_LOG"
        else
            echo "暂无错误日志"
        fi
        ;;
    7)
        echo "📈 训练统计信息:"
        echo "----------------------------------------"
        echo "📂 日志目录: $LOG_DIR"
        echo "📄 日志文件数量: $(find "$LOG_DIR" -name "*.log" -type f | wc -l)"
        echo "📊 日志文件大小:"
        du -h "$LOG_DIR"/*.log 2>/dev/null || echo "无日志文件"
        echo ""
        echo "🕒 最新训练时间:"
        if [ -f "$LATEST_LOG" ]; then
            echo "开始时间: $(head -20 "$LATEST_LOG" | grep "训练开始时间" | cut -d' ' -f2-)"
            echo "最后更新: $(stat -c '%y' "$LATEST_LOG")"
        fi
        echo ""
        echo "📝 训练状态:"
        if pgrep -f "llamafactory-cli train" >/dev/null; then
            echo "✅ 训练进程正在运行"
        else
            echo "⏹️  训练进程未运行"
        fi
        ;;
    *)
        echo "❌ 无效选择，请输入 1-7"
        exit 1
        ;;
esac 