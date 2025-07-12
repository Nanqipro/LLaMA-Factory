#!/bin/bash

# OpenR1-Math-220k LoRA 微调启动脚本
# 用法：bash train_openr1_math_220k.sh

echo "开始 OpenR1-Math-220k 数据集的 LoRA 微调..."

# 设置日志目录和文件
LOG_DIR="logs/qwen2.5-3b-openr1-math-220k/lora/sft"
LOG_FILE="${LOG_DIR}/training_$(date +%Y%m%d_%H%M%S).log"
ERROR_LOG="${LOG_DIR}/error_$(date +%Y%m%d_%H%M%S).log"

# 创建日志目录
mkdir -p "$LOG_DIR"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 指定GPU，多卡可设置为 0,1,2,3
export WANDB_DISABLED=true     # 禁用wandb日志

echo "日志将保存到: $LOG_FILE"
echo "错误日志将保存到: $ERROR_LOG"

# 检查数据集是否存在
if [ ! -d "LLM-models-datasets/OpenR1-Math-220k_default/open-r1___open_r1-math-220k/default-f85bbd709552ea85/0.0.0/master" ]; then
    echo "错误：数据集文件不存在，请确保 OpenR1-Math-220k 数据集已正确下载！" | tee -a "$ERROR_LOG"
    exit 1
fi

# 检查是否有足够的arrow文件
ARROW_COUNT=$(find "LLM-models-datasets/OpenR1-Math-220k_default/open-r1___open_r1-math-220k/default-f85bbd709552ea85/0.0.0/master/" -name "*.arrow" | wc -l)
if [ "$ARROW_COUNT" -lt 10 ]; then
    echo "错误：数据集文件不完整，预期10个arrow文件，实际找到 $ARROW_COUNT 个！" | tee -a "$ERROR_LOG"
    exit 1
fi

# 输出系统信息到日志
{
    echo "=========================================="
    echo "训练开始时间: $(date)"
    echo "=========================================="
    echo "系统信息:"
    echo "- 操作系统: $(uname -a)"
    echo "- Python版本: $(python --version)"
    echo "- CUDA设备: $CUDA_VISIBLE_DEVICES"
    echo "- GPU信息:"
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits 2>/dev/null || echo "无法获取GPU信息"
    echo "- 数据集信息:"
    echo "  - 数据集名称: OpenR1-Math-220k"
    echo "  - 数据集样本数: 93,733"
    echo "  - Arrow文件数量: $ARROW_COUNT"
    echo "=========================================="
    echo "配置文件内容:"
    cat examples/train_lora/openr1_math_220k_lora_sft.yaml
    echo "=========================================="
    echo "开始训练..."
} > "$LOG_FILE"

# 开始训练，输出到日志文件
{
    echo "训练命令: llamafactory-cli train examples/train_lora/openr1_math_220k_lora_sft.yaml"
    echo "----------------------------------------"
    llamafactory-cli train examples/train_lora/openr1_math_220k_lora_sft.yaml
    TRAIN_EXIT_CODE=$?
    
    echo "----------------------------------------"
    echo "训练结束时间: $(date)"
    echo "训练退出代码: $TRAIN_EXIT_CODE"
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        echo "✅ 训练成功完成！"
        echo "📁 模型保存在: saves/qwen2.5-3b-openr1-math-220k/lora/sft/"
        echo "📊 训练日志保存在: $LOG_FILE"
        echo "📈 TensorBoard日志保存在: logs/qwen2.5-3b-openr1-math-220k/lora/sft/"
        
        # 输出最终的模型信息
        if [ -d "saves/qwen2.5-3b-openr1-math-220k/lora/sft" ]; then
            echo "🗂️  输出文件列表:"
            ls -la saves/qwen2.5-3b-openr1-math-220k/lora/sft/
        fi
    else
        echo "❌ 训练失败，退出代码: $TRAIN_EXIT_CODE"
        echo "请检查错误日志: $ERROR_LOG"
    fi
    
} 2>&1 | tee -a "$LOG_FILE"

# 将错误信息也保存到错误日志
if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo "训练失败，详细错误信息:" > "$ERROR_LOG"
    tail -100 "$LOG_FILE" >> "$ERROR_LOG"
fi

echo "=========================================="
echo "训练流程结束"
echo "📋 完整日志: $LOG_FILE"
echo "🔍 错误日志: $ERROR_LOG"
echo "📊 TensorBoard日志: $LOG_DIR"
echo "=========================================="

# 可选：训练完成后显示模型性能概览
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "🎯 训练完成总结:"
    echo "  - 数据集: OpenR1-Math-220k (93,733 样本)"
    echo "  - 模型: Qwen2.5-3B"
    echo "  - 方法: LoRA微调"
    echo "  - 输出目录: saves/qwen2.5-3b-openr1-math-220k/lora/sft/"
    echo ""
    echo "🚀 下一步操作建议:"
    echo "  1. 使用 llamafactory-cli chat 进行交互式测试"
    echo "  2. 使用 llamafactory-cli export 导出完整模型"
    echo "  3. 在数学推理任务上评估模型性能"
fi 