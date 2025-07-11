#!/bin/bash

# Sky-T1 Data 17k 思维链推理数据集 LoRA 微调启动脚本
# 用法：bash train_bespoke_stratos.sh

echo "开始 Sky-T1 Data 17k 思维链推理数据集的 LoRA 微调..."

# 设置日志目录和文件
LOG_DIR="logs/x-r1-3b-sky-t1/lora/sft"
LOG_FILE="${LOG_DIR}/training_$(date +%Y%m%d_%H%M%S).log"
ERROR_LOG="${LOG_DIR}/error_$(date +%Y%m%d_%H%M%S).log"

# 创建日志目录
mkdir -p "$LOG_DIR"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=2,3  # 指定GPU，多卡可设置为 0,1,2,3
export WANDB_DISABLED=true     # 禁用wandb日志

echo "日志将保存到: $LOG_FILE"
echo "错误日志将保存到: $ERROR_LOG"

# 检查数据集是否存在
if [ ! -f "LLM-models-datasets/Sky-T1_data_17k_default/NovaSky-AI___sky-t1_data_17k/default-a3f903710ddeb1b7/0.0.0/master/sky-t1_data_17k-train.arrow" ]; then
    echo "错误：数据集文件不存在，请确保 Sky-T1 Data 17k 数据集已正确放置在指定路径！" | tee -a "$ERROR_LOG"
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
    echo "=========================================="
    echo "配置文件内容:"
    cat examples/train_lora/sky_t1_lora_sft.yaml
    echo "=========================================="
    echo "开始训练..."
} > "$LOG_FILE"

# 开始训练，输出到日志文件
{
    echo "训练命令: llamafactory-cli train examples/train_lora/sky_t1_lora_sft.yaml"
    echo "----------------------------------------"
    llamafactory-cli train examples/train_lora/sky_t1_lora_sft.yaml
    TRAIN_EXIT_CODE=$?
    
    echo "----------------------------------------"
    echo "训练结束时间: $(date)"
    echo "训练退出代码: $TRAIN_EXIT_CODE"
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        echo "✅ 训练成功完成！"
        echo "📁 模型保存在: saves/x-r1-3b-sky-t1/lora/sft/"
        echo "📊 训练日志保存在: $LOG_FILE"
        echo "📈 TensorBoard日志保存在: logs/x-r1-3b-sky-t1/lora/sft/"
        
        # 输出最终的模型信息
        if [ -d "saves/x-r1-3b-sky-t1/lora/sft" ]; then
            echo "🗂️  输出文件列表:"
            ls -la saves/x-r1-3b-sky-t1/lora/sft/
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