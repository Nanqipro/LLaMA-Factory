#!/bin/bash
#
# LLaMA Factory LoRA 后处理脚本
#
# 功能:
# 1. 交互式测试: 直接加载基础模型和LoRA适配器进行对话，快速验证效果。
# 2. 模型合并: 将LoRA适配器与基础模型合并，导出一个完整的、可直接部署的新模型。
#
# 使用方法:
# 1. 根据您的实际路径修改下面的 "配置参数" 部分。
# 2. 在终端中直接运行此脚本: ./post_train_lora.sh
# 3. 根据菜单提示选择要执行的操作。
#

# --- 配置参数 ---
# 请根据您的环境修改以下路径
BASE_MODEL_PATH="./LLM-models-datasets/Qwen2.5-3B"
ADAPTER_PATH="saves/qwen2.5-3b-generated-x-r1/lora/sft"
MERGED_MODEL_PATH="saves/qwen2.5-3b-generated-x-r1-merged"
TEMPLATE="qwen"
# 模型导出分片大小(GB)，根据需要调整
EXPORT_SIZE=4

# --- 脚本函数 ---

# 函数: 交互式测试
run_chat_test() {
    echo "----------------------------------------"
    echo "🚀 开始交互式测试 (加载LoRA适配器)..."
    echo "----------------------------------------"
    echo "基础模型: $BASE_MODEL_PATH"
    echo "适配器: $ADAPTER_PATH"
    echo "说明: "
    echo " - 直接与微调后的模型进行对话。"
    echo " - 测试完成后，按 Ctrl+C 退出对话模式。"
    echo ""

    llamafactory-cli chat \
        --model_name_or_path "$BASE_MODEL_PATH" \
        --adapter_name_or_path "$ADAPTER_PATH" \
        --template "$TEMPLATE"

    echo "✅ 交互式测试结束。"
}

# 函数: 合并模型
run_merge_model() {
    echo "----------------------------------------"
    echo "📦 开始合并模型..."
    echo "----------------------------------------"
    echo "基础模型: $BASE_MODEL_PATH"
    echo "适配器: $ADAPTER_PATH"
    echo "合并后输出到: $MERGED_MODEL_PATH"
    echo "分片大小: ${EXPORT_SIZE}GB"
    echo ""

    # 检查输出目录是否已存在
    if [ -d "$MERGED_MODEL_PATH" ]; then
        read -p "⚠️ 警告: 输出目录 '$MERGED_MODEL_PATH' 已存在。是否覆盖? (y/n): " confirm
        if [ "$confirm" != "y" ]; then
            echo "操作已取消。"
            return
        fi
    fi

    llamafactory-cli export \
        --model_name_or_path "$BASE_MODEL_PATH" \
        --adapter_name_or_path "$ADAPTER_PATH" \
        --export_dir "$MERGED_MODEL_PATH" \
        --export_size "$EXPORT_SIZE" \
        --template "$TEMPLATE"

    if [ $? -eq 0 ]; then
        echo "✅ 模型合并成功！"
        echo "合并后的完整模型保存在: $MERGED_MODEL_PATH"
    else
        echo "❌ 模型合并失败。"
    fi
}

# --- 主逻辑 ---

# 主菜单
while true; do
    echo "========================================"
    echo "      LLaMA Factory LoRA 后处理"
    echo "========================================"
    echo "请选择要执行的操作:"
    echo "  1. 交互式测试 (Chat with Adapter)"
    echo "  2. 合并模型 (Merge to Standalone Model)"
    echo "  3. 退出脚本"
    echo "----------------------------------------"
    read -p "请输入选项 [1-3]: " choice

    case $choice in
        1)
            run_chat_test
            read -p "按任意键返回主菜单..."
            ;;
        2)
            run_merge_model
            read -p "按任意键返回主菜单..."
            ;;
        3)
            echo "👋 脚本退出。"
            exit 0
            ;;
        *)
            echo "❌ 无效选项，请输入 1, 2, 或 3."
            sleep 2
            ;;
    esac
    echo -e "\n\n"
done 