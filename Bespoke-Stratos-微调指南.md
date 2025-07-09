# Bespoke-Stratos-17k 数据集 LoRA 微调指南

## 概述

本指南介绍如何使用 LLaMA-Factory 对 Bespoke-Stratos-17k 数据集进行 LoRA 微调。该数据集包含 16,710 个高质量的对话样本，采用 sharegpt 格式，非常适合用于指令微调。

## 数据集信息

- **数据集名称**: Bespoke-Stratos-17k
- **样本数量**: 16,710
- **格式**: sharegpt（包含 system 和 conversations 字段）
- **文件类型**: Arrow 格式
- **兼容性**: 完全兼容 LLaMA-Factory

## 预备条件

### 1. 环境要求
- Python 3.8+
- PyTorch 1.13+
- CUDA 11.8+ (推荐)
- 显存要求: 至少 12GB（8B 模型）

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 数据集检查
确保数据集文件存在：
```bash
ls -la LLM-models-datasets/Bespoke-Stratos-17k/bespokelabs___bespoke-stratos-17k/default-45716a10dbb21a2b/0.0.0/master/bespoke-stratos-17k-train.arrow
```

## 快速开始

### 1. 一键启动训练
```bash
bash train_bespoke_stratos.sh
```

### 2. 手动启动训练
```bash
python -m llamafactory.train examples/train_lora/bespoke_stratos_lora_sft.yaml
```

### 3. 测试微调后的模型
```bash
# 测试微调后的模型
python test_bespoke_model.py

# 测试原始模型（对比）
python test_bespoke_model.py --no_lora
```

## 配置说明

### 数据集配置 (data/dataset_info.json)
```json
"bespoke_stratos_17k": {
  "file_name": "../LLM-models-datasets/Bespoke-Stratos-17k/bespokelabs___bespoke-stratos-17k/default-45716a10dbb21a2b/0.0.0/master/bespoke-stratos-17k-train.arrow",
  "formatting": "sharegpt",
  "columns": {
    "messages": "conversations",
    "system": "system"
  },
  "tags": {
    "role_tag": "from",
    "content_tag": "value",
    "user_tag": "human",
    "assistant_tag": "gpt"
  }
}
```

### 训练配置关键参数

| 参数 | 值 | 说明 |
|------|----|----|
| model_name_or_path | meta-llama/Meta-Llama-3-8B-Instruct | 基础模型 |
| lora_rank | 8 | LoRA 秩，控制适配器大小 |
| lora_target | all | 应用 LoRA 的层 |
| cutoff_len | 4096 | 最大序列长度 |
| learning_rate | 5.0e-5 | 学习率 |
| num_train_epochs | 3.0 | 训练轮数 |
| per_device_train_batch_size | 2 | 批大小 |

## 高级配置

### 1. 模型选择
支持多种模型，修改 `model_name_or_path` 即可：
- `meta-llama/Meta-Llama-3-8B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`

### 2. 显存优化
#### 小显存配置（<12GB）
```yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
fp16: true  # 使用 fp16 而非 bf16
```

#### 量化训练
```yaml
quantization_bit: 4
quantization_type: nf4
```

### 3. 数据采样
#### 快速测试（1000样本）
```yaml
max_samples: 1000
num_train_epochs: 1.0
```

#### 完整训练（全部数据）
```yaml
max_samples: 16710
num_train_epochs: 3.0
```

## 监控与评估

### 1. 训练日志系统
我们已经配置了完整的日志系统，所有训练过程都会记录到指定路径：

#### 📋 日志文件结构
```
logs/qwen2.5-3b-bespoke-stratos/lora/sft/
├── training_YYYYMMDD_HHMMSS.log  # 详细训练日志
├── error_YYYYMMDD_HHMMSS.log     # 错误日志
└── tensorboard事件文件            # TensorBoard可视化数据
```

#### 🔍 实时监控训练
```bash
# 启动训练监控工具
bash monitor_training.sh

# 选项包括：
# 1) 实时显示最新日志
# 2) 查看完整日志内容  
# 3) 查看训练进度和损失
# 4) 查看GPU使用情况
# 5) 启动TensorBoard
# 6) 查看错误日志
# 7) 显示训练统计信息
```

#### 📊 TensorBoard可视化
```bash
# 启动TensorBoard
bash start_tensorboard.sh

# 访问地址: http://localhost:6006
# 远程服务器请使用端口转发:
# ssh -L 6006:localhost:6006 username@server_ip
```

#### 📈 日志分析和报告
```bash
# 分析训练日志，生成统计报告和可视化图表
python analyze_training_logs.py

# 指定特定日志文件
python analyze_training_logs.py --log-file logs/xxx/training_xxx.log

# 自定义输出目录
python analyze_training_logs.py --output-dir my_analysis
```

### 2. 训练日志功能

#### ✅ 自动记录的信息
- **系统信息**: 操作系统、Python版本、CUDA设备
- **GPU状态**: 显存使用情况、GPU型号
- **配置信息**: 完整的训练配置文件内容
- **训练进度**: 步数、轮次、损失值、学习率
- **时间戳**: 训练开始/结束时间
- **错误信息**: 详细的错误堆栈信息

#### 📋 日志内容示例
```
==========================================
训练开始时间: 2024-12-09 20:49:03
==========================================
系统信息:
- 操作系统: Linux DGX-Station 6.8.0-60-generic
- Python版本: Python 3.11.0
- CUDA设备: 0,1,2,3
- GPU信息:
NVIDIA RTX 4090, 24564, 1024, 23540
==========================================
配置文件内容:
[配置文件完整内容]
==========================================
开始训练...
训练命令: llamafactory-cli train examples/train_lora/bespoke_stratos_lora_sft.yaml
----------------------------------------
[训练过程详细日志]
----------------------------------------
训练结束时间: 2024-12-09 23:45:12
训练退出代码: 0
✅ 训练成功完成！
📁 模型保存在: saves/qwen2.5-3b-bespoke-stratos/lora/sft/
📊 训练日志保存在: logs/xxx/training_xxx.log
📈 TensorBoard日志保存在: logs/qwen2.5-3b-bespoke-stratos/lora/sft/
```

### 3. 验证评估
启用验证集评估：
```yaml
val_size: 0.1
eval_strategy: steps
eval_steps: 500
```

### 4. 外部监控工具
支持多种监控工具：
```yaml
report_to: tensorboard  # 选项: tensorboard, wandb, mlflow
logging_dir: logs/qwen2.5-3b-bespoke-stratos/lora/sft
```

## 推理使用

### 1. API 服务
```bash
python -m llamafactory.api \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --adapter_name_or_path saves/bespoke-stratos/lora/sft \
    --template llama3
```

### 2. 命令行聊天
```bash
python -m llamafactory.chat \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --adapter_name_or_path saves/bespoke-stratos/lora/sft \
    --template llama3
```

### 3. Web UI
```bash
python -m llamafactory.webui
```

## 故障排除

### 常见问题

1. **数据集路径错误**
   ```
   确保路径正确: ../LLM-models-datasets/Bespoke-Stratos-17k/...
   ```

2. **显存不足**
   ```yaml
   # 减小批大小和序列长度
   per_device_train_batch_size: 1
   cutoff_len: 2048
   ```

3. **训练不收敛**
   ```yaml
   # 调整学习率
   learning_rate: 2.0e-5  # 更小的学习率
   ```

4. **模板不匹配**
   ```yaml
   # 根据模型选择正确模板
   template: llama3  # 或 qwen, mistral 等
   ```

### 日志分析
查看训练日志：
```bash
tail -f saves/bespoke-stratos/lora/sft/trainer_log.jsonl
```

## 性能优化建议

### 1. 硬件配置
- 推荐: RTX 4090 / A100 (24GB+)
- 最低: RTX 3090 / V100 (12GB+)

### 2. 训练技巧
- 使用 DeepSpeed ZeRO-2 提升效率
- 启用梯度检查点节省显存
- 合理设置数据并行

### 3. 超参数调优
- LoRA rank: 8-32 (平衡效果与效率)
- 学习率: 1e-5 到 1e-4
- Warmup ratio: 0.05-0.1

## 结论

Bespoke-Stratos-17k 数据集完全兼容 LLaMA-Factory，无需修改数据格式。通过合理的配置，可以高效地进行 LoRA 微调，提升模型的对话能力和指令遵循性能。

建议从小样本开始测试，确认配置无误后再进行完整训练。 