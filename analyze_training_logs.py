#!/usr/bin/env python3
"""
训练日志分析脚本
功能：分析训练日志，提取关键信息，生成统计报告和可视化图表

用法：python analyze_training_logs.py
"""

import os
import re
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

def setup_chinese_fonts():
    """设置中文字体支持"""
    try:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("Warning: Cannot set Chinese fonts, using default fonts")

def find_latest_log(log_dir: str) -> str:
    """查找最新的训练日志文件"""
    log_dir = Path(log_dir)
    if not log_dir.exists():
        raise FileNotFoundError(f"Log directory not found: {log_dir}")
    
    log_files = list(log_dir.glob("training_*.log"))
    if not log_files:
        raise FileNotFoundError(f"No training log files found in {log_dir}")
    
    # 按修改时间排序，返回最新的
    latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
    return str(latest_log)

def parse_training_log(log_file: str) -> dict:
    """解析训练日志文件"""
    print(f"Parsing log file: {log_file}")
    
    data = {
        'training_steps': [],
        'train_losses': [],
        'eval_losses': [],
        'learning_rates': [],
        'timestamps': [],
        'epochs': [],
        'gpu_info': [],
        'system_info': {},
        'config_info': {},
        'training_status': 'unknown'
    }
    
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取系统信息
    system_match = re.search(r'- Operating System: (.+)', content)
    if system_match:
        data['system_info']['os'] = system_match.group(1)
    
    python_match = re.search(r'- Python Version: (.+)', content)
    if python_match:
        data['system_info']['python'] = python_match.group(1)
    
    cuda_match = re.search(r'- CUDA Devices: (.+)', content)
    if cuda_match:
        data['system_info']['cuda_devices'] = cuda_match.group(1)
    
    # 提取训练状态
    if 'Training completed successfully!' in content or '✅ 训练成功完成' in content:
        data['training_status'] = 'completed'
    elif 'Training failed' in content or '❌ 训练失败' in content:
        data['training_status'] = 'failed'
    elif 'Training started' in content or '开始训练' in content:
        data['training_status'] = 'running'
    
    # 提取训练指标 (JSON格式的日志行)
    json_pattern = r'\{[^{}]*"train_loss"[^{}]*\}'
    json_matches = re.findall(json_pattern, content)
    
    for match in json_matches:
        try:
            log_data = json.loads(match)
            if 'train_loss' in log_data:
                data['training_steps'].append(log_data.get('step', 0))
                data['train_losses'].append(log_data.get('train_loss', 0))
                data['learning_rates'].append(log_data.get('learning_rate', 0))
                data['epochs'].append(log_data.get('epoch', 0))
                
                # 尝试解析时间戳
                if 'timestamp' in log_data:
                    try:
                        timestamp = datetime.fromisoformat(log_data['timestamp'])
                        data['timestamps'].append(timestamp)
                    except:
                        data['timestamps'].append(datetime.now())
                else:
                    data['timestamps'].append(datetime.now())
        except json.JSONDecodeError:
            continue
    
    # 提取评估损失
    eval_pattern = r'"eval_loss":\s*([\d.]+)'
    eval_matches = re.findall(eval_pattern, content)
    data['eval_losses'] = [float(loss) for loss in eval_matches]
    
    return data

def generate_training_report(data: dict, output_dir: str):
    """生成训练报告"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成文本报告
    report_file = output_dir / 'training_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("Bespoke-Stratos-17k Training Analysis Report\n")
        f.write("="*60 + "\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 系统信息
        f.write("System Information:\n")
        f.write("-" * 30 + "\n")
        for key, value in data['system_info'].items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        # 训练状态
        f.write(f"Training Status: {data['training_status']}\n")
        f.write(f"Total Training Steps: {len(data['training_steps'])}\n")
        f.write(f"Total Epochs: {max(data['epochs']) if data['epochs'] else 0:.2f}\n\n")
        
        # 损失统计
        if data['train_losses']:
            f.write("Training Loss Statistics:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Initial Loss: {data['train_losses'][0]:.4f}\n")
            f.write(f"Final Loss: {data['train_losses'][-1]:.4f}\n")
            f.write(f"Best Loss: {min(data['train_losses']):.4f}\n")
            f.write(f"Loss Reduction: {(data['train_losses'][0] - data['train_losses'][-1]):.4f}\n")
            f.write(f"Loss Reduction %: {((data['train_losses'][0] - data['train_losses'][-1]) / data['train_losses'][0] * 100):.2f}%\n\n")
        
        if data['eval_losses']:
            f.write("Evaluation Loss Statistics:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Best Eval Loss: {min(data['eval_losses']):.4f}\n")
            f.write(f"Latest Eval Loss: {data['eval_losses'][-1]:.4f}\n\n")
        
        # 学习率统计
        if data['learning_rates']:
            f.write("Learning Rate Statistics:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Initial LR: {data['learning_rates'][0]:.2e}\n")
            f.write(f"Final LR: {data['learning_rates'][-1]:.2e}\n")
            f.write(f"Max LR: {max(data['learning_rates']):.2e}\n")
            f.write(f"Min LR: {min(data['learning_rates']):.2e}\n")
    
    print(f"Training report saved to: {report_file}")

def create_visualizations(data: dict, output_dir: str):
    """创建可视化图表"""
    setup_chinese_fonts()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not data['training_steps']:
        print("No training data found for visualization")
        return
    
    # 创建训练损失图
    plt.figure(figsize=(12, 8))
    
    # 子图1: 训练损失
    plt.subplot(2, 2, 1)
    plt.plot(data['training_steps'], data['train_losses'], 'b-', linewidth=2, label='Training Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 子图2: 学习率
    if data['learning_rates']:
        plt.subplot(2, 2, 2)
        plt.plot(data['training_steps'], data['learning_rates'], 'r-', linewidth=2, label='Learning Rate')
        plt.xlabel('Steps')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.yscale('log')
    
    # 子图3: 训练和验证损失对比
    if data['eval_losses']:
        plt.subplot(2, 2, 3)
        plt.plot(data['training_steps'], data['train_losses'], 'b-', linewidth=2, label='Train Loss')
        # 假设评估损失按相同间隔记录
        eval_steps = np.linspace(0, max(data['training_steps']), len(data['eval_losses']))
        plt.plot(eval_steps, data['eval_losses'], 'g-', linewidth=2, label='Eval Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Training vs Evaluation Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    # 子图4: Epoch进度
    if data['epochs']:
        plt.subplot(2, 2, 4)
        plt.plot(data['training_steps'], data['epochs'], 'purple', linewidth=2, label='Epoch Progress')
        plt.xlabel('Steps')
        plt.ylabel('Epoch')
        plt.title('Training Epoch Progress')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    
    # 保存图表
    plot_file = output_dir / 'training_plots.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training plots saved to: {plot_file}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='分析训练日志')
    parser.add_argument('--log-dir', default='logs/qwen2.5-3b-bespoke-stratos/lora/sft',
                        help='日志目录路径')
    parser.add_argument('--output-dir', default='logs/analysis',
                        help='分析结果输出目录')
    parser.add_argument('--log-file', help='指定具体的日志文件')
    
    args = parser.parse_args()
    
    try:
        # 查找或使用指定的日志文件
        if args.log_file:
            log_file = args.log_file
        else:
            log_file = find_latest_log(args.log_dir)
        
        print(f"Analyzing log file: {log_file}")
        
        # 解析日志
        data = parse_training_log(log_file)
        
        # 生成报告
        generate_training_report(data, args.output_dir)
        
        # 创建可视化
        create_visualizations(data, args.output_dir)
        
        print(f"\nAnalysis completed! Results saved to: {args.output_dir}")
        print(f"Training status: {data['training_status']}")
        if data['train_losses']:
            print(f"Training progress: {len(data['training_steps'])} steps, Latest loss: {data['train_losses'][-1]:.4f}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main() 