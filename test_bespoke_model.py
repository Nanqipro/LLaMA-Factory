#!/usr/bin/env python3
"""
Bespoke-Stratos-17k 微调模型测试脚本
用法：python test_bespoke_model.py
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import argparse

def test_model(model_path, lora_path=None, test_prompts=None):
    """
    测试微调后的模型
    
    Args:
        model_path: 基础模型路径
        lora_path: LoRA适配器路径
        test_prompts: 测试提示词列表
    """
    
    # 默认测试提示词
    if test_prompts is None:
        test_prompts = [
            "解释一下什么是人工智能？",
            "How does machine learning work?",
            "请用简单的话解释深度学习。",
            "What are the benefits of renewable energy?",
            "编写一个Python函数来计算斐波那契数列。"
        ]
    
    print(f"正在加载基础模型: {model_path}")
    
    # 配置量化（如果需要）
    # quantization_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_compute_dtype=torch.float16,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4"
    # )
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        # quantization_config=quantization_config,  # 如果需要量化
    )
    
    # 如果提供了LoRA路径，加载LoRA适配器
    if lora_path:
        print(f"正在加载LoRA适配器: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()  # 合并LoRA权重
    
    model.eval()
    
    print("\n开始测试模型...")
    print("=" * 50)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n测试 {i}: {prompt}")
        print("-" * 30)
        
        # 格式化输入
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # 应用聊天模板
        if hasattr(tokenizer, 'apply_chat_template'):
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        # 编码
        inputs = tokenizer.encode(formatted_prompt, return_tensors="pt").to(model.device)
        
        # 生成
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # 解码响应
        response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        print(f"回答: {response.strip()}")
        print("=" * 50)

def main():
    parser = argparse.ArgumentParser(description="测试 Bespoke-Stratos-17k 微调模型")
    parser.add_argument("--base_model", default="meta-llama/Meta-Llama-3-8B-Instruct", 
                       help="基础模型路径")
    parser.add_argument("--lora_path", default="saves/bespoke-stratos/lora/sft",
                       help="LoRA适配器路径")
    parser.add_argument("--no_lora", action="store_true", help="不使用LoRA适配器")
    
    args = parser.parse_args()
    
    lora_path = None if args.no_lora else args.lora_path
    
    test_model(args.base_model, lora_path)

if __name__ == "__main__":
    main() 