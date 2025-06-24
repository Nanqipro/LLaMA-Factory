import transformers
import torch

# 切换为你下载的模型文件目录, 这里的demo是Qwen2.5-3B-Instruct
# 修正模型路径为实际的模型文件所在目录
model_id = "/home/nanchang/ZJ/LLM-models-datasets/Qwen2.5-3B-Instruct/qwen/Qwen2___5-3B-Instruct"

print("开始加载模型...")
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)
print("模型加载完成！")

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

print("构建提示词...")
prompt = pipeline.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
)

# 简化终止符设置，避免可能的兼容性问题
terminators = [
    pipeline.tokenizer.eos_token_id,
]

print("开始生成回复...")
outputs = pipeline(
    prompt,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
print("生成完成！")
print("回复内容:")
print(outputs[0]["generated_text"][len(prompt):])