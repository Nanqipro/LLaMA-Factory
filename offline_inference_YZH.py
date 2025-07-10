# import os
# # 在导入任何PyTorch相关模块之前设置环境变量
# os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"  # 禁用transformers的在线功能
# os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"  # 明确禁用torchvision检查

# import json
# import sys
# from unittest.mock import MagicMock

# # 创建更完整的torchvision模拟
# class TorchVisionMock(MagicMock):
#     def __getattr__(self, name):
#         if name == "__spec__":
#             return MagicMock()
#         return super().__getattr__(name)
    
#     def __bool__(self):
#         return False

# # 模拟缺失的模块
# sys.modules['torchvision'] = TorchVisionMock()
# sys.modules['torchvision.transforms'] = TorchVisionMock()
# sys.modules['torchvision.models'] = TorchVisionMock()

# # 在导入vllm之前设置环境变量
# os.environ["VLLM_NO_TORCHVISION"] = "1"

# from vllm import LLM, SamplingParams
# from transformers import AutoTokenizer

# class Infer:
#     def __init__(self, model="/home/ma-user/work/X-R1-3B"):
#         if not os.path.exists(model):
#             print(f"Error: Model path '{model}' does not exist.")
#             exit(1)
        
#         # 初始化tokenizer - 简化版本
#         try:
#             print("Initializing tokenizer...")
#             self.tokenizer = AutoTokenizer.from_pretrained(
#                 model, 
#                 trust_remote_code=True,
#                 use_fast=True,
#                 padding_side="left"  # 添加填充方向
#             )
#             print("Tokenizer initialized successfully.")
#         except Exception as e:
#             print(f"Tokenizer initialization failed: {e}")
#             # 尝试回退到简单tokenizer
#             try:
#                 print("Trying fallback tokenizer...")
#                 self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
#                 print("Fallback tokenizer initialized successfully.")
#             except Exception as fallback_e:
#                 print(f"Fallback tokenizer failed: {fallback_e}")
#                 exit(1)
        
#         # 配置引擎参数 - 简化版本
#         try:
#             print("Initializing LLM engine...")
#             self.llm = LLM(
#                 model=model,
#                 tokenizer=model,
#                 tokenizer_mode="auto",
#                 trust_remote_code=True,
#                 dtype="bfloat16",
#                 tensor_parallel_size=1,
#                 max_model_len=32768,
#                 device="npu",
#                 enforce_eager=True,
#                 disable_custom_all_reduce=True  # 禁用自定义all_reduce
#             )
#             print("LLM engine initialized successfully.")
#         except Exception as e:
#             print(f"LLM initialization failed: {e}")
#             # 尝试回退到基本配置
#             try:
#                 print("Trying fallback LLM configuration...")
#                 self.llm = LLM(model=model, device="npu", enforce_eager=True)
#                 print("Fallback LLM initialized successfully.")
#             except Exception as fallback_e:
#                 print(f"Fallback LLM failed: {fallback_e}")
#                 exit(1)
        
#         # 简化的采样参数
#         self.sampling_params = {
#             "choice": SamplingParams(max_tokens=1024, temperature=0.8, top_p=0.95),
#             "code-generate": SamplingParams(max_tokens=2048, temperature=0.8, top_p=0.95),
#             "generic-generate": SamplingParams(max_tokens=128, temperature=0.8, top_p=0.95),
#             "math": SamplingParams(max_tokens=512, temperature=0.8, top_p=0.95)
#         }
        
#         # 简化的提示模板
#         self.prompt_templates = {
#             "choice": "Question: {Question}\nA) {A}\nB) {B}\nC) {C}\nD) {D}\nAnswer:",
#             "code-generate": "Implement this function:\n{prompt}",
#             "generic-generate": "Answer this question:\n{prompt}\nAnswer:",
#             "math": "Solve this math problem:\n{Question}\nAnswer:"
#         }

#     def infer(self, data_file="A-data.jsonl"):
#         a_datas = []
#         try:
#             with open(data_file, 'r', encoding="utf-8") as f:
#                 for line in f:
#                     a_data = json.loads(line)
#                     a_datas.append(a_data)
#         except FileNotFoundError:
#             print(f"Error: Data file '{data_file}' not found.")
#             return {"error": f"File not found: {data_file}"}
#         except json.JSONDecodeError as e:
#             print(f"Error: Invalid JSON in '{data_file}'. Details: {e}")
#             return {"error": f"Invalid JSON in {data_file}: {e}"}

#         res = {"result": {"results": []}}

#         for i, a_data in enumerate(a_datas):
#             type_ = a_data.get("type")
#             id_ = a_data.get("id")
#             print(f"\n--- Processing entry {i+1} (ID: {id_}, Type: {type_}) ---")

#             if type_ not in self.prompt_templates:
#                 print(f"WARNING: Unknown type '{type_}' for ID '{id_}'. Skipping.")
#                 continue

#             template = self.prompt_templates[type_]
#             prompt = ""
            
#             try:
#                 if type_ == "choice":
#                     choices = a_data["choices"]
#                     prompt = template.format(
#                         Question=a_data["prompt"],
#                         A=choices.get("A", ""),
#                         B=choices.get("B", ""),
#                         C=choices.get("C", ""),
#                         D=choices.get("D", "")
#                     )
#                 elif type_ == "math":
#                     prompt = template.format(Question=a_data["prompt"])
#                 else:
#                     prompt = template.format(prompt=a_data["prompt"])

#                 print(f"DEBUG: Generated prompt for ID '{id_}':\n{prompt}\n")
                
#                 generated_text = []
#                 if not prompt.strip():
#                     print(f"WARNING: Empty prompt for ID '{id_}'. Skipping.")
#                     generated_text = "ERROR: Empty prompt"
#                 else:
#                     # 使用LLM生成文本
#                     outputs = self.llm.generate(prompts=[prompt], 
#                                                sampling_params=self.sampling_params[type_])
                    
#                     generated_text = []
#                     for output in outputs:
#                         for choice in output.outputs:
#                             generated_text.append(choice.text)
                    
#                     generated_text = generated_text[0] if len(generated_text) == 1 else generated_text
                
#                 res["result"]["results"].append({"id": id_, "content": generated_text})
            
#             except Exception as e:
#                 print(f"ERROR: {e}")
#                 res["result"]["results"].append({"id": id_, "content": f"ERROR: {e}"})
        
#         return res

# if __name__ == "__main__":
#     data_file_name = "A-data.jsonl"
#     infer = Infer(model="/home/ma-user/work/X-R1-3B")
#     res = infer.infer(data_file=data_file_name)
    
#     output_file_name = "res.json"
#     with open(output_file_name, "w", encoding="utf-8") as f:
#         json.dump(res, f, ensure_ascii=False, indent=2)

#     print(f"\n--- Inference complete. Results saved to '{output_file_name}' ---")

import json
from vllm import LLM, SamplingParams
import os

class Infer:
    def __init__(self, model="./LLM-models-datasets/Qwen2.5-3B"):
        if not os.path.exists(model):
            print(f"Error: Model path '{model}' does not exist.")
            exit(1)
        
        self.llm = LLM(
    model=model,
    trust_remote_code=True,
    tokenizer_mode="slow",  # 添加这行
    # max_model_len=8192  # 增加此行以限制最大序列长度，减少显存占用
)
        self.sampling_params = {
            "choice": SamplingParams(max_tokens=4096, temperature=0.8, top_p=0.95),
            "code-generate": SamplingParams(n=3, max_tokens=4096, temperature=0.8, top_p=0.95),
            "generic-generate": SamplingParams(max_tokens=4096, temperature=0.8, top_p=0.95),
            "math": SamplingParams(max_tokens=4096, temperature=0.8, top_p=0.95)
        }
        
        # 修复数学模板中的大括号转义问题
        self.prompt_templates = {
            "choice": "Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.\n\n{Question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}",
            "code-generate": "Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.\n",
            "generic-generate": "You will be asked to read a passage and answer a question. Think step by step, then write a line of the form 'Answer: $ANSWER' at the end of your response.",
            "math": "Solve the following math problem step by step. The last line of your response should be of the form Answer: \\$ANSWER (without quotes) where $ANSWER is the answer to the problem.\n\n{Question}\n\nRemember to put your answer on its own line after 'Answer:', and indicate your final answer in boxed LaTeX. For example, if the final answer is \\sqrt{{3}}, write it as \\boxed{{\\sqrt{{3}}}}."
        }
        
    def escape_braces(self, text):
        """转义文本中的所有大括号，防止format解析错误"""
        return text.replace("{", "{{").replace("}", "}}")

    def infer(self, data_file="A-data.jsonl"):
        a_datas = []
        try:
            with open(data_file, 'r', encoding="utf-8") as f:
                for line in f:
                    a_data = json.loads(line)
                    a_datas.append(a_data)
        except FileNotFoundError:
            print(f"Error: Data file '{data_file}' not found.")
            return {"error": f"File not found: {data_file}"}
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in '{data_file}'. Details: {e}")
            return {"error": f"Invalid JSON in {data_file}: {e}"}

        res = {"result": {"results": []}}

        for i, a_data in enumerate(a_datas):
            type_ = a_data.get("type")
            id_ = a_data.get("id")
            print(f"\n--- Processing entry {i+1} (ID: {id_}, Type: {type_}) ---")

            if type_ not in self.prompt_templates:
                print(f"WARNING: Unknown type '{type_}' for ID '{id_}'. Skipping.")
                continue

            template = self.prompt_templates[type_]
            prompt = ""
            
            try:
                # 对输入内容进行大括号转义
                if type_ == "choice":
                    choices = a_data["choices"]
                    escaped_choices = {k: self.escape_braces(v) for k, v in choices.items()}
                    prompt = template.format(
                        Question=self.escape_braces(a_data["prompt"]),
                        A=escaped_choices.get("A", ""),
                        B=escaped_choices.get("B", ""),
                        C=escaped_choices.get("C", ""),
                        D=escaped_choices.get("D", "")
                    )
                else:
                    # 对所有其他类型的内容进行转义
                    escaped_prompt = self.escape_braces(a_data["prompt"])
                    if type_ == "math":
                        prompt = template.format(Question=escaped_prompt)
                    else:  # code-generate 和 generic-generate
                        prompt = template + escaped_prompt

                print(f"DEBUG: Generated prompt for ID '{id_}':\n{prompt}\n")
                
                generated_text = []
                if not prompt.strip():
                    print(f"WARNING: Empty prompt for ID '{id_}'. Skipping.")
                    generated_text = "ERROR: Empty prompt"
                else:
                    outputs = self.llm.generate(prompt, self.sampling_params[type_])
                    for output in outputs:
                        for o in output.outputs:
                            generated_text.append(o.text)
                    generated_text = generated_text[0] if len(generated_text) == 1 else generated_text

                res["result"]["results"].append({"id": id_, "content": generated_text})

            except KeyError as ke:
                print(f"ERROR: Missing key in data for ID '{id_}': {ke}")
                res["result"]["results"].append({"id": id_, "content": f"ERROR: Missing data - {ke}"})
            except Exception as e:
                print(f"ERROR: Unexpected error for ID '{id_}': {e}")
                res["result"]["results"].append({"id": id_, "content": f"ERROR: {e}"})

        return res

if __name__ == "__main__":
    data_file_name = "./A-data/A-data.jsonl"
    infer = Infer(model="./LLM-models-datasets/Qwen2.5-3B")
    res = infer.infer(data_file=data_file_name)
    
    output_file_name = "./A-data/res.json"
    with open(output_file_name, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    print(f"\n--- Inference complete. Results saved to '{output_file_name}' ---")