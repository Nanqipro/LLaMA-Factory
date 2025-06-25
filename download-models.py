"""
模型下载脚本 - 将模型下载到指定目录
"""
from modelscope import snapshot_download

def download_model_to_path() -> str:
    """
    下载LLaMA模型到指定路径
    
    Returns
    -------
    str
        模型下载后的本地路径
    """
    # 指定模型下载的目标路径
    target_path: str = '/home/nanchang/ZJ/LLM-models-datasets'
    
    # 下载模型到指定路径
    model_dir: str = snapshot_download(
        'LLM-Research/Meta-Llama-3-8B-Instruct',
        cache_dir=target_path
    )
    
    print(f"模型已下载到: {model_dir}")
    return model_dir

if __name__ == "__main__":
    download_model_to_path()