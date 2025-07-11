from datasets import load_dataset, DatasetDict, Dataset, IterableDatasetDict, IterableDataset
import os
from typing import Optional

def download_dataset_from_huggingface(dataset_name: str, local_dir: str, split: Optional[str] = None) -> bool:
    """
    从 Hugging Face 下载数据集

    Parameters
    ----------
    dataset_name : str
        数据集名称，格式为 "组织名/数据集名" 或 "数据集名"
    local_dir : str
        本地存储目录路径
    split : Optional[str], default=None
        数据集分割，如 'train', 'test', 'validation'，None表示下载所有分割

    Returns
    -------
    bool
        下载是否成功
    """
    try:
        print(f"从 Hugging Face 下载数据集 {dataset_name} 到 {local_dir}...")
        
        # 确保本地目录存在
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)
            print(f"创建目录: {local_dir}")

        # 使用 Hugging Face datasets 库下载数据集
        # 通过 streaming=True 来判断是否只能流式加载
        try:
            dataset = load_dataset(dataset_name, cache_dir=local_dir, split=split)
        except Exception:
            print("标准加载失败，尝试流式加载...")
            dataset = load_dataset(dataset_name, cache_dir=local_dir, split=split, streaming=True)

        # 处理 DatasetDict 和 IterableDatasetDict (当 split is None)
        if isinstance(dataset, (DatasetDict, IterableDatasetDict)):
            for split_name, ds in dataset.items():
                if isinstance(ds, Dataset):
                    save_path = os.path.join(local_dir, f"{dataset_name.replace('/', '_')}_{split_name}")
                    ds.save_to_disk(save_path)
                    print(f"数据集分割 '{split_name}' 已保存到 {save_path}")
                elif isinstance(ds, IterableDataset):
                    print(f"检测到流式数据集分割 '{split_name}'，将保存为 JSONL 文件。")
                    json_save_path = os.path.join(local_dir, f"{dataset_name.replace('/', '_')}_{split_name}.jsonl")
                    # 流式写入json
                    with open(json_save_path, "w", encoding="utf-8") as f:
                        for item in ds:
                            f.write(str(item) + "\n")
                    print(f"流式数据集分割 '{split_name}' 已保存到 {json_save_path}")

        # 处理单个 Dataset 和 IterableDataset
        elif isinstance(dataset, Dataset):
            save_path = os.path.join(local_dir, dataset_name.replace('/', '_'))
            dataset.save_to_disk(save_path)
            print(f"数据集下载成功到 {save_path}")
        elif isinstance(dataset, IterableDataset):
            print("检测到流式数据集，将保存为 JSONL 文件。")
            json_save_path = os.path.join(local_dir, f"{dataset_name.replace('/', '_')}.jsonl")
            # 流式写入json
            with open(json_save_path, "w", encoding="utf-8") as f:
                for item in dataset:
                    f.write(str(item) + "\n")
            print(f"流式数据集已保存到 {json_save_path}")

        return True

    except Exception as e:
        print(f"下载数据集时出错: {e}")
        return False

def main() -> None:
    """
    主函数：执行数据集下载流程
    """
    # Hugging Face 上的数据集名称示例
    dataset_name = "squad"  # Stanford Question Answering Dataset
    # 备选数据集：dataset_name = "glue", "imdb", "wikitext"
    
    # 定义本地存储路径
    base_dir = "../LLM-models-datasets"
    dataset_root_dir = os.path.join(base_dir, "datasets")
    local_dataset_dir = os.path.join(dataset_root_dir, dataset_name.replace('/', '_'))
    
    # 确保基础目录存在
    if not os.path.exists(dataset_root_dir):
        os.makedirs(dataset_root_dir)
        print(f"创建目录: {dataset_root_dir}")
    
    # 执行下载 - 可以指定特定分割或下载全部
    success = download_dataset_from_huggingface(
        dataset_name=dataset_name,
        local_dir=local_dataset_dir,
        split=None  # None表示下载所有分割，也可以指定 'train', 'test' 等
    )
    
    if success:
        print("✅ 数据集下载完成！")
        print(f"数据集位置: {local_dataset_dir}")
    else:
        print("❌ 数据集下载失败，请检查网络连接和数据集名称")

if __name__ == "__main__":
    main()
