import os
import asyncio
from typing import Optional, Dict, Any
from datasets import load_dataset, DownloadMode
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatasetDownloader:
    """
    Hugging Face数据集下载器
    
    提供从Hugging Face平台下载各种类型数据集的功能
    """
    
    def __init__(self, base_dir: str = "../LLM-models-datasets") -> None:
        """
        初始化数据集下载器
        
        Parameters
        ----------
        base_dir : str
            数据集存储的基础目录路径
        """
        self.base_dir = base_dir
        self._ensure_base_dir_exists()
        
    def _ensure_base_dir_exists(self) -> None:
        """
        确保基础目录存在，不存在则创建
        """
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir, exist_ok=True)
            logger.info(f"创建基础目录: {self.base_dir}")
    
    async def download_dataset_async(
        self, 
        dataset_name: str, 
        config_name: Optional[str] = None,
        split: Optional[str] = None,
        download_mode: str = "force_redownload",
        use_auth_token: Optional[str] = None
    ) -> bool:
        """
        异步下载指定数据集
        
        Parameters
        ----------
        dataset_name : str
            数据集名称，格式为 "组织名/数据集名" 或 "数据集名"
        config_name : Optional[str]
            配置名称（子数据集）（可选）
        split : Optional[str]
            数据集分割类型，如 "train", "test", "validation" 等
        download_mode : str
            下载模式，默认为 "force_redownload"
        use_auth_token : Optional[str]
            认证令牌，用于访问私有数据集
            
        Returns
        -------
        bool
            下载是否成功
        """
        try:
            logger.info(f"开始异步下载数据集: {dataset_name}")
            
            # 构建本地存储路径
            dataset_local_name = dataset_name.replace('/', '_')
            if config_name:
                dataset_local_name += f"_{config_name}"
            
            local_path = os.path.join(self.base_dir, dataset_local_name)
            
            # 从环境变量获取认证令牌（如果未提供）
            if use_auth_token is None:
                use_auth_token = os.getenv('HF_TOKEN')
            
            # 确定下载模式
            if download_mode == "force_redownload":
                dl_mode = DownloadMode.FORCE_REDOWNLOAD
            elif download_mode == "reuse_cache_if_exists":
                dl_mode = DownloadMode.REUSE_CACHE_IF_EXISTS
            else:
                dl_mode = DownloadMode.REUSE_DATASET_IF_EXISTS
            
            # 使用Hugging Face数据集API下载到指定目录
            dataset = load_dataset(
                dataset_name,
                name=config_name,
                split=split,
                cache_dir=local_path,
                download_mode=dl_mode,
                token=use_auth_token
            )
            
            logger.info(f"数据集已下载到: {local_path}")
            try:
                # 尝试获取数据集大小，使用 type: ignore 来避免类型检查错误
                size = len(dataset)  # type: ignore
                logger.info(f"数据集大小: {size} 条记录")
            except (TypeError, AttributeError):
                logger.info("数据集为可迭代类型，无法直接获取大小")
            logger.info(f"✅ 数据集 {dataset_name} 下载成功")
            return True
            
        except Exception as e:
            logger.error(f"❌ 下载数据集 {dataset_name} 时出错: {e}")
            return False
    
    def download_dataset_sync(
        self, 
        dataset_name: str, 
        config_name: Optional[str] = None,
        split: Optional[str] = None,
        use_auth_token: Optional[str] = None
    ) -> bool:
        """
        同步下载指定数据集
        
        Parameters
        ----------
        dataset_name : str
            数据集名称，格式为 "组织名/数据集名" 或 "数据集名"
        config_name : Optional[str]
            配置名称（子数据集）（可选）
        split : Optional[str]
            数据集分割类型，如 "train", "test", "validation" 等
        use_auth_token : Optional[str]
            认证令牌，用于访问私有数据集
            
        Returns
        -------
        bool
            下载是否成功
        """
        try:
            logger.info(f"开始同步下载数据集: {dataset_name}")
            
            # 构建本地存储路径
            dataset_local_name = dataset_name.replace('/', '_')
            if config_name:
                dataset_local_name += f"_{config_name}"
            
            local_path = os.path.join(self.base_dir, dataset_local_name)
            
            # 从环境变量获取认证令牌（如果未提供）
            if use_auth_token is None:
                use_auth_token = os.getenv('HF_TOKEN')
            
            # 使用Hugging Face数据集API下载到指定目录
            dataset = load_dataset(
                dataset_name,
                name=config_name,
                split=split,
                cache_dir=local_path,
                token=use_auth_token
            )
            
            logger.info(f"数据集已下载到: {local_path}")
            try:
                # 尝试获取数据集大小，使用 type: ignore 来避免类型检查错误
                size = len(dataset)  # type: ignore
                logger.info(f"数据集大小: {size} 条记录")
            except (TypeError, AttributeError):
                logger.info("数据集为可迭代类型，无法直接获取大小")
            logger.info(f"✅ 数据集 {dataset_name} 下载成功")
            return True
            
        except Exception as e:
            logger.error(f"❌ 下载数据集 {dataset_name} 时出错: {e}")
            return False
    
    async def download_multiple_datasets_async(
        self, 
        dataset_configs: list[Dict[str, Any]]
    ) -> Dict[str, bool]:
        """
        异步批量下载多个数据集
        
        Parameters
        ----------
        dataset_configs : list[Dict[str, Any]]
            数据集配置列表，每个配置包含数据集下载参数
            
        Returns
        -------
        Dict[str, bool]
            每个数据集的下载结果
        """
        results = {}
        tasks = []
        
        for config in dataset_configs:
            dataset_name = config.get('dataset_name')
            if dataset_name:
                task = self.download_dataset_async(**config)
                tasks.append((dataset_name, task))
        
        # 并发执行所有下载任务
        for dataset_name, task in tasks:
            try:
                result = await task
                results[dataset_name] = result
            except Exception as e:
                logger.error(f"批量下载中出错 {dataset_name}: {e}")
                results[dataset_name] = False
        
        return results


async def main() -> None:
    """
    主函数：演示数据集下载功能
    """
    # 初始化下载器
    downloader = DatasetDownloader()
    
    logger.info("=== Hugging Face 数据集下载工具 ===")
    
    # 单个数据集同步下载示例
    logger.info("\n单个数据集下载示例:")
    success = downloader.download_dataset_sync(
        dataset_name="NovaSky-AI/Sky-T1_data_17k",
        split="train"
    )
    
    if success:
        logger.info("✅ 数据集下载完成")
    else:
        logger.error("❌ 数据集下载失败")
    
    # # 异步批量下载示例
    # logger.info("\n批量异步下载示例:")
    # dataset_configs = [
    #     {
    #         "dataset_name": "glue",
    #         "config_name": "cola",
    #         "split": "train"
    #     }
    # ]
    
    # results = await downloader.download_multiple_datasets_async(dataset_configs)
    
    # logger.info("\n批量下载结果:")
    # for dataset_name, success in results.items():
    #     status = "✅ 成功" if success else "❌ 失败"
    #     logger.info(f"{dataset_name}: {status}")


if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())
