'''
Time: 2025.8.20
We will upload the comments for the code as soon as possible and update the document content so that you can better understand the code.
'''
import os
import json
import logging
import argparse
from pathlib import Path
from traffic_splitter import TrafficSplitter
from data_cleaner import DataCleaner

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PreprocessingPipeline:
    def __init__(self, dataset_config_path="config/dataset_config.json"):
        self.dataset_config = self._load_config(dataset_config_path)
        self.splitcap_path = self.dataset_config.get("splitcap_path", None)
        
        self.splitter = TrafficSplitter(self.splitcap_path)
        self.cleaner = DataCleaner(
            max_payload_length=self.dataset_config.get("max_payload_length", 1500),
            remove_empty=self.dataset_config.get("remove_empty", True)
        )
        
        self._create_directories()
    
    def _load_config(self, config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"成功加载配置文件: {config_path}")
            return config
        except Exception as e:
            logger.error(f"加载配置文件失败: {str(e)}")
            raise
    
    def _create_directories(self):

        for dataset in ['iscx', 'ustc']:
            if f"{dataset}_path" in self.dataset_config:
                Path(self.dataset_config.get(f"{dataset}_split_path", f"datasets/{dataset}/split")).mkdir(parents=True, exist_ok=True)
                Path(self.dataset_config.get(f"{dataset}_cleaned_path", f"datasets/{dataset}/cleaned")).mkdir(parents=True, exist_ok=True)
        
        logger.info("预处理目录准备就绪")
    
    def process_dataset(self, dataset_name):
        if dataset_name not in ['iscx', 'ustc']:
            logger.error(f"不支持的数据集: {dataset_name}，支持的数据集为'iscx'和'ustc'")
            return
        
        raw_path = self.dataset_config.get(f"{dataset_name}_path")
        split_path = self.dataset_config.get(f"{dataset_name}_split_path", f"datasets/{dataset_name}/split")
        cleaned_path = self.dataset_config.get(f"{dataset_name}_cleaned_path", f"datasets/{dataset_name}/cleaned")
        
        if not raw_path or not os.path.exists(raw_path):
            logger.error(f"数据集{dataset_name}的原始路径不存在: {raw_path}")
            return
        
        logger.info(f"开始处理数据集: {dataset_name}")
        logger.info(f"原始数据路径: {raw_path}")
        logger.info(f"分割后路径: {split_path}")
        logger.info(f"清洗后路径: {cleaned_path}")
        
        try:
            # 步骤1: 分割流量
            logger.info("===== 开始分割流量 =====")
            split_count = self.splitter.batch_split(raw_path, split_path, recursive=True)
            
            if split_count == 0:
                logger.warning("没有分割出任何文件，可能原始数据为空或格式不正确")
                return
            
            # 步骤2: 清洗和标准化数据
            logger.info("===== 开始清洗数据 =====")
            clean_count = self.cleaner.batch_clean(split_path, cleaned_path, recursive=True)
            
            logger.info(f"数据集{dataset_name}预处理完成，共生成{clean_count}个清洗后的文件")
            
        except Exception as e:
            logger.error(f"处理数据集{dataset_name}时出错: {str(e)}")
    
    def process_all_datasets(self):
        logger.info("开始处理所有数据集")
        self.process_dataset("iscx")
        self.process_dataset("ustc")
        logger.info("所有数据集处理完成")

def main():
    parser = argparse.ArgumentParser(description='FGFR-Net数据预处理流水线')
    parser.add_argument('--dataset', type=str, choices=['iscx', 'ustc', 'all'], 
                      default='all', help='要处理的数据集，默认为所有数据集')
    parser.add_argument('--config', type=str, default='config/dataset_config.json',
                      help='数据集配置文件路径')
    
    args = parser.parse_args()
    
    pipeline = PreprocessingPipeline(args.config)
    
    if args.dataset == 'all':
        pipeline.process_all_datasets()
    else:
        pipeline.process_dataset(args.dataset)

if __name__ == "__main__":
    main()
