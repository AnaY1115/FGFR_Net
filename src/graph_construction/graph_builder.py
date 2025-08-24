'''
Time: 2025.8.20
We will upload the comments for the code as soon as possible and update the document content so that you can better understand the code.
'''
import os
import json
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import torch
from torch_geometric.data import Data, Dataset
from node_constructor import NodeConstructor
from edge_constructor import EdgeConstructor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrafficGraphBuilder:
    """
    流量图构建器，协调节点构造器和边构造器，将预处理后的流量数据转换为图结构
    """
    
    def __init__(self, config_path: str = "config/model_config.json"):
        """
        初始化流量图构建器
        
        Args:
            config_path: 模型配置文件路径
        """
        self.config = self._load_config(config_path)
        graph_config = self.config.get("graph_construction", {})
        
        self.node_constructor = NodeConstructor(
            include_position_features=graph_config.get("include_position_features", True),
            include_statistical_features=graph_config.get("include_statistical_features", True),
            max_node_features=graph_config.get("max_node_features", 32)
        )
        
        self.edge_constructor = EdgeConstructor(
            window_size=graph_config.get("window_size", 10),
            llr_threshold=graph_config.get("llr_threshold", 0.1),
            include_directional_edges=graph_config.get("include_directional_edges", True),
            max_edges_per_node=graph_config.get("max_edges_per_node", 20)
        )
        
        self._create_output_directories()
        
        logger.info("流量图构建器初始化完成")
    
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"成功加载配置文件: {config_path}")
            return config
        except Exception as e:
            logger.warning(f"加载配置文件失败，使用默认配置: {str(e)}")
            return {"graph_construction": {}}
    
    def _create_output_directories(self):
        """创建图数据的输出目录"""
        for dataset in ['iscx', 'ustc']:
            graph_path = self.config.get("dataset", {}).get(f"{dataset}_graph_path", 
                                                          f"datasets/{dataset}/graphs")
            Path(graph_path).mkdir(parents=True, exist_ok=True)
    
    def build_graph(self, payload: np.ndarray, label: Optional[int] = None) -> Tuple[Data, Dict]:
        """
        从单个流量payload构建图
        
        Args:
            payload: 预处理后的流量payload，形状为[N]
            label: 可选的流量标签
            
        Returns:
            PyTorch Geometric的Data对象，包含图数据
            构建信息字典
        """
        if len(payload) == 0:
            logger.warning("空payload，无法构建图")
            return None, {"error": "空payload"}
        
        node_features, node_info = self.node_constructor.construct_nodes(payload)
        
        if node_info["num_nodes"] == 0:
            logger.warning("未生成任何节点，无法构建图")
            return None, {"error": "未生成任何节点"}
        
        edge_index, edge_weights, edge_info = self.edge_constructor.construct_edges(payload)
        
        data = Data(
            x=torch.tensor(node_features, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_weights, dtype=torch.float32).view(-1, 1) if len(edge_weights) > 0 else None,
            y=torch.tensor([label], dtype=torch.long) if label is not None else None,
            num_nodes=node_info["num_nodes"]
        )
        
        info = {
            "num_nodes": node_info["num_nodes"],
            "num_edges": edge_info["num_edges"],
            "node_feature_dim": node_info["feature_dim"],
            "mean_edge_weight": edge_info["mean_weight"],
            "window_size_used": edge_info["window_size_used"],
            "construction_time": None  # 可以添加时间戳
        }
        
        return data, info
    
    def _load_cleaned_payloads(self, dataset_path: str, limit: Optional[int] = None) -> Tuple[List[np.ndarray], List[int]]:
        """
        从清洗后的数据集目录加载payload数据
        
        Args:
            dataset_path: 清洗后的数据集目录
            limit: 加载的样本数量限制，None表示加载全部
            
        Returns:
            payload列表
            对应的标签列表
        """
        payloads = []
        labels = []
        
        for label_dir in os.listdir(dataset_path):
            label_path = os.path.join(dataset_path, label_dir)
            if not os.path.isdir(label_path):
                continue
                
            try:
                label = int(label_dir)
            except ValueError:
                logger.warning(f"无法将目录名'{label_dir}'转换为标签，跳过该目录")
                continue
                
            file_count = 0
            for filename in os.listdir(label_path):
                if filename.endswith(".npy"):
                    file_path = os.path.join(label_path, filename)
                    try:
                        payload = np.load(file_path)
                        payloads.append(payload)
                        labels.append(label)
                        file_count += 1
                        
                        if limit is not None and len(payloads) >= limit:
                            logger.info(f"已达到加载限制{limit}，停止加载")
                            return payloads, labels
                    except Exception as e:
                        logger.error(f"加载文件{file_path}失败: {str(e)}")
            
            logger.info(f"从标签{label}加载了{file_count}个payload")
        
        logger.info(f"共加载了{len(payloads)}个payload")
        return payloads, labels
    
    def batch_build_graphs(self, dataset_name: str, limit: Optional[int] = None, 
                           train_distribution: bool = True) -> Tuple[List[Data], List[Dict]]:
        """
        批量为指定数据集构建图
        
        Args:
            dataset_name: 数据集名称，"iscx"或"ustc"
            limit: 处理的样本数量限制
            train_distribution: 是否使用数据集训练字节分布模型
            
        Returns:
            图数据对象列表
            构建信息列表
        """
        if dataset_name not in ['iscx', 'ustc']:
            logger.error(f"不支持的数据集: {dataset_name}")
            return [], []
        
        cleaned_path = self.config.get("dataset", {}).get(f"{dataset_name}_cleaned_path",
                                                        f"datasets/{dataset_name}/cleaned")
        graph_path = self.config.get("dataset", {}).get(f"{dataset_name}_graph_path", 
                                                      f"datasets/{dataset_name}/graphs")
        
        if not os.path.exists(cleaned_path):
            logger.error(f"清洗后的数据集路径不存在: {cleaned_path}")
            return [], []
        
        logger.info(f"开始加载{dataset_name}数据集的清洗后数据...")
        payloads, labels = self._load_cleaned_payloads(cleaned_path, limit)
        
        if not payloads:
            logger.error("未加载到任何payload数据，无法构建图")
            return [], []
        
        if train_distribution:
            train_size = min(1000, int(len(payloads) * 0.1))
            train_payloads = payloads[:train_size]
            logger.info(f"使用{train_size}个样本训练字节分布模型...")
            self.edge_constructor._train_byte_distributions(train_payloads)
        
        logger.info(f"开始为{dataset_name}数据集构建图，共{len(payloads)}个样本...")
        
        logger.info("批量构建节点特征...")
        all_node_features, all_node_infos = self.node_constructor.batch_construct_nodes(payloads)
        
        logger.info("批量构建边...")
        all_edge_indices, all_edge_weights, all_edge_infos = self.edge_constructor.batch_construct_edges(payloads)
        
        graphs = []
        infos = []
        
        for i in range(len(payloads)):
            if all_node_infos[i]["num_nodes"] == 0:
                logger.warning(f"样本{i}没有节点，跳过")
                continue
                
            data = Data(
                x=torch.tensor(all_node_features[i], dtype=torch.float32),
                edge_index=torch.tensor(all_edge_indices[i], dtype=torch.long),
                edge_attr=torch.tensor(all_edge_weights[i], dtype=torch.float32).view(-1, 1) if len(all_edge_weights[i]) > 0 else None,
                y=torch.tensor([labels[i]], dtype=torch.long),
                num_nodes=all_node_infos[i]["num_nodes"]
            )
            
            graph_file = os.path.join(graph_path, f"graph_{i}.pt")
            torch.save(data, graph_file)
            
            graphs.append(data)
            
            info = {
                "index": i,
                "label": labels[i],
                "num_nodes": all_node_infos[i]["num_nodes"],
                "num_edges": all_edge_infos[i]["num_edges"],
                "node_feature_dim": all_node_infos[i]["feature_dim"],
                "mean_edge_weight": all_edge_infos[i]["mean_weight"],
                "saved_path": graph_file
            }
            infos.append(info)
            
            if (i + 1) % 100 == 0:
                logger.info(f"已处理{ i + 1 }/{ len(payloads) }个样本")
        
        info_file = os.path.join(graph_path, "construction_info.json")
        with open(info_file, 'w') as f:
            json.dump(infos, f, indent=2)
        
        logger.info(f"批量图构建完成，共生成{len(graphs)}个图，保存至{graph_path}")
        
        return graphs, infos

class TrafficGraphDataset(Dataset):
    """PyTorch Geometric数据集类，用于加载构建好的流量图"""
    
    def __init__(self, root: str, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.graph_files = [f for f in os.listdir(self.processed_dir) if f.endswith(".pt")]
        self.graph_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
        
        info_file = os.path.join(self.processed_dir, "construction_info.json")
        if os.path.exists(info_file):
            with open(info_file, 'r') as f:
                self.infos = json.load(f)
        else:
            self.infos = []
        
        logger.info(f"加载了{len(self.graph_files)}个图数据")
    
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return self.graph_files
    
    def download(self):
        pass
    
    def process(self):
        pass
    
    def len(self):
        return len(self.graph_files)
    
    def get(self, idx):
        graph_file = os.path.join(self.processed_dir, self.graph_files[idx])
        data = torch.load(graph_file)
        return data
    