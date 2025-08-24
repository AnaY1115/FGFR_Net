'''
Time: 2025.8.20
We will upload the comments for the code as soon as possible and update the document content so that you can better understand the code.
'''
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NodeConstructor:
    """
    节点构造器，负责从流量数据中提取和构建节点特征
    """
    
    def __init__(self, include_position_features: bool = True, 
                 include_statistical_features: bool = True,
                 max_node_features: int = 32):
        """
        初始化节点构造器
        
        Args:
            include_position_features: 是否包含位置特征
            include_statistical_features: 是否包含统计特征
            max_node_features: 节点特征的最大维度
        """
        self.include_position_features = include_position_features
        self.include_statistical_features = include_statistical_features
        self.max_node_features = max_node_features
        
        self.position_embeddings = self._create_position_embeddings(
            max_length=1500,
            embedding_dim=16
        )
        
        logger.info("节点构造器初始化完成")
    
    def _create_position_embeddings(self, max_length: int, embedding_dim: int) -> np.ndarray:
        """
        创建位置嵌入查找表，使用正弦余弦函数
        
        Args:
            max_length: 最大位置长度
            embedding_dim: 嵌入维度
            
        Returns:
            位置嵌入矩阵，形状为[max_length, embedding_dim]
        """
        position = np.arange(max_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, embedding_dim, 2) * (-np.log(10000.0) / embedding_dim))
        
        embeddings = np.zeros((max_length, embedding_dim))
        embeddings[:, 0::2] = np.sin(position * div_term)
        embeddings[:, 1::2] = np.cos(position * div_term)
        
        return embeddings
    
    def _extract_statistical_features(self, byte_value: int, payload: np.ndarray) -> np.ndarray:
        """
        提取字节的统计特征
        
        Args:
            byte_value: 当前字节值
            payload: 整个 payload 数据
            
        Returns:
            统计特征向量
        """
        frequency = np.mean(payload == byte_value)
        
        indices = np.where(payload == byte_value)[0]
        prev_bytes = []
        next_bytes = []
        
        for idx in indices:
            if idx > 0:
                prev_bytes.append(payload[idx-1])
            if idx < len(payload) - 1:
                next_bytes.append(payload[idx+1])
        
        prev_mean = np.mean(prev_bytes) if prev_bytes else 0.0
        prev_var = np.var(prev_bytes) if prev_bytes else 0.0
        next_mean = np.mean(next_bytes) if next_bytes else 0.0
        next_var = np.var(next_bytes) if next_bytes else 0.0
        
        return np.array([
            frequency,
            prev_mean / 255.0,  # 标准化到0-1范围
            prev_var / (255.0**2),
            next_mean / 255.0,
            next_var / (255.0**2)
        ])
    
    def construct_nodes(self, payload: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        从payload数据构建节点特征
        
        Args:
            payload: 预处理后的流量payload数据，形状为[N]，N为字节数
            
        Returns:
            节点特征矩阵，形状为[N, F]，其中F是特征维度
            包含节点构建信息的字典
        """
        if len(payload) == 0:
            return np.array([]), {"num_nodes": 0, "feature_dim": 0}
        
        num_nodes = len(payload)
        features = []
        
        for i, byte_value in enumerate(payload):
            byte_feature = np.array([byte_value / 255.0])
            
            position_feature = np.array([])
            if self.include_position_features and i < len(self.position_embeddings):
                position_feature = self.position_embeddings[i]
            
            stat_feature = np.array([])
            if self.include_statistical_features:
                stat_feature = self._extract_statistical_features(byte_value, payload)
            
            node_feature = np.concatenate([byte_feature, position_feature, stat_feature])
            
            if len(node_feature) > self.max_node_features:
                node_feature = node_feature[:self.max_node_features]
            elif len(node_feature) < self.max_node_features:
                node_feature = np.pad(
                    node_feature, 
                    (0, self.max_node_features - len(node_feature)),
                    mode='constant'
                )
            
            features.append(node_feature)
        
        node_features = np.array(features)
        
        info = {
            "num_nodes": num_nodes,
            "feature_dim": node_features.shape[1],
            "include_position": self.include_position_features,
            "include_statistics": self.include_statistical_features
        }
        
        return node_features, info
    
    def batch_construct_nodes(self, payloads: List[np.ndarray]) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        批量构建多个payload的节点特征
        
        Args:
            payloads: 多个payload数据的列表
            
        Returns:
            节点特征矩阵的列表
            信息字典的列表
        """
        all_node_features = []
        all_infos = []
        
        for payload in payloads:
            nodes, info = self.construct_nodes(payload)
            all_node_features.append(nodes)
            all_infos.append(info)
        
        logger.info(f"批量处理完成，共处理{len(payloads)}个payload，生成{sum(info['num_nodes'] for info in all_infos)}个节点")
        
        return all_node_features, all_infos
    