'''
Time: 2025.8.20
We will upload the comments for the code as soon as possible and update the document content so that you can better understand the code.
'''
import numpy as np
import logging
from scipy.stats import entropy
from typing import List, Tuple, Dict, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EdgeConstructor:
    """
    边构造器，负责计算节点之间的连接关系和权重
    实现了基于LLR(Log-Likelihood Ratio)的边权重计算
    """
    
    def __init__(self, window_size: int = 10, 
                 llr_threshold: float = 0.1,
                 include_directional_edges: bool = True,
                 max_edges_per_node: int = 20):
        """
        初始化边构造器
        
        Args:
            window_size: 考虑的上下文窗口大小，用于计算相邻关系
            llr_threshold: LLR阈值，低于此值的边将被过滤
            include_directional_edges: 是否包含有向边
            max_edges_per_node: 每个节点的最大边数，防止图过于稠密
        """
        self.window_size = window_size
        self.llr_threshold = llr_threshold
        self.include_directional_edges = include_directional_edges
        self.max_edges_per_node = max_edges_per_node
        
        self.byte_pair_probs = None
        
        logger.info("边构造器初始化完成")
    
    def _train_byte_distributions(self, payloads: List[np.ndarray]) -> None:
        """
        从训练数据中学习字节和字节对的概率分布
        
        Args:
            payloads: 用于训练的payload列表
        """
        logger.info("开始训练字节分布模型...")
        
        byte_counts = np.zeros(256, dtype=np.float64)
        byte_pair_counts = np.zeros((256, 256), dtype=np.float64)
        total_bytes = 0
        total_pairs = 0
        
        for payload in payloads:
            if len(payload) < 2:
                continue
                
            byte_values = payload.astype(int)
            np.add.at(byte_counts, byte_values, 1)
            total_bytes += len(payload)
            
            pairs = np.stack([byte_values[:-1], byte_values[1:]], axis=1)
            for a, b in pairs:
                byte_pair_counts[a, b] += 1
                total_pairs += 1
                if not self.include_directional_edges:
                    byte_pair_counts[b, a] += 1
                    total_pairs += 1
        
        smooth = 1e-10
        byte_probs = (byte_counts + smooth) / (total_bytes + 256 * smooth)
        
        self.byte_pair_probs = (byte_pair_counts + smooth) / (total_pairs + (256**2) * smooth)
        
        self.independent_probs = np.outer(byte_probs, byte_probs)
        
        logger.info("字节分布模型训练完成")
    
    def calculate_llr(self, a: int, b: int) -> float:
        """
        计算两个字节之间的LLR(Log-Likelihood Ratio)
        
        LLR(a,b) = log(P(a,b) / (P(a)P(b)))
        
        当LLR为正时，表示两个字节倾向于共同出现；为负时，表示它们倾向于避免共同出现
        
        Args:
            a: 第一个字节值
            b: 第二个字节值
            
        Returns:
            LLR值
        """
        if self.byte_pair_probs is None:
            raise RuntimeError("字节分布模型尚未训练，请先调用train_byte_distributions方法")
        
        a_clamped = np.clip(a, 0, 255)
        b_clamped = np.clip(b, 0, 255)
        
        joint_prob = self.byte_pair_probs[a_clamped, b_clamped]
        indep_prob = self.independent_probs[a_clamped, b_clamped]
        
        if indep_prob < 1e-20:
            return 0.0
            
        return np.log(joint_prob / indep_prob)
    
    def _calculate_transition_entropy(self, payload: np.ndarray) -> float:
        """
        计算payload中字节转换的熵，用于衡量字节序列的随机性
        
        Args:
            payload: 字节序列
            
        Returns:
            转换熵值
        """
        if len(payload) < 2:
            return 0.0
            
        transitions = np.stack([payload[:-1], payload[1:]], axis=1)
        counts = {}
        
        for a, b in transitions:
            counts[(a, b)] = counts.get((a, b), 0) + 1
            
        total = len(transitions)
        probs = [c / total for c in counts.values()]
        
        return entropy(probs)
    
    def construct_edges(self, payload: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        为单个payload构建边
        
        Args:
            payload: 字节序列，形状为[N]
            
        Returns:
            边索引，形状为[2, E]
            边权重，形状为[E]
            包含边构建信息的字典
        """
        if len(payload) < 2:
            return np.array([[], []]), np.array([]), {"num_edges": 0, "mean_weight": 0.0}
        
        num_nodes = len(payload)
        edges = []
        weights = []
        
        transition_entropy = self._calculate_transition_entropy(payload)
        adaptive_window = max(2, int(self.window_size * (1 - min(0.8, transition_entropy / 8))))
        
        for i in range(num_nodes):
            current_byte = payload[i]
            
            start = max(0, i - adaptive_window)
            end = min(num_nodes, i + adaptive_window + 1)
            
            candidates = []
            for j in range(start, end):
                if i != j:
                    candidate_byte = payload[j]
                    weight = self.calculate_llr(current_byte, candidate_byte)
                    #
                    if abs(weight) >= self.llr_threshold:
                        candidates.append((j, weight))
            
            if len(candidates) > self.max_edges_per_node:
                candidates.sort(key=lambda x: abs(x[1]), reverse=True)
                candidates = candidates[:self.max_edges_per_node]
            
            for j, weight in candidates:
                edges.append([i, j])
                weights.append(weight)
                
                if not self.include_directional_edges and i < j:
                    edges.append([j, i])
                    weights.append(weight)
        
        edge_index = np.array(edges).T if edges else np.array([[], []])
        edge_weights = np.array(weights) if weights else np.array([])
        
        if edge_index.shape[1] > 0:
            if not self.include_directional_edges:
                mask = edge_index[0] > edge_index[1]
                edge_index[:, mask] = edge_index[::-1, mask]
            
            unique_edges = np.unique(edge_index.T, axis=0)
            edge_index = unique_edges.T
            
            if len(edge_weights) > 0:
                unique_weights = []
                for u, v in unique_edges:
                    mask = (edge_index[0] == u) & (edge_index[1] == v)
                    unique_weights.append(np.mean(edge_weights[mask]))
                edge_weights = np.array(unique_weights)
        
        info = {
            "num_edges": edge_index.shape[1],
            "mean_weight": np.mean(np.abs(edge_weights)) if len(edge_weights) > 0 else 0.0,
            "window_size_used": adaptive_window,
            "transition_entropy": transition_entropy
        }
        
        return edge_index, edge_weights, info
    
    def batch_construct_edges(self, payloads: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict]]:
        """
        批量为多个payload构建边
        
        Args:
            payloads: 多个字节序列的列表
            
        Returns:
            边索引列表
            边权重列表
            信息字典列表
        """
        if self.byte_pair_probs is None:
            self._train_byte_distributions(payloads)
        
        all_edge_indices = []
        all_edge_weights = []
        all_infos = []
        
        for payload in payloads:
            edges, weights, info = self.construct_edges(payload)
            all_edge_indices.append(edges)
            all_edge_weights.append(weights)
            all_infos.append(info)
        
        total_edges = sum(info["num_edges"] for info in all_infos)
        avg_edges = total_edges / len(payloads) if payloads else 0
        logger.info(f"批量处理完成，共处理{len(payloads)}个payload，生成{total_edges}条边，平均每个图{avg_edges:.2f}条边")
        
        return all_edge_indices, all_edge_weights, all_infos
    