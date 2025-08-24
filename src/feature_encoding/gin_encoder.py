'''
Time: 2025.8.20
We will upload the comments for the code as soon as possible and update the document content so that you can better understand the code.
'''
import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.data import Batch
import json
import logging
from typing import Tuple, Optional, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GINEncoder(torch.nn.Module):
    """
    基于图同构网络(GIN)的特征编码器，将字节级流量图转换为固定维度的特征向量
    参考论文: How Powerful are Graph Neural Networks? (ICLR 2019)
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 128,
                 output_dim: int = 256,
                 num_layers: int = 5,
                 mlp_layers: int = 2,
                 dropout: float = 0.3,
                 pooling_type: str = "mean",
                 residual: bool = True,
                 config_path: Optional[str] = None):
        """
        初始化GIN编码器
        
        Args:
            input_dim: 节点特征的输入维度
            hidden_dim: 隐藏层维度
            output_dim: 输出特征向量维度
            num_layers: GIN卷积层数量
            mlp_layers: 每个GIN层中MLP的层数
            dropout: Dropout概率
            pooling_type: 全局池化类型，可选"mean"、"add"、"max"或"concat"
            residual: 是否使用残差连接
            config_path: 配置文件路径，用于从文件加载参数(可选)
        """
        super(GINEncoder, self).__init__()
        
        if config_path:
            config = self._load_config(config_path)
            self._init_from_config(config)
        else:
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.output_dim = output_dim
            self.num_layers = num_layers
            self.mlp_layers = mlp_layers
            self.dropout = dropout
            self.pooling_type = pooling_type
            self.residual = residual
        
        self._validate_parameters()
        
        self.conv_layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        
        self.conv_layers.append(self._build_gin_layer(self.input_dim, self.hidden_dim))
        self.batch_norms.append(BatchNorm1d(self.hidden_dim))
        
        for _ in range(self.num_layers - 1):
            self.conv_layers.append(self._build_gin_layer(self.hidden_dim, self.hidden_dim))
            self.batch_norms.append(BatchNorm1d(self.hidden_dim))
        
        self.output_projection = Linear(self.hidden_dim, self.output_dim)
        
        self.dropout_layer = Dropout(self.dropout)
        
        logger.info(f"GIN编码器初始化完成 - 卷积层: {self.num_layers}, 隐藏维度: {self.hidden_dim}, "
                   f"输出维度: {self.output_dim}, 池化类型: {self.pooling_type}")
    
    def _load_config(self, config_path: str) -> Dict:
        """从配置文件加载参数"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            encoder_config = config.get("gin_encoder", {})
            logger.info(f"从配置文件加载GIN编码器参数: {config_path}")
            return encoder_config
        except Exception as e:
            logger.error(f"加载配置文件失败: {str(e)}，使用默认参数")
            return {}
    
    def _init_from_config(self, config: Dict):
        """从配置字典初始化参数"""
        self.input_dim = config.get("input_dim", 32)
        self.hidden_dim = config.get("hidden_dim", 128)
        self.output_dim = config.get("output_dim", 256)
        self.num_layers = config.get("num_layers", 5)
        self.mlp_layers = config.get("mlp_layers", 2)
        self.dropout = config.get("dropout", 0.3)
        self.pooling_type = config.get("pooling_type", "mean")
        self.residual = config.get("residual", True)
    
    def _validate_parameters(self):
        """验证参数有效性"""
        assert self.num_layers >= 1, "卷积层数量必须至少为1"
        assert self.mlp_layers >= 1, "MLP层数必须至少为1"
        assert 0 <= self.dropout < 1, "dropout概率必须在[0, 1)范围内"
        assert self.pooling_type in ["mean", "add", "max", "concat"], \
            "池化类型必须是'mean', 'add', 'max'或'concat'"
        assert self.hidden_dim > 0 and self.output_dim > 0, "隐藏层和输出层维度必须为正数"
    
    def _build_mlp(self, input_dim: int, output_dim: int) -> torch.nn.Sequential:
        """构建GIN卷积中使用的MLP"""
        layers = []
        in_dim = input_dim
        
        for _ in range(self.mlp_layers - 1):
            layers.append(Linear(in_dim, self.hidden_dim))
            layers.append(BatchNorm1d(self.hidden_dim))
            layers.append(ReLU())
            in_dim = self.hidden_dim
        
        layers.append(Linear(in_dim, output_dim))
        
        return torch.nn.Sequential(*layers)
    
    def _build_gin_layer(self, input_dim: int, output_dim: int) -> GINConv:
        mlp = self._build_mlp(input_dim, output_dim)
        return GINConv(mlp, eps=0.0, train_eps=False)
    
    def _global_pooling(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        对节点特征进行全局池化，得到图级特征
        
        Args:
            x: 节点特征矩阵，形状为[num_nodes, hidden_dim]
            batch: 批次索引，形状为[num_nodes]
            
        Returns:
            图特征向量，形状为[batch_size, output_dim]
        """
        if self.pooling_type == "mean":
            return global_mean_pool(x, batch)
        elif self.pooling_type == "add":
            return global_add_pool(x, batch)
        elif self.pooling_type == "max":
            return global_max_pool(x, batch)
        elif self.pooling_type == "concat":
            # 拼接多种池化结果
            mean_pool = global_mean_pool(x, batch)
            add_pool = global_add_pool(x, batch)
            max_pool = global_max_pool(x, batch)
            return torch.cat([mean_pool, add_pool, max_pool], dim=1)
    
    def forward(self, data: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播函数
        
        Args:
            data: PyTorch Geometric的Batch对象，包含:
                - x: 节点特征，形状为[num_nodes, input_dim]
                - edge_index: 边索引，形状为[2, num_edges]
                - batch: 批次索引，形状为[num_nodes]
                
        Returns:
            graph_emb: 图级别特征向量，形状为[batch_size, output_dim]
            node_emb: 节点级别特征向量，形状为[num_nodes, hidden_dim]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        h = x  # 初始特征
        
        for i in range(self.num_layers):
            conv = self.conv_layers[i]
            bn = self.batch_norms[i]
            
            h_new = conv(h, edge_index)
            h_new = bn(h_new)
            h_new = F.relu(h_new)
            h_new = self.dropout_layer(h_new)
            
            if self.residual and h.size(-1) == h_new.size(-1):
                h = h + h_new
            else:
                h = h_new
        
        node_emb = h
        
        graph_emb = self._global_pooling(h, batch)
        
        if self.pooling_type == "concat":
            graph_emb = self.output_projection(graph_emb)
        else:
            graph_emb = self.output_projection(graph_emb)
        
        return graph_emb, node_emb
    
    def get_graph_embedding(self, data: Batch) -> torch.Tensor:
        """便捷方法，仅返回图级别特征"""
        graph_emb, _ = self.forward(data)
        return graph_emb
    
    def get_node_embedding(self, data: Batch) -> torch.Tensor:
        """便捷方法，仅返回节点级别特征"""
        _, node_emb = self.forward(data)
        return node_emb

def test_gin_encoder():
    """测试GIN编码器的基本功能"""
    num_nodes = 100
    input_dim = 32
    batch_size = 5
    
    x = torch.randn(num_nodes, input_dim)
    edge_index = torch.randint(0, num_nodes, (2, 200))
    batch = torch.randint(0, batch_size, (num_nodes,))
    
    data = Batch(x=x, edge_index=edge_index, batch=batch)
    
    encoder = GINEncoder(
        input_dim=input_dim,
        hidden_dim=128,
        output_dim=256,
        num_layers=3,
        dropout=0.3
    )
    
    graph_emb, node_emb = encoder(data)
    
    assert graph_emb.shape == (batch_size, 256), f"图嵌入形状错误: {graph_emb.shape}"
    assert node_emb.shape == (num_nodes, 128), f"节点嵌入形状错误: {node_emb.shape}"
    
    logger.info("GIN编码器测试通过")

if __name__ == "__main__":
    test_gin_encoder()
    