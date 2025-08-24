'''
Time: 2025.8.20
We will upload the comments for the code as soon as possible and update the document content so that you can better understand the code.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """
    通道注意力机制模块
    参考论文: Squeeze-and-Excitation Networks (CVPR 2018)
    对特征图的每个通道进行权重调整，增强重要特征通道
    """
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        """
        初始化通道注意力模块
        
        Args:
            in_channels: 输入特征图的通道数
            reduction_ratio: 通道数缩减比例，用于控制中间层维度
        """
        super(ChannelAttention, self).__init__()
        
        self.reduction_ratio = reduction_ratio if in_channels > reduction_ratio else 1
        self.intermediate_channels = in_channels // self.reduction_ratio
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, self.intermediate_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.intermediate_channels, in_channels, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征图，形状为 [batch_size, channels, length]
            
        Returns:
            经过通道注意力加权的特征图，形状与输入相同
        """
        batch_size, channels, length = x.size()
        
        avg_pool = self.avg_pool(x).view(batch_size, channels)  # [batch_size, channels]
        max_pool = self.max_pool(x).view(batch_size, channels)  # [batch_size, channels]
        
        avg_att = self.mlp(avg_pool).view(batch_size, channels, 1)  # [batch_size, channels, 1]
        max_att = self.mlp(max_pool).view(batch_size, channels, 1)  # [batch_size, channels, 1]
        
        att_weights = self.sigmoid(avg_att + max_att)  # [batch_size, channels, 1]
        
        return x * att_weights  # [batch_size, channels, length]

def test_channel_attention():
    """测试通道注意力模块"""
    x = torch.randn(2, 64, 100)
    
    ca = ChannelAttention(in_channels=64, reduction_ratio=16)
    
    output = ca(x)
    
    assert output.shape == x.shape, f"输出形状错误: {output.shape}，预期: {x.shape}"
    
    assert torch.all(output >= 0), "输出值不应为负"
    
    print("通道注意力模块测试通过")

if __name__ == "__main__":
    test_channel_attention()
