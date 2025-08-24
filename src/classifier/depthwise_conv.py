'''
Time: 2025.8.20
We will upload the comments for the code as soon as possible and update the document content so that you can better understand the code.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv1d(nn.Module):
    """
    1D深度可分离卷积层
    将标准卷积分解为深度卷积(depthwise convolution)和逐点卷积(pointwise convolution)
    能够在保持性能的同时显著减少参数量和计算量
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 dilation: int = 1,
                 depth_multiplier: int = 1,
                 bias: bool = True):
        """
        初始化1D深度可分离卷积层
        
        Args:
            in_channels: 输入特征通道数
            out_channels: 输出特征通道数
            kernel_size: 卷积核大小
            stride: 步长
            padding: 填充大小
            dilation: 膨胀率
            depth_multiplier: 深度卷积的通道倍数
            bias: 是否使用偏置
        """
        super(DepthwiseSeparableConv1d, self).__init__()
        
        self.depthwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels * depth_multiplier,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False
        )
        
        self.pointwise = nn.Conv1d(
            in_channels=in_channels * depth_multiplier,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias
        )
        
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化卷积层权重"""
        nn.init.kaiming_normal_(self.depthwise.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.pointwise.weight, mode='fan_out', nonlinearity='relu')
        if self.pointwise.bias is not None:
            nn.init.zeros_(self.pointwise.bias)
        nn.init.ones_(self.bn.weight)
        nn.init.zeros_(self.bn.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征，形状为 [batch_size, in_channels, length]
            
        Returns:
            输出特征，形状为 [batch_size, out_channels, out_length]
        """
        x = self.depthwise(x)
        
        x = self.pointwise(x)
        
        x = self.bn(x)
        x = self.relu(x)
        
        return x

def test_depthwise_separable_conv1d():
    """测试1D深度可分离卷积层"""
    x = torch.randn(2, 32, 100)
    
    dwconv = DepthwiseSeparableConv1d(
        in_channels=32,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        depth_multiplier=1
    )
    
    output = dwconv(x)
    
    assert output.shape == (2, 64, 100), f"输出形状错误: {output.shape}，预期: (2, 64, 100)"

    print("1D深度可分离卷积层测试通过")

if __name__ == "__main__":
    test_depthwise_separable_conv1d()
