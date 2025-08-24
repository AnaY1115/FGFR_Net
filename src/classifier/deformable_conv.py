'''
Time: 2025.8.20
We will upload the comments for the code as soon as possible and update the document content so that you can better understand the code.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

class DeformableConv1d(nn.Module):
    """
    1D可变形卷积层
    参考论文: Deformable Convolutional Networks (ICCV 2017)
    能够根据输入特征动态调整卷积核的采样位置，增强对不规则特征的捕捉能力
    """
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True):
        """
        初始化1D可变形卷积层
        
        Args:
            in_channels: 输入特征通道数
            out_channels: 输出特征通道数
            kernel_size: 卷积核大小
            stride: 步长
            padding: 填充大小
            dilation: 膨胀率
            groups: 分组卷积参数
            bias: 是否使用偏置
        """
        super(DeformableConv1d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)[0]  # 确保为整数
        self.stride = _pair(stride)[0]
        self.padding = _pair(padding)[0]
        self.dilation = _pair(dilation)[0]
        self.groups = groups
        
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, self.kernel_size)
        )
        
        self.offset_conv = nn.Conv1d(
            in_channels,
            self.kernel_size,  # 每个卷积核位置预测1个偏移量
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True
        )
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化卷积权重和偏移量卷积层"""
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征，形状为 [batch_size, in_channels, length]
            
        Returns:
            输出特征，形状为 [batch_size, out_channels, out_length]
        """
        batch_size, channels, length = x.size()
        
        offsets = self.offset_conv(x)
        
        out_length = (length + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1

        grid = torch.linspace(-1.0, 1.0, out_length, device=x.device)
        grid = grid.view(1, -1, 1)  # [1, out_length, 1]
        grid = grid.repeat(batch_size, 1, self.kernel_size)  # [batch_size, out_length, kernel_size]
        
        kernel_offsets = torch.arange(-(self.kernel_size // 2),
                                     (self.kernel_size + 1) // 2, 
                                     device=x.device)
        kernel_offsets = kernel_offsets.view(1, 1, -1)  # [1, 1, kernel_size]
        
        stride_scale = 2.0 / (length - 1) if length > 1 else 0.0
        
        offsets = offsets.permute(0, 2, 1)  # [batch_size, out_length, kernel_size]
        grid = grid + (kernel_offsets + offsets) * stride_scale
        
        grid = grid.unsqueeze(1)
        
        x = x.unsqueeze(2)
        
        sampled_x = F.grid_sample(
            x, 
            grid, 
            mode='bilinear', 
            padding_mode='zeros', 
            align_corners=True
        )
        
        sampled_x = sampled_x.view(batch_size, channels, out_length, self.kernel_size)
        
        sampled_x = sampled_x.permute(0, 2, 1, 3).contiguous()
        sampled_x = sampled_x.view(-1, channels, self.kernel_size)
                output = F.conv1d(
            sampled_x,
            self.weight,
            bias=None,
            stride=1,
            padding=0,
            groups=self.groups
        )
        
        output = output.view(batch_size, out_length, self.out_channels)
        output = output.permute(0, 2, 1).contiguous()
        
        if self.bias is not None:
            output += self.bias.view(1, -1, 1)
        
        return output

# 测试代码
def test_deformable_conv1d():
    """测试1D可变形卷积层"""
    x = torch.randn(2, 32, 100)
    
    dconv = DeformableConv1d(
        in_channels=32,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1
    )
    
    output = dconv(x)
    
    assert output.shape == (2, 64, 100), f"输出形状错误: {output.shape}，预期: (2, 64, 100)"
    
    print("1D可变形卷积层测试通过")

if __name__ == "__main__":
    test_deformable_conv1d()
