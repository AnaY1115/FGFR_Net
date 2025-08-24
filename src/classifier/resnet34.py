'''
Time: 2025.8.20
We will upload the comments for the code as soon as possible and update the document content so that you can better understand the code.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, Tuple, Union
from deformable_conv import DeformableConv1d
from depthwise_conv import DepthwiseSeparableConv1d
from channel_attention import ChannelAttention

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BasicBlock(nn.Module):
    """ResNet的基本残差块"""
    expansion = 1
    
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 use_deformable: bool = False,
                 use_depthwise: bool = False,
                 use_attention: bool = False):
        """
        初始化基本残差块
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            stride: 步长
            downsample: 下采样模块，用于匹配残差连接的维度
            use_deformable: 是否使用可变形卷积
            use_depthwise: 是否使用深度可分离卷积
            use_attention: 是否使用通道注意力机制
        """
        super(BasicBlock, self).__init__()
        
        conv_layer = None
        if use_deformable:
            conv_layer = DeformableConv1d
        elif use_depthwise:
            conv_layer = DepthwiseSeparableConv1d
        else:
            conv_layer = nn.Conv1d
        
        self.conv1 = conv_layer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = conv_layer(
            in_channels=out_channels,
            out_channels=out_channels * self.expansion,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels * self.expansion)
        
        self.attention = ChannelAttention(out_channels * self.expansion) if use_attention else None
        
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.attention is not None:
            out = self.attention(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out

class ResNet34(nn.Module):
    """
    改进的ResNet-34模型，用于加密流量分类
    支持集成可变形卷积、深度可分离卷积和通道注意力机制
    """
    def __init__(self, 
                 input_dim: int = 256,
                 num_classes: int = 12,
                 in_channels: int = 1,
                 base_width: int = 64,
                 use_deformable: Union[bool, Tuple[bool, ...]] = (False, False, True, True),
                 use_depthwise: Union[bool, Tuple[bool, ...]] = (False, False, False, False),
                 use_attention: Union[bool, Tuple[bool, ...]] = (True, True, True, True)):
        """
        初始化改进的ResNet-34模型
        
        Args:
            input_dim: 输入特征的维度（长度）
            num_classes: 分类的类别数
            in_channels: 输入特征的通道数
            base_width: 基础通道数
            use_deformable: 是否在各阶段使用可变形卷积，可传入布尔值或四元组
            use_depthwise: 是否在各阶段使用深度可分离卷积，可传入布尔值或四元组
            use_attention: 是否在各阶段使用通道注意力，可传入布尔值或四元组
        """
        super(ResNet34, self).__init__()
        
        self._validate_parameters(use_deformable, use_depthwise, use_attention)
        
        self.use_deformable = self._to_tuple(use_deformable, 4)
        self.use_depthwise = self._to_tuple(use_depthwise, 4)
        self.use_attention = self._to_tuple(use_attention, 4)
        
        self.base_width = base_width
        self.input_dim = input_dim
        
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=self.base_width,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(self.base_width)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(
            block=BasicBlock,
            out_channels=self.base_width,
            blocks=3,
            stride=1,
            stage=0
        )
        self.layer2 = self._make_layer(
            block=BasicBlock,
            out_channels=self.base_width * 2,
            blocks=4,
            stride=2,
            stage=1
        )
        self.layer3 = self._make_layer(
            block=BasicBlock,
            out_channels=self.base_width * 4,
            blocks=6,
            stride=2,
            stage=2
        )
        self.layer4 = self._make_layer(
            block=BasicBlock,
            out_channels=self.base_width * 8,
            blocks=3,
            stride=2,
            stage=3
        )
        
        self.final_length = self._calculate_final_length()
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(self.base_width * 8 * BasicBlock.expansion, num_classes)
        
        self._initialize_weights()
        
        logger.info(f"ResNet34初始化完成 - 类别数: {num_classes}, 输入维度: {input_dim}, "
                   f"基础通道数: {base_width}")
    
    def _validate_parameters(self, *params):
        """验证参数有效性"""
        for param in params:
            if not isinstance(param, (bool, tuple)):
                raise ValueError("参数必须是布尔值或四元组")
            if isinstance(param, tuple) and len(param) != 4:
                raise ValueError("元组参数必须包含4个元素，对应四个阶段")
    
    def _to_tuple(self, param: Union[bool, Tuple[bool, ...]], length: int) -> Tuple[bool, ...]:
        """将参数转换为指定长度的元组"""
        if isinstance(param, bool):
            return (param,) * length
        return param
    
    def _make_layer(self, 
                    block: nn.Module,
                    out_channels: int,
                    blocks: int,
                    stride: int,
                    stage: int) -> nn.Sequential:
        """
        构建ResNet的一个阶段
        
        Args:
            block: 残差块类型
            out_channels: 输出通道数
            blocks: 残差块数量
            stride: 步长
            stage: 阶段索引
            
        Returns:
            由多个残差块组成的序列
        """
        downsample = None
        in_channels = self.base_width * (2 ** stage)
        
        if stride != 1 or in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm1d(out_channels * block.expansion),
            )
        
        layers = []
        layers.append(block(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            downsample=downsample,
            use_deformable=self.use_deformable[stage],
            use_depthwise=self.use_depthwise[stage],
            use_attention=self.use_attention[stage]
        ))
        
        in_channels = out_channels * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=1,
                downsample=None,
                use_deformable=self.use_deformable[stage],
                use_depthwise=self.use_depthwise[stage],
                use_attention=self.use_attention[stage]
            ))
        
        return nn.Sequential(*layers)
    
    def _calculate_final_length(self) -> int:
        length = self.input_dim
        
        length = (length + 2*3 - 7) // 2 + 1  # conv1
        length = (length + 2*1 - 3) // 2 + 1  # maxpool
        
        strides = [1, 2, 2, 2]
        for i in range(4):
            length = (length + 2*1 - 3) // strides[i] + 1  # 第一个块的conv1
            length = (length + 2*1 - 3) // 1 + 1  # 第一个块的conv2
            
            blocks = [3, 4, 6, 3][i]
            for _ in range(1, blocks):
                length = (length + 2*1 - 3) // 1 + 1  # conv1
                length = (length + 2*1 - 3) // 1 + 1  # conv2
        
        return length
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征，形状为 [batch_size, input_dim] 或 [batch_size, in_channels, input_dim]
            
        Returns:
            分类结果，形状为 [batch_size, num_classes]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # 添加通道维度
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

def test_resnet34():
    """测试改进的ResNet34模型"""
    x = torch.randn(2, 256)
    
    model = ResNet34(
        input_dim=256,
        num_classes=12,
        use_deformable=(False, False, True, True),
        use_attention=True
    )
    
    output = model(x)
    
    assert output.shape == (2, 12), f"输出形状错误: {output.shape}，预期: (2, 12)"
    
    logger.info("ResNet34模型测试通过")

if __name__ == "__main__":
    test_resnet34()
