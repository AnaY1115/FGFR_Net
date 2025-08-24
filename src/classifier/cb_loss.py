'''
Time: 2025.8.20
We will upload the comments for the code as soon as possible and update the document content so that you can better understand the code.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union

class ClassBalancedLoss(nn.Module):
    """
    类别平衡损失函数
    参考论文: Class-Balanced Loss Based on Effective Number of Samples (CVPR 2019)
    解决类别不平衡问题，通过调整每个类别的权重，使训练过程更加关注样本较少的类别
    """
    def __init__(self, 
                 num_classes: int,
                 beta: float = 0.999,
                 loss_type: str = "cross_entropy",
                 samples_per_class: Optional[Union[np.ndarray, torch.Tensor]] = None):
        """
        初始化类别平衡损失函数
        
        Args:
            num_classes: 类别数量
            beta: 平衡参数，范围[0, 1)，值越大平衡效果越强
            loss_type: 基础损失类型，可选"cross_entropy"或"focal"
            samples_per_class: 每个类别的样本数量，用于计算类别权重
        """
        super(ClassBalancedLoss, self).__init__()
        
        assert 0 <= beta < 1, "beta参数必须在[0, 1)范围内"
        assert loss_type in ["cross_entropy", "focal"], "损失类型必须是'cross_entropy'或'focal'"
        
        self.num_classes = num_classes
        self.beta = beta
        self.loss_type = loss_type
        
        if samples_per_class is not None:
            if isinstance(samples_per_class, np.ndarray):
                samples_per_class = torch.from_numpy(samples_per_class)
            self.samples_per_class = samples_per_class.float()
        else:
            self.samples_per_class = torch.ones(num_classes, dtype=torch.float32)
        
        self.class_weights = self._calculate_class_weights()
        
        self.gamma = 2.0 if loss_type == "focal" else 0.0
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def _calculate_class_weights(self) -> torch.Tensor:
        """计算类别权重"""
        effective_num = 1.0 - torch.pow(self.beta, self.samples_per_class)
        
        weights = (1.0 - self.beta) / effective_num
        
        weights = weights / torch.sum(weights) * self.num_classes
        
        return weights
    
    def update_samples_per_class(self, samples_per_class: Union[np.ndarray, torch.Tensor]):
        """更新每个类别的样本数量，并重新计算权重"""
        if isinstance(samples_per_class, np.ndarray):
            samples_per_class = torch.from_numpy(samples_per_class)
        
        self.samples_per_class = samples_per_class.float().to(self.device)
        self.class_weights = self._calculate_class_weights()
    
    def forward(self, 
                logits: torch.Tensor, 
                targets: torch.Tensor,
                reduction: str = "mean") -> torch.Tensor:
        """
        前向传播计算损失
        
        Args:
            logits: 模型输出的logits，形状为 [batch_size, num_classes]
            targets: 标签，形状为 [batch_size]
            reduction: 损失聚合方式，可选"mean"、"sum"或"none"
            
        Returns:
            计算得到的损失值
        """
        batch_weights = self.class_weights[targets]
        
        if self.loss_type == "cross_entropy":
            loss = F.cross_entropy(logits, targets, reduction="none")
            
            loss = loss * batch_weights
            
        elif self.loss_type == "focal":
            log_probs = F.log_softmax(logits, dim=1)
            probs = torch.exp(log_probs)
            
            ce_loss = F.nll_loss(log_probs, targets, reduction="none")
            
            one_hot = F.one_hot(targets, num_classes=self.num_classes)
            modulating_factor = torch.pow(1.0 - probs, self.gamma)
            
            loss = modulating_factor * ce_loss * batch_weights
        
        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        else:  # "none"
            return loss

def test_class_balanced_loss():
    """测试类别平衡损失函数"""
    num_classes = 5
    samples_per_class = np.array([1000, 500, 200, 100, 50], dtype=np.float32)
    
    batch_size = 32
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    cb_cross_entropy = ClassBalancedLoss(
        num_classes=num_classes,
        beta=0.999,
        loss_type="cross_entropy",
        samples_per_class=samples_per_class
    )
    loss_ce = cb_cross_entropy(logits, targets)
    assert not torch.isnan(loss_ce), "交叉熵损失计算出现NaN"
    
    cb_focal = ClassBalancedLoss(
        num_classes=num_classes,
        beta=0.99,
        loss_type="focal",
        samples_per_class=samples_per_class
    )
    loss_focal = cb_focal(logits, targets)
    assert not torch.isnan(loss_focal), "Focal损失计算出现NaN"
    
    new_samples = np.array([1000, 500, 200, 100, 100], dtype=np.float32)
    cb_cross_entropy.update_samples_per_class(new_samples)
    
    print("类别平衡损失函数测试通过")

if __name__ == "__main__":
    test_class_balanced_loss()
