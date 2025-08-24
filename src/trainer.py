'''
Time: 2025.8.20
We will upload the comments for the code as soon as possible and update the document content so that you can better understand the code.
'''
import os
import json
import logging
import argparse
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeometricDataLoader
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from feature_encoding.gin_encoder import GINEncoder
from classifier.resnet34 import ResNet34
from classifier.cb_loss import ClassBalancedLoss
from graph_construction.graph_builder import TrafficGraphDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Trainer:
    """模型训练器，负责FGFR-Net模型的端到端训练"""
    
    def __init__(self, dataset_name: str, config_path: str = "config/model_config.json"):
        """
        初始化训练器
        
        Args:
            dataset_name: 数据集名称，"iscx"或"ustc"
            config_path: 模型配置文件路径
        """
        self.config = self._load_config(config_path)
        self.dataset_name = dataset_name
        
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.config["use_cuda"] else "cpu")
        logger.info(f"使用设备: {self.device}")
        
        self._create_directories()
        
        self._load_dataset()
        

        self._init_model()
        
        self._init_optimizer()
        

        self._init_loss_function()
        

        self.best_val_acc = 0.0
        self.train_history = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": []
        }
    
    def _load_config(self, config_path: str) -> dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"成功加载配置文件: {config_path}")
            return config
        except Exception as e:
            logger.error(f"加载配置文件失败: {str(e)}")
            raise
    
    def _create_directories(self):
        """创建训练所需的目录"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_save_dir = os.path.join(
            self.config["model_save_path"],
            f"{self.dataset_name}_{timestamp}"
        )
        Path(self.model_save_dir).mkdir(parents=True, exist_ok=True)
        

        self.log_dir = os.path.join(self.config["log_path"], self.dataset_name)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"模型将保存至: {self.model_save_dir}")
    
    def _load_dataset(self):
        """加载并准备数据集"""

        graph_path = self.config["dataset"][f"{self.dataset_name}_graph_path"]
        if not os.path.exists(graph_path):
            logger.error(f"图数据路径不存在: {graph_path}")
            raise FileNotFoundError(f"图数据路径不存在: {graph_path}")
        

        logger.info(f"加载{dataset_name}数据集...")
        full_dataset = TrafficGraphDataset(root=graph_path)
        

        train_indices, val_indices = train_test_split(
            range(len(full_dataset)),
            test_size=self.config["val_split"],
            random_state=self.config["random_seed"],
            shuffle=True
        )
        
        self.train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        self.val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
        
        self.train_loader = GeometricDataLoader(
            self.train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config["num_workers"]
        )
        
        self.val_loader = GeometricDataLoader(
            self.val_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["num_workers"]
        )
        
        logger.info(f"数据集加载完成 - 训练样本: {len(self.train_dataset)}, 验证样本: {len(self.val_dataset)}")
        
        self._calculate_class_distribution()
    
    def _calculate_class_distribution(self):
        """计算训练集中每个类别的样本数量"""
        labels = []
        for idx in self.train_dataset.indices:
            data = self.train_dataset.dataset[idx]
            labels.append(data.y.item())
        
        self.num_classes = len(np.unique(labels))
        self.samples_per_class = np.bincount(labels, minlength=self.num_classes)
        logger.info(f"类别分布: {self.samples_per_class}, 总类别数: {self.num_classes}")
    
    def _init_model(self):
        """初始化模型组件"""
        self.gin_encoder = GINEncoder(
            input_dim=self.config["gin_encoder"]["input_dim"],
            hidden_dim=self.config["gin_encoder"]["hidden_dim"],
            output_dim=self.config["gin_encoder"]["output_dim"],
            num_layers=self.config["gin_encoder"]["num_layers"],
            dropout=self.config["gin_encoder"]["dropout"]
        ).to(self.device)
        
        self.classifier = ResNet34(
            input_dim=self.config["gin_encoder"]["output_dim"],
            num_classes=self.num_classes,
            use_deformable_conv=self.config["classifier"]["use_deformable_conv"],
            use_depthwise_conv=self.config["classifier"]["use_depthwise_conv"],
            use_channel_attention=self.config["classifier"]["use_channel_attention"]
        ).to(self.device)
        
        self._log_model_parameters()
    
    def _log_model_parameters(self):
        """计算并记录模型参数数量"""
        encoder_params = sum(p.numel() for p in self.gin_encoder.parameters() if p.requires_grad)
        classifier_params = sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)
        total_params = encoder_params + classifier_params
        
        logger.info(f"模型参数数量 - 编码器: {encoder_params:,}, 分类器: {classifier_params:,}, 总计: {total_params:,}")
    
    def _init_optimizer(self):
        """初始化优化器和学习率调度器"""
        params = list(self.gin_encoder.parameters()) + list(self.classifier.parameters())
        
        if self.config["optimizer"] == "adam":
            self.optimizer = optim.Adam(
                params,
                lr=self.config["learning_rate"],
                weight_decay=self.config["weight_decay"]
            )
        elif self.config["optimizer"] == "sgd":
            self.optimizer = optim.SGD(
                params,
                lr=self.config["learning_rate"],
                momentum=0.9,
                weight_decay=self.config["weight_decay"]
            )
        else:
            raise ValueError(f"不支持的优化器: {self.config['optimizer']}")
        
        if self.config["lr_scheduler"] == "reduce_on_plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.1,
                patience=5,
                verbose=True
            )
        elif self.config["lr_scheduler"] == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config["epochs"],
                eta_min=1e-6
            )
        else:
            self.scheduler = None
    
    def _init_loss_function(self):
        """初始化损失函数"""
        self.criterion = ClassBalancedLoss(
            num_classes=self.num_classes,
            beta=self.config["loss"]["beta"],
            loss_type=self.config["loss"]["type"],
            samples_per_class=self.samples_per_class
        ).to(self.device)
    
    def _train_one_epoch(self, epoch: int) -> tuple:
        """
        训练一个epoch
        
        Args:
            epoch: 当前epoch编号
            
        Returns:
            平均训练损失和训练准确率
        """
        self.gin_encoder.train()
        self.classifier.train()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        start_time = time.time()
        
        for batch_idx, data in enumerate(self.train_loader):
            data = data.to(self.device)
            
            self.optimizer.zero_grad()
            
            graph_emb, _ = self.gin_encoder(data)
            logits = self.classifier(graph_emb)
            
            loss = self.criterion(logits, data.y)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
            
            if (batch_idx + 1) % self.config["log_interval"] == 0:
                logger.info(f"Epoch [{epoch}/{self.config['epochs']}], Batch [{batch_idx+1}/{len(self.train_loader)}], Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(self.train_loader)
        acc = accuracy_score(all_labels, all_preds)
        
        epoch_time = time.time() - start_time
        
        logger.info(f"Epoch [{epoch}/{self.config['epochs']}] 训练完成 - 平均损失: {avg_loss:.4f}, 准确率: {acc:.4f}, 耗时: {epoch_time:.2f}秒")
        
        return avg_loss, acc
    
    def _validate(self, epoch: int) -> tuple:
        """
        在验证集上评估模型
        
        Args:
            epoch: 当前epoch编号
            
        Returns:
            平均验证损失和验证准确率
        """
        self.gin_encoder.eval()
        self.classifier.eval()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data in self.val_loader:
                data = data.to(self.device)
                
                graph_emb, _ = self.gin_encoder(data)
                logits = self.classifier(graph_emb)
                
                loss = self.criterion(logits, data.y)
                total_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(data.y.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        acc = accuracy_score(all_labels, all_preds)
        
        logger.info(f"Epoch [{epoch}/{self.config['epochs']}] 验证完成 - 平均损失: {avg_loss:.4f}, 准确率: {acc:.4f}")
        
        if epoch % 10 == 0:
            logger.info("\n" + classification_report(all_labels, all_preds))
        
        return avg_loss, acc
    
    def _save_model(self, epoch: int, is_best: bool = False):
        """
        保存模型状态
        
        Args:
            epoch: 当前epoch编号
            is_best: 是否为目前最佳模型
        """
        model_state = {
            "epoch": epoch,
            "gin_encoder": self.gin_encoder.state_dict(),
            "classifier": self.classifier.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_val_acc": self.best_val_acc,
            "train_history": self.train_history
        }
        
        current_path = os.path.join(self.model_save_dir, f"model_epoch_{epoch}.pth")
        torch.save(model_state, current_path)
        logger.info(f"模型已保存至: {current_path}")
        
        if is_best:
            best_path = os.path.join(self.model_save_dir, "best_model.pth")
            torch.save(model_state, best_path)
            logger.info(f"最佳模型已更新并保存至: {best_path}")
    
    def _plot_training_history(self):
        """绘制训练历史曲线"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_history["train_loss"], label="训练损失")
        plt.plot(self.train_history["val_loss"], label="验证损失")
        plt.xlabel("Epoch")
        plt.ylabel("损失")
        plt.title("训练和验证损失")
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_history["train_acc"], label="训练准确率")
        plt.plot(self.train_history["val_acc"], label="验证准确率")
        plt.xlabel("Epoch")
        plt.ylabel("准确率")
        plt.title("训练和验证准确率")
        plt.legend()
        
        history_plot_path = os.path.join(self.log_dir, "training_history.png")
        plt.tight_layout()
        plt.savefig(history_plot_path)
        plt.close()
        
        logger.info(f"训练历史曲线已保存至: {history_plot_path}")
    
    def train(self):
        """执行完整的训练过程"""
        logger.info("开始模型训练...")
        start_time = time.time()
        
        try:
            for epoch in range(1, self.config["epochs"] + 1):
                train_loss, train_acc = self._train_one_epoch(epoch)
                
                val_loss, val_acc = self._validate(epoch)
                
                self.train_history["train_loss"].append(train_loss)
                self.train_history["train_acc"].append(train_acc)
                self.train_history["val_loss"].append(val_loss)
                self.train_history["val_acc"].append(val_acc)
                
                if self.scheduler is not None:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
                
                is_best = val_acc > self.best_val_acc
                if is_best:
                    self.best_val_acc = val_acc
                
                if epoch % self.config["save_interval"] == 0 or is_best:
                    self._save_model(epoch, is_best)
            
            self._save_model(self.config["epochs"], False)
            
            self._plot_training_history()
            
            history_path = os.path.join(self.log_dir, "training_history.json")
            with open(history_path, 'w') as f:
                json.dump(self.train_history, f, indent=2)
            
            total_time = time.time() - start_time
            logger.info(f"训练完成! 总耗时: {total_time:.2f}秒, 最佳验证准确率: {self.best_val_acc:.4f}")
            
        except KeyboardInterrupt:
            logger.warning("训练被手动中断")
            self._save_model(epoch, False)
            logger.info(f"已保存中断时的模型状态 (epoch {epoch})")

def main():
    parser = argparse.ArgumentParser(description='FGFR-Net模型训练器')
    parser.add_argument('--dataset', type=str, choices=['iscx', 'ustc'], 
                      required=True, help='要使用的数据集')
    parser.add_argument('--config', type=str, default='config/model_config.json',
                      help='模型配置文件路径')
    parser.add_argument('--epochs', type=int, help='训练轮次，覆盖配置文件中的设置')
    parser.add_argument('--batch_size', type=int, help='批次大小，覆盖配置文件中的设置')
    
    args = parser.parse_args()
    
    trainer = Trainer(args.dataset, args.config)
    
    if args.epochs:
        trainer.config["epochs"] = args.epochs
    if args.batch_size:
        trainer.config["batch_size"] = args.batch_size
    
    trainer.train()

if __name__ == "__main__":
    main()
