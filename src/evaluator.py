'''
Time: 2025.8.20
We will upload the comments for the code as soon as possible and update the document content so that you can better understand the code.
'''
import os
import json
import logging
import argparse
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch_geometric.loader import DataLoader as GeometricDataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, classification_report, confusion_matrix,
                             roc_curve, auc, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from feature_encoding.gin_encoder import GINEncoder
from classifier.resnet34 import ResNet34
from graph_construction.graph_builder import TrafficGraphDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Evaluator:
    """模型评估器，负责评估训练好的FGFR-Net模型性能"""
    
    def __init__(self, dataset_name: str, model_path: str, 
                 config_path: str = "config/model_config.json",
                 test_split: float = 0.2):
        """
        初始化评估器
        
        Args:
            dataset_name: 数据集名称，"iscx"或"ustc"
            model_path: 训练好的模型权重路径
            config_path: 模型配置文件路径
            test_split: 从完整数据集中划分测试集的比例（如果没有单独的测试集）
        """
        self.config = self._load_config(config_path)
        self.dataset_name = dataset_name
        self.model_path = model_path
        self.test_split = test_split
        
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.config["use_cuda"] else "cpu")
        logger.info(f"使用设备: {self.device}")
        
        self._create_result_directories()
        
        self._load_dataset()
        
        self._load_model()
        
        self.results = {}
    
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
    
    def _create_result_directories(self):
        """创建评估结果保存目录"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.result_dir = os.path.join(
            self.config["results_path"],
            self.dataset_name,
            f"eval_{timestamp}"
        )
        
        Path(self.result_dir).mkdir(parents=True, exist_ok=True)
        
        self.metrics_dir = os.path.join(self.result_dir, "metrics")
        self.confusion_dir = os.path.join(self.result_dir, "confusion_matrices")
        self.roc_dir = os.path.join(self.result_dir, "roc_curves")
        
        Path(self.metrics_dir).mkdir(parents=True, exist_ok=True)
        Path(self.confusion_dir).mkdir(parents=True, exist_ok=True)
        Path(self.roc_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"评估结果将保存至: {self.result_dir}")
    
    def _load_dataset(self):
        """加载并准备测试数据集"""
        graph_path = self.config["dataset"][f"{self.dataset_name}_graph_path"]
        if not os.path.exists(graph_path):
            logger.error(f"图数据路径不存在: {graph_path}")
            raise FileNotFoundError(f"图数据路径不存在: {graph_path}")
        
        logger.info(f"加载{dataset_name}数据集...")
        full_dataset = TrafficGraphDataset(root=graph_path)
        
        if f"{self.dataset_name}_test_path" in self.config["dataset"]:
            test_graph_path = self.config["dataset"][f"{self.dataset_name}_test_path"]
            if os.path.exists(test_graph_path):
                self.test_dataset = TrafficGraphDataset(root=test_graph_path)
                logger.info(f"使用单独的测试集，样本数量: {len(self.test_dataset)}")
            else:
                logger.warning(f"指定的测试集路径不存在: {test_graph_path}，将从完整数据集中划分")
                self._split_test_dataset(full_dataset)
        else:
            self._split_test_dataset(full_dataset)
        
        self.test_loader = GeometricDataLoader(
            self.test_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["num_workers"]
        )
        
        self._get_class_info()
    
    def _split_test_dataset(self, full_dataset):
        """从完整数据集中划分测试集"""
        _, test_indices = train_test_split(
            range(len(full_dataset)),
            test_size=self.test_split,
            random_state=self.config["random_seed"],
            shuffle=True
        )
        self.test_dataset = Subset(full_dataset, test_indices)
        logger.info(f"从完整数据集中划分测试集 - 测试样本: {len(self.test_dataset)}")
    
    def _get_class_info(self):
        """获取类别信息，包括类别数量和名称（如果有）"""
        labels = []
        for idx in range(len(self.test_dataset)):
            if isinstance(self.test_dataset, Subset):
                data = self.test_dataset.dataset[self.test_dataset.indices[idx]]
            else:
                data = self.test_dataset[idx]
            labels.append(data.y.item())
        
        self.num_classes = len(np.unique(labels))
        self.class_names = [str(i) for i in range(self.num_classes)]
        
        if "class_names" in self.config["dataset"] and len(self.config["dataset"]["class_names"]) == self.num_classes:
            self.class_names = self.config["dataset"]["class_names"]
        
        logger.info(f"测试集类别数量: {self.num_classes}, 类别: {self.class_names}")
    
    def _load_model(self):
        """加载训练好的模型"""
        if not os.path.exists(self.model_path):
            logger.error(f"模型文件不存在: {self.model_path}")
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        logger.info(f"加载模型: {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
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
        
        self.gin_encoder.load_state_dict(checkpoint["gin_encoder"])
        self.classifier.load_state_dict(checkpoint["classifier"])
        
        self.gin_encoder.eval()
        self.classifier.eval()
        
        logger.info("模型加载完成")
    
    def _run_inference(self) -> tuple:
        """
        在测试集上运行推理
        
        Returns:
            真实标签、预测标签、预测概率
        """
        logger.info("开始在测试集上进行推理...")
        
        all_labels = []
        all_preds = []
        all_probs = []
        
        start_time = time.time()
        total_samples = 0
        
        with torch.no_grad():
            for batch_idx, data in enumerate(self.test_loader):
                data = data.to(self.device)
                batch_size = data.num_graphs
                
                graph_emb, _ = self.gin_encoder(data)
                logits = self.classifier(graph_emb)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_labels.extend(data.y.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
                total_samples += batch_size
                
                if (batch_idx + 1) % self.config["log_interval"] == 0:
                    logger.info(f"处理批次 [{batch_idx+1}/{len(self.test_loader)}], 已处理样本: {total_samples}")
        
        inference_time = time.time() - start_time
        logger.info(f"推理完成 - 总样本: {total_samples}, 耗时: {inference_time:.2f}秒, 平均速度: {total_samples/inference_time:.2f}样本/秒")
        
        return np.array(all_labels), np.array(all_preds), np.array(all_probs)
    
    def _calculate_metrics(self, labels: np.ndarray, preds: np.ndarray, probs: np.ndarray):
        """
        计算评估指标
        
        Args:
            labels: 真实标签
            preds: 预测标签
            probs: 预测概率
        """
        logger.info("计算评估指标...")
        
        self.results["overall"] = {
            "accuracy": accuracy_score(labels, preds),
            "macro_precision": precision_score(labels, preds, average='macro'),
            "macro_recall": recall_score(labels, preds, average='macro'),
            "macro_f1": f1_score(labels, preds, average='macro'),
            "weighted_precision": precision_score(labels, preds, average='weighted'),
            "weighted_recall": recall_score(labels, preds, average='weighted'),
            "weighted_f1": f1_score(labels, preds, average='weighted')
        }
        
        self.results["per_class"] = {
            "precision": precision_score(labels, preds, average=None).tolist(),
            "recall": recall_score(labels, preds, average=None).tolist(),
            "f1": f1_score(labels, preds, average=None).tolist()
        }
        
        self.results["classification_report"] = classification_report(
            labels, preds, target_names=self.class_names, output_dict=True
        )
        
        if self.num_classes == 2:
            self.results["roc_auc"] = roc_auc_score(labels, probs[:, 1])
        else:
            self.results["roc_auc_ovo"] = roc_auc_score(
                labels, probs, average='macro', multi_class='ovo'
            )
        
        logger.info("\n整体评估指标:")
        for metric, value in self.results["overall"].items():
            logger.info(f"  {metric}: {value:.4f}")
    
    def _plot_confusion_matrix(self, labels: np.ndarray, preds: np.ndarray):
        """
        绘制混淆矩阵
        
        Args:
            labels: 真实标签
            preds: 预测标签
        """
        logger.info("绘制混淆矩阵...")
        
        cm = confusion_matrix(labels, preds)
        
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names)
        plt.xlabel('预测类别')
        plt.ylabel('真实类别')
        plt.title('混淆矩阵（数量）')
        
        cm_count_path = os.path.join(self.confusion_dir, 'confusion_matrix_count.png')
        plt.tight_layout()
        plt.savefig(cm_count_path)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names)
        plt.xlabel('预测类别')
        plt.ylabel('真实类别')
        plt.title('混淆矩阵（归一化）')
        
        cm_norm_path = os.path.join(self.confusion_dir, 'confusion_matrix_normalized.png')
        plt.tight_layout()
        plt.savefig(cm_norm_path)
        
        plt.close('all')
        
        logger.info(f"混淆矩阵已保存至: {self.confusion_dir}")
        
        self.results["confusion_matrix"] = {
            "count": cm.tolist(),
            "normalized": cm_normalized.tolist()
        }
    
    def _plot_roc_curves(self, labels: np.ndarray, probs: np.ndarray):
        """
        绘制ROC曲线（仅对二分类或多分类）
        
        Args:
            labels: 真实标签
            probs: 预测概率
        """
        if self.num_classes == 1:
            logger.warning("单类别问题不支持ROC曲线")
            return
        
        logger.info("绘制ROC曲线...")
        
        if self.num_classes == 2:
            fpr, tpr, _ = roc_curve(labels, probs[:, 1])
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC曲线 (面积 = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('假正例率 (FPR)')
            plt.ylabel('真正例率 (TPR)')
            plt.title('ROC曲线')
            plt.legend(loc="lower right")
            
            roc_path = os.path.join(self.roc_dir, 'roc_curve.png')
            plt.tight_layout()
            plt.savefig(roc_path)
            
            self.results["roc_curve"] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "auc": float(roc_auc)
            }
        
        else:
            plt.figure(figsize=(10, 8))
            
            encoder = OneHotEncoder(sparse=False)
            labels_onehot = encoder.fit_transform(labels.reshape(-1, 1))
            
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            for i in range(self.num_classes):
                fpr[i], tpr[i], _ = roc_curve(labels_onehot[:, i], probs[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
                plt.plot(fpr[i], tpr[i], lw=2, 
                        label=f'类别 {self.class_names[i]} (面积 = {roc_auc[i]:.2f})')
            
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('假正例率 (FPR)')
            plt.ylabel('真正例率 (TPR)')
            plt.title('多类别ROC曲线 (One-vs-Rest)')
            plt.legend(loc="lower right")
            
            roc_path = os.path.join(self.roc_dir, 'multiclass_roc_curve.png')
            plt.tight_layout()
            plt.savefig(roc_path)
            
            self.results["roc_curve"] = {
                "fpr": {str(i): fpr[i].tolist() for i in range(self.num_classes)},
                "tpr": {str(i): tpr[i].tolist() for i in range(self.num_classes)},
                "auc": {str(i): float(roc_auc[i]) for i in range(self.num_classes)}
            }
        
        plt.close('all')
        logger.info(f"ROC曲线已保存至: {self.roc_dir}")
    
    def _save_results(self):
        """保存评估结果到文件"""
        metrics_path = os.path.join(self.metrics_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        report_path = os.path.join(self.metrics_dir, 'classification_report.txt')
        with open(report_path, 'w') as f:
            f.write(classification_report(
                self.results["classification_report"], 
                target_names=self.class_names
            ))
        
        logger.info(f"评估指标已保存至: {self.metrics_dir}")
    
    def evaluate(self):
        """执行完整的评估流程"""
        logger.info("开始模型评估...")
        
        labels, preds, probs = self._run_inference()
        
        self._calculate_metrics(labels, preds, probs)
        
        self._plot_confusion_matrix(labels, preds)
        
        self._plot_roc_curves(labels, probs)
        
        self._save_results()
        
        logger.info(f"评估完成! 所有结果已保存至: {self.result_dir}")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='FGFR-Net模型评估器')
    parser.add_argument('--dataset', type=str, choices=['iscx', 'ustc'], 
                      required=True, help='要使用的数据集')
    parser.add_argument('--model', type=str, required=True, 
                      help='训练好的模型文件路径')
    parser.add_argument('--config', type=str, default='config/model_config.json',
                      help='模型配置文件路径')
    parser.add_argument('--test_split', type=float, default=0.2,
                      help='测试集划分比例（如果没有单独的测试集）')
    
    args = parser.parse_args()
    
    evaluator = Evaluator(
        dataset_name=args.dataset,
        model_path=args.model,
        config_path=args.config,
        test_split=args.test_split
    )
    
    evaluator.evaluate()

if __name__ == "__main__":
    main()
