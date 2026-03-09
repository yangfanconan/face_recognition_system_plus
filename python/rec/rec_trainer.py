"""
Face Recognition Trainer Module (人脸识别训练器模块)
=====================================================
支持 AMP 混合精度训练、分布式训练、在线三元组挖掘
包含完整的训练循环、验证、日志记录功能
"""

import os
import time
import json
import math
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

# 本地导入
from ultra_precise_rec import (
    UltraPreciseRecognizer,
    AdaArcV2,
    CenterLoss,
    ContrastiveLoss,
    DistillationLoss,
)
from rec_dataset import OnlineTripletMiner


# ============================================================================
# 训练配置
# ============================================================================
class RecognitionTrainerConfig:
    """识别器训练配置"""
    
    def __init__(
        self,
        # 数据配置
        img_size: int = 112,
        batch_size: int = 128,
        num_workers: int = 8,
        
        # 优化器配置
        lr: float = 0.1,
        weight_decay: float = 5e-4,
        momentum: float = 0.9,
        
        # 学习率调度
        epochs: int = 100,
        warmup_epochs: int = 5,
        lr_scheduler: str = 'cosine',
        
        # 损失函数配置
        arcface_margin: float = 0.5,
        arcface_scale: float = 64.0,
        center_loss_weight: float = 0.001,
        contrastive_margin: float = 1.0,
        distillation_temperature: float = 4.0,
        
        # 训练策略
        amp: bool = True,
        grad_clip: float = 1.0,
        ema: bool = True,
        ema_decay: float = 0.999,
        
        # 日志与保存
        log_dir: str = 'logs/recognition',
        checkpoint_dir: str = 'checkpoints/recognition',
        save_freq: int = 10,
        eval_freq: int = 5,
        
        # 分布式
        distributed: bool = False,
        local_rank: int = 0,
    ):
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.lr_scheduler = lr_scheduler
        self.arcface_margin = arcface_margin
        self.arcface_scale = arcface_scale
        self.center_loss_weight = center_loss_weight
        self.contrastive_margin = contrastive_margin
        self.distillation_temperature = distillation_temperature
        self.amp = amp
        self.grad_clip = grad_clip
        self.ema = ema
        self.ema_decay = ema_decay
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.save_freq = save_freq
        self.eval_freq = eval_freq
        self.distributed = distributed
        self.local_rank = local_rank
    
    def to_dict(self) -> Dict:
        return self.__dict__
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'RecognitionTrainerConfig':
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


# ============================================================================
# EMA (指数移动平均)
# ============================================================================
class ModelEMA:
    """模型指数移动平均"""
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.ema_model = self._copy_model(model)
        self.ema_model.eval()
        self.updates = 0
    
    def _copy_model(self, model: nn.Module) -> nn.Module:
        ema_model = type(model)()
        ema_model.load_state_dict(model.state_dict())
        return ema_model
    
    def update(self):
        """更新 EMA 模型"""
        self.updates += 1
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        
        with torch.no_grad():
            for ema_param, param in zip(
                self.ema_model.parameters(),
                self.model.parameters()
            ):
                ema_param.mul_(decay).add_(param, alpha=1 - decay)
    
    def get_state_dict(self) -> Dict:
        return self.ema_model.state_dict()


# ============================================================================
# 学习率调度器
# ============================================================================
class CosineAnnealingLR:
    """
    带预热的余弦退火学习率调度器
    """
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        total_epochs: int,
        warmup_epochs: int,
        base_lr: float,
        min_lr: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
    
    def step(self, epoch: int):
        """更新学习率"""
        if epoch < self.warmup_epochs:
            # 线性预热
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # 余弦退火
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']


# ============================================================================
# 识别器训练器
# ============================================================================
class RecognitionTrainer:
    """
    人脸识别器训练器
    
    Args:
        model: 识别器模型
        config: 训练配置
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        device: 训练设备
        num_classes: 类别数
    """
    
    def __init__(
        self,
        model: UltraPreciseRecognizer,
        config: RecognitionTrainerConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        num_classes: int,
    ):
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_classes = num_classes
        
        # 损失函数
        self.arcface_loss = AdaArcV2(
            embedding_size=512,
            num_classes=num_classes,
            margin=config.arcface_margin,
            scale=config.arcface_scale,
        )
        self.center_loss = CenterLoss(num_classes=num_classes, feature_dim=512)
        self.contrastive_loss = ContrastiveLoss(margin=config.contrastive_margin)
        self.distillation_loss = DistillationLoss(temperature=config.distillation_temperature)
        
        # 三元组挖掘
        self.triplet_miner = OnlineTripletMiner(margin=config.contrastive_margin)
        
        # 优化器
        self.optimizer = optim.SGD(
            [
                {'params': self.model.parameters(), 'lr': config.lr},
                {'params': self.arcface_loss.parameters(), 'lr': config.lr},
                {'params': self.center_loss.parameters(), 'lr': config.lr},
            ],
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
        
        # 学习率调度器
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            total_epochs=config.epochs,
            warmup_epochs=config.warmup_epochs,
            base_lr=config.lr,
        )
        
        # AMP
        self.scaler = GradScaler() if config.amp else None
        
        # EMA
        self.ema = ModelEMA(model, decay=config.ema_decay) if config.ema else None
        
        # 分布式
        if config.distributed:
            self.model = DDP(
                self.model,
                device_ids=[config.local_rank],
                find_unused_parameters=True,
            )
        
        # 日志
        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 训练统计
        self.best_accuracy = 0.0
        self.training_history = []
    
    def train_epoch(self, epoch: int) -> Dict:
        """训练一个 epoch"""
        self.model.train()
        
        total_loss = 0.0
        arcface_loss_sum = 0.0
        center_loss_sum = 0.0
        contrastive_loss_sum = 0.0
        
        pbar = tqdm(
            self.train_loader,
            desc=f'Epoch {epoch}/{self.config.epochs}',
            disable=(self.config.distributed and self.config.local_rank != 0),
        )
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['images'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # AMP 前向传播
            if self.config.amp and self.scaler is not None:
                with autocast():
                    # 提取特征
                    features = self.model(images)
                    id_features = features['id']
                    
                    # 计算损失
                    arcface_loss = self.arcface_loss(id_features, labels)
                    center_loss = self.center_loss(id_features, labels)
                    
                    # 三元组损失 (在线挖掘)
                    if batch_size := len(labels) >= 3:
                        anchor_idx, pos_idx, neg_idx = self.triplet_miner.mine(
                            id_features.detach(), labels
                        )
                        if len(anchor_idx) > 0:
                            contrastive_loss = self.contrastive_loss(
                                id_features[anchor_idx],
                                id_features[pos_idx],
                                id_features[neg_idx],
                            )
                        else:
                            contrastive_loss = torch.tensor(0.0, device=self.device)
                    else:
                        contrastive_loss = torch.tensor(0.0, device=self.device)
                    
                    # 总损失
                    total_batch_loss = (
                        arcface_loss +
                        self.config.center_loss_weight * center_loss +
                        contrastive_loss
                    )
                
                # 反向传播
                self.optimizer.zero_grad()
                self.scaler.scale(total_batch_loss).backward()
                
                # 梯度裁剪
                if self.config.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.grad_clip,
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # 非 AMP 训练
                features = self.model(images)
                id_features = features['id']
                
                arcface_loss = self.arcface_loss(id_features, labels)
                center_loss = self.center_loss(id_features, labels)
                
                total_batch_loss = (
                    arcface_loss +
                    self.config.center_loss_weight * center_loss
                )
                
                self.optimizer.zero_grad()
                total_batch_loss.backward()
                
                if self.config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.grad_clip,
                    )
                
                self.optimizer.step()
            
            # 更新 EMA
            if self.ema is not None:
                self.ema.update()
            
            # 累计损失
            total_loss += total_batch_loss.item()
            arcface_loss_sum += arcface_loss.item()
            center_loss_sum += center_loss.item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{total_loss / (batch_idx + 1):.4f}',
                'arcface': f'{arcface_loss_sum / (batch_idx + 1):.4f}',
                'center': f'{center_loss_sum / (batch_idx + 1):.4f}',
            })
        
        # 更新学习率
        self.scheduler.step(epoch)
        
        return {
            'total_loss': total_loss / len(self.train_loader),
            'arcface_loss': arcface_loss_sum / len(self.train_loader),
            'center_loss': center_loss_sum / len(self.train_loader),
            'lr': self.scheduler.get_lr(),
        }
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict:
        """验证"""
        self.model.eval()
        
        all_embeddings = []
        all_labels = []
        
        pbar = tqdm(
            self.val_loader,
            desc=f'Validating Epoch {epoch}',
            disable=(self.config.distributed and self.config.local_rank != 0),
        )
        
        for batch in pbar:
            images = batch['images'].to(self.device)
            labels = batch['labels']
            
            # 提取特征
            if self.config.amp:
                with autocast():
                    features = self.model(images)
            else:
                features = self.model(images)
            
            id_features = features['id']
            
            all_embeddings.append(id_features.cpu())
            all_labels.extend(labels.numpy())
        
        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_labels = np.array(all_labels)
        
        # 计算类内距离和类间距离
        intra_dist, inter_dist = self._compute_distances(all_embeddings, all_labels)
        
        return {
            'intra_distance': intra_dist,
            'inter_distance': inter_dist,
            'separability': inter_dist / (intra_dist + 1e-7),
        }
    
    def _compute_distances(
        self,
        embeddings: torch.Tensor,
        labels: np.ndarray,
    ) -> Tuple[float, float]:
        """计算类内距离和类间距离"""
        unique_labels = np.unique(labels)
        
        intra_distances = []
        inter_distances = []
        
        for label in unique_labels:
            mask = labels == label
            class_embeddings = embeddings[mask]
            
            if len(class_embeddings) > 1:
                # 类内距离
                for i in range(len(class_embeddings)):
                    for j in range(i + 1, len(class_embeddings)):
                        dist = F.pairwise_distance(
                            class_embeddings[i:i+1],
                            class_embeddings[j:j+1],
                        ).item()
                        intra_distances.append(dist)
        
        # 类间距离 (随机采样)
        for i in range(min(1000, len(embeddings))):
            for j in range(min(1000, len(embeddings))):
                if labels[i] != labels[j]:
                    dist = F.pairwise_distance(
                        embeddings[i:i+1],
                        embeddings[j:j+1],
                    ).item()
                    inter_distances.append(dist)
        
        return (
            np.mean(intra_distances) if intra_distances else 0,
            np.mean(inter_distances) if inter_distances else 0,
        )
    
    def train(self) -> Dict:
        """完整训练流程"""
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            # 训练
            train_metrics = self.train_epoch(epoch)
            
            # 验证
            if (epoch + 1) % self.config.eval_freq == 0:
                val_metrics = self.validate(epoch)
                
                # 保存最佳模型
                if val_metrics.get('separability', 0) > self.best_accuracy:
                    self.best_accuracy = val_metrics['separability']
                    self._save_checkpoint(epoch, 'best')
            
            # 保存检查点
            if (epoch + 1) % self.config.save_freq == 0:
                self._save_checkpoint(epoch, f'epoch_{epoch + 1}')
            
            # 记录历史
            self.training_history.append({
                'epoch': epoch,
                'train': train_metrics,
                'val': val_metrics if (epoch + 1) % self.config.eval_freq == 0 else {},
            })
            
            # 保存日志
            self._save_log()
        
        total_time = time.time() - start_time
        
        return {
            'best_separability': self.best_accuracy,
            'total_time': total_time,
            'history': self.training_history,
        }
    
    def _save_checkpoint(self, epoch: int, name: str):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'ema_state_dict': self.ema.get_state_dict() if self.ema else None,
            'arcface_loss_state_dict': self.arcface_loss.state_dict(),
            'center_loss_state_dict': self.center_loss.state_dict(),
            'best_separability': self.best_accuracy,
            'config': self.config.to_dict(),
        }
        
        save_path = self.checkpoint_dir / f'checkpoint_{name}.pth'
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved: {save_path}")
    
    def _save_log(self):
        """保存日志"""
        log_path = self.log_dir / 'training_log.json'
        with open(log_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)


# ============================================================================
# 训练入口函数
# ============================================================================
def train_recognition(
    config: RecognitionTrainerConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    pretrained_weights: Optional[str] = None,
) -> RecognitionTrainer:
    """
    训练识别器
    
    Args:
        config: 训练配置
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        num_classes: 类别数
        pretrained_weights: 预训练权重路径
        
    Returns:
        trainer: 训练器
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 模型
    model = UltraPreciseRecognizer()
    
    if pretrained_weights is not None:
        checkpoint = torch.load(pretrained_weights, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"Loaded pretrained weights from {pretrained_weights}")
    
    # 训练器
    trainer = RecognitionTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_classes=num_classes,
    )
    
    # 开始训练
    history = trainer.train()
    
    print(f"\nTraining completed!")
    print(f"Best separability: {history['best_separability']:.4f}")
    print(f"Total time: {history['total_time'] / 3600:.2f} hours")
    
    return trainer


if __name__ == '__main__':
    # 示例配置
    config = RecognitionTrainerConfig(
        img_size=112,
        batch_size=128,
        epochs=100,
        lr=0.1,
    )
    
    print("Training config:")
    print(json.dumps(config.to_dict(), indent=2))
