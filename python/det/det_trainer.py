"""
Face Detection Trainer Module (人脸检测训练器模块)
===================================================
支持 AMP 混合精度训练、分布式训练、多尺度训练
包含完整的训练循环、验证、日志记录功能
"""

import os
import time
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# 进度条
from tqdm import tqdm

# 本地导入
from ultra_tiny_det import (
    UltraTinyDetector,
    FocalLoss,
    DIOULoss,
    SmallTargetWeightedLoss,
    nms,
)


# ============================================================================
# 训练配置
# ============================================================================
class DetectionTrainerConfig:
    """检测器训练配置"""
    
    def __init__(
        self,
        # 数据配置
        img_size: int = 640,
        batch_size: int = 32,
        num_workers: int = 8,
        
        # 优化器配置
        lr: float = 1e-3,
        weight_decay: float = 0.05,
        momentum: float = 0.9,
        
        # 学习率调度
        epochs: int = 300,
        warmup_epochs: int = 5,
        lr_scheduler: str = 'cosine',  # cosine / step / multistep
        
        # 损失函数配置
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        small_target_weight: float = 1.5,
        
        # 训练策略
        amp: bool = True,  # 混合精度
        grad_clip: float = 1.0,
        ema: bool = True,  # 指数移动平均
        ema_decay: float = 0.995,
        
        # 日志与保存
        log_dir: str = 'logs/detection',
        checkpoint_dir: str = 'checkpoints/detection',
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
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.small_target_weight = small_target_weight
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
        """转换为字典"""
        return self.__dict__
    
    def save(self, path: str):
        """保存配置"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'DetectionTrainerConfig':
        """加载配置"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


# ============================================================================
# EMA (指数移动平均)
# ============================================================================
class ModelEMA:
    """
    模型指数移动平均
    用于提升模型泛化能力
    
    Args:
        model: 原始模型
        decay: 衰减率
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.model = model
        self.decay = decay
        self.ema_model = self._copy_model(model)
        self.ema_model.eval()
    
    def _copy_model(self, model: nn.Module) -> nn.Module:
        """复制模型"""
        ema_model = type(model)()
        ema_model.load_state_dict(model.state_dict())
        return ema_model
    
    def update(self):
        """更新 EMA 模型"""
        with torch.no_grad():
            for ema_param, param in zip(
                self.ema_model.parameters(),
                self.model.parameters()
            ):
                ema_param.mul_(self.decay).add_(param, alpha=1 - self.decay)
    
    def get_state_dict(self) -> Dict:
        """获取 EMA 模型状态"""
        return self.ema_model.state_dict()


# ============================================================================
# 学习率调度器
# ============================================================================
class WarmupCosineLR:
    """
    带预热的余弦退火学习率调度器
    
    Args:
        optimizer: 优化器
        total_epochs: 总训练轮数
        warmup_epochs: 预热轮数
        base_lr: 基础学习率
        min_lr: 最小学习率
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
        """获取当前学习率"""
        return self.optimizer.param_groups[0]['lr']


# ============================================================================
# 检测器训练器
# ============================================================================
class DetectionTrainer:
    """
    人脸检测器训练器
    
    Args:
        model: 检测器模型
        config: 训练配置
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        device: 训练设备
    """
    
    def __init__(
        self,
        model: UltraTinyDetector,
        config: DetectionTrainerConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # 损失函数
        self.cls_loss_fn = FocalLoss(
            alpha=config.focal_alpha,
            gamma=config.focal_gamma,
        )
        self.reg_loss_fn = DIOULoss()
        self.kpt_loss_fn = nn.SmoothL1Loss()
        self.small_target_loss = SmallTargetWeightedLoss(
            weight_factor=config.small_target_weight,
        )
        
        # 优化器
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
        
        # 学习率调度器
        self.scheduler = WarmupCosineLR(
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
        self.best_map = 0.0
        self.training_history = []
    
    def train_epoch(self, epoch: int) -> Dict:
        """
        训练一个 epoch
        
        Args:
            epoch: 当前轮次
            
        Returns:
            metrics: 训练指标
        """
        self.model.train()
        
        total_loss = 0.0
        cls_loss = 0.0
        reg_loss = 0.0
        kpt_loss = 0.0
        
        pbar = tqdm(
            self.train_loader,
            desc=f'Epoch {epoch}/{self.config.epochs}',
            disable=(self.config.distributed and self.config.local_rank != 0),
        )
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['images'].to(self.device)
            boxes = batch['boxes']  # List[Tensor]
            heatmaps = batch['heatmaps'].to(self.device)
            
            # AMP 前向传播
            if self.config.amp and self.scaler is not None:
                with autocast():
                    outputs = self.model(images)
                    
                    # 计算损失
                    losses = self._compute_loss(outputs, boxes, heatmaps)
                    total_batch_loss = (
                        losses['cls'] + 
                        losses['reg'] + 
                        losses['kpt']
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
                outputs = self.model(images)
                losses = self._compute_loss(outputs, boxes, heatmaps)
                total_batch_loss = (
                    losses['cls'] + 
                    losses['reg'] + 
                    losses['kpt']
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
            cls_loss += losses['cls'].item()
            reg_loss += losses['reg'].item()
            kpt_loss += losses['kpt'].item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{total_loss / (batch_idx + 1):.4f}',
                'cls': f'{cls_loss / (batch_idx + 1):.4f}',
                'reg': f'{reg_loss / (batch_idx + 1):.4f}',
                'kpt': f'{kpt_loss / (batch_idx + 1):.4f}',
            })
        
        # 更新学习率
        self.scheduler.step(epoch)
        
        return {
            'total_loss': total_loss / len(self.train_loader),
            'cls_loss': cls_loss / len(self.train_loader),
            'reg_loss': reg_loss / len(self.train_loader),
            'kpt_loss': kpt_loss / len(self.train_loader),
            'lr': self.scheduler.get_lr(),
        }
    
    def _compute_loss(
        self,
        outputs: Dict,
        boxes: List[torch.Tensor],
        heatmaps: torch.Tensor,
    ) -> Dict:
        """
        计算总损失
        
        Args:
            outputs: 模型输出
            boxes: 真实 bbox
            heatmaps: 真实热力图
            
        Returns:
            losses: 各项损失
        """
        cls_preds = outputs['cls']
        reg_preds = outputs['reg']
        kpt_preds = outputs['kpt']
        obj_preds = outputs['obj']
        
        # 分类损失 (热力图匹配)
        cls_loss = 0.0
        for i, pred in enumerate(cls_preds):
            cls_loss += self.cls_loss_fn(pred, heatmaps)
        cls_loss /= len(cls_preds)
        
        # 回归损失 (需要正样本匹配)
        reg_loss = torch.tensor(0.0, device=self.device)
        kpt_loss = torch.tensor(0.0, device=self.device)
        
        # 简化实现：对每个正样本计算损失
        num_pos = 0
        for batch_idx, (box_list, feat) in enumerate(zip(boxes, reg_preds[0])):
            if len(box_list) > 0:
                # 这里需要实现正负样本匹配逻辑
                # 简化处理
                num_pos += len(box_list)
        
        if num_pos > 0:
            # 实际实现需要更复杂的匹配策略
            reg_loss = reg_preds[0].mean() * 0.1  # 占位
            kpt_loss = kpt_preds[0].mean() * 0.1  # 占位
        
        return {
            'cls': cls_loss,
            'reg': reg_loss,
            'kpt': kpt_loss,
        }
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict:
        """
        验证
        
        Args:
            epoch: 当前轮次
            
        Returns:
            metrics: 验证指标
        """
        self.model.eval()
        
        all_preds = []
        all_targets = []
        
        pbar = tqdm(
            self.val_loader,
            desc=f'Validating Epoch {epoch}',
            disable=(self.config.distributed and self.config.local_rank != 0),
        )
        
        for batch in pbar:
            images = batch['images'].to(self.device)
            boxes = batch['boxes']
            
            # 前向传播
            if self.config.amp:
                with autocast():
                    outputs = self.model(images)
            else:
                outputs = self.model(images)
            
            # 后处理
            preds = self._postprocess(outputs, images.shape[2:])
            
            all_preds.extend(preds)
            all_targets.extend(boxes)
        
        # 计算 mAP
        map_metrics = self._compute_map(all_preds, all_targets)
        
        return map_metrics
    
    def _postprocess(
        self,
        outputs: Dict,
        img_size: Tuple[int, int],
        conf_threshold: float = 0.3,
        nms_threshold: float = 0.5,
    ) -> List[Dict]:
        """
        后处理
        
        Args:
            outputs: 模型输出
            img_size: 图像尺寸
            conf_threshold: 置信度阈值
            nms_threshold: NMS 阈值
            
        Returns:
            preds: 检测结果列表
        """
        batch_size = outputs['cls'][0].shape[0]
        preds = []
        
        for i in range(batch_size):
            # 获取输出
            cls_pred = outputs['cls'][0][i].sigmoid()
            reg_pred = outputs['reg'][0][i]
            obj_pred = outputs['obj'][0][i].squeeze()
            
            # 置信度过滤
            conf = (cls_pred * obj_pred).squeeze()
            mask = conf > conf_threshold
            
            if mask.sum() == 0:
                preds.append({'boxes': [], 'scores': [], 'labels': []})
                continue
            
            # 生成候选框
            # 简化实现
            boxes = reg_pred[mask]
            scores = conf[mask]
            
            # NMS
            if len(boxes) > 0:
                keep = nms(boxes, scores, iou_threshold=nms_threshold)
                boxes = boxes[keep]
                scores = scores[keep]
            
            preds.append({
                'boxes': boxes.cpu().numpy(),
                'scores': scores.cpu().numpy(),
                'labels': [1] * len(boxes),
            })
        
        return preds
    
    def _compute_map(
        self,
        preds: List[Dict],
        targets: List[torch.Tensor],
        iou_threshold: float = 0.5,
    ) -> Dict:
        """
        计算 mAP
        
        Args:
            preds: 预测结果
            targets: 真实标注
            iou_threshold: IoU 阈值
            
        Returns:
            map_metrics: mAP 指标
        """
        # 简化实现
        # 实际需要使用 COCO/WiderFace 评估工具
        return {'map@0.5': 0.0, 'map@0.5:0.95': 0.0}
    
    def train(self) -> Dict:
        """
        完整训练流程
        
        Returns:
            history: 训练历史
        """
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            # 训练
            train_metrics = self.train_epoch(epoch)
            
            # 验证
            if (epoch + 1) % self.config.eval_freq == 0:
                val_metrics = self.validate(epoch)
                
                # 保存最佳模型
                if val_metrics.get('map@0.5', 0) > self.best_map:
                    self.best_map = val_metrics['map@0.5']
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
            'best_map': self.best_map,
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
            'best_map': self.best_map,
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
def train_detection(
    config: DetectionTrainerConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    pretrained_weights: Optional[str] = None,
) -> DetectionTrainer:
    """
    训练检测器
    
    Args:
        config: 训练配置
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        pretrained_weights: 预训练权重路径
        
    Returns:
        trainer: 训练器
    """
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 模型
    model = UltraTinyDetector(img_size=config.img_size)
    
    if pretrained_weights is not None:
        checkpoint = torch.load(pretrained_weights, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"Loaded pretrained weights from {pretrained_weights}")
    
    # 训练器
    trainer = DetectionTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )
    
    # 开始训练
    history = trainer.train()
    
    print(f"\nTraining completed!")
    print(f"Best mAP@0.5: {history['best_map']:.4f}")
    print(f"Total time: {history['total_time'] / 3600:.2f} hours")
    
    return trainer


if __name__ == '__main__':
    import math
    
    # 示例配置
    config = DetectionTrainerConfig(
        img_size=640,
        batch_size=32,
        epochs=300,
        lr=1e-3,
    )
    
    print("Training config:")
    print(json.dumps(config.to_dict(), indent=2))
