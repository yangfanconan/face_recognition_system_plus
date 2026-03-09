"""
Face Detection Dataset Module (人脸检测数据集模块)
===================================================
支持 WiderFace、FDDB 等主流数据集
提供数据增强、小目标增强、混合精度训练支持
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ============================================================================
# 数据增强策略
# ============================================================================
def get_train_transforms(
    img_size: int = 640,
    small_target_enhance: bool = True,
) -> A.Compose:
    """
    获取训练数据增强
    
    Args:
        img_size: 目标图像大小
        small_target_enhance: 是否启用小目标增强
        
    Returns:
        transforms: 数据增强组合
    """
    # 基础增强
    base_transforms = [
        A.RandomResizedCrop(
            height=img_size,
            width=img_size,
            scale=(0.6, 1.0),
            ratio=(0.8, 1.25),
            p=0.8,
        ),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=15,
            p=0.5,
        ),
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
            p=0.5,
        ),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0)),
            A.ISONoise(),
            A.MotionBlur(blur_limit=3),
        ], p=0.3),
    ]
    
    # 小目标增强
    if small_target_enhance:
        base_transforms.append(
            A.OneOf([
                A.Perspective(scale=(0.05, 0.1)),  # 模拟远距离小目标
                A.ZoomBlur(max_factor=1.2),
            ], p=0.3)
        )
    
    base_transforms.extend([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])
    
    return A.Compose(base_transforms, bbox_params=A.BboxParams(
        format='xyxy',
        min_area=16,  # 最小 16x16 像素
        min_visibility=0.3,
        label_fields=['labels'],
    ))


def get_val_transforms(img_size: int = 640) -> A.Compose:
    """
    获取验证数据增强
    
    Args:
        img_size: 目标图像大小
        
    Returns:
        transforms: 数据增强组合
    """
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format='xyxy',
        min_area=16,
        label_fields=['labels'],
    ))


# ============================================================================
# WiderFace 数据集
# ============================================================================
class WiderFaceDataset(Dataset):
    """
    WiderFace 人脸检测数据集
    
    Args:
        root_dir: 数据集根目录
        split: 数据集划分 (train/val/test)
        img_size: 图像大小
        transforms: 数据增强
        min_face_size: 最小人脸尺寸
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        img_size: int = 640,
        transforms: Optional[Callable] = None,
        min_face_size: int = 16,
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        self.img_size = img_size
        self.transforms = transforms
        self.min_face_size = min_face_size
        
        # 加载标注
        self.images = []
        self.annotations = []
        self._load_annotations()
    
    def _load_annotations(self):
        """加载标注文件"""
        if self.split == 'train':
            anno_path = self.root_dir / 'wider_face_split' / 'wider_face_train_bbx_gt.txt'
            image_dir = self.root_dir / 'WIDER_train' / 'images'
        elif self.split == 'val':
            anno_path = self.root_dir / 'wider_face_split' / 'wider_face_val_bbx_gt.txt'
            image_dir = self.root_dir / 'WIDER_val' / 'images'
        else:
            raise ValueError(f"Unsupported split: {self.split}")
        
        with open(anno_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            # 读取图像路径
            image_path = lines[i].strip()
            i += 1
            
            # 读取人脸数量
            num_faces = int(lines[i].strip())
            i += 1
            
            # 读取 bbox 和关键点
            boxes = []
            landmarks = []
            for _ in range(num_faces):
                parts = lines[i].strip().split()
                i += 1
                
                x, y, w, h = map(int, parts[:4])
                
                # 过滤过小人脸
                if w >= self.min_face_size and h >= self.min_face_size:
                    boxes.append([x, y, x + w, y + h])
                    
                    # 关键点 (如果有)
                    if len(parts) >= 14:
                        kpt = []
                        for j in range(5):
                            kpt_x = int(parts[4 + j * 2])
                            kpt_y = int(parts[4 + j * 2 + 1])
                            kpt.append([kpt_x, kpt_y])
                        landmarks.append(kpt)
                    else:
                        landmarks.append([[0, 0]] * 5)  # 占位
            
            if len(boxes) > 0:
                self.images.append(image_dir / image_path)
                self.annotations.append({
                    'boxes': boxes,
                    'landmarks': landmarks,
                    'labels': [1] * len(boxes),  # 人脸类别
                })
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            sample: 样本字典
        """
        # 读取图像
        image_path = str(self.images[idx])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 获取标注
        anno = self.annotations[idx]
        boxes = np.array(anno['boxes'], dtype=np.float32)
        labels = np.array(anno['labels'], dtype=np.int64)
        landmarks = np.array(anno['landmarks'], dtype=np.float32)
        
        # 数据增强
        if self.transforms is not None:
            transformed = self.transforms(
                image=image,
                bboxes=boxes,
                labels=labels,
            )
            image = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['labels']
        
        # 转换为 tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # 生成热力图目标 (用于 Anchor-Free 检测)
        h, w = image.shape[1], image.shape[2]
        heatmap_size = h // 4  # P1 特征图尺寸
        heatmap = self._generate_heatmap(boxes, heatmap_size)
        
        return {
            'image': image,
            'boxes': boxes,
            'labels': labels,
            'heatmap': heatmap,
            'image_path': image_path,
        }
    
    def _generate_heatmap(
        self,
        boxes: np.ndarray,
        heatmap_size: int,
        sigma: float = 2.0,
    ) -> torch.Tensor:
        """
        生成高斯热力图
        
        Args:
            boxes: bbox [N, 4]
            heatmap_size: 热力图尺寸
            sigma: 高斯核标准差
            
        Returns:
            heatmap: 热力图 [1, H, W]
        """
        heatmap = np.zeros((heatmap_size, heatmap_size), dtype=np.float32)
        
        scale_x = heatmap_size / self.img_size
        scale_y = heatmap_size / self.img_size
        
        for box in boxes:
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2 * scale_x
            cy = (y1 + y2) / 2 * scale_y
            w = (x2 - x1) * scale_x
            h = (y2 - y1) * scale_y
            
            # 自适应 sigma
            radius = max(w, h) / 4
            radius = max(1, int(radius))
            
            # 生成高斯分布
            y, x = np.ogrid[:heatmap_size, :heatmap_size]
            dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            gaussian = np.exp(-dist ** 2 / (2 * sigma ** 2))
            
            # 取最大值 (多个人脸重叠时)
            heatmap = np.maximum(heatmap, gaussian)
        
        return torch.from_numpy(heatmap).unsqueeze(0)


# ============================================================================
# FDDB 数据集 (简化版)
# ============================================================================
class FDBBDataset(Dataset):
    """
    FDDB 人脸检测数据集
    
    Args:
        root_dir: 数据集根目录
        img_size: 图像大小
        transforms: 数据增强
    """
    
    def __init__(
        self,
        root_dir: str,
        img_size: int = 640,
        transforms: Optional[Callable] = None,
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.img_size = img_size
        self.transforms = transforms
        
        self.images = []
        self.annotations = []
        self._load_annotations()
    
    def _load_annotations(self):
        """加载 FDDB 标注"""
        # FDDB 标注格式较复杂，这里简化处理
        # 实际使用需要解析 ellipseList.txt
        pass
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Dict:
        # 类似 WiderFace 实现
        pass


# ============================================================================
# 数据加载器
# ============================================================================
def build_detection_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 8,
    distributed: bool = False,
    pin_memory: bool = True,
) -> DataLoader:
    """
    构建检测数据加载器
    
    Args:
        dataset: 数据集
        batch_size: 批次大小
        num_workers: 工作线程数
        distributed: 是否分布式训练
        pin_memory: 是否锁定内存
        
    Returns:
        dataloader: 数据加载器
    """
    sampler = None
    shuffle = True
    
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        shuffle = False
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=detection_collate_fn,
    )
    
    return dataloader


def detection_collate_fn(batch: List[Dict]) -> Dict:
    """
    检测任务 collate 函数
    
    Args:
        batch: 批次数据
        
    Returns:
        batched: 批量化数据
    """
    images = torch.stack([item['image'] for item in batch])
    boxes = [item['boxes'] for item in batch]
    labels = [item['labels'] for item in batch]
    heatmaps = torch.stack([item['heatmap'] for item in batch])
    image_paths = [item['image_path'] for item in batch]
    
    return {
        'images': images,
        'boxes': boxes,
        'labels': labels,
        'heatmaps': heatmaps,
        'image_paths': image_paths,
    }


# ============================================================================
# 数据集统计
# ============================================================================
def analyze_dataset(dataset: Dataset) -> Dict:
    """
    分析数据集统计信息
    
    Args:
        dataset: 数据集
        
    Returns:
        stats: 统计信息字典
    """
    total_images = len(dataset)
    total_faces = 0
    face_sizes = []
    
    for i in range(len(dataset)):
        anno = dataset.annotations[i]
        boxes = anno['boxes']
        total_faces += len(boxes)
        
        for box in boxes:
            w, h = box[2] - box[0], box[3] - box[1]
            face_sizes.append(min(w, h))
    
    face_sizes = np.array(face_sizes)
    
    stats = {
        'total_images': total_images,
        'total_faces': total_faces,
        'avg_faces_per_image': total_faces / total_images,
        'min_face_size': int(face_sizes.min()),
        'max_face_size': int(face_sizes.max()),
        'median_face_size': int(np.median(face_sizes)),
        'tiny_faces_ratio': float((face_sizes < 32).sum() / len(face_sizes)),
    }
    
    return stats


if __name__ == '__main__':
    # 测试数据集
    from det_dataset import WiderFaceDataset, get_train_transforms
    
    transforms = get_train_transforms(img_size=640)
    dataset = WiderFaceDataset(
        root_dir='/data/WiderFace',
        split='train',
        img_size=640,
        transforms=transforms,
    )
    
    # 分析数据集
    stats = analyze_dataset(dataset)
    print("Dataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 测试数据加载
    sample = dataset[0]
    print(f"\nSample image shape: {sample['image'].shape}")
    print(f"Number of faces: {len(sample['boxes'])}")
