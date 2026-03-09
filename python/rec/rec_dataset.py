"""
Face Recognition Dataset Module (人脸识别数据集模块)
=====================================================
支持 CASIA-WebFace、MS1M-V2、VGGFace2 等主流数据集
提供在线三元组挖掘、混合精度训练支持
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ============================================================================
# 数据增强策略
# ============================================================================
def get_train_transforms(
    img_size: int = 112,
    use_cutout: bool = True,
    use_blur: bool = True,
) -> A.Compose:
    """
    获取训练数据增强
    
    Args:
        img_size: 目标图像大小
        use_cutout: 是否使用 Cutout
        use_blur: 是否使用模糊增强
        
    Returns:
        transforms: 数据增强组合
    """
    augmentations = [
        A.RandomResizedCrop(
            height=img_size,
            width=img_size,
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
            p=0.8,
        ),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
            p=0.5,
        ),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 30.0)),
            A.ISONoise(),
        ], p=0.2),
    ]
    
    if use_cutout:
        augmentations.append(
            A.CoarseDropout(
                max_holes=4,
                max_height=img_size // 4,
                max_width=img_size // 4,
                p=0.3,
            )
        )
    
    if use_blur:
        augmentations.append(
            A.OneOf([
                A.MotionBlur(blur_limit=5),
                A.GaussianBlur(blur_limit=5),
                A.MedianBlur(blur_limit=5),
            ], p=0.2)
        )
    
    augmentations.extend([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
        ToTensorV2(),
    ])
    
    return A.Compose(augmentations)


def get_val_transforms(img_size: int = 112) -> A.Compose:
    """获取验证数据增强"""
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
        ToTensorV2(),
    ])


# ============================================================================
# 人脸识别数据集基类
# ============================================================================
class FaceRecognitionDataset(Dataset):
    """
    人脸识别数据集基类
    
    Args:
        root_dir: 数据集根目录
        transforms: 数据增强
    """
    
    def __init__(
        self,
        root_dir: str,
        transforms: Optional[Callable] = None,
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.transforms = transforms
        
        # 图像路径和标签
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        
        # 加载数据
        self._load_data()
    
    def _load_data(self):
        """加载数据"""
        raise NotImplementedError
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict:
        """获取单个样本"""
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 读取图像
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 数据增强
        if self.transforms is not None:
            image = self.transforms(image=image)['image']
        
        return {
            'image': image,
            'label': label,
            'image_path': str(image_path),
        }


# ============================================================================
# CASIA-WebFace 数据集
# ============================================================================
class CASIAWebFaceDataset(FaceRecognitionDataset):
    """
    CASIA-WebFace 数据集
    
    Args:
        root_dir: 数据集根目录
        transforms: 数据增强
        min_samples_per_class: 每类最小样本数
    """
    
    def __init__(
        self,
        root_dir: str,
        transforms: Optional[Callable] = None,
        min_samples_per_class: int = 10,
    ):
        self.min_samples_per_class = min_samples_per_class
        super().__init__(root_dir, transforms)
    
    def _load_data(self):
        """加载 CASIA-WebFace 数据"""
        # CASIA-WebFace 目录结构:
        # root/
        #   0000000/
        #     000.jpg
        #     001.jpg
        #   ...
        
        class_idx = 0
        valid_classes = []
        
        for class_dir in sorted(self.root_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            
            images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            
            # 过滤样本数过少的类
            if len(images) < self.min_samples_per_class:
                continue
            
            valid_classes.append(class_dir.name)
            self.class_to_idx[class_dir.name] = class_idx
            
            for image_path in images:
                self.image_paths.append(image_path)
                self.labels.append(class_idx)
            
            class_idx += 1
        
        print(f"Loaded CASIA-WebFace: {len(self.image_paths)} images, {class_idx} classes")


# ============================================================================
# MS1M-V2 数据集
# ============================================================================
class MS1MDataset(FaceRecognitionDataset):
    """
    MS1M-V2 数据集
    
    Args:
        root_dir: 数据集根目录
        transforms: 数据增强
    """
    
    def _load_data(self):
        """加载 MS1M 数据"""
        # MS1M 通常使用 .lst 文件或 pickle 文件
        # 这里简化处理，实际需要根据具体格式解析
        pass


# ============================================================================
# 平衡批次采样器 (用于训练)
# ============================================================================
class BalancedBatchSampler(Sampler):
    """
    平衡批次采样器
    每个批次包含 P 个类别，每个类别 K 个样本
    
    Args:
        labels: 所有样本的标签
        n_classes: 类别数
        n_samples_per_class: 每类样本数
        batch_size: 批次大小
    """
    
    def __init__(
        self,
        labels: List[int],
        n_classes: int,
        n_samples_per_class: int,
        batch_size: int,
    ):
        self.labels = labels
        self.n_classes = n_classes
        self.n_samples_per_class = n_samples_per_class
        self.batch_size = batch_size
        
        # 构建标签到索引的映射
        self.labels_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.labels_to_indices[label].append(idx)
        
        # 验证参数
        assert batch_size % n_samples_per_class == 0
        self.n_classes_per_batch = batch_size // n_samples_per_class
    
    def __iter__(self):
        """生成批次索引"""
        batch = []
        
        # 随机打乱每个类的索引
        for label in self.labels_to_indices:
            np.random.shuffle(self.labels_to_indices[label])
        
        # 生成批次
        while True:
            # 随机选择类别
            selected_classes = np.random.choice(
                list(self.labels_to_indices.keys()),
                size=self.n_classes_per_batch,
                replace=False,
            )
            
            # 从每个类中选择样本
            for label in selected_classes:
                indices = self.labels_to_indices[label]
                if len(indices) < self.n_samples_per_class:
                    # 样本不足时重复采样
                    selected_indices = np.random.choice(indices, self.n_samples_per_class, replace=True)
                else:
                    selected_indices = indices[:self.n_samples_per_class]
                
                batch.extend(selected_indices)
            
            if len(batch) >= self.batch_size:
                yield batch[:self.batch_size]
                batch = batch[self.batch_size:]
    
    def __len__(self) -> int:
        return len(self.labels) // self.batch_size


# ============================================================================
# 在线三元组挖掘
# ============================================================================
class OnlineTripletMiner:
    """
    在线三元组挖掘
    
    Args:
        margin: 边界
        mining_strategy: 挖掘策略 (hard/semi-hard/easy)
    """
    
    def __init__(self, margin: float = 0.3, mining_strategy: str = 'hard'):
        self.margin = margin
        self.mining_strategy = mining_strategy
    
    def mine(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        挖掘三元组
        
        Args:
            embeddings: 嵌入特征 [B, D]
            labels: 标签 [B]
            
        Returns:
            anchor_indices: 锚点索引
            positive_indices: 正样本索引
            negative_indices: 负样本索引
        """
        # 计算距离矩阵
        distances = self._pairwise_distance(embeddings)
        
        if self.mining_strategy == 'hard':
            return self._mine_hard_triplets(distances, labels)
        elif self.mining_strategy == 'semi-hard':
            return self._mine_semi_hard_triplets(distances, labels)
        else:
            return self._mine_random_triplets(distances, labels)
    
    def _pairwise_distance(self, embeddings: torch.Tensor) -> torch.Tensor:
        """计算成对距离"""
        # 余弦距离
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        similarity = embeddings_norm @ embeddings_norm.t()
        distance = 1 - similarity
        return distance
    
    def _mine_hard_triplets(
        self,
        distances: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """挖掘困难三元组"""
        n = len(labels)
        
        # 正样本对掩码
        pos_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        
        # 负样本对掩码
        neg_mask = ~pos_mask
        
        # 对于每个锚点，选择最远的正样本
        pos_distances = distances.masked_fill(pos_mask, -1e12)
        hardest_pos = pos_distances.argmax(dim=1)
        
        # 对于每个锚点，选择最近的负样本
        neg_distances = distances.masked_fill(neg_mask, 1e12)
        hardest_neg = neg_distances.argmin(dim=1)
        
        # 构建三元组
        anchor_indices = torch.arange(n, device=distances.device)
        positive_indices = hardest_pos
        negative_indices = hardest_neg
        
        # 过滤有效三元组
        valid_mask = (
            distances[anchor_indices, positive_indices] <
            distances[anchor_indices, negative_indices] - self.margin
        )
        
        return (
            anchor_indices[valid_mask],
            positive_indices[valid_mask],
            negative_indices[valid_mask],
        )
    
    def _mine_semi_hard_triplets(
        self,
        distances: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """挖掘半困难三元组"""
        # 类似困难三元组，但负样本距离在 [margin, 1] 范围内
        pass
    
    def _mine_random_triplets(
        self,
        distances: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """随机挖掘三元组"""
        n = len(labels)
        anchor_indices = torch.arange(n, device=distances.device)
        
        # 随机选择正负样本
        positive_indices = torch.randint(0, n, (n,), device=distances.device)
        negative_indices = torch.randint(0, n, (n,), device=distances.device)
        
        return anchor_indices, positive_indices, negative_indices


# ============================================================================
# 数据加载器构建
# ============================================================================
def build_recognition_dataloader(
    dataset: Dataset,
    batch_size: int = 128,
    num_workers: int = 8,
    distributed: bool = False,
    use_balanced_sampler: bool = True,
) -> DataLoader:
    """
    构建识别数据加载器
    
    Args:
        dataset: 数据集
        batch_size: 批次大小
        num_workers: 工作线程数
        distributed: 是否分布式
        use_balanced_sampler: 是否使用平衡采样器
        
    Returns:
        dataloader: 数据加载器
    """
    sampler = None
    shuffle = True
    
    if distributed:
        sampler = DistributedSampler(dataset)
        shuffle = False
    elif use_balanced_sampler and hasattr(dataset, 'labels'):
        n_classes = len(set(dataset.labels))
        sampler = BalancedBatchSampler(
            labels=dataset.labels,
            n_classes=n_classes,
            n_samples_per_class=4,
            batch_size=batch_size,
        )
        shuffle = False
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=recognition_collate_fn,
    )
    
    return dataloader


def recognition_collate_fn(batch: List[Dict]) -> Dict:
    """识别任务 collate 函数"""
    images = torch.stack([item['image'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    image_paths = [item['image_path'] for item in batch]
    
    return {
        'images': images,
        'labels': labels,
        'image_paths': image_paths,
    }


# ============================================================================
# 评估工具
# ============================================================================
def evaluate_on_lfw(
    model: torch.nn.Module,
    lfw_pairs_path: str,
    img_size: int = 112,
    batch_size: int = 128,
) -> Dict:
    """
    在 LFW 上评估
    
    Args:
        model: 模型
        lfw_pairs_path: LFW pairs.txt 路径
        img_size: 图像大小
        batch_size: 批次大小
        
    Returns:
        metrics: 评估指标
    """
    model.eval()
    
    # 加载 pairs.txt
    pairs = load_lfw_pairs(lfw_pairs_path)
    
    embeddings1 = []
    embeddings2 = []
    labels = []
    
    with torch.no_grad():
        for img1_path, img2_path, label in pairs:
            # 加载图像
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)
            
            # 预处理
            transform = get_val_transforms(img_size)
            img1 = transform(image=img1)['image'].unsqueeze(0)
            img2 = transform(image=img2)['image'].unsqueeze(0)
            
            # 提取特征
            feat1 = model.extract_id_feature(img1.cuda())
            feat2 = model.extract_id_feature(img2.cuda())
            
            embeddings1.append(feat1.cpu())
            embeddings2.append(feat2.cpu())
            labels.append(label)
    
    # 计算相似度
    embeddings1 = torch.cat(embeddings1, dim=0)
    embeddings2 = torch.cat(embeddings2, dim=0)
    
    similarities = F.cosine_similarity(embeddings1, embeddings2)
    
    # 计算准确率
    thresholds = np.linspace(0, 1, 100)
    best_accuracy = 0
    best_threshold = 0
    
    for threshold in thresholds:
        predictions = (similarities.numpy() > threshold).astype(int)
        accuracy = (predictions == np.array(labels)).mean()
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    
    return {
        'accuracy': best_accuracy,
        'threshold': best_threshold,
    }


def load_lfw_pairs(pairs_path: str) -> List[Tuple[str, str, int]]:
    """加载 LFW pairs.txt"""
    pairs = []
    
    with open(pairs_path, 'r') as f:
        lines = f.readlines()
    
    # 跳过第一行 (样本数)
    for line in lines[1:]:
        parts = line.strip().split()
        
        if len(parts) == 3:
            # 同类别
            name = parts[0]
            img1 = int(parts[1])
            img2 = int(parts[2])
            img1_path = f"{pairs_path}/../{name}/{name}_{img1:04d}.jpg"
            img2_path = f"{pairs_path}/../{name}/{name}_{img2:04d}.jpg"
            pairs.append((img1_path, img2_path, 1))
        else:
            # 不同类别
            name1 = parts[0]
            img1 = int(parts[1])
            name2 = parts[2]
            img2 = int(parts[3])
            img1_path = f"{pairs_path}/../{name1}/{name1}_{img1:04d}.jpg"
            img2_path = f"{pairs_path}/../{name2}/{name2}_{img2:04d}.jpg"
            pairs.append((img1_path, img2_path, 0))
    
    return pairs


if __name__ == '__main__':
    # 测试数据集
    from ultra_precise_rec import build_ultra_precise_recognizer
    
    transforms = get_train_transforms(img_size=112)
    dataset = CASIAWebFaceDataset(
        root_dir='/data/CASIA-WebFace',
        transforms=transforms,
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of classes: {len(dataset.class_to_idx)}")
    
    # 测试数据加载
    sample = dataset[0]
    print(f"Sample image shape: {sample['image'].shape}")
