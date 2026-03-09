"""
Ultra-Face Recognition System - Utils Module
=============================================
工具函数和辅助类
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import torch


def draw_detections(
    image: np.ndarray,
    detections: List[dict],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    绘制检测结果
    
    Args:
        image: 输入图像
        detections: 检测结果列表
        color: 边界框颜色 (BGR)
        thickness: 线条粗细
        
    Returns:
        drawn_image: 绘制后的图像
    """
    drawn = image.copy()
    
    for det in detections:
        bbox = det.get('bbox', [])
        if len(bbox) < 4:
            continue
        
        x1, y1, x2, y2 = map(int, bbox[:4])
        cv2.rectangle(drawn, (x1, y1), (x2, y2), color, thickness)
        
        # 绘制置信度
        conf = det.get('confidence', 0)
        label = f"{conf:.2f}"
        cv2.putText(drawn, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
    
    return drawn


def draw_landmarks(
    image: np.ndarray,
    landmarks: List[List[float]],
    color: Tuple[int, int, int] = (255, 0, 0),
    radius: int = 2,
) -> np.ndarray:
    """
    绘制关键点
    
    Args:
        image: 输入图像
        landmarks: 关键点列表
        color: 颜色 (BGR)
        radius: 点半径
        
    Returns:
        drawn_image: 绘制后的图像
    """
    drawn = image.copy()
    
    for kpt in landmarks:
        x, y = int(kpt[0]), int(kpt[1])
        cv2.circle(drawn, (x, y), radius, color, -1)
    
    return drawn


def crop_face(
    image: np.ndarray,
    bbox: List[float],
    padding: float = 0.3,
    output_size: Tuple[int, int] = (112, 112),
) -> np.ndarray:
    """
    裁剪人脸
    
    Args:
        image: 输入图像
        bbox: 边界框 [x1, y1, x2, y2]
        padding: 填充比例
        output_size: 输出尺寸
        
    Returns:
        face_image: 裁剪后的人脸
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    
    # 填充
    box_w = x2 - x1
    box_h = y2 - y1
    pad_x = int(box_w * padding)
    pad_y = int(box_h * padding)
    
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    
    face_image = image[y1:y2, x1:x2]
    
    # 调整尺寸
    face_image = cv2.resize(face_image, output_size)
    
    return face_image


def normalize_feature(feature: np.ndarray) -> np.ndarray:
    """
    L2 归一化特征
    
    Args:
        feature: 特征向量
        
    Returns:
        normalized_feature: 归一化后的特征
    """
    norm = np.linalg.norm(feature)
    if norm > 1e-7:
        return feature / norm
    return feature


def compute_similarity(
    feature1: np.ndarray,
    feature2: np.ndarray,
) -> float:
    """
    计算余弦相似度
    
    Args:
        feature1: 特征 1
        feature2: 特征 2
        
    Returns:
        similarity: 相似度
    """
    feat1_norm = normalize_feature(feature1)
    feat2_norm = normalize_feature(feature2)
    
    return float(np.dot(feat1_norm, feat2_norm))


def preprocess_image(
    image: np.ndarray,
    input_size: Tuple[int, int] = (640, 640),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    """
    预处理图像
    
    Args:
        image: 输入图像 (BGR)
        input_size: 输入尺寸
        mean: 均值
        std: 标准差
        
    Returns:
        tensor: 预处理后的张量
    """
    # 转换颜色空间
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 调整尺寸
    image_resized = cv2.resize(image_rgb, input_size)
    
    # 归一化
    image_normalized = image_resized.astype(np.float32) / 255.0
    image_normalized = (image_normalized - mean) / std
    
    # HWC to CHW
    image_tensor = torch.from_numpy(image_normalized.transpose(2, 0, 1))
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor


def xyxy_to_xywh(bbox: List[float]) -> List[float]:
    """xyxy 转 xywh"""
    x1, y1, x2, y2 = bbox
    return [x1, y1, x2 - x1, y2 - y1]


def xywh_to_xyxy(bbox: List[float]) -> List[float]:
    """xywh 转 xyxy"""
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def compute_iou(
    box1: List[float],
    box2: List[float],
) -> float:
    """
    计算 IoU
    
    Args:
        box1: 框 1 (x1, y1, x2, y2)
        box2: 框 2 (x1, y1, x2, y2)
        
    Returns:
        iou: IoU 值
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - inter
    
    return inter / (union + 1e-7)
