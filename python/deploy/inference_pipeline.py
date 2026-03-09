"""
Inference Pipeline Module (全链路推理管道模块)
==============================================
封装检测→识别→检索端到端推理接口
支持图片/视频输入、批量推理、多 GPU 推理
"""

import os
import cv2
import time
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict
import threading

import torch
import torch.nn.functional as F

# 本地导入
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from det.ultra_tiny_det import UltraTinyDetector, nms, build_ultra_tiny_detector
from rec.ultra_precise_rec import UltraPreciseRecognizer, build_ultra_precise_recognizer
from retrieval.search_engine import FaceSearchEngine, FaceSearchResult, SearchResponse


# ============================================================================
# 数据结构
# ============================================================================
@dataclass
class DetectedFace:
    """检测到的人脸"""
    
    # 边界框 [x1, y1, x2, y2]
    bbox: List[float]
    
    # 置信度
    confidence: float
    
    # 关键点 (5 个)
    landmarks: Optional[List[List[float]]] = None
    
    # 人脸图像 (裁剪后)
    face_image: Optional[np.ndarray] = None
    
    # 识别特征
    feature: Optional[np.ndarray] = None
    
    # 识别结果
    identity_id: Optional[int] = None
    identity_score: float = 0.0
    
    # 检索结果
    search_results: Optional[List[FaceSearchResult]] = None
    
    def to_dict(self) -> Dict:
        return {
            'bbox': self.bbox,
            'confidence': self.confidence,
            'landmarks': self.landmarks,
            'identity_id': self.identity_id,
            'identity_score': self.identity_score,
            'search_results': [r.to_dict() for r in self.search_results] if self.search_results else None,
        }


@dataclass
class InferenceResult:
    """推理结果"""
    
    # 检测到的人脸列表
    faces: List[DetectedFace]
    
    # 推理耗时 (ms)
    latency_ms: Dict[str, float]
    
    # 图像尺寸
    image_shape: Tuple[int, int]
    
    def to_dict(self) -> Dict:
        return {
            'faces': [f.to_dict() for f in self.faces],
            'latency_ms': self.latency_ms,
            'image_shape': self.image_shape,
        }


# ============================================================================
# 配置
# ============================================================================
@dataclass
class PipelineConfig:
    """推理管道配置"""
    
    # 模型路径
    det_model_path: str = ''
    rec_model_path: str = ''
    search_index_path: str = ''
    
    # 检测参数
    det_img_size: int = 640
    det_conf_threshold: float = 0.3
    det_nms_threshold: float = 0.5
    
    # 识别参数
    rec_img_size: int = 112
    
    # 检索参数
    search_top_k: int = 10
    search_threshold: float = 0.5
    
    # 设备配置
    device: str = 'cuda'
    use_amp: bool = True
    
    # 性能配置
    max_batch_size: int = 32
    num_workers: int = 4
    
    def to_dict(self) -> Dict:
        return asdict(self)


# ============================================================================
# 人脸检测器封装
# ============================================================================
class FaceDetector:
    """
    人脸检测器封装
    
    Args:
        model_path: 模型路径
        device: 设备
        img_size: 输入尺寸
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        img_size: int = 640,
    ):
        self.device = device
        self.img_size = img_size
        
        # 加载模型
        self.model = build_ultra_tiny_detector(img_size=img_size)
        
        if model_path:
            checkpoint = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(f"Loaded detector from {model_path}")
        
        self.model.to(device)
        self.model.eval()
    
    @torch.no_grad()
    def detect(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.3,
        nms_threshold: float = 0.5,
    ) -> List[DetectedFace]:
        """
        检测人脸
        
        Args:
            image: 输入图像
            conf_threshold: 置信度阈值
            nms_threshold: NMS 阈值
            
        Returns:
            faces: 检测到的人脸
        """
        # 预处理
        input_tensor = self._preprocess(image)
        
        # 推理
        with torch.cuda.amp.autocast():
            outputs = self.model(input_tensor)
        
        # 后处理
        faces = self._postprocess(
            outputs,
            image.shape,
            conf_threshold,
            nms_threshold,
        )
        
        return faces
    
    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """预处理"""
        # Resize
        h, w = image.shape[:2]
        image_resized = cv2.resize(image, (self.img_size, self.img_size))
        
        # Normalize
        image_normalized = image_resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_normalized = (image_normalized - mean) / std
        
        # HWC to CHW
        image_tensor = torch.from_numpy(image_normalized.transpose(2, 0, 1))
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        return image_tensor
    
    def _postprocess(
        self,
        outputs: Dict,
        image_shape: Tuple[int, int],
        conf_threshold: float,
        nms_threshold: float,
    ) -> List[DetectedFace]:
        """后处理"""
        faces = []
        h, w = image_shape[:2]
        scale = min(h, w) / self.img_size
        
        # 获取输出
        cls_preds = outputs['cls'][0].sigmoid()
        obj_preds = outputs['obj'][0]
        reg_preds = outputs['reg'][0]
        kpt_preds = outputs['kpt'][0]
        
        # 合并置信度
        conf = (cls_preds * obj_preds).squeeze()
        
        # 阈值过滤
        mask = conf > conf_threshold
        if mask.sum() == 0:
            return faces
        
        # 生成候选框
        # 简化实现：假设 reg_preds 直接是 bbox
        boxes = reg_preds[mask]
        scores = conf[mask]
        kpts = kpt_preds[mask]
        
        # NMS
        if len(boxes) > 0:
            keep = nms(boxes, scores, iou_threshold=nms_threshold)
            
            for idx in keep:
                box = boxes[idx].cpu().numpy()
                score = scores[idx].cpu().item()
                
                # 缩放到原图
                box = box / self.img_size * min(h, w)
                
                face = DetectedFace(
                    bbox=box.tolist(),
                    confidence=score,
                )
                faces.append(face)
        
        return faces


# ============================================================================
# 人脸识别器封装
# ============================================================================
class FaceRecognizer:
    """
    人脸识别器封装
    
    Args:
        model_path: 模型路径
        device: 设备
        img_size: 输入尺寸
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        img_size: int = 112,
    ):
        self.device = device
        self.img_size = img_size
        
        # 加载模型
        self.model = build_ultra_precise_recognizer()
        
        if model_path:
            checkpoint = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(f"Loaded recognizer from {model_path}")
        
        self.model.to(device)
        self.model.eval()
    
    @torch.no_grad()
    def extract_feature(
        self,
        face_image: np.ndarray,
    ) -> np.ndarray:
        """
        提取特征
        
        Args:
            face_image: 人脸图像
            
        Returns:
            feature: 特征向量
        """
        # 预处理
        input_tensor = self._preprocess(face_image)
        
        # 推理
        with torch.cuda.amp.autocast():
            features = self.model(input_tensor)
        
        feature = features['id'].cpu().numpy()[0]
        
        # L2 归一化
        feature = feature / (np.linalg.norm(feature) + 1e-7)
        
        return feature
    
    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """预处理"""
        # Resize
        image_resized = cv2.resize(image, (self.img_size, self.img_size))
        
        # Normalize
        image_normalized = image_resized.astype(np.float32) / 255.0
        image_normalized = (image_normalized - 0.5) / 0.5
        
        # HWC to CHW
        image_tensor = torch.from_numpy(image_normalized.transpose(2, 0, 1))
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        return image_tensor
    
    @torch.no_grad()
    def verify(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
    ) -> Tuple[bool, float]:
        """
        人脸验证
        
        Args:
            image1: 图像 1
            image2: 图像 2
            
        Returns:
            is_same: 是否同一人
            similarity: 相似度
        """
        feat1 = self.extract_feature(image1)
        feat2 = self.extract_feature(image2)
        
        similarity = np.dot(feat1, feat2)
        
        return similarity > 0.5, similarity


# ============================================================================
# 推理管道
# ============================================================================
class FaceInferencePipeline:
    """
    全链路推理管道
    检测 → 识别 → 检索
    
    Args:
        config: 管道配置
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        
        # 初始化检测器
        self.detector = FaceDetector(
            model_path=config.det_model_path,
            device=config.device,
            img_size=config.det_img_size,
        )
        
        # 初始化识别器
        self.recognizer = FaceRecognizer(
            model_path=config.rec_model_path,
            device=config.device,
            img_size=config.rec_img_size,
        )
        
        # 初始化检索引擎 (可选)
        self.search_engine = None
        if config.search_index_path:
            from retrieval.search_engine import create_search_engine
            self.search_engine = create_search_engine(
                index_path=config.search_index_path,
                top_k=config.search_top_k,
            )
        
        # 统计
        self.total_inferences = 0
        self.total_latency = 0.0
    
    def infer(
        self,
        image: np.ndarray,
        do_recognition: bool = True,
        do_search: bool = True,
    ) -> InferenceResult:
        """
        端到端推理
        
        Args:
            image: 输入图像
            do_recognition: 是否识别
            do_search: 是否检索
            
        Returns:
            result: 推理结果
        """
        start_time = time.perf_counter()
        latencies = {}
        
        # 阶段 1: 检测
        det_start = time.perf_counter()
        faces = self.detector.detect(
            image,
            conf_threshold=self.config.det_conf_threshold,
            nms_threshold=self.config.det_nms_threshold,
        )
        latencies['detection'] = (time.perf_counter() - det_start) * 1000
        
        # 阶段 2: 识别 (提取特征)
        if do_recognition and faces:
            rec_start = time.perf_counter()
            for face in faces:
                # 裁剪人脸
                face_image = self._crop_face(image, face.bbox)
                face.face_image = face_image
                
                # 提取特征
                feature = self.recognizer.extract_feature(face_image)
                face.feature = feature
            latencies['recognition'] = (time.perf_counter() - rec_start) * 1000
        
        # 阶段 3: 检索
        if do_search and self.search_engine and faces:
            search_start = time.perf_counter()
            for face in faces:
                if face.feature is not None:
                    response = self.search_engine.search(face.feature)
                    face.search_results = response.results
            latencies['search'] = (time.perf_counter() - search_start) * 1000
        
        # 阶段 4: 1:N 识别
        if self.search_engine and faces:
            for face in faces:
                if face.search_results and face.search_results[0].score > self.config.search_threshold:
                    face.identity_id = face.search_results[0].face_id
                    face.identity_score = face.search_results[0].score
        
        total_latency = (time.perf_counter() - start_time) * 1000
        latencies['total'] = total_latency
        
        # 更新统计
        self.total_inferences += 1
        self.total_latency += total_latency
        
        return InferenceResult(
            faces=faces,
            latency_ms=latencies,
            image_shape=(image.shape[0], image.shape[1]),
        )
    
    def batch_infer(
        self,
        images: List[np.ndarray],
    ) -> List[InferenceResult]:
        """
        批量推理
        
        Args:
            images: 图像列表
            
        Returns:
            results: 推理结果列表
        """
        results = []
        for image in images:
            result = self.infer(image)
            results.append(result)
        return results
    
    def _crop_face(
        self,
        image: np.ndarray,
        bbox: List[float],
        padding: float = 0.3,
    ) -> np.ndarray:
        """
        裁剪人脸
        
        Args:
            image: 原图
            bbox: 边界框
            padding: 填充比例
            
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
        
        return face_image
    
    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        display: bool = False,
    ):
        """
        处理视频
        
        Args:
            video_path: 视频路径
            output_path: 输出路径
            display: 是否显示
        """
        cap = cv2.VideoCapture(video_path)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 视频写入器
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 推理
            result = self.infer(frame)
            
            # 绘制结果
            frame = self._draw_results(frame, result)
            
            # 写入
            if writer:
                writer.write(frame)
            
            # 显示
            if display:
                cv2.imshow('Face Recognition', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames")
        
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        print(f"Video processing completed. Total frames: {frame_count}")
    
    def _draw_results(
        self,
        image: np.ndarray,
        result: InferenceResult,
    ) -> np.ndarray:
        """
        绘制结果
        
        Args:
            image: 图像
            result: 推理结果
            
        Returns:
            drawn_image: 绘制后的图像
        """
        for face in result.faces:
            x1, y1, x2, y2 = map(int, face.bbox)
            
            # 绘制边界框
            color = (0, 255, 0) if face.identity_id is not None else (0, 255, 255)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签
            if face.identity_id is not None:
                label = f"ID: {face.identity_id} ({face.identity_score:.2f})"
            else:
                label = f"Unknown ({face.confidence:.2f})"
            
            cv2.putText(image, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 绘制关键点
            if face.landmarks:
                for kpt in face.landmarks:
                    cv2.circle(image, (int(kpt[0]), int(kpt[1])), 2, (255, 0, 0), -1)
        
        return image
    
    def stats(self) -> Dict:
        """获取统计信息"""
        avg_latency = self.total_latency / max(self.total_inferences, 1)
        
        return {
            'total_inferences': self.total_inferences,
            'avg_latency_ms': avg_latency,
            'fps': 1000 / avg_latency if avg_latency > 0 else 0,
        }


# ============================================================================
# 入口函数
# ============================================================================
def build_pipeline(
    det_model_path: str,
    rec_model_path: str,
    search_index_path: Optional[str] = None,
    device: str = 'cuda',
) -> FaceInferencePipeline:
    """
    构建推理管道
    
    Args:
        det_model_path: 检测模型路径
        rec_model_path: 识别模型路径
        search_index_path: 检索索引路径
        device: 设备
        
    Returns:
        pipeline: 推理管道
    """
    config = PipelineConfig(
        det_model_path=det_model_path,
        rec_model_path=rec_model_path,
        search_index_path=search_index_path,
        device=device,
    )
    
    return FaceInferencePipeline(config)


if __name__ == '__main__':
    # 示例
    print("Face Inference Pipeline")
    print("=======================")
    
    config = PipelineConfig()
    print(f"Config: {config.to_dict()}")
