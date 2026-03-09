"""
Ultra-Face Recognition System - Python Package
===============================================
工业级超极限人脸识别全栈系统 - Python 实现
"""

__version__ = '1.0.0'
__author__ = 'Face Recognition Team'

# 检测模块
from .det.ultra_tiny_det import (
    UltraTinyDetector,
    build_ultra_tiny_detector,
    DCNv4,
    TinyViT,
    UltraFPN,
    UltraDetHead,
)

# 识别模块
from .rec.ultra_precise_rec import (
    UltraPreciseRecognizer,
    build_ultra_precise_recognizer,
    SpatialBranch,
    FrequencyBranch,
    DepthBranch,
    GlobalTransformer,
)

# 检索模块
from .retrieval.billion_iadm import (
    BillionScaleSearchEngine,
    ProductQuantizer,
    HNSWIndex,
)

from .retrieval.search_engine import (
    FaceSearchEngine,
    Face1NRecognizer,
    create_search_engine,
)

# 部署模块
from .deploy.inference_pipeline import (
    FaceInferencePipeline,
    build_pipeline,
    PipelineConfig,
)

__all__ = [
    # 检测
    'UltraTinyDetector',
    'build_ultra_tiny_detector',
    'DCNv4',
    'TinyViT',
    'UltraFPN',
    'UltraDetHead',
    
    # 识别
    'UltraPreciseRecognizer',
    'build_ultra_precise_recognizer',
    'SpatialBranch',
    'FrequencyBranch',
    'DepthBranch',
    'GlobalTransformer',
    
    # 检索
    'BillionScaleSearchEngine',
    'ProductQuantizer',
    'HNSWIndex',
    'FaceSearchEngine',
    'Face1NRecognizer',
    'create_search_engine',
    
    # 部署
    'FaceInferencePipeline',
    'build_pipeline',
    'PipelineConfig',
]
