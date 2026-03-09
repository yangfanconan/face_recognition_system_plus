"""
FastAPI Server Module (FastAPI 服务部署模块)
============================================
高并发人脸检索服务
支持批量推理、图片/视频输入、WebSocket 实时处理
"""

import os
import io
import time
import base64
import uuid
import json
from typing import List, Dict, Optional, Union
from pathlib import Path
from contextlib import asynccontextmanager
from dataclasses import dataclass
import logging

import numpy as np
from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    BackgroundTasks,
    Depends,
)
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import cv2

# 本地导入
from inference_pipeline import (
    FaceInferencePipeline,
    PipelineConfig,
    InferenceResult,
    build_pipeline,
)


# ============================================================================
# 日志配置
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


# ============================================================================
# 全局状态
# ============================================================================
class AppState:
    """应用状态"""
    
    def __init__(self):
        self.pipeline: Optional[FaceInferencePipeline] = None
        self.request_count = 0
        self.start_time = time.time()
    
    def get_pipeline(self) -> FaceInferencePipeline:
        if self.pipeline is None:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        return self.pipeline


app_state = AppState()


# ============================================================================
# 请求/响应模型
# ============================================================================
class DetectRequest(BaseModel):
    """检测请求"""
    
    # 置信度阈值
    conf_threshold: float = Field(default=0.3, ge=0, le=1)
    
    # NMS 阈值
    nms_threshold: float = Field(default=0.5, ge=0, le=1)
    
    # 是否识别
    do_recognition: bool = True
    
    # 是否检索
    do_search: bool = True


class DetectResponse(BaseModel):
    """检测响应"""
    
    # 请求 ID
    request_id: str
    
    # 检测到的人脸
    faces: List[Dict]
    
    # 延迟 (ms)
    latency_ms: Dict[str, float]
    
    # 图像尺寸
    image_shape: tuple
    
    # 状态
    status: str = 'success'


class VerifyRequest(BaseModel):
    """验证请求"""
    
    # 特征 1
    feature1: List[float]
    
    # 特征 2
    feature2: List[float]
    
    # 阈值
    threshold: float = 0.5


class VerifyResponse(BaseModel):
    """验证响应"""
    
    # 是否同一人
    is_same: bool
    
    # 相似度
    similarity: float
    
    # 请求 ID
    request_id: str


class SearchRequest(BaseModel):
    """检索请求"""
    
    # 特征向量
    feature: List[float]
    
    # 返回数量
    top_k: int = 10
    
    # 分数阈值
    threshold: float = 0.5


class RegisterFaceRequest(BaseModel):
    """注册人脸请求"""
    
    # 人脸 ID
    face_id: Optional[str] = None
    
    # 元数据
    metadata: Optional[Dict] = None


class HealthResponse(BaseModel):
    """健康检查响应"""
    
    status: str
    uptime_seconds: float
    request_count: int
    avg_latency_ms: float


# ============================================================================
# 生命周期管理
# ============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期"""
    # 启动时初始化
    logger.info("Initializing face recognition pipeline...")
    
    config = PipelineConfig(
        det_model_path=os.getenv('DET_MODEL_PATH', 'checkpoints/detection/best.pth'),
        rec_model_path=os.getenv('REC_MODEL_PATH', 'checkpoints/recognition/best.pth'),
        search_index_path=os.getenv('SEARCH_INDEX_PATH', 'indexes/face_index'),
        device=os.getenv('DEVICE', 'cuda'),
    )
    
    app_state.pipeline = build_pipeline(
        det_model_path=config.det_model_path,
        rec_model_path=config.rec_model_path,
        search_index_path=config.search_index_path,
        device=config.device,
    )
    
    logger.info("Pipeline initialized successfully")
    
    yield
    
    # 关闭时清理
    logger.info("Shutting down...")


# ============================================================================
# FastAPI 应用
# ============================================================================
app = FastAPI(
    title="Ultra-Face Recognition API",
    description="工业级超极限人脸识别服务 - 检测 + 识别 + 检索",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# 辅助函数
# ============================================================================
def generate_request_id() -> str:
    """生成请求 ID"""
    return str(uuid.uuid4())


def decode_image(image_data: bytes) -> np.ndarray:
    """解码图像"""
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Failed to decode image")
    
    return image


def encode_image_to_base64(image: np.ndarray) -> str:
    """编码图像为 Base64"""
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')


def get_pipeline() -> FaceInferencePipeline:
    """获取管道依赖"""
    return app_state.get_pipeline()


# ============================================================================
# API 端点
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """根路径"""
    return {
        "service": "Ultra-Face Recognition API",
        "version": "1.0.0",
        "status": "running",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(pipeline: FaceInferencePipeline = Depends(get_pipeline)):
    """健康检查"""
    stats = pipeline.stats()
    
    return HealthResponse(
        status="healthy",
        uptime_seconds=time.time() - app_state.start_time,
        request_count=app_state.request_count,
        avg_latency_ms=stats.get('avg_latency_ms', 0),
    )


@app.post("/detect", response_model=DetectResponse, tags=["Detection"])
async def detect(
    file: UploadFile = File(..., description="上传的图像"),
    request: DetectRequest = DetectRequest(),
    pipeline: FaceInferencePipeline = Depends(get_pipeline),
):
    """
    人脸检测
    
    - 支持 JPG/PNG 格式
    - 返回人脸位置、置信度、特征
    """
    request_id = generate_request_id()
    
    try:
        # 读取图像
        image_data = await file.read()
        image = decode_image(image_data)
        
        # 推理
        result = pipeline.infer(
            image,
            do_recognition=request.do_recognition,
            do_search=request.do_search,
        )
        
        # 更新统计
        app_state.request_count += 1
        
        return DetectResponse(
            request_id=request_id,
            faces=[f.to_dict() for f in result.faces],
            latency_ms=result.latency_ms,
            image_shape=result.image_shape,
            status='success',
        )
    
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect/base64", response_model=DetectResponse, tags=["Detection"])
async def detect_base64(
    image_base64: str,
    request: DetectRequest = DetectRequest(),
    pipeline: FaceInferencePipeline = Depends(get_pipeline),
):
    """
    人脸检测 (Base64 输入)
    """
    request_id = generate_request_id()
    
    try:
        # 解码 Base64
        image_data = base64.b64decode(image_base64)
        image = decode_image(image_data)
        
        # 推理
        result = pipeline.infer(image)
        
        app_state.request_count += 1
        
        return DetectResponse(
            request_id=request_id,
            faces=[f.to_dict() for f in result.faces],
            latency_ms=result.latency_ms,
            image_shape=result.image_shape,
            status='success',
        )
    
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/verify", response_model=VerifyResponse, tags=["Verification"])
async def verify(
    request: VerifyRequest,
    pipeline: FaceInferencePipeline = Depends(get_pipeline),
):
    """
    人脸验证 (1:1)
    
    比较两个人脸特征是否属于同一人
    """
    request_id = generate_request_id()
    
    try:
        feat1 = np.array(request.feature1)
        feat2 = np.array(request.feature2)
        
        # 归一化
        feat1 = feat1 / (np.linalg.norm(feat1) + 1e-7)
        feat2 = feat2 / (np.linalg.norm(feat2) + 1e-7)
        
        # 计算相似度
        similarity = float(np.dot(feat1, feat2))
        is_same = similarity >= request.threshold
        
        app_state.request_count += 1
        
        return VerifyResponse(
            is_same=is_same,
            similarity=similarity,
            request_id=request_id,
        )
    
    except Exception as e:
        logger.error(f"Verification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", tags=["Search"])
async def search(
    request: SearchRequest,
    pipeline: FaceInferencePipeline = Depends(get_pipeline),
):
    """
    人脸检索 (1:N)
    
    在人脸库中搜索相似人脸
    """
    request_id = generate_request_id()
    
    try:
        feature = np.array(request.feature)
        
        # 使用检索引擎
        if pipeline.search_engine:
            response = pipeline.search_engine.search(
                feature,
                k=request.top_k,
            )
            
            app_state.request_count += 1
            
            return {
                'request_id': request_id,
                'results': [r.to_dict() for r in response.results],
                'latency_ms': response.latency_ms,
                'status': 'success',
            }
        else:
            raise HTTPException(status_code=503, detail="Search engine not available")
    
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/register", tags=["Management"])
async def register_face(
    file: UploadFile = File(...),
    face_id: Optional[str] = None,
    metadata: Optional[str] = Form(None),
    pipeline: FaceInferencePipeline = Depends(get_pipeline),
):
    """
    注册人脸
    
    将新人脸添加到人脸库
    """
    request_id = generate_request_id()
    
    try:
        # 读取图像
        image_data = await file.read()
        image = decode_image(image_data)
        
        # 检测并提取特征
        result = pipeline.infer(image, do_recognition=True, do_search=False)
        
        if not result.faces:
            raise HTTPException(status_code=400, detail="No face detected")
        
        # 使用第一张人脸
        face = result.faces[0]
        feature = face.feature
        
        # 添加到检索引擎
        if pipeline.search_engine:
            parsed_metadata = json.loads(metadata) if metadata else {}
            parsed_metadata['face_id'] = face_id or str(uuid.uuid4())
            
            face_id_assigned = pipeline.search_engine.add_face(
                feature,
                metadata=parsed_metadata,
            )
            
            app_state.request_count += 1
            
            return {
                'request_id': request_id,
                'face_id': face_id_assigned,
                'status': 'success',
            }
        else:
            raise HTTPException(status_code=503, detail="Search engine not available")
    
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/face/{face_id}", tags=["Management"])
async def delete_face(
    face_id: int,
    pipeline: FaceInferencePipeline = Depends(get_pipeline),
):
    """
    删除人脸
    """
    try:
        if pipeline.search_engine:
            success = pipeline.search_engine.delete_face(face_id)
            
            if success:
                return {'status': 'success', 'message': f'Face {face_id} deleted'}
            else:
                raise HTTPException(status_code=404, detail=f'Face {face_id} not found')
        else:
            raise HTTPException(status_code=503, detail="Search engine not available")
    
    except Exception as e:
        logger.error(f"Delete error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/face/{face_id}", tags=["Management"])
async def get_face(
    face_id: int,
    pipeline: FaceInferencePipeline = Depends(get_pipeline),
):
    """
    获取人脸信息
    """
    try:
        if pipeline.search_engine:
            metadata = pipeline.search_engine.get_face(face_id)
            
            if metadata:
                return {
                    'face_id': face_id,
                    'metadata': metadata,
                    'status': 'success',
                }
            else:
                raise HTTPException(status_code=404, detail=f'Face {face_id} not found')
        else:
            raise HTTPException(status_code=503, detail="Search engine not available")
    
    except Exception as e:
        logger.error(f"Get face error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", tags=["Monitoring"])
async def get_stats(pipeline: FaceInferencePipeline = Depends(get_pipeline)):
    """
    获取服务统计
    """
    stats = pipeline.stats()
    
    return {
        'pipeline_stats': stats,
        'app_stats': {
            'request_count': app_state.request_count,
            'uptime_seconds': time.time() - app_state.start_time,
        },
    }


# ============================================================================
# WebSocket 实时处理
# ============================================================================
@app.websocket("/ws/realtime")
async def websocket_realtime(
    websocket: WebSocket,
    pipeline: FaceInferencePipeline = Depends(get_pipeline),
):
    """
    WebSocket 实时人脸检测
    """
    await websocket.accept()
    
    try:
        while True:
            # 接收图像
            data = await websocket.receive_bytes()
            
            # 解码
            image = decode_image(data)
            
            # 推理
            result = pipeline.infer(image)
            
            # 绘制结果
            drawn_image = pipeline._draw_results(image, result)
            
            # 编码返回
            _, buffer = cv2.imencode('.jpg', drawn_image)
            await websocket.send_bytes(buffer)
            
            # 发送元数据
            await websocket.send_json({
                'faces': [f.to_dict() for f in result.faces],
                'latency_ms': result.latency_ms,
            })
    
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()


# ============================================================================
# 批量处理端点
# ============================================================================
@app.post("/batch/detect", tags=["Batch"])
async def batch_detect(
    files: List[UploadFile] = File(...),
    pipeline: FaceInferencePipeline = Depends(get_pipeline),
):
    """
    批量人脸检测
    """
    request_id = generate_request_id()
    
    try:
        results = []
        
        for file in files:
            image_data = await file.read()
            image = decode_image(image_data)
            
            result = pipeline.infer(image)
            results.append(result.to_dict())
        
        app_state.request_count += len(files)
        
        return {
            'request_id': request_id,
            'results': results,
            'total_images': len(files),
            'status': 'success',
        }
    
    except Exception as e:
        logger.error(f"Batch detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# 视频处理 (后台任务)
# ============================================================================
@app.post("/process/video", tags=["Video"])
async def process_video(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    pipeline: FaceInferencePipeline = Depends(get_pipeline),
):
    """
    视频处理 (异步)
    """
    request_id = generate_request_id()
    
    # 保存上传的视频
    video_path = f"/tmp/{request_id}_{file.filename}"
    
    with open(video_path, 'wb') as f:
        content = await file.read()
        f.write(content)
    
    # 输出路径
    output_path = f"/tmp/{request_id}_output.mp4"
    
    # 后台处理
    background_tasks.add_task(
        pipeline.process_video,
        video_path,
        output_path,
        display=False,
    )
    
    return {
        'request_id': request_id,
        'status': 'processing',
        'message': 'Video processing started in background',
    }


# ============================================================================
# 错误处理
# ============================================================================
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            'error': exc.detail,
            'status_code': exc.status_code,
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            'error': 'Internal server error',
            'status_code': 500,
        },
    )


# ============================================================================
# 启动命令
# ============================================================================
if __name__ == '__main__':
    import uvicorn
    
    uvicorn.run(
        app,
        host='0.0.0.0',
        port=8000,
        workers=4,
        log_level='info',
    )
