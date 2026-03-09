# 项目部署文档

## 目录结构

```
face_recognition_system_plus/
├── python/                          # Python 全栈实现
│   ├── det/                         # 人脸检测模块
│   ├── rec/                         # 人脸识别模块
│   ├── retrieval/                   # 检索模块
│   ├── deploy/                      # 部署模块
│   ├── configs/                     # 配置文件
│   ├── utils/                       # 工具函数
│   ├── main.py                      # 主入口
│   └── requirements.txt             # 依赖
│
├── cpp/                             # C++/CUDA 高性能实现
│   ├── include/                     # 头文件
│   ├── src/                         # 源文件
│   │   ├── det/                     # 检测实现
│   │   ├── rec/                     # 识别实现
│   │   ├── retrieval/               # 检索实现
│   │   └── sdk/                     # SDK 实现
│   ├── examples/                    # 示例程序
│   ├── tools/                       # 工具程序
│   └── CMakeLists.txt               # CMake 配置
│
├── scripts/                         # 构建脚本
├── docs/                            # 文档
└── README.md                        # 项目说明
```

## Python 环境安装

### 1. 创建虚拟环境（推荐）

```bash
# Linux/Mac
cd python
python3 -m venv venv
source venv/bin/activate

# Windows
cd python
python -m venv venv
venv\Scripts\activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 验证安装

```bash
python -c "import torch; print(torch.__version__)"
python -c "import cv2; print(cv2.__version__)"
```

## C++ 编译

### Linux

```bash
cd cpp
mkdir build && cd build

# 配置
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCUDA_ARCHITECTURES="70;75;80" \
         -DBUILD_PYTHON_BINDINGS=OFF

# 编译
make -j$(nproc)

# 安装（可选）
sudo make install
```

### Windows

```bash
cd cpp
mkdir build && cd build

# 使用 Visual Studio 生成器
cmake .. -G "Visual Studio 17 2022" -A x64

# 编译
cmake --build . --config Release
```

### macOS

```bash
cd cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(sysctl -n hw.ncpu)
```

## 模型准备

### 1. 下载预训练模型

```bash
# 创建目录
mkdir -p python/checkpoints/detection
mkdir -p python/checkpoints/recognition
mkdir -p python/indexes
```

### 2. 训练模型

```bash
cd python

# 训练检测器
python main.py train_det --config configs/config.yaml

# 训练识别器
python main.py train_rec --config configs/config.yaml
```

### 3. 导出 TensorRT 引擎

```bash
# Python 导出 ONNX
python -c "
from det.ultra_tiny_det import build_ultra_tiny_detector
model = build_ultra_tiny_detector()
torch.onnx.export(model, torch.randn(1,3,640,640), 'det.onnx')
"

# C++ 构建 TensorRT 引擎
cd cpp/build
./model_converter ../checkpoints/det.onnx ../checkpoints/det.engine
```

### 4. 构建检索索引

```bash
cd python
python main.py build_index --data_dir /data/features --output indexes/face_index
```

## 运行示例

### Python 推理

```bash
cd python

# 单图推理
python main.py inference --image test.jpg --config configs/config.yaml

# 批量推理
python -c "
from deploy.inference_pipeline import build_pipeline
import cv2

pipeline = build_pipeline(
    det_model_path='checkpoints/det.pth',
    rec_model_path='checkpoints/rec.pth',
)

image = cv2.imread('test.jpg')
result = pipeline.infer(image)
print(f'Faces: {len(result.faces)}')
"
```

### C++ 测试

```bash
cd cpp/build

# SDK 测试
./sdk_test

# 性能基准
./benchmark ../test.jpg
```

### 启动 API 服务

```bash
cd python

# 启动服务
python -m uvicorn deploy.fastapi_server:app --host 0.0.0.0 --port 8000

# 后台运行
nohup python -m uvicorn deploy.fastapi_server:app --port 8000 &
```

### API 调用示例

```bash
# 检测人脸
curl -X POST "http://localhost:8000/detect" \
  -F "file=@test.jpg"

# 健康检查
curl "http://localhost:8000/health"

# 获取统计
curl "http://localhost:8000/stats"
```

## Docker 部署

### 构建镜像

```bash
docker build -t face-recognition:latest .
```

### 运行容器

```bash
docker run -d \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/indexes:/app/indexes \
  face-recognition:latest
```

## 故障排查

### 常见问题

1. **CUDA 不可用**
   ```bash
   nvidia-smi  # 检查 GPU 驱动
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **TensorRT 加载失败**
   ```bash
   # 检查 TensorRT 安装
   ldconfig -p | grep nvinfer
   ```

3. **FAISS 导入错误**
   ```bash
   # 重新安装
   pip uninstall faiss-gpu
   pip install faiss-gpu
   ```

4. **内存不足**
   - 减小批次大小
   - 使用 FP16 模式
   - 启用梯度检查点

### 性能优化

1. **启用 TensorRT**
   ```python
   config.use_tensorrt = True
   config.use_fp16 = True
   ```

2. **多 GPU 推理**
   ```python
   pipeline = build_pipeline(device='cuda:0')
   ```

3. **批量推理**
   ```python
   results = pipeline.batch_infer([image1, image2, image3])
   ```

## 更新日志

### v1.0.0 (2026-03-09)
- 初始版本
- Python 全栈实现完成
- C++/CUDA 高性能实现完成
- FastAPI 服务部署完成
