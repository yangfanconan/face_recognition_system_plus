# Ultra-Face Recognition System - Training Guide
# ===============================================
# 训练指南

## 自动训练流程

### 1. 一键启动训练

```bash
cd python

# 完整训练（检测器 + 识别器）
python automated_training.py --all

# 只训练检测器
python automated_training.py --det-only

# 只训练识别器
python automated_training.py --rec-only

# 自定义训练轮数
python automated_training.py --all --det-epochs 100 --rec-epochs 50
```

### 2. 分步训练

```bash
# 步骤 1: 准备数据
python automated_training.py --prepare-data

# 步骤 2: 训练检测器
python automated_training.py --det-only --det-epochs 300

# 步骤 3: 训练识别器
python automated_training.py --rec-only --rec-epochs 100

# 步骤 4: 导出模型
python automated_training.py --export

# 步骤 5: 构建检索索引
python automated_training.py --build-index
```

### 3. 使用训练脚本

```bash
# 快速训练（默认 10 轮）
python train.py

# 自定义模式
python train.py --mode det --epochs 50
python train.py --mode rec --epochs 30
```

## 训练配置

编辑 `configs/config.yaml` 修改训练参数：

```yaml
training:
  # 检测器训练
  det_batch_size: 32
  det_epochs: 300
  det_lr: 0.001
  det_warmup_epochs: 5
  
  # 识别器训练
  rec_batch_size: 128
  rec_epochs: 100
  rec_lr: 0.1
  
  # 训练策略
  amp: true          # 混合精度训练
  ema: true          # 指数移动平均
  ema_decay: 0.999   # EMA 衰减率
```

## 数据集准备

### WiderFace (检测)

1. 下载：http://shuoyang1213.me/WIDERFACE/
2. 解压到 `data/widerface/`
3. 目录结构：
   ```
   data/widerface/
   ├── WIDER_train/images/
   ├── WIDER_val/images/
   └── wider_face_split/
       ├── wider_face_train_bbx_gt.txt
       └── wider_face_val_bbx_gt.txt
   ```

### CASIA-WebFace (识别)

1. 申请：http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html
2. 解压到 `data/casia_webface/`
3. 目录结构：
   ```
   data/casia_webface/
   ├── 0000000/
   │   ├── 000.jpg
   │   └── ...
   └── ...
   ```

## 训练监控

### TensorBoard

```bash
# 启动 TensorBoard
tensorboard --logdir=logs/

# 访问 http://localhost:6006
```

### 训练日志

- 检测器日志：`logs/detection/training_log.json`
- 识别器日志：`logs/recognition/training_log.json`

## 模型导出

### ONNX 格式

```bash
python automated_training.py --export
```

输出：
- `checkpoints/detection/ultra_tiny_det.onnx`
- `checkpoints/recognition/ultra_precise_rec.onnx`

### TensorRT 引擎

```bash
cd ../cpp/build
./model_converter ../checkpoints/det.onnx ../checkpoints/det.engine
```

## 训练时间估算

| GPU | 检测器 (300 轮) | 识别器 (100 轮) |
|-----|----------------|----------------|
| RTX 4090 | ~6 小时 | ~4 小时 |
| RTX 3080 | ~10 小时 | ~7 小时 |
| RTX 3060 | ~18 小时 | ~12 小时 |
| CPU | ~48 小时 | ~36 小时 |

## 常见问题

### Q: CUDA out of memory
A: 减小 batch_size 或使用梯度累积

### Q: 训练损失不下降
A: 检查学习率，尝试 warmup 或降低初始学习率

### Q: 数据集下载失败
A: 使用镜像源或手动下载后解压

## 下一步

训练完成后：
1. 运行推理测试：`python main.py inference --image test.jpg`
2. 启动 API 服务：`python -m uvicorn deploy.fastapi_server:app --port 8000`
3. 部署到生产环境：参考 `docs/DEPLOYMENT.md`
