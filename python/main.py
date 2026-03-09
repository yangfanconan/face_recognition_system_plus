"""
Ultra-Face Recognition System - Training & Inference Scripts
=============================================================
工业级超极限人脸识别系统 - 训练和推理启动脚本

启动示例:
----------
1. 训练检测器:
   python train_detector.py --config configs/config.yaml

2. 训练识别器:
   python train_recognizer.py --config configs/config.yaml

3. 构建检索索引:
   python build_index.py --data_dir /data/features --output indexes/face_index

4. 启动推理服务:
   python fastapi_server.py --host 0.0.0.0 --port 8000

5. 单图推理测试:
   python inference_test.py --image test.jpg
"""

import os
import sys
import argparse
import yaml
from pathlib import Path


# ============================================================================
# 训练检测器
# ============================================================================
def train_detector(config_path: str):
    """训练检测器"""
    from det.det_trainer import DetectionTrainerConfig, train_detection
    from det.det_dataset import WiderFaceDataset, get_train_transforms, get_val_transforms, build_detection_dataloader
    
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 训练配置
    trainer_config = DetectionTrainerConfig(
        img_size=config['detector']['img_size'],
        batch_size=config['training']['det_batch_size'],
        epochs=config['training']['det_epochs'],
        lr=config['training']['det_lr'],
        warmup_epochs=config['training']['det_warmup_epochs'],
        amp=config['training']['amp'],
        ema=config['training']['ema'],
    )
    
    # 数据集
    train_dataset = WiderFaceDataset(
        root_dir='/data/WiderFace',
        split='train',
        img_size=trainer_config.img_size,
        transforms=get_train_transforms(trainer_config.img_size),
    )
    
    val_dataset = WiderFaceDataset(
        root_dir='/data/WiderFace',
        split='val',
        img_size=trainer_config.img_size,
        transforms=get_val_transforms(trainer_config.img_size),
    )
    
    # 数据加载器
    train_loader = build_detection_dataloader(
        train_dataset,
        batch_size=trainer_config.batch_size,
        num_workers=8,
    )
    
    val_loader = build_detection_dataloader(
        val_dataset,
        batch_size=trainer_config.batch_size,
        num_workers=8,
    )
    
    # 开始训练
    print("Starting detector training...")
    trainer = train_detection(
        config=trainer_config,
        train_loader=train_loader,
        val_loader=val_loader,
    )
    
    print(f"Training completed! Best mAP: {trainer.best_map:.4f}")


# ============================================================================
# 训练识别器
# ============================================================================
def train_recognizer(config_path: str):
    """训练识别器"""
    from rec.rec_trainer import RecognitionTrainerConfig, train_recognition
    from rec.rec_dataset import CASIAWebFaceDataset, get_train_transforms, build_recognition_dataloader
    
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 训练配置
    trainer_config = RecognitionTrainerConfig(
        img_size=config['recognizer']['img_size'],
        batch_size=config['training']['rec_batch_size'],
        epochs=config['training']['rec_epochs'],
        lr=config['training']['rec_lr'],
        warmup_epochs=config['training']['det_warmup_epochs'],
        amp=config['training']['amp'],
        ema=config['training']['ema'],
    )
    
    # 数据集
    train_dataset = CASIAWebFaceDataset(
        root_dir='/data/CASIA-WebFace',
        transforms=get_train_transforms(trainer_config.img_size),
    )
    
    # 数据加载器
    train_loader = build_recognition_dataloader(
        train_dataset,
        batch_size=trainer_config.batch_size,
        num_workers=8,
    )
    
    # 开始训练
    print("Starting recognizer training...")
    num_classes = len(train_dataset.class_to_idx)
    
    trainer = train_recognition(
        config=trainer_config,
        train_loader=train_loader,
        val_loader=None,  # 验证集可选
        num_classes=num_classes,
    )
    
    print(f"Training completed!")


# ============================================================================
# 构建检索索引
# ============================================================================
def build_index(data_dir: str, output_path: str):
    """构建检索索引"""
    import numpy as np
    from retrieval.billion_iadm import build_billion_scale_index
    
    print(f"Building index from {data_dir}...")
    
    # 加载特征
    features = []
    metadata = []
    
    for feat_file in Path(data_dir).glob('*.npy'):
        feat = np.load(feat_file)
        features.append(feat)
        
        # 元数据
        meta = {'path': str(feat_file)}
        metadata.append(meta)
    
    features = np.vstack(features)
    print(f"Loaded {len(features)} features")
    
    # 构建索引
    engine = build_billion_scale_index(
        vectors=features,
        metadata=metadata,
        save_path=output_path,
    )
    
    stats = engine.get_stats()
    print(f"Index built successfully!")
    print(f"  Total vectors: {stats['total_vectors']}")
    print(f"  Memory usage: {stats['memory_usage_mb']:.2f} MB")


# ============================================================================
# 推理测试
# ============================================================================
def inference_test(image_path: str, config_path: str):
    """推理测试"""
    import cv2
    from deploy.inference_pipeline import build_pipeline
    
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 构建管道
    pipeline = build_pipeline(
        det_model_path=config['models']['det_model_path'],
        rec_model_path=config['models']['rec_model_path'],
        search_index_path=config['models']['search_index_path'],
    )
    
    # 读取图像
    image = cv2.imread(image_path)
    
    # 推理
    result = pipeline.infer(image)
    
    # 输出结果
    print(f"\nInference Results:")
    print(f"  Image: {image_path}")
    print(f"  Faces detected: {len(result.faces)}")
    print(f"  Total latency: {result.latency_ms['total']:.2f} ms")
    
    for i, face in enumerate(result.faces):
        print(f"\n  Face {i+1}:")
        print(f"    BBox: {face.bbox}")
        print(f"    Confidence: {face.confidence:.4f}")
        if face.identity_id is not None:
            print(f"    Identity: ID={face.identity_id}, Score={face.identity_score:.4f}")
        if face.search_results:
            print(f"    Top search result: ID={face.search_results[0].face_id}, Score={face.search_results[0].score:.4f}")
    
    # 绘制结果
    drawn = pipeline._draw_results(image.copy(), result)
    cv2.imwrite('output.jpg', drawn)
    print(f"\nResult saved to output.jpg")


# ============================================================================
# 主函数
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Ultra-Face Recognition System')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # 训练检测器
    parser_det = subparsers.add_parser('train_det', help='Train detector')
    parser_det.add_argument('--config', default='configs/config.yaml', help='Config path')
    
    # 训练识别器
    parser_rec = subparsers.add_parser('train_rec', help='Train recognizer')
    parser_rec.add_argument('--config', default='configs/config.yaml', help='Config path')
    
    # 构建索引
    parser_index = subparsers.add_parser('build_index', help='Build search index')
    parser_index.add_argument('--data_dir', required=True, help='Feature data directory')
    parser_index.add_argument('--output', required=True, help='Output path')
    
    # 推理测试
    parser_infer = subparsers.add_parser('inference', help='Run inference')
    parser_infer.add_argument('--image', required=True, help='Input image path')
    parser_infer.add_argument('--config', default='configs/config.yaml', help='Config path')
    
    args = parser.parse_args()
    
    if args.command == 'train_det':
        train_detector(args.config)
    elif args.command == 'train_rec':
        train_recognizer(args.config)
    elif args.command == 'build_index':
        build_index(args.data_dir, args.output)
    elif args.command == 'inference':
        inference_test(args.image, args.config)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
