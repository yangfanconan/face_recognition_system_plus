"""
Face Recognition Training - 人脸识别训练
=========================================
使用合成数据实际训练识别器
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from datetime import datetime
import numpy as np
import json

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from rec.ultra_precise_rec import UltraPreciseRecognizer


def train_recognizer(epochs=5, batch_size=4, lr=0.01, img_size=112):
    """实际训练识别器"""
    print("=" * 60)
    print("Training Recognizer (Ultra-Precise Face Recognizer)")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # 创建模型
    print("\nCreating model...")
    model = UltraPreciseRecognizer().to(device)
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params / 1e6:.2f}M")
    
    # 损失函数
    num_classes = 100
    arcface_weight = nn.Parameter(torch.randn(num_classes, 512)).to(device)
    nn.init.xavier_uniform_(arcface_weight)
    
    # 优化器
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    training_log = []
    best_loss = float('inf')
    
    print(f"\nStarting training for {epochs} epochs...")
    print(f"Batch Size: {batch_size}, Learning Rate: {lr}")
    print("-" * 60)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        start_time = time.time()
        
        num_batches = 30
        correct = 0
        total = 0
        
        for batch_idx in range(num_batches):
            # 生成随机输入
            images = torch.randn(batch_size, 3, img_size, img_size).to(device)
            labels = torch.randint(0, num_classes, (batch_size,)).to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            features = model(images)
            id_features = features['id']
            
            # 计算 ArcFace 风格损失（简化）
            id_features_norm = nn.functional.normalize(id_features, p=2, dim=1)
            weight_norm = nn.functional.normalize(arcface_weight, p=2, dim=1)
            cosine = nn.functional.linear(id_features_norm, weight_norm)
            
            # 交叉熵损失
            loss = nn.functional.cross_entropy(cosine * 64, labels)
            
            # 添加中心损失
            centers = torch.randn(num_classes, 512).to(device)
            center_loss = nn.functional.mse_loss(id_features, centers[labels]) * 0.001
            loss = loss + center_loss
            
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            # 计算准确率
            _, predicted = cosine.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{num_batches}, Loss: {loss.item():.4f}")
        
        # 更新学习率
        scheduler.step()
        
        avg_loss = total_loss / num_batches
        accuracy = 100. * correct / total
        elapsed_time = time.time() - start_time
        
        print(f"  Epoch {epoch+1}/{epochs} completed - Loss: {avg_loss:.4f}, Acc: {accuracy:.1f}%, Time: {elapsed_time:.1f}s")
        
        training_log.append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'accuracy': accuracy / 100,
            'lr': scheduler.get_last_lr()[0],
            'time': datetime.now().isoformat()
        })
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = Path('checkpoints/recognition/ultra_precise_rec_best.pth')
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': {
                    'best_loss': best_loss,
                    'best_accuracy': accuracy / 100
                },
                'config': {
                    'img_size': img_size,
                    'batch_size': batch_size,
                    'lr': lr,
                    'epochs': epochs,
                    'num_classes': num_classes
                },
                'training_completed': datetime.now().isoformat()
            }
            
            torch.save(checkpoint, save_path)
            file_size = save_path.stat().st_size / (1024 * 1024)
            print(f"  [OK] Best model saved: {save_path} ({file_size:.2f} MB)")
    
    # 保存训练日志
    log_path = Path('logs/recognition/training_log.json')
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"[OK] Recognizer training completed!")
    print(f"Best Loss: {best_loss:.4f}")
    print(f"Final Accuracy: {training_log[-1]['accuracy']*100:.1f}%")
    print("=" * 60)
    
    return training_log


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("Ultra-Face Recognition System - Recognition Training")
    print("=" * 60)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print("=" * 60 + "\n")
    
    # 训练识别器
    train_recognizer(epochs=5, batch_size=4, lr=0.01)
    
    # 打印总结
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nGenerated Files:")
    print("  - checkpoints/recognition/ultra_precise_rec_best.pth")
    print("  - logs/recognition/training_log.json")
    print("\nNote: This is trained with synthetic data for demonstration.")
    print("      For production use, train with CASIA-WebFace or MS1M.")


if __name__ == '__main__':
    main()
