"""
Face Database Builder - 人脸库构建器
=====================================
使用训练好的识别器构建人脸检索库
"""

import torch
import numpy as np
from pathlib import Path
import sys
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from rec.ultra_precise_rec import UltraPreciseRecognizer


class FaceDatabase:
    """人脸数据库"""
    
    def __init__(self, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        # 加载识别器
        print("Loading face recognizer...")
        model_path = Path('checkpoints/recognition/ultra_precise_rec_best.pth')
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            self.model = UltraPreciseRecognizer().to(device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"[OK] Loaded model from epoch {checkpoint['epoch']}")
        else:
            print("[WARN] Model not found, using untrained model")
            self.model = UltraPreciseRecognizer().to(device)
        
        self.model.eval()
        
        # 人脸库
        self.features = []
        self.metadata = []
        self.ids = []
    
    def add_face(self, image: np.ndarray, person_id: str, metadata: dict = None):
        """
        添加人脸到数据库
        
        Args:
            image: 人脸图像 (112x112 RGB)
            person_id: 人员 ID
            metadata: 元数据
        """
        # 提取特征
        with torch.no_grad():
            image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            
            output = self.model(image_tensor)
            feature = output['id'][0].cpu().numpy()
        
        # 归一化
        feature = feature / (np.linalg.norm(feature) + 1e-7)
        
        # 添加到数据库
        self.features.append(feature)
        self.ids.append(person_id)
        self.metadata.append(metadata or {})
        
        return len(self.features) - 1
    
    def search(self, image: np.ndarray, top_k=5):
        """
        搜索相似人脸
        
        Args:
            image: 查询图像
            top_k: 返回数量
            
        Returns:
            results: 搜索结果列表
        """
        if len(self.features) == 0:
            return []
        
        # 提取查询特征
        with torch.no_grad():
            image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            
            output = self.model(image_tensor)
            query_feature = output['id'][0].cpu().numpy()
        
        # 归一化
        query_feature = query_feature / (np.linalg.norm(query_feature) + 1e-7)
        
        # 计算相似度
        features_matrix = np.stack(self.features)
        similarities = np.dot(features_matrix, query_feature)
        
        # 排序
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'id': self.ids[idx],
                'score': float(similarities[idx]),
                'metadata': self.metadata[idx]
            })
        
        return results
    
    def save(self, path: str):
        """保存数据库"""
        data = {
            'features': [f.tolist() for f in self.features],
            'ids': self.ids,
            'metadata': self.metadata,
            'created': datetime.now().isoformat()
        }
        
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"[OK] Database saved to {path}")
        print(f"  - Total faces: {len(self.features)}")
    
    def load(self, path: str):
        """加载数据库"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.features = [np.array(f) for f in data['features']]
        self.ids = data['ids']
        self.metadata = data['metadata']
        
        print(f"[OK] Database loaded from {path}")
        print(f"  - Total faces: {len(self.features)}")
    
    def stats(self):
        """返回统计信息"""
        return {
            'total_faces': len(self.features),
            'unique_ids': len(set(self.ids)),
            'feature_dim': len(self.features[0]) if self.features else 0
        }


def build_demo_database():
    """构建演示数据库"""
    print("=" * 60)
    print("Building Demo Face Database")
    print("=" * 60)
    
    db = FaceDatabase()
    
    # 生成随机人脸特征（演示用）
    print("\nGenerating demo faces...")
    num_people = 100
    faces_per_person = 5
    
    for i in range(num_people):
        person_id = f"person_{i:03d}"
        for j in range(faces_per_person):
            # 生成随机人脸图像
            image = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
            
            metadata = {
                'name': f'Person {i}',
                'group': f'group_{i // 10}',
                'image_idx': j
            }
            
            db.add_face(image, person_id, metadata)
        
        if (i + 1) % 20 == 0:
            print(f"  Added {i + 1}/{num_people} people...")
    
    # 打印统计
    stats = db.stats()
    print(f"\n[OK] Database built:")
    print(f"  - Total faces: {stats['total_faces']}")
    print(f"  - Unique IDs: {stats['unique_ids']}")
    print(f"  - Feature dim: {stats['feature_dim']}")
    
    # 保存数据库
    db.save('indexes/demo_face_db.json')
    
    # 测试搜索
    print("\nTesting search...")
    query_image = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    results = db.search(query_image, top_k=5)
    
    print(f"\nSearch results:")
    for i, result in enumerate(results):
        print(f"  {i+1}. ID: {result['id']}, Score: {result['score']:.4f}")
    
    return db


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("Face Database Builder")
    print("=" * 60)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60 + "\n")
    
    # 构建演示数据库
    db = build_demo_database()
    
    # 总结
    print("\n" + "=" * 60)
    print("Database Build Complete!")
    print("=" * 60)
    print("\nGenerated Files:")
    print("  - indexes/demo_face_db.json")
    print("\nNext Steps:")
    print("  1. Add real face images to the database")
    print("  2. Test search with real query images")
    print("  3. Integrate with detection pipeline")


if __name__ == '__main__':
    main()
