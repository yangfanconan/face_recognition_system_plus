"""
Index Builder Module (索引构建模块)
====================================
支持批量构建、增量更新、分布式构建
"""

import os
import json
import pickle
import numpy as np
from typing import List, Dict, Tuple, Optional, Iterator
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing

from billion_iadm import (
    BillionScaleSearchEngine,
    ProductQuantizer,
    IVFIndex,
    HNSWIndex,
    DistributedIndexBuilder,
)


# ============================================================================
# 数据配置
# ============================================================================
@dataclass
class IndexBuildConfig:
    """索引构建配置"""
    
    # 数据配置
    dim: int = 512
    batch_size: int = 10000
    
    # PQ 配置
    pq_code_size: int = 64
    pq_nbits: int = 8
    
    # IVF 配置
    ivf_nlist: int = 65536
    ivf_nprobe: int = 32
    
    # HNSW 配置
    hnsw_M: int = 128
    hnsw_ef_construction: int = 200
    
    # 分布式配置
    num_shards: int = 8
    num_workers: int = 8
    
    # 内存配置
    max_memory_gb: float = 32.0
    
    def to_dict(self) -> Dict:
        return self.__dict__
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'IndexBuildConfig':
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


# ============================================================================
# 特征数据加载器
# ============================================================================
class FeatureDataLoader:
    """
    特征数据加载器
    支持多种数据源：numpy 文件、数据库、文件列表
    
    Args:
        data_source: 数据源路径
        batch_size: 批次大小
    """
    
    def __init__(
        self,
        data_source: str,
        batch_size: int = 10000,
    ):
        self.data_source = Path(data_source)
        self.batch_size = batch_size
        
        # 数据源类型
        if self.data_source.suffix == '.npy':
            self.source_type = 'npy'
        elif self.data_source.is_dir():
            self.source_type = 'dir'
        else:
            self.source_type = 'list'
        
        self.current_index = 0
        self.total_count = self._count_total()
    
    def _count_total(self) -> int:
        """统计总数据量"""
        if self.source_type == 'npy':
            data = np.load(self.data_source, mmap_mode='r')
            return len(data)
        elif self.source_type == 'dir':
            files = list(self.data_source.glob('*.npy'))
            return sum(np.load(f, mmap_mode='r').shape[0] for f in files)
        else:
            with open(self.data_source, 'r') as f:
                return sum(1 for _ in f)
    
    def __iter__(self) -> Iterator[Tuple[np.ndarray, List[Dict]]]:
        """迭代批次数据"""
        self.current_index = 0
        
        if self.source_type == 'npy':
            data = np.load(self.data_source, mmap_mode='r')
            for i in range(0, len(data), self.batch_size):
                batch = data[i:i + self.batch_size]
                metadata = [{'index': i + j} for j in range(len(batch))]
                yield batch, metadata
        elif self.source_type == 'dir':
            files = sorted(self.data_source.glob('*.npy'))
            accumulated_metadata = []
            accumulated_data = []
            
            for file in files:
                file_data = np.load(file, mmap_mode='r')
                file_meta = [{'file': file.name, 'index': i} for i in range(len(file_data))]
                
                accumulated_data.append(file_data)
                accumulated_metadata.extend(file_meta)
                
                if len(accumulated_data) * self.batch_size >= self.batch_size:
                    batch = np.vstack(accumulated_data)
                    yield batch, accumulated_metadata[:len(batch)]
                    accumulated_data = []
                    accumulated_metadata = accumulated_metadata[len(batch):]
        else:
            # 文件列表模式
            with open(self.data_source, 'r') as f:
                batch_paths = []
                batch_metadata = []
                
                for line in f:
                    path = line.strip()
                    batch_paths.append(path)
                    batch_metadata.append({'path': path})
                    
                    if len(batch_paths) >= self.batch_size:
                        # 加载批次数据
                        batch = self._load_batch(batch_paths)
                        yield batch, batch_metadata
                        batch_paths = []
                        batch_metadata = []
    
    def _load_batch(self, paths: List[str]) -> np.ndarray:
        """加载批次数据"""
        # 简化实现：从文件加载特征
        features = []
        for path in paths:
            # 实际实现需要根据具体格式调整
            if Path(path).exists():
                feat = np.load(path)
                features.append(feat)
        return np.vstack(features) if features else np.array([])
    
    def __len__(self) -> int:
        return (self.total_count + self.batch_size - 1) // self.batch_size


# ============================================================================
# 索引构建器
# ============================================================================
class IndexBuilder:
    """
    索引构建器
    支持全量构建、增量更新、分布式构建
    
    Args:
        config: 构建配置
    """
    
    def __init__(self, config: IndexBuildConfig):
        self.config = config
        self.engine = None
        self.built = False
    
    def build(
        self,
        data_loader: FeatureDataLoader,
        save_path: str,
    ) -> BillionScaleSearchEngine:
        """
        构建索引
        
        Args:
            data_loader: 数据加载器
            save_path: 保存路径
            
        Returns:
            engine: 检索引擎
        """
        print(f"Starting index build with config: {self.config.to_dict()}")
        
        # 创建引擎
        self.engine = BillionScaleSearchEngine(
            dim=self.config.dim,
            pq_code_size=self.config.pq_code_size,
            ivf_nlist=self.config.ivf_nlist,
            hnsw_M=self.config.hnsw_M,
        )
        
        # 收集所有数据
        all_vectors = []
        all_metadata = []
        
        print("Loading data...")
        for batch_vectors, batch_metadata in data_loader:
            all_vectors.append(batch_vectors)
            all_metadata.extend(batch_metadata)
            print(f"  Loaded {len(all_metadata)} vectors")
        
        all_vectors = np.vstack(all_vectors)
        print(f"Total vectors: {len(all_vectors)}")
        
        # 添加到引擎
        print("Adding vectors to index...")
        self.engine.add(all_vectors, all_metadata)
        
        # 构建 HNSW
        print("Building HNSW index...")
        self.engine.build_hnsw(all_vectors)
        
        # 保存
        print(f"Saving index to {save_path}...")
        self.engine.save(save_path)
        
        self.built = True
        
        # 释放内存
        del all_vectors
        
        return self.engine
    
    def build_distributed(
        self,
        data_loader: FeatureDataLoader,
        base_save_path: str,
    ) -> List[BillionScaleSearchEngine]:
        """
        分布式构建索引
        
        Args:
            data_loader: 数据加载器
            base_save_path: 基础保存路径
            
        Returns:
            engines: 检索引擎列表
        """
        print(f"Starting distributed build with {self.config.num_shards} shards")
        
        # 收集所有数据
        all_vectors = []
        all_metadata = []
        
        for batch_vectors, batch_metadata in data_loader:
            all_vectors.append(batch_vectors)
            all_metadata.extend(batch_metadata)
        
        all_vectors = np.vstack(all_vectors)
        n_vectors = len(all_vectors)
        
        # 分片
        shard_size = (n_vectors + self.config.num_shards - 1) // self.config.num_shards
        
        builders = []
        for shard_id in range(self.config.num_shards):
            start = shard_id * shard_size
            end = min(start + shard_size, n_vectors)
            
            if start >= n_vectors:
                break
            
            shard_vectors = all_vectors[start:end]
            shard_metadata = all_metadata[start:end]
            
            builder = DistributedIndexBuilder(
                num_shards=1,
                dim=self.config.dim,
            )
            builder.shards = [[shard_vectors]]
            builders.append((builder, shard_id, f"{base_save_path}/shard_{shard_id}"))
        
        # 并行构建
        engines = []
        with ProcessPoolExecutor(max_workers=self.config.num_workers) as executor:
            futures = []
            for builder, shard_id, save_path in builders:
                future = executor.submit(
                    self._build_single_shard,
                    builder,
                    shard_id,
                    save_path,
                )
                futures.append(future)
            
            for future in futures:
                engine = future.result()
                engines.append(engine)
        
        return engines
    
    def _build_single_shard(
        self,
        builder: DistributedIndexBuilder,
        shard_id: int,
        save_path: str,
    ) -> BillionScaleSearchEngine:
        """构建单个分片"""
        engine = builder.build_shard(shard_id, save_path)
        return engine
    
    def update(
        self,
        new_vectors: np.ndarray,
        new_metadata: List[Dict],
        save_path: Optional[str] = None,
    ):
        """
        增量更新索引
        
        Args:
            new_vectors: 新增向量
            new_metadata: 新增元数据
            save_path: 保存路径
        """
        assert self.built, "Index not built yet"
        
        print(f"Adding {len(new_vectors)} new vectors...")
        self.engine.add(new_vectors, new_metadata)
        
        if save_path:
            self.engine.save(save_path)
    
    def merge(
        self,
        index_paths: List[str],
        save_path: str,
    ):
        """
        合并多个索引
        
        Args:
            index_paths: 索引路径列表
            save_path: 合并后保存路径
        """
        print(f"Merging {len(index_paths)} indexes...")
        
        # 加载所有索引
        engines = []
        all_vectors = []
        all_metadata = []
        
        for path in index_paths:
            engine = BillionScaleSearchEngine()
            engine.load(path)
            engines.append(engine)
            
            # 这里需要获取原始向量用于重建
            # 简化实现：假设向量可从元数据重建
        
        # 创建新引擎
        merged_engine = BillionScaleSearchEngine(dim=self.config.dim)
        
        # 合并逻辑 (简化)
        # 实际实现需要合并 IVF 和 HNSW 索引
        
        merged_engine.save(save_path)
        print(f"Merged index saved to {save_path}")


# ============================================================================
# 索引质量评估
# ============================================================================
class IndexEvaluator:
    """
    索引质量评估器
    
    Args:
        engine: 检索引擎
    """
    
    def __init__(self, engine: BillionScaleSearchEngine):
        self.engine = engine
    
    def evaluate_recall(
        self,
        queries: np.ndarray,
        ground_truth: np.ndarray,
        k: int = 10,
    ) -> float:
        """
        评估召回率
        
        Args:
            queries: 查询向量
            ground_truth: 真实最近邻 [N, k]
            k: 检索数量
            
        Returns:
            recall: 召回率
        """
        recalls = []
        
        for i, query in enumerate(queries):
            results = self.engine.search(query, k=k)
            result_ids = set(r.id for r in results)
            gt_ids = set(ground_truth[i])
            
            intersection = result_ids & gt_ids
            recall = len(intersection) / len(gt_ids) if gt_ids else 0
            recalls.append(recall)
        
        return np.mean(recalls)
    
    def evaluate_latency(
        self,
        queries: np.ndarray,
        k: int = 10,
        n_warmup: int = 10,
    ) -> Dict:
        """
        评估延迟
        
        Args:
            queries: 查询向量
            k: 检索数量
            n_warmup: 预热次数
            
        Returns:
            latency_stats: 延迟统计
        """
        import time
        
        latencies = []
        
        # 预热
        for i in range(n_warmup):
            query = queries[i % len(queries)]
            self.engine.search(query, k=k)
        
        # 测试
        for query in queries:
            start = time.perf_counter()
            self.engine.search(query, k=k)
            end = time.perf_counter()
            
            latencies.append((end - start) * 1000)  # ms
        
        return {
            'mean': np.mean(latencies),
            'median': np.median(latencies),
            'p95': np.percentile(latencies, 95),
            'p99': np.percentile(latencies, 99),
            'min': np.min(latencies),
            'max': np.max(latencies),
        }
    
    def evaluate_throughput(
        self,
        queries: np.ndarray,
        k: int = 10,
        duration_seconds: float = 10.0,
    ) -> float:
        """
        评估吞吐量
        
        Args:
            queries: 查询向量
            k: 检索数量
            duration_seconds: 测试时长
            
        Returns:
            qps: 每秒查询数
        """
        import time
        
        start_time = time.time()
        query_count = 0
        
        while time.time() - start_time < duration_seconds:
            query = queries[query_count % len(queries)]
            self.engine.search(query, k=k)
            query_count += 1
        
        elapsed = time.time() - start_time
        qps = query_count / elapsed
        
        return qps
    
    def full_evaluation(
        self,
        queries: np.ndarray,
        ground_truth: Optional[np.ndarray] = None,
        k: int = 10,
    ) -> Dict:
        """
        完整评估
        
        Args:
            queries: 查询向量
            ground_truth: 真实最近邻
            k: 检索数量
            
        Returns:
            metrics: 评估指标
        """
        metrics = {}
        
        # 延迟
        print("Evaluating latency...")
        metrics['latency'] = self.evaluate_latency(queries, k)
        
        # 吞吐量
        print("Evaluating throughput...")
        metrics['throughput'] = self.evaluate_throughput(queries, k)
        
        # 召回率
        if ground_truth is not None:
            print("Evaluating recall...")
            metrics['recall'] = self.evaluate_recall(queries, ground_truth, k)
        
        # 索引统计
        metrics['index_stats'] = self.engine.get_stats()
        
        return metrics


# ============================================================================
# 构建入口函数
# ============================================================================
def build_index(
    data_source: str,
    save_path: str,
    dim: int = 512,
    batch_size: int = 10000,
    distributed: bool = False,
) -> BillionScaleSearchEngine:
    """
    构建索引入口函数
    
    Args:
        data_source: 数据源路径
        save_path: 保存路径
        dim: 特征维度
        batch_size: 批次大小
        distributed: 是否分布式构建
        
    Returns:
        engine: 检索引擎
    """
    # 配置
    config = IndexBuildConfig(
        dim=dim,
        batch_size=batch_size,
        num_shards=8 if distributed else 1,
    )
    
    # 数据加载器
    data_loader = FeatureDataLoader(data_source, batch_size=batch_size)
    
    # 构建器
    builder = IndexBuilder(config)
    
    if distributed:
        engines = builder.build_distributed(data_loader, save_path)
        return engines[0] if engines else None
    else:
        engine = builder.build(data_loader, save_path)
        return engine


def evaluate_index(
    index_path: str,
    queries: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
) -> Dict:
    """
    评估索引
    
    Args:
        index_path: 索引路径
        queries: 查询向量
        ground_truth: 真实最近邻
        
    Returns:
        metrics: 评估指标
    """
    engine = BillionScaleSearchEngine()
    engine.load(index_path)
    
    evaluator = IndexEvaluator(engine)
    metrics = evaluator.full_evaluation(queries, ground_truth)
    
    return metrics


if __name__ == '__main__':
    # 示例：构建索引
    config = IndexBuildConfig(dim=512)
    config.save('index_config.json')
    
    print("Index build config saved.")
    print(f"Config: {config.to_dict()}")
