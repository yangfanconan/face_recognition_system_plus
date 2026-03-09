"""
Search Engine Module (检索引擎模块)
====================================
封装检索接口，支持多种检索模式、缓存、并发控制
"""

import os
import time
import json
import threading
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import OrderedDict
import numpy as np

from billion_iadm import BillionScaleSearchEngine, SearchResult


# ============================================================================
# 缓存实现
# ============================================================================
class LRUCache:
    """
    LRU 缓存
    用于缓存热门查询结果
    
    Args:
        capacity: 缓存容量
    """
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[List[SearchResult]]:
        """获取缓存"""
        with self.lock:
            if key in self.cache:
                # 移动到末尾 (最近使用)
                self.cache.move_to_end(key)
                return self.cache[key]
            return None
    
    def put(self, key: str, value: List[SearchResult]):
        """放入缓存"""
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value
            
            # 超出容量时移除最旧的
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
    
    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
    
    def stats(self) -> Dict:
        """缓存统计"""
        return {
            'size': len(self.cache),
            'capacity': self.capacity,
            'hit_rate': self._compute_hit_rate(),
        }
    
    def _compute_hit_rate(self) -> float:
        """计算命中率 (需要外部记录)"""
        return 0.0


# ============================================================================
# 查询结果
# ============================================================================
@dataclass
class FaceSearchResult:
    """人脸检索结果"""
    
    # 人脸 ID
    face_id: int
    
    # 相似度分数
    score: float
    
    # 距离
    distance: float
    
    # 元数据
    metadata: Optional[Dict] = None
    
    # 边界框 (如果有)
    bbox: Optional[List[float]] = None
    
    # 关键点 (如果有)
    landmarks: Optional[List[List[float]]] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SearchResponse:
    """检索响应"""
    
    # 检索结果
    results: List[FaceSearchResult]
    
    # 检索耗时 (ms)
    latency_ms: float
    
    # 检索的库大小
    index_size: int
    
    # 使用的参数
    params: Dict
    
    # 缓存命中
    cache_hit: bool = False
    
    def to_dict(self) -> Dict:
        return {
            'results': [r.to_dict() for r in self.results],
            'latency_ms': self.latency_ms,
            'index_size': self.index_size,
            'params': self.params,
            'cache_hit': self.cache_hit,
        }


# ============================================================================
# 检索引擎配置
# ============================================================================
@dataclass
class SearchEngineConfig:
    """检索引擎配置"""
    
    # 索引路径
    index_path: str
    
    # 检索参数
    top_k: int = 10
    coarse_k: int = 1000
    rerank_k: int = 100
    
    # 缓存配置
    cache_enabled: bool = True
    cache_capacity: int = 10000
    
    # 并发配置
    max_concurrent_requests: int = 100
    num_search_threads: int = 8
    
    # 阈值配置
    score_threshold: float = 0.5
    distance_threshold: float = 0.6
    
    # 日志配置
    log_enabled: bool = True
    log_path: str = 'logs/search.log'
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'SearchEngineConfig':
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


# ============================================================================
# 检索引擎
# ============================================================================
class FaceSearchEngine:
    """
    人脸检索引擎
    封装亿级检索接口，支持缓存、并发、日志
    
    Args:
        config: 引擎配置
    """
    
    def __init__(self, config: SearchEngineConfig):
        self.config = config
        
        # 加载索引
        print(f"Loading index from {config.index_path}...")
        self.engine = BillionScaleSearchEngine()
        self.engine.load(config.index_path)
        
        # 缓存
        self.cache = LRUCache(capacity=config.cache_capacity) if config.cache_enabled else None
        
        # 并发控制
        self.semaphore = threading.Semaphore(config.max_concurrent_requests)
        
        # 统计
        self.request_count = 0
        self.cache_hits = 0
        self.total_latency = 0.0
        
        # 日志
        self.log_enabled = config.log_enabled
        if self.log_enabled:
            log_dir = Path(config.log_path).parent
            log_dir.mkdir(parents=True, exist_ok=True)
            self.log_file = open(config.log_path, 'a')
    
    def search(
        self,
        feature: np.ndarray,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        use_cache: bool = True,
        use_rerank: bool = True,
    ) -> SearchResponse:
        """
        单图检索
        
        Args:
            feature: 特征向量 [dim]
            top_k: 返回结果数
            score_threshold: 分数阈值
            use_cache: 是否使用缓存
            use_rerank: 是否使用重排
            
        Returns:
            response: 检索响应
        """
        start_time = time.perf_counter()
        
        # 参数
        top_k = top_k or self.config.top_k
        score_threshold = score_threshold or self.config.score_threshold
        
        # 生成缓存键
        cache_key = None
        if use_cache and self.cache is not None:
            cache_key = self._generate_cache_key(feature)
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                self.cache_hits += 1
                return SearchResponse(
                    results=cached_result,
                    latency_ms=0,
                    index_size=self.engine.total_vectors,
                    params={'top_k': top_k},
                    cache_hit=True,
                )
        
        # 并发控制
        with self.semaphore:
            # 检索
            results = self.engine.search(
                query=feature,
                k=top_k,
                coarse_k=self.config.coarse_k,
                rerank_k=self.config.rerank_k,
                use_rerank=use_rerank,
            )
        
        # 过滤和转换
        face_results = []
        for r in results:
            if r.score < score_threshold:
                continue
            
            face_result = FaceSearchResult(
                face_id=r.id,
                score=r.score,
                distance=r.distance,
                metadata=r.metadata,
            )
            face_results.append(face_result)
        
        # 计算延迟
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # 更新统计
        self.request_count += 1
        self.total_latency += latency_ms
        
        # 缓存结果
        if use_cache and self.cache is not None and len(face_results) > 0:
            self.cache.put(cache_key, face_results)
        
        # 日志
        if self.log_enabled:
            self._log_search(feature, face_results, latency_ms)
        
        # 构建响应
        response = SearchResponse(
            results=face_results,
            latency_ms=latency_ms,
            index_size=self.engine.total_vectors,
            params={
                'top_k': top_k,
                'score_threshold': score_threshold,
                'use_rerank': use_rerank,
            },
        )
        
        return response
    
    def batch_search(
        self,
        features: np.ndarray,
        top_k: Optional[int] = None,
        num_threads: Optional[int] = None,
    ) -> List[SearchResponse]:
        """
        批量检索
        
        Args:
            features: 特征向量 [N, dim]
            top_k: 返回结果数
            num_threads: 线程数
            
        Returns:
            responses: 检索响应列表
        """
        from concurrent.futures import ThreadPoolExecutor
        
        num_threads = num_threads or self.config.num_search_threads
        top_k = top_k or self.config.top_k
        
        def search_single(feature):
            return self.search(feature, top_k=top_k, use_cache=False)
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            responses = list(executor.map(search_single, features))
        
        return responses
    
    def search_with_metadata(
        self,
        feature: np.ndarray,
        metadata_filter: Dict,
        top_k: int = 10,
    ) -> SearchResponse:
        """
        带元数据过滤的检索
        
        Args:
            feature: 特征向量
            metadata_filter: 元数据过滤条件
            top_k: 返回结果数
            
        Returns:
            response: 检索响应
        """
        # 先检索更多候选
        raw_results = self.engine.search(
            query=feature,
            k=top_k * 10,  # 获取更多用于过滤
        )
        
        # 过滤元数据
        filtered_results = []
        for r in raw_results:
            if r.metadata is None:
                continue
            
            match = True
            for key, value in metadata_filter.items():
                if key not in r.metadata or r.metadata[key] != value:
                    match = False
                    break
            
            if match:
                face_result = FaceSearchResult(
                    face_id=r.id,
                    score=r.score,
                    distance=r.distance,
                    metadata=r.metadata,
                )
                filtered_results.append(face_result)
            
            if len(filtered_results) >= top_k:
                break
        
        # 计算延迟
        response = SearchResponse(
            results=filtered_results,
            latency_ms=0,
            index_size=self.engine.total_vectors,
            params={'top_k': top_k, 'metadata_filter': metadata_filter},
        )
        
        return response
    
    def add_face(
        self,
        feature: np.ndarray,
        metadata: Optional[Dict] = None,
    ) -> int:
        """
        添加人脸到索引
        
        Args:
            feature: 特征向量
            metadata: 元数据
            
        Returns:
            face_id: 分配的人脸 ID
        """
        ids = self.engine.add(
            vectors=feature.reshape(1, -1),
            metadata=[metadata] if metadata else None,
        )
        return ids[0]
    
    def delete_face(self, face_id: int) -> bool:
        """
        删除人脸
        
        Args:
            face_id: 人脸 ID
            
        Returns:
            success: 是否成功
        """
        # FAISS 不支持直接删除，需要标记或重建
        # 简化实现：从元数据中移除
        if face_id in self.engine.id_to_metadata:
            del self.engine.id_to_metadata[face_id]
            return True
        return False
    
    def get_face(self, face_id: int) -> Optional[Dict]:
        """
        获取人脸信息
        
        Args:
            face_id: 人脸 ID
            
        Returns:
            metadata: 元数据
        """
        return self.engine.id_to_metadata.get(face_id)
    
    def stats(self) -> Dict:
        """获取引擎统计"""
        avg_latency = self.total_latency / max(self.request_count, 1)
        cache_hit_rate = self.cache_hits / max(self.request_count, 1)
        
        return {
            'request_count': self.request_count,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': cache_hit_rate,
            'avg_latency_ms': avg_latency,
            'index_size': self.engine.total_vectors,
            'index_stats': self.engine.get_stats(),
            'cache_stats': self.cache.stats() if self.cache else None,
        }
    
    def _generate_cache_key(self, feature: np.ndarray) -> str:
        """生成缓存键"""
        # 使用特征哈希
        return str(hash(feature.tobytes()))
    
    def _log_search(
        self,
        feature: np.ndarray,
        results: List[FaceSearchResult],
        latency_ms: float,
    ):
        """记录搜索日志"""
        log_entry = {
            'timestamp': time.time(),
            'feature_dim': len(feature),
            'num_results': len(results),
            'latency_ms': latency_ms,
            'top_score': results[0].score if results else 0,
        }
        
        self.log_file.write(json.dumps(log_entry) + '\n')
        self.log_file.flush()
    
    def close(self):
        """关闭引擎"""
        if self.log_enabled and hasattr(self, 'log_file'):
            self.log_file.close()


# ============================================================================
# 1:N 识别器
# ============================================================================
class Face1NRecognizer:
    """
    1:N 人脸识别器
    基于检索引擎实现
    
    Args:
        search_engine: 检索引擎
    """
    
    def __init__(self, search_engine: FaceSearchEngine):
        self.engine = search_engine
        self.threshold = search_engine.config.score_threshold
    
    def recognize(
        self,
        feature: np.ndarray,
        top_k: int = 1,
    ) -> Tuple[Optional[int], float, List[FaceSearchResult]]:
        """
        识别人脸
        
        Args:
            feature: 特征向量
            top_k: 候选数
            
        Returns:
            face_id: 识别结果 (None 表示未知)
            confidence: 置信度
            candidates: 候选列表
        """
        response = self.engine.search(feature, top_k=top_k)
        
        if not response.results:
            return None, 0.0, []
        
        top_result = response.results[0]
        
        if top_result.score >= self.threshold:
            return top_result.face_id, top_result.score, response.results
        else:
            return None, top_result.score, response.results
    
    def verify(
        self,
        feature1: np.ndarray,
        feature2: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Tuple[bool, float]:
        """
        人脸验证 (1:1)
        
        Args:
            feature1: 特征 1
            feature2: 特征 2
            threshold: 阈值
            
        Returns:
            is_same: 是否同一人
            similarity: 相似度
        """
        threshold = threshold or self.threshold
        
        # 计算余弦相似度
        similarity = np.dot(feature1, feature2) / (
            np.linalg.norm(feature1) * np.linalg.norm(feature2) + 1e-7
        )
        
        return similarity >= threshold, similarity
    
    def identify_batch(
        self,
        features: np.ndarray,
        top_k: int = 1,
    ) -> List[Tuple[Optional[int], float]]:
        """
        批量识别
        
        Args:
            features: 特征向量 [N, dim]
            top_k: 候选数
            
        Returns:
            results: (face_id, confidence) 列表
        """
        responses = self.engine.batch_search(features, top_k=top_k)
        
        results = []
        for response in responses:
            if response.results:
                top = response.results[0]
                if top.score >= self.threshold:
                    results.append((top.face_id, top.score))
                else:
                    results.append((None, top.score))
            else:
                results.append((None, 0.0))
        
        return results


# ============================================================================
# 入口函数
# ============================================================================
def create_search_engine(
    index_path: str,
    top_k: int = 10,
    cache_enabled: bool = True,
) -> FaceSearchEngine:
    """
    创建检索引擎
    
    Args:
        index_path: 索引路径
        top_k: 默认返回结果数
        cache_enabled: 是否启用缓存
        
    Returns:
        engine: 检索引擎
    """
    config = SearchEngineConfig(
        index_path=index_path,
        top_k=top_k,
        cache_enabled=cache_enabled,
    )
    
    return FaceSearchEngine(config)


def create_1n_recognizer(
    index_path: str,
    threshold: float = 0.5,
) -> Face1NRecognizer:
    """
    创建 1:N 识别器
    
    Args:
        index_path: 索引路径
        threshold: 识别阈值
        
    Returns:
        recognizer: 识别器
    """
    config = SearchEngineConfig(
        index_path=index_path,
        score_threshold=threshold,
    )
    
    engine = FaceSearchEngine(config)
    return Face1NRecognizer(engine)


if __name__ == '__main__':
    # 示例
    print("Search Engine Module")
    print("====================")
    
    # 创建配置
    config = SearchEngineConfig(
        index_path='test_index',
        top_k=5,
    )
    
    print(f"Config: {config.to_dict()}")
