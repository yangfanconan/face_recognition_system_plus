/**
 * @file search_util.h
 * @brief Billion-Scale Face Search Utility Header (亿级人脸检索工具头文件)
 * 
 * 高性能检索工具
 * - FAISS C++ 接口封装
 * - HNSW 索引优化
 * - PQ 量化
 * - 多线程并发检索
 * 
 * @version 1.0
 * @date 2026-03-09
 */

#ifndef SEARCH_UTIL_H
#define SEARCH_UTIL_H

#include <vector>
#include <string>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexFlat.h>
#include <faiss/MetaIndexes.h>

namespace face {

/**
 * @brief 检索结果结构体
 */
struct SearchResultSet {
    std::vector<int64_t> ids;       ///< 结果 ID
    std::vector<float> distances;   ///< 距离
    std::vector<float> scores;      ///< 相似度分数
    
    size_t size() const { return ids.size(); }
    
    void clear() {
        ids.clear();
        distances.clear();
        scores.clear();
    }
};

/**
 * @brief 检索配置
 */
struct SearchConfig {
    int dim = 512;                   ///< 特征维度
    int top_k = 10;                  ///< 返回结果数
    
    // HNSW 配置
    int hnsw_M = 64;                 ///< HNSW 连接数
    int hnsw_ef_construction = 200;  ///< HNSW 构建参数
    int hnsw_ef_search = 64;         ///< HNSW 搜索参数
    
    // IVF 配置
    int ivf_nlist = 4096;            ///< IVF 簇数
    int ivf_nprobe = 32;             ///< IVF 探测数
    
    // PQ 配置
    int pq_m = 32;                   ///< PQ 子量化器数
    int pq_nbits = 8;                ///< PQ 每子段比特数
    
    // 性能配置
    int num_threads = 8;             ///< 线程数
    bool use_gpu = false;            ///< 是否使用 GPU
    int gpu_id = 0;                  ///< GPU ID
    
    // 内存配置
    size_t max_memory_mb = 8192;     ///< 最大内存 (MB)
};

/**
 * @brief 元数据结构
 */
struct FaceMetadata {
    int64_t id;                      ///< 人脸 ID
    std::string name;                ///< 姓名
    std::string image_path;          ///< 图像路径
    std::vector<float> bbox;         ///< 边界框
    int64_t timestamp;               ///< 时间戳
    std::map<std::string, std::string> extra;  ///< 额外信息
};

/**
 * @brief 亿级检索引擎类
 * 
 * 支持分层检索：IVF 粗排 -> HNSW 精排
 * 
 * 使用示例:
 * @code
 * SearchConfig config;
 * BillionScaleSearchEngine engine(config);
 * 
 * // 添加数据
 * engine.add(vectors, metadata);
 * 
 * // 检索
 * SearchResultSet results;
 * engine.search(query, results);
 * @endcode
 */
class BillionScaleSearchEngine {
public:
    /**
     * @brief 构造函数
     * @param config 检索配置
     */
    explicit BillionScaleSearchEngine(const SearchConfig& config = SearchConfig());
    
    /**
     * @brief 析构函数
     */
    ~BillionScaleSearchEngine();
    
    /**
     * @brief 添加向量
     * @param vectors 特征向量 [N, dim]
     * @param metadata 元数据
     * @return 起始 ID
     */
    int64_t add(
        const std::vector<std::vector<float>>& vectors,
        const std::vector<FaceMetadata>& metadata = {}
    );
    
    /**
     * @brief 添加单个向量
     * @param vector 特征向量
     * @param metadata 元数据
     * @return 分配的 ID
     */
    int64_t add_single(
        const std::vector<float>& vector,
        const FaceMetadata& metadata = {}
    );
    
    /**
     * @brief 构建索引 (训练 + 添加)
     * @param vectors 特征向量
     * @return 是否成功
     */
    bool build(const std::vector<std::vector<float>>& vectors);
    
    /**
     * @brief 检索
     * @param query 查询向量
     * @param results 检索结果
     * @param k 返回数量
     * @return 是否成功
     */
    bool search(
        const std::vector<float>& query,
        SearchResultSet& results,
        int k = -1
    );
    
    /**
     * @brief 批量检索
     * @param queries 查询向量批次
     * @param all_results 所有结果
     * @param k 每查询返回数量
     * @return 是否成功
     */
    bool batch_search(
        const std::vector<std::vector<float>>& queries,
        std::vector<SearchResultSet>& all_results,
        int k = -1
    );
    
    /**
     * @brief 保存索引
     * @param path 保存路径
     * @return 是否成功
     */
    bool save(const std::string& path);
    
    /**
     * @brief 加载索引
     * @param path 加载路径
     * @return 是否成功
     */
    bool load(const std::string& path);
    
    /**
     * @brief 获取索引大小
     * @return 向量数量
     */
    size_t size() const;
    
    /**
     * @brief 获取元数据
     * @param id 人脸 ID
     * @return 元数据指针
     */
    const FaceMetadata* get_metadata(int64_t id) const;
    
    /**
     * @brief 删除人脸
     * @param id 人脸 ID
     * @return 是否成功
     */
    bool remove(int64_t id);
    
    /**
     * @brief 清空索引
     */
    void clear();
    
    /**
     * @brief 获取统计信息
     * @return 统计信息
     */
    std::map<std::string, double> get_stats() const;

private:
    /**
     * @brief 创建 IVF 索引
     */
    void create_ivf_index();
    
    /**
     * @brief 创建 HNSW 索引
     */
    void create_hnsw_index();
    
    /**
     * @brief 训练索引
     * @param vectors 训练数据
     */
    void train_index(const std::vector<std::vector<float>>& vectors);

private:
    SearchConfig config_;                          ///< 配置
    
    // 主索引 (IVF-PQ)
    std::unique_ptr<faiss::IndexIVFPQ> ivf_index_;
    
    // 精排索引 (HNSW)
    std::unique_ptr<faiss::IndexHNSWFlat> hnsw_index_;
    
    // 元数据存储
    std::map<int64_t, FaceMetadata> metadata_;
    
    // 线程安全
    mutable std::shared_mutex mutex_;
    
    // ID 分配
    int64_t next_id_ = 0;
    
    // 是否已训练
    bool is_trained_ = false;
};

/**
 * @brief 检索管理器 (单例)
 */
class SearchManager {
public:
    /**
     * @brief 获取单例实例
     */
    static SearchManager& instance();
    
    /**
     * @brief 创建检索引擎
     * @param id 引擎 ID
     * @param config 配置
     * @return 是否成功
     */
    bool create(int id, const SearchConfig& config);
    
    /**
     * @brief 加载索引
     * @param id 引擎 ID
     * @param path 索引路径
     * @return 是否成功
     */
    bool load(int id, const std::string& path);
    
    /**
     * @brief 获取引擎
     * @param id 引擎 ID
     * @return 引擎指针
     */
    BillionScaleSearchEngine* get(int id);
    
    /**
     * @brief 销毁引擎
     * @param id 引擎 ID
     */
    void destroy(int id);

private:
    SearchManager() = default;
    ~SearchManager();
    
    SearchManager(const SearchManager&) = delete;
    SearchManager& operator=(const SearchManager&) = delete;
    
    std::map<int, std::unique_ptr<BillionScaleSearchEngine>> engines_;
};

// ============================================================================
// 工具函数
// ============================================================================

/**
 * @brief 计算余弦相似度
 * @param v1 向量 1
 * @param v2 向量 2
 * @return 相似度
 */
inline float cosine_similarity(
    const std::vector<float>& v1,
    const std::vector<float>& v2
) {
    float dot = 0.0f;
    float norm1 = 0.0f;
    float norm2 = 0.0f;
    
    for (size_t i = 0; i < v1.size() && i < v2.size(); ++i) {
        dot += v1[i] * v2[i];
        norm1 += v1[i] * v1[i];
        norm2 += v2[i] * v2[i];
    }
    
    float norm = std::sqrt(norm1 * norm2);
    return (norm > 1e-7f) ? (dot / norm) : 0.0f;
}

/**
 * @brief L2 归一化
 * @param v 向量
 */
inline void l2_normalize(std::vector<float>& v) {
    float norm = 0.0f;
    for (float x : v) norm += x * x;
    norm = std::sqrt(norm);
    if (norm > 1e-7f) {
        for (float& x : v) x /= norm;
    }
}

/**
 * @brief 将距离转换为相似度
 * @param distance 欧氏距离
 * @return 相似度
 */
inline float distance_to_similarity(float distance) {
    return 1.0f / (1.0f + distance);
}

} // namespace face

#endif // SEARCH_UTIL_H
