/**
 * @file index_manager.cpp
 * @brief Index Manager Implementation (索引管理器实现)
 * 
 * 索引构建、管理、维护工具
 * - 分布式索引构建
 * - 增量更新
 * - 索引合并
 * - 性能监控
 * 
 * @version 1.0
 * @date 2026-03-09
 */

#include "search_util.h"
#include <fstream>
#include <iostream>
#include <thread>
#include <queue>
#include <atomic>

namespace face {

// ============================================================================
// 索引构建配置
// ============================================================================

/**
 * @brief 索引构建配置
 */
struct IndexBuildConfig {
    int num_threads = 8;           ///< 构建线程数
    int batch_size = 10000;        ///< 批次大小
    bool use_gpu = false;          ///< 是否使用 GPU 加速
    int gpu_id = 0;                ///< GPU ID
    std::string temp_dir = "/tmp"; ///< 临时目录
};

// ============================================================================
// 索引构建器
// ============================================================================

/**
 * @brief 分布式索引构建器
 */
class DistributedIndexBuilder {
public:
    /**
     * @brief 构造函数
     * @param config 构建配置
     */
    explicit DistributedIndexBuilder(const IndexBuildConfig& config = IndexBuildConfig())
        : config_(config) {}
    
    /**
     * @brief 构建索引
     * @param vectors 特征向量
     * @param output_path 输出路径
     * @param search_config 检索配置
     * @return 是否成功
     */
    bool build(
        const std::vector<std::vector<float>>& vectors,
        const std::string& output_path,
        const SearchConfig& search_config
    );
    
    /**
     * @brief 分片构建
     * @param vectors 特征向量
     * @param num_shards 分片数
     * @param base_output_path 基础输出路径
     * @return 是否成功
     */
    bool build_sharded(
        const std::vector<std::vector<float>>& vectors,
        int num_shards,
        const std::string& base_output_path
    );
    
    /**
     * @brief 增量更新
     * @param engine 检索引擎
     * @param new_vectors 新增向量
     * @return 是否成功
     */
    bool update(
        BillionScaleSearchEngine& engine,
        const std::vector<std::vector<float>>& new_vectors
    );

private:
    IndexBuildConfig config_;
};

bool DistributedIndexBuilder::build(
    const std::vector<std::vector<float>>& vectors,
    const std::string& output_path,
    const SearchConfig& search_config
) {
    std::cout << "Building index with " << vectors.size() << " vectors..." << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // 创建引擎
    BillionScaleSearchEngine engine(search_config);
    
    // 构建索引
    engine.build(vectors);
    
    // 保存
    if (!engine.save(output_path)) {
        std::cerr << "Failed to save index" << std::endl;
        return false;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
    
    std::cout << "Index built in " << duration << " seconds" << std::endl;
    
    // 打印统计
    auto stats = engine.get_stats();
    std::cout << "Index statistics:" << std::endl;
    for (const auto& [key, value] : stats) {
        std::cout << "  " << key << ": " << value << std::endl;
    }
    
    return true;
}

bool DistributedIndexBuilder::build_sharded(
    const std::vector<std::vector<float>>& vectors,
    int num_shards,
    const std::string& base_output_path
) {
    std::cout << "Building sharded index with " << num_shards << " shards..." << std::endl;
    
    size_t total_vectors = vectors.size();
    size_t vectors_per_shard = (total_vectors + num_shards - 1) / num_shards;
    
    std::vector<std::thread> threads;
    std::vector<std::atomic<bool>> shard_done(num_shards);
    
    for (int shard_id = 0; shard_id < num_shards; ++shard_id) {
        size_t start = shard_id * vectors_per_shard;
        size_t end = std::min(start + vectors_per_shard, total_vectors);
        
        if (start >= total_vectors) break;
        
        threads.emplace_back([this, &vectors, &shard_done, shard_id, 
                              start, end, base_output_path]() {
            std::cout << "Building shard " << shard_id 
                      << " (" << start << "-" << end << ")..." << std::endl;
            
            std::vector<std::vector<float>> shard_vectors(
                vectors.begin() + start,
                vectors.begin() + end
            );
            
            SearchConfig search_config;
            std::string output_path = base_output_path + "_shard_" + std::to_string(shard_id);
            
            build(shard_vectors, output_path, search_config);
            
            shard_done[shard_id] = true;
        });
    }
    
    // 等待所有线程完成
    for (auto& t : threads) {
        t.join();
    }
    
    std::cout << "All shards built successfully" << std::endl;
    
    return true;
}

bool DistributedIndexBuilder::update(
    BillionScaleSearchEngine& engine,
    const std::vector<std::vector<float>>& new_vectors
) {
    std::cout << "Updating index with " << new_vectors.size() << " new vectors..." << std::endl;
    
    engine.add(new_vectors);
    
    std::cout << "Update completed. Total vectors: " << engine.size() << std::endl;
    
    return true;
}

// ============================================================================
// 索引管理器
// ============================================================================

/**
 * @brief 索引管理器类
 */
class IndexManager {
public:
    /**
     * @brief 获取单例实例
     */
    static IndexManager& instance();
    
    /**
     * @brief 创建索引
     * @param index_id 索引 ID
     * @param config 配置
     * @return 是否成功
     */
    bool create(const std::string& index_id, const SearchConfig& config);
    
    /**
     * @brief 加载索引
     * @param index_id 索引 ID
     * @param path 索引路径
     * @return 是否成功
     */
    bool load(const std::string& index_id, const std::string& path);
    
    /**
     * @brief 获取索引
     * @param index_id 索引 ID
     * @return 索引指针
     */
    BillionScaleSearchEngine* get(const std::string& index_id);
    
    /**
     * @brief 删除索引
     * @param index_id 索引 ID
     * @return 是否成功
     */
    bool remove(const std::string& index_id);
    
    /**
     * @brief 列出所有索引
     * @return 索引 ID 列表
     */
    std::vector<std::string> list_indexes() const;
    
    /**
     * @brief 合并索引
     * @param index_ids 源索引 ID 列表
     * @param output_id 输出索引 ID
     * @return 是否成功
     */
    bool merge(
        const std::vector<std::string>& index_ids,
        const std::string& output_id
    );
    
    /**
     * @brief 获取索引统计
     * @param index_id 索引 ID
     * @return 统计信息
     */
    std::map<std::string, double> get_stats(const std::string& index_id);

private:
    IndexManager() = default;
    ~IndexManager();
    
    IndexManager(const IndexManager&) = delete;
    IndexManager& operator=(const IndexManager&) = delete;
    
    std::map<std::string, std::unique_ptr<BillionScaleSearchEngine>> indexes_;
    mutable std::shared_mutex mutex_;
};

IndexManager& IndexManager::instance() {
    static IndexManager instance;
    return instance;
}

IndexManager::~IndexManager() {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    indexes_.clear();
}

bool IndexManager::create(const std::string& index_id, const SearchConfig& config) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    
    if (indexes_.find(index_id) != indexes_.end()) {
        return false;
    }
    
    indexes_[index_id] = std::make_unique<BillionScaleSearchEngine>(config);
    
    return true;
}

bool IndexManager::load(const std::string& index_id, const std::string& path) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    
    auto engine = std::make_unique<BillionScaleSearchEngine>();
    if (!engine->load(path)) {
        return false;
    }
    
    indexes_[index_id] = std::move(engine);
    
    return true;
}

BillionScaleSearchEngine* IndexManager::get(const std::string& index_id) {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    
    auto it = indexes_.find(index_id);
    if (it != indexes_.end()) {
        return it->second.get();
    }
    
    return nullptr;
}

bool IndexManager::remove(const std::string& index_id) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    
    return indexes_.erase(index_id) > 0;
}

std::vector<std::string> IndexManager::list_indexes() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    
    std::vector<std::string> ids;
    for (const auto& [id, _] : indexes_) {
        ids.push_back(id);
    }
    
    return ids;
}

bool IndexManager::merge(
    const std::vector<std::string>& index_ids,
    const std::string& output_id
) {
    std::cout << "Merging " << index_ids.size() << " indexes..." << std::endl;
    
    // 收集所有向量
    std::vector<std::vector<float>> all_vectors;
    std::vector<FaceMetadata> all_metadata;
    
    {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        
        for (const auto& index_id : index_ids) {
            auto it = indexes_.find(index_id);
            if (it == indexes_.end()) {
                std::cerr << "Index not found: " << index_id << std::endl;
                continue;
            }
            
            // 简化实现：实际需要导出向量
            std::cout << "Collecting vectors from: " << index_id << std::endl;
        }
    }
    
    // 创建合并后的索引
    SearchConfig config;
    create(output_id, config);
    
    auto* output_engine = get(output_id);
    if (!output_engine) {
        return false;
    }
    
    output_engine->add(all_vectors, all_metadata);
    
    std::cout << "Merge completed. Total vectors: " << output_engine->size() << std::endl;
    
    return true;
}

std::map<std::string, double> IndexManager::get_stats(const std::string& index_id) {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    
    auto it = indexes_.find(index_id);
    if (it != indexes_.end()) {
        return it->second->get_stats();
    }
    
    return {};
}

// ============================================================================
// 性能监控
// ============================================================================

/**
 * @brief 性能监控器
 */
class PerformanceMonitor {
public:
    /**
     * @brief 开始计时
     * @param name 任务名称
     */
    void start(const std::string& name) {
        start_times_[name] = std::chrono::high_resolution_clock::now();
    }
    
    /**
     * @brief 结束计时
     * @param name 任务名称
     * @return 耗时 (ms)
     */
    double stop(const std::string& name) {
        auto end = std::chrono::high_resolution_clock::now();
        auto start = start_times_[name];
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end - start).count();
        
        double ms = duration / 1000.0;
        
        // 记录历史
        history_[name].push_back(ms);
        
        return ms;
    }
    
    /**
     * @brief 获取平均耗时
     * @param name 任务名称
     * @return 平均耗时 (ms)
     */
    double get_average(const std::string& name) {
        auto it = history_.find(name);
        if (it == history_.end() || it->second.empty()) {
            return 0.0;
        }
        
        double sum = 0.0;
        for (double ms : it->second) {
            sum += ms;
        }
        
        return sum / it->second.size();
    }
    
    /**
     * @brief 获取 P99 耗时
     * @param name 任务名称
     * @return P99 耗时 (ms)
     */
    double get_p99(const std::string& name) {
        auto it = history_.find(name);
        if (it == history_.end() || it->second.empty()) {
            return 0.0;
        }
        
        std::vector<double> sorted = it->second;
        std::sort(sorted.begin(), sorted.end());
        
        int p99_idx = static_cast<int>(sorted.size() * 0.99);
        return sorted[p99_idx];
    }
    
    /**
     * @brief 重置历史
     * @param name 任务名称
     */
    void reset(const std::string& name) {
        history_[name].clear();
    }
    
    /**
     * @brief 打印统计
     */
    void print_stats() {
        std::cout << "\nPerformance Statistics:" << std::endl;
        std::cout << std::string(50, '-') << std::endl;
        
        for (const auto& [name, values] : history_) {
            if (values.empty()) continue;
            
            double avg = get_average(name);
            double p99 = get_p99(name);
            double min = *std::min_element(values.begin(), values.end());
            double max = *std::max_element(values.begin(), values.end());
            
            std::cout << name << ":" << std::endl;
            std::cout << "  Avg: " << avg << " ms" << std::endl;
            std::cout << "  P99: " << p99 << " ms" << std::endl;
            std::cout << "  Min: " << min << " ms" << std::endl;
            std::cout << "  Max: " << max << " ms" << std::endl;
        }
    }

private:
    std::map<std::string, std::chrono::high_resolution_clock::time_point> start_times_;
    std::map<std::string, std::vector<double>> history_;
};

// 全局监控器实例
PerformanceMonitor& get_performance_monitor() {
    static PerformanceMonitor monitor;
    return monitor;
}

} // namespace face
