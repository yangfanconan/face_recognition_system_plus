/**
 * @file sdk_interface.h
 * @brief Face Recognition SDK Interface Header (人脸识别 SDK 接口头文件)
 * 
 * 统一 SDK 接口
 * - C/C++/Python 调用支持
 * - 检测→识别→检索端到端流程
 * - 并发安全
 * - 高性能推理
 * 
 * @version 1.0
 * @date 2026-03-09
 */

#ifndef SDK_INTERFACE_H
#define SDK_INTERFACE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

// ============================================================================
// 类型定义
// ============================================================================

/**
 * @brief SDK 句柄
 */
typedef void* FaceSDKHandle;

/**
 * @brief 边界框
 */
typedef struct {
    float x1, y1, x2, y2;
} FaceBBox;

/**
 * @brief 关键点 (5 个)
 */
typedef struct {
    float x[5];
    float y[5];
} FaceLandmarks;

/**
 * @brief 检测到的人脸
 */
typedef struct {
    FaceBBox bbox;              ///< 边界框
    float confidence;           ///< 置信度
    FaceLandmarks landmarks;    ///< 关键点
    float* feature;             ///< 特征向量 (调用者负责释放)
    int feature_dim;            ///< 特征维度
    int64_t identity_id;        ///< 识别 ID (-1 表示未知)
    float identity_score;       ///< 识别置信度
} FaceDetection;

/**
 * @brief 检索结果
 */
typedef struct {
    int64_t id;                 ///< 人脸 ID
    float score;                ///< 相似度
    char* name;                 ///< 姓名
    char* image_path;           ///< 图像路径
} SearchResult;

/**
 * @brief SDK 配置
 */
typedef struct {
    // 模型路径
    const char* det_model_path;     ///< 检测模型路径
    const char* rec_model_path;     ///< 识别模型路径
    const char* search_index_path;  ///< 检索索引路径
    
    // 检测参数
    int det_img_size;               ///< 检测输入尺寸
    float det_conf_threshold;       ///< 检测置信度阈值
    float det_nms_threshold;        ///< NMS 阈值
    
    // 识别参数
    int rec_img_size;               ///< 识别输入尺寸
    float verify_threshold;         ///< 验证阈值
    
    // 检索参数
    int search_top_k;               ///< 检索返回数量
    float search_threshold;         ///< 检索阈值
    
    // 设备配置
    int gpu_id;                     ///< GPU ID
    bool use_fp16;                  ///< 是否使用 FP16
    
    // 性能配置
    int max_batch_size;             ///< 最大批次大小
    int num_threads;                ///< 线程数
} FaceSDKConfig;

/**
 * @brief SDK 统计信息
 */
typedef struct {
    int64_t total_inferences;       ///< 总推理次数
    double avg_latency_ms;          ///< 平均延迟 (ms)
    double fps;                     ///< FPS
    int64_t num_faces_detected;     ///< 检测到的人脸数
    int64_t index_size;             ///< 索引大小
} FaceSDKStats;

// ============================================================================
// SDK 核心接口
// ============================================================================

/**
 * @brief 初始化 SDK
 * @param config 配置
 * @return SDK 句柄，失败返回 NULL
 */
FaceSDKHandle face_sdk_init(const FaceSDKConfig* config);

/**
 * @brief 销毁 SDK
 * @param handle SDK 句柄
 */
void face_sdk_destroy(FaceSDKHandle handle);

/**
 * @brief 人脸检测 + 识别
 * @param handle SDK 句柄
 * @param image 输入图像数据 (RGB, uint8)
 * @param width 图像宽度
 * @param height 图像高度
 * @param faces 输出人脸数组
 * @param num_faces 输出人脸数量
 * @return 是否成功
 */
bool face_sdk_detect(
    FaceSDKHandle handle,
    const uint8_t* image,
    int width,
    int height,
    FaceDetection** faces,
    int* num_faces
);

/**
 * @brief 人脸验证 (1:1)
 * @param handle SDK 句柄
 * @param feature1 特征 1
 * @param feature2 特征 2
 * @param dim 特征维度
 * @param is_same 是否同一人
 * @param similarity 相似度
 * @return 是否成功
 */
bool face_sdk_verify(
    FaceSDKHandle handle,
    const float* feature1,
    const float* feature2,
    int dim,
    bool* is_same,
    float* similarity
);

/**
 * @brief 人脸检索 (1:N)
 * @param handle SDK 句柄
 * @param feature 特征向量
 * @param dim 特征维度
 * @param results 检索结果
 * @param num_results 结果数量
 * @return 是否成功
 */
bool face_sdk_search(
    FaceSDKHandle handle,
    const float* feature,
    int dim,
    SearchResult** results,
    int* num_results
);

/**
 * @brief 注册人脸
 * @param handle SDK 句柄
 * @param feature 特征向量
 * @param dim 特征维度
 * @param name 姓名
 * @param face_id 输出人脸 ID
 * @return 是否成功
 */
bool face_sdk_register(
    FaceSDKHandle handle,
    const float* feature,
    int dim,
    const char* name,
    int64_t* face_id
);

/**
 * @brief 删除人脸
 * @param handle SDK 句柄
 * @param face_id 人脸 ID
 * @return 是否成功
 */
bool face_sdk_delete(FaceSDKHandle handle, int64_t face_id);

/**
 * @brief 获取统计信息
 * @param handle SDK 句柄
 * @param stats 统计信息
 * @return 是否成功
 */
bool face_sdk_get_stats(FaceSDKHandle handle, FaceSDKStats* stats);

/**
 * @brief 重置统计
 * @param handle SDK 句柄
 * @return 是否成功
 */
bool face_sdk_reset_stats(FaceSDKHandle handle);

// ============================================================================
// 工具函数
// ============================================================================

/**
 * @brief 释放检测结果
 * @param faces 人脸数组
 * @param num_faces 人脸数量
 */
void face_sdk_free_detections(FaceDetection* faces, int num_faces);

/**
 * @brief 释放检索结果
 * @param results 结果数组
 * @param num_results 结果数量
 */
void face_sdk_free_results(SearchResult* results, int num_results);

/**
 * @brief 获取 SDK 版本
 * @return 版本字符串
 */
const char* face_sdk_get_version();

/**
 * @brief 获取错误信息
 * @return 错误信息字符串
 */
const char* face_sdk_get_last_error();

#ifdef __cplusplus
}
#endif

// ============================================================================
// C++ 封装类
// ============================================================================

#ifdef __cplusplus
#include <vector>
#include <string>
#include <memory>

namespace face {

/**
 * @brief C++ SDK 封装类
 */
class FaceSDK {
public:
    /**
     * @brief 构造函数
     * @param config 配置
     */
    explicit FaceSDK(const FaceSDKConfig& config);
    
    /**
     * @brief 析构函数
     */
    ~FaceSDK();
    
    /**
     * @brief 检测人脸
     * @param image 输入图像
     * @param width 宽度
     * @param height 高度
     * @return 检测结果
     */
    std::vector<FaceDetection> detect(
        const uint8_t* image,
        int width,
        int height
    );
    
    /**
     * @brief 验证人脸
     * @param feature1 特征 1
     * @param feature2 特征 2
     * @return (是否同一人，相似度)
     */
    std::pair<bool, float> verify(
        const std::vector<float>& feature1,
        const std::vector<float>& feature2
    );
    
    /**
     * @brief 检索人脸
     * @param feature 特征
     * @param top_k 返回数量
     * @return 检索结果
     */
    std::vector<SearchResult> search(
        const std::vector<float>& feature,
        int top_k = 10
    );
    
    /**
     * @brief 注册人脸
     * @param feature 特征
     * @param name 姓名
     * @return 人脸 ID
     */
    int64_t register_face(
        const std::vector<float>& feature,
        const std::string& name
    );
    
    /**
     * @brief 删除人脸
     * @param face_id 人脸 ID
     * @return 是否成功
     */
    bool delete_face(int64_t face_id);
    
    /**
     * @brief 获取统计
     * @return 统计信息
     */
    FaceSDKStats get_stats() const;

private:
    FaceSDKHandle handle_;
};

} // namespace face
#endif

#endif // SDK_INTERFACE_H
