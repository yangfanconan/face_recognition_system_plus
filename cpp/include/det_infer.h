/**
 * @file det_infer.h
 * @brief Ultra-Tiny Face Detection Inference Header (超小人脸检测推理头文件)
 * 
 * 高性能人脸检测推理接口
 * - 支持 TensorRT 推理
 * - FP16/INT8 量化
 * - CUDA 加速后处理
 * 
 * @version 1.0
 * @date 2026-03-09
 */

#ifndef DET_INFER_H
#define DET_INFER_H

#include <vector>
#include <string>
#include <memory>
#include <cuda_runtime.h>
#include <NvInfer.h>

namespace face {

/**
 * @brief 检测结果结构体
 */
struct Detection {
    float x1, y1, x2, y2;  ///< 边界框
    float confidence;       ///< 置信度
    float landmarks[10];    ///< 5 个关键点 (x, y)
    
    Detection() : x1(0), y1(0), x2(0), y2(0), confidence(0) {
        for (int i = 0; i < 10; ++i) landmarks[i] = 0;
    }
};

/**
 * @brief 检测器配置
 */
struct DetectorConfig {
    int input_width = 640;           ///< 输入宽度
    int input_height = 640;          ///< 输入高度
    float conf_threshold = 0.3f;     ///< 置信度阈值
    float nms_threshold = 0.5f;      ///< NMS 阈值
    int max_num_detections = 1000;   ///< 最大检测数
    bool use_fp16 = true;            ///< 是否使用 FP16
    bool use_int8 = false;           ///< 是否使用 INT8
    int gpu_id = 0;                  ///< GPU ID
};

/**
 * @brief CUDA 错误检查宏
 */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

/**
 * @brief TensorRT Logger
 */
class TensorRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            printf("[TensorRT] %s\n", msg);
        }
    }
};

/**
 * @brief Ultra-Tiny 检测器推理类
 * 
 * 支持 TensorRT 加速的人脸检测推理
 * 
 * 使用示例:
 * @code
 * DetectorConfig config;
 * config.conf_threshold = 0.5f;
 * 
 * UltraTinyDetector detector(config);
 * detector.load("model.engine");
 * 
 * std::vector<Detection> results;
 * detector.infer(image_data, results);
 * @endcode
 */
class UltraTinyDetector {
public:
    /**
     * @brief 构造函数
     * @param config 检测器配置
     */
    explicit UltraTinyDetector(const DetectorConfig& config = DetectorConfig());
    
    /**
     * @brief 析构函数
     */
    ~UltraTinyDetector();
    
    /**
     * @brief 从文件加载 TensorRT 引擎
     * @param engine_path TensorRT 引擎文件路径
     * @return 是否成功
     */
    bool load(const std::string& engine_path);
    
    /**
     * @brief 从 ONNX 模型构建 TensorRT 引擎
     * @param onnx_path ONNX 模型路径
     * @return 是否成功
     */
    bool build_from_onnx(const std::string& onnx_path);
    
    /**
     * @brief 执行推理
     * @param input 输入图像数据 [H, W, 3], RGB 格式，归一化到 [0, 1]
     * @param detections 输出检测结果
     * @return 是否成功
     */
    bool infer(const std::vector<float>& input, std::vector<Detection>& detections);
    
    /**
     * @brief 批量推理
     * @param inputs 输入图像批次 [N, H*W*3]
     * @param all_detections 所有检测结果
     * @return 是否成功
     */
    bool batch_infer(
        const std::vector<std::vector<float>>& inputs,
        std::vector<std::vector<Detection>>& all_detections
    );
    
    /**
     * @brief 获取输入尺寸
     * @return 输入宽度
     */
    int get_input_width() const { return config_.input_width; }
    
    /**
     * @brief 获取输入高度
     * @return 输入高度
     */
    int get_input_height() const { return config_.input_height; }
    
    /**
     * @brief 同步设备
     */
    void synchronize();

private:
    /**
     * @brief 预处理输入图像
     * @param input 原始图像
     * @param device_input 设备端输入
     */
    void preprocess(const std::vector<float>& input, float* device_input);
    
    /**
     * @brief 后处理模型输出
     * @param host_output 主机端输出
     * @param detections 检测结果
     */
    void postprocess(const float* host_output, std::vector<Detection>& detections);
    
    /**
     * @brief CUDA NMS
     * @param boxes 边界框
     * @param scores 置信度
     * @param keep 保留的索引
     * @return 保留的数量
     */
    int cuda_nms(
        const float* boxes,
        const float* scores,
        int num_boxes,
        int* keep
    );

private:
    DetectorConfig config_;                    ///< 配置
    TensorRTLogger logger_;                    ///< TensorRT 日志
    
    nvinfer1::ICudaEngine* engine_ = nullptr;  ///< TensorRT 引擎
    nvinfer1::IExecutionContext* context_ = nullptr;  ///< 执行上下文
    
    void* input_buffer_ = nullptr;             ///< 输入缓冲区 (设备)
    void* output_buffers_[4] = {nullptr};      ///< 输出缓冲区 (设备)
    
    std::vector<size_t> input_sizes_;          ///< 输入尺寸
    std::vector<size_t> output_sizes_;         ///< 输出尺寸
    
    cudaStream_t stream_ = nullptr;            ///< CUDA 流
};

/**
 * @brief 检测器管理类
 * 
 * 单例模式管理检测器实例
 */
class DetectorManager {
public:
    /**
     * @brief 获取单例实例
     * @return DetectorManager 实例
     */
    static DetectorManager& instance();
    
    /**
     * @brief 创建检测器
     * @param id 检测器 ID
     * @param config 配置
     * @return 是否成功
     */
    bool create(int id, const DetectorConfig& config);
    
    /**
     * @brief 加载模型
     * @param id 检测器 ID
     * @param model_path 模型路径
     * @return 是否成功
     */
    bool load(int id, const std::string& model_path);
    
    /**
     * @brief 获取检测器
     * @param id 检测器 ID
     * @return 检测器指针
     */
    UltraTinyDetector* get(int id);
    
    /**
     * @brief 销毁检测器
     * @param id 检测器 ID
     */
    void destroy(int id);

private:
    DetectorManager() = default;
    ~DetectorManager();
    
    DetectorManager(const DetectorManager&) = delete;
    DetectorManager& operator=(const DetectorManager&) = delete;
    
    std::map<int, std::unique_ptr<UltraTinyDetector>> detectors_;
};

} // namespace face

#endif // DET_INFER_H
