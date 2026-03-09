/**
 * @file rec_feature.h
 * @brief Ultra-Precise Face Recognition Feature Header (极致人脸识别特征头文件)
 * 
 * 高性能人脸识别特征提取接口
 * - 三分支特征提取 (空域 + 频域 + 深度)
 * - Transformer 全局建模
 * - 特征解耦 (512d 身份 +128d 属性 +64d 深度)
 * 
 * @version 1.0
 * @date 2026-03-09
 */

#ifndef REC_FEATURE_H
#define REC_FEATURE_H

#include <vector>
#include <string>
#include <memory>
#include <cuda_runtime.h>
#include <NvInfer.h>

namespace face {

/**
 * @brief 识别特征结构体
 */
struct RecognitionFeature {
    std::vector<float> id_feature;      ///< 身份特征 (512d)
    std::vector<float> attr_feature;    ///< 属性特征 (128d)
    std::vector<float> depth_feature;   ///< 深度特征 (64d)
    
    /// 归一化身份特征
    void normalize_id() {
        float norm = 0.0f;
        for (float v : id_feature) norm += v * v;
        norm = std::sqrt(norm);
        if (norm > 1e-7f) {
            for (float& v : id_feature) v /= norm;
        }
    }
    
    /// 计算与另一个特征的相似度
    float similarity(const RecognitionFeature& other) const {
        float dot = 0.0f;
        for (size_t i = 0; i < id_feature.size() && i < other.id_feature.size(); ++i) {
            dot += id_feature[i] * other.id_feature[i];
        }
        return dot;
    }
};

/**
 * @brief 识别器配置
 */
struct RecognizerConfig {
    int input_width = 112;             ///< 输入宽度
    int input_height = 112;            ///< 输入高度
    int id_dim = 512;                  ///< 身份特征维度
    int attr_dim = 128;                ///< 属性特征维度
    int depth_dim = 64;                ///< 深度特征维度
    bool use_fp16 = true;              ///< 是否使用 FP16
    bool use_trt = true;               ///< 是否使用 TensorRT
    int gpu_id = 0;                    ///< GPU ID
    float verify_threshold = 0.5f;     ///< 验证阈值
};

/**
 * @brief Ultra-Precise 识别器类
 * 
 * 支持三分支特征提取的人脸识别器
 * 
 * 使用示例:
 * @code
 * RecognizerConfig config;
 * UltraPreciseRecognizer recognizer(config);
 * recognizer.load("rec.engine");
 * 
 * RecognitionFeature feat;
 * recognizer.extract(image_data, feat);
 * @endcode
 */
class UltraPreciseRecognizer {
public:
    /**
     * @brief 构造函数
     * @param config 识别器配置
     */
    explicit UltraPreciseRecognizer(const RecognizerConfig& config = RecognizerConfig());
    
    /**
     * @brief 析构函数
     */
    ~UltraPreciseRecognizer();
    
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
     * @brief 提取特征
     * @param input 输入人脸图像 [H, W, 3], RGB 格式
     * @param feature 输出特征
     * @return 是否成功
     */
    bool extract(const std::vector<float>& input, RecognitionFeature& feature);
    
    /**
     * @brief 批量提取特征
     * @param inputs 输入图像批次
     * @param features 输出特征批次
     * @return 是否成功
     */
    bool batch_extract(
        const std::vector<std::vector<float>>& inputs,
        std::vector<RecognitionFeature>& features
    );
    
    /**
     * @brief 人脸验证 (1:1)
     * @param feature1 特征 1
     * @param feature2 特征 2
     * @param threshold 阈值
     * @return 是否同一人
     */
    bool verify(
        const RecognitionFeature& feature1,
        const RecognitionFeature& feature2,
        float threshold = -1.0f
    ) const;
    
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
     * @brief 获取特征维度
     * @return 身份特征维度
     */
    int get_feature_dim() const { return config_.id_dim; }
    
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
     * @param feature 输出特征
     */
    void postprocess(const float* host_output, RecognitionFeature& feature);
    
    /**
     * @brief L2 归一化
     * @param data 数据
     * @param size 大小
     */
    void l2_normalize(float* data, int size);

private:
    RecognizerConfig config_;                  ///< 配置
    TensorRTLogger logger_;                    ///< TensorRT 日志
    
    nvinfer1::ICudaEngine* engine_ = nullptr;  ///< TensorRT 引擎
    nvinfer1::IExecutionContext* context_ = nullptr;  ///< 执行上下文
    
    void* input_buffer_ = nullptr;             ///< 输入缓冲区 (设备)
    void* output_buffers_[3] = {nullptr};      ///< 输出缓冲区 (设备): id, attr, depth
    
    std::vector<size_t> input_sizes_;          ///< 输入尺寸
    std::vector<size_t> output_sizes_;         ///< 输出尺寸
    
    cudaStream_t stream_ = nullptr;            ///< CUDA 流
};

/**
 * @brief 识别器管理类
 */
class RecognizerManager {
public:
    /**
     * @brief 获取单例实例
     */
    static RecognizerManager& instance();
    
    /**
     * @brief 创建识别器
     * @param id 识别器 ID
     * @param config 配置
     * @return 是否成功
     */
    bool create(int id, const RecognizerConfig& config);
    
    /**
     * @brief 加载模型
     * @param id 识别器 ID
     * @param model_path 模型路径
     * @return 是否成功
     */
    bool load(int id, const std::string& model_path);
    
    /**
     * @brief 获取识别器
     * @param id 识别器 ID
     * @return 识别器指针
     */
    UltraPreciseRecognizer* get(int id);
    
    /**
     * @brief 销毁识别器
     * @param id 识别器 ID
     */
    void destroy(int id);

private:
    RecognizerManager() = default;
    ~RecognizerManager();
    
    RecognizerManager(const RecognizerManager&) = delete;
    RecognizerManager& operator=(const RecognizerManager&) = delete;
    
    std::map<int, std::unique_ptr<UltraPreciseRecognizer>> recognizers_;
};

} // namespace face

#endif // REC_FEATURE_H
