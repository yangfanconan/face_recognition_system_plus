/**
 * @file sdk_test.cpp
 * @brief Face SDK Test Example (SDK 测试示例)
 * 
 * 测试 Face SDK 的基本功能
 */

#include "sdk_interface.h"
#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char** argv) {
    std::cout << "Face SDK Test" << std::endl;
    std::cout << "=============" << std::endl;
    
    // 配置
    FaceSDKConfig config = {};
    config.det_model_path = "../checkpoints/det.engine";
    config.rec_model_path = "../checkpoints/rec.engine";
    config.search_index_path = "../indexes/face_index";
    config.det_img_size = 640;
    config.rec_img_size = 112;
    config.det_conf_threshold = 0.5f;
    config.verify_threshold = 0.5f;
    config.search_top_k = 10;
    config.gpu_id = 0;
    config.use_fp16 = true;
    
    // 初始化 SDK
    std::cout << "Initializing SDK..." << std::endl;
    FaceSDKHandle handle = face_sdk_init(&config);
    
    if (!handle) {
        std::cerr << "Failed to initialize SDK: " << face_sdk_get_last_error() << std::endl;
        return 1;
    }
    
    std::cout << "SDK initialized successfully!" << std::endl;
    std::cout << "SDK Version: " << face_sdk_get_version() << std::endl;
    
    // 加载测试图像
    cv::Mat image = cv::imread("../test.jpg");
    if (image.empty()) {
        std::cerr << "Failed to load test image" << std::endl;
        face_sdk_destroy(handle);
        return 1;
    }
    
    std::cout << "Image loaded: " << image.cols << "x" << image.rows << std::endl;
    
    // 转换颜色空间
    cv::Mat rgb_image;
    cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);
    
    // 检测人脸
    std::cout << "Detecting faces..." << std::endl;
    FaceDetection* faces = nullptr;
    int num_faces = 0;
    
    bool success = face_sdk_detect(
        handle,
        rgb_image.data,
        image.cols,
        image.rows,
        &faces,
        &num_faces
    );
    
    if (success) {
        std::cout << "Detected " << num_faces << " face(s)" << std::endl;
        
        for (int i = 0; i < num_faces; ++i) {
            std::cout << "  Face " << (i + 1) << ":" << std::endl;
            std::cout << "    BBox: [" << faces[i].bbox.x1 << ", " 
                      << faces[i].bbox.y1 << ", "
                      << faces[i].bbox.x2 << ", " 
                      << faces[i].bbox.y2 << "]" << std::endl;
            std::cout << "    Confidence: " << faces[i].confidence << std::endl;
            std::cout << "    Feature dim: " << faces[i].feature_dim << std::endl;
        }
        
        // 释放检测结果
        face_sdk_free_detections(faces, num_faces);
    } else {
        std::cerr << "Detection failed: " << face_sdk_get_last_error() << std::endl;
    }
    
    // 获取统计
    FaceSDKStats stats;
    face_sdk_get_stats(handle, &stats);
    
    std::cout << "\nStatistics:" << std::endl;
    std::cout << "  Total inferences: " << stats.total_inferences << std::endl;
    std::cout << "  Avg latency: " << stats.avg_latency_ms << " ms" << std::endl;
    std::cout << "  FPS: " << stats.fps << std::endl;
    std::cout << "  Faces detected: " << stats.num_faces_detected << std::endl;
    std::cout << "  Index size: " << stats.index_size << std::endl;
    
    // 销毁 SDK
    face_sdk_destroy(handle);
    
    std::cout << "\nTest completed!" << std::endl;
    
    return 0;
}
