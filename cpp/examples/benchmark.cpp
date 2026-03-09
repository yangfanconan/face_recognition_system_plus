/**
 * @file benchmark.cpp
 * @brief Performance Benchmark Example (性能基准测试示例)
 * 
 * 测试检测和识别模块的性能
 */

#include "sdk_interface.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <vector>

void benchmark_detection(FaceSDKHandle handle, const cv::Mat& image, int iterations = 100) {
    std::cout << "Benchmarking Detection..." << std::endl;
    
    cv::Mat rgb_image;
    cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);
    
    // 预热
    FaceDetection* faces = nullptr;
    int num_faces = 0;
    face_sdk_detect(handle, rgb_image.data, image.cols, image.rows, &faces, &num_faces);
    face_sdk_free_detections(faces, num_faces);
    
    // 测试
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        face_sdk_detect(handle, rgb_image.data, image.cols, image.rows, &faces, &num_faces);
        face_sdk_free_detections(faces, num_faces);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    float avg_latency = static_cast<float>(duration) / iterations;
    float fps = 1000.0f / avg_latency;
    
    std::cout << "  Iterations: " << iterations << std::endl;
    std::cout << "  Avg Latency: " << avg_latency << " ms" << std::endl;
    std::cout << "  FPS: " << fps << std::endl;
}

void benchmark_verification(FaceSDKHandle handle, int dim = 512, int iterations = 10000) {
    std::cout << "\nBenchmarking Verification..." << std::endl;
    
    // 生成随机特征
    std::vector<float> feat1(dim, 0.5f);
    std::vector<float> feat2(dim, 0.5f);
    
    // 预热
    bool is_same = false;
    float similarity = 0.0f;
    face_sdk_verify(handle, feat1.data(), feat2.data(), dim, &is_same, &similarity);
    
    // 测试
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        face_sdk_verify(handle, feat1.data(), feat2.data(), dim, &is_same, &similarity);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    float avg_latency = static_cast<float>(duration) / iterations;
    float qps = 1000.0f / avg_latency;
    
    std::cout << "  Iterations: " << iterations << std::endl;
    std::cout << "  Avg Latency: " << avg_latency * 1000 << " us" << std::endl;
    std::cout << "  QPS: " << qps << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "Face SDK Benchmark" << std::endl;
    std::cout << "==================" << std::endl;
    
    // 配置
    FaceSDKConfig config = {};
    config.det_model_path = "../checkpoints/det.engine";
    config.rec_model_path = "../checkpoints/rec.engine";
    config.det_img_size = 640;
    config.rec_img_size = 112;
    config.gpu_id = 0;
    config.use_fp16 = true;
    
    // 初始化
    FaceSDKHandle handle = face_sdk_init(&config);
    if (!handle) {
        std::cerr << "Failed to initialize SDK" << std::endl;
        return 1;
    }
    
    // 加载测试图像
    cv::Mat image;
    if (argc > 1) {
        image = cv::imread(argv[1]);
    } else {
        image = cv::Mat(480, 640, CV_8UC3, cv::Scalar(100, 100, 100));
    }
    
    if (image.empty()) {
        std::cerr << "Failed to load image" << std::endl;
        face_sdk_destroy(handle);
        return 1;
    }
    
    std::cout << "Image: " << image.cols << "x" << image.rows << std::endl;
    std::cout << std::endl;
    
    // 基准测试
    benchmark_detection(handle, image, 100);
    benchmark_verification(handle, 512, 10000);
    
    // 统计
    FaceSDKStats stats;
    face_sdk_get_stats(handle, &stats);
    
    std::cout << "\nFinal Statistics:" << std::endl;
    std::cout << "  Total inferences: " << stats.total_inferences << std::endl;
    std::cout << "  Avg latency: " << stats.avg_latency_ms << " ms" << std::endl;
    
    face_sdk_destroy(handle);
    
    std::cout << "\nBenchmark completed!" << std::endl;
    
    return 0;
}
