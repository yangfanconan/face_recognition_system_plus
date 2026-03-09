# CMakeLists.txt for QWEncode Face Recognition SDK
# ==================================================
# 高性能轻量化人脸识别编码实现
# 基于 C++17 + CUDA 优化，支持 TensorRT 推理、亿级检索

cmake_minimum_required(VERSION 3.18)
project(QWEncode VERSION 1.0.0 LANGUAGES CXX CUDA)

# C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# CUDA 架构
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)

# 编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")

# 查找包
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(TensorRT REQUIRED)

# FAISS (需要手动指定路径)
set(FAISS_ROOT "/usr/local/faiss" CACHE PATH "FAISS installation root")
find_library(FAISS_LIBRARY faiss ${FAISS_ROOT}/lib)
find_path(FAISS_INCLUDE faiss/Index.h ${FAISS_ROOT}/include)

# ============================================================================
# 头文件目录
# ============================================================================
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIRS}
    ${FAISS_INCLUDE}
)

# ============================================================================
# 检测模块 (Detection)
# ============================================================================
# CUDA 源文件
set(DET_CUDA_SOURCES
    src/det/ultra_det_cuda.cu
)

# C++ 源文件
set(DET_CPP_SOURCES
    src/det/ultra_det_trt.cpp
)

# 检测库
add_library(detector SHARED
    ${DET_CUDA_SOURCES}
    ${DET_CPP_SOURCES}
)

target_link_libraries(detector
    ${CUDA_LIBRARIES}
    ${TENSORRT_LIBRARIES}
)

set_target_properties(detector PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# ============================================================================
# 识别模块 (Recognition)
# ============================================================================
set(REC_CUDA_SOURCES
    src/rec/ultra_rec_cuda.cu
)

set(REC_CPP_SOURCES
    src/rec/ultra_rec_trt.cpp
)

add_library(recognizer SHARED
    ${REC_CUDA_SOURCES}
    ${REC_CPP_SOURCES}
)

target_link_libraries(recognizer
    ${CUDA_LIBRARIES}
    ${TENSORRT_LIBRARIES}
)

set_target_properties(recognizer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# ============================================================================
# 检索模块 (Retrieval)
# ============================================================================
set(RETRIEVAL_SOURCES
    src/retrieval/billion_search.cpp
    src/retrieval/index_manager.cpp
)

add_library(retrieval SHARED
    ${RETRIEVAL_SOURCES}
)

target_link_libraries(retrieval
    ${FAISS_LIBRARY}
    ${CUDA_LIBRARIES}
    pthread
)

# ============================================================================
# SDK 模块
# ============================================================================
set(SDK_SOURCES
    src/sdk/face_sdk.cpp
)

add_library(face_sdk SHARED
    ${SDK_SOURCES}
)

target_link_libraries(face_sdk
    detector
    recognizer
    retrieval
    ${OpenCV_LIBS}
)

# 设置输出名称
set_target_properties(face_sdk PROPERTIES
    OUTPUT_NAME "face_sdk"
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
)

# ============================================================================
# 可执行文件
# ============================================================================

# SDK 测试程序
add_executable(sdk_test examples/sdk_test.cpp)
target_link_libraries(sdk_test face_sdk)

# 性能基准测试
add_executable(benchmark examples/benchmark.cpp)
target_link_libraries(benchmark face_sdk)

# 模型转换工具
add_executable(model_converter tools/model_converter.cpp)
target_link_libraries(model_converter
    detector
    recognizer
    ${TENSORRT_LIBRARIES}
)

# 索引构建工具
add_executable(index_builder tools/index_builder.cpp)
target_link_libraries(index_builder
    retrieval
    recognizer
    ${OpenCV_LIBS}
)

# ============================================================================
# Python 绑定 (可选)
# ============================================================================
option(BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)

if(BUILD_PYTHON_BINDINGS)
    find_package(Python3 COMPONENTS Interpreter Development)
    find_package(pybind11 QUIET)
    
    if(pybind11_FOUND)
        pybind11_add_module(py_face_sdk python/py_face_sdk.cpp)
        target_link_libraries(py_face_sdk PRIVATE face_sdk)
        
        install(TARGETS py_face_sdk LIBRARY DESTINATION .)
    endif()
endif()

# ============================================================================
# 安装
# ============================================================================
install(TARGETS
    face_sdk
    detector
    recognizer
    retrieval
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include)

# ============================================================================
# 配置输出
# ============================================================================
message(STATUS "")
message(STATUS "QWEncode Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  OpenCV Version: ${OpenCV_VERSION}")
message(STATUS "  TensorRT Include: ${TENSORRT_INCLUDE_DIRS}")
message(STATUS "  FAISS Library: ${FAISS_LIBRARY}")
message(STATUS "  Python Bindings: ${BUILD_PYTHON_BINDINGS}")
message(STATUS "")
