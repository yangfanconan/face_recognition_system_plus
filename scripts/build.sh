#!/bin/bash
# Ultra-Face Recognition System - Build Script
# =============================================
# 快速构建脚本

set -e

echo "========================================"
echo "Ultra-Face Recognition System Build"
echo "========================================"

# 检查 Python 环境
echo ""
echo "Checking Python environment..."
python3 --version || python --version

# 安装 Python 依赖
echo ""
echo "Installing Python dependencies..."
cd python
pip install -r requirements.txt
cd ..

# 编译 C++ 项目
echo ""
echo "Building C++ project..."
cd cpp
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd ../..

echo ""
echo "========================================"
echo "Build completed successfully!"
echo "========================================"
echo ""
echo "Usage:"
echo "  Python training:  cd python && python main.py train_det --config configs/config.yaml"
echo "  Python inference: cd python && python main.py inference --image test.jpg"
echo "  C++ test:         cd cpp/build && ./sdk_test"
echo ""
