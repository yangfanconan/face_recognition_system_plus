@echo off
REM Ultra-Face Recognition System - Windows Build Script
REM =====================================================

echo ========================================
echo Ultra-Face Recognition System Build
echo ========================================

REM 检查 Python 环境
echo.
echo Checking Python environment...
python --version

REM 安装 Python 依赖
echo.
echo Installing Python dependencies...
cd python
pip install -r requirements.txt
cd ..

REM 编译 C++ 项目
echo.
echo Building C++ project...
cd cpp
if not exist build mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j %NUMBER_OF_PROCESSORS%
cd ../..

echo.
echo ========================================
echo Build completed successfully!
echo ========================================
echo.
echo Usage:
echo   Python training:  cd python ^&^& python main.py train_det --config configs/config.yaml
echo   Python inference: cd python ^&^& python main.py inference --image test.jpg
echo   C++ test:         cd cpp\build ^&^& sdk_test.exe
echo.
