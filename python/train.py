"""
Quick Training Script - 快速训练脚本
=====================================
一键启动训练，自动处理所有依赖和配置
"""

import subprocess
import sys
from pathlib import Path

def install_dependencies():
    """安装依赖"""
    print("=" * 60)
    print("安装依赖...")
    print("=" * 60)
    
    requirements = Path("requirements.txt")
    if requirements.exists():
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements)])
    else:
        print("requirements.txt 不存在，跳过依赖安装")


def run_training(mode="all", epochs=10):
    """运行训练"""
    print("=" * 60)
    print(f"开始训练 - 模式：{mode}, Epochs: {epochs}")
    print("=" * 60)
    
    cmd = [
        sys.executable, "automated_training.py",
        f"--{mode}",
        f"--det-epochs", str(epochs),
        f"--rec-epochs", str(epochs),
    ]
    
    subprocess.run(cmd)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['all', 'det', 'rec'], default='all')
    parser.add_argument('--epochs', type=int, default=10)
    
    args = parser.parse_args()
    
    install_dependencies()
    run_training(mode=args.mode, epochs=args.epochs)
