"""
GPU 环境测试脚本。
运行方式：python tools/test_gpu.py  （从 models/ 目录执行）
"""
import sys
import os
import time

# 确保 models/ 根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def check_python():
    print(f"Python 版本: {sys.version}")


def check_torch():
    try:
        import torch
        print(f"\n[PyTorch]")
        print(f"  版本:       {torch.__version__}")
        print(f"  CUDA 可用:  {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {p.name}  显存={p.total_memory/1024**3:.1f}GB  算力={p.major}.{p.minor}")
            print(f"  CUDA 版本:  {torch.version.cuda}")
            print(f"  cuDNN 版本: {torch.backends.cudnn.version()}")
        else:
            print("  [警告] 未检测到 GPU，将使用 CPU")
        return torch
    except ImportError:
        print("[错误] PyTorch 未安装")
        return None


def check_torchvision():
    try:
        import torchvision
        print(f"\n[torchvision]  版本: {torchvision.__version__}")
    except ImportError:
        print("[错误] torchvision 未安装")


def check_opencv():
    try:
        import cv2
        print(f"[OpenCV]       版本: {cv2.__version__}")
    except ImportError:
        print("[错误] opencv-python 未安装")


def check_numpy():
    try:
        import numpy as np
        print(f"[NumPy]        版本: {np.__version__}")
    except ImportError:
        print("[错误] numpy 未安装")


def benchmark_gpu(torch):
    if not torch.cuda.is_available():
        return
    print("\n[GPU 基准] 矩阵乘法 4096×4096 ×10 次...")
    device = torch.device('cuda')
    a = torch.randn(4096, 4096, device=device)
    b = torch.randn(4096, 4096, device=device)
    torch.cuda.synchronize(); _ = a @ b; torch.cuda.synchronize()  # 预热

    start = time.time()
    for _ in range(10):
        _ = a @ b
    torch.cuda.synchronize()
    print(f"  平均耗时: {(time.time()-start)/10*1000:.1f} ms")
    print(f"  显存占用: {torch.cuda.memory_allocated(device)/1024**2:.0f} MB")


def test_model_forward(torch):
    print("\n[模型推理测试]")
    try:
        from config_loader import load_config
        from networks.generator import Generator

        cfg = load_config()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        G = Generator(cfg=cfg['model']).to(device).eval()

        dummy = torch.randn(1, 3, 512, 512, device=device)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            Ms, Mb, Ic4, Ic2, Ic1, Ire, Icomp = G(dummy)
        if device.type == 'cuda':
            torch.cuda.synchronize()

        print(f"  设备:    {device}")
        print(f"  耗时:    {(time.time()-t0)*1000:.1f} ms")
        print(f"  Icomp:   {tuple(Icomp.shape)}")
        print(f"  参数量:  {sum(p.numel() for p in G.parameters())/1e6:.2f} M")
    except Exception as e:
        print(f"  [失败] {e}")


def main():
    print("=" * 50)
    print("  GPU 环境检测")
    print("=" * 50)
    check_python()
    torch = check_torch()
    check_torchvision()
    check_opencv()
    check_numpy()
    if torch:
        benchmark_gpu(torch)
        test_model_forward(torch)
    print("\n" + "=" * 50)
    print("  检测完成")
    print("=" * 50)


if __name__ == '__main__':
    main()
