AIT102 风格迁移演示系统 (Real-time Style Transfer Server)
=====================================================

1. 项目概述 (Project Overview)
-------------------
本项目是一个专为现场演示的高性能神经风格迁移系统。
利用基于 Transformer 的快速风格迁移网络 (Fast Style Transfer Network)，并针对双卡 NVIDIA RTX 3090 服务器进行了深度优化。

核心亮点：
- **风格矩阵 (Style Matrix)**：利用双卡并发能力，一次生成 5+ 张不同风格的图片。
- **上帝模式 (God Mode)**：基于 WebRTC 的实时视频风格迁移，低延迟体验。
- **前后端分离架构**：
  - 后端：Python FastAPI (高性能异步服务)
  - 前端：Next.js + TailwindCSS (现代化交互界面)

2. 环境要求 (Requirements)
-------------------
硬件需求：
- 操作系统：Ubuntu 22.04 LTS (推荐) / Windows (支持 WSL)
- GPU：NVIDIA RTX 3090 x2 (最低要求：任意 8GB+ 显存的 NVIDIA 显卡)
- CUDA 版本：11.8+

软件需求：
- Python 3.9+
- Node.js 16+ (用于前端)
- Conda (Miniconda/Anaconda)

Python 依赖库 (安装命令见下文)：
- tensorflow>=2.14.0
- fastapi
- uvicorn
- opencv-python
- pillow
- tqdm

3. 快速开始 (How to Run)
-------------------
步骤 1: 环境配置
   打开终端并创建 conda 环境：
   $ conda create -n ait python=3.9 -y
   $ conda activate ait
   $ pip install -r src/requirements.txt

步骤 2: 启动统一启动器 (Project Launcher)
   这是管理整个项目的推荐方式：
   $ python main.py

   > 菜单选项说明：
   > [1] Start Unified Server & Frontend (一键启动)
   >     - 自动开启两个新窗口，分别运行后端 (Port 8000) 和前端 (Port 3000)。
   >     - 这是演示 Demo 的最佳方式。
   >
   > [2] Train New Style (训练模式)
   >     - 输入风格名称（如 `monet`），自动调用训练管理器。
   >     - 需要双 RTX 3090 (48GB VRAM) 支持。

4. 核心脚本说明 (Core Scripts)
-----------------
如果你想单独运行某个模块，可以使用以下命令：

[A] 启动后端服务 (Server)
    $ python src/server.py
    # 启动 FastAPI 服务，监听 8000 端口
    # 同时加载 TensorFlow模型 和 ONNX上帝模式模型

[B] 模型训练管理器 (Training Manager)
    $ python src/train_manager.py --style <style_name>
    # 例如: python src/train_manager.py --style picasso
    # 自动执行：数据加载 -> VGG特征提取 -> 双卡训练 -> 模型导出 -> ONNX转换

[C] ONNX 模型优化 (Optimizer)
    $ python src/export_onnx.py --model_dir src/models/exported --fp16
    # 将训练好的 .h5模型 转换为 .onnx 并进行 FP16 半精度优化
    # 这是实现 RTX 4060 实时推理的关键

5. 目录结构 (File Structure)
-----------------
.
├── src/
│   ├── server.py           # 核心后端：集成 TF推理 + ONNX实时流 + MJPEG服务
│   ├── train_manager.py    # 训练管理器：负责调用训练流程管理
│   ├── export_onnx.py      # 模型转换器：H5 -> ONNX (FP16/FP32)
│   ├── models/             # 存放训练模型 (.h5, .onnx)
│   ├── styles/             # 风格参考图片 (.jpg)
│   ├── data/               # 训练数据集 (COCO)
│   └── frontend/           # Next.js 前端项目
├── main.py                 # 项目统一入口 (CLI Launcher)
├── README.txt              # 英文说明文档
└── README_cn.md            # 中文说明文档 (本文)

6. 致谢与引用 (Credits)
-----------------------
- 基础算法改编自 TensorFlow 官方教程
- 快速风格迁移网络 proposed by Johnson et al. (ECCV 2016)
- 预训练 VGG16 权重来自 Keras Applications
