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

步骤 2: 启动统一启动器
   在项目根目录下运行：
   $ python main.py

步骤 3: 选择功能模式
   - [1] 启动演示服务器 (Start Demo Server)
     启动后访问Web界面：http://localhost:3000 (前端) 或 http://localhost:8000 (API文档)
   
   - [2] 启动模型训练 (Start Model Training)
     使用 `src/data/` 中的图片重新训练风格模型。
     注意：这是一个计算密集型任务。

   - [3] 启动前端 (Start Frontend)
     即 Next.js 现代化界面。请新建一个终端窗口：
     ```bash
     cd frontend
     npm install  # 安装依赖 (首次运行需要)
     npm run dev  # 启动开发服务器
     ```
     访问 UI：http://localhost:3000

4. 目录结构 (File Structure)
-----------------
.
├── src/
│   ├── style_transfer/     # 核心算法 (Transformer, VGG Loss)
│   ├── checkpoints/        # 训练好的模型权重
│   ├── frontend/           # Next.js 前端源代码
│   ├── app.py              # 后端 API 服务 (FastAPI)
│   ├── train_v2.py         # V2版本训练脚本 (VGG16 + Pixel Loss)
│   └── finetune_v2.py      # 微调脚本
├── data/
│   └── val2017/            # COCO 验证集 (训练素材)
├── docs/                   # 技术文档
│   └── technical_handover_guide.md # 技术交接指南
├── main.py                 # 项目统一入口
├── README.txt              # 英文说明文档
└── README_cn.md            # 中文说明文档 (本文)

5. 致谢与引用 (Credits)
-----------------------
- 基础算法改编自 TensorFlow 官方教程
- 快速风格迁移网络 proposed by Johnson et al. (ECCV 2016)
- 预训练 VGG16 权重来自 Keras Applications

-----------------------------------------------------
声明：本项目为 AIT102 课程原创作品。
