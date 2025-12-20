AIT102 Group Project: Real-time Style Transfer Server
=====================================================

1. PROJECT OVERVIEW
-------------------
This project implements a High-Performance Neural Style Transfer system designed for real-time demonstrations. 
It utilizes a Transformer-based Fast Style Transfer network, optimized for Dual NVIDIA RTX 3090 GPUs.

Key features include:
- "Style Matrix": Parallel generation of 5+ styles simultaneously.
- "God Mode": Real-time video style transfer using WebRTC.
- Server-Client Architecture: Python FastAPI backend + Next.js Frontend (planned).

2. REQUIREMENTS
---------------
Hardware:
- Operating System: Ubuntu 22.04 LTS (Tested)
- GPU: NVIDIA RTX 3090 x2 (Minimum: 1x NVIDIA GPU with 8GB+ VRAM)
- CUDA: 11.8+

Software:
- Python 3.9
- Conda (Miniconda/Anaconda)

Python Libraries (Install via src/requirements.txt):
- tensorflow>=2.14.0
- fastapi
- uvicorn
- opencv-python
- pillow
- tqdm

3. HOW TO RUN
-------------
Step 1: Environment Setup
   Open a terminal and create the conda environment:
   $ conda create -n ait python=3.9 -y
   $ conda activate ait
   $ pip install -r src/requirements.txt

Step 2: Unified Launcher (Recommended)
   Run the main menu from the project root:
   $ python main.py

   > Menu Options:
   > [1] Start Full Stack Demo (One-Click)
   >     - Launches Backend (Port 8000) and Frontend (Port 3000) in separate windows.
   >     - Best for presentations.
   >
   > [2] Train New Style (Manager Mode)
   >     - Prompts for style name (e.g., 'monet').
   >     - Requires Dual RTX 3090 GPUs (48GB VRAM) for stability.
   >     - Automatically manages training, checkpoints, and export.

4. CORE SCRIPTS CLI
-------------------
You can also run individual components manually:

[A] Backend Server
    $ python src/server.py
    # Starts FastAPI server (TF + ONNX God Mode) on Port 8000.

[B] Training Manager
    $ python src/train_manager.py --style <style_name>
    # Example: python src/train_manager.py --style picasso
    # runs the full training pipeline using DistributedStrategy.

[C] ONNX Converter
    $ python src/export_onnx.py --model_dir src/models/exported --fp16
    # Converts trained .h5 models to .onnx and optimizes for FP16.
    # Essential for Real-time Video performance.

5. FILE STRUCTURE
-----------------
.
├── src/
│   ├── server.py           # Unified Backend (API + Streaming)
│   ├── train_manager.py    # Training orchestration script
│   ├── export_onnx.py      # Model converter (H5 -> ONNX FP16)
│   ├── models/             # Exported models (.h5, .onnx)
│   ├── styles/             # Style reference images
│   ├── data/               # Training dataset (COCO)
│   └── frontend/           # Next.js Frontend source
├── main.py                 # Project Launcher
├── README.txt              # This file
└── README_cn.md            # Chinese Documentation

6. CREDITS & REFERENCES
-----------------------
- Base logic adapted from TensorFlow Tutorials: https://www.tensorflow.org/tutorials/generative/style_transfer
- Fast Style Transfer Network: Johnson et al. (2016)
- VGG16 Pre-trained Weights: Keras Applications
