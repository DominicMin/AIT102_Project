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

Step 2: Run the Main Program
   Execute the unified launcher from the project root:
   $ python main.py

Step 3: Select an Option
   - Option [1]: Starts the Demo Server (Backend).
     Once running, access the web interface at: http://localhost:8000
   
   - Option [2]: Starts Model Training.
     This will retrain the style transfer models using the images in 'src/data/'.
     Note: This process is computationally intensive.

4. FILE STRUCTURE
-----------------
/src
  /style_transfer   - Core model definitions (Transformer, VGG Loss)
  /checkpoints      - Trained model weights
  train_v2.py       - Main training script (Dual GPU optimized)
  server.py         - Backend server code (FastAPI)
main.py             - Project Entry Point (Launcher)
README.txt          - This file
docs/               - Technical documentation & plans

5. CREDITS & REFERENCES
-----------------------
- Base logic adapted from TensorFlow Tutorials: https://www.tensorflow.org/tutorials/generative/style_transfer
- Fast Style Transfer Network: Johnson et al. (2016)
- VGG16 Pre-trained Weights: Keras Applications

-----------------------------------------------------
Declaration: This work is original and submitted for AIT102.
