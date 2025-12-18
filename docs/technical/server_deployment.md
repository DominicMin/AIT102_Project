# Server Deployment Guide (V100 + TF 2.9)

Your code is fully compatible with TensorFlow 2.9 and Python 3.8.
Since V100 servers usually run Linux, you don't need WSL or special Windows hacks. The native `tensorflow` package on Linux includes GPU support.

## 1. What to Upload
Upload the entire `src/` folder to your server, BUT exclude:
- `.venv` (You should recreate this on the server)
- `.venv_wsl`
- `__pycache__`
- `run_wsl_training.ps1` (Windows only)

## 2. Setup on Server
Run these commands on your server terminal:

```bash
# 1. Enter directory
cd src

# 2. Create Virtual Environment (Optional but recommended)
python3.8 -m venv .venv
source .venv/bin/activate

# 3. Install Dependencies
# TF 2.9.0 is compatible with CUDA 11.2
pip install -r requirements_server.txt

# 4. Run Training
# GPU will be detected automatically on Linux
python train_model.py
```

## 3. Monitor
To view TensorBoard on your local machine while training on server:
```bash
# On Server:
tensorboard --logdir logs --port 6006
```
Then use SSH Tunneling on your local machine:
`ssh -L 6006:localhost:6006 user@your_server_ip`
And open `localhost:6006` locally.
